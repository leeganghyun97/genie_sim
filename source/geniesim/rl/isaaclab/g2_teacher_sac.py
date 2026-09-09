"""Privileged-state observation contract for the Milestone 7 G2 SAC teacher.

This module deliberately has no Isaac Lab imports.  It separates the teacher
state from the RGB-D student contract and makes the exact ordering testable
without launching Kit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import math
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from geniesim.rl.sac.stage2_sac import (
    QNetwork,
    SACConfig,
    polyak_update,
    soft_bellman_target,
    squashed_gaussian_log_prob,
)

from .g2_quaternion import canonicalize_quaternion_xyzw


G2_TEACHER_OBSERVATION_SCHEMA = "g2_privileged_teacher_canonical_xyzw_v3"
G2_TEACHER_ACTION_SCHEMA = "g2_se3_gripper_v1"
G2_TEACHER_POLICY_DISTRIBUTION_SCHEMA = (
    "g2_teacher_hybrid_sac_gaussian6_bernoulli1_v1"
)
G2_TEACHER_STATEFUL_ACTION_TRANSFORM_SCHEMA = (
    "g2_teacher_normalized_request_to_bounded_slew_controller_v1"
)
G2_TEACHER_ACTION_DIM = 7
G2_TEACHER_ARM_ACTION_DIM = 6
G2_TEACHER_GRIPPER_ACTION_INDEX = 6


@dataclass(frozen=True)
class G2TeacherActionContract:
    """Exact controller boundary used by the privileged teacher.

    The arm request is continuous, while Isaac Lab's
    ``BinaryJointPositionAction`` consumes only the sign of the final channel.
    A seven-dimensional squashed Gaussian is therefore useful as a diagnostic
    baseline, but is *not* an exact SAC distribution for this mixed action
    space.  Keeping that fact in an executable contract prevents a large run
    from silently treating the pre-threshold scalar as the action seen by the
    environment.
    """

    arm_dimension: int = G2_TEACHER_ARM_ACTION_DIM
    gripper_index: int = G2_TEACHER_GRIPPER_ACTION_INDEX

    @property
    def action_dimension(self) -> int:
        return self.arm_dimension + 1

    @property
    def continuous_gaussian_sac_is_exact(self) -> bool:
        return False

    @property
    def hybrid_sac_is_exact(self) -> bool:
        return True

    @property
    def policy_distribution_schema(self) -> str:
        return G2_TEACHER_POLICY_DISTRIBUTION_SCHEMA

    @property
    def large_scale_blocker(self) -> None:
        # The environment boundary remains mixed continuous/discrete, but the
        # dedicated learner below now evaluates the Bernoulli branch exactly.
        return None

    def project_environment_action(self, action: torch.Tensor) -> torch.Tensor:
        """Return the command actually consumed by the mixed controller.

        This projection is for environment execution and replay provenance.
        It deliberately does not claim that hard thresholding supplies a
        differentiable or probability-correct SAC gripper policy.
        """

        if action.ndim != 2 or action.shape[-1] != self.action_dimension:
            raise ValueError(
                f"teacher action must be [N,{self.action_dimension}]"
            )
        if not bool(torch.isfinite(action).all()):
            raise ValueError("teacher action contains non-finite values")
        projected = action.clone()
        gripper = projected[:, self.gripper_index]
        projected[:, self.gripper_index] = torch.where(
            gripper >= 0.0, torch.ones_like(gripper), -torch.ones_like(gripper)
        )
        return projected


@dataclass(frozen=True)
class G2TeacherStatefulActionTransform:
    """Map a normalized policy request to the bounded controller surface.

    Replay stores the normalized policy request because that is the random
    variable sampled by SAC.  The prior applied controller action is part of
    the 59-D observation, making the slew limiter a deterministic Markov
    action wrapper instead of an unmodelled action substitution.
    """

    maximum_arm_action_magnitude: float = 0.10
    arm_action_slew_per_policy_step: float = 0.02

    def validated(self) -> "G2TeacherStatefulActionTransform":
        if not (
            0.0
            < self.arm_action_slew_per_policy_step
            <= self.maximum_arm_action_magnitude
            <= 1.0
        ):
            raise ValueError("invalid Teacher stateful action transform")
        return self

    def as_dict(self) -> dict[str, float | str]:
        self.validated()
        return {
            "schema": G2_TEACHER_STATEFUL_ACTION_TRANSFORM_SCHEMA,
            "maximum_arm_action_magnitude": self.maximum_arm_action_magnitude,
            "arm_action_slew_per_policy_step": self.arm_action_slew_per_policy_step,
            "replay_action_authority": "NORMALIZED_POLICY_REQUEST_PRE_TRANSFORM",
            "observation_state": "PREVIOUS_APPLIED_CONTROLLER_ACTION",
        }

    def apply(
        self,
        normalized_policy_action: torch.Tensor,
        previous_applied_controller_action: torch.Tensor,
    ) -> torch.Tensor:
        self.validated()
        requested = G2TeacherActionContract().project_environment_action(
            normalized_policy_action
        )
        if previous_applied_controller_action.shape != requested.shape:
            raise ValueError("previous applied action shape differs from request")
        if not bool(torch.isfinite(previous_applied_controller_action).all()):
            raise ValueError("previous applied action contains non-finite values")
        if bool((requested[:, :G2_TEACHER_ARM_ACTION_DIM].abs() > 1.0).any()):
            raise ValueError("normalized Teacher arm request must remain in [-1,1]")
        target = requested.clone()
        target[:, :G2_TEACHER_ARM_ACTION_DIM] *= float(
            self.maximum_arm_action_magnitude
        )
        applied = previous_applied_controller_action.clone()
        applied[:, :G2_TEACHER_ARM_ACTION_DIM] += torch.clamp(
            target[:, :G2_TEACHER_ARM_ACTION_DIM]
            - applied[:, :G2_TEACHER_ARM_ACTION_DIM],
            -self.arm_action_slew_per_policy_step,
            self.arm_action_slew_per_policy_step,
        )
        applied[:, G2_TEACHER_GRIPPER_ACTION_INDEX] = target[
            :, G2_TEACHER_GRIPPER_ACTION_INDEX
        ]
        return applied


class _G2TeacherHybridActor(nn.Module):
    """Six reparameterized arm dimensions plus one exact Bernoulli gripper."""

    def __init__(self, config: SACConfig) -> None:
        super().__init__()
        if config.action_dim != G2_TEACHER_ACTION_DIM:
            raise ValueError("G2 Teacher hybrid actor requires a 6+1 action")
        if tuple(config.action_mask) != (1.0,) * G2_TEACHER_ACTION_DIM:
            raise ValueError("G2 Teacher hybrid actor requires all action channels")
        layers: list[nn.Module] = []
        previous = config.observation_dim
        for width in config.hidden_sizes:
            linear = nn.Linear(previous, width)
            nn.init.orthogonal_(linear.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.ReLU()))
            previous = width
        self.config = config
        self.trunk = nn.Sequential(*layers)
        self.arm_mean = nn.Linear(previous, G2_TEACHER_ARM_ACTION_DIM)
        self.arm_log_std = nn.Linear(previous, G2_TEACHER_ARM_ACTION_DIM)
        self.gripper_logit = nn.Linear(previous, 1)
        nn.init.orthogonal_(self.arm_mean.weight, gain=0.01)
        nn.init.zeros_(self.arm_mean.bias)
        nn.init.orthogonal_(self.arm_log_std.weight, gain=0.01)
        nn.init.constant_(self.arm_log_std.bias, -2.0)
        nn.init.orthogonal_(self.gripper_logit.weight, gain=0.01)
        nn.init.zeros_(self.gripper_logit.bias)

    def distribution_parameters(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(observation)
        mean = self.arm_mean(features)
        raw_log_std = torch.tanh(self.arm_log_std(features))
        log_std = self.config.log_std_min + 0.5 * (
            self.config.log_std_max - self.config.log_std_min
        ) * (raw_log_std + 1.0)
        return mean, log_std, self.gripper_logit(features)

    def sample_arm(
        self, observation: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, gripper_logit = self.distribution_parameters(observation)
        pre_tanh = (
            mean
            if deterministic
            else torch.distributions.Normal(mean, log_std.exp()).rsample()
        )
        arm_action = torch.tanh(pre_tanh)
        arm_log_probability = squashed_gaussian_log_prob(
            pre_tanh, mean, log_std
        )
        return (
            arm_action,
            arm_log_probability,
            torch.tanh(mean),
            log_std,
            gripper_logit,
        )

    def sample(
        self, observation: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        arm, arm_logp, mean_arm, _, gripper_logit = self.sample_arm(
            observation, deterministic=deterministic
        )
        probability_open = torch.sigmoid(gripper_logit)
        open_sample = (
            probability_open >= 0.5
            if deterministic
            else torch.bernoulli(probability_open).to(torch.bool)
        )
        gripper = torch.where(
            open_sample,
            torch.ones_like(probability_open),
            -torch.ones_like(probability_open),
        )
        selected_probability = torch.where(
            open_sample, probability_open, 1.0 - probability_open
        )
        log_probability = arm_logp + torch.log(
            selected_probability.clamp_min(1.0e-8)
        )
        mean_gripper = torch.where(
            probability_open >= 0.5,
            torch.ones_like(probability_open),
            -torch.ones_like(probability_open),
        )
        return (
            torch.cat((arm, gripper), dim=-1),
            log_probability,
            torch.cat((mean_arm, mean_gripper), dim=-1),
        )


class G2TeacherHybridSACAgent:
    """State-only SAC with an exact Gaussian-arm/Bernoulli-gripper policy.

    The critic consumes the exact binary command stored in replay.  Actor and
    target losses enumerate both gripper values, so no hard-thresholded latent
    Gaussian is mistaken for the action distribution seen by the simulator.
    """

    state_schema = "g2_teacher_hybrid_sac_agent_v1"

    def __init__(
        self, config: SACConfig, *, device: str | torch.device = "cpu"
    ) -> None:
        if config.action_dim != G2_TEACHER_ACTION_DIM:
            raise ValueError("G2 Teacher hybrid SAC requires action_dim=7")
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA SAC device requested but torch CUDA is unavailable")
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        self.actor = _G2TeacherHybridActor(config).to(self.device)
        self.critic_1 = QNetwork(config).to(self.device)
        self.critic_2 = QNetwork(config).to(self.device)
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)
        self.target_critic_1.requires_grad_(False)
        self.target_critic_2.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=config.critic_lr,
        )
        self.log_alpha = torch.tensor(
            math.log(config.initial_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.alpha_lr
        )
        self.update_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def target_entropy(self) -> float:
        if self.config.target_entropy is not None:
            return float(self.config.target_entropy)
        return -(float(G2_TEACHER_ARM_ACTION_DIM) + math.log(2.0))

    @staticmethod
    def _hybrid_expectation(
        arm_action: torch.Tensor,
        arm_log_probability: torch.Tensor,
        gripper_logit: torch.Tensor,
        critic_1: nn.Module,
        critic_2: nn.Module,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probability_open = torch.sigmoid(gripper_logit)
        closed = torch.cat((arm_action, -torch.ones_like(probability_open)), dim=-1)
        opened = torch.cat((arm_action, torch.ones_like(probability_open)), dim=-1)
        closed_q = torch.minimum(
            critic_1(observation, closed), critic_2(observation, closed)
        )
        open_q = torch.minimum(
            critic_1(observation, opened), critic_2(observation, opened)
        )
        expected_q = (1.0 - probability_open) * closed_q + probability_open * open_q
        binary_expected_logp = (
            (1.0 - probability_open)
            * torch.log((1.0 - probability_open).clamp_min(1.0e-8))
            + probability_open * torch.log(probability_open.clamp_min(1.0e-8))
        )
        return expected_q, arm_log_probability + binary_expected_logp, probability_open

    @staticmethod
    def _tensor(
        batch: Mapping[str, np.ndarray], name: str, device: torch.device
    ) -> torch.Tensor:
        if name not in batch:
            raise KeyError(f"missing SAC batch field: {name}")
        return torch.as_tensor(batch[name], dtype=torch.float32, device=device)

    @torch.no_grad()
    def select_actions(
        self, observations: np.ndarray, *, deterministic: bool = False
    ) -> np.ndarray:
        array = np.asarray(observations, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.config.observation_dim:
            raise ValueError(
                f"observations must have shape (N,{self.config.observation_dim})"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("observations contain non-finite values")
        action, _, mean_action = self.actor.sample(
            torch.from_numpy(array).to(self.device), deterministic=deterministic
        )
        selected = mean_action if deterministic else action
        return selected.cpu().numpy().astype(np.float32, copy=False)

    @torch.no_grad()
    def select_action(
        self, observation: np.ndarray, *, deterministic: bool = False
    ) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float32)
        if array.shape != (self.config.observation_dim,):
            raise ValueError(
                f"observation must have shape ({self.config.observation_dim},)"
            )
        return self.select_actions(
            array.reshape(1, -1), deterministic=deterministic
        )[0]

    def update(self, batch: Mapping[str, np.ndarray]) -> dict[str, float]:
        observation = self._tensor(batch, "observations", self.device)
        action = self._tensor(batch, "actions", self.device)
        reward = self._tensor(batch, "rewards", self.device)
        next_observation = self._tensor(batch, "next_observations", self.device)
        terminated = self._tensor(batch, "terminated", self.device)
        importance = (
            self._tensor(batch, "importance_weights", self.device)
            if "importance_weights" in batch
            else torch.ones_like(reward)
        )
        batch_size = observation.shape[0]
        if observation.shape != (batch_size, self.config.observation_dim):
            raise ValueError("invalid observations batch shape")
        if next_observation.shape != observation.shape:
            raise ValueError("invalid next_observations batch shape")
        if action.shape != (batch_size, G2_TEACHER_ACTION_DIM):
            raise ValueError("invalid actions batch shape")
        if reward.shape != (batch_size, 1) or terminated.shape != (batch_size, 1):
            raise ValueError("reward and terminated must have shape [batch,1]")
        if importance.shape != (batch_size, 1):
            raise ValueError("importance_weights must have shape [batch,1]")
        tensors = (observation, action, reward, next_observation, terminated, importance)
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("SAC batch contains non-finite values")
        if bool((importance <= 0.0).any()):
            raise ValueError("importance_weights must be positive")
        gripper = action[:, G2_TEACHER_GRIPPER_ACTION_INDEX]
        if bool((action[:, :G2_TEACHER_ARM_ACTION_DIM].abs() > 1.0).any()):
            raise ValueError("replay arm action must remain in [-1,1]")
        if not bool(torch.equal(gripper.abs(), torch.ones_like(gripper))):
            raise ValueError("replay gripper action must be exactly binary -1/+1")

        with torch.no_grad():
            next_arm, next_arm_logp, _, _, next_gripper_logit = self.actor.sample_arm(
                next_observation
            )
            target_q_value, next_logp, _ = self._hybrid_expectation(
                next_arm,
                next_arm_logp,
                next_gripper_logit,
                self.target_critic_1,
                self.target_critic_2,
                next_observation,
            )
            target_q = soft_bellman_target(
                reward,
                terminated,
                self.config.gamma,
                target_q_value - self.alpha.detach() * next_logp,
            )

        current_q_1 = self.critic_1(observation, action)
        current_q_2 = self.critic_2(observation, action)
        critic_loss = (
            importance * (current_q_1 - target_q).pow(2)
        ).mean() + (
            importance * (current_q_2 - target_q).pow(2)
        ).mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_gradient_norm = nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            self.config.gradient_clip,
        )
        self.critic_optimizer.step()

        self.critic_1.requires_grad_(False)
        self.critic_2.requires_grad_(False)
        arm, arm_logp, _, log_std, gripper_logit = self.actor.sample_arm(observation)
        policy_q, policy_logp, probability_open = self._hybrid_expectation(
            arm,
            arm_logp,
            gripper_logit,
            self.critic_1,
            self.critic_2,
            observation,
        )
        actor_loss = (self.alpha.detach() * policy_logp - policy_q).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_gradient_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.gradient_clip
        )
        self.actor_optimizer.step()
        self.critic_1.requires_grad_(True)
        self.critic_2.requires_grad_(True)

        alpha_loss = -(
            self.log_alpha * (policy_logp.detach() + self.target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_gradient_norm = nn.utils.clip_grad_norm_(
            [self.log_alpha], self.config.gradient_clip
        )
        self.alpha_optimizer.step()
        polyak_update(self.critic_1, self.target_critic_1, self.config.tau)
        polyak_update(self.critic_2, self.target_critic_2, self.config.tau)
        self.update_count += 1

        q_data = torch.cat((current_q_1.detach(), current_q_2.detach()), dim=1)
        td_error = torch.cat(
            (
                target_q.detach() - current_q_1.detach(),
                target_q.detach() - current_q_2.detach(),
            ),
            dim=1,
        )
        metrics = {
            "loss/critic": float(critic_loss.detach()),
            "loss/actor": float(actor_loss.detach()),
            "loss/alpha": float(alpha_loss.detach()),
            "temperature/alpha": float(self.alpha.detach()),
            "temperature/target_entropy": self.target_entropy,
            "policy/entropy": float(-policy_logp.detach().mean()),
            "policy/log_probability": float(policy_logp.detach().mean()),
            "policy/log_std_mean": float(log_std.detach().mean()),
            "policy/gripper_open_probability_mean": float(
                probability_open.detach().mean()
            ),
            "policy/gripper_binary_entropy_mean": float(
                -(
                    probability_open * torch.log(probability_open.clamp_min(1.0e-8))
                    + (1.0 - probability_open)
                    * torch.log((1.0 - probability_open).clamp_min(1.0e-8))
                ).detach().mean()
            ),
            "q/data_min": float(q_data.min()),
            "q/data_mean": float(q_data.mean()),
            "q/data_max": float(q_data.max()),
            "q/policy_mean": float(policy_q.detach().mean()),
            "q/target_mean": float(target_q.detach().mean()),
            "td_error/max_abs": float(td_error.abs().max()),
            "gradient/actor_norm": float(actor_gradient_norm),
            "gradient/critic_norm": float(critic_gradient_norm),
            "gradient/alpha_norm": float(alpha_gradient_norm),
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"non-finite hybrid SAC metric: {metrics}")
        return metrics

    def parameter_checksum(self) -> str:
        digest = hashlib.sha256()
        for module in (self.actor, self.critic_1, self.critic_2):
            for tensor in module.state_dict().values():
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.state_schema,
            "policy_distribution_schema": G2_TEACHER_POLICY_DISTRIBUTION_SCHEMA,
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != self.state_schema:
            raise ValueError("checkpoint is not an exact G2 Teacher hybrid SAC agent")
        if state.get("policy_distribution_schema") != G2_TEACHER_POLICY_DISTRIBUTION_SCHEMA:
            raise ValueError("checkpoint Teacher policy distribution differs")
        saved_config = dict(state["config"])
        current_config = asdict(self.config)
        for name in ("hidden_sizes", "action_mask"):
            if name in saved_config:
                saved_config[name] = tuple(saved_config[name])
        if saved_config != current_config:
            raise ValueError("checkpoint Teacher hybrid SAC config differs")
        self.actor.load_state_dict(state["actor"])
        self.critic_1.load_state_dict(state["critic_1"])
        self.critic_2.load_state_dict(state["critic_2"])
        self.target_critic_1.load_state_dict(state["target_critic_1"])
        self.target_critic_2.load_state_dict(state["target_critic_2"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.log_alpha.data.copy_(
            torch.as_tensor(state["log_alpha"], device=self.device)
        )
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.update_count = int(state["update_count"])
        for optimizer in (
            self.actor_optimizer,
            self.critic_optimizer,
            self.alpha_optimizer,
        ):
            for optimizer_state in optimizer.state.values():
                for key, value in optimizer_state.items():
                    if torch.is_tensor(value):
                        optimizer_state[key] = value.to(self.device)


@dataclass(frozen=True)
class G2TeacherTrainingCadence:
    """Units and update ratios for one vectorized Teacher SAC run."""

    physics_dt_s: float
    control_decimation: int
    num_envs: int
    learning_starts_transitions: int
    updates_per_vector_step: int = 1

    def validated(self) -> "G2TeacherTrainingCadence":
        if not math.isfinite(self.physics_dt_s) or self.physics_dt_s <= 0.0:
            raise ValueError("physics_dt_s must be finite and positive")
        for name in (
            "control_decimation",
            "num_envs",
            "learning_starts_transitions",
            "updates_per_vector_step",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        return self

    @property
    def policy_dt_s(self) -> float:
        return self.physics_dt_s * self.control_decimation

    @property
    def policy_hz(self) -> float:
        return 1.0 / self.policy_dt_s

    @property
    def optimizer_updates_per_transition(self) -> float:
        return self.updates_per_vector_step / self.num_envs

    def as_dict(self) -> dict[str, float | int | str]:
        self.validated()
        return {
            "physics_dt_s": self.physics_dt_s,
            "control_decimation": self.control_decimation,
            "policy_dt_s": self.policy_dt_s,
            "policy_hz": self.policy_hz,
            "num_envs": self.num_envs,
            "learning_starts": self.learning_starts_transitions,
            "learning_starts_unit": "TRANSITIONS_ACROSS_ALL_ENVS",
            "updates_per_vector_step": self.updates_per_vector_step,
            "optimizer_updates_per_transition": self.optimizer_updates_per_transition,
        }


@dataclass(frozen=True)
class G2TeacherCheckpointStorageContract:
    """Bound full replay storage while preserving 2k model checkpoints."""

    replay_capacity: int
    observation_dim: int
    action_dim: int

    @property
    def replay_payload_bytes_upper_bound(self) -> int:
        # observations + next observations + action + reward/done/priority
        # arrays + phase byte.  Pickle/container overhead is reported
        # separately by the actual file hash/size.
        return self.replay_capacity * (
            2 * self.observation_dim * 4
            + self.action_dim * 4
            + 4 + 4 + 4 + 4 + 1
        )

    def as_dict(self) -> dict[str, int | str | bool]:
        if self.replay_capacity <= 0 or self.observation_dim <= 0 or self.action_dim <= 0:
            raise ValueError("checkpoint storage dimensions must be positive")
        return {
            "schema": "g2_teacher_bounded_replay_checkpoint_storage_v1",
            "periodic_checkpoint_kind": "MODEL_ONLY_EVALUATION",
            "periodic_checkpoint_contains_replay": False,
            "rolling_recovery_kind": "FULL_AGENT_REPLAY_RNG_ATOMIC_REPLACE",
            "maximum_live_full_replay_files_during_run": 1,
            "maximum_full_replay_files_after_finalization": 2,
            "replay_payload_bytes_upper_bound_each": self.replay_payload_bytes_upper_bound,
        }


def integrate_reward_rate(
    weighted_reward_rate: torch.Tensor, policy_dt_s: float
) -> torch.Tensor:
    """Convert Isaac Lab RewardManager's weighted rate to transition reward."""

    if not math.isfinite(policy_dt_s) or policy_dt_s <= 0.0:
        raise ValueError("policy_dt_s must be finite and positive")
    if not bool(torch.isfinite(weighted_reward_rate).all()):
        raise ValueError("weighted reward rate contains non-finite values")
    return weighted_reward_rate * float(policy_dt_s)


@dataclass(frozen=True)
class G2TeacherObservationContract:
    """Ordered, privileged state supplied to the state-only SAC actor."""

    controlled_joint_count: int = 8
    phase_feature_count: int = 4
    contact_force_scale_n: float = 100.0

    @property
    def fields(self) -> tuple[tuple[str, int], ...]:
        return (
            ("controlled_joint_position_relative_rad", self.controlled_joint_count),
            ("controlled_joint_velocity_rad_s", self.controlled_joint_count),
            ("end_effector_pose_root_xyzw", 7),
            ("cube_pose_root_xyzw", 7),
            ("cube_linear_angular_velocity", 6),
            ("goal_position_root_m", 3),
            ("end_effector_to_cube_m", 3),
            ("cube_to_goal_m", 3),
            ("bilateral_contact_features", 3),
            ("curriculum_phase_features", self.phase_feature_count),
            ("previous_action", G2_TEACHER_ACTION_DIM),
        )

    @property
    def observation_dim(self) -> int:
        return sum(width for _, width in self.fields)

    @property
    def slices(self) -> Mapping[str, slice]:
        result: dict[str, slice] = {}
        cursor = 0
        for name, width in self.fields:
            result[name] = slice(cursor, cursor + width)
            cursor += width
        return result

    def build(
        self,
        *,
        joint_position_relative_rad: torch.Tensor,
        joint_velocity_rad_s: torch.Tensor,
        end_effector_pose_root_xyzw: torch.Tensor,
        cube_pose_root_xyzw: torch.Tensor,
        cube_linear_angular_velocity: torch.Tensor,
        goal_position_root_m: torch.Tensor,
        bilateral_contact_force_n: torch.Tensor,
        curriculum_phase_features: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        """Build one batched observation without image or student-only data."""

        tensors = {
            "joint_position_relative_rad": joint_position_relative_rad,
            "joint_velocity_rad_s": joint_velocity_rad_s,
            "end_effector_pose_root_xyzw": end_effector_pose_root_xyzw,
            "cube_pose_root_xyzw": cube_pose_root_xyzw,
            "cube_linear_angular_velocity": cube_linear_angular_velocity,
            "goal_position_root_m": goal_position_root_m,
            "bilateral_contact_force_n": bilateral_contact_force_n,
            "curriculum_phase_features": curriculum_phase_features,
            "previous_action": previous_action,
        }
        batch = joint_position_relative_rad.shape[0]
        expected = {
            "joint_position_relative_rad": self.controlled_joint_count,
            "joint_velocity_rad_s": self.controlled_joint_count,
            "end_effector_pose_root_xyzw": 7,
            "cube_pose_root_xyzw": 7,
            "cube_linear_angular_velocity": 6,
            "goal_position_root_m": 3,
            "bilateral_contact_force_n": 2,
            "curriculum_phase_features": self.phase_feature_count,
            "previous_action": G2_TEACHER_ACTION_DIM,
        }
        for name, tensor in tensors.items():
            if tensor.ndim != 2 or tuple(tensor.shape) != (batch, expected[name]):
                raise ValueError(
                    f"{name} must have shape {(batch, expected[name])}; "
                    f"got {tuple(tensor.shape)}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains non-finite values")

        # Dataset/replay quaternions have one normalized, deterministic XYZW
        # representation.  This prevents q/-q sign flips from becoming false
        # observation discontinuities and rejects invalid zero quaternions.
        end_effector_pose_root_xyzw = torch.cat(
            (
                end_effector_pose_root_xyzw[:, :3],
                canonicalize_quaternion_xyzw(end_effector_pose_root_xyzw[:, 3:7]),
            ),
            dim=-1,
        )
        cube_pose_root_xyzw = torch.cat(
            (
                cube_pose_root_xyzw[:, :3],
                canonicalize_quaternion_xyzw(cube_pose_root_xyzw[:, 3:7]),
            ),
            dim=-1,
        )

        ee_position = end_effector_pose_root_xyzw[:, :3]
        cube_position = cube_pose_root_xyzw[:, :3]
        force = torch.clamp(
            bilateral_contact_force_n / self.contact_force_scale_n, 0.0, 1.0
        )
        bilateral = (
            (bilateral_contact_force_n[:, 0] > 1.0)
            & (bilateral_contact_force_n[:, 1] > 1.0)
        ).to(force.dtype).unsqueeze(-1)
        contact_features = torch.cat((force, bilateral), dim=-1)
        observation = torch.cat(
            (
                joint_position_relative_rad,
                joint_velocity_rad_s,
                end_effector_pose_root_xyzw,
                cube_pose_root_xyzw,
                cube_linear_angular_velocity,
                goal_position_root_m,
                cube_position - ee_position,
                goal_position_root_m - cube_position,
                contact_features,
                curriculum_phase_features,
                previous_action,
            ),
            dim=-1,
        )
        if observation.shape[-1] != self.observation_dim:
            raise RuntimeError("teacher observation assembly dimension mismatch")
        return observation

    def validate_flat_observation(self, observation: torch.Tensor) -> None:
        """Fail closed on persisted/replay observations outside this schema."""

        if observation.ndim < 1 or observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"teacher observation must end in {self.observation_dim} fields"
            )
        if not bool(torch.isfinite(observation).all()):
            raise ValueError("teacher observation contains non-finite values")
        for pose_name in (
            "end_effector_pose_root_xyzw",
            "cube_pose_root_xyzw",
        ):
            pose = observation[..., self.slices[pose_name]]
            canonical = canonicalize_quaternion_xyzw(pose[..., 3:7])
            if not bool(torch.allclose(pose[..., 3:7], canonical, atol=1.0e-6, rtol=1.0e-6)):
                raise ValueError(f"{pose_name} is not canonical normalized XYZW")
        ee = observation[..., self.slices["end_effector_pose_root_xyzw"]][..., :3]
        cube = observation[..., self.slices["cube_pose_root_xyzw"]][..., :3]
        goal = observation[..., self.slices["goal_position_root_m"]]
        if not bool(
            torch.allclose(
                observation[..., self.slices["end_effector_to_cube_m"]],
                cube - ee,
                atol=1.0e-5,
                rtol=1.0e-5,
            )
        ):
            raise ValueError("end_effector_to_cube_m is inconsistent with poses")
        if not bool(
            torch.allclose(
                observation[..., self.slices["cube_to_goal_m"]],
                goal - cube,
                atol=1.0e-5,
                rtol=1.0e-5,
            )
        ):
            raise ValueError("cube_to_goal_m is inconsistent with poses")
        contact = observation[..., self.slices["bilateral_contact_features"]]
        phase = observation[..., self.slices["curriculum_phase_features"]]
        if bool(((contact < 0.0) | (contact > 1.0)).any()):
            raise ValueError("bilateral contact features must be in [0,1]")
        if bool(((phase < 0.0) | (phase > 1.0)).any()):
            raise ValueError("curriculum phase features must be in [0,1]")


__all__ = [
    "G2_TEACHER_ACTION_DIM",
    "G2_TEACHER_ARM_ACTION_DIM",
    "G2_TEACHER_GRIPPER_ACTION_INDEX",
    "G2_TEACHER_ACTION_SCHEMA",
    "G2_TEACHER_POLICY_DISTRIBUTION_SCHEMA",
    "G2_TEACHER_STATEFUL_ACTION_TRANSFORM_SCHEMA",
    "G2_TEACHER_OBSERVATION_SCHEMA",
    "G2TeacherActionContract",
    "G2TeacherStatefulActionTransform",
    "G2TeacherCheckpointStorageContract",
    "G2TeacherHybridSACAgent",
    "G2TeacherObservationContract",
    "G2TeacherTrainingCadence",
    "integrate_reward_rate",
]
