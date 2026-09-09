"""Small, dependency-light Soft Actor-Critic implementation for Stage 2.

This module contains no simulator or middleware imports.  It intentionally
implements only online, state-based, reward-driven SAC: a tanh-squashed
Gaussian actor, independent twin critics and targets, automatic entropy
tuning, a transition replay buffer, and complete checkpoint state.

The equations and default hyperparameters follow the checked-out RLinf SAC
implementation/config, while avoiding its ROS/MuJoCo GenieSim worker path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


STAGE2_OBSERVATION_DIM = 76
STAGE2_ACTION_DIM = 7
CHECKPOINT_VERSION = 1


def _positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class SACConfig:
    """Configuration for a state-based continuous-action SAC agent."""

    observation_dim: int = STAGE2_OBSERVATION_DIM
    critic_observation_dim: int | None = None
    action_dim: int = STAGE2_ACTION_DIM
    hidden_sizes: tuple[int, ...] = (256, 256)
    gamma: float = 0.96
    tau: float = 0.005
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    initial_alpha: float = 0.01
    target_entropy: float | None = None
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    gradient_clip: float = 5.0
    pose_auxiliary_weight: float = 0.1
    pose_auxiliary_rotation_weight: float = 0.1
    action_mask: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    seed: int = 42

    def __post_init__(self) -> None:
        if self.observation_dim <= 0 or self.action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        if self.critic_observation_dim is not None and (
            self.critic_observation_dim < self.observation_dim
        ):
            raise ValueError(
                "critic_observation_dim must include the complete actor observation"
            )
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive widths")
        if not math.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1]")
        if not math.isfinite(self.tau) or not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be finite and in (0, 1]")
        for name in (
            "actor_lr",
            "critic_lr",
            "alpha_lr",
            "initial_alpha",
            "gradient_clip",
            "pose_auxiliary_weight",
            "pose_auxiliary_rotation_weight",
        ):
            _positive_finite(name, float(getattr(self, name)))
        if not math.isfinite(self.log_std_min) or not math.isfinite(
            self.log_std_max
        ):
            raise ValueError("log_std bounds must be finite")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max")
        if len(self.action_mask) != self.action_dim:
            raise ValueError("action_mask length must equal action_dim")
        mask = np.asarray(self.action_mask, dtype=np.float64)
        if not np.all(np.isfinite(mask)) or not np.all((mask == 0.0) | (mask == 1.0)):
            raise ValueError("action_mask entries must be exactly 0 or 1")
        if not np.any(mask):
            raise ValueError("action_mask must enable at least one action dimension")
        if self.target_entropy is not None and not math.isfinite(
            self.target_entropy
        ):
            raise ValueError("target_entropy must be finite when supplied")

    @property
    def resolved_target_entropy(self) -> float:
        if self.target_entropy is not None:
            return float(self.target_entropy)
        return -float(sum(self.action_mask))

    @property
    def resolved_critic_observation_dim(self) -> int:
        return int(self.critic_observation_dim or self.observation_dim)


def _mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_sizes:
        linear = nn.Linear(previous, width)
        nn.init.orthogonal_(linear.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(linear.bias)
        layers.extend((linear, nn.ReLU()))
        previous = width
    output = nn.Linear(previous, output_dim)
    nn.init.orthogonal_(output.weight, gain=1.0)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


def squashed_gaussian_log_prob(
    pre_tanh: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the tanh-corrected diagonal Gaussian log probability.

    The stable Jacobian expression is the one used by the original SAC
    implementation and CleanRL-style policies.  The result has shape ``[B,1]``.
    """

    if pre_tanh.shape != mean.shape or mean.shape != log_std.shape:
        raise ValueError("pre_tanh, mean, and log_std must have identical shapes")
    normal = torch.distributions.Normal(mean, log_std.exp())
    per_dimension = normal.log_prob(pre_tanh)
    correction = 2.0 * (
        math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh)
    )
    per_dimension = per_dimension - correction
    if action_mask is not None:
        if action_mask.ndim != 1 or action_mask.shape[0] != pre_tanh.shape[-1]:
            raise ValueError("action_mask must be one-dimensional and match action dim")
        per_dimension = per_dimension * action_mask
    return per_dimension.sum(dim=-1, keepdim=True)


class SquashedGaussianActor(nn.Module):
    """MLP actor producing a reparameterized action in ``[-1, 1]``."""

    def __init__(self, config: SACConfig) -> None:
        super().__init__()
        self.config = config
        trunk_widths = config.hidden_sizes
        layers: list[nn.Module] = []
        previous = config.observation_dim
        for width in trunk_widths:
            linear = nn.Linear(previous, width)
            nn.init.orthogonal_(linear.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.ReLU()))
            previous = width
        self.trunk = nn.Sequential(*layers)
        self.mean = nn.Linear(previous, config.action_dim)
        self.log_std = nn.Linear(previous, config.action_dim)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        nn.init.orthogonal_(self.log_std.weight, gain=0.01)
        nn.init.constant_(self.log_std.bias, -2.0)
        self.register_buffer(
            "action_mask",
            torch.as_tensor(config.action_mask, dtype=torch.float32),
        )

    def distribution_parameters(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(observation)
        mean = self.mean(features)
        raw_log_std = torch.tanh(self.log_std(features))
        log_std = self.config.log_std_min + 0.5 * (
            self.config.log_std_max - self.config.log_std_min
        ) * (raw_log_std + 1.0)
        return mean, log_std

    def sample(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution_parameters(observation)
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = torch.distributions.Normal(mean, log_std.exp()).rsample()
        action = torch.tanh(pre_tanh) * self.action_mask
        mean_action = torch.tanh(mean) * self.action_mask
        log_probability = squashed_gaussian_log_prob(
            pre_tanh,
            mean,
            log_std,
            self.action_mask,
        )
        return action, log_probability, mean_action


class QNetwork(nn.Module):
    """One scalar state-action critic."""

    def __init__(self, config: SACConfig) -> None:
        super().__init__()
        self.network = _mlp(
            config.resolved_critic_observation_dim + config.action_dim,
            config.hidden_sizes,
            1,
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((observation, action), dim=-1))


class VisualPoseAuxiliaryHead(nn.Module):
    """Predict object position/rotation from camera-derived 53-D fields."""

    def __init__(self, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.network = _mlp(7, hidden_sizes, 9)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # [23:26]=fused visual centroid, [26:29]=surface normal,
        # [29]=multi-camera confidence.  No privileged state enters here.
        return self.network(observation[..., 23:30])


def _rotation6d_matrix(value: torch.Tensor) -> torch.Tensor:
    first = F.normalize(value[..., :3], dim=-1)
    second_raw = value[..., 3:6]
    second = F.normalize(
        second_raw - (first * second_raw).sum(-1, keepdim=True) * first,
        dim=-1,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def _quaternion_wxyz_matrix(value: torch.Tensor) -> torch.Tensor:
    q = F.normalize(value, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def soft_bellman_target(
    reward: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
    target_value: torch.Tensor,
) -> torch.Tensor:
    """SAC backup that bootstraps through time-limit truncation.

    Callers deliberately pass only ``terminated``.  A separate ``truncated``
    flag is stored in replay but must not enter this mask.
    """

    return reward + float(gamma) * (1.0 - terminated) * target_value


@torch.no_grad()
def polyak_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    """Apply one in-place Polyak update to ``target``."""

    if not math.isfinite(tau) or not 0.0 < tau <= 1.0:
        raise ValueError("tau must be finite and in (0, 1]")
    for source, destination in zip(
        online.parameters(), target.parameters(), strict=True
    ):
        destination.mul_(1.0 - tau).add_(source, alpha=tau)


class SACAgent:
    """Twin-Q SAC learner with automatic entropy tuning."""

    def __init__(self, config: SACConfig, *, device: str | torch.device = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA SAC device requested but torch CUDA is unavailable")
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        self.actor = SquashedGaussianActor(config).to(self.device)
        self.visual_pose_auxiliary = VisualPoseAuxiliaryHead(
            config.hidden_sizes
        ).to(self.device)
        self.critic_1 = QNetwork(config).to(self.device)
        self.critic_2 = QNetwork(config).to(self.device)
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)
        self.target_critic_1.requires_grad_(False)
        self.target_critic_2.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.visual_pose_auxiliary.parameters()),
            lr=config.actor_lr,
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

    def zero_initialize_residual_mean_head(self) -> None:
        """Make the fresh deterministic residual exactly zero at step zero."""

        nn.init.zeros_(self.actor.mean.weight)
        nn.init.zeros_(self.actor.mean.bias)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float32)
        if array.shape != (self.config.observation_dim,):
            raise ValueError(
                f"observation must have shape ({self.config.observation_dim},), "
                f"got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("observation contains non-finite values")
        with torch.no_grad():
            tensor = torch.from_numpy(array).to(self.device).unsqueeze(0)
            action, _, mean_action = self.actor.sample(
                tensor, deterministic=deterministic
            )
            selected = mean_action if deterministic else action
        return selected.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    def select_actions(
        self,
        observations: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select a batch of actions with one device transfer and actor call."""

        array = np.asarray(observations, dtype=np.float32)
        expected = (array.shape[0], self.config.observation_dim) if array.ndim == 2 else None
        if expected is None or array.shape != expected:
            raise ValueError(
                "observations must have shape (N, "
                f"{self.config.observation_dim}); got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("observations contain non-finite values")
        with torch.no_grad():
            tensor = torch.from_numpy(array).to(self.device)
            action, _, mean_action = self.actor.sample(
                tensor, deterministic=deterministic
            )
            selected = mean_action if deterministic else action
        return selected.cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _tensor(
        batch: Mapping[str, np.ndarray], name: str, device: torch.device
    ) -> torch.Tensor:
        if name not in batch:
            raise KeyError(f"missing SAC batch field: {name}")
        return torch.as_tensor(batch[name], dtype=torch.float32, device=device)

    def update(self, batch: Mapping[str, np.ndarray]) -> dict[str, float]:
        observation = self._tensor(batch, "observations", self.device)
        action = self._tensor(batch, "actions", self.device)
        reward = self._tensor(batch, "rewards", self.device)
        next_observation = self._tensor(batch, "next_observations", self.device)
        critic_observation = (
            self._tensor(batch, "critic_observations", self.device)
            if "critic_observations" in batch
            else observation
        )
        next_critic_observation = (
            self._tensor(batch, "next_critic_observations", self.device)
            if "next_critic_observations" in batch
            else next_observation
        )
        terminated = self._tensor(batch, "terminated", self.device)
        importance_weights = (
            self._tensor(batch, "importance_weights", self.device)
            if "importance_weights" in batch
            else torch.ones_like(reward)
        )

        expected_observation = (observation.shape[0], self.config.observation_dim)
        expected_action = (observation.shape[0], self.config.action_dim)
        if observation.shape != expected_observation:
            raise ValueError("invalid observations batch shape")
        if next_observation.shape != expected_observation:
            raise ValueError("invalid next_observations batch shape")
        if action.shape != expected_action:
            raise ValueError("invalid actions batch shape")
        expected_critic = (
            observation.shape[0], self.config.resolved_critic_observation_dim
        )
        if critic_observation.shape != expected_critic:
            raise ValueError("invalid critic_observations batch shape")
        if next_critic_observation.shape != expected_critic:
            raise ValueError("invalid next_critic_observations batch shape")
        if reward.shape != (observation.shape[0], 1):
            raise ValueError("rewards must have shape [batch, 1]")
        if terminated.shape != (observation.shape[0], 1):
            raise ValueError("terminated must have shape [batch, 1]")
        if importance_weights.shape != (observation.shape[0], 1):
            raise ValueError("importance_weights must have shape [batch, 1]")
        for tensor in (
            observation, critic_observation, action, reward, next_observation,
            next_critic_observation, terminated,
            importance_weights,
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError("SAC batch contains non-finite values")
        if torch.any(importance_weights <= 0.0):
            raise ValueError("importance_weights must be positive")

        with torch.no_grad():
            next_action, next_log_probability, _ = self.actor.sample(
                next_observation
            )
            target_q_1 = self.target_critic_1(
                next_critic_observation, next_action
            )
            target_q_2 = self.target_critic_2(
                next_critic_observation, next_action
            )
            target_soft_value = (
                torch.minimum(target_q_1, target_q_2)
                - self.alpha.detach() * next_log_probability
            )
            target_q = soft_bellman_target(
                reward,
                terminated,
                self.config.gamma,
                target_soft_value,
            )

        current_q_1 = self.critic_1(critic_observation, action)
        current_q_2 = self.critic_2(critic_observation, action)
        # CEBP/HER-force sampling changes only which replay rows are drawn.
        # Importance sampling corrects that replay bias in the Bellman loss;
        # reward, success and physical termination remain untouched.
        critic_loss = (
            importance_weights * (current_q_1 - target_q).pow(2)
        ).mean() + (
            importance_weights * (current_q_2 - target_q).pow(2)
        ).mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            self.config.gradient_clip,
        )
        self.critic_optimizer.step()

        self.critic_1.requires_grad_(False)
        self.critic_2.requires_grad_(False)
        policy_action, log_probability, _ = self.actor.sample(observation)
        policy_q_1 = self.critic_1(critic_observation, policy_action)
        policy_q_2 = self.critic_2(critic_observation, policy_action)
        policy_q = torch.minimum(policy_q_1, policy_q_2)
        sac_actor_loss = (
            self.alpha.detach() * log_probability - policy_q
        ).mean()
        pose_prediction = self.visual_pose_auxiliary(observation)
        if self.config.resolved_critic_observation_dim > self.config.observation_dim:
            privileged_pose = critic_observation[
                :, self.config.observation_dim : self.config.observation_dim + 7
            ]
            position_error = torch.abs(
                pose_prediction[:, :3] - privileged_pose[:, :3]
            ).mean()
            predicted_rotation = _rotation6d_matrix(pose_prediction[:, 3:9])
            target_rotation = _quaternion_wxyz_matrix(privileged_pose[:, 3:7])
            relative_trace = torch.diagonal(
                target_rotation.transpose(-1, -2) @ predicted_rotation,
                dim1=-2,
                dim2=-1,
            ).sum(-1)
            rotation_error = torch.acos(
                torch.clamp((relative_trace - 1.0) * 0.5, -1.0 + 1e-6, 1.0 - 1e-6)
            ).mean()
            pose_auxiliary_loss = position_error + (
                self.config.pose_auxiliary_rotation_weight * rotation_error
            )
        else:
            position_error = torch.zeros((), device=self.device)
            rotation_error = torch.zeros((), device=self.device)
            pose_auxiliary_loss = torch.zeros((), device=self.device)
        actor_loss = sac_actor_loss + (
            self.config.pose_auxiliary_weight * pose_auxiliary_loss
        )
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.visual_pose_auxiliary.parameters()),
            self.config.gradient_clip,
        )
        self.actor_optimizer.step()
        self.critic_1.requires_grad_(True)
        self.critic_2.requires_grad_(True)

        alpha_loss = -(
            self.log_alpha
            * (log_probability + self.config.resolved_target_entropy).detach()
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_gradient_norm = torch.nn.utils.clip_grad_norm_(
            [self.log_alpha], self.config.gradient_clip
        )
        self.alpha_optimizer.step()

        polyak_update(self.critic_1, self.target_critic_1, self.config.tau)
        polyak_update(self.critic_2, self.target_critic_2, self.config.tau)
        self.update_count += 1

        q_data = torch.cat((current_q_1.detach(), current_q_2.detach()), dim=1)
        td_error_1 = target_q.detach() - current_q_1.detach()
        td_error_2 = target_q.detach() - current_q_2.detach()
        td_error = torch.cat((td_error_1, td_error_2), dim=1)
        with torch.no_grad():
            _, policy_log_std = self.actor.distribution_parameters(observation)
            action_saturation_ratio = torch.mean(
                (torch.abs(policy_action) >= 0.95).to(torch.float32)
            )
        metrics = {
            "loss/critic": float(critic_loss.detach()),
            "loss/actor": float(actor_loss.detach()),
            "loss/actor_sac": float(sac_actor_loss.detach()),
            "auxiliary/pose_loss": float(pose_auxiliary_loss.detach()),
            "auxiliary/position_error_m": float(position_error.detach()),
            "auxiliary/rotation_error_rad": float(rotation_error.detach()),
            "loss/alpha": float(alpha_loss.detach()),
            "temperature/alpha": float(self.alpha.detach()),
            "policy/entropy": float(-log_probability.detach().mean()),
            "policy/log_probability": float(log_probability.detach().mean()),
            "q/data_min": float(q_data.min()),
            "q/data_mean": float(q_data.mean()),
            "q/data_max": float(q_data.max()),
            "q/q1_mean": float(current_q_1.detach().mean()),
            "q/q1_std": float(current_q_1.detach().std(unbiased=False)),
            "q/q1_max": float(current_q_1.detach().max()),
            "q/q2_mean": float(current_q_2.detach().mean()),
            "q/q2_std": float(current_q_2.detach().std(unbiased=False)),
            "q/q2_max": float(current_q_2.detach().max()),
            "q/policy_mean": float(policy_q.detach().mean()),
            "q/target_min": float(target_q.min()),
            "q/target_mean": float(target_q.mean()),
            "q/target_max": float(target_q.max()),
            "td_error/mean": float(td_error.mean()),
            "td_error/std": float(td_error.std(unbiased=False)),
            "td_error/max_abs": float(torch.abs(td_error).max()),
            "gradient/actor_norm": float(actor_gradient_norm),
            "gradient/critic_norm": float(critic_gradient_norm),
            "gradient/alpha_norm": float(alpha_gradient_norm),
            "temperature/target_entropy": float(
                self.config.resolved_target_entropy
            ),
            "policy/log_std_mean": float(policy_log_std.mean()),
            "policy/log_std_min": float(policy_log_std.min()),
            "policy/log_std_max": float(policy_log_std.max()),
            "policy/action_saturation_ratio": float(action_saturation_ratio),
            "replay/importance_weight_mean": float(
                importance_weights.detach().mean()
            ),
            "replay/importance_weight_min": float(
                importance_weights.detach().min()
            ),
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"non-finite SAC metric: {metrics}")
        return metrics

    def parameter_checksum(self) -> str:
        digest = hashlib.sha256()
        for module in (
            self.actor, self.visual_pose_auxiliary, self.critic_1, self.critic_2
        ):
            for tensor in module.state_dict().values():
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "visual_pose_auxiliary": self.visual_pose_auxiliary.state_dict(),
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
        saved_config = dict(state["config"])
        current_config = asdict(self.config)
        # Version-1 symmetric checkpoints predate the explicit critic input
        # dimension.  They remain loadable only into a symmetric agent.
        saved_config.setdefault("critic_observation_dim", None)
        if set(saved_config) != set(current_config):
            raise ValueError("checkpoint SAC config fields do not match")
        for name, current_value in current_config.items():
            saved_value = saved_config[name]
            if name in ("hidden_sizes", "action_mask"):
                matches = tuple(saved_value) == tuple(current_value)
            else:
                matches = saved_value == current_value
            if not matches:
                raise ValueError(f"checkpoint SAC config mismatch for {name}")
        self.actor.load_state_dict(state["actor"])
        self.visual_pose_auxiliary.load_state_dict(state["visual_pose_auxiliary"])
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
        self._move_optimizer_state(self.actor_optimizer)
        self._move_optimizer_state(self.critic_optimizer)
        self._move_optimizer_state(self.alpha_optimizer)

    def load_actor_only(
        self,
        state: Mapping[str, Any],
        *,
        zero_initialize_output: bool,
    ) -> None:
        """Load only an actor while keeping fresh critics/targets/alpha/replay."""

        actor_state = state.get("actor")
        if not isinstance(actor_state, Mapping):
            raise ValueError("actor-only checkpoint lacks actor parameters")
        self.actor.load_state_dict(actor_state)
        if zero_initialize_output:
            nn.init.zeros_(self.actor.mean.weight)
            nn.init.zeros_(self.actor.mean.bias)
        # Never import the old actor optimizer moments into a fresh run.
        self.actor_optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.visual_pose_auxiliary.parameters()),
            lr=self.config.actor_lr,
        )
        self.update_count = 0

    def _move_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)


class ReplayBuffer:
    """Seeded circular transition replay with exact resumable RNG state."""

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        *,
        seed: int,
        critic_observation_dim: int | None = None,
    ) -> None:
        if capacity <= 0 or observation_dim <= 0 or action_dim <= 0:
            raise ValueError("replay dimensions and capacity must be positive")
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.critic_observation_dim = (
            None
            if critic_observation_dim is None
            else int(critic_observation_dim)
        )
        if self.critic_observation_dim is not None and (
            self.critic_observation_dim < self.observation_dim
        ):
            raise ValueError("critic replay observation cannot omit actor fields")
        self.observations = np.empty(
            (capacity, observation_dim), dtype=np.float32
        )
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_observations = np.empty(
            (capacity, observation_dim), dtype=np.float32
        )
        self.critic_observations = (
            None
            if self.critic_observation_dim is None
            else np.empty(
                (capacity, self.critic_observation_dim), dtype=np.float32
            )
        )
        self.next_critic_observations = (
            None
            if self.critic_observation_dim is None
            else np.empty(
                (capacity, self.critic_observation_dim), dtype=np.float32
            )
        )
        self.terminated = np.empty((capacity, 1), dtype=np.float32)
        self.truncated = np.empty((capacity, 1), dtype=np.float32)
        self.priorities = np.ones((capacity,), dtype=np.float32)
        self.phase_labels = np.zeros((capacity,), dtype=np.int8)
        # Monotonic insertion identity prevents delayed episode-level priority
        # updates from mutating a circular slot that another environment (or a
        # HER row) has already overwritten.
        self.insertion_ids = np.full((capacity,), -1, dtype=np.int64)
        self.phase_names = ("REACH", "CONTACT", "STABLE_GRASP", "LIFT")
        self.phase_to_id = {
            name: index for index, name in enumerate(self.phase_names)
        }
        self.phase_stratification_enabled = False
        self.prioritized = False
        self.priority_alpha = 0.6
        self.importance_beta = 0.4
        self.uniform_mix = 0.5
        self.position = 0
        self.size = 0
        self.total_inserted = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def ready(self, minimum_transitions: int) -> bool:
        if minimum_transitions < 0:
            raise ValueError("minimum_transitions cannot be negative")
        return self.size >= minimum_transitions

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
        *,
        priority: float = 1.0,
        critic_observation: np.ndarray | None = None,
        next_critic_observation: np.ndarray | None = None,
        phase_label: str = "REACH",
    ) -> int:
        validated = self._validate_transition(
            observation,
            action,
            reward,
            next_observation,
            terminated,
            truncated,
        )
        critic_validated = self._validate_critic_transition(
            critic_observation, next_critic_observation
        )
        return self._commit_validated(
            validated,
            priority=priority,
            critic_transition=critic_validated,
            phase_label=phase_label,
        )

    def configure_phase_stratification(self, *, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("phase stratification enabled flag must be bool")
        self.phase_stratification_enabled = enabled

    def _phase_id(self, phase_label: str) -> int:
        if not isinstance(phase_label, str) or phase_label not in self.phase_to_id:
            raise ValueError(
                "replay phase_label must be REACH, CONTACT, STABLE_GRASP, or LIFT"
            )
        return int(self.phase_to_id[phase_label])

    def _validate_critic_transition(
        self,
        critic_observation: np.ndarray | None,
        next_critic_observation: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if self.critic_observation_dim is None:
            if critic_observation is not None or next_critic_observation is not None:
                raise ValueError("symmetric replay received privileged critic state")
            return None
        if critic_observation is None or next_critic_observation is None:
            raise ValueError("asymmetric replay requires both critic observations")
        current = np.asarray(critic_observation, dtype=np.float32)
        following = np.asarray(next_critic_observation, dtype=np.float32)
        expected = (self.critic_observation_dim,)
        if current.shape != expected or following.shape != expected:
            raise ValueError("invalid replay critic observation shape")
        if not np.all(np.isfinite(current)) or not np.all(np.isfinite(following)):
            raise ValueError("replay critic observation contains non-finite values")
        return (
            np.ascontiguousarray(current.copy()),
            np.ascontiguousarray(following.copy()),
        )

    def configure_prioritization(
        self,
        *,
        enabled: bool,
        alpha: float = 0.6,
        beta: float = 0.4,
        uniform_mix: float = 0.5,
    ) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("prioritization enabled flag must be bool")
        for name, value in (("alpha", alpha), ("beta", beta)):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"priority {name} must be within [0, 1]")
        if (
            not math.isfinite(float(uniform_mix))
            or not 0.0 <= float(uniform_mix) <= 1.0
        ):
            raise ValueError("uniform_mix must be within [0, 1]")
        self.prioritized = enabled
        self.priority_alpha = float(alpha)
        self.importance_beta = float(beta)
        self.uniform_mix = float(uniform_mix)

    def _validate_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, bool]:
        observation_array = np.asarray(observation, dtype=np.float32)
        next_array = np.asarray(next_observation, dtype=np.float32)
        action_array = np.asarray(action, dtype=np.float32)
        if observation_array.shape != (self.observation_dim,):
            raise ValueError("invalid replay observation shape")
        if next_array.shape != (self.observation_dim,):
            raise ValueError("invalid replay next_observation shape")
        if action_array.shape != (self.action_dim,):
            raise ValueError("invalid replay action shape")
        if not all(
            np.all(np.isfinite(value))
            for value in (observation_array, next_array, action_array)
        ) or not math.isfinite(float(reward)):
            raise ValueError("replay transition contains non-finite values")
        return (
            np.ascontiguousarray(observation_array.copy()),
            np.ascontiguousarray(action_array.copy()),
            float(reward),
            np.ascontiguousarray(next_array.copy()),
            bool(terminated),
            bool(truncated),
        )

    def _commit_validated(
        self,
        transition: tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, bool],
        *,
        priority: float = 1.0,
        critic_transition: tuple[np.ndarray, np.ndarray] | None = None,
        phase_label: str = "REACH",
    ) -> int:
        if not math.isfinite(float(priority)) or float(priority) <= 0.0:
            raise ValueError("replay priority must be finite and positive")
        (
            observation_array,
            action_array,
            reward,
            next_array,
            terminated,
            truncated,
        ) = transition
        index = self.position
        self.observations[index] = observation_array
        self.actions[index] = action_array
        self.rewards[index, 0] = np.float32(reward)
        self.next_observations[index] = next_array
        if self.critic_observation_dim is not None:
            if critic_transition is None:
                raise ValueError("asymmetric replay commit lacks critic state")
            assert self.critic_observations is not None
            assert self.next_critic_observations is not None
            self.critic_observations[index] = critic_transition[0]
            self.next_critic_observations[index] = critic_transition[1]
        self.terminated[index, 0] = np.float32(bool(terminated))
        self.truncated[index, 0] = np.float32(bool(truncated))
        self.priorities[index] = np.float32(priority)
        self.phase_labels[index] = np.int8(self._phase_id(phase_label))
        self.insertion_ids[index] = np.int64(self.total_inserted)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_inserted += 1
        return index

    def set_recent_priorities(self, insertion_count: int, priority: float) -> None:
        """Assign one episode priority to its most recent physical/HER rows."""

        if (
            isinstance(insertion_count, bool)
            or not isinstance(insertion_count, int)
            or insertion_count <= 0
            or insertion_count > self.size
            or insertion_count > self.capacity
        ):
            raise ValueError("recent replay insertion count is invalid")
        if not math.isfinite(float(priority)) or float(priority) <= 0.0:
            raise ValueError("replay priority must be finite and positive")
        indices = (
            self.position - insertion_count + np.arange(insertion_count)
        ) % self.capacity
        self.priorities[indices] = np.float32(priority)

    def recent_indices(self, insertion_count: int) -> np.ndarray:
        """Return circular row indices for the newest committed insertions."""

        if (
            isinstance(insertion_count, bool)
            or not isinstance(insertion_count, int)
            or insertion_count <= 0
            or insertion_count > self.size
            or insertion_count > self.capacity
        ):
            raise ValueError("recent replay insertion count is invalid")
        return (
            self.position - insertion_count + np.arange(insertion_count)
        ) % self.capacity

    def set_priorities(
        self, indices: Sequence[int] | np.ndarray, priority: float
    ) -> None:
        """Assign one episode priority to explicit interleaved replay rows."""

        rows = np.asarray(indices, dtype=np.int64)
        if (
            rows.ndim != 1
            or rows.size == 0
            or np.any(rows < 0)
            or np.any(rows >= self.capacity)
            or len(np.unique(rows)) != rows.size
        ):
            raise ValueError("replay priority indices are invalid")
        if not math.isfinite(float(priority)) or float(priority) <= 0.0:
            raise ValueError("replay priority must be finite and positive")
        self.priorities[rows] = np.float32(priority)

    def set_priorities_if_current(
        self,
        handles: Sequence[tuple[int, int]],
        priority: float,
    ) -> tuple[int, int]:
        """Update only circular slots that still contain the named insertion.

        Returns ``(updated, stale)``.  Stale handles are expected when a long
        vectorized episode outlives replay capacity; silently applying its
        priority to the replacement row would corrupt HER-force semantics.
        """

        if not math.isfinite(float(priority)) or float(priority) <= 0.0:
            raise ValueError("replay priority must be finite and positive")
        if not handles:
            return 0, 0
        rows = np.asarray([value[0] for value in handles], dtype=np.int64)
        identities = np.asarray([value[1] for value in handles], dtype=np.int64)
        if (
            rows.ndim != 1
            or identities.shape != rows.shape
            or np.any(rows < 0)
            or np.any(rows >= self.capacity)
            or np.any(identities < 0)
            or len(np.unique(identities)) != identities.size
        ):
            raise ValueError("replay insertion handles are invalid")
        current = self.insertion_ids[rows] == identities
        self.priorities[rows[current]] = np.float32(priority)
        return int(np.count_nonzero(current)), int(np.count_nonzero(~current))

    def add_many_atomic(
        self,
        transitions: Sequence[
            tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, bool]
        ],
    ) -> None:
        """Validate a transition batch fully, then commit it transactionally.

        Validation never mutates replay.  The write phase backs up only the
        circular rows touched by the batch, so even an unexpected NumPy write
        error restores data and cursor counters without copying a large replay
        buffer in full.
        """

        if not isinstance(transitions, Sequence):
            raise ValueError("atomic replay batch must be a sequence")
        validated: list[
            tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, bool]
        ] = []
        for transition in transitions:
            if not isinstance(transition, Sequence) or len(transition) not in (6, 7, 8, 9):
                raise ValueError("atomic replay transition must contain 6/7/8/9 fields")
            validated.append(self._validate_transition(*transition[:6]))
        if not validated:
            return

        affected_indices = np.unique(
            (self.position + np.arange(len(validated), dtype=np.int64))
            % self.capacity
        )
        backups = {
            "observations": self.observations[affected_indices].copy(),
            "actions": self.actions[affected_indices].copy(),
            "rewards": self.rewards[affected_indices].copy(),
            "next_observations": self.next_observations[affected_indices].copy(),
            "terminated": self.terminated[affected_indices].copy(),
            "truncated": self.truncated[affected_indices].copy(),
            "priorities": self.priorities[affected_indices].copy(),
            "phase_labels": self.phase_labels[affected_indices].copy(),
            "insertion_ids": self.insertion_ids[affected_indices].copy(),
        }
        if self.critic_observation_dim is not None:
            assert self.critic_observations is not None
            assert self.next_critic_observations is not None
            backups["critic_observations"] = self.critic_observations[
                affected_indices
            ].copy()
            backups["next_critic_observations"] = self.next_critic_observations[
                affected_indices
            ].copy()
        cursor_before = (self.position, self.size, self.total_inserted)
        try:
            for original, transition in zip(transitions, validated, strict=True):
                has_phase = len(original) in (7, 9)
                phase_label = str(original[-1]) if has_phase else "REACH"
                critic_fields = (
                    original[6:8] if len(original) in (8, 9) else (None, None)
                )
                critic_transition = self._validate_critic_transition(
                    *critic_fields
                )
                self._commit_validated(
                    transition,
                    critic_transition=critic_transition,
                    phase_label=phase_label,
                )
        except Exception:
            self.observations[affected_indices] = backups["observations"]
            self.actions[affected_indices] = backups["actions"]
            self.rewards[affected_indices] = backups["rewards"]
            self.next_observations[affected_indices] = backups[
                "next_observations"
            ]
            self.terminated[affected_indices] = backups["terminated"]
            self.truncated[affected_indices] = backups["truncated"]
            self.priorities[affected_indices] = backups["priorities"]
            self.phase_labels[affected_indices] = backups["phase_labels"]
            self.insertion_ids[affected_indices] = backups["insertion_ids"]
            if self.critic_observation_dim is not None:
                assert self.critic_observations is not None
                assert self.next_critic_observations is not None
                self.critic_observations[affected_indices] = backups[
                    "critic_observations"
                ]
                self.next_critic_observations[affected_indices] = backups[
                    "next_critic_observations"
                ]
            self.position, self.size, self.total_inserted = cursor_before
            raise

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.size < batch_size:
            raise ValueError(
                f"replay contains {self.size} transitions, need {batch_size}"
            )
        phase_ids_present = np.unique(self.phase_labels[: self.size])
        stratified = bool(
            self.phase_stratification_enabled and len(phase_ids_present) > 1
        )
        if stratified:
            # Equal phase quotas prevent rare physical contact/stable/lift
            # rows from being submerged by long REACH rollouts.  Replacement
            # is intentional for a newly populated rare stratum.
            counts = np.full(len(phase_ids_present), batch_size // len(phase_ids_present))
            counts[: batch_size % len(phase_ids_present)] += 1
            chosen: list[np.ndarray] = []
            selected_probabilities: list[np.ndarray] = []
            for phase_id, count in zip(phase_ids_present, counts, strict=True):
                rows = np.flatnonzero(self.phase_labels[: self.size] == phase_id)
                if self.prioritized:
                    scaled = np.power(
                        np.maximum(self.priorities[rows].astype(np.float64), 1.0e-12),
                        self.priority_alpha,
                    )
                    within = scaled / float(np.sum(scaled))
                    within = self.uniform_mix / len(rows) + (1.0 - self.uniform_mix) * within
                else:
                    within = np.full(len(rows), 1.0 / len(rows), dtype=np.float64)
                local = self.rng.choice(len(rows), size=int(count), replace=True, p=within)
                chosen.append(rows[local])
                selected_probabilities.append(
                    within[local] / float(len(phase_ids_present))
                )
            indices = np.concatenate(chosen)
            selected_probability = np.concatenate(selected_probabilities)
            permutation = self.rng.permutation(batch_size)
            indices = indices[permutation]
            selected_probability = selected_probability[permutation]
            probabilities = selected_probability
            weights = np.power(
                self.size * np.maximum(selected_probability, 1.0e-12),
                -self.importance_beta,
            )
            weights /= float(np.max(weights))
        elif self.prioritized:
            scaled = np.power(
                np.maximum(self.priorities[: self.size].astype(np.float64), 1.0e-12),
                self.priority_alpha,
            )
            probabilities = scaled / float(np.sum(scaled))
            probabilities = (
                self.uniform_mix / self.size
                + (1.0 - self.uniform_mix) * probabilities
            )
            indices = self.rng.choice(
                self.size, size=batch_size, replace=True, p=probabilities
            )
            weights = np.power(
                self.size * probabilities[indices], -self.importance_beta
            )
            weights /= float(np.max(weights))
        else:
            indices = self.rng.integers(0, self.size, size=batch_size)
            probabilities = None
            weights = None
        result = {
            "observations": self.observations[indices].copy(),
            "actions": self.actions[indices].copy(),
            "rewards": self.rewards[indices].copy(),
            "next_observations": self.next_observations[indices].copy(),
            "terminated": self.terminated[indices].copy(),
            "truncated": self.truncated[indices].copy(),
        }
        if self.critic_observation_dim is not None:
            assert self.critic_observations is not None
            assert self.next_critic_observations is not None
            result["critic_observations"] = self.critic_observations[
                indices
            ].copy()
            result["next_critic_observations"] = self.next_critic_observations[
                indices
            ].copy()
        if weights is not None and probabilities is not None:
            result.update(
                {
                    "importance_weights": weights.astype(np.float32).reshape(-1, 1),
                    "sampling_probabilities": (
                        probabilities.astype(np.float32).reshape(-1, 1)
                        if stratified
                        else probabilities[indices].astype(np.float32).reshape(-1, 1)
                    ),
                    "sampled_priorities": self.priorities[indices].copy().reshape(
                        -1, 1
                    ),
                }
            )
        if self.phase_stratification_enabled:
            result["phase_labels"] = self.phase_labels[indices].copy().reshape(-1, 1)
        return result

    def state_dict(self) -> dict[str, Any]:
        stored = self.size
        result = {
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "critic_observation_dim": self.critic_observation_dim,
            "position": self.position,
            "size": self.size,
            "total_inserted": self.total_inserted,
            "observations": self.observations[:stored].copy(),
            "actions": self.actions[:stored].copy(),
            "rewards": self.rewards[:stored].copy(),
            "next_observations": self.next_observations[:stored].copy(),
            "terminated": self.terminated[:stored].copy(),
            "truncated": self.truncated[:stored].copy(),
            "priorities": self.priorities[:stored].copy(),
            "phase_labels": self.phase_labels[:stored].copy(),
            "insertion_ids": self.insertion_ids[:stored].copy(),
            "phase_stratification": {
                "schema": "geniesim_phase_stratified_replay_v1",
                "enabled": self.phase_stratification_enabled,
                "phase_names": self.phase_names,
            },
            "prioritization": {
                "enabled": self.prioritized,
                "alpha": self.priority_alpha,
                "beta": self.importance_beta,
                "uniform_mix": self.uniform_mix,
            },
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }
        if self.critic_observation_dim is not None:
            assert self.critic_observations is not None
            assert self.next_critic_observations is not None
            result["critic_observations"] = self.critic_observations[:stored].copy()
            result["next_critic_observations"] = self.next_critic_observations[
                :stored
            ].copy()
        return result

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for name, expected in (
            ("capacity", self.capacity),
            ("observation_dim", self.observation_dim),
            ("action_dim", self.action_dim),
        ):
            if int(state[name]) != expected:
                raise ValueError(f"replay checkpoint mismatch for {name}")
        saved_critic_dim = state.get("critic_observation_dim")
        if saved_critic_dim != self.critic_observation_dim:
            raise ValueError("replay checkpoint mismatch for critic_observation_dim")
        size = int(state["size"])
        position = int(state["position"])
        if not 0 <= size <= self.capacity or not 0 <= position < self.capacity:
            raise ValueError("invalid replay checkpoint cursor")
        fields = (
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "terminated",
            "truncated",
        )
        for field in fields:
            saved = np.asarray(state[field], dtype=np.float32)
            target = getattr(self, field)
            if saved.shape != target[:size].shape or not np.all(np.isfinite(saved)):
                raise ValueError(f"invalid replay checkpoint field: {field}")
            target[:size] = saved
        if self.critic_observation_dim is not None:
            assert self.critic_observations is not None
            assert self.next_critic_observations is not None
            for field, target in (
                ("critic_observations", self.critic_observations),
                ("next_critic_observations", self.next_critic_observations),
            ):
                saved = np.asarray(state[field], dtype=np.float32)
                if (
                    saved.shape != target[:size].shape
                    or not np.all(np.isfinite(saved))
                ):
                    raise ValueError(f"invalid replay checkpoint field: {field}")
                target[:size] = saved
        self.size = size
        self.position = position
        self.total_inserted = int(state["total_inserted"])
        saved_insertion_ids = state.get("insertion_ids")
        if saved_insertion_ids is None:
            # Backward-compatible reconstruction for older replay checkpoints.
            # Circular slot ownership is fully determined by total inserts,
            # capacity and the next-write cursor.
            if size < self.capacity:
                reconstructed = np.arange(size, dtype=np.int64)
            else:
                base = self.total_inserted - self.capacity
                reconstructed = base + (
                    (np.arange(size, dtype=np.int64) - position) % self.capacity
                )
            self.insertion_ids[:size] = reconstructed
        else:
            insertion_ids = np.asarray(saved_insertion_ids, dtype=np.int64)
            if (
                insertion_ids.shape != (size,)
                or np.any(insertion_ids < 0)
                or len(np.unique(insertion_ids)) != size
                or (size and int(np.max(insertion_ids)) >= self.total_inserted)
            ):
                raise ValueError("invalid replay checkpoint insertion IDs")
            self.insertion_ids[:size] = insertion_ids
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        saved_priorities = state.get("priorities")
        if saved_priorities is None:
            self.priorities[:size] = 1.0
        else:
            priorities = np.asarray(saved_priorities, dtype=np.float32)
            if (
                priorities.shape != (size,)
                or not np.all(np.isfinite(priorities))
                or np.any(priorities <= 0.0)
            ):
                raise ValueError("invalid replay checkpoint priorities")
            self.priorities[:size] = priorities
        phase_state = state.get("phase_stratification")
        saved_phase_labels = state.get("phase_labels")
        if phase_state is None or saved_phase_labels is None:
            self.phase_labels[:size] = 0
            self.phase_stratification_enabled = False
        else:
            if (
                not isinstance(phase_state, Mapping)
                or phase_state.get("schema") != "geniesim_phase_stratified_replay_v1"
                or tuple(phase_state.get("phase_names", ())) != self.phase_names
            ):
                raise ValueError("replay phase stratification schema differs")
            labels = np.asarray(saved_phase_labels, dtype=np.int8)
            if labels.shape != (size,) or np.any(labels < 0) or np.any(labels >= len(self.phase_names)):
                raise ValueError("invalid replay phase labels")
            self.phase_labels[:size] = labels
            self.phase_stratification_enabled = bool(phase_state.get("enabled"))
        prioritization = state.get("prioritization")
        if prioritization is not None:
            if not isinstance(prioritization, Mapping):
                raise ValueError("invalid replay prioritization checkpoint")
            self.configure_prioritization(
                enabled=bool(prioritization.get("enabled")),
                alpha=float(prioritization.get("alpha")),
                beta=float(prioritization.get("beta")),
                uniform_mix=float(prioritization.get("uniform_mix")),
            )


class RunningMeanStd:
    """Numerically stable observation normalizer with checkpoint state."""

    def __init__(self, shape: int | Sequence[int], *, clip: float = 10.0) -> None:
        normalized_shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if not normalized_shape or any(dimension <= 0 for dimension in normalized_shape):
            raise ValueError("normalizer shape must be positive")
        _positive_finite("clip", clip)
        self.shape = normalized_shape
        self.clip = float(clip)
        self.count = 0
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.m2 = np.zeros(self.shape, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.shape == self.shape:
            array = array.reshape((1,) + self.shape)
        if array.ndim != len(self.shape) + 1 or array.shape[1:] != self.shape:
            raise ValueError(f"normalizer expected (*,{self.shape}), got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("normalizer input contains non-finite values")
        batch_count = array.shape[0]
        if batch_count == 0:
            return
        batch_mean = array.mean(axis=0)
        batch_m2 = ((array - batch_mean) ** 2).sum(axis=0)
        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
            self.count = batch_count
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * (batch_count / total)
        self.m2 += batch_m2 + delta**2 * self.count * batch_count / total
        self.count = total

    @property
    def variance(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.shape, dtype=np.float64)
        return np.maximum(self.m2 / (self.count - 1), 1.0e-6)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-len(self.shape) :] != self.shape:
            raise ValueError("normalizer input has incompatible trailing shape")
        normalized = (array.astype(np.float64) - self.mean) / np.sqrt(self.variance)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "clip": self.clip,
            "count": self.count,
            "mean": self.mean.copy(),
            "m2": self.m2.copy(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if tuple(state["shape"]) != self.shape:
            raise ValueError("normalizer checkpoint shape mismatch")
        if float(state["clip"]) != self.clip:
            raise ValueError("normalizer checkpoint clip mismatch")
        count = int(state["count"])
        mean = np.asarray(state["mean"], dtype=np.float64)
        m2 = np.asarray(state["m2"], dtype=np.float64)
        if count < 0 or mean.shape != self.shape or m2.shape != self.shape:
            raise ValueError("invalid normalizer checkpoint")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(m2)):
            raise ValueError("non-finite normalizer checkpoint")
        self.count = count
        self.mean = mean.copy()
        self.m2 = m2.copy()


def capture_rng_state() -> dict[str, Any]:
    """Capture process RNG state used by SAC/checkpoint tests."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_torch_checkpoint(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Atomically write a local Stage 2 checkpoint."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, destination)


def load_torch_checkpoint(
    path: Path | str, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load a Stage 2 checkpoint produced by :func:`save_torch_checkpoint`."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Stage 2 checkpoint root must be a mapping")
    if int(payload.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError("unsupported Stage 2 checkpoint version")
    return payload


__all__ = [
    "CHECKPOINT_VERSION",
    "QNetwork",
    "ReplayBuffer",
    "RunningMeanStd",
    "SACAgent",
    "SACConfig",
    "STAGE2_ACTION_DIM",
    "STAGE2_OBSERVATION_DIM",
    "SquashedGaussianActor",
    "capture_rng_state",
    "load_torch_checkpoint",
    "polyak_update",
    "restore_rng_state",
    "save_torch_checkpoint",
    "soft_bellman_target",
    "squashed_gaussian_log_prob",
]
