"""Task-relevant RGB-D models for the G2 teacher/student pipeline.

RGB and depth are encoded separately, and head/wrist weight sharing is an
explicit experiment option.  The deployment student consumes only cameras,
robot state, previous action and camera age.  Simulator state is restricted
to critic and auxiliary *targets* during training.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import copy
import math
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from geniesim.rl.sac.stage2_sac import polyak_update, squashed_gaussian_log_prob
from .g2_quaternion import canonicalize_quaternion_xyzw
from .g2_teacher_sac import G2TeacherObservationContract


G2_VISUAL_SAC_SCHEMA = "g2_rgbd_task_aux_asymmetric_hybrid_sac_v9"
G2_VISUAL_REPLAY_SCHEMA = "g2_rgbd_masked_episode_sequence_replay_v6"
G2_VISUAL_CAMERA_SHAPE = (2, 6, 48, 64)
G2_VISUAL_PROPRIO_DIM = 45
G2_VISUAL_PRIVILEGED_DIM = 59
G2_VISUAL_ACTION_DIM = 7
G2_VISUAL_ARM_ACTION_DIM = 6
G2_VISUAL_GRIPPER_ACTION_INDEX = 6
G2_RECURRENT_STUDENT_SCHEMA = "g2_rgbd_gated_cross_camera_gru_student_v9"
G2_RECURRENT_VISUAL_POLICY_SCHEMA = "g2_rgbd_gated_cross_camera_fusion_gru_policy_v5"
G2_CAMERA_ENCODER_PROFILES = {
    "baseline_4layer": (32, 64, 96, 128),
    "cnn5_160": (32, 64, 96, 128, 160),
    "cnn7_160": (32, 64, 96, 128, 160, 160, 160),
    "cnn9_160": (32, 64, 96, 128, 160, 160, 160, 160, 160),
    "cnn11_160": (32, 64, 96, 128, 160, 160, 160, 160, 160, 160, 160),
    "cnn13_160": (
        32, 64, 96, 128, 160, 160, 160, 160, 160, 160, 160, 160, 160
    ),
}
G2_STUDENT_FAILURE_CLASSES = (
    "miss", "contact", "stable_grasp", "slip", "collision", "lift", "place"
)
G2_DEMONSTRATION_PHASES = (
    "REACH", "PRE_GRASP", "CONTACT", "STABLE_GRASP", "LIFT"
)
G2_ONLINE_REPLAY_PHASES = ("REACH", "CONTACT", "STABLE_GRASP", "LIFT")


def preprocess_demonstration_expert_actions(
    expert_action: torch.Tensor, *, arm_action_scale: float
) -> tuple[torch.Tensor, dict[str, float | str]]:
    """Create a BC-only speed-retimed view of canonical expert actions.

    Only the six continuous arm labels are scaled.  The recorded gripper
    command (binary in current live collections, potentially continuous in
    canonical legacy/test fixtures) and every state/reward/next-state remain
    bit-identical, so this function must not be used to rewrite an RL
    transition action.
    """

    if expert_action.ndim < 2 or expert_action.shape[-1] != G2_VISUAL_ACTION_DIM:
        raise ValueError("expert action must end in the canonical 7-D action")
    if not 0.0 < arm_action_scale <= 1.0:
        raise ValueError("expert arm action scale must be in (0,1]")
    if not bool(torch.isfinite(expert_action).all()):
        raise ValueError("expert action contains non-finite values")
    gripper = expert_action[..., G2_VISUAL_GRIPPER_ACTION_INDEX]
    processed = expert_action.clone()
    before = processed[..., :G2_VISUAL_ARM_ACTION_DIM].abs()
    processed[..., :G2_VISUAL_ARM_ACTION_DIM] *= float(arm_action_scale)
    after = processed[..., :G2_VISUAL_ARM_ACTION_DIM].abs()
    if not torch.equal(processed[..., G2_VISUAL_GRIPPER_ACTION_INDEX], gripper):
        raise RuntimeError("expert gripper changed during arm speed preprocessing")
    return processed, {
        "schema": "g2_demonstration_bc_arm_speed_retime_v1",
        "arm_action_scale": float(arm_action_scale),
        "arm_absolute_max_before": float(before.max()) if before.numel() else 0.0,
        "arm_absolute_max_after": float(after.max()) if after.numel() else 0.0,
        "gripper_contract": "RECORDED_VALUE_EXACTLY_UNCHANGED",
        "usage": "BEHAVIOR_CLONING_LABEL_ONLY_NOT_RL_TRANSITION_REWRITE",
    }


def classify_demonstration_sequence_phases(
    sequences: Mapping[str, torch.Tensor | int],
) -> torch.Tensor:
    """Return one highest-achieved phase code for every expert window.

    The labels are derived from the canonical Teacher state and physical
    contact/stable targets.  No simulator-only value is added to the actor
    input: this code is used solely by the offline batch sampler.
    """

    required = (
        "teacher_state_target",
        "expert_action_target",
        "bilateral_contact_target",
        "stable_grasp_target",
        "padding_mask",
    )
    if any(not isinstance(sequences.get(name), torch.Tensor) for name in required):
        raise ValueError("demonstration sequences lack canonical phase tensors")
    state = sequences["teacher_state_target"]
    action = sequences["expert_action_target"]
    bilateral = sequences["bilateral_contact_target"]
    stable = sequences["stable_grasp_target"]
    padding = sequences["padding_mask"].to(torch.bool)
    assert isinstance(state, torch.Tensor)
    assert isinstance(action, torch.Tensor)
    assert isinstance(bilateral, torch.Tensor)
    assert isinstance(stable, torch.Tensor)
    assert isinstance(padding, torch.Tensor)
    if state.ndim != 3 or state.shape[-1] != G2_VISUAL_PRIVILEGED_DIM:
        raise ValueError("teacher_state_target must be [B,T,59]")
    if action.shape != (*padding.shape, G2_VISUAL_ACTION_DIM):
        raise ValueError("expert_action_target must be [B,T,7]")
    burn_in = int(sequences.get("burn_in_steps", 0))
    learning = padding.clone()
    learning[:, :burn_in] = False
    phase_slice = G2TeacherObservationContract().slices[
        "curriculum_phase_features"
    ]
    phase = state[..., phase_slice]
    valid = learning.unsqueeze(-1)
    contact = ((phase[..., 1:2] > 0.5) | (bilateral > 0.5)) & valid
    stable_mask = (stable > 0.5) & valid
    lift = ((phase[..., 2:3] > 0.5) | (phase[..., 3:4] > 0.5)) & valid
    pregrasp = (action[..., 6:7] < 0.0) & ~contact & valid
    code = torch.zeros(padding.shape[0], dtype=torch.long, device=state.device)
    code[pregrasp.any(dim=(1, 2))] = 1
    code[contact.any(dim=(1, 2))] = 2
    code[stable_mask.any(dim=(1, 2))] = 3
    code[lift.any(dim=(1, 2))] = 4
    return code


def sample_phase_stratified_demonstration_indices(
    phase_codes: torch.Tensor,
    success: torch.Tensor,
    *,
    batch_size: int,
    phase_probabilities: tuple[float, float, float, float, float],
    success_priority_weight: float,
    rng: np.random.Generator,
    force_priorities: torch.Tensor | None = None,
    force_priority_mixture: float = 0.25,
) -> tuple[np.ndarray, dict[str, int]]:
    """Sample expert windows by phase, success and real-contact energy.

    Force priority is mixed with the phase-local base distribution instead of
    multiplying it without a bound.  The latter made the 12% of windows with
    force evidence occupy roughly half of every expert batch in a live run,
    repeatedly training on a very small set of trajectories and degrading the
    reference/contact curriculum.  A mixture keeps real force evidence active
    while preserving phase-local demonstration diversity.
    """

    if phase_codes.ndim != 1 or success.shape != phase_codes.shape:
        raise ValueError("phase_codes and success must both be [windows]")
    if batch_size <= 0 or phase_codes.numel() == 0:
        raise ValueError("expert sampling requires a positive non-empty batch")
    probability = np.asarray(phase_probabilities, dtype=np.float64)
    if probability.shape != (len(G2_DEMONSTRATION_PHASES),) or np.any(probability < 0):
        raise ValueError("invalid demonstration phase probabilities")
    if not np.isclose(probability.sum(), 1.0):
        raise ValueError("demonstration phase probabilities must sum to one")
    if success_priority_weight < 1.0:
        raise ValueError("success priority weight must be at least one")
    if not 0.0 <= force_priority_mixture <= 1.0:
        raise ValueError("force_priority_mixture must be in [0,1]")
    if force_priorities is not None:
        if force_priorities.shape != phase_codes.shape:
            raise ValueError("force_priorities must be [windows]")
        if not bool(torch.isfinite(force_priorities).all()) or bool(
            (force_priorities < 1.0).any()
        ):
            raise ValueError("force_priorities must be finite and at least one")
    phase_numpy = phase_codes.detach().cpu().numpy()
    success_numpy = success.detach().cpu().numpy().astype(bool)
    force_numpy = (
        np.ones_like(phase_numpy, dtype=np.float64)
        if force_priorities is None
        else force_priorities.detach().cpu().numpy().astype(np.float64)
    )
    all_indices = np.arange(len(phase_numpy), dtype=np.int64)
    available = np.asarray(
        [bool(np.any(phase_numpy == phase)) for phase in range(len(probability))],
        dtype=np.bool_,
    )
    effective_probability = probability * available
    if effective_probability.sum() <= 0.0:
        raise ValueError("demonstration batch has no available phase")
    # Reallocate the probability of an absent physical phase over phases that
    # actually exist.  Falling back to all windows silently destroyed the
    # requested phase balance in datasets without standalone STABLE windows.
    effective_probability /= effective_probability.sum()
    requested = rng.choice(
        len(effective_probability), size=batch_size, p=effective_probability
    )
    selected: list[int] = []
    counts = {name: 0 for name in G2_DEMONSTRATION_PHASES}
    for phase_code in requested:
        candidates = all_indices[phase_numpy == phase_code]
        if candidates.size == 0:
            raise RuntimeError("sampler selected an unavailable demonstration phase")
        base_weights = np.where(
            success_numpy[candidates], success_priority_weight, 1.0
        ).astype(np.float64)
        base_probability = base_weights / base_weights.sum()
        force_weights = base_weights * force_numpy[candidates]
        force_probability = force_weights / force_weights.sum()
        weights = (
            (1.0 - force_priority_mixture) * base_probability
            + force_priority_mixture * force_probability
        )
        chosen = int(rng.choice(candidates, p=weights))
        selected.append(chosen)
        counts[G2_DEMONSTRATION_PHASES[int(phase_numpy[chosen])]] += 1
    return np.asarray(selected, dtype=np.int64), counts


def demonstration_policy_mastery(
    evaluation_rates: tuple[float, float, float],
    *,
    evaluation_episodes: int,
    minimum_evaluation_episodes: int = 10,
    thresholds: tuple[float, float, float] = (0.50, 0.30, 0.20),
) -> float:
    """Measure Q-filter trust earned by held-out policy task outcomes."""

    if evaluation_episodes < minimum_evaluation_episodes:
        return 0.0
    rates = np.asarray(evaluation_rates, dtype=np.float64)
    limits = np.asarray(thresholds, dtype=np.float64)
    if rates.shape != (3,) or limits.shape != (3,) or np.any(limits <= 0.0):
        raise ValueError("demonstration policy mastery expects three positive gates")
    if np.any(~np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("evaluation rates must be finite and non-negative")
    return float(np.clip(np.min(rates / limits), 0.0, 1.0))


def demonstration_bc_schedule(
    *,
    replay_transitions: int,
    learning_starts: int,
    decay_transitions: int,
    initial_coefficient: float,
    minimum_coefficient: float,
    policy_mastery: float,
) -> tuple[float, float]:
    """Return performance-gated BC coefficient and Q-filter reliability."""

    if decay_transitions <= 0:
        raise ValueError("demonstration BC decay transitions must be positive")
    if not 0.0 <= minimum_coefficient <= initial_coefficient:
        raise ValueError("invalid demonstration BC coefficient range")
    if not 0.0 <= policy_mastery <= 1.0:
        raise ValueError("policy mastery must be in [0,1]")
    time_progress = np.clip(
        max(int(replay_transitions) - int(learning_starts), 0)
        / float(decay_transitions),
        0.0,
        1.0,
    )
    trusted_progress = float(time_progress) * float(policy_mastery)
    coefficient = initial_coefficient + (
        minimum_coefficient - initial_coefficient
    ) * trusted_progress
    return float(coefficient), float(policy_mastery)


def demonstration_force_energy_priorities(
    sequences: Mapping[str, torch.Tensor | int],
    *,
    temperature_j: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build bounded window priorities from recorded physical contact work.

    The authority is the original two-finger force and consecutive cube pose
    displacement.  Contact labels, safe-for-force attestation and recurrent
    padding all have to agree.  This does not synthesize contact and does not
    turn demonstrations into HER transitions; it only prioritizes real force
    evidence in the actor's expert batch.
    """

    if temperature_j <= 0.0:
        raise ValueError("demonstration force temperature must be positive")
    required = (
        "teacher_state_target",
        "contact_force_target_n",
        "bilateral_contact_target",
        "safe_for_her_force",
        "stable_grasp_target",
        "padding_mask",
    )
    if any(not isinstance(sequences.get(name), torch.Tensor) for name in required):
        raise ValueError("demonstration sequences lack force-priority tensors")
    state = sequences["teacher_state_target"]
    force = sequences["contact_force_target_n"]
    bilateral = sequences["bilateral_contact_target"]
    safe = sequences["safe_for_her_force"]
    stable = sequences["stable_grasp_target"]
    padding = sequences["padding_mask"]
    assert isinstance(state, torch.Tensor)
    assert isinstance(force, torch.Tensor)
    assert isinstance(bilateral, torch.Tensor)
    assert isinstance(safe, torch.Tensor)
    assert isinstance(stable, torch.Tensor)
    assert isinstance(padding, torch.Tensor)
    if force.shape != (*padding.shape, 2):
        raise ValueError("contact_force_target_n must be [B,T,2]")
    cube_slice = G2TeacherObservationContract().slices["cube_pose_root_xyzw"]
    cube_position = state[..., cube_slice][..., :3]
    displacement = torch.zeros_like(padding, dtype=torch.float32)
    displacement[:, 1:] = torch.linalg.vector_norm(
        cube_position[:, 1:] - cube_position[:, :-1], dim=-1
    )
    burn_in = int(sequences.get("burn_in_steps", 0))
    # ``Tensor.to`` may return the original object when it is already bool.
    # Never mutate the canonical recurrent padding mask while excluding the
    # burn-in prefix for force-priority estimation.
    valid = padding.to(torch.bool).clone()
    valid[:, :burn_in] = False
    valid &= safe.to(torch.bool).squeeze(-1)
    valid &= bilateral.to(torch.bool).squeeze(-1)
    total_force_n = force.to(torch.float32).clamp_min(0.0).sum(dim=-1)
    energy_j = torch.where(valid, total_force_n * displacement, torch.zeros_like(displacement))
    row_priority = 1.0 + torch.clamp(energy_j / temperature_j, 0.0, 4.0)
    stable_valid = stable.to(torch.bool).squeeze(-1) & valid
    row_priority = torch.where(
        stable_valid,
        torch.maximum(row_priority, torch.full_like(row_priority, 3.0)),
        row_priority,
    )
    window_priority = row_priority.max(dim=1).values
    return window_priority, {
        "demonstration_force/window_priority_mean": float(window_priority.mean()),
        "demonstration_force/window_priority_max": float(window_priority.max()),
        "demonstration_force/prioritized_window_fraction": float(
            (window_priority > 1.0).to(torch.float32).mean()
        ),
        "demonstration_force/contact_energy_mean_j": float(
            energy_j[valid].mean() if bool(valid.any()) else 0.0
        ),
        "demonstration_force/contact_energy_max_j": float(energy_j.max()),
        "demonstration_force/temperature_j": float(temperature_j),
    }


def g2_reference_controller_action(
    teacher_state: torch.Tensor,
    *,
    end_effector_to_cube_slice: slice,
    cube_to_goal_slice: slice,
    bilateral_contact: torch.Tensor,
    stable_grasp: torch.Tensor,
    maximum_arm_action_magnitude: float,
    translation_scale_m: float = 0.0225,
    target_cube_minus_ee_m: tuple[float, float, float] = (0.0, 0.0, -0.020),
    gripper_close_position_tolerance_m: float = 0.010,
) -> torch.Tensor:
    """Build the bounded privileged reference used only for data collection.

    Before contact the hand approaches a calibrated side-pinch relative pose.
    ``target_cube_minus_ee_m`` is expressed in the robot-root frame and may be
    estimated only from physically successful, collision-free demonstrations.
    A stable grasp changes authority to the episode's object goal in all three
    translation axes, so independently randomized cube/goal XY positions do
    not make the canonical 5 mm success condition unreachable.
    """

    if teacher_state.ndim != 2 or teacher_state.shape[-1] != G2_VISUAL_PRIVILEGED_DIM:
        raise ValueError("reference controller requires [N,59] teacher state")
    batch = teacher_state.shape[0]
    for name, value in (
        ("bilateral_contact", bilateral_contact),
        ("stable_grasp", stable_grasp),
    ):
        if value.shape != (batch,) or value.dtype != torch.bool:
            raise ValueError(f"{name} must be a boolean [N] tensor")
    if not 0.0 < maximum_arm_action_magnitude <= 1.0:
        raise ValueError("maximum arm action magnitude must be in (0,1]")
    if translation_scale_m <= 0.0:
        raise ValueError("translation scale must be positive")
    if len(target_cube_minus_ee_m) != 3 or not all(
        math.isfinite(float(value)) for value in target_cube_minus_ee_m
    ):
        raise ValueError("reference grasp offset must contain three finite values")
    if gripper_close_position_tolerance_m <= 0.0:
        raise ValueError("reference gripper close tolerance must be positive")

    result = torch.zeros(
        (batch, G2_VISUAL_ACTION_DIM),
        dtype=teacher_state.dtype,
        device=teacher_state.device,
    )
    target_relative = torch.as_tensor(
        target_cube_minus_ee_m,
        dtype=teacher_state.dtype,
        device=teacher_state.device,
    )
    # Moving the EE by +delta changes (cube - EE) by -delta.  Therefore the
    # Cartesian command that drives the measured relative pose to the
    # calibrated target is current_relative - target_relative.
    approach = (
        teacher_state[:, end_effector_to_cube_slice] - target_relative
    )
    result[:, :3] = torch.clamp(
        approach / translation_scale_m,
        -maximum_arm_action_magnitude,
        maximum_arm_action_magnitude,
    )
    result[bilateral_contact, :G2_VISUAL_ARM_ACTION_DIM] = 0.0
    goal_delta = teacher_state[:, cube_to_goal_slice]
    result[stable_grasp, :3] = torch.clamp(
        goal_delta[stable_grasp] / translation_scale_m,
        -maximum_arm_action_magnitude,
        maximum_arm_action_magnitude,
    )
    # Approaching with a closed parallel gripper made the outer pad strike and
    # push the cube while the inner pad never contacted it.  Preserve the
    # audited open pose until the calibrated side-pinch center is reached;
    # only then close.  This is reference collection logic, not actor input.
    position_error = torch.linalg.vector_norm(approach, dim=-1)
    close_gripper = (
        position_error <= float(gripper_close_position_tolerance_m)
    ) | bilateral_contact
    result[:, G2_VISUAL_GRIPPER_ACTION_INDEX] = torch.where(
        close_gripper,
        -torch.ones_like(position_error),
        torch.ones_like(position_error),
    )
    return result


@dataclass(frozen=True)
class G2RecurrentVisualPolicyContract:
    """One sealed input/backbone contract for keyboard, teacher and student."""

    schema: str = G2_RECURRENT_VISUAL_POLICY_SCHEMA
    camera_shape: tuple[int, int, int, int] = G2_VISUAL_CAMERA_SHAPE
    proprioception_dim: int = G2_VISUAL_PROPRIO_DIM
    action_dim: int = G2_VISUAL_ACTION_DIM
    hidden_dim: int = 256
    gru_num_layers: int = 1
    sequence_length: int = 16
    burn_in_steps: int = 4
    sequence_stride: int = 12
    camera_encoder_channels: tuple[int, ...] = G2_CAMERA_ENCODER_PROFILES[
        "baseline_4layer"
    ]

    def validated(self) -> "G2RecurrentVisualPolicyContract":
        if self.camera_shape != (2, 6, 48, 64):
            raise ValueError("visual policy requires head/wrist packed RGB-D")
        if self.proprioception_dim != 45 or self.action_dim != 7:
            raise ValueError("visual policy input/output dimension drift")
        if self.hidden_dim <= 0:
            raise ValueError("visual policy hidden dimension must be positive")
        if self.gru_num_layers != 1:
            raise ValueError("visual policy contract requires exactly one GRU layer")
        if not 0 <= self.burn_in_steps < self.sequence_length:
            raise ValueError("visual policy burn-in must be inside the sequence")
        if not 0 < self.sequence_stride <= self.sequence_length:
            raise ValueError("visual policy sequence stride must be in [1, sequence_length]")
        if tuple(self.camera_encoder_channels) not in G2_CAMERA_ENCODER_PROFILES.values():
            raise ValueError("unknown camera encoder channel contract")
        return self

    def serializable(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "camera_order": ["head", "right_wrist"],
            "camera_channels": ["rgb_r", "rgb_g", "rgb_b", "depth_hi", "depth_lo", "depth_valid"],
            "camera_shape": list(self.camera_shape),
            "proprioception_dim": self.proprioception_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "gru_num_layers": self.gru_num_layers,
            "sequence_length": self.sequence_length,
            "burn_in_steps": self.burn_in_steps,
            "sequence_stride": self.sequence_stride,
            "backbone": "SPLIT_RGB_DEPTH_GATED_HEAD_WRIST_CROSS_ATTENTION_GRU",
            "camera_encoder_channels": list(self.camera_encoder_channels),
            "camera_fusion": "PER_CAMERA_RGB_DEPTH_GATE_THEN_2_TOKEN_ATTENTION",
            "training_auxiliary_objectives": [
                "CROSS_CAMERA_SYMMETRIC_INFONCE",
                "TEMPORAL_OBJECT_IN_EE_POSE_RESIDUAL",
            ],
            "torso_policy_output_enabled": False,
        }


def _quat_multiply_xyzw(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lx, ly, lz, lw = left.unbind(-1)
    rx, ry, rz, rw = right.unbind(-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


def _quat_apply_inverse_xyzw(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    conjugate = torch.cat((-quaternion[..., :3], quaternion[..., 3:4]), dim=-1)
    pure = torch.cat((vector, torch.zeros_like(vector[..., :1])), dim=-1)
    return _quat_multiply_xyzw(_quat_multiply_xyzw(conjugate, pure), quaternion)[..., :3]


def _axis_angle_quaternion_xyzw(
    axis: tuple[float, float, float], angle_rad: float
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(component * component for component in axis))
    scale = math.sin(angle_rad / 2.0) / norm
    return (
        axis[0] * scale,
        axis[1] * scale,
        axis[2] * scale,
        math.cos(angle_rad / 2.0),
    )


# The 24 proper rotations of a cube, represented in the cube-local frame.
# A visually and physically symmetric cube orientation is therefore q*S.
_CUBE_ROTATIONAL_SYMMETRIES_XYZW = (
    ((0.0, 0.0, 0.0, 1.0),)
    + tuple(
        _axis_angle_quaternion_xyzw(axis, angle)
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        for angle in (math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
    )
    + tuple(
        _axis_angle_quaternion_xyzw(axis, angle)
        for axis in (
            (1.0, 1.0, 1.0),
            (1.0, 1.0, -1.0),
            (1.0, -1.0, 1.0),
            (-1.0, 1.0, 1.0),
        )
        for angle in (2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
    )
    + tuple(
        _axis_angle_quaternion_xyzw(axis, math.pi)
        for axis in (
            (1.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 0.0, 1.0),
            (1.0, 0.0, -1.0),
            (0.0, 1.0, 1.0),
            (0.0, 1.0, -1.0),
        )
    )
)


def relative_pose_target_from_teacher_state(state: torch.Tensor) -> torch.Tensor:
    """Return cube pose in the EE frame from the privileged XYZW contract."""

    if state.shape[-1] != G2_VISUAL_PRIVILEGED_DIM:
        raise ValueError("teacher state must have 59 fields")
    slices = G2TeacherObservationContract().slices
    ee = state[..., slices["end_effector_pose_root_xyzw"]]
    cube = state[..., slices["cube_pose_root_xyzw"]]
    ee_quaternion = canonicalize_quaternion_xyzw(ee[..., 3:7])
    cube_quaternion = canonicalize_quaternion_xyzw(cube[..., 3:7])
    relative_position = _quat_apply_inverse_xyzw(
        ee_quaternion, cube[..., :3] - ee[..., :3]
    )
    ee_conjugate = torch.cat((-ee_quaternion[..., :3], ee_quaternion[..., 3:4]), dim=-1)
    relative_quaternion = canonicalize_quaternion_xyzw(
        _quat_multiply_xyzw(ee_conjugate, cube_quaternion)
    )
    return torch.cat((relative_position, relative_quaternion), dim=-1)


def relative_pose_auxiliary_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
    cube_symmetry_invariant: bool = True,
) -> torch.Tensor:
    """Position plus sign- and cube-symmetry-invariant orientation loss."""

    if prediction.shape != target.shape or prediction.shape[-1] != 7:
        raise ValueError("relative pose prediction/target must have matching [...,7] shape")
    position = F.smooth_l1_loss(prediction[..., :3], target[..., :3], reduction="none").mean(-1)
    predicted_quaternion = canonicalize_quaternion_xyzw(prediction[..., 3:7])
    target_quaternion = canonicalize_quaternion_xyzw(target[..., 3:7])
    if cube_symmetry_invariant:
        symmetries = target_quaternion.new_tensor(_CUBE_ROTATIONAL_SYMMETRIES_XYZW)
        equivalent_targets = _quat_multiply_xyzw(
            target_quaternion.unsqueeze(-2), symmetries
        )
        difference = (
            predicted_quaternion.unsqueeze(-2) - equivalent_targets
        ).square().mean(-1)
        opposite_difference = (
            predicted_quaternion.unsqueeze(-2) + equivalent_targets
        ).square().mean(-1)
        orientation = torch.minimum(difference, opposite_difference).amin(-1)
    else:
        orientation = torch.minimum(
            (predicted_quaternion - target_quaternion).square().mean(-1),
            (predicted_quaternion + target_quaternion).square().mean(-1),
        )
    value = position + orientation
    if reduction == "none":
        return value
    if reduction == "mean":
        return value.mean()
    raise ValueError("relative pose loss reduction must be 'none' or 'mean'")


def relative_pose_consistency_loss(
    original_prediction: torch.Tensor,
    shifted_prediction: torch.Tensor,
    *,
    rotation_weight: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """Pose invariance loss for original and aligned RGB-D shift views.

    Position uses the requested L1 metric.  Orientation uses the sign-invariant
    quaternion geodesic distance in radians.  No privileged target is consumed:
    both arguments are predictions from the deployable visual/GRU path.
    """

    if (
        original_prediction.shape != shifted_prediction.shape
        or original_prediction.shape[-1] != 7
    ):
        raise ValueError("pose consistency predictions must have matching [...,7] shape")
    if rotation_weight < 0.0:
        raise ValueError("pose consistency rotation weight cannot be negative")
    position = (original_prediction[..., :3] - shifted_prediction[..., :3]).abs().sum(-1)
    original_quaternion = canonicalize_quaternion_xyzw(
        original_prediction[..., 3:7]
    )
    shifted_quaternion = canonicalize_quaternion_xyzw(
        shifted_prediction[..., 3:7]
    )
    cosine_half_angle = (original_quaternion * shifted_quaternion).sum(-1).abs()
    # Avoid the undefined sqrt/atan2 gradient at exactly identical unit
    # quaternions; this introduces only a sub-milliradian numerical floor.
    cosine_half_angle = cosine_half_angle.clamp(0.0, 1.0 - 1.0e-7)
    sine_half_angle = torch.sqrt((1.0 - cosine_half_angle.square()).clamp_min(0.0))
    rotation = 2.0 * torch.atan2(sine_half_angle, cosine_half_angle.clamp_min(1.0e-8))
    value = position + float(rotation_weight) * rotation
    if reduction == "none":
        return value
    if reduction == "mean":
        return value.mean()
    raise ValueError("pose consistency reduction must be 'none' or 'mean'")


def cross_camera_contrastive_loss(
    projected_camera_tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    relative_pose_target: torch.Tensor | None = None,
    temperature: float = 0.1,
    false_negative_position_threshold_m: float = 0.01,
) -> torch.Tensor:
    """Symmetric head/wrist InfoNCE over camera-specific projection tokens.

    The positive pair is the head and right-wrist observation from the same
    environment and timestep.  Other samples whose privileged relative
    positions are nearly identical are removed from the denominator rather
    than treated as false negatives.  Privileged pose is used only to build
    this learner-side mask and never enters the deployable actor input.
    """

    if projected_camera_tokens.ndim != 4 or projected_camera_tokens.shape[2] != 2:
        raise ValueError("projected camera tokens must be [B,T,2,D]")
    if tuple(valid_mask.shape) != tuple(projected_camera_tokens.shape[:2]):
        raise ValueError("contrastive valid mask must be [B,T]")
    if temperature <= 0.0:
        raise ValueError("contrastive temperature must be positive")
    if false_negative_position_threshold_m < 0.0:
        raise ValueError("false-negative position threshold cannot be negative")
    selected = projected_camera_tokens[valid_mask.to(torch.bool)]
    if selected.shape[0] < 2:
        return projected_camera_tokens.sum() * 0.0
    head = F.normalize(selected[:, 0], dim=-1)
    wrist = F.normalize(selected[:, 1], dim=-1)
    logits = head @ wrist.transpose(0, 1) / float(temperature)
    allowed = torch.ones_like(logits, dtype=torch.bool)
    if relative_pose_target is not None:
        if tuple(relative_pose_target.shape[:2]) != tuple(valid_mask.shape) or relative_pose_target.shape[-1] != 7:
            raise ValueError("contrastive relative-pose target must be [B,T,7]")
        positions = relative_pose_target[valid_mask.to(torch.bool), :3]
        separation = torch.cdist(positions, positions)
        allowed &= separation > float(false_negative_position_threshold_m)
    allowed.fill_diagonal_(True)
    floor = torch.finfo(logits.dtype).min
    row_logits = logits.masked_fill(~allowed, floor)
    column_logits = logits.transpose(0, 1).masked_fill(~allowed.transpose(0, 1), floor)
    diagonal = torch.arange(logits.shape[0], device=logits.device)
    row_loss = -row_logits[diagonal, diagonal] + torch.logsumexp(row_logits, dim=1)
    column_loss = -column_logits[diagonal, diagonal] + torch.logsumexp(column_logits, dim=1)
    # An untrained InfoNCE objective is approximately log(N).  Normalize by
    # that value so its configured weight has the same meaning for offline BC
    # and online SAC batches of different sizes.
    normalizer = max(math.log(float(logits.shape[0])), 1.0)
    return 0.5 * (row_loss.mean() + column_loss.mean()) / normalizer


def temporal_pose_residual_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    rotation_weight: float = 0.1,
) -> torch.Tensor:
    """Match consecutive object-in-EE pose changes without crossing resets."""

    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 7:
        raise ValueError("temporal pose prediction/target must be matching [B,T,7]")
    if tuple(valid_mask.shape) != tuple(prediction.shape[:2]):
        raise ValueError("temporal pose valid mask must be [B,T]")
    if rotation_weight < 0.0:
        raise ValueError("temporal pose rotation weight cannot be negative")
    if prediction.shape[1] < 2:
        return prediction.sum() * 0.0
    pair_valid = valid_mask[:, 1:].to(torch.bool) & valid_mask[:, :-1].to(torch.bool)
    if not bool(pair_valid.any()):
        return prediction.sum() * 0.0
    predicted_position_delta = prediction[:, 1:, :3] - prediction[:, :-1, :3]
    target_position_delta = target[:, 1:, :3] - target[:, :-1, :3]
    position = F.smooth_l1_loss(
        predicted_position_delta, target_position_delta, reduction="none"
    ).mean(-1)

    predicted_previous = canonicalize_quaternion_xyzw(prediction[:, :-1, 3:7])
    predicted_next = canonicalize_quaternion_xyzw(prediction[:, 1:, 3:7])
    target_previous = canonicalize_quaternion_xyzw(target[:, :-1, 3:7])
    target_next = canonicalize_quaternion_xyzw(target[:, 1:, 3:7])
    predicted_delta = canonicalize_quaternion_xyzw(
        _quat_multiply_xyzw(
            torch.cat((-predicted_previous[..., :3], predicted_previous[..., 3:4]), dim=-1),
            predicted_next,
        )
    )
    target_delta = canonicalize_quaternion_xyzw(
        _quat_multiply_xyzw(
            torch.cat((-target_previous[..., :3], target_previous[..., 3:4]), dim=-1),
            target_next,
        )
    )
    cosine_half_angle = (predicted_delta * target_delta).sum(-1).abs().clamp(0.0, 1.0 - 1.0e-7)
    sine_half_angle = torch.sqrt((1.0 - cosine_half_angle.square()).clamp_min(0.0))
    rotation = 2.0 * torch.atan2(sine_half_angle, cosine_half_angle.clamp_min(1.0e-8))
    value = position + float(rotation_weight) * rotation
    weight = pair_valid.to(value.dtype)
    return (weight * value).sum() / weight.sum().clamp_min(1.0)


@dataclass(frozen=True)
class G2StudentObservationContract:
    """Ordered deployment input excluding all privileged task oracles."""

    arm_hand_joint_count: int = 8
    torso_joint_count: int = 5
    previous_action_dim: int = G2_VISUAL_ACTION_DIM

    @property
    def fields(self) -> tuple[tuple[str, int], ...]:
        return (
            ("arm_hand_joint_position_relative_rad", self.arm_hand_joint_count),
            ("arm_hand_joint_velocity_rad_s", self.arm_hand_joint_count),
            ("torso_joint_position_relative_rad", self.torso_joint_count),
            ("torso_joint_velocity_rad_s", self.torso_joint_count),
            ("end_effector_pose_root_xyzw", 7),
            ("goal_position_root_m", 3),
            ("previous_action", self.previous_action_dim),
            ("camera_frame_age_s", 2),
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

    def build(self, **values: torch.Tensor) -> torch.Tensor:
        expected_names = tuple(name for name, _ in self.fields)
        if tuple(values) != expected_names:
            raise ValueError(f"student observation fields/order differ: {tuple(values)}")
        batch = next(iter(values.values())).shape[0]
        pieces = []
        for name, width in self.fields:
            value = values[name]
            if tuple(value.shape) != (batch, width):
                raise ValueError(f"{name} must be {(batch, width)}, got {tuple(value.shape)}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains non-finite values")
            if name == "end_effector_pose_root_xyzw":
                value = torch.cat(
                    (value[:, :3], canonicalize_quaternion_xyzw(value[:, 3:7])),
                    dim=-1,
                )
            pieces.append(value)
        observation = torch.cat(pieces, dim=-1)
        if observation.shape[-1] != G2_VISUAL_PROPRIO_DIM:
            raise RuntimeError("student observation dimension contract mismatch")
        return observation


def pack_rgbd(
    head_rgb: torch.Tensor,
    head_depth_m: torch.Tensor,
    wrist_rgb: torch.Tensor,
    wrist_depth_m: torch.Tensor,
    *,
    output_hw: tuple[int, int] = (48, 64),
    maximum_depth_m: float = 2.0,
) -> torch.Tensor:
    """Return compact uint8 ``[N,2,6,H,W]`` RGB/depth-hi/depth-lo/valid.

    Metric depth is clipped then encoded as an unsigned 16-bit integer split
    across two uint8 channels.  At the default 2 m range the quantization is
    about 0.031 mm, rather than the former 7.84 mm single-byte step.
    """

    if maximum_depth_m <= 0.0:
        raise ValueError("maximum_depth_m must be positive")

    def one(rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[-1] < 3:
            raise ValueError("RGB must have shape [N,H,W,C>=3]")
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 3 or depth.shape[:3] != rgb.shape[:3]:
            raise ValueError("depth must match RGB batch/height/width")
        rgb_chw = rgb[..., :3].permute(0, 3, 1, 2).float()
        if rgb.dtype != torch.uint8:
            rgb_chw = rgb_chw.clamp(0.0, 1.0).mul(255.0)
        valid = torch.isfinite(depth) & (depth > 0)
        valid_depth = torch.where(valid, depth, 0.0)
        depth_chw = valid_depth.clamp_max(maximum_depth_m).unsqueeze(1)
        valid_chw = valid.to(depth_chw.dtype).unsqueeze(1)
        rgb_small = F.interpolate(rgb_chw, output_hw, mode="area")
        depth_small = F.interpolate(depth_chw, output_hw, mode="nearest")
        valid_small = F.interpolate(valid_chw, output_hw, mode="nearest")
        depth_u16 = depth_small.mul(65535.0 / maximum_depth_m).round().clamp(0, 65535)
        depth_high = torch.floor(depth_u16 / 256.0)
        depth_low = depth_u16 - depth_high * 256.0
        return torch.cat(
            (rgb_small, depth_high, depth_low, valid_small.mul(255.0)), dim=1
        ).round().clamp(0, 255).to(torch.uint8)

    return torch.stack((one(head_rgb, head_depth_m), one(wrist_rgb, wrist_depth_m)), dim=1)


def random_shift_rgbd_sequences(
    rgbd_u8: torch.Tensor,
    *,
    pad: int = 4,
    return_principal_point_shift: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply one aligned integer random shift per sequence and camera.

    RGB, the two-byte metric depth encoding, and the depth-valid mask always
    receive the same crop.  The crop is also held constant across time so the
    augmentation cannot manufacture apparent camera or object motion for the
    GRU.  Augmentation is a learner-only operation; rollout and deployment
    observations are unchanged.
    """

    if rgbd_u8.ndim != 6 or tuple(rgbd_u8.shape[2:]) != G2_VISUAL_CAMERA_SHAPE:
        raise ValueError("RGB-D sequence must be [B,T,2,6,48,64]")
    if rgbd_u8.dtype != torch.uint8:
        raise ValueError("RGB-D random shift requires packed uint8 input")
    if pad < 0:
        raise ValueError("random-shift padding cannot be negative")
    batch, steps, cameras, channels, height, width = rgbd_u8.shape
    if pad == 0:
        principal_point_shift = torch.zeros(
            (batch, steps, cameras, 2),
            dtype=torch.float32,
            device=rgbd_u8.device,
        )
        return (
            (rgbd_u8, principal_point_shift)
            if return_principal_point_shift
            else rgbd_u8
        )
    # Keep the time dimension next to channels while applying a single crop
    # offset to each (batch, camera) pair.
    packed = rgbd_u8.permute(0, 2, 1, 3, 4, 5).reshape(
        batch * cameras, steps * channels, height, width
    )
    padded = F.pad(packed, (pad, pad, pad, pad), mode="replicate")
    windows = padded.unfold(2, height, 1).unfold(3, width, 1)
    offsets = torch.randint(
        0, 2 * pad + 1, (batch * cameras, 2), device=rgbd_u8.device
    )
    selected = windows[
        torch.arange(batch * cameras, device=rgbd_u8.device),
        :,
        offsets[:, 0],
        offsets[:, 1],
    ]
    shifted = selected.reshape(
        batch, cameras, steps, channels, height, width
    ).permute(0, 2, 1, 3, 4, 5).contiguous()
    # Padding followed by a crop at (offset_v, offset_u) moves the nominal
    # principal point by (pad-offset_u, pad-offset_v).  Focal length is
    # unchanged, so this is the exact K -> K' update for this augmentation.
    principal_point_shift = torch.stack(
        (
            float(pad) - offsets[:, 1].to(torch.float32),
            float(pad) - offsets[:, 0].to(torch.float32),
        ),
        dim=-1,
    ).reshape(batch, cameras, 2)
    principal_point_shift = principal_point_shift[:, None].expand(
        batch, steps, cameras, 2
    )
    return (
        (shifted, principal_point_shift)
        if return_principal_point_shift
        else shifted
    )


class _CameraRefinementStage(nn.Module):
    """One convolutional stage with a shape-safe residual connection."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, stride=1, padding=1
        )
        self.activation = nn.SiLU()
        self.residual = input_channels == output_channels

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        refined = self.activation(self.convolution(value))
        return value + refined if self.residual else refined


def _camera_encoder(
    input_channels: int,
    latent_dim: int,
    encoder_channels: tuple[int, ...],
) -> nn.Sequential:
    channels = tuple(int(value) for value in encoder_channels)
    if channels not in G2_CAMERA_ENCODER_PROFILES.values():
        raise ValueError("camera encoder channels must use a registered profile")
    layers: list[nn.Module] = []
    previous = int(input_channels)
    for index, output in enumerate(channels):
        # Four stride-2 stages preserve the established 48x64 -> 3x4 feature
        # geometry.  Optional later stages enrich capacity without changing
        # the visual embedding or GRU interface.
        if index >= 4:
            layers.append(_CameraRefinementStage(previous, output))
            previous = output
            continue
        kernel = 5 if index == 0 else 3
        padding = 2 if index == 0 else 1
        layers.extend((nn.Conv2d(previous, output, kernel, 2, padding), nn.SiLU()))
        previous = output
    layers.extend(
        (
            nn.Flatten(),
            nn.Linear(channels[-1] * 3 * 4, latent_dim),
            nn.LayerNorm(latent_dim),
        )
    )
    return nn.Sequential(*layers)


class G2TaskRelevantVisualEncoder(nn.Module):
    """Separate RGB/depth encoders for head and wrist observations.

    Depth receives its valid mask as a second channel.  Camera weight sharing
    is disabled by default because overview and eye-in-hand images have very
    different geometry; it remains a constructor option for ablation.
    """

    def __init__(
        self,
        *,
        latent_per_modality: int = 64,
        output_dim: int = 64,
        share_camera_encoder_weights: bool = False,
        encoder_channels: tuple[int, ...] = G2_CAMERA_ENCODER_PROFILES[
            "baseline_4layer"
        ],
    ) -> None:
        super().__init__()
        self.latent_per_modality = int(latent_per_modality)
        self.output_dim = int(output_dim)
        self.share_camera_encoder_weights = bool(share_camera_encoder_weights)
        self.encoder_channels = tuple(int(value) for value in encoder_channels)
        if self.encoder_channels not in G2_CAMERA_ENCODER_PROFILES.values():
            raise ValueError("unknown camera encoder profile")
        camera_count = 1 if self.share_camera_encoder_weights else 2
        self.rgb_encoders = nn.ModuleList(
            [
                _camera_encoder(3, self.latent_per_modality, self.encoder_channels)
                for _ in range(camera_count)
            ]
        )
        self.depth_encoders = nn.ModuleList(
            [
                _camera_encoder(2, self.latent_per_modality, self.encoder_channels)
                for _ in range(camera_count)
            ]
        )
        self.camera_gates = nn.ModuleList(
            [nn.Linear(2 * self.latent_per_modality, self.latent_per_modality) for _ in range(2)]
        )
        self.camera_residual_fusion = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.latent_per_modality, self.latent_per_modality),
                    nn.SiLU(),
                    nn.LayerNorm(self.latent_per_modality),
                )
                for _ in range(2)
            ]
        )
        self.intrinsic_shift_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2, self.latent_per_modality),
                    nn.SiLU(),
                    nn.Linear(self.latent_per_modality, self.latent_per_modality),
                )
                for _ in range(2)
            ]
        )
        self.cross_camera_attention = nn.MultiheadAttention(
            self.latent_per_modality, num_heads=4, batch_first=True
        )
        self.cross_camera_norm = nn.LayerNorm(self.latent_per_modality)
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.latent_per_modality, self.output_dim),
            nn.SiLU(),
            nn.LayerNorm(self.output_dim),
        )

    def encode_with_camera_tokens(
        self,
        rgbd_u8: torch.Tensor,
        principal_point_shift_px: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(rgbd_u8.shape[1:]) != G2_VISUAL_CAMERA_SHAPE:
            raise ValueError(f"RGB-D must be [N,{G2_VISUAL_CAMERA_SHAPE}], got {tuple(rgbd_u8.shape)}")
        if principal_point_shift_px is None:
            principal_point_shift_px = torch.zeros(
                (rgbd_u8.shape[0], 2, 2),
                dtype=torch.float32,
                device=rgbd_u8.device,
            )
        if tuple(principal_point_shift_px.shape) != (rgbd_u8.shape[0], 2, 2):
            raise ValueError("principal-point shift must be [N,2,2] in (du,dv) pixels")
        camera_features = []
        for camera_index in range(2):
            encoder_index = 0 if self.share_camera_encoder_weights else camera_index
            packed = rgbd_u8[:, camera_index]
            rgb = packed[:, :3].float().div(255.0)
            depth_code = (
                packed[:, 3].float() * 256.0 + packed[:, 4].float()
            )
            depth_normalized = depth_code.div(65535.0).unsqueeze(1)
            valid = packed[:, 5:6].float().div(255.0)
            depth_and_mask = torch.cat((depth_normalized, valid), dim=1)
            rgb_feature = self.rgb_encoders[encoder_index](rgb)
            depth_feature = self.depth_encoders[encoder_index](depth_and_mask)
            paired = torch.cat((rgb_feature, depth_feature), dim=-1)
            gate = torch.sigmoid(self.camera_gates[camera_index](paired))
            gated = gate * rgb_feature + (1.0 - gate) * depth_feature
            normalized_shift = principal_point_shift_px[:, camera_index].to(
                device=paired.device, dtype=paired.dtype
            ) / paired.new_tensor((64.0, 48.0))
            camera_features.append(
                gated
                + self.camera_residual_fusion[camera_index](paired)
                + self.intrinsic_shift_encoders[camera_index](normalized_shift)
            )
        # Preserve the pre-attention, camera-specific tokens for the
        # cross-view objective.  Using post-attention tokens would leak each
        # camera into the other and make the positive-pair task trivial.
        camera_tokens = torch.stack(camera_features, dim=1)
        tokens = camera_tokens
        attended, _ = self.cross_camera_attention(
            tokens, tokens, tokens, need_weights=False
        )
        tokens = self.cross_camera_norm(tokens + attended)
        return self.fusion(tokens.flatten(start_dim=1)), camera_tokens

    def encode(
        self,
        rgbd_u8: torch.Tensor,
        principal_point_shift_px: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused, _ = self.encode_with_camera_tokens(
            rgbd_u8, principal_point_shift_px
        )
        return fused


class _VisualActor(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(feature_dim, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU())
        if action_dim != G2_VISUAL_ACTION_DIM:
            raise ValueError("G2 visual actor requires the sealed 6+1 action contract")
        self.mean = nn.Linear(256, G2_VISUAL_ARM_ACTION_DIM)
        self.log_std = nn.Linear(256, G2_VISUAL_ARM_ACTION_DIM)
        self.gripper_logit = nn.Linear(256, 1)
        nn.init.zeros_(self.mean.bias)
        nn.init.constant_(self.log_std.bias, -2.0)
        nn.init.zeros_(self.gripper_logit.bias)

    def distribution(self, features: torch.Tensor):
        hidden = self.trunk(features)
        mean = self.mean(hidden)
        log_std = -5.0 + 3.5 * (torch.tanh(self.log_std(hidden)) + 1.0)
        return mean, log_std, self.gripper_logit(hidden)

    def sample_arm(self, features: torch.Tensor, deterministic: bool = False):
        mean, log_std, gripper_logit = self.distribution(features)
        pre_tanh = (
            mean
            if deterministic
            else torch.distributions.Normal(mean, log_std.exp()).rsample()
        )
        arm_action = torch.tanh(pre_tanh)
        arm_log_prob = squashed_gaussian_log_prob(pre_tanh, mean, log_std)
        return arm_action, arm_log_prob, torch.tanh(mean), log_std, gripper_logit

    def sample(self, features: torch.Tensor, deterministic: bool = False):
        arm, arm_log_prob, arm_mean, log_std, gripper_logit = self.sample_arm(
            features, deterministic
        )
        probability_open = torch.sigmoid(gripper_logit)
        if deterministic:
            open_sample = probability_open >= 0.5
        else:
            open_sample = torch.bernoulli(probability_open).to(torch.bool)
        gripper = torch.where(
            open_sample,
            torch.ones_like(probability_open),
            -torch.ones_like(probability_open),
        )
        selected_probability = torch.where(
            open_sample, probability_open, 1.0 - probability_open
        )
        log_probability = arm_log_prob + torch.log(
            selected_probability.clamp_min(1.0e-8)
        )
        mean_action = torch.cat(
            (
                arm_mean,
                torch.where(
                    probability_open >= 0.5,
                    torch.ones_like(probability_open),
                    -torch.ones_like(probability_open),
                ),
            ),
            dim=-1,
        )
        return torch.cat((arm, gripper), dim=-1), log_probability, mean_action, log_std


class _Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(G2_VISUAL_PRIVILEGED_DIM + G2_VISUAL_ACTION_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((state, action), dim=-1))


@dataclass(frozen=True)
class G2VisualSACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3.0e-4
    initial_alpha: float = 0.1
    target_entropy: float | None = None
    relative_pose_weight: float = 0.2
    pose_consistency_weight: float = 0.1
    pose_consistency_rotation_weight: float = 0.1
    contact_weight: float = 0.1
    depth_validity_weight: float = 0.05
    share_camera_encoder_weights: bool = False
    camera_encoder_channels: tuple[int, ...] = G2_CAMERA_ENCODER_PROFILES[
        "baseline_4layer"
    ]
    gradient_clip: float = 5.0

    @property
    def resolved_target_entropy(self) -> float:
        return (
            -(float(G2_VISUAL_ARM_ACTION_DIM) + math.log(2.0))
            if self.target_entropy is None
            else float(self.target_entropy)
        )


class G2VisualAsymmetricSAC:
    """Feed-forward visual actor and privileged twin-Q critics.

    A recurrent policy is deliberately not used: camera capture is faster than
    the 50 Hz policy boundary, and single-transition replay remains valid.  If
    later occlusion metrics show temporal aliasing, a sequence replay contract
    must be introduced before GRU/LSTM is enabled.
    """

    def __init__(self, config: G2VisualSACConfig | None = None, *, device="cpu") -> None:
        self.config = config or G2VisualSACConfig()
        self.device = torch.device(device)
        self.visual = G2TaskRelevantVisualEncoder(
            share_camera_encoder_weights=self.config.share_camera_encoder_weights,
            encoder_channels=self.config.camera_encoder_channels,
        ).to(self.device)
        self.actor = _VisualActor(64 + G2_VISUAL_PROPRIO_DIM, G2_VISUAL_ACTION_DIM).to(self.device)
        self.relative_pose_head = nn.Sequential(
            nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 7)
        ).to(self.device)
        self.contact_head = nn.Sequential(
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 3)
        ).to(self.device)
        self.depth_validity_head = nn.Sequential(
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 2)
        ).to(self.device)
        self.q1, self.q2 = _Critic().to(self.device), _Critic().to(self.device)
        self.tq1, self.tq2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.tq1.requires_grad_(False); self.tq2.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            list(self.visual.parameters())
            + list(self.actor.parameters())
            + list(self.relative_pose_head.parameters())
            + list(self.contact_head.parameters())
            + list(self.depth_validity_head.parameters()),
            lr=self.config.learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.config.learning_rate
        )
        self.log_alpha = torch.tensor(math.log(self.config.initial_alpha), device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.learning_rate)
        self.update_count = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _features(self, rgbd, proprio):
        return torch.cat((self.visual.encode(rgbd), proprio), dim=-1)

    @staticmethod
    def _hybrid_expectation(
        arm_action: torch.Tensor,
        arm_log_probability: torch.Tensor,
        gripper_logit: torch.Tensor,
        q1,
        q2,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return exact binary-gripper expectation for Q and log pi."""

        probability_open = torch.sigmoid(gripper_logit)
        closed_action = torch.cat(
            (arm_action, -torch.ones_like(probability_open)), dim=-1
        )
        open_action = torch.cat(
            (arm_action, torch.ones_like(probability_open)), dim=-1
        )
        closed_q = torch.minimum(q1(state, closed_action), q2(state, closed_action))
        open_q = torch.minimum(q1(state, open_action), q2(state, open_action))
        expected_q = (1.0 - probability_open) * closed_q + probability_open * open_q
        binary_expected_log_probability = (
            (1.0 - probability_open)
            * torch.log((1.0 - probability_open).clamp_min(1.0e-8))
            + probability_open * torch.log(probability_open.clamp_min(1.0e-8))
        )
        return (
            expected_q,
            arm_log_probability + binary_expected_log_probability,
            probability_open,
        )

    @torch.no_grad()
    def select_actions(self, rgbd_u8, proprio, *, deterministic=False) -> np.ndarray:
        rgbd = torch.as_tensor(rgbd_u8, dtype=torch.uint8, device=self.device)
        prop = torch.as_tensor(proprio, dtype=torch.float32, device=self.device)
        action, _, mean, _ = self.actor.sample(self._features(rgbd, prop), deterministic)
        return (mean if deterministic else action).cpu().numpy()

    def update(self, batch: Mapping[str, np.ndarray]) -> dict[str, float]:
        t = lambda name, dtype=torch.float32: torch.as_tensor(batch[name], dtype=dtype, device=self.device)
        rgbd, next_rgbd = t("rgbd", torch.uint8), t("next_rgbd", torch.uint8)
        prop, next_prop = t("proprio"), t("next_proprio")
        state, next_state = t("privileged"), t("next_privileged")
        action, reward, terminated = t("actions"), t("rewards"), t("terminated")
        with torch.no_grad():
            next_features = self._features(next_rgbd, next_prop)
            next_arm, next_arm_logp, _, _, next_gripper_logit = (
                self.actor.sample_arm(next_features)
            )
            target_q, next_logp, _ = self._hybrid_expectation(
                next_arm,
                next_arm_logp,
                next_gripper_logit,
                self.tq1,
                self.tq2,
                next_state,
            )
            target = reward + self.config.gamma * (1.0 - terminated) * (
                target_q - self.alpha.detach() * next_logp
            )
        q1, q2 = self.q1(state, action), self.q2(state, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True); critic_loss.backward()
        critic_norm = nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), self.config.gradient_clip)
        self.critic_optimizer.step()

        features = self._features(rgbd, prop)
        sampled_arm, arm_logp, _, log_std, gripper_logit = self.actor.sample_arm(
            features
        )
        for parameter in list(self.q1.parameters()) + list(self.q2.parameters()):
            parameter.requires_grad_(False)
        expected_q, logp, probability_open = self._hybrid_expectation(
            sampled_arm, arm_logp, gripper_logit, self.q1, self.q2, state
        )
        actor_loss = (self.alpha.detach() * logp - expected_q).mean()
        for parameter in list(self.q1.parameters()) + list(self.q2.parameters()):
            parameter.requires_grad_(True)
        visual_latent = features[:, :64]
        # Privileged values supervise task-relevant heads only.  They are not
        # concatenated into the visual actor's deployment input.
        relative_target = relative_pose_target_from_teacher_state(state)
        relative_prediction = self.relative_pose_head(visual_latent)
        relative_pose_loss = relative_pose_auxiliary_loss(relative_prediction, relative_target)
        contact_slice = G2TeacherObservationContract().slices["bilateral_contact_features"]
        contact_target = state[:, contact_slice].clamp(0.0, 1.0)
        contact_logits = self.contact_head(visual_latent)
        contact_loss = F.binary_cross_entropy_with_logits(contact_logits, contact_target)
        depth_validity_target = rgbd[:, :, 5].float().div(255.0).mean(dim=(-1, -2))
        depth_validity_logits = self.depth_validity_head(visual_latent)
        depth_validity_loss = F.binary_cross_entropy_with_logits(
            depth_validity_logits, depth_validity_target
        )
        visual_loss = (
            self.config.relative_pose_weight * relative_pose_loss
            + self.config.contact_weight * contact_loss
            + self.config.depth_validity_weight * depth_validity_loss
        )
        total_actor_loss = actor_loss + visual_loss
        self.actor_optimizer.zero_grad(set_to_none=True); total_actor_loss.backward()
        actor_norm = nn.utils.clip_grad_norm_(
            list(self.visual.parameters())
            + list(self.actor.parameters())
            + list(self.relative_pose_head.parameters())
            + list(self.contact_head.parameters())
            + list(self.depth_validity_head.parameters()),
            self.config.gradient_clip,
        )
        self.actor_optimizer.step()

        # Match the repository SAC temperature contract exactly.  The default
        # target entropy is -action_dim; using +action_dim here would reverse
        # the configured entropy target and make visual/teacher SAC diverge.
        alpha_loss = -(
            self.log_alpha
            * (logp.detach() + self.config.resolved_target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True); alpha_loss.backward(); self.alpha_optimizer.step()
        polyak_update(self.q1, self.tq1, self.config.tau); polyak_update(self.q2, self.tq2, self.config.tau)
        self.update_count += 1
        return {
            "loss/actor": float(actor_loss.detach()), "loss/critic": float(critic_loss.detach()),
            "loss/visual_relative_pose": float(relative_pose_loss.detach()),
            "loss/visual_contact": float(contact_loss.detach()),
            "loss/visual_depth_validity": float(depth_validity_loss.detach()),
            "loss/actor_total": float(total_actor_loss.detach()),
            "loss/alpha": float(alpha_loss.detach()),
            "visual/relative_position_error_m": float(
                torch.linalg.vector_norm(
                    relative_prediction.detach()[:, :3] - relative_target[:, :3], dim=-1
                ).mean()
            ),
            "visual/contact_probability_mean": float(torch.sigmoid(contact_logits.detach()).mean()),
            "q/q1_mean": float(q1.detach().mean()), "q/q2_mean": float(q2.detach().mean()),
            "q/disagreement_mean": float(torch.abs(q1.detach() - q2.detach()).mean()),
            "entropy/alpha": float(self.alpha.detach()), "entropy/log_std_mean": float(log_std.detach().mean()),
            "entropy/gripper_open_probability_mean": float(
                probability_open.detach().mean()
            ),
            "entropy/target": float(self.config.resolved_target_entropy),
            "gradient/actor_visual_norm": float(actor_norm), "gradient/critic_norm": float(critic_norm),
        }

    def state_dict(self) -> dict:
        return {
            "schema": G2_VISUAL_SAC_SCHEMA,
            "config": self.config.__dict__,
            "visual": self.visual.state_dict(), "actor": self.actor.state_dict(),
            "relative_pose_head": self.relative_pose_head.state_dict(),
            "contact_head": self.contact_head.state_dict(),
            "depth_validity_head": self.depth_validity_head.state_dict(),
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "tq1": self.tq1.state_dict(), "tq2": self.tq2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "update_count": self.update_count,
        }


class G2RecurrentVisualStudent(nn.Module):
    """Deployment-safe RGB-D GRU used for BC/distillation and DAgger.

    Sequences must never cross an episode boundary.  Simulator cube/contact/
    phase state is used only as an auxiliary or value target during training;
    it is absent from :meth:`forward_sequence` inputs at deployment.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        gru_num_layers: int = 1,
        value_loss_weight: float = 0.2,
        auxiliary_loss_weight: float = 0.2,
        pose_consistency_loss_weight: float = 0.1,
        pose_consistency_rotation_weight: float = 0.1,
        cross_camera_contrastive_loss_weight: float = 0.02,
        cross_camera_contrastive_temperature: float = 0.1,
        cross_camera_false_negative_position_threshold_m: float = 0.01,
        temporal_pose_residual_loss_weight: float = 0.05,
        temporal_pose_residual_rotation_weight: float = 0.1,
        contact_loss_weight: float = 0.1,
        stable_grasp_loss_weight: float = 0.1,
        slip_speed_loss_weight: float = 0.1,
        depth_validity_loss_weight: float = 0.05,
        failure_loss_weight: float = 0.1,
        share_camera_encoder_weights: bool = False,
        camera_encoder_channels: tuple[int, ...] = G2_CAMERA_ENCODER_PROFILES[
            "baseline_4layer"
        ],
        torso_control_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.visual = G2TaskRelevantVisualEncoder(
            share_camera_encoder_weights=share_camera_encoder_weights,
            encoder_channels=camera_encoder_channels,
        )
        self.cross_camera_projection = nn.Sequential(
            nn.Linear(self.visual.latent_per_modality, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(G2_VISUAL_PROPRIO_DIM, 64), nn.SiLU(), nn.LayerNorm(64)
        )
        self.hidden_dim = int(hidden_dim)
        self.gru_num_layers = int(gru_num_layers)
        if self.gru_num_layers != 1:
            raise ValueError("G2 recurrent visual policy requires exactly one GRU layer")
        self.value_loss_weight = float(value_loss_weight)
        self.auxiliary_loss_weight = float(auxiliary_loss_weight)
        self.pose_consistency_loss_weight = float(pose_consistency_loss_weight)
        self.pose_consistency_rotation_weight = float(pose_consistency_rotation_weight)
        self.cross_camera_contrastive_loss_weight = float(
            cross_camera_contrastive_loss_weight
        )
        self.cross_camera_contrastive_temperature = float(
            cross_camera_contrastive_temperature
        )
        self.cross_camera_false_negative_position_threshold_m = float(
            cross_camera_false_negative_position_threshold_m
        )
        self.temporal_pose_residual_loss_weight = float(
            temporal_pose_residual_loss_weight
        )
        self.temporal_pose_residual_rotation_weight = float(
            temporal_pose_residual_rotation_weight
        )
        self.contact_loss_weight = float(contact_loss_weight)
        self.stable_grasp_loss_weight = float(stable_grasp_loss_weight)
        self.slip_speed_loss_weight = float(slip_speed_loss_weight)
        self.depth_validity_loss_weight = float(depth_validity_loss_weight)
        self.failure_loss_weight = float(failure_loss_weight)
        if self.cross_camera_contrastive_loss_weight < 0.0:
            raise ValueError("cross-camera contrastive weight cannot be negative")
        if self.cross_camera_contrastive_temperature <= 0.0:
            raise ValueError("cross-camera contrastive temperature must be positive")
        if self.cross_camera_false_negative_position_threshold_m < 0.0:
            raise ValueError("cross-camera false-negative threshold cannot be negative")
        if self.temporal_pose_residual_loss_weight < 0.0:
            raise ValueError("temporal pose-residual weight cannot be negative")
        if self.temporal_pose_residual_rotation_weight < 0.0:
            raise ValueError("temporal pose-residual rotation weight cannot be negative")
        self.torso_control_enabled = bool(torso_control_enabled)
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=self.hidden_dim,
            num_layers=self.gru_num_layers,
            batch_first=True,
        )
        self.arm_action_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128), nn.SiLU(), nn.Linear(128, 6), nn.Tanh()
        )
        self.gripper_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1), nn.Tanh()
        )
        self.torso_action_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64), nn.SiLU(), nn.Linear(64, 5), nn.Tanh()
        )
        self.torso_gate_head = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
        if not self.torso_control_enabled:
            self.torso_action_head.requires_grad_(False)
            self.torso_gate_head.requires_grad_(False)
        self.value_head = nn.Sequential(nn.Linear(self.hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1))
        self.relative_pose_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64), nn.SiLU(), nn.Linear(64, 7)
        )
        self.contact_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64), nn.SiLU(), nn.Linear(64, 3)
        )
        self.stable_grasp_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 32), nn.SiLU(), nn.Linear(32, 1)
        )
        self.slip_speed_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 32), nn.SiLU(), nn.Linear(32, 1)
        )
        self.depth_validity_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 32), nn.SiLU(), nn.Linear(32, 2)
        )
        self.failure_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64), nn.SiLU(),
            nn.Linear(64, len(G2_STUDENT_FAILURE_CLASSES)),
        )

    def initial_hidden(self, batch: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
        parameter = next(self.parameters())
        return torch.zeros(
            (self.gru_num_layers, int(batch), self.hidden_dim),
            device=parameter.device if device is None else device,
            dtype=dtype,
        )

    @staticmethod
    def validate_sequence_boundaries(
        episode_id: torch.Tensor,
        sequence_step: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if episode_id.ndim != 2 or sequence_step.shape != episode_id.shape:
            raise ValueError("episode_id and sequence_step must both be [B,T]")
        if padding_mask is None:
            padding_mask = torch.ones_like(episode_id, dtype=torch.bool)
        if padding_mask.shape != episode_id.shape:
            raise ValueError("padding_mask must be [B,T]")
        if sequence_lengths is not None:
            if tuple(sequence_lengths.shape) != (episode_id.shape[0],):
                raise ValueError("sequence_lengths must be [B]")
            expected = torch.arange(episode_id.shape[1], device=episode_id.device)[None, :]
            expected = expected < sequence_lengths[:, None]
            if not torch.equal(expected, padding_mask.to(torch.bool)):
                raise ValueError("sequence_lengths and padding_mask disagree")
        adjacent = padding_mask[:, 1:] & padding_mask[:, :-1]
        if bool(((episode_id[:, 1:] != episode_id[:, :-1]) & adjacent).any()):
            raise ValueError("student sequence crosses an episode boundary")
        if bool(((sequence_step[:, 1:] != sequence_step[:, :-1] + 1) & adjacent).any()):
            raise ValueError("student sequence is not temporally contiguous")
        return padding_mask.to(torch.bool)

    def forward_sequence(
        self,
        rgbd_u8: torch.Tensor,
        deployable_proprioception: torch.Tensor,
        hidden: torch.Tensor | None = None,
        hidden_reset_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        camera_principal_point_shift_px: torch.Tensor | None = None,
        *,
        return_camera_tokens: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        encoded = self.encode_recurrent_features(
            rgbd_u8,
            deployable_proprioception,
            hidden,
            hidden_reset_mask,
            padding_mask,
            camera_principal_point_shift_px,
            return_camera_tokens=return_camera_tokens,
        )
        if return_camera_tokens:
            recurrent, next_hidden, projected_camera_tokens = encoded
        else:
            recurrent, next_hidden = encoded
            projected_camera_tokens = None
        arm_action = self.arm_action_head(recurrent)
        gripper_action = self.gripper_head(recurrent)
        torso_proposal = self.torso_action_head(recurrent)
        torso_gate_probability = self.torso_gate_head(recurrent)
        if self.torso_control_enabled:
            torso_action = torso_proposal * torso_gate_probability
        else:
            torso_action = torch.zeros_like(torso_proposal)
            torso_gate_probability = torch.zeros_like(torso_gate_probability)
        result = {
            "arm_action": arm_action,
            "gripper_action": gripper_action,
            "action": torch.cat((arm_action, gripper_action), dim=-1),
            "torso_action": torso_action,
            "torso_action_proposal": torso_proposal,
            "torso_gate_probability": torso_gate_probability,
            "value": self.value_head(recurrent),
            "relative_pose_aux": self.relative_pose_head(recurrent),
            "contact_logits": self.contact_head(recurrent),
            "stable_grasp_logits": self.stable_grasp_head(recurrent),
            # Softplus keeps the physical regression non-negative.  The loss
            # below normalizes by the sealed 0.06 m/s stable-grasp threshold.
            "slip_speed_m_s": F.softplus(self.slip_speed_head(recurrent)) * 0.06,
            "depth_validity_logits": self.depth_validity_head(recurrent),
            "failure_logits": self.failure_head(recurrent),
        }
        if projected_camera_tokens is not None:
            result["projected_camera_tokens"] = projected_camera_tokens
        return result, next_hidden

    def encode_recurrent_features(
        self,
        rgbd_u8: torch.Tensor,
        deployable_proprioception: torch.Tensor,
        hidden: torch.Tensor | None = None,
        hidden_reset_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        camera_principal_point_shift_px: torch.Tensor | None = None,
        *,
        return_camera_tokens: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the canonical keyboard/teacher input with the shared GRU.

        Both online visual SAC and offline keyboard distillation call this
        exact method.  Keeping the recurrent backbone in one class prevents
        camera channel, proprioception ordering, and hidden-reset drift.
        """

        if rgbd_u8.ndim != 6 or tuple(rgbd_u8.shape[2:]) != G2_VISUAL_CAMERA_SHAPE:
            raise ValueError("student RGB-D must be [B,T,2,6,48,64]")
        batch, steps = rgbd_u8.shape[:2]
        if tuple(deployable_proprioception.shape) != (batch, steps, G2_VISUAL_PROPRIO_DIM):
            raise ValueError(f"student proprioception must be [B,T,{G2_VISUAL_PROPRIO_DIM}]")
        if camera_principal_point_shift_px is None:
            camera_principal_point_shift_px = torch.zeros(
                (batch, steps, 2, 2), dtype=torch.float32, device=rgbd_u8.device
            )
        if tuple(camera_principal_point_shift_px.shape) != (batch, steps, 2, 2):
            raise ValueError("camera principal-point shift must be [B,T,2,2]")
        visual, camera_tokens = self.visual.encode_with_camera_tokens(
            rgbd_u8.reshape(batch * steps, *G2_VISUAL_CAMERA_SHAPE),
            camera_principal_point_shift_px.reshape(batch * steps, 2, 2),
        )
        projected_camera_tokens = self.cross_camera_projection(camera_tokens).reshape(
            batch, steps, 2, -1
        )
        state = self.state_encoder(deployable_proprioception.reshape(batch * steps, -1))
        feature = torch.cat((visual, state), dim=-1).reshape(batch, steps, -1)
        if hidden is None:
            hidden = self.initial_hidden(batch, device=feature.device, dtype=feature.dtype)
        if hidden_reset_mask is None:
            hidden_reset_mask = torch.zeros((batch, steps), dtype=torch.bool, device=feature.device)
            hidden_reset_mask[:, 0] = True
        if tuple(hidden_reset_mask.shape) != (batch, steps):
            raise ValueError("hidden_reset_mask must be [B,T]")
        if padding_mask is None:
            padding_mask = torch.ones((batch, steps), dtype=torch.bool, device=feature.device)
        if tuple(padding_mask.shape) != (batch, steps):
            raise ValueError("padding_mask must be [B,T]")
        padding_mask = padding_mask.to(device=feature.device, dtype=torch.bool)
        if bool((padding_mask[:, 1:] & ~padding_mask[:, :-1]).any()):
            raise ValueError("padding_mask must be a left-aligned valid prefix")
        if bool((hidden_reset_mask.to(torch.bool) & ~padding_mask).any()):
            raise ValueError("hidden reset cannot occur on a padded timestep")
        outputs = []
        next_hidden = hidden
        for step in range(steps):
            reset = hidden_reset_mask[:, step].to(device=feature.device, dtype=torch.bool)
            next_hidden = torch.where(
                reset.view(1, batch, 1), torch.zeros_like(next_hidden), next_hidden
            )
            proposed_output, proposed_hidden = self.gru(
                feature[:, step : step + 1], next_hidden
            )
            active = padding_mask[:, step]
            next_hidden = torch.where(
                active.view(1, batch, 1), proposed_hidden, next_hidden
            )
            outputs.append(
                torch.where(
                    active.view(batch, 1, 1),
                    proposed_output,
                    torch.zeros_like(proposed_output),
                )
            )
        recurrent = torch.cat(outputs, dim=1)
        if return_camera_tokens:
            return recurrent, next_hidden, projected_camera_tokens
        return recurrent, next_hidden

    def distillation_loss(
        self,
        *,
        rgbd_u8: torch.Tensor,
        pose_consistency_rgbd_u8: torch.Tensor | None = None,
        camera_principal_point_shift_px: torch.Tensor | None = None,
        deployable_proprioception: torch.Tensor,
        expert_action: torch.Tensor,
        expert_confidence: torch.Tensor,
        episode_id: torch.Tensor,
        sequence_step: torch.Tensor,
        teacher_value: torch.Tensor | None = None,
        teacher_value_valid: torch.Tensor | None = None,
        relative_pose_target: torch.Tensor | None = None,
        contact_target: torch.Tensor | None = None,
        stable_grasp_target: torch.Tensor | None = None,
        slip_speed_target_m_s: torch.Tensor | None = None,
        depth_validity_target: torch.Tensor | None = None,
        failure_target: torch.Tensor | None = None,
        failure_target_valid: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
        hidden_reset_mask: torch.Tensor | None = None,
        burn_in_steps: int = 0,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        if burn_in_steps < 0 or burn_in_steps >= rgbd_u8.shape[1]:
            raise ValueError("burn_in_steps must be in [0,T)")
        padding_mask = self.validate_sequence_boundaries(
            episode_id, sequence_step, padding_mask, sequence_lengths
        )
        output, next_hidden = self.forward_sequence(
            rgbd_u8,
            deployable_proprioception,
            hidden,
            hidden_reset_mask,
            padding_mask,
            camera_principal_point_shift_px,
            return_camera_tokens=True,
        )
        if expert_action.shape != output["action"].shape:
            raise ValueError("expert action target must be [B,T,7]")
        if expert_confidence.shape != output["value"].shape:
            raise ValueError("expert confidence must be [B,T,1]")
        loss_mask = padding_mask.unsqueeze(-1).to(expert_confidence.dtype)
        loss_mask[:, :burn_in_steps] = 0.0
        confidence = torch.clamp(expert_confidence, 0.0, 1.0) * loss_mask
        denominator = torch.clamp(confidence.sum(), min=1.0)
        imitation = (
            confidence * (output["action"] - expert_action).square().mean(dim=-1, keepdim=True)
        ).sum() / denominator
        value = output["value"].new_zeros(())
        if teacher_value is not None:
            if teacher_value.shape != output["value"].shape:
                raise ValueError("teacher value target must be [B,T,1]")
            if teacher_value_valid is None:
                teacher_value_valid = torch.ones_like(teacher_value, dtype=torch.bool)
            if teacher_value_valid.shape != teacher_value.shape:
                raise ValueError("teacher value validity must be [B,T,1]")
            value_weight = confidence * teacher_value_valid.to(confidence.dtype)
            value_denominator = torch.clamp(value_weight.sum(), min=1.0)
            value = (
                value_weight * (output["value"] - teacher_value).square()
            ).sum() / value_denominator
        relative_pose = output["value"].new_zeros(())
        if relative_pose_target is not None:
            if relative_pose_target.shape != output["relative_pose_aux"].shape:
                raise ValueError("relative pose target must be [B,T,7]")
            relative_error = relative_pose_auxiliary_loss(
                output["relative_pose_aux"], relative_pose_target, reduction="none"
            ).unsqueeze(-1)
            relative_pose = (loss_mask * relative_error).sum() / torch.clamp(loss_mask.sum(), min=1.0)
        pose_consistency = output["value"].new_zeros(())
        if pose_consistency_rgbd_u8 is not None:
            if pose_consistency_rgbd_u8.shape != rgbd_u8.shape:
                raise ValueError("pose consistency RGB-D view must match RGB-D input")
            consistency_output, _ = self.forward_sequence(
                pose_consistency_rgbd_u8,
                deployable_proprioception,
                hidden,
                hidden_reset_mask,
                padding_mask,
                None,
            )
            consistency_error = relative_pose_consistency_loss(
                consistency_output["relative_pose_aux"],
                output["relative_pose_aux"],
                rotation_weight=self.pose_consistency_rotation_weight,
                reduction="none",
            ).unsqueeze(-1)
            pose_consistency = (
                loss_mask * consistency_error
            ).sum() / torch.clamp(loss_mask.sum(), min=1.0)
        cross_camera_contrastive = output["value"].new_zeros(())
        temporal_pose_residual = output["value"].new_zeros(())
        if relative_pose_target is not None:
            contrastive_valid = loss_mask[..., 0].to(torch.bool)
            if depth_validity_target is not None:
                contrastive_valid &= (depth_validity_target > 0.0).all(dim=-1)
            cross_camera_contrastive = cross_camera_contrastive_loss(
                output["projected_camera_tokens"],
                contrastive_valid,
                relative_pose_target=relative_pose_target,
                temperature=self.cross_camera_contrastive_temperature,
                false_negative_position_threshold_m=(
                    self.cross_camera_false_negative_position_threshold_m
                ),
            )
            temporal_pose_residual = temporal_pose_residual_loss(
                output["relative_pose_aux"],
                relative_pose_target,
                loss_mask[..., 0].to(torch.bool),
                rotation_weight=self.temporal_pose_residual_rotation_weight,
            )
        contact = output["value"].new_zeros(())
        if contact_target is not None:
            if contact_target.shape != output["contact_logits"].shape:
                raise ValueError("contact target must be [B,T,3]")
            contact_error = F.binary_cross_entropy_with_logits(
                output["contact_logits"], contact_target, reduction="none"
            ).mean(dim=-1, keepdim=True)
            contact = (loss_mask * contact_error).sum() / torch.clamp(loss_mask.sum(), min=1.0)
        stable_grasp = output["value"].new_zeros(())
        if stable_grasp_target is not None:
            if stable_grasp_target.shape != output["stable_grasp_logits"].shape:
                raise ValueError("stable-grasp target must be [B,T,1]")
            stable_error = F.binary_cross_entropy_with_logits(
                output["stable_grasp_logits"], stable_grasp_target, reduction="none"
            )
            stable_grasp = (loss_mask * stable_error).sum() / torch.clamp(
                loss_mask.sum(), min=1.0
            )
        slip_speed = output["value"].new_zeros(())
        if slip_speed_target_m_s is not None:
            if slip_speed_target_m_s.shape != output["slip_speed_m_s"].shape:
                raise ValueError("slip-speed target must be [B,T,1]")
            if bool((slip_speed_target_m_s < 0.0).any()):
                raise ValueError("slip-speed target cannot be negative")
            slip_error = F.smooth_l1_loss(
                output["slip_speed_m_s"] / 0.06,
                slip_speed_target_m_s / 0.06,
                reduction="none",
            )
            slip_speed = (loss_mask * slip_error).sum() / torch.clamp(
                loss_mask.sum(), min=1.0
            )
        depth_validity = output["value"].new_zeros(())
        if depth_validity_target is not None:
            if depth_validity_target.shape != output["depth_validity_logits"].shape:
                raise ValueError("depth-validity target must be [B,T,2]")
            depth_error = F.binary_cross_entropy_with_logits(
                output["depth_validity_logits"], depth_validity_target, reduction="none"
            ).mean(dim=-1, keepdim=True)
            depth_validity = (loss_mask * depth_error).sum() / torch.clamp(loss_mask.sum(), min=1.0)
        failure = output["value"].new_zeros(())
        if failure_target is not None:
            if failure_target.shape != episode_id.shape:
                raise ValueError("failure target must be [B,T]")
            if failure_target_valid is None:
                failure_target_valid = torch.ones_like(
                    failure_target, dtype=torch.bool
                )
            if failure_target_valid.shape != episode_id.shape:
                raise ValueError("failure target validity must be [B,T]")
            failure_error = F.cross_entropy(
                output["failure_logits"].reshape(-1, len(G2_STUDENT_FAILURE_CLASSES)),
                failure_target.reshape(-1),
                reduction="none",
            ).reshape(*episode_id.shape, 1)
            failure_weight = loss_mask * failure_target_valid.unsqueeze(-1).to(
                loss_mask.dtype
            )
            failure = (failure_weight * failure_error).sum() / torch.clamp(
                failure_weight.sum(), min=1.0
            )
        auxiliary = (
            relative_pose
            + self.pose_consistency_loss_weight * pose_consistency
            + self.cross_camera_contrastive_loss_weight
            * cross_camera_contrastive
            + self.temporal_pose_residual_loss_weight * temporal_pose_residual
        )
        total = (
            imitation
            + self.value_loss_weight * value
            + self.auxiliary_loss_weight * relative_pose
            + self.pose_consistency_loss_weight * pose_consistency
            + self.cross_camera_contrastive_loss_weight
            * cross_camera_contrastive
            + self.temporal_pose_residual_loss_weight * temporal_pose_residual
            + self.contact_loss_weight * contact
            + self.stable_grasp_loss_weight * stable_grasp
            + self.slip_speed_loss_weight * slip_speed
            + self.depth_validity_loss_weight * depth_validity
            + self.failure_loss_weight * failure
        )
        return total, {
            "loss/student_imitation": imitation,
            "loss/student_value": value,
            "loss/student_auxiliary": auxiliary,
            "loss/student_relative_pose": relative_pose,
            "loss/student_pose_consistency": pose_consistency,
            "loss/student_cross_camera_contrastive": cross_camera_contrastive,
            "loss/student_temporal_pose_residual": temporal_pose_residual,
            "loss/student_contact": contact,
            "loss/student_stable_grasp": stable_grasp,
            "loss/student_slip_speed": slip_speed,
            "loss/student_depth_validity": depth_validity,
            "loss/student_failure": failure,
            "loss/student_total": total,
        }, next_hidden

@dataclass(frozen=True)
class G2RecurrentVisualSACConfig(G2VisualSACConfig):
    """SAC settings for the shared keyboard/teacher recurrent backbone."""

    hidden_dim: int = 256
    gru_num_layers: int = 1
    sequence_length: int = 16
    burn_in_steps: int = 4
    sequence_stride: int | None = None
    gradient_clip: float = 5.0
    random_shift_pad: int = 4
    demonstration_q_filter_temperature: float = 0.1
    student_head_distillation_weight: float = 0.1
    near_contact_auxiliary_distance_m: float = 0.10
    near_contact_relative_pose_multiplier: float = 4.0
    cross_camera_contrastive_weight: float = 0.02
    cross_camera_contrastive_temperature: float = 0.1
    cross_camera_false_negative_position_threshold_m: float = 0.01
    temporal_pose_residual_weight: float = 0.05
    temporal_pose_residual_rotation_weight: float = 0.1

    @property
    def resolved_sequence_stride(self) -> int:
        """Use sequence-minus-burn-in for the selected production or test contract."""

        return (
            self.sequence_length - self.burn_in_steps
            if self.sequence_stride is None
            else int(self.sequence_stride)
        )


class _RecurrentVisualActor(nn.Module):
    """Stochastic SAC head over the exact deployable recurrent model."""

    def __init__(self, config: G2RecurrentVisualSACConfig) -> None:
        super().__init__()
        self.policy = G2RecurrentVisualStudent(
            hidden_dim=config.hidden_dim,
            gru_num_layers=config.gru_num_layers,
            pose_consistency_loss_weight=config.pose_consistency_weight,
            pose_consistency_rotation_weight=config.pose_consistency_rotation_weight,
            cross_camera_contrastive_loss_weight=(
                config.cross_camera_contrastive_weight
            ),
            cross_camera_contrastive_temperature=(
                config.cross_camera_contrastive_temperature
            ),
            cross_camera_false_negative_position_threshold_m=(
                config.cross_camera_false_negative_position_threshold_m
            ),
            temporal_pose_residual_loss_weight=(
                config.temporal_pose_residual_weight
            ),
            temporal_pose_residual_rotation_weight=(
                config.temporal_pose_residual_rotation_weight
            ),
            share_camera_encoder_weights=config.share_camera_encoder_weights,
            camera_encoder_channels=config.camera_encoder_channels,
            torso_control_enabled=False,
        )
        self.mean = nn.Linear(config.hidden_dim, G2_VISUAL_ARM_ACTION_DIM)
        self.log_std = nn.Linear(config.hidden_dim, G2_VISUAL_ARM_ACTION_DIM)
        self.gripper_logit = nn.Linear(config.hidden_dim, 1)
        nn.init.zeros_(self.mean.bias)
        nn.init.constant_(self.log_std.bias, -2.0)
        nn.init.zeros_(self.gripper_logit.bias)

    def initial_hidden(self, batch: int, *, device=None) -> torch.Tensor:
        return self.policy.initial_hidden(batch, device=device)

    def recurrent_features(
        self,
        rgbd_u8: torch.Tensor,
        proprioception: torch.Tensor,
        hidden: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(rgbd_u8.shape[0])
        if reset_mask is None:
            reset_mask = torch.zeros(batch, dtype=torch.bool, device=rgbd_u8.device)
        features, next_hidden = self.policy.encode_recurrent_features(
            rgbd_u8.unsqueeze(1),
            proprioception.unsqueeze(1),
            hidden,
            reset_mask.to(torch.bool).unsqueeze(1),
            torch.ones((batch, 1), dtype=torch.bool, device=rgbd_u8.device),
        )
        return features[:, 0], next_hidden

    def sample_from_features(
        self, features: torch.Tensor, *, deterministic: bool = False
    ):
        mean = self.mean(features)
        log_std = -5.0 + 3.5 * (torch.tanh(self.log_std(features)) + 1.0)
        pre_tanh = (
            mean
            if deterministic
            else torch.distributions.Normal(mean, log_std.exp()).rsample()
        )
        arm = torch.tanh(pre_tanh)
        arm_log_probability = squashed_gaussian_log_prob(
            pre_tanh, mean, log_std
        )
        gripper_logit = self.gripper_logit(features)
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
        log_probability = arm_log_probability + torch.log(
            selected_probability.clamp_min(1.0e-8)
        )
        return (
            torch.cat((arm, gripper), dim=-1),
            log_probability,
            torch.tanh(mean),
            log_std,
            gripper_logit,
        )


class G2RecurrentVisualAsymmetricSAC:
    """RGB-D encoder + proprioception fusion + GRU asymmetric SAC.

    Replay samples contiguous episode sequences.  The stored rollout state is
    only the starting point: burn-in observations refresh it with the current
    encoder/GRU before losses are applied, avoiding a single-transition GRU
    update with a stale history summary.  Keyboard BC and this online teacher
    share :class:`G2RecurrentVisualStudent` exactly.
    """

    def __init__(
        self,
        config: G2RecurrentVisualSACConfig | None = None,
        *,
        device="cpu",
    ) -> None:
        self.config = config or G2RecurrentVisualSACConfig()
        if self.config.sequence_length <= 0:
            raise ValueError("recurrent sequence length must be positive")
        if self.config.gru_num_layers != 1:
            raise ValueError("recurrent SAC requires exactly one GRU layer")
        if not 0 <= self.config.burn_in_steps < self.config.sequence_length:
            raise ValueError("recurrent burn-in must be inside the sequence")
        if not 0 < self.config.resolved_sequence_stride <= self.config.sequence_length:
            raise ValueError("recurrent sequence stride must be in [1, sequence_length]")
        if self.config.random_shift_pad < 0:
            raise ValueError("random-shift padding cannot be negative")
        if self.config.pose_consistency_weight < 0.0:
            raise ValueError("pose-consistency weight cannot be negative")
        if self.config.pose_consistency_rotation_weight < 0.0:
            raise ValueError("pose-consistency rotation weight cannot be negative")
        if self.config.demonstration_q_filter_temperature <= 0.0:
            raise ValueError("demonstration Q-filter temperature must be positive")
        if self.config.student_head_distillation_weight < 0.0:
            raise ValueError("student head distillation weight cannot be negative")
        if self.config.near_contact_auxiliary_distance_m <= 0.0:
            raise ValueError("near-contact auxiliary distance must be positive")
        if self.config.near_contact_relative_pose_multiplier < 1.0:
            raise ValueError("near-contact relative-pose multiplier must be at least one")
        if self.config.cross_camera_contrastive_weight < 0.0:
            raise ValueError("cross-camera contrastive weight cannot be negative")
        if self.config.cross_camera_contrastive_temperature <= 0.0:
            raise ValueError("cross-camera contrastive temperature must be positive")
        if self.config.cross_camera_false_negative_position_threshold_m < 0.0:
            raise ValueError("cross-camera false-negative threshold cannot be negative")
        if self.config.temporal_pose_residual_weight < 0.0:
            raise ValueError("temporal pose-residual weight cannot be negative")
        if self.config.temporal_pose_residual_rotation_weight < 0.0:
            raise ValueError("temporal pose-residual rotation weight cannot be negative")
        self.device = torch.device(device)
        self.actor = _RecurrentVisualActor(self.config).to(self.device)
        self.q1, self.q2 = _Critic().to(self.device), _Critic().to(self.device)
        self.tq1, self.tq2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.tq1.requires_grad_(False)
        self.tq2.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.config.learning_rate,
        )
        self.log_alpha = torch.tensor(
            math.log(self.config.initial_alpha),
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.config.learning_rate
        )
        self.update_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def initial_hidden(self, batch: int) -> torch.Tensor:
        return self.actor.initial_hidden(batch, device=self.device)

    @torch.no_grad()
    def select_actions(
        self,
        rgbd_u8,
        proprioception,
        hidden: torch.Tensor,
        *,
        reset_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, torch.Tensor]:
        rgbd = torch.as_tensor(rgbd_u8, dtype=torch.uint8, device=self.device)
        proprio = torch.as_tensor(
            proprioception, dtype=torch.float32, device=self.device
        )
        features, next_hidden = self.actor.recurrent_features(
            rgbd, proprio, hidden.to(self.device), reset_mask
        )
        action, _, mean, _, _ = self.actor.sample_from_features(
            features, deterministic=deterministic
        )
        if deterministic:
            probability_open = torch.sigmoid(self.actor.gripper_logit(features))
            mean = torch.cat(
                (
                    mean,
                    torch.where(
                        probability_open >= 0.5,
                        torch.ones_like(probability_open),
                        -torch.ones_like(probability_open),
                    ),
                ),
                dim=-1,
            )
            action = mean
        return action.cpu().numpy(), next_hidden.detach()

    def behavior_clone(self, batch: Mapping[str, torch.Tensor | int]) -> dict[str, float]:
        """Warm-start the online Teacher actor from canonical keyboard data.

        This updates only the deployable RGB-D/proprioception/GRU actor path.
        Critics, target critics, alpha and online replay remain fresh.  The
        caller must supply episode-safe sequence windows produced by
        :class:`G2StudentSequenceContract`.
        """

        def value(name: str, dtype=None) -> torch.Tensor:
            item = batch.get(name)
            if not isinstance(item, torch.Tensor):
                raise ValueError(f"teacher BC batch is missing tensor {name}")
            return item.to(device=self.device, dtype=dtype)

        original_rgbd = value("rgbd_u8", torch.uint8)
        rgbd, principal_point_shift_px = random_shift_rgbd_sequences(
            original_rgbd,
            pad=self.config.random_shift_pad,
            return_principal_point_shift=True,
        )
        proprio = value("deployable_proprioception", torch.float32)
        target = value("expert_action_target", torch.float32)
        confidence = value("expert_confidence", torch.float32)
        padding = value("padding_mask", torch.bool)
        reset = value("hidden_reset_mask", torch.bool)
        episode_id = value("episode_id", torch.long)
        sequence_step = value("sequence_step", torch.long)
        lengths = value("sequence_lengths", torch.long)
        G2RecurrentVisualStudent.validate_sequence_boundaries(
            episode_id, sequence_step, padding, lengths
        )
        if target.shape != (*padding.shape, G2_VISUAL_ACTION_DIM):
            raise ValueError("teacher BC action target must be [B,T,7]")
        if confidence.shape != (*padding.shape, 1):
            raise ValueError("teacher BC confidence must be [B,T,1]")
        burn_in = int(batch.get("burn_in_steps", self.config.burn_in_steps))
        if not 0 <= burn_in < padding.shape[1]:
            raise ValueError("teacher BC burn-in must be in [0,T)")

        features, _, projected_camera_tokens = self.actor.policy.encode_recurrent_features(
            rgbd, proprio, self.actor.initial_hidden(rgbd.shape[0], device=self.device),
            reset, padding, principal_point_shift_px,
            return_camera_tokens=True,
        )
        original_features, _ = self.actor.policy.encode_recurrent_features(
            original_rgbd,
            proprio,
            self.actor.initial_hidden(rgbd.shape[0], device=self.device),
            reset,
            padding,
        )
        mask = padding.unsqueeze(-1).to(torch.float32)
        mask[:, :burn_in] = 0.0
        weight = mask * confidence.clamp(0.0, 1.0)
        denominator = weight.sum().clamp_min(1.0)
        arm_prediction = torch.tanh(self.actor.mean(features))
        arm_loss = (
            weight
            * (arm_prediction - target[..., :G2_VISUAL_ARM_ACTION_DIM])
            .square()
            .mean(dim=-1, keepdim=True)
        ).sum() / denominator
        gripper_target_open = (target[..., 6:7] > 0.0).to(torch.float32)
        gripper_loss = (
            weight
            * F.binary_cross_entropy_with_logits(
                self.actor.gripper_logit(features),
                gripper_target_open,
                reduction="none",
            )
        ).sum() / denominator

        relative_target = value("relative_pose_target", torch.float32)
        relative_prediction = self.actor.policy.relative_pose_head(features)
        relative_error = relative_pose_auxiliary_loss(
            relative_prediction,
            relative_target,
            reduction="none",
        ).unsqueeze(-1)
        relative_loss = (mask * relative_error).sum() / mask.sum().clamp_min(1.0)
        consistency_error = relative_pose_consistency_loss(
            self.actor.policy.relative_pose_head(original_features),
            self.actor.policy.relative_pose_head(features),
            rotation_weight=self.config.pose_consistency_rotation_weight,
            reduction="none",
        ).unsqueeze(-1)
        consistency_loss = (
            mask * consistency_error
        ).sum() / mask.sum().clamp_min(1.0)
        contact_target = value("contact_target", torch.float32)
        contact_error = F.binary_cross_entropy_with_logits(
            self.actor.policy.contact_head(features), contact_target, reduction="none"
        ).mean(dim=-1, keepdim=True)
        contact_loss = (mask * contact_error).sum() / mask.sum().clamp_min(1.0)
        depth_target = value("depth_validity_target", torch.float32)
        depth_error = F.binary_cross_entropy_with_logits(
            self.actor.policy.depth_validity_head(features),
            depth_target,
            reduction="none",
        ).mean(dim=-1, keepdim=True)
        depth_loss = (mask * depth_error).sum() / mask.sum().clamp_min(1.0)
        contrastive_valid = mask[..., 0].to(torch.bool) & (
            depth_target > 0.0
        ).all(dim=-1)
        cross_camera_contrastive = cross_camera_contrastive_loss(
            projected_camera_tokens,
            contrastive_valid,
            relative_pose_target=relative_target,
            temperature=self.config.cross_camera_contrastive_temperature,
            false_negative_position_threshold_m=(
                self.config.cross_camera_false_negative_position_threshold_m
            ),
        )
        temporal_pose_residual = temporal_pose_residual_loss(
            relative_prediction,
            relative_target,
            mask[..., 0].to(torch.bool),
            rotation_weight=self.config.temporal_pose_residual_rotation_weight,
        )
        loss = (
            arm_loss
            + gripper_loss
            + self.config.relative_pose_weight * relative_loss
            + self.config.pose_consistency_weight * consistency_loss
            + self.config.cross_camera_contrastive_weight
            * cross_camera_contrastive
            + self.config.temporal_pose_residual_weight * temporal_pose_residual
            + self.config.contact_weight * contact_loss
            + self.config.depth_validity_weight * depth_loss
        )
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.gradient_clip
        )
        self.actor_optimizer.step()
        return {
            "demonstration_bc/loss_total": float(loss.detach()),
            "demonstration_bc/loss_arm": float(arm_loss.detach()),
            "demonstration_bc/loss_gripper": float(gripper_loss.detach()),
            "demonstration_bc/loss_relative_pose": float(relative_loss.detach()),
            "demonstration_bc/loss_pose_consistency": float(consistency_loss.detach()),
            "demonstration_bc/loss_cross_camera_contrastive": float(
                cross_camera_contrastive.detach()
            ),
            "demonstration_bc/loss_temporal_pose_residual": float(
                temporal_pose_residual.detach()
            ),
            "demonstration_bc/loss_contact": float(contact_loss.detach()),
            "demonstration_bc/loss_depth_validity": float(depth_loss.detach()),
            "demonstration_bc/gradient_norm": float(gradient_norm),
            "demonstration_bc/supervised_rows": float(weight.sum().detach()),
        }

    @staticmethod
    def _hybrid_expectation(
        arm_action,
        arm_log_probability,
        gripper_logit,
        q1,
        q2,
        state,
    ):
        return G2VisualAsymmetricSAC._hybrid_expectation(
            arm_action,
            arm_log_probability,
            gripper_logit,
            q1,
            q2,
            state,
        )

    def _demonstration_regularization_loss(
        self,
        batch: Mapping[str, torch.Tensor | int],
        *,
        q_filter_reliability: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute actor-only, soft-Q-filtered BC on canonical sequences.

        The privileged Teacher state is consumed only by the frozen critics
        that calculate the Q filter.  Actor features use exactly the same
        RGB-D, 45-D deployable proprioception and GRU path as the Student.
        """

        if not 0.0 <= q_filter_reliability <= 1.0:
            raise ValueError("Q-filter reliability must be in [0,1]")

        def value(name: str, dtype=None) -> torch.Tensor:
            item = batch.get(name)
            if not isinstance(item, torch.Tensor):
                raise ValueError(f"online BC batch is missing tensor {name}")
            return item.to(device=self.device, dtype=dtype)

        rgbd, principal_point_shift_px = random_shift_rgbd_sequences(
            value("rgbd_u8", torch.uint8),
            pad=self.config.random_shift_pad,
            return_principal_point_shift=True,
        )
        proprio = value("deployable_proprioception", torch.float32)
        expert_action = value("expert_action_target", torch.float32)
        confidence = value("expert_confidence", torch.float32)
        teacher_state = value("teacher_state_target", torch.float32)
        padding = value("padding_mask", torch.bool)
        reset = value("hidden_reset_mask", torch.bool)
        episode_id = value("episode_id", torch.long)
        sequence_step = value("sequence_step", torch.long)
        lengths = value("sequence_lengths", torch.long)
        G2RecurrentVisualStudent.validate_sequence_boundaries(
            episode_id, sequence_step, padding, lengths
        )
        if teacher_state.shape != (*padding.shape, G2_VISUAL_PRIVILEGED_DIM):
            raise ValueError("online BC teacher state must be [B,T,59]")
        if expert_action.shape != (*padding.shape, G2_VISUAL_ACTION_DIM):
            raise ValueError("online BC expert action must be [B,T,7]")
        burn_in = int(batch.get("burn_in_steps", self.config.burn_in_steps))
        if not 0 <= burn_in < padding.shape[1]:
            raise ValueError("online BC burn-in must be in [0,T)")

        features, _ = self.actor.policy.encode_recurrent_features(
            rgbd,
            proprio,
            self.actor.initial_hidden(rgbd.shape[0], device=self.device),
            reset,
            padding,
            principal_point_shift_px,
        )
        learning = padding.clone()
        learning[:, :burn_in] = False
        flat_learning = learning.reshape(-1)
        flat_features = features.reshape(-1, self.config.hidden_dim)[flat_learning]
        flat_state = teacher_state.reshape(-1, G2_VISUAL_PRIVILEGED_DIM)[
            flat_learning
        ]
        flat_expert = expert_action.reshape(-1, G2_VISUAL_ACTION_DIM)[flat_learning]
        flat_confidence = confidence.reshape(-1, 1)[flat_learning].clamp(0.0, 1.0)
        if flat_features.shape[0] == 0:
            raise ValueError("online BC batch has no post-burn-in rows")

        arm_prediction = torch.tanh(self.actor.mean(flat_features))
        gripper_logit = self.actor.gripper_logit(flat_features)
        gripper_prediction = torch.where(
            gripper_logit >= 0.0,
            torch.ones_like(gripper_logit),
            -torch.ones_like(gripper_logit),
        )
        policy_action = torch.cat((arm_prediction, gripper_prediction), dim=-1)
        # Q is an eligibility/strength filter only.  Detaching prevents the
        # demonstration path from updating critics or exploiting Q gradients.
        with torch.no_grad():
            expert_q = torch.minimum(
                self.q1(flat_state, flat_expert), self.q2(flat_state, flat_expert)
            )
            policy_q = torch.minimum(
                self.q1(flat_state, policy_action.detach()),
                self.q2(flat_state, policy_action.detach()),
            )
            advantage = expert_q - policy_q
            raw_q_weight = torch.sigmoid(
                advantage / self.config.demonstration_q_filter_temperature
            )
            # Online-only critics are not allowed to veto expert labels until
            # held-out policy rollouts have demonstrated real task mastery.
            q_weight = (
                (1.0 - float(q_filter_reliability))
                + float(q_filter_reliability) * raw_q_weight
            )
        weight = flat_confidence * q_weight
        denominator = weight.sum().clamp_min(1.0)
        arm_loss = (
            weight
            * (arm_prediction - flat_expert[:, :G2_VISUAL_ARM_ACTION_DIM])
            .square()
            .mean(dim=-1, keepdim=True)
        ).sum() / denominator
        gripper_target_open = (flat_expert[:, 6:7] > 0.0).to(torch.float32)
        gripper_loss = (
            weight
            * F.binary_cross_entropy_with_logits(
                gripper_logit, gripper_target_open, reduction="none"
            )
        ).sum() / denominator
        # The stochastic Teacher uses distribution heads around the shared
        # recurrent backbone, whereas deployment calls the deterministic
        # Student heads on that same backbone.  Train both against the exact
        # canonical 7-D label so a Teacher checkpoint contains a usable
        # Student action surface instead of only compatible feature weights.
        student_arm = self.actor.policy.arm_action_head(flat_features)
        student_gripper = self.actor.policy.gripper_head(flat_features)
        student_arm_loss = (
            weight
            * (student_arm - flat_expert[:, :G2_VISUAL_ARM_ACTION_DIM])
            .square()
            .mean(dim=-1, keepdim=True)
        ).sum() / denominator
        student_gripper_loss = (
            weight * (student_gripper - flat_expert[:, 6:7]).square()
        ).sum() / denominator
        student_compatibility_loss = student_arm_loss + student_gripper_loss
        loss = (
            arm_loss
            + gripper_loss
            + self.config.student_head_distillation_weight
            * student_compatibility_loss
        )
        return loss, {
            "demonstration_online_bc/loss": float(loss.detach()),
            "demonstration_online_bc/loss_arm": float(arm_loss.detach()),
            "demonstration_online_bc/loss_gripper": float(gripper_loss.detach()),
            "demonstration_online_bc/loss_student_compatibility": float(
                student_compatibility_loss.detach()
            ),
            "demonstration_online_bc/q_advantage_mean": float(advantage.mean()),
            "demonstration_online_bc/q_filter_weight_mean": float(q_weight.mean()),
            "demonstration_online_bc/q_filter_raw_weight_mean": float(
                raw_q_weight.mean()
            ),
            "demonstration_online_bc/q_filter_reliability": float(
                q_filter_reliability
            ),
            "demonstration_online_bc/q_filter_expert_better_fraction": float(
                (advantage > 0.0).float().mean()
            ),
            "demonstration_online_bc/supervised_rows": float(
                (flat_confidence > 0.0).sum()
            ),
        }

    def update(
        self,
        batch: Mapping[str, np.ndarray],
        *,
        demonstration_batch: Mapping[str, torch.Tensor | int] | None = None,
        demonstration_bc_coefficient: float = 0.0,
        demonstration_q_filter_reliability: float = 1.0,
    ) -> dict[str, float]:
        if demonstration_bc_coefficient < 0.0:
            raise ValueError("demonstration BC coefficient cannot be negative")
        if demonstration_bc_coefficient > 0.0 and demonstration_batch is None:
            raise ValueError("positive demonstration BC coefficient requires a batch")
        tensor = lambda name, dtype=torch.float32: torch.as_tensor(
            batch[name], dtype=dtype, device=self.device
        )
        rgbd, next_rgbd = tensor("rgbd", torch.uint8), tensor(
            "next_rgbd", torch.uint8
        )
        proprio, next_proprio = tensor("proprio"), tensor("next_proprio")
        state, next_state = tensor("privileged"), tensor("next_privileged")
        action, reward, terminated = (
            tensor("actions"),
            tensor("rewards"),
            tensor("terminated"),
        )
        # Preserve the legacy one-transition update for small unit diagnostics,
        # but production replay supplies [B,T,...] sequences.
        if rgbd.ndim == 5:
            rgbd, next_rgbd = rgbd.unsqueeze(1), next_rgbd.unsqueeze(1)
            proprio, next_proprio = proprio.unsqueeze(1), next_proprio.unsqueeze(1)
            state, next_state = state.unsqueeze(1), next_state.unsqueeze(1)
            action, reward, terminated = (
                action.unsqueeze(1), reward.unsqueeze(1), terminated.unsqueeze(1)
            )
        if rgbd.ndim != 6 or rgbd.shape != next_rgbd.shape:
            raise ValueError("recurrent SAC RGB-D replay must be [B,T,2,6,48,64]")
        batch_size, sequence_steps = rgbd.shape[:2]
        if proprio.shape[:2] != (batch_size, sequence_steps):
            raise ValueError("recurrent SAC proprioception sequence shape mismatch")
        burn_in = min(
            self.config.burn_in_steps if sequence_steps > 1 else 0,
            sequence_steps - 1,
        )
        hidden_value = tensor("recurrent_hidden")
        # Sequence replay returns only the state before its first observation;
        # legacy replay returns one state per sampled transition.
        if hidden_value.ndim == 3:
            hidden_value = hidden_value[:, 0]
        hidden = hidden_value.unsqueeze(0)
        if tuple(hidden.shape) != (1, batch_size, self.config.hidden_dim):
            raise ValueError("recurrent replay initial hidden-state shape mismatch")

        if "episode_id" in batch and "sequence_step" in batch:
            episode_id = tensor("episode_id", torch.long)
            sequence_step = tensor("sequence_step", torch.long)
            G2RecurrentVisualStudent.validate_sequence_boundaries(
                episode_id, sequence_step
            )

        # Build one recurrent chain: [s_0, s_1, ..., s_T].  Contiguous replay
        # guarantees next(s_t)==s_(t+1), while the final next state supplies
        # the bootstrap observation.  This gives the target action the exact
        # history implied by the sampled sequence.
        original_observation_chain = torch.cat((rgbd[:, :1], next_rgbd), dim=1)
        proprio_chain = torch.cat((proprio[:, :1], next_proprio), dim=1)
        observation_chain, principal_point_shift_px = random_shift_rgbd_sequences(
            original_observation_chain,
            pad=self.config.random_shift_pad,
            return_principal_point_shift=True,
        )
        original_hidden = hidden.detach().clone()
        if burn_in:
            with torch.no_grad():
                _, hidden = self.actor.policy.encode_recurrent_features(
                    observation_chain[:, :burn_in],
                    proprio_chain[:, :burn_in],
                    hidden,
                    torch.zeros(
                        (batch_size, burn_in), dtype=torch.bool, device=self.device
                    ),
                    torch.ones(
                        (batch_size, burn_in), dtype=torch.bool, device=self.device
                    ),
                    principal_point_shift_px[:, :burn_in],
                )
            hidden = hidden.detach()
        learning_rgbd = observation_chain[:, burn_in:]
        learning_proprio = proprio_chain[:, burn_in:]
        all_features, _, projected_camera_tokens = self.actor.policy.encode_recurrent_features(
            learning_rgbd,
            learning_proprio,
            hidden,
            torch.zeros(
                learning_rgbd.shape[:2], dtype=torch.bool, device=self.device
            ),
            torch.ones(
                learning_rgbd.shape[:2], dtype=torch.bool, device=self.device
            ),
            principal_point_shift_px[:, burn_in:],
            return_camera_tokens=True,
        )
        learning_features = all_features[:, :-1]
        features = learning_features.reshape(-1, self.config.hidden_dim)
        next_features = all_features[:, 1:].reshape(-1, self.config.hidden_dim)
        if burn_in:
            with torch.no_grad():
                _, original_hidden = self.actor.policy.encode_recurrent_features(
                    original_observation_chain[:, :burn_in],
                    proprio_chain[:, :burn_in],
                    original_hidden,
                    torch.zeros(
                        (batch_size, burn_in), dtype=torch.bool, device=self.device
                    ),
                    torch.ones(
                        (batch_size, burn_in), dtype=torch.bool, device=self.device
                    ),
                )
            original_hidden = original_hidden.detach()
        original_learning_rgbd = original_observation_chain[:, burn_in:]
        original_all_features, _ = self.actor.policy.encode_recurrent_features(
            original_learning_rgbd,
            proprio_chain[:, burn_in:],
            original_hidden,
            torch.zeros(
                original_learning_rgbd.shape[:2], dtype=torch.bool, device=self.device
            ),
            torch.ones(
                original_learning_rgbd.shape[:2], dtype=torch.bool, device=self.device
            ),
        )
        original_features = original_all_features[:, :-1].reshape(
            -1, self.config.hidden_dim
        )
        learning_state = state[:, burn_in:]
        relative_target_sequence = relative_pose_target_from_teacher_state(
            learning_state
        )
        relative_prediction_sequence = self.actor.policy.relative_pose_head(
            learning_features
        )
        temporal_valid = torch.ones(
            relative_target_sequence.shape[:2], dtype=torch.bool, device=self.device
        )
        if terminated.shape[1] > burn_in + 1:
            temporal_valid[:, 1:] &= ~terminated[:, burn_in:-1, 0].to(torch.bool)
        temporal_pose_residual = temporal_pose_residual_loss(
            relative_prediction_sequence,
            relative_target_sequence,
            temporal_valid,
            rotation_weight=self.config.temporal_pose_residual_rotation_weight,
        )
        contrastive_rgbd = learning_rgbd[:, :-1]
        contrastive_valid = (
            contrastive_rgbd[:, :, :, 5].float().mean(dim=(-1, -2)) > 0.0
        ).all(dim=-1)
        cross_camera_contrastive = cross_camera_contrastive_loss(
            projected_camera_tokens[:, :-1],
            contrastive_valid,
            relative_pose_target=relative_target_sequence,
            temperature=self.config.cross_camera_contrastive_temperature,
            false_negative_position_threshold_m=(
                self.config.cross_camera_false_negative_position_threshold_m
            ),
        )
        state = learning_state.reshape(-1, G2_VISUAL_PRIVILEGED_DIM)
        next_state = next_state[:, burn_in:].reshape(-1, G2_VISUAL_PRIVILEGED_DIM)
        action = action[:, burn_in:].reshape(-1, G2_VISUAL_ACTION_DIM)
        reward = reward[:, burn_in:].reshape(-1, 1)
        terminated = terminated[:, burn_in:].reshape(-1, 1)
        auxiliary_rgbd = learning_rgbd[:, :-1].reshape(
            -1, *G2_VISUAL_CAMERA_SHAPE
        )

        with torch.no_grad():
            next_features_target = next_features.detach()
            next_mean = self.actor.mean(next_features_target)
            next_log_std = -5.0 + 3.5 * (
                torch.tanh(self.actor.log_std(next_features_target)) + 1.0
            )
            next_pre_tanh = torch.distributions.Normal(
                next_mean, next_log_std.exp()
            ).rsample()
            next_arm = torch.tanh(next_pre_tanh)
            next_arm_logp = squashed_gaussian_log_prob(
                next_pre_tanh, next_mean, next_log_std
            )
            next_gripper_logit = self.actor.gripper_logit(next_features_target)
            target_q, next_logp, _ = self._hybrid_expectation(
                next_arm,
                next_arm_logp,
                next_gripper_logit,
                self.tq1,
                self.tq2,
                next_state,
            )
            target = reward + self.config.gamma * (1.0 - terminated) * (
                target_q - self.alpha.detach() * next_logp
            )

        q1, q2 = self.q1(state, action), self.q2(state, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_norm = nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.config.gradient_clip,
        )
        self.critic_optimizer.step()

        mean = self.actor.mean(features)
        log_std = -5.0 + 3.5 * (torch.tanh(self.actor.log_std(features)) + 1.0)
        pre_tanh = torch.distributions.Normal(mean, log_std.exp()).rsample()
        arm = torch.tanh(pre_tanh)
        arm_logp = squashed_gaussian_log_prob(pre_tanh, mean, log_std)
        gripper_logit = self.actor.gripper_logit(features)
        for parameter in list(self.q1.parameters()) + list(self.q2.parameters()):
            parameter.requires_grad_(False)
        expected_q, logp, probability_open = self._hybrid_expectation(
            arm, arm_logp, gripper_logit, self.q1, self.q2, state
        )
        actor_loss = (self.alpha.detach() * logp - expected_q).mean()
        for parameter in list(self.q1.parameters()) + list(self.q2.parameters()):
            parameter.requires_grad_(True)

        policy = self.actor.policy
        relative_target = relative_target_sequence.reshape(
            -1, relative_target_sequence.shape[-1]
        )
        relative_prediction = relative_prediction_sequence.reshape(
            -1, relative_prediction_sequence.shape[-1]
        )
        original_relative_prediction = policy.relative_pose_head(original_features)
        relative_error = relative_pose_auxiliary_loss(
            relative_prediction, relative_target, reduction="none"
        )
        ee_cube_slice = G2TeacherObservationContract().slices[
            "end_effector_to_cube_m"
        ]
        near_contact = (
            torch.linalg.vector_norm(state[:, ee_cube_slice], dim=-1)
            <= self.config.near_contact_auxiliary_distance_m
        )
        relative_weight = torch.where(
            near_contact,
            torch.full_like(
                relative_error,
                self.config.near_contact_relative_pose_multiplier,
            ),
            torch.ones_like(relative_error),
        )
        relative_loss = (relative_error * relative_weight).sum() / relative_weight.sum()
        pose_consistency_loss = relative_pose_consistency_loss(
            original_relative_prediction,
            relative_prediction,
            rotation_weight=self.config.pose_consistency_rotation_weight,
        )
        contact_slice = G2TeacherObservationContract().slices[
            "bilateral_contact_features"
        ]
        contact_target = state[:, contact_slice].clamp(0.0, 1.0)
        contact_logits = policy.contact_head(features)
        contact_loss = F.binary_cross_entropy_with_logits(
            contact_logits, contact_target
        )
        depth_target = auxiliary_rgbd[:, :, 5].float().div(255.0).mean(dim=(-1, -2))
        depth_logits = policy.depth_validity_head(features)
        depth_loss = F.binary_cross_entropy_with_logits(depth_logits, depth_target)
        auxiliary_loss = (
            self.config.relative_pose_weight * relative_loss
            + self.config.pose_consistency_weight * pose_consistency_loss
            + self.config.cross_camera_contrastive_weight
            * cross_camera_contrastive
            + self.config.temporal_pose_residual_weight * temporal_pose_residual
            + self.config.contact_weight * contact_loss
            + self.config.depth_validity_weight * depth_loss
        )
        # Continuously distil the current deterministic Teacher action into
        # the deployment Student heads.  Features and Teacher targets are
        # detached so this term cannot distort SAC's shared encoder/GRU; it
        # updates only the Student action heads carried in actor.policy.
        detached_features = features.detach()
        teacher_action_target = torch.cat(
            (torch.tanh(mean.detach()), torch.tanh(gripper_logit.detach())), dim=-1
        )
        student_action_prediction = torch.cat(
            (
                policy.arm_action_head(detached_features),
                policy.gripper_head(detached_features),
            ),
            dim=-1,
        )
        student_head_distillation_loss = F.smooth_l1_loss(
            student_action_prediction, teacher_action_target
        )
        demonstration_loss = actor_loss.new_zeros(())
        demonstration_metrics: dict[str, float] = {
            "demonstration_online_bc/loss": 0.0,
            "demonstration_online_bc/coefficient": float(
                demonstration_bc_coefficient
            ),
        }
        if demonstration_batch is not None and demonstration_bc_coefficient > 0.0:
            demonstration_loss, computed = self._demonstration_regularization_loss(
                demonstration_batch,
                q_filter_reliability=demonstration_q_filter_reliability,
            )
            demonstration_metrics.update(computed)
        total_actor_loss = (
            actor_loss
            + auxiliary_loss
            + self.config.student_head_distillation_weight
            * student_head_distillation_loss
            + float(demonstration_bc_coefficient) * demonstration_loss
        )
        self.actor_optimizer.zero_grad(set_to_none=True)
        total_actor_loss.backward()
        actor_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.gradient_clip
        )
        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha
            * (logp.detach() + self.config.resolved_target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        polyak_update(self.q1, self.tq1, self.config.tau)
        polyak_update(self.q2, self.tq2, self.config.tau)
        self.update_count += 1
        q_data = torch.cat((q1.detach(), q2.detach()), dim=-1)
        td_error = torch.cat(
            (target.detach() - q1.detach(), target.detach() - q2.detach()),
            dim=-1,
        )
        metrics = {
            "loss/actor": float(actor_loss.detach()),
            "loss/actor_total": float(total_actor_loss.detach()),
            "loss/critic": float(critic_loss.detach()),
            "loss/visual_relative_pose": float(relative_loss.detach()),
            "loss/visual_pose_consistency": float(pose_consistency_loss.detach()),
            "loss/visual_cross_camera_contrastive": float(
                cross_camera_contrastive.detach()
            ),
            "loss/visual_temporal_pose_residual": float(
                temporal_pose_residual.detach()
            ),
            "loss/visual_contact": float(contact_loss.detach()),
            "loss/visual_depth_validity": float(depth_loss.detach()),
            "loss/teacher_student_action_head_distillation": float(
                student_head_distillation_loss.detach()
            ),
            "loss/alpha": float(alpha_loss.detach()),
            "entropy/alpha": float(self.alpha.detach()),
            "entropy/log_std_mean": float(log_std.detach().mean()),
            "entropy/gripper_open_probability_mean": float(
                probability_open.detach().mean()
            ),
            "gradient/actor_visual_gru_norm": float(actor_norm),
            "gradient/critic_norm": float(critic_norm),
            "q/data_min": float(q_data.min()),
            "q/data_mean": float(q_data.mean()),
            "q/data_max": float(q_data.max()),
            "q/policy_mean": float(expected_q.detach().mean()),
            "q/target_mean": float(target.detach().mean()),
            "td_error/max_abs": float(td_error.abs().max()),
            "replay/sequence_length": float(sequence_steps),
            "replay/burn_in_steps": float(burn_in),
            "replay/learning_steps": float(sequence_steps - burn_in),
            "replay/augmentation_random_shift_pad": float(
                self.config.random_shift_pad
            ),
            "visual/relative_position_error_m": float(
                torch.linalg.vector_norm(
                    relative_prediction.detach()[:, :3] - relative_target[:, :3],
                    dim=-1,
                ).mean()
            ),
            "visual/near_contact_fraction": float(near_contact.float().mean()),
            "visual/near_contact_relative_position_error_m": float(
                torch.linalg.vector_norm(
                    relative_prediction.detach()[:, :3] - relative_target[:, :3],
                    dim=-1,
                )[near_contact].mean()
                if bool(near_contact.any())
                else 0.0
            ),
        }
        metrics.update(demonstration_metrics)
        return metrics

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": "g2_recurrent_rgbd_sequence_burnin_asymmetric_sac_v4",
            "shared_policy_contract": G2RecurrentVisualPolicyContract(
                hidden_dim=self.config.hidden_dim,
                gru_num_layers=self.config.gru_num_layers,
                sequence_length=self.config.sequence_length,
                burn_in_steps=self.config.burn_in_steps,
                sequence_stride=self.config.resolved_sequence_stride,
            ).validated().serializable(),
            "config": self.config.__dict__,
            "actor": self.actor.state_dict(),
            "deployment_student": self.actor.policy.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "tq1": self.tq1.state_dict(),
            "tq2": self.tq2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "update_count": self.update_count,
        }


class G2ReverseCurriculum:
    """Evaluation-success-gated GRASP_READY→PRE_GRASP→REACH mixture.

    Training outcomes are tracked separately for diagnostics.  Promotion must
    be driven by deterministic evaluation episodes; using noisy replay-
    collection episodes creates an optimistic, policy-dependent environment
    shift and does not match the COHER evaluation-rollout contract.
    """

    def __init__(self, *, initial_probability=.60, minimum_probability=.20, decrement=.05, window=100):
        self.beta = float(initial_probability); self.minimum = float(minimum_probability)
        self.decrement = float(decrement); self.window = int(window)
        self.history = deque(maxlen=self.window)
        self.training_history = deque(maxlen=self.window)
        self.promotion_count = 0

    def sample_near(self, rng: np.random.Generator, count: int) -> np.ndarray:
        return rng.random(count) < self.beta

    def record_episode(self, *, contact: bool, stable: bool, lift: bool) -> bool:
        return self.record_episodes(
            contact=[contact], stable=[stable], lift=[lift]
        )

    def record_training_episodes(self, *, contact, stable, lift) -> None:
        for values in zip(contact, stable, lift, strict=True):
            self.training_history.append(tuple(bool(value) for value in values))

    def record_evaluation_episodes(self, *, contact, stable, lift) -> bool:
        return self.record_episodes(contact=contact, stable=stable, lift=lift)

    @staticmethod
    def _rates(history: deque) -> tuple[float, float, float]:
        if not history:
            return (0.0, 0.0, 0.0)
        values = np.asarray(history, dtype=np.float32).mean(0)
        return tuple(float(value) for value in values)

    @property
    def evaluation_rates(self) -> tuple[float, float, float]:
        return self._rates(self.history)

    @property
    def training_rates(self) -> tuple[float, float, float]:
        return self._rates(self.training_history)

    def record_episodes(self, *, contact, stable, lift) -> bool:
        """Record one vector completion batch and promote at most once.

        Without this batch boundary, thousands of simultaneous completions
        can repeatedly clear/refill a short deque and collapse beta through
        every curriculum level in one environment step.
        """

        values = list(zip(contact, stable, lift, strict=True))
        for contact_value, stable_value, lift_value in values:
            self.history.append(
                (bool(contact_value), bool(stable_value), bool(lift_value))
            )
        if len(self.history) < self.window:
            return False
        rates = np.asarray(self.history, dtype=np.float32).mean(0)
        if rates[0] >= .50 and rates[1] >= .30 and rates[2] >= .20 and self.beta > self.minimum:
            self.beta = max(self.minimum, self.beta - self.decrement)
            self.history.clear()
            self.promotion_count += 1
            return True
        return False


class G2EpisodeReferenceBehavior:
    """Latch reference-vs-policy authority for complete episodes.

    Resampling authority at every transition lets a privileged reference
    rescue arbitrary fragments of a policy rollout.  The resulting contact
    counts cannot attest that either controller completed a coherent episode.
    This scheduler samples only at reset boundaries and keeps held-out
    evaluation environments policy-only.
    """

    schema = "g2_episode_latched_reference_behavior_v1"

    def __init__(
        self,
        training_mask,
        *,
        seed: int,
    ) -> None:
        mask = np.asarray(training_mask, dtype=np.bool_)
        if mask.ndim != 1 or mask.size == 0 or not bool(mask.any()):
            raise ValueError("reference behavior needs a non-empty training mask")
        self.training_mask = mask.copy()
        self.reference_mask = mask.copy()
        self.rng = np.random.default_rng(int(seed))

    def current_mask(self) -> np.ndarray:
        return self.reference_mask.copy()

    def reset_episodes(
        self,
        env_ids,
        *,
        learning_started: bool,
        reference_probability: float,
        force_policy_mask=None,
    ) -> None:
        probability = float(reference_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("reference probability must be in [0,1]")
        ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        forced = (
            np.zeros(ids.shape, dtype=np.bool_)
            if force_policy_mask is None
            else np.asarray(force_policy_mask, dtype=np.bool_).reshape(-1)
        )
        if forced.shape != ids.shape:
            raise ValueError("force-policy mask must align with reset environment ids")
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= self.training_mask.size):
            raise IndexError("reference behavior environment index is out of range")
        for offset, env_id in enumerate(ids):
            if not bool(self.training_mask[env_id]):
                self.reference_mask[env_id] = False
            elif bool(forced[offset]) and learning_started:
                self.reference_mask[env_id] = False
            elif not learning_started:
                self.reference_mask[env_id] = True
            else:
                self.reference_mask[env_id] = bool(self.rng.random() < probability)

    def serializable(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "authority_sampling_boundary": "EPISODE_RESET_ONLY",
            "training_mask": self.training_mask.tolist(),
            "reference_mask": self.reference_mask.tolist(),
            "held_out_evaluation_policy_only": True,
        }


class G2VisualReplayBuffer:
    """RAM-bounded replay that keeps compact RGB-D as uint8."""

    def __init__(self, capacity: int, *, seed: int = 42, storage_dir=None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity, self.size, self.position = int(capacity), 0, 0
        if storage_dir is not None:
            from pathlib import Path
            root = Path(storage_dir); root.mkdir(parents=True, exist_ok=True)
            def allocate(name, shape, dtype):
                return np.memmap(root / f"{name}.mmap", mode="w+", shape=shape, dtype=dtype)
        else:
            allocate = lambda _name, shape, dtype: np.empty(shape, dtype=dtype)
        shape = (capacity, *G2_VISUAL_CAMERA_SHAPE)
        self.rgbd = allocate("rgbd", shape, np.uint8); self.next_rgbd = allocate("next_rgbd", shape, np.uint8)
        self.proprio = allocate("proprio", (capacity, G2_VISUAL_PROPRIO_DIM), np.float32)
        self.next_proprio = allocate("next_proprio", self.proprio.shape, np.float32)
        self.privileged = allocate("privileged", (capacity, G2_VISUAL_PRIVILEGED_DIM), np.float32)
        self.next_privileged = allocate("next_privileged", self.privileged.shape, np.float32)
        self.actions = allocate("actions", (capacity, G2_VISUAL_ACTION_DIM), np.float32)
        self.rewards = allocate("rewards", (capacity, 1), np.float32)
        self.terminated = allocate("terminated", (capacity, 1), np.float32)
        self.truncated = allocate("truncated", (capacity, 1), np.float32)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add_batch(self, **batch) -> None:
        count = len(batch["rewards"])
        required = {"rgbd", "next_rgbd", "proprio", "next_proprio", "privileged", "next_privileged", "actions", "rewards", "terminated", "truncated"}
        if set(batch) != required:
            raise ValueError(f"visual replay fields differ: {set(batch) ^ required}")
        if not 0 < count <= self.capacity:
            raise ValueError("visual replay batch must fit in replay capacity")
        actions = np.asarray(batch["actions"])
        if not np.all(np.isfinite(actions)):
            raise ValueError("visual replay actions contain non-finite values")
        if not np.all(np.isin(actions[:, G2_VISUAL_GRIPPER_ACTION_INDEX], (-1.0, 1.0))):
            raise ValueError("visual replay gripper actions must be exactly -1 or +1")
        indices = (np.arange(count, dtype=np.int64) + self.position) % self.capacity
        for name in required:
            source = np.asarray(batch[name])
            destination = getattr(self, name)
            if source.shape != (count, *destination.shape[1:]):
                raise ValueError(
                    f"visual replay {name} shape mismatch: {source.shape} != "
                    f"{(count, *destination.shape[1:])}"
                )
            destination[indices] = source
        self.position = (self.position + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if not 0 < batch_size <= self.size:
            raise ValueError("invalid batch size")
        indices = self.rng.integers(0, self.size, batch_size)
        return {name: getattr(self, name)[indices] for name in (
            "rgbd", "next_rgbd", "proprio", "next_proprio", "privileged",
            "next_privileged", "actions", "rewards", "terminated", "truncated",
        )}

    def flush(self) -> None:
        for name in ("rgbd", "next_rgbd", "proprio", "next_proprio", "privileged",
                     "next_privileged", "actions", "rewards", "terminated", "truncated"):
            value = getattr(self, name)
            if isinstance(value, np.memmap):
                value.flush()


class G2RecurrentVisualReplayBuffer(G2VisualReplayBuffer):
    """Episode-safe recurrent replay with stored start states and burn-in."""

    def __init__(
        self,
        capacity: int,
        *,
        hidden_dim: int = 128,
        seed: int = 42,
        storage_dir=None,
    ) -> None:
        super().__init__(capacity, seed=seed, storage_dir=storage_dir)
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError("recurrent replay hidden dimension must be positive")
        shape = (capacity, self.hidden_dim)
        if storage_dir is None:
            self.recurrent_hidden = np.empty(shape, dtype=np.float32)
            self.next_recurrent_hidden = np.empty(shape, dtype=np.float32)
            self.env_id = np.full((capacity,), -1, dtype=np.int64)
            self.episode_id = np.full((capacity,), -1, dtype=np.int64)
            self.sequence_step = np.full((capacity,), -1, dtype=np.int64)
        else:
            from pathlib import Path

            root = Path(storage_dir)
            self.recurrent_hidden = np.memmap(
                root / "recurrent_hidden.mmap",
                mode="w+",
                shape=shape,
                dtype=np.float32,
            )
            self.next_recurrent_hidden = np.memmap(
                root / "next_recurrent_hidden.mmap",
                mode="w+",
                shape=shape,
                dtype=np.float32,
            )
            self.env_id = np.memmap(
                root / "env_id.mmap", mode="w+", shape=(capacity,), dtype=np.int64
            )
            self.episode_id = np.memmap(
                root / "episode_id.mmap", mode="w+", shape=(capacity,), dtype=np.int64
            )
            self.sequence_step = np.memmap(
                root / "sequence_step.mmap", mode="w+", shape=(capacity,), dtype=np.int64
            )
            self.env_id[:] = self.episode_id[:] = self.sequence_step[:] = -1
        if storage_dir is None:
            self.phase_code = np.zeros((capacity,), dtype=np.int8)
            self.force_priority = np.ones((capacity,), dtype=np.float32)
            self.hindsight_relabel = np.zeros((capacity,), dtype=np.bool_)
        else:
            self.phase_code = np.memmap(
                root / "phase_code.mmap", mode="w+", shape=(capacity,), dtype=np.int8
            )
            self.force_priority = np.memmap(
                root / "force_priority.mmap", mode="w+", shape=(capacity,), dtype=np.float32
            )
            self.hindsight_relabel = np.memmap(
                root / "hindsight_relabel.mmap", mode="w+", shape=(capacity,), dtype=np.bool_
            )
            self.phase_code[:] = 0
            self.force_priority[:] = 1.0
            self.hindsight_relabel[:] = False
        self._sequence_index: dict[tuple[int, int, int], int] = {}
        self._phase_slots: tuple[set[int], ...] = tuple(
            set() for _ in G2_ONLINE_REPLAY_PHASES
        )

    def add_batch(self, **batch) -> None:
        count = len(batch["rewards"])
        phase_code = np.asarray(
            batch.pop("phase_code", np.zeros(count, dtype=np.int8)), dtype=np.int8
        )
        force_priority = np.asarray(
            batch.pop("force_priority", np.ones(count, dtype=np.float32)),
            dtype=np.float32,
        )
        hindsight_relabel = np.asarray(
            batch.pop("hindsight_relabel", np.zeros(count, dtype=np.bool_)),
            dtype=np.bool_,
        )
        recurrent = np.asarray(batch.pop("recurrent_hidden"), dtype=np.float32)
        next_recurrent = np.asarray(
            batch.pop("next_recurrent_hidden"), dtype=np.float32
        )
        env_id = np.asarray(
            batch.pop("env_id", np.arange(count, dtype=np.int64)), dtype=np.int64
        )
        episode_id = np.asarray(
            batch.pop("episode_id", np.zeros(count, dtype=np.int64)), dtype=np.int64
        )
        sequence_step = np.asarray(
            batch.pop("sequence_step", np.zeros(count, dtype=np.int64)), dtype=np.int64
        )
        expected = (count, self.hidden_dim)
        if recurrent.shape != expected or next_recurrent.shape != expected:
            raise ValueError(
                "recurrent replay hidden state must have shape " + str(expected)
            )
        if not np.isfinite(recurrent).all() or not np.isfinite(next_recurrent).all():
            raise ValueError("recurrent replay hidden state contains non-finite values")
        for name, value in (
            ("env_id", env_id),
            ("episode_id", episode_id),
            ("sequence_step", sequence_step),
        ):
            if value.shape != (count,) or np.any(value < 0):
                raise ValueError(f"recurrent replay {name} must be non-negative [N]")
        if phase_code.shape != (count,) or np.any(phase_code < 0) or np.any(
            phase_code >= len(G2_ONLINE_REPLAY_PHASES)
        ):
            raise ValueError("recurrent replay phase_code must be a valid [N] phase")
        if force_priority.shape != (count,) or not np.all(np.isfinite(force_priority)) or np.any(
            force_priority < 1.0
        ):
            raise ValueError("recurrent replay force_priority must be finite [N] >= 1")
        if hindsight_relabel.shape != (count,):
            raise ValueError("recurrent replay hindsight_relabel must be boolean [N]")
        indices = (np.arange(count, dtype=np.int64) + self.position) % self.capacity
        # The parent advances ``position``; write the recurrent fields at the
        # same pre-advance indices first.
        self.recurrent_hidden[indices] = recurrent
        self.next_recurrent_hidden[indices] = next_recurrent
        for index in indices:
            for slots in self._phase_slots:
                slots.discard(int(index))
            old_key = (
                int(self.env_id[index]),
                int(self.episode_id[index]),
                int(self.sequence_step[index]),
            )
            if old_key[0] >= 0 and self._sequence_index.get(old_key) == int(index):
                del self._sequence_index[old_key]
        self.env_id[indices] = env_id
        self.episode_id[indices] = episode_id
        self.sequence_step[indices] = sequence_step
        self.phase_code[indices] = phase_code
        self.force_priority[indices] = force_priority
        self.hindsight_relabel[indices] = hindsight_relabel
        for offset, index in enumerate(indices):
            key = (int(env_id[offset]), int(episode_id[offset]), int(sequence_step[offset]))
            if key in self._sequence_index:
                raise ValueError(f"duplicate recurrent replay sequence key: {key}")
            self._sequence_index[key] = int(index)
            self._phase_slots[int(phase_code[offset])].add(int(index))
        super().add_batch(**batch)

    def _sequence_for_end(self, end_index: int, length: int) -> list[int] | None:
        env = int(self.env_id[end_index])
        episode = int(self.episode_id[end_index])
        end_step = int(self.sequence_step[end_index])
        if env < 0 or episode < 0 or end_step < length - 1:
            return None
        indices = [
            self._sequence_index.get((env, episode, step))
            for step in range(end_step - length + 1, end_step + 1)
        ]
        if any(index is None for index in indices):
            return None
        resolved = [int(index) for index in indices]
        # A terminal transition may end a sampled sequence, never precede a
        # later transition in that same training sequence.
        if length > 1 and bool(np.asarray(self.terminated[resolved[:-1]]).any()):
            return None
        return resolved

    def has_sequences(self, length: int) -> bool:
        if length <= 0:
            raise ValueError("sequence length must be positive")
        return any(
            self._sequence_for_end(index, length) is not None
            for index in self._sequence_index.values()
        )

    def sample_sequences(self, batch_size: int, length: int) -> dict[str, np.ndarray]:
        if batch_size <= 0 or length <= 0:
            raise ValueError("sequence batch size and length must be positive")
        # Do not rebuild an O(replay_capacity) candidate list for every SAC
        # update.  Physical replay slots are sampled with replacement and are
        # accepted only when their env/episode keys form a complete sequence.
        # Once the first ``length`` vector steps have been collected, the
        # acceptance rate is high while memory and CPU cost stay O(batch).
        slot_count = self.capacity if self.size == self.capacity else self.size
        sequences: list[list[int]] = []
        attempts = 0
        maximum_attempts = max(batch_size * 128, min(slot_count * 2, 1_000_000))
        while len(sequences) < batch_size and attempts < maximum_attempts:
            end_index = int(self.rng.integers(0, slot_count))
            sequence = self._sequence_for_end(end_index, length)
            if sequence is not None:
                sequences.append(sequence)
            attempts += 1
        if len(sequences) < batch_size:
            # Sparse early buffers need a deterministic bounded fallback.  It
            # exits as soon as enough sequences are found and never retains a
            # capacity-sized Python object between updates.
            for end_index in self._sequence_index.values():
                sequence = self._sequence_for_end(end_index, length)
                if sequence is not None:
                    sequences.append(sequence)
                    if len(sequences) >= batch_size:
                        break
        if not sequences:
            raise ValueError("recurrent replay has no complete episode-safe sequence")
        while len(sequences) < batch_size:
            sequences.append(sequences[int(self.rng.integers(0, len(sequences)))])
        indices = np.asarray(sequences[:batch_size])
        sequence_names = (
            "rgbd", "next_rgbd", "proprio", "next_proprio", "privileged",
            "next_privileged", "actions", "rewards", "terminated", "truncated",
            "next_recurrent_hidden", "env_id", "episode_id", "sequence_step",
        )
        result = {name: np.asarray(getattr(self, name)[indices]) for name in sequence_names}
        result["recurrent_hidden"] = np.asarray(self.recurrent_hidden[indices[:, 0]])
        return result

    def sample_sequences_stratified(
        self,
        batch_size: int,
        length: int,
        *,
        phase_probabilities: tuple[float, float, float, float],
        force_priority_mixture: float = 0.25,
    ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        """Sample complete sequences by final physical phase.

        Contact/stable/lift endpoints may carry a bounded force-energy weight.
        A missing early-training stratum falls back to any complete sequence;
        realized rather than requested counts are returned for audit.
        """

        probability = np.asarray(phase_probabilities, dtype=np.float64)
        if (
            probability.shape != (len(G2_ONLINE_REPLAY_PHASES),)
            or np.any(probability < 0.0)
            or not np.isclose(probability.sum(), 1.0)
        ):
            raise ValueError(
                "online replay phase probabilities must be four non-negative values summing to one"
            )
        if batch_size <= 0 or length <= 0:
            raise ValueError("sequence batch size and length must be positive")
        if not 0.0 <= force_priority_mixture <= 1.0:
            raise ValueError("force priority mixture must be in [0,1]")
        requested = self.rng.choice(len(probability), size=batch_size, p=probability)
        sequences: list[list[int]] = []
        counts = {name: 0 for name in G2_ONLINE_REPLAY_PHASES}
        all_valid: list[list[int]] | None = None
        for requested_phase in requested:
            candidates = [
                index
                for index in self._phase_slots[int(requested_phase)]
                if self._sequence_for_end(index, length) is not None
            ]
            if candidates:
                force_weights = np.asarray(
                    [self.force_priority[index] for index in candidates],
                    dtype=np.float64,
                )
                force_probability = force_weights / force_weights.sum()
                uniform_probability = np.full(
                    len(candidates), 1.0 / len(candidates), dtype=np.float64
                )
                # Normal HER/phase replay remains the primary distribution.
                # Physical contact energy is a bounded local refinement, not
                # the sole authority inside CONTACT/STABLE/LIFT strata.
                weights = (
                    (1.0 - force_priority_mixture) * uniform_probability
                    + force_priority_mixture * force_probability
                )
                end_index = int(self.rng.choice(candidates, p=weights))
                sequence = self._sequence_for_end(end_index, length)
                assert sequence is not None
            else:
                if all_valid is None:
                    all_valid = [
                        sequence
                        for index in self._sequence_index.values()
                        if (sequence := self._sequence_for_end(index, length)) is not None
                    ]
                if not all_valid:
                    raise ValueError("recurrent replay has no complete episode-safe sequence")
                sequence = all_valid[int(self.rng.integers(0, len(all_valid)))]
            sequences.append(sequence)
            realized_phase = int(self.phase_code[sequence[-1]])
            counts[G2_ONLINE_REPLAY_PHASES[realized_phase]] += 1
        indices = np.asarray(sequences, dtype=np.int64)
        sequence_names = (
            "rgbd", "next_rgbd", "proprio", "next_proprio", "privileged",
            "next_privileged", "actions", "rewards", "terminated", "truncated",
            "next_recurrent_hidden", "env_id", "episode_id", "sequence_step",
            "phase_code", "force_priority", "hindsight_relabel",
        )
        result = {name: np.asarray(getattr(self, name)[indices]) for name in sequence_names}
        result["recurrent_hidden"] = np.asarray(self.recurrent_hidden[indices[:, 0]])
        return result, counts

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if not 0 < batch_size <= self.size:
            raise ValueError("invalid batch size")
        indices = self.rng.integers(0, self.size, batch_size)
        names = (
            "rgbd",
            "next_rgbd",
            "proprio",
            "next_proprio",
            "privileged",
            "next_privileged",
            "actions",
            "rewards",
            "terminated",
            "truncated",
            "recurrent_hidden",
            "next_recurrent_hidden",
        )
        return {name: getattr(self, name)[indices] for name in names}

    def flush(self) -> None:
        super().flush()
        for value in (
            self.recurrent_hidden,
            self.next_recurrent_hidden,
            self.env_id,
            self.episode_id,
            self.sequence_step,
            self.phase_code,
            self.force_priority,
            self.hindsight_relabel,
        ):
            if isinstance(value, np.memmap):
                value.flush()


__all__ = [name for name in globals() if name.startswith("G2_") or name in {"pack_rgbd"}]
