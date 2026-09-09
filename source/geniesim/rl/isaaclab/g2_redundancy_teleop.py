"""Seven-DoF G2 keyboard redundancy-control primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


G2_REDUNDANCY_ACTION_SCHEMA = "g2_se3_elbow_nullspace_gripper_v1"
ELBOW_NEGATIVE_KEYS = frozenset({"LEFT_BRACKET", "BRACKET_LEFT", "["})
ELBOW_POSITIVE_KEYS = frozenset({"RIGHT_BRACKET", "BRACKET_RIGHT", "]"})


@dataclass(frozen=True)
class G2NullspaceProjection:
    joint_delta: torch.Tensor
    task_leakage: torch.Tensor
    clipped: torch.Tensor


def damped_nullspace_project(
    jacobian: torch.Tensor,
    command: torch.Tensor,
    *,
    seed_joint_index: int = 2,
    damping: float = 0.05,
    maximum_joint_delta_rad: float = 0.002,
) -> G2NullspaceProjection:
    """Project a signed elbow request through a batched 6-by-7 Jacobian.

    Joint3 is only the elbow-plane seed. The projected output is a coordinated
    seven-joint delta and does not directly overwrite joint3 or joint4.
    """

    if jacobian.ndim != 3 or jacobian.shape[-2:] != (6, 7):
        raise ValueError(f"jacobian must have shape (N, 6, 7), got {tuple(jacobian.shape)}")
    if command.shape != (jacobian.shape[0],):
        raise ValueError(
            f"command must have shape ({jacobian.shape[0]},), got {tuple(command.shape)}"
        )
    if not 0 <= seed_joint_index < 7:
        raise ValueError("seed_joint_index must be in [0, 6]")
    if damping < 0.0 or maximum_joint_delta_rad <= 0.0:
        raise ValueError("damping must be non-negative and maximum delta must be positive")
    if not bool(torch.isfinite(jacobian).all() and torch.isfinite(command).all()):
        raise ValueError("jacobian and command must be finite")

    batch = jacobian.shape[0]
    identity6 = torch.eye(6, dtype=jacobian.dtype, device=jacobian.device).expand(batch, -1, -1)
    identity7 = torch.eye(7, dtype=jacobian.dtype, device=jacobian.device).expand(batch, -1, -1)
    pseudo_inverse = jacobian.transpose(-1, -2) @ torch.linalg.solve(
        jacobian @ jacobian.transpose(-1, -2) + damping**2 * identity6,
        identity6,
    )
    projector = identity7 - pseudo_inverse @ jacobian
    seed = torch.zeros((batch, 7), dtype=jacobian.dtype, device=jacobian.device)
    seed[:, seed_joint_index] = torch.clamp(command, -1.0, 1.0)
    unbounded = torch.bmm(projector, seed.unsqueeze(-1)).squeeze(-1)
    unbounded = unbounded * maximum_joint_delta_rad
    joint_delta = torch.clamp(unbounded, -maximum_joint_delta_rad, maximum_joint_delta_rad)
    task_leakage = torch.linalg.vector_norm(
        torch.bmm(jacobian, joint_delta.unsqueeze(-1)).squeeze(-1), dim=-1
    )
    clipped = torch.any(joint_delta != unbounded, dim=-1)
    return G2NullspaceProjection(joint_delta, task_leakage, clipped)


class G2ElbowKeyState:
    """Press/release state for the two elbow-swivel keys."""

    def __init__(self) -> None:
        self._negative = False
        self._positive = False

    @property
    def command(self) -> float:
        return float(self._positive) - float(self._negative)

    def reset(self) -> None:
        self._negative = False
        self._positive = False

    def handle(self, key_name: str, *, pressed: bool) -> bool:
        key = key_name.upper()
        if key in ELBOW_NEGATIVE_KEYS:
            self._negative = pressed
            return True
        if key in ELBOW_POSITIVE_KEYS:
            self._positive = pressed
            return True
        return False


def tensor_value(value: Any) -> torch.Tensor:
    """Return the tensor exposed by either Isaac Lab 2.x or 3.x APIs."""

    return value.torch if hasattr(value, "torch") else value


def ee_rotation_only_mask(
    cartesian_action: torch.Tensor, *, epsilon: float = 1.0e-8
) -> torch.Tensor:
    """Return rows that request EE rotation without Cartesian translation.

    The first three values are root-frame translation and the next three are
    the EE rotation vector.  Keeping this classification in the shared
    teleoperation contract makes terminal, teacher, and student action paths
    agree on what a rotation-only request means.
    """

    if cartesian_action.ndim != 2 or cartesian_action.shape[1] != 6:
        raise ValueError(
            "cartesian_action must have shape (N, 6), got "
            f"{tuple(cartesian_action.shape)}"
        )
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")
    translation_requested = torch.any(
        torch.abs(cartesian_action[:, :3]) > epsilon, dim=-1
    )
    rotation_requested = torch.any(
        torch.abs(cartesian_action[:, 3:6]) > epsilon, dim=-1
    )
    return rotation_requested & ~translation_requested


def wrist_only_rotation_jacobian(
    jacobian: torch.Tensor, *, wrist_start_joint_index: int = 4
) -> torch.Tensor:
    """Remove shoulder/elbow authority from pure EE-orientation IK.

    G2 right-arm joints 1..4 form the proximal chain and joints 5..7 form the
    wrist.  The position rows remain active, so wrist motion still minimizes
    translation of the EE rather than assuming ideal intersecting wrist axes.
    """

    if jacobian.ndim != 3 or jacobian.shape[-2:] != (6, 7):
        raise ValueError(
            f"jacobian must have shape (N, 6, 7), got {tuple(jacobian.shape)}"
        )
    if not 1 <= wrist_start_joint_index < 7:
        raise ValueError("wrist_start_joint_index must be in [1, 6]")
    if not bool(torch.isfinite(jacobian).all()):
        raise ValueError("jacobian must be finite")
    result = jacobian.clone()
    result[:, :, :wrist_start_joint_index] = 0.0
    return result


__all__ = [
    "ELBOW_NEGATIVE_KEYS",
    "ELBOW_POSITIVE_KEYS",
    "G2ElbowKeyState",
    "G2NullspaceProjection",
    "G2_REDUNDANCY_ACTION_SCHEMA",
    "damped_nullspace_project",
    "ee_rotation_only_mask",
    "tensor_value",
    "wrist_only_rotation_jacobian",
]
