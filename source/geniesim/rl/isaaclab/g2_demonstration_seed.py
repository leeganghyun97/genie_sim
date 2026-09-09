"""Fail-closed reset seeds extracted from canonical keyboard demonstrations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import h5py
import numpy as np

from .g2_keyboard_pose import (
    G2_KEYBOARD_PHOTO_POSE_PROFILE,
    G2_KEYBOARD_PHOTO_RIGHT_ARM_Q,
)
from .g2_teacher_sac import G2TeacherObservationContract


@dataclass(frozen=True)
class G2DemonstrationResetSeed:
    dataset: str
    episode: str
    row: int
    ee_cube_distance_m: float
    right_arm_q_rad: tuple[float, ...]

    def serializable(self) -> dict[str, object]:
        return asdict(self)


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def select_safe_pregrasp_seed(
    datasets: list[Path] | tuple[Path, ...],
    *,
    target_distance_m: float = 0.10,
    minimum_distance_m: float = 0.08,
    maximum_distance_m: float = 0.12,
    required_dataset: str | None = None,
    required_episode: str | None = None,
) -> G2DemonstrationResetSeed:
    """Select the closest measured, collision-attested pre-contact arm state.

    The selected row is a reset state only.  It is not a behavior-cloning
    action label and therefore does not turn a trajectory into imitation data.
    """

    if not datasets:
        raise ValueError("at least one demonstration dataset is required")
    if not 0.0 < minimum_distance_m < target_distance_m < maximum_distance_m:
        raise ValueError("invalid pregrasp distance interval")
    slices = G2TeacherObservationContract().slices
    joint_slice = slices["controlled_joint_position_relative_rad"]
    ee_slice = slices["end_effector_pose_root_xyzw"]
    cube_slice = slices["cube_pose_root_xyzw"]
    best: tuple[float, G2DemonstrationResetSeed] | None = None
    initial_q = np.asarray(G2_KEYBOARD_PHOTO_RIGHT_ARM_Q, dtype=np.float64)

    for dataset_path in datasets:
        dataset = Path(dataset_path).resolve()
        if required_dataset is not None and str(dataset) != str(Path(required_dataset).resolve()):
            continue
        with h5py.File(dataset, "r") as file:
            data = file.get("data")
            if data is None or "env_args" not in data.attrs:
                raise ValueError(f"demonstration has no env_args: {dataset}")
            environment = json.loads(_text(data.attrs["env_args"]))
            pose = environment.get("initial_pose_contract")
            if not isinstance(pose, dict):
                raise ValueError("demonstration initial pose contract is missing")
            if pose.get("profile") != G2_KEYBOARD_PHOTO_POSE_PROFILE:
                raise ValueError("demonstration reset profile differs")
            observed_initial = np.asarray(pose.get("right_arm_q_rad"), dtype=np.float64)
            if observed_initial.shape != (7,) or not np.allclose(
                observed_initial, initial_q, rtol=0.0, atol=1.0e-9
            ):
                raise ValueError("demonstration initial right-arm pose differs")

            for episode_name, group in data.items():
                if not episode_name.startswith("demo_"):
                    continue
                if required_episode is not None and episode_name != required_episode:
                    continue
                required = (
                    "teacher_observation",
                    "forbidden_collision",
                    "forbidden_collision_valid",
                    "bilateral_contact",
                )
                if any(name not in group for name in required):
                    raise ValueError(f"demonstration episode is incomplete: {episode_name}")
                observation = np.asarray(group["teacher_observation"], dtype=np.float64)
                collision = np.asarray(group["forbidden_collision"]).reshape(-1).astype(bool)
                collision_valid = (
                    np.asarray(group["forbidden_collision_valid"])
                    .reshape(-1)
                    .astype(bool)
                )
                contact = np.asarray(group["bilateral_contact"]).reshape(-1).astype(bool)
                ee = observation[:, ee_slice.start : ee_slice.start + 3]
                cube = observation[:, cube_slice.start : cube_slice.start + 3]
                distance = np.linalg.norm(cube - ee, axis=-1)
                eligible = (
                    collision_valid
                    & ~collision
                    & ~contact
                    & np.isfinite(distance)
                    & (distance >= minimum_distance_m)
                    & (distance <= maximum_distance_m)
                )
                for row in np.flatnonzero(eligible):
                    relative = observation[row, joint_slice.start : joint_slice.start + 7]
                    measured_q = initial_q + relative
                    if not np.isfinite(measured_q).all():
                        continue
                    seed = G2DemonstrationResetSeed(
                        dataset=str(dataset),
                        episode=episode_name,
                        row=int(row),
                        ee_cube_distance_m=float(distance[row]),
                        right_arm_q_rad=tuple(float(value) for value in measured_q),
                    )
                    score = abs(seed.ee_cube_distance_m - target_distance_m)
                    if best is None or score < best[0]:
                        best = (score, seed)
    if best is None:
        raise ValueError("demonstration has no safe collision-attested pregrasp seed")
    return best[1]


__all__ = ["G2DemonstrationResetSeed", "select_safe_pregrasp_seed"]
