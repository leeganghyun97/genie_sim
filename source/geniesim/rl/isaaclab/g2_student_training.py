"""Offline recurrent-student training and deployment contracts for G2.

The privileged teacher and RGB-D collection paths deliberately remain separate
from this module.  Only the current canonical keyboard/teacher dataset may be
loaded.  Older quaternion/depth/action schemas require an explicit migration
and are never guessed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np
import torch
from torch import nn

from .g2_keyboard_teacher_dataset import (
    G2_KEYBOARD_TEACHER_DATASET_SCHEMA,
    G2_KEYBOARD_TEACHER_PUBLIC_HDF_KEYS,
    G2KeyboardTeacherTransitionContract,
    G2StudentSequenceContract,
)
from .g2_keyboard_pose import (
    G2_KEYBOARD_CUBE_CENTER_WORLD_M,
    G2_KEYBOARD_PHOTO_POSE_PROFILE,
    G2_KEYBOARD_PHOTO_RIGHT_ARM_Q,
)
from .g2_lift_methodology import G2LiftSandboxContract
from .g2_visual_sac import (
    G2_RECURRENT_STUDENT_SCHEMA,
    G2RecurrentVisualPolicyContract,
    G2_STUDENT_FAILURE_CLASSES,
    G2_VISUAL_ACTION_DIM,
    G2_VISUAL_CAMERA_SHAPE,
    G2_VISUAL_PROPRIO_DIM,
    G2RecurrentVisualStudent,
    preprocess_demonstration_expert_actions,
    random_shift_rgbd_sequences,
)


G2_STUDENT_TRAINING_CHECKPOINT_SCHEMA = "g2_recurrent_student_checkpoint_v3"
G2_STUDENT_COMPACT_RGBD_BYTES_PER_TRANSITION = int(np.prod(G2_VISUAL_CAMERA_SHAPE))
G2_STUDENT_RAW_RGBD_BYTES_PER_TRANSITION = 2 * 192 * 256 * (3 + 4 + 1)


def estimate_student_rgbd_storage(transitions: int) -> dict[str, int | float]:
    """Return an uncompressed RGB-D-only capacity floor.

    The raw estimate includes two uint8 RGB images, two float32 metric-depth
    images and two uint8 valid masks.  HDF5 metadata, proprioception, labels,
    replay next-state images, and recurrent-window overlap are intentionally
    excluded and therefore can only increase actual storage.
    """

    if transitions < 0:
        raise ValueError("transition count cannot be negative")
    compact = int(transitions) * G2_STUDENT_COMPACT_RGBD_BYTES_PER_TRANSITION
    raw = int(transitions) * G2_STUDENT_RAW_RGBD_BYTES_PER_TRANSITION
    return {
        "transitions": int(transitions),
        "compact_rgbd_bytes": compact,
        "compact_rgbd_tib": compact / float(1 << 40),
        "raw_rgbd_bytes": raw,
        "raw_rgbd_tib": raw / float(1 << 40),
    }


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _dataset_environment(file: h5py.File) -> dict[str, object] | None:
    data = file.get("data")
    if data is None or "env_args" not in data.attrs:
        return None
    try:
        environment = json.loads(_text(data.attrs["env_args"]))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("dataset env_args is not valid JSON") from error
    if not isinstance(environment, dict):
        raise ValueError("dataset env_args must be a JSON object")
    return environment


def _dataset_schema(file: h5py.File) -> str | None:
    """Read either the contract-oracle or Isaac Lab HDF5 schema location."""

    if "schema" in file.attrs:
        return _text(file.attrs["schema"])
    environment = _dataset_environment(file)
    if environment is None:
        return None
    schema = environment.get("dataset_schema")
    return None if schema is None else str(schema)


def _validate_initial_pose_profile(file: h5py.File) -> None:
    """Require the reset distribution that produced the keyboard data."""

    environment = _dataset_environment(file)
    # Root-schema files are bounded contract-oracle fixtures, not collector
    # output.  Live Isaac datasets always own env_args and are strict here.
    if environment is None:
        return
    pose = environment.get("initial_pose_contract")
    profile = pose.get("profile") if isinstance(pose, dict) else None
    if profile != G2_KEYBOARD_PHOTO_POSE_PROFILE:
        raise ValueError(
            "student dataset initial pose differs: "
            f"{profile!r} != {G2_KEYBOARD_PHOTO_POSE_PROFILE!r}"
        )
    right_arm_q = pose.get("right_arm_q_rad") if isinstance(pose, dict) else None
    if not isinstance(right_arm_q, list) or len(right_arm_q) != 7:
        raise ValueError("student dataset initial right-arm pose is missing")
    if not np.allclose(
        np.asarray(right_arm_q, dtype=np.float64),
        np.asarray(G2_KEYBOARD_PHOTO_RIGHT_ARM_Q, dtype=np.float64),
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError("student dataset initial right-arm pose differs")


def _validate_recurrent_visual_policy_contract(file: h5py.File) -> None:
    """Require data-facing dimensions while allowing a newer model backbone.

    GRU width, temporal windowing, and encoder topology are model/checkpoint
    properties, not stored observation semantics. Canonical files store full
    episode timelines, so compatible keyboard RGB-D can be re-windowed under a
    newer recurrent training contract without changing observations.
    """

    environment = _dataset_environment(file)
    if environment is None:
        return
    expected = G2RecurrentVisualPolicyContract().validated().serializable()
    observed = environment.get("recurrent_visual_policy_contract")
    data_fields = (
        "camera_order",
        "camera_channels",
        "camera_shape",
        "proprioception_dim",
        "action_dim",
        "torso_policy_output_enabled",
    )
    # Build the field list without treating hidden width/backbone as dataset
    # dimensions. Older keyboard recordings remain valid training data for a
    # new GRU width/window, while incompatible tensors still fail closed.
    if not isinstance(observed, dict) or any(
        observed.get(name) != expected.get(name) for name in data_fields
    ):
        raise ValueError(
            "student dataset recurrent visual policy contract differs"
        )


def _validate_task_object_geometry(file: h5py.File) -> None:
    """Reject demonstrations recorded with the superseded 30 mm cube."""

    environment = _dataset_environment(file)
    if environment is None:
        return
    geometry = environment.get("keyboard_task_geometry")
    sandbox = G2LiftSandboxContract().validated()
    if not isinstance(geometry, dict):
        raise ValueError("student dataset task geometry is missing")
    if tuple(geometry.get("cube_size_m", ())) != sandbox.cube_size_m:
        raise ValueError("student dataset object dimensions differ")
    center = geometry.get("cube_center_world_m")
    if not isinstance(center, list) or not np.allclose(
        np.asarray(center, dtype=np.float64),
        np.asarray(G2_KEYBOARD_CUBE_CENTER_WORLD_M, dtype=np.float64),
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError("student dataset object resting center differs")


def list_canonical_episodes(path: str | Path) -> tuple[str, ...]:
    """Return numerically ordered episode keys after strict schema validation."""

    source = Path(path)
    with h5py.File(source, "r") as file:
        schema = _dataset_schema(file)
        if schema != G2_KEYBOARD_TEACHER_DATASET_SCHEMA:
            raise ValueError(
                "student dataset schema differs: "
                f"{schema!r} != {G2_KEYBOARD_TEACHER_DATASET_SCHEMA!r}"
            )
        _validate_initial_pose_profile(file)
        _validate_recurrent_visual_policy_contract(file)
        _validate_task_object_geometry(file)
        if "data" not in file:
            raise ValueError("student dataset has no data group")
        keys = [name for name in file["data"] if name.startswith("demo_")]
        if not keys:
            raise ValueError("student dataset contains no episodes")
        try:
            return tuple(sorted(keys, key=lambda value: int(value.rsplit("_", 1)[1])))
        except (IndexError, ValueError) as error:
            raise ValueError("student episode names are not canonical demo_N keys") from error


def load_canonical_episode(
    path: str | Path,
    episode_key: str,
) -> dict[str, torch.Tensor]:
    """Materialize and validate one canonical episode on CPU.

    The file is closed before tensors are returned, which prevents h5py handles
    from leaking into DataLoader workers or Isaac/Kit finalization.
    """

    source = Path(path)
    with h5py.File(source, "r") as file:
        schema = _dataset_schema(file)
        if schema != G2_KEYBOARD_TEACHER_DATASET_SCHEMA:
            raise ValueError(
                "student dataset schema differs: "
                f"{schema!r} != {G2_KEYBOARD_TEACHER_DATASET_SCHEMA!r}"
            )
        _validate_initial_pose_profile(file)
        _validate_recurrent_visual_policy_contract(file)
        _validate_task_object_geometry(file)
        group_path = f"data/{episode_key}"
        if group_path not in file:
            raise KeyError(f"student episode does not exist: {group_path}")
        group = file[group_path]
        required = (
            G2KeyboardTeacherTransitionContract().required_keys
            | G2_KEYBOARD_TEACHER_PUBLIC_HDF_KEYS
        )
        missing = sorted(required.difference(group.keys()))
        if missing:
            raise ValueError(f"student episode is missing canonical fields: {missing}")
        episode = {
            name: torch.from_numpy(np.asarray(group[name]).copy())
            for name in required
        }

    if episode["head_rgb"].ndim != 4:
        raise ValueError("head_rgb must be [T,H,W,3]")
    height, width = map(int, episode["head_rgb"].shape[1:3])
    contract = G2KeyboardTeacherTransitionContract(height=height, width=width)
    validation = contract.validate_hdf_episode(episode)
    if not validation["pass"]:
        raise ValueError(f"invalid canonical student episode: {validation}")
    return episode


def load_canonical_sequences(
    path: str | Path,
    *,
    sequence_contract: G2StudentSequenceContract | None = None,
    successful_episodes_only: bool = False,
    expert_arm_action_scale: float = 1.0,
) -> dict[str, torch.Tensor | int]:
    """Load compact recurrent sequences from all episodes in one canonical file.

    This convenience loader is intentionally bounded to one dataset file.  It
    is suitable for smoke and moderate offline datasets, not a claim that tens
    of millions of full-resolution RGB-D transitions fit in host memory.
    """

    sequence_contract = sequence_contract or G2StudentSequenceContract()
    tensor_names = (
        "rgbd_u8",
        "deployable_proprioception",
        "expert_action_target",
        "teacher_state_target",
        "success",
        "expert_confidence",
        "teacher_q_value",
        "teacher_value_valid",
        "relative_pose_target",
        "contact_target",
        "contact_force_target_n",
        "bilateral_contact_target",
        "stable_grasp_target",
        "slip_speed_target_m_s",
        "push_before_grasp_trial",
        "recovery_fail",
        "depth_validity_target",
        "failure_class_target",
        "failure_class_valid",
        "safe_for_her_force",
        "camera_frame_age",
        "episode_id",
        # Derived provenance label. ``success`` is a terminal transition label,
        # so it cannot identify early windows from a successful episode.
        "episode_outcome_success",
        "sequence_step",
        "padding_mask",
        "hidden_reset_mask",
        "sequence_lengths",
    )
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in tensor_names}
    for episode_key in list_canonical_episodes(path):
        episode = load_canonical_episode(path, episode_key)
        episode_outcome_success = bool(episode["success"].to(torch.bool).any())
        if successful_episodes_only and not episode_outcome_success:
            continue
        height, width = map(int, episode["head_rgb"].shape[1:3])
        sequence = sequence_contract.build(
            episode,
            transition_contract=G2KeyboardTeacherTransitionContract(
                height=height, width=width
            ),
        )
        lengths = sequence["sequence_lengths"]
        if not isinstance(lengths, torch.Tensor):
            raise RuntimeError("student sequence lengths are not a tensor")
        eligible = lengths > sequence_contract.burn_in_steps
        sequence["episode_outcome_success"] = torch.full(
            (int(lengths.shape[0]),), episode_outcome_success, dtype=torch.bool
        )
        # A tail shorter than burn-in carries no supervised timestep and must
        # never become a randomly sampled optimizer batch on its own.
        if not bool(eligible.any()):
            continue
        for name in tensor_names:
            value = sequence[name]
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"student sequence {name} is not a tensor")
            collected[name].append(value[eligible])
    if not collected["rgbd_u8"]:
        raise ValueError("student dataset has no sequence longer than burn-in")
    result: dict[str, torch.Tensor | int] = {
        name: torch.cat(values, dim=0) for name, values in collected.items()
    }
    # The generic sequence pad is zero-filled.  A zero quaternion is not a
    # valid relative-pose target even though padded rows carry no loss weight,
    # because quaternion canonicalization happens before reduction.  Preserve
    # the mask while making only those non-supervised rows a valid identity
    # rotation.  Real dataset targets are never rewritten.
    padding_mask = result["padding_mask"]
    relative_pose = result["relative_pose_target"]
    if not isinstance(padding_mask, torch.Tensor) or not isinstance(
        relative_pose, torch.Tensor
    ):
        raise RuntimeError("student padding/relative-pose tensors are unavailable")
    padded = ~padding_mask.to(torch.bool)
    if bool(padded.any()):
        relative_pose = relative_pose.clone()
        # ``padded`` indexes the [sequence,time] axes; apply it to the
        # quaternion view explicitly.  ``relative_pose[padded, 3:7]`` mixes
        # advanced and basic indexing and incorrectly treats 3:7 as a time
        # slice instead of the final pose dimension.
        relative_pose[..., 3:7][padded] = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], dtype=relative_pose.dtype,
            device=relative_pose.device,
        )
        result["relative_pose_target"] = relative_pose
    result["burn_in_steps"] = sequence_contract.burn_in_steps
    result["successful_episodes_only"] = bool(successful_episodes_only)
    expert_action = result["expert_action_target"]
    if not isinstance(expert_action, torch.Tensor):
        raise RuntimeError("expert action target is unavailable")
    processed_action, preprocessing = preprocess_demonstration_expert_actions(
        expert_action, arm_action_scale=expert_arm_action_scale
    )
    result["expert_action_target"] = processed_action
    result["expert_action_preprocessing"] = preprocessing
    return result


@dataclass(frozen=True)
class G2StudentTrainingConfig:
    hidden_dim: int = 256
    gru_num_layers: int = 1
    learning_rate: float = 3.0e-4
    gradient_clip: float = 5.0
    sequence_length: int = 16
    burn_in_steps: int = 4
    stride: int = 12
    value_loss_weight: float = 0.2
    auxiliary_loss_weight: float = 0.2
    pose_consistency_loss_weight: float = 0.1
    pose_consistency_rotation_weight: float = 0.1
    cross_camera_contrastive_loss_weight: float = 0.02
    cross_camera_contrastive_temperature: float = 0.1
    cross_camera_false_negative_position_threshold_m: float = 0.01
    temporal_pose_residual_loss_weight: float = 0.05
    temporal_pose_residual_rotation_weight: float = 0.1
    contact_loss_weight: float = 0.1
    stable_grasp_loss_weight: float = 0.1
    slip_speed_loss_weight: float = 0.1
    depth_validity_loss_weight: float = 0.05
    failure_loss_weight: float = 0.1
    share_camera_encoder_weights: bool = False
    camera_encoder_channels: tuple[int, ...] = (32, 64, 96, 128)
    torso_control_enabled: bool = False
    random_shift_pad: int = 4
    seed: int = 42

    def validate(self) -> "G2StudentTrainingConfig":
        G2StudentSequenceContract(
            sequence_length=self.sequence_length,
            burn_in_steps=self.burn_in_steps,
            stride=self.stride,
        )
        if self.hidden_dim <= 0 or self.learning_rate <= 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("student optimizer/model parameters must be positive")
        if self.gru_num_layers != 1:
            raise ValueError("student training requires exactly one GRU layer")
        if self.random_shift_pad < 0:
            raise ValueError("student random-shift padding cannot be negative")
        for name in (
            "value_loss_weight", "auxiliary_loss_weight",
            "pose_consistency_loss_weight", "pose_consistency_rotation_weight",
            "cross_camera_contrastive_loss_weight",
            "cross_camera_false_negative_position_threshold_m",
            "temporal_pose_residual_loss_weight",
            "temporal_pose_residual_rotation_weight",
            "contact_loss_weight",
            "stable_grasp_loss_weight", "slip_speed_loss_weight",
            "depth_validity_loss_weight", "failure_loss_weight",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if self.cross_camera_contrastive_temperature <= 0.0:
            raise ValueError("cross-camera contrastive temperature must be positive")
        return self

    def sequence_contract(self) -> G2StudentSequenceContract:
        return G2StudentSequenceContract(
            sequence_length=self.sequence_length,
            burn_in_steps=self.burn_in_steps,
            stride=self.stride,
        )

    def model(self) -> G2RecurrentVisualStudent:
        return G2RecurrentVisualStudent(
            hidden_dim=self.hidden_dim,
            gru_num_layers=self.gru_num_layers,
            value_loss_weight=self.value_loss_weight,
            auxiliary_loss_weight=self.auxiliary_loss_weight,
            pose_consistency_loss_weight=self.pose_consistency_loss_weight,
            pose_consistency_rotation_weight=self.pose_consistency_rotation_weight,
            cross_camera_contrastive_loss_weight=(
                self.cross_camera_contrastive_loss_weight
            ),
            cross_camera_contrastive_temperature=(
                self.cross_camera_contrastive_temperature
            ),
            cross_camera_false_negative_position_threshold_m=(
                self.cross_camera_false_negative_position_threshold_m
            ),
            temporal_pose_residual_loss_weight=(
                self.temporal_pose_residual_loss_weight
            ),
            temporal_pose_residual_rotation_weight=(
                self.temporal_pose_residual_rotation_weight
            ),
            contact_loss_weight=self.contact_loss_weight,
            stable_grasp_loss_weight=self.stable_grasp_loss_weight,
            slip_speed_loss_weight=self.slip_speed_loss_weight,
            depth_validity_loss_weight=self.depth_validity_loss_weight,
            failure_loss_weight=self.failure_loss_weight,
            share_camera_encoder_weights=self.share_camera_encoder_weights,
            camera_encoder_channels=self.camera_encoder_channels,
            torso_control_enabled=self.torso_control_enabled,
        )


class G2RecurrentStudentTrainer:
    """One explicit optimizer/checkpoint boundary for canonical distillation."""

    def __init__(
        self,
        config: G2StudentTrainingConfig | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = (config or G2StudentTrainingConfig()).validate()
        self.device = torch.device(device)
        torch.manual_seed(self.config.seed)
        self.rng = np.random.default_rng(self.config.seed)
        self.model = self.config.model().to(self.device)
        self.optimizer = torch.optim.Adam(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=self.config.learning_rate,
        )
        self.update_count = 0

    def _tensor(self, batch: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
        if name not in batch or not isinstance(batch[name], torch.Tensor):
            raise ValueError(f"student batch is missing tensor {name}")
        return batch[name].to(self.device)

    def update(self, batch: Mapping[str, torch.Tensor | int]) -> dict[str, float]:
        burn_in = int(batch.get("burn_in_steps", self.config.burn_in_steps))
        padding = self._tensor(batch, "padding_mask").to(torch.bool)
        if int(padding[:, burn_in:].sum()) <= 0:
            raise ValueError("student batch has no supervised post-burn-in samples")
        self.model.train()
        original_rgbd_u8 = self._tensor(batch, "rgbd_u8").to(torch.uint8)
        rgbd_u8, principal_point_shift_px = random_shift_rgbd_sequences(
            original_rgbd_u8,
            pad=self.config.random_shift_pad,
            return_principal_point_shift=True,
        )
        loss, components, _ = self.model.distillation_loss(
            rgbd_u8=rgbd_u8,
            pose_consistency_rgbd_u8=original_rgbd_u8,
            camera_principal_point_shift_px=principal_point_shift_px,
            deployable_proprioception=self._tensor(batch, "deployable_proprioception").float(),
            expert_action=self._tensor(batch, "expert_action_target").float(),
            expert_confidence=self._tensor(batch, "expert_confidence").float(),
            episode_id=self._tensor(batch, "episode_id").long(),
            sequence_step=self._tensor(batch, "sequence_step").long(),
            teacher_value=self._tensor(batch, "teacher_q_value").float(),
            teacher_value_valid=self._tensor(batch, "teacher_value_valid").to(torch.bool),
            relative_pose_target=self._tensor(batch, "relative_pose_target").float(),
            contact_target=self._tensor(batch, "contact_target").float(),
            stable_grasp_target=self._tensor(batch, "stable_grasp_target").float(),
            slip_speed_target_m_s=self._tensor(batch, "slip_speed_target_m_s").float(),
            depth_validity_target=self._tensor(batch, "depth_validity_target").float(),
            failure_target=self._tensor(batch, "failure_class_target").long(),
            failure_target_valid=self._tensor(batch, "failure_class_valid").to(torch.bool),
            padding_mask=padding,
            sequence_lengths=self._tensor(batch, "sequence_lengths").long(),
            hidden_reset_mask=self._tensor(batch, "hidden_reset_mask").to(torch.bool),
            burn_in_steps=burn_in,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            self.config.gradient_clip,
        )
        self.optimizer.step()
        self.update_count += 1
        confidence = self._tensor(batch, "expert_confidence").float()
        value_valid = self._tensor(batch, "teacher_value_valid").to(torch.bool)
        valid_rows = padding.unsqueeze(-1)
        valid_count = torch.clamp(valid_rows.sum(), min=1)
        supervised_rows = valid_rows.clone()
        supervised_rows[:, :burn_in] = False
        supervised_count = torch.clamp(supervised_rows.sum(), min=1)
        expert_action = self._tensor(batch, "expert_action_target").float()
        bilateral = self._tensor(batch, "bilateral_contact_target").float()
        stable = self._tensor(batch, "stable_grasp_target").float()
        slip = self._tensor(batch, "slip_speed_target_m_s").float()
        push = self._tensor(batch, "push_before_grasp_trial").float()
        recovery_fail = self._tensor(batch, "recovery_fail").float()
        failure_class = self._tensor(batch, "failure_class_target").long()
        failure_class_valid = self._tensor(batch, "failure_class_valid").to(torch.bool)
        camera_age = self._tensor(batch, "camera_frame_age").float()
        metrics = {name: float(value.detach()) for name, value in components.items()}
        metrics.update(
            {
                "student/update": float(self.update_count),
                "student/gradient_norm": float(gradient_norm),
                "student/learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "student/padding_valid_fraction": float(padding.float().mean()),
                "student/expert_confidence_mean": float(
                    (confidence * valid_rows).sum() / valid_count
                ),
                "student/teacher_value_valid_fraction": float(
                    (value_valid & valid_rows).sum() / valid_count
                ),
                "student/target_action_l2_mean": float(
                    (
                        torch.linalg.vector_norm(expert_action, dim=-1, keepdim=True)
                        * supervised_rows
                    ).sum()
                    / supervised_count
                ),
                "student/target_gripper_open_fraction": float(
                    ((expert_action[..., 6:7] > 0.0) & supervised_rows).sum()
                    / supervised_count
                ),
                "student/target_bilateral_contact_fraction": float(
                    (bilateral * supervised_rows).sum() / supervised_count
                ),
                "student/target_stable_grasp_fraction": float(
                    (stable * supervised_rows).sum() / supervised_count
                ),
                "student/target_slip_speed_mean_m_s": float(
                    (slip * supervised_rows).sum() / supervised_count
                ),
                "student/target_pregrasp_push_fraction": float(
                    (push * supervised_rows).sum() / supervised_count
                ),
                "student/target_recovery_failure_fraction": float(
                    (recovery_fail * supervised_rows).sum() / supervised_count
                ),
                "student/failure_label_valid_fraction": float(
                    (failure_class_valid.unsqueeze(-1) & supervised_rows).sum()
                    / supervised_count
                ),
                "student/camera_frame_age_mean_s": float(
                    (camera_age * supervised_rows).sum()
                    / (supervised_count * camera_age.shape[-1])
                ),
                "student/burn_in_steps": float(burn_in),
                "student/random_shift_pad_pixels": float(
                    self.config.random_shift_pad
                ),
            }
        )
        for class_index, class_name in enumerate(G2_STUDENT_FAILURE_CLASSES):
            failure_supervised = supervised_rows & failure_class_valid.unsqueeze(-1)
            failure_denominator = torch.clamp(failure_supervised.sum(), min=1)
            metrics[f"student/target_phase_{class_name}_fraction"] = float(
                ((failure_class == class_index).unsqueeze(-1) & failure_supervised).sum()
                / failure_denominator
            )
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("student update produced a non-finite metric")
        return metrics

    def state_dict(self) -> dict[str, object]:
        policy_contract = G2RecurrentVisualPolicyContract(
            hidden_dim=self.config.hidden_dim,
            gru_num_layers=self.config.gru_num_layers,
            sequence_length=self.config.sequence_length,
            burn_in_steps=self.config.burn_in_steps,
            sequence_stride=self.config.stride,
        ).validated()
        return {
            "schema": G2_STUDENT_TRAINING_CHECKPOINT_SCHEMA,
            "dataset_schema": G2_KEYBOARD_TEACHER_DATASET_SCHEMA,
            "model_schema": G2_RECURRENT_STUDENT_SCHEMA,
            "camera_shape": G2_VISUAL_CAMERA_SHAPE,
            "proprio_dim": G2_VISUAL_PROPRIO_DIM,
            "action_dim": G2_VISUAL_ACTION_DIM,
            "recurrent_visual_policy_contract": policy_contract.serializable(),
            "config": asdict(self.config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": (
                torch.cuda.get_rng_state_all()
                if self.device.type == "cuda" and torch.cuda.is_available()
                else None
            ),
            "numpy_rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        policy_contract = G2RecurrentVisualPolicyContract(
            hidden_dim=self.config.hidden_dim,
            gru_num_layers=self.config.gru_num_layers,
            sequence_length=self.config.sequence_length,
            burn_in_steps=self.config.burn_in_steps,
            sequence_stride=self.config.stride,
        ).validated()
        expected = {
            "schema": G2_STUDENT_TRAINING_CHECKPOINT_SCHEMA,
            "dataset_schema": G2_KEYBOARD_TEACHER_DATASET_SCHEMA,
            "model_schema": G2_RECURRENT_STUDENT_SCHEMA,
            "camera_shape": G2_VISUAL_CAMERA_SHAPE,
            "proprio_dim": G2_VISUAL_PROPRIO_DIM,
            "action_dim": G2_VISUAL_ACTION_DIM,
            "recurrent_visual_policy_contract": policy_contract.serializable(),
            "config": asdict(self.config),
        }
        for name, value in expected.items():
            checkpoint_value = state.get(name)
            if name == "camera_shape" and checkpoint_value is not None:
                checkpoint_value = tuple(checkpoint_value)
            if checkpoint_value != value:
                raise ValueError(f"student checkpoint {name} differs")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.update_count = int(state["update_count"])
        torch.set_rng_state(state["torch_rng_state"])
        cuda_rng_state = state.get("torch_cuda_rng_state_all")
        if (
            cuda_rng_state is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(cuda_rng_state)
        self.rng.bit_generator.state = state["numpy_rng_state"]

    def save_checkpoint(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(self.state_dict(), temporary)
        os.replace(temporary, destination)
        return destination

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "G2RecurrentStudentTrainer":
        # Checkpoints are trusted local training artifacts.  Never load an
        # untrusted pickle through this API.
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        if state.get("schema") != G2_STUDENT_TRAINING_CHECKPOINT_SCHEMA:
            raise ValueError("student checkpoint schema differs")
        trainer = cls(G2StudentTrainingConfig(**state["config"]), device=device)
        trainer.load_state_dict(state)
        return trainer


class G2StudentDeploymentPolicy:
    """Stateful vector-env GRU inference with explicit reset semantics."""

    def __init__(
        self,
        model: G2RecurrentVisualStudent,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.hidden: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None, *, batch: int | None = None) -> None:
        if self.hidden is None:
            if batch is None:
                return
            self.hidden = self.model.initial_hidden(batch, device=self.device)
            return
        if env_ids is None:
            self.hidden.zero_()
        else:
            self.hidden[:, env_ids.to(self.device, dtype=torch.long)] = 0.0

    @torch.no_grad()
    def step(
        self,
        rgbd_u8: torch.Tensor,
        deployable_proprioception: torch.Tensor,
        *,
        reset_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if tuple(rgbd_u8.shape[1:]) != G2_VISUAL_CAMERA_SHAPE:
            raise ValueError("deployment RGB-D shape differs")
        batch = int(rgbd_u8.shape[0])
        if tuple(deployable_proprioception.shape) != (batch, G2_VISUAL_PROPRIO_DIM):
            raise ValueError("deployment proprioception shape differs")
        if self.hidden is None:
            self.reset(batch=batch)
        elif self.hidden.shape[1] != batch:
            raise ValueError("deployment batch changed without an explicit policy reset")
        if reset_mask is None:
            reset_mask = torch.zeros(batch, dtype=torch.bool, device=self.device)
        if tuple(reset_mask.shape) != (batch,):
            raise ValueError("deployment reset_mask must be [N]")
        output, self.hidden = self.model.forward_sequence(
            rgbd_u8.to(self.device, dtype=torch.uint8).unsqueeze(1),
            deployable_proprioception.to(self.device, dtype=torch.float32).unsqueeze(1),
            self.hidden,
            reset_mask.to(self.device, dtype=torch.bool).unsqueeze(1),
            torch.ones((batch, 1), dtype=torch.bool, device=self.device),
        )
        raw_action = output["action"][:, 0]
        controller_action = raw_action.clone()
        # The G2 environment exposes a BinaryJointPositionAction.  Distillation
        # learns its signed command; deployment makes that adapter explicit.
        controller_action[:, 6] = torch.where(
            raw_action[:, 6] >= 0.0,
            torch.ones_like(raw_action[:, 6]),
            -torch.ones_like(raw_action[:, 6]),
        )
        # A recurrent hidden-state reset is also a physical episode boundary.
        # Never let the prior episode's close decision become the first
        # command of the freshly opened gripper.
        controller_action[reset_mask, 6] = 1.0
        if not bool(torch.isfinite(raw_action).all()):
            raise FloatingPointError("student deployment action is non-finite")
        return {
            "raw_student_action": raw_action,
            "controller_action": controller_action,
            "hidden": self.hidden,
        }


__all__ = [
    "G2_STUDENT_TRAINING_CHECKPOINT_SCHEMA",
    "G2_STUDENT_COMPACT_RGBD_BYTES_PER_TRANSITION",
    "G2_STUDENT_RAW_RGBD_BYTES_PER_TRANSITION",
    "G2RecurrentStudentTrainer",
    "G2StudentDeploymentPolicy",
    "G2StudentTrainingConfig",
    "list_canonical_episodes",
    "load_canonical_episode",
    "load_canonical_sequences",
    "estimate_student_rgbd_storage",
]
