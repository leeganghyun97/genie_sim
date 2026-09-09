"""Teacher-compatible RGB-D transition schema for 8-D G2 keyboard control."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping

import torch

from .g2_teacher_sac import G2_TEACHER_ACTION_DIM, G2TeacherObservationContract
from .g2_lift_methodology import G2LiftSandboxContract
from .g2_quaternion import canonicalize_quaternion_xyzw
from .g2_teleop_dataset import (
    G2_ROTATION_ACTION_SCALE_RAD,
    G2_TRANSLATION_ACTION_SCALE_M,
    G2_REDUNDANCY_TELEOP_ACTION_DIM,
    G2RedundancyKeyboardTeleopContract,
)
from .g2_visual_sac import (
    G2RecurrentVisualPolicyContract,
    G2StudentObservationContract,
    pack_rgbd,
    relative_pose_target_from_teacher_state,
)


G2_KEYBOARD_TEACHER_DATASET_SCHEMA = (
    "g2_keyboard_teacher_rgbd_canonical_xyzw_rotvec_rect_object_v25"
)
G2_KEYBOARD_TEACHER_PUBLIC_HDF_KEYS = frozenset({"actions"})

# Persist these labels beside the executable contract.  This prevents the
# collection script from documenting an order that differs from validation.
G2_KEYBOARD_PHYSICAL_ACTION_LABELS = (
    "dx_m", "dy_m", "dz_m", "drotvec_x_rad", "drotvec_y_rad", "drotvec_z_rad",
    "elbow_nullspace_normalized", "gripper_binary",
)
G2_KEYBOARD_NORMALIZED_ACTION_LABELS = (
    "dx_normalized", "dy_normalized", "dz_normalized",
    "drotvec_x_normalized", "drotvec_y_normalized", "drotvec_z_normalized",
    "elbow_nullspace_normalized", "gripper_binary",
)
G2_CANONICAL_POLICY_ACTION_LABELS = (
    "dx_normalized", "dy_normalized", "dz_normalized",
    "drotvec_x_normalized", "drotvec_y_normalized", "drotvec_z_normalized",
    "gripper_binary",
)
G2_CONTROLLED_JOINT_LABELS = (
    "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3",
    "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6",
    "idx67_arm_r_joint7", "idx81_gripper_r_outer_joint1",
)


def keyboard_teacher_dataset_metadata() -> dict[str, object]:
    """Return machine-readable dimensions, units, frames and time alignment."""

    teacher_fields: list[dict[str, object]] = []
    cursor = 0
    field_specs = (
        ("controlled_joint_position_relative_rad", 8, "rad", "robot_joint"),
        ("controlled_joint_velocity_rad_s", 8, "rad/s", "robot_joint"),
        ("end_effector_pose_root_xyzw", 7, "m+unit_quaternion_xyzw", "robot_root"),
        ("cube_pose_root_xyzw", 7, "m+unit_quaternion_xyzw", "robot_root"),
        ("cube_linear_angular_velocity", 6, "m/s+rad/s", "robot_root"),
        ("goal_position_root_m", 3, "m", "robot_root"),
        ("end_effector_to_cube_m", 3, "m", "robot_root"),
        ("cube_to_goal_m", 3, "m", "robot_root"),
        ("bilateral_contact_features", 3, "normalized_force+binary", "inner_outer"),
        ("curriculum_phase_features", 4, "binary", "precontact_contact_lift_success"),
        ("previous_action", 7, "normalized", "canonical_policy_action"),
    )
    for name, width, unit, frame in field_specs:
        teacher_fields.append(
            {
                "name": name,
                "start": cursor,
                "stop": cursor + width,
                "width": width,
                "unit": unit,
                "frame": frame,
            }
        )
        cursor += width
    if cursor != G2TeacherObservationContract().observation_dim:
        raise RuntimeError("teacher observation metadata dimension mismatch")

    sandbox = G2LiftSandboxContract().validated()
    return {
        "dataset_schema": G2_KEYBOARD_TEACHER_DATASET_SCHEMA,
        "task_object_geometry": {
            "shape": "cuboid",
            "size_xyz_m": list(sandbox.cube_size_m),
            "resting_center_z_m": sandbox.cube_nominal_world_m[2],
            "live_lift_success_center_z_m": (
                sandbox.lift_success_cube_center_z_m
            ),
        },
        "recurrent_visual_policy_contract": (
            G2RecurrentVisualPolicyContract().validated().serializable()
        ),
        "canonical_quaternion_order": "xyzw",
        "canonical_quaternion_rules": "unit_norm; deterministic_q_neg_q_sign",
        "teacher_observation_dim": cursor,
        "teacher_observation_fields": teacher_fields,
        "controlled_joint_order": list(G2_CONTROLLED_JOINT_LABELS),
        "keyboard_physical_action": {
            "shape": [8],
            "labels": list(G2_KEYBOARD_PHYSICAL_ACTION_LABELS),
            "units": ["m", "m", "m", "rad", "rad", "rad", "normalized", "binary"],
        },
        "keyboard_normalized_action": {
            "shape": [8],
            "labels": list(G2_KEYBOARD_NORMALIZED_ACTION_LABELS),
            "unit": "normalized",
            "physical_to_normalized": {
                "translation_divisor_m": G2_TRANSLATION_ACTION_SCALE_M,
                "rotation_vector_divisor_rad": G2_ROTATION_ACTION_SCALE_RAD,
                "elbow_nullspace": "identity_then_clip_-1_1",
                "gripper": "sign_to_binary_minus1_plus1",
                "normalized_limit": 1.0,
            },
            "rotation_execution_authority": {
                "pure_rpy_controlled_joints": [
                    "idx65_arm_r_joint5",
                    "idx66_arm_r_joint6",
                    "idx67_arm_r_joint7",
                ],
                "pure_rpy_held_joints": [
                    "idx61_arm_r_joint1",
                    "idx62_arm_r_joint2",
                    "idx63_arm_r_joint3",
                    "idx64_arm_r_joint4",
                ],
                "ee_position_lock": True,
                "neutral_packets_retain_outstanding_orientation_target": True,
            },
        },
        "canonical_teacher_student_action": {
            "shape": [7],
            "labels": list(G2_CANONICAL_POLICY_ACTION_LABELS),
            "unit": "normalized",
        },
        "public_hdf_actions": {
            "shape": [8],
            "equals": "applied_teleop_action",
            "semantics": "normalized_controller_request_submitted_to_environment",
        },
        "applied_actuator_joint_target_rad": {
            "shape": [8],
            "labels": list(G2_CONTROLLED_JOINT_LABELS),
            "unit": "rad",
            "semantics": "last_emitted_target_after_ik_limit_and_rate_limit",
        },
        "camera_layout": {
            "rgb": "[transition,height,width,rgb_channel]_uint8",
            "metric_depth": "[transition,height,width,1]_float_m",
            "valid_mask": "[transition,height,width,1]_uint8",
            "camera_order": ["head", "right_wrist"],
        },
        "time_units": {
            "timestamp": "episode_seconds",
            "camera_timestamp": "episode_seconds",
            "camera_frame_age": "seconds",
            "remote_source_timestamp": "source_clock_seconds",
            "remote_receive_age": "seconds",
        },
        "transition_alignment": {
            "pre_action_t": [
                "teacher_observation", "keyboard_physical_command", "teleop_action",
                "head_rgb", "head_depth", "right_wrist_rgb", "right_wrist_depth",
                "contact_force_n", "bilateral_contact", "stable_grasp",
                "slip_speed_m_s", "timestamp", "camera_timestamp", "camera_frame_age",
            ],
            "action_t": [
                "actions", "applied_teleop_action", "canonical_policy_action",
                "applied_actuator_joint_target_rad",
            ],
            "post_action_t_plus_1": [
                "next_teacher_observation", "reward_components", "reward",
                "terminated", "truncated", "success", "outcome_code",
                "forbidden_collision", "forbidden_collision_valid",
            ],
        },
        "collision_labels": {
            "forbidden_collision": "binary result from a live full-body authority",
            "forbidden_collision_valid": "1 only when that authority was measured for the row",
            "unmeasured_semantics": "forbidden_collision=0, forbidden_collision_valid=0; never safe",
            "her_force_safe_mask": "forbidden_collision_valid AND NOT forbidden_collision",
        },
    }


class G2ExpertSource(IntEnum):
    KEYBOARD = 0
    TEACHER = 1
    REMOTE_KEYBOARD = 2


class G2EpisodeOutcome(IntEnum):
    ONGOING = 0
    SUCCESS = 1
    TIMEOUT = 2
    OBJECT_DROP = 3
    FORBIDDEN_COLLISION = 4
    CAMERA_FAILURE = 5
    MANUAL_FAILURE = 6
    EMERGENCY_STOP = 7
    RECOVERY_FAILURE = 8
    REMOTE_COMMAND_STALE = 9
    # A real touch/stable-grasp/lift milestone was reached, but the object was
    # no longer stably held at the operator/episode boundary.  This is kept
    # distinct from SUCCESS so evaluation cannot inflate the success rate.
    HALF_SUCCESS = 10


def teacher_compatible_keyboard_controller_action(
    normalized_keyboard_action: torch.Tensor,
    *,
    allow_elbow_nullspace: bool = False,
) -> torch.Tensor:
    """Return the applied 8-D controller action for canonical data collection.

    The common teacher/student policy has 6-D SE(3) plus one gripper command.
    The independent keyboard elbow-nullspace channel cannot be represented by
    that contract, so the collection default disables it before control while
    retaining the unmodified raw key sample separately.
    """

    if normalized_keyboard_action.ndim != 2 or normalized_keyboard_action.shape[1] != 8:
        raise ValueError("normalized keyboard action must be [N,8]")
    applied = normalized_keyboard_action.clone()
    if not allow_elbow_nullspace:
        applied[:, 6] = 0.0
    return applied


@dataclass(frozen=True)
class G2KeyboardTeacherTransitionContract:
    height: int = 192
    width: int = 256
    # Stored actions are float32.  A full-scale 0.0225 m command round-trips as
    # 0.99999994, so 1e-8 incorrectly rejected an otherwise exact canonical
    # conversion.  This is numerical dtype tolerance, not a control or safety
    # limit relaxation.
    teacher_action_compatibility_tolerance: float = 1.0e-6

    @property
    def required_keys(self) -> frozenset[str]:
        return frozenset(
            {
                "teacher_observation",
                "next_teacher_observation",
                "teacher_policy_action",
                "canonical_policy_action",
                "expert_policy_action",
                "expert_source_code",
                "remote_command_valid",
                "remote_sequence_id",
                "remote_source_timestamp",
                "remote_receive_age",
                "expert_confidence",
                "teacher_q_value",
                "teacher_value_valid",
                "teacher_action_valid",
                "student_policy_action",
                "student_action_valid",
                "keyboard_physical_command",
                "teleop_action",
                "applied_teleop_action",
                "applied_actuator_joint_target_rad",
                "processed_arm_command",
                "nullspace_joint_delta",
                "nullspace_task_leakage",
                "nullspace_limit_clipped",
                "teacher_action_compatible",
                "head_rgb",
                "head_depth",
                "head_depth_valid",
                "right_wrist_rgb",
                "right_wrist_depth",
                "right_wrist_depth_valid",
                "contact_force_n",
                "bilateral_contact",
                "stable_grasp",
                "slip_speed_m_s",
                "cube_planar_displacement_m",
                "push_before_grasp_trial",
                "recovery_fail",
                "reward_components",
                "reward",
                "terminated",
                "truncated",
                "success",
                "outcome_code",
                "forbidden_collision",
                "forbidden_collision_valid",
                "timestamp",
                "camera_timestamp",
                "camera_frame_age",
                "torso_joint_position_relative_rad",
                "torso_joint_velocity_rad_s",
                "episode_id",
                "sequence_step",
            }
        )

    def build(
        self,
        *,
        teacher_observation: torch.Tensor,
        next_teacher_observation: torch.Tensor,
        keyboard_physical_command: torch.Tensor,
        teleop_action: torch.Tensor,
        applied_teleop_action: torch.Tensor,
        processed_arm_command: torch.Tensor,
        nullspace_joint_delta: torch.Tensor,
        nullspace_task_leakage: torch.Tensor,
        nullspace_limit_clipped: torch.Tensor,
        head_rgb: torch.Tensor,
        head_depth: torch.Tensor,
        right_wrist_rgb: torch.Tensor,
        right_wrist_depth: torch.Tensor,
        contact_force_n: torch.Tensor,
        stable_grasp: torch.Tensor,
        slip_speed_m_s: torch.Tensor,
        reward_components: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        success: torch.Tensor,
        outcome_code: torch.Tensor,
        timestamp: torch.Tensor,
        camera_timestamp: torch.Tensor,
        camera_frame_age: torch.Tensor,
        torso_joint_position_relative_rad: torch.Tensor,
        torso_joint_velocity_rad_s: torch.Tensor,
        bilateral_contact: torch.Tensor | None = None,
        cube_planar_displacement_m: torch.Tensor | None = None,
        push_before_grasp_trial: torch.Tensor | None = None,
        applied_actuator_joint_target_rad: torch.Tensor | None = None,
        expert_source_code: torch.Tensor | None = None,
        remote_command_valid: torch.Tensor | None = None,
        remote_sequence_id: torch.Tensor | None = None,
        remote_source_timestamp: torch.Tensor | None = None,
        remote_receive_age: torch.Tensor | None = None,
        expert_confidence: torch.Tensor | None = None,
        teacher_policy_label: torch.Tensor | None = None,
        teacher_q_value: torch.Tensor | None = None,
        student_policy_action: torch.Tensor | None = None,
        episode_id: torch.Tensor | None = None,
        sequence_step: torch.Tensor | None = None,
        forbidden_collision: torch.Tensor | None = None,
        forbidden_collision_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Validate and return one batch of transition tensors."""

        teacher_dim = G2TeacherObservationContract().observation_dim
        batch = teacher_observation.shape[0]
        if applied_actuator_joint_target_rad is None:
            applied_actuator_joint_target_rad = torch.zeros(
                (batch, 8), dtype=teleop_action.dtype, device=teleop_action.device
            )
        if expert_source_code is None:
            expert_source_code = torch.full(
                (batch, 1), int(G2ExpertSource.KEYBOARD), dtype=torch.uint8,
                device=teleop_action.device,
            )
        if remote_command_valid is None:
            remote_command_valid = torch.zeros(
                (batch, 1), dtype=torch.uint8, device=teleop_action.device
            )
        if remote_sequence_id is None:
            remote_sequence_id = torch.full(
                (batch, 1), -1, dtype=torch.int64, device=teleop_action.device
            )
        if remote_source_timestamp is None:
            remote_source_timestamp = torch.zeros(
                (batch, 1), dtype=torch.float64, device=teleop_action.device
            )
        if remote_receive_age is None:
            remote_receive_age = torch.zeros(
                (batch, 1), dtype=torch.float32, device=teleop_action.device
            )
        if expert_confidence is None:
            expert_confidence = torch.ones((batch, 1), device=teleop_action.device)
        if teacher_policy_label is None:
            teacher_policy_label = torch.zeros(
                (batch, G2_TEACHER_ACTION_DIM), device=teleop_action.device
            )
            teacher_action_valid = torch.zeros((batch, 1), dtype=torch.uint8, device=teleop_action.device)
        else:
            teacher_action_valid = torch.ones((batch, 1), dtype=torch.uint8, device=teleop_action.device)
        if teacher_q_value is None:
            teacher_q_value = torch.zeros((batch, 1), device=teleop_action.device)
            teacher_value_valid = torch.zeros(
                (batch, 1), dtype=torch.uint8, device=teleop_action.device
            )
        else:
            teacher_value_valid = torch.ones(
                (batch, 1), dtype=torch.uint8, device=teleop_action.device
            )
        if student_policy_action is None:
            student_policy_action = torch.zeros(
                (batch, G2_TEACHER_ACTION_DIM), device=teleop_action.device
            )
            student_action_valid = torch.zeros((batch, 1), dtype=torch.uint8, device=teleop_action.device)
        else:
            student_action_valid = torch.ones((batch, 1), dtype=torch.uint8, device=teleop_action.device)
        if episode_id is None:
            episode_id = torch.zeros((batch, 1), dtype=torch.int64, device=teleop_action.device)
        if sequence_step is None:
            sequence_step = torch.arange(batch, dtype=torch.int64, device=teleop_action.device).unsqueeze(-1)
        if forbidden_collision is None:
            forbidden_collision = torch.zeros(
                (batch, 1), dtype=torch.uint8, device=teleop_action.device
            )
        if forbidden_collision_valid is None:
            forbidden_collision_valid = torch.zeros(
                (batch, 1), dtype=torch.uint8, device=teleop_action.device
            )
        if bilateral_contact is None:
            bilateral_contact = (
                (contact_force_n[:, 0:1] > 1.0)
                & (contact_force_n[:, 1:2] > 1.0)
            )
        if cube_planar_displacement_m is None:
            cube_planar_displacement_m = torch.zeros(
                (batch, 1), dtype=teleop_action.dtype, device=teleop_action.device
            )
        if push_before_grasp_trial is None:
            push_before_grasp_trial = torch.zeros(
                (batch, 1), dtype=torch.bool, device=teleop_action.device
            )
        expected_shapes = {
            "teacher_observation": (batch, teacher_dim),
            "next_teacher_observation": (batch, teacher_dim),
            "keyboard_physical_command": (batch, G2_REDUNDANCY_TELEOP_ACTION_DIM),
            "teleop_action": (batch, G2_REDUNDANCY_TELEOP_ACTION_DIM),
            "applied_teleop_action": (batch, G2_REDUNDANCY_TELEOP_ACTION_DIM),
            "applied_actuator_joint_target_rad": (batch, 8),
            "expert_source_code": (batch, 1),
            "remote_command_valid": (batch, 1),
            "remote_sequence_id": (batch, 1),
            "remote_source_timestamp": (batch, 1),
            "remote_receive_age": (batch, 1),
            "expert_confidence": (batch, 1),
            "teacher_policy_label": (batch, G2_TEACHER_ACTION_DIM),
            "teacher_q_value": (batch, 1),
            "teacher_value_valid": (batch, 1),
            "student_policy_action": (batch, G2_TEACHER_ACTION_DIM),
            "episode_id": (batch, 1),
            "sequence_step": (batch, 1),
            "processed_arm_command": (batch, 7),
            "nullspace_joint_delta": (batch, 7),
            "nullspace_task_leakage": (batch, 1),
            "nullspace_limit_clipped": (batch, 1),
            "head_rgb": (batch, self.height, self.width, 3),
            "head_depth": (batch, self.height, self.width, 1),
            "right_wrist_rgb": (batch, self.height, self.width, 3),
            "right_wrist_depth": (batch, self.height, self.width, 1),
            "contact_force_n": (batch, 2),
            "bilateral_contact": (batch, 1),
            "stable_grasp": (batch, 1),
            "slip_speed_m_s": (batch, 1),
            "cube_planar_displacement_m": (batch, 1),
            "push_before_grasp_trial": (batch, 1),
            "reward": (batch, 1),
            "terminated": (batch, 1),
            "truncated": (batch, 1),
            "success": (batch, 1),
            "outcome_code": (batch, 1),
            "forbidden_collision": (batch, 1),
            "forbidden_collision_valid": (batch, 1),
            "timestamp": (batch, 1),
            "camera_timestamp": (batch, 2),
            "camera_frame_age": (batch, 2),
            "torso_joint_position_relative_rad": (batch, 5),
            "torso_joint_velocity_rad_s": (batch, 5),
        }
        values = locals()
        for name, shape in expected_shapes.items():
            tensor = values[name]
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}; got {tuple(tensor.shape)}")
        for name in ("forbidden_collision", "forbidden_collision_valid"):
            tensor = values[name]
            if not bool(((tensor == 0) | (tensor == 1)).all()):
                raise ValueError(f"{name} must be binary")
        if bool(
            (
                forbidden_collision.to(torch.bool)
                & ~forbidden_collision_valid.to(torch.bool)
            ).any()
        ):
            raise ValueError("forbidden collision cannot be asserted without authority")
        if reward_components.ndim != 2 or reward_components.shape[0] != batch:
            raise ValueError("reward_components must have shape (N, R)")
        if not bool(torch.isfinite(teacher_observation).all()):
            raise ValueError("teacher_observation contains non-finite values")
        if not bool(torch.isfinite(next_teacher_observation).all()):
            raise ValueError("next_teacher_observation contains non-finite values")
        teacher_slices = G2TeacherObservationContract().slices
        for field in ("teacher_observation", "next_teacher_observation"):
            observation = values[field]
            for pose_name in (
                "end_effector_pose_root_xyzw",
                "cube_pose_root_xyzw",
            ):
                pose = observation[:, teacher_slices[pose_name]]
                canonical = canonicalize_quaternion_xyzw(pose[:, 3:7])
                if not bool(
                    torch.allclose(
                        pose[:, 3:7], canonical, atol=1.0e-6, rtol=1.0e-6
                    )
                ):
                    raise ValueError(
                        f"{field} {pose_name} is not canonical normalized XYZW"
                    )

        for name in (
            "teleop_action",
            "applied_teleop_action",
            "keyboard_physical_command",
            "processed_arm_command",
            "nullspace_joint_delta",
            "nullspace_task_leakage",
            "contact_force_n",
            "slip_speed_m_s",
            "cube_planar_displacement_m",
            "reward_components",
            "reward",
            "timestamp",
            "camera_timestamp",
            "camera_frame_age",
            "torso_joint_position_relative_rad",
            "torso_joint_velocity_rad_s",
            "applied_actuator_joint_target_rad",
            "expert_confidence",
            "remote_source_timestamp",
            "remote_receive_age",
            "teacher_policy_label",
            "teacher_q_value",
            "student_policy_action",
        ):
            if not bool(torch.isfinite(values[name]).all()):
                raise ValueError(f"{name} contains non-finite values")
        if head_rgb.dtype != torch.uint8 or right_wrist_rgb.dtype != torch.uint8:
            raise ValueError("RGB tensors must be uint8")

        expected_teleop_action = G2RedundancyKeyboardTeleopContract().normalize(
            keyboard_physical_command
        )
        if not bool(
            torch.allclose(
                teleop_action,
                expected_teleop_action,
                atol=self.teacher_action_compatibility_tolerance,
                rtol=0.0,
            )
        ):
            raise ValueError(
                "teleop_action does not match the canonical physical-command scale"
            )
        if not bool(
            torch.allclose(
                applied_teleop_action[:, :6],
                teleop_action[:, :6],
                atol=self.teacher_action_compatibility_tolerance,
                rtol=0.0,
            )
            and torch.allclose(
                applied_teleop_action[:, 7:8],
                teleop_action[:, 7:8],
                atol=self.teacher_action_compatibility_tolerance,
                rtol=0.0,
            )
        ):
            raise ValueError(
                "applied teleop SE(3)/gripper differs from normalized input"
            )
        elbow_is_identity = torch.isclose(
            applied_teleop_action[:, 6:7],
            teleop_action[:, 6:7],
            atol=self.teacher_action_compatibility_tolerance,
            rtol=0.0,
        )
        elbow_is_disabled = torch.abs(applied_teleop_action[:, 6:7]) <= (
            self.teacher_action_compatibility_tolerance
        )
        if not bool((elbow_is_identity | elbow_is_disabled).all()):
            raise ValueError(
                "applied elbow channel must be unchanged or explicitly disabled"
            )

        head_valid = torch.isfinite(head_depth) & (head_depth > 0.0)
        wrist_valid = torch.isfinite(right_wrist_depth) & (right_wrist_depth > 0.0)
        if not bool(head_valid.any(dim=(1, 2, 3)).all()):
            raise ValueError("head depth has no finite positive pixels")
        if not bool(wrist_valid.any(dim=(1, 2, 3)).all()):
            raise ValueError("right-wrist depth has no finite positive pixels")

        # Dropping elbow_nullspace is only valid when that applied channel is zero.
        teacher_action = torch.cat(
            (applied_teleop_action[:, :6], applied_teleop_action[:, 7:8]), dim=-1
        )
        compatible = (
            torch.abs(applied_teleop_action[:, 6:7])
            <= self.teacher_action_compatibility_tolerance
        )
        source_code = expert_source_code.to(torch.int64)
        source_is_teacher = source_code == int(G2ExpertSource.TEACHER)
        source_is_keyboard = (source_code == int(G2ExpertSource.KEYBOARD)) | (
            source_code == int(G2ExpertSource.REMOTE_KEYBOARD)
        )
        source_is_remote = source_code == int(G2ExpertSource.REMOTE_KEYBOARD)
        valid_source = source_is_keyboard | source_is_teacher
        if not bool(valid_source.all()):
            raise ValueError(
                "expert_source_code must be KEYBOARD, REMOTE_KEYBOARD or TEACHER"
            )
        if bool((source_is_teacher & ~teacher_action_valid.to(torch.bool)).any()):
            raise ValueError("teacher-source rows require a teacher policy label")
        remote_valid = remote_command_valid.to(torch.bool)
        if bool((source_is_remote & ~remote_valid).any()):
            raise ValueError("remote-keyboard rows require remote command provenance")
        if bool((~source_is_remote & remote_valid).any()):
            raise ValueError("only remote-keyboard rows may carry remote command provenance")
        if bool(
            (
                source_is_remote
                & ((remote_sequence_id < 0) | (remote_receive_age < 0.0))
            ).any()
        ):
            raise ValueError("remote command provenance contains invalid values")
        expert_action = torch.where(source_is_teacher, teacher_policy_label, teacher_action)
        teacher_label_in_range = (
            teacher_policy_label.abs() <= 1.0 + 1.0e-6
        ).all(dim=-1, keepdim=True)
        teacher_gripper_is_binary = (
            teacher_policy_label[:, -1:].abs() - 1.0
        ).abs() <= 1.0e-6
        expert_compatible = torch.where(
            source_is_teacher,
            teacher_action_valid.to(torch.bool)
            & teacher_label_in_range
            & teacher_gripper_is_binary,
            compatible,
        )
        recovery_fail = (
            outcome_code.to(torch.int64) == int(G2EpisodeOutcome.RECOVERY_FAILURE)
        )
        result = {
            "teacher_observation": teacher_observation,
            "next_teacher_observation": next_teacher_observation,
            "teacher_policy_action": teacher_action,
            "canonical_policy_action": teacher_action,
            "expert_policy_action": expert_action,
            "expert_source_code": expert_source_code.to(torch.uint8),
            "remote_command_valid": remote_command_valid.to(torch.uint8),
            "remote_sequence_id": remote_sequence_id.to(torch.int64),
            "remote_source_timestamp": remote_source_timestamp.to(torch.float64),
            "remote_receive_age": remote_receive_age,
            "expert_confidence": expert_confidence,
            "teacher_q_value": teacher_q_value,
            "teacher_value_valid": teacher_value_valid,
            "teacher_action_valid": teacher_action_valid,
            "student_policy_action": student_policy_action,
            "student_action_valid": student_action_valid,
            "keyboard_physical_command": keyboard_physical_command,
            "teleop_action": teleop_action,
            "applied_teleop_action": applied_teleop_action,
            "applied_actuator_joint_target_rad": applied_actuator_joint_target_rad,
            "processed_arm_command": processed_arm_command,
            "nullspace_joint_delta": nullspace_joint_delta,
            "nullspace_task_leakage": nullspace_task_leakage,
            "nullspace_limit_clipped": nullspace_limit_clipped.to(torch.uint8),
            "teacher_action_compatible": expert_compatible.to(torch.uint8),
            "head_rgb": head_rgb,
            "head_depth": head_depth,
            "head_depth_valid": head_valid.to(torch.uint8),
            "right_wrist_rgb": right_wrist_rgb,
            "right_wrist_depth": right_wrist_depth,
            "right_wrist_depth_valid": wrist_valid.to(torch.uint8),
            "contact_force_n": contact_force_n,
            "bilateral_contact": bilateral_contact.to(torch.uint8),
            "stable_grasp": stable_grasp.to(torch.uint8),
            "slip_speed_m_s": slip_speed_m_s,
            "cube_planar_displacement_m": cube_planar_displacement_m,
            "push_before_grasp_trial": push_before_grasp_trial.to(torch.uint8),
            "recovery_fail": recovery_fail.to(torch.uint8),
            "reward_components": reward_components,
            "reward": reward,
            "terminated": terminated.to(torch.uint8),
            "truncated": truncated.to(torch.uint8),
            "success": success.to(torch.uint8),
            "outcome_code": outcome_code.to(torch.int16),
            "forbidden_collision": forbidden_collision.to(torch.uint8),
            "forbidden_collision_valid": forbidden_collision_valid.to(torch.uint8),
            "timestamp": timestamp,
            "camera_timestamp": camera_timestamp,
            "camera_frame_age": camera_frame_age,
            "torso_joint_position_relative_rad": torso_joint_position_relative_rad,
            "torso_joint_velocity_rad_s": torso_joint_velocity_rad_s,
            "episode_id": episode_id.to(torch.int64),
            "sequence_step": sequence_step.to(torch.int64),
        }
        if set(result) != set(self.required_keys):
            raise RuntimeError("keyboard teacher transition key mismatch")
        return result

    def validate_episode(self, episode: Mapping[str, torch.Tensor]) -> dict[str, object]:
        missing = sorted(self.required_keys.difference(episode))
        lengths = {
            key: int(value.shape[0])
            for key, value in episode.items()
            if key in self.required_keys and isinstance(value, torch.Tensor) and value.ndim > 0
        }
        unique_lengths = sorted(set(lengths.values()))
        teacher_shape = (
            list(episode["teacher_observation"].shape[1:])
            if "teacher_observation" in episode
            else None
        )
        next_shape = (
            list(episode["next_teacher_observation"].shape[1:])
            if "next_teacher_observation" in episode
            else None
        )
        semantic_errors: list[str] = []
        if not missing and len(unique_lengths) == 1:
            count = unique_lengths[0]
            expected_trailing_shapes = {
                "teacher_observation": (59,),
                "next_teacher_observation": (59,),
                "teacher_policy_action": (7,),
                "canonical_policy_action": (7,),
                "expert_policy_action": (7,),
                "expert_source_code": (1,),
                "remote_command_valid": (1,),
                "remote_sequence_id": (1,),
                "remote_source_timestamp": (1,),
                "remote_receive_age": (1,),
                "expert_confidence": (1,),
                "teacher_q_value": (1,),
                "teacher_value_valid": (1,),
                "teacher_action_valid": (1,),
                "student_policy_action": (7,),
                "student_action_valid": (1,),
                "keyboard_physical_command": (8,),
                "teleop_action": (8,),
                "applied_teleop_action": (8,),
                "applied_actuator_joint_target_rad": (8,),
                "processed_arm_command": (7,),
                "nullspace_joint_delta": (7,),
                "nullspace_task_leakage": (1,),
                "nullspace_limit_clipped": (1,),
                "teacher_action_compatible": (1,),
                "head_rgb": (self.height, self.width, 3),
                "head_depth": (self.height, self.width, 1),
                "head_depth_valid": (self.height, self.width, 1),
                "right_wrist_rgb": (self.height, self.width, 3),
                "right_wrist_depth": (self.height, self.width, 1),
                "right_wrist_depth_valid": (self.height, self.width, 1),
                "contact_force_n": (2,),
                "bilateral_contact": (1,),
                "stable_grasp": (1,),
                "slip_speed_m_s": (1,),
                "cube_planar_displacement_m": (1,),
                "push_before_grasp_trial": (1,),
                "recovery_fail": (1,),
                "reward": (1,),
                "terminated": (1,),
                "truncated": (1,),
                "success": (1,),
                "outcome_code": (1,),
                "forbidden_collision": (1,),
                "forbidden_collision_valid": (1,),
                "timestamp": (1,),
                "camera_timestamp": (2,),
                "camera_frame_age": (2,),
                "torso_joint_position_relative_rad": (5,),
                "torso_joint_velocity_rad_s": (5,),
                "episode_id": (1,),
                "sequence_step": (1,),
            }
            for name, trailing in expected_trailing_shapes.items():
                value = episode[name]
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != (
                    count,
                    *trailing,
                ):
                    semantic_errors.append(f"shape:{name}")
            expected_dtypes = {
                "expert_source_code": torch.uint8,
                "remote_command_valid": torch.uint8,
                "teacher_value_valid": torch.uint8,
                "teacher_action_valid": torch.uint8,
                "student_action_valid": torch.uint8,
                "nullspace_limit_clipped": torch.uint8,
                "teacher_action_compatible": torch.uint8,
                "head_rgb": torch.uint8,
                "head_depth_valid": torch.uint8,
                "right_wrist_rgb": torch.uint8,
                "right_wrist_depth_valid": torch.uint8,
                "bilateral_contact": torch.uint8,
                "stable_grasp": torch.uint8,
                "push_before_grasp_trial": torch.uint8,
                "recovery_fail": torch.uint8,
                "terminated": torch.uint8,
                "truncated": torch.uint8,
                "success": torch.uint8,
                "forbidden_collision": torch.uint8,
                "forbidden_collision_valid": torch.uint8,
                "outcome_code": torch.int16,
                "remote_sequence_id": torch.int64,
                "episode_id": torch.int64,
                "sequence_step": torch.int64,
                "remote_source_timestamp": torch.float64,
            }
            for name, dtype in expected_dtypes.items():
                if episode[name].dtype != dtype:
                    semantic_errors.append(f"dtype:{name}")
            binary_fields = (
                "remote_command_valid", "teacher_value_valid",
                "teacher_action_valid", "student_action_valid",
                "nullspace_limit_clipped", "teacher_action_compatible",
                "head_depth_valid", "right_wrist_depth_valid",
                "bilateral_contact", "stable_grasp", "push_before_grasp_trial",
                "recovery_fail", "terminated", "truncated", "success",
                "forbidden_collision", "forbidden_collision_valid",
            )
            for name in binary_fields:
                if bool(((episode[name] != 0) & (episode[name] != 1)).any()):
                    semantic_errors.append(f"nonbinary:{name}")
            if episode["reward_components"].ndim != 2:
                semantic_errors.append("shape:reward_components")
            if episode["head_rgb"].dtype != torch.uint8 or episode[
                "right_wrist_rgb"
            ].dtype != torch.uint8:
                semantic_errors.append("rgb_dtype")
            raw_depth_names = {"head_depth", "right_wrist_depth"}
            for name in self.required_keys.difference(raw_depth_names):
                value = episode[name]
                if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
                    semantic_errors.append(f"nonfinite:{name}")
            for prefix in ("head", "right_wrist"):
                depth = episode[f"{prefix}_depth"]
                stored_valid = episode[f"{prefix}_depth_valid"].to(torch.bool)
                actual_valid = torch.isfinite(depth) & (depth > 0.0)
                if not torch.equal(stored_valid, actual_valid):
                    semantic_errors.append(f"depth_valid_mismatch:{prefix}")
                if not bool(actual_valid.reshape(count, -1).any(dim=1).all()):
                    semantic_errors.append(f"no_valid_depth:{prefix}")
            teacher_slices = G2TeacherObservationContract().slices
            for field in ("teacher_observation", "next_teacher_observation"):
                for pose_name in (
                    "end_effector_pose_root_xyzw",
                    "cube_pose_root_xyzw",
                ):
                    pose = episode[field][:, teacher_slices[pose_name]][:, 3:7]
                    try:
                        canonical = canonicalize_quaternion_xyzw(pose)
                    except ValueError:
                        semantic_errors.append(f"invalid_quaternion:{field}:{pose_name}")
                        continue
                    if not bool(
                        torch.allclose(pose, canonical, atol=1.0e-6, rtol=1.0e-6)
                    ):
                        semantic_errors.append(
                            f"noncanonical_quaternion:{field}:{pose_name}"
                        )
            source = episode["expert_source_code"].to(torch.int64)
            valid_source = (
                (source == int(G2ExpertSource.KEYBOARD))
                | (source == int(G2ExpertSource.TEACHER))
                | (source == int(G2ExpertSource.REMOTE_KEYBOARD))
            )
            if not bool(valid_source.all()):
                semantic_errors.append("invalid_expert_source")
            source_is_remote = source == int(G2ExpertSource.REMOTE_KEYBOARD)
            remote_valid = episode["remote_command_valid"].to(torch.bool)
            if not torch.equal(source_is_remote, remote_valid):
                semantic_errors.append("remote_provenance_source_mismatch")
            if bool(
                (
                    remote_valid
                    & (
                        (episode["remote_sequence_id"] < 0)
                        | (episode["remote_receive_age"] < 0.0)
                    )
                ).any()
            ):
                semantic_errors.append("invalid_remote_provenance")
            remote_rows = remote_valid.reshape(-1)
            if int(remote_rows.sum()) > 1:
                sequence = episode["remote_sequence_id"].reshape(-1)[remote_rows]
                source_time = episode["remote_source_timestamp"].reshape(-1)[remote_rows]
                # One accepted packet is intentionally held until the next
                # heartbeat. Repeated rows may therefore share its sequence id;
                # the adapter itself rejects duplicate submitted packets.
                if not bool((sequence[1:] >= sequence[:-1]).all()):
                    semantic_errors.append("nonmonotonic_remote_sequence")
                if not bool((source_time[1:] >= source_time[:-1]).all()):
                    semantic_errors.append("nonmonotonic_remote_source_timestamp")
            episode_ids = episode["episode_id"].reshape(-1)
            if episode_ids.numel() and not bool((episode_ids == episode_ids[0]).all()):
                semantic_errors.append("mixed_episode_ids")
            steps = episode["sequence_step"].reshape(-1)
            if steps.numel() and int(steps[0]) != 0:
                semantic_errors.append("sequence_does_not_start_at_zero")
            if steps.numel() > 1 and not bool((steps[1:] == steps[:-1] + 1).all()):
                semantic_errors.append("noncontiguous_sequence_steps")
            timestamp = episode["timestamp"].reshape(-1)
            if timestamp.numel() > 1 and not bool((timestamp[1:] >= timestamp[:-1]).all()):
                semantic_errors.append("nonmonotonic_timestamp")
            if not bool((episode["camera_frame_age"] >= 0.0).all()):
                semantic_errors.append("negative_camera_age")
            expected_age = episode["timestamp"] - episode["camera_timestamp"]
            if not bool(
                torch.allclose(
                    expected_age,
                    episode["camera_frame_age"],
                    atol=1.0e-5,
                    rtol=1.0e-5,
                )
            ):
                semantic_errors.append("camera_time_age_mismatch")
            if bool(
                (
                    episode["terminated"].to(torch.bool)
                    & episode["truncated"].to(torch.bool)
                ).any()
            ):
                semantic_errors.append("terminated_and_truncated")
            component_sum = episode["reward_components"].sum(dim=-1, keepdim=True)
            if not bool(torch.allclose(component_sum, episode["reward"], atol=1.0e-5, rtol=1.0e-5)):
                semantic_errors.append("reward_component_sum_mismatch")
            success = episode["success"].to(torch.bool).reshape(-1)
            outcome = episode["outcome_code"].to(torch.int64).reshape(-1)
            forbidden_collision = episode["forbidden_collision"].to(torch.bool).reshape(-1)
            forbidden_collision_valid = episode["forbidden_collision_valid"].to(torch.bool).reshape(-1)
            if bool((forbidden_collision & ~forbidden_collision_valid).any()):
                semantic_errors.append("forbidden_collision_without_live_authority")
            collision_outcome = outcome == int(G2EpisodeOutcome.FORBIDDEN_COLLISION)
            if not torch.equal(collision_outcome, forbidden_collision):
                semantic_errors.append("forbidden_collision_outcome_mismatch")
            if bool((success & forbidden_collision).any()):
                semantic_errors.append("success_with_forbidden_collision")
            if not torch.equal(success, outcome == int(G2EpisodeOutcome.SUCCESS)):
                semantic_errors.append("success_outcome_mismatch")
            expected_recovery_fail = (
                outcome == int(G2EpisodeOutcome.RECOVERY_FAILURE)
            )
            if not torch.equal(
                episode["recovery_fail"].to(torch.bool).reshape(-1),
                expected_recovery_fail,
            ):
                semantic_errors.append("recovery_fail_outcome_mismatch")
            expected_bilateral = (
                (episode["contact_force_n"][:, 0] > 1.0)
                & (episode["contact_force_n"][:, 1] > 1.0)
            )
            if not torch.equal(
                episode["bilateral_contact"].to(torch.bool).reshape(-1),
                expected_bilateral,
            ):
                semantic_errors.append("bilateral_contact_force_mismatch")
            expected_teacher_action = torch.cat(
                (
                    episode["applied_teleop_action"][:, :6],
                    episode["applied_teleop_action"][:, 7:8],
                ),
                dim=-1,
            )
            if not torch.allclose(
                episode["teacher_policy_action"], expected_teacher_action,
                atol=self.teacher_action_compatibility_tolerance, rtol=0.0,
            ):
                semantic_errors.append("teacher_action_applied_request_mismatch")
            if not torch.equal(
                episode["canonical_policy_action"],
                episode["teacher_policy_action"],
            ):
                semantic_errors.append("canonical_teacher_action_mismatch")
            # For keyboard rows the expert target must be the exact action
            # applied to the common 7-D policy surface.  Teacher-source labels
            # are separately identified by teacher_action_valid and may differ
            # from the executed controller request by design.
            teacher_rows = source.reshape(-1) == int(G2ExpertSource.TEACHER)
            keyboard_rows = ~teacher_rows
            if not torch.allclose(
                episode["expert_policy_action"][keyboard_rows],
                expected_teacher_action[keyboard_rows],
                atol=self.teacher_action_compatibility_tolerance, rtol=0.0,
            ):
                semantic_errors.append("expert_action_source_mismatch")
            confidence = episode["expert_confidence"]
            if bool(((confidence < 0.0) | (confidence > 1.0)).any()):
                semantic_errors.append("expert_confidence_out_of_range")
            for action_name in (
                "teleop_action", "applied_teleop_action",
                "teacher_policy_action", "canonical_policy_action",
            ):
                if bool((episode[action_name].abs() > 1.0 + 1.0e-6).any()):
                    semantic_errors.append(f"normalized_action_out_of_range:{action_name}")
            expected_teleop = G2RedundancyKeyboardTeleopContract().normalize(
                episode["keyboard_physical_command"]
            )
            if not torch.allclose(
                episode["teleop_action"], expected_teleop,
                atol=self.teacher_action_compatibility_tolerance, rtol=0.0,
            ):
                semantic_errors.append("physical_to_normalized_scale_mismatch")
            if not torch.allclose(
                episode["applied_teleop_action"][:, :6],
                episode["teleop_action"][:, :6],
                atol=self.teacher_action_compatibility_tolerance, rtol=0.0,
            ) or not torch.allclose(
                episode["applied_teleop_action"][:, 7:8],
                episode["teleop_action"][:, 7:8],
                atol=self.teacher_action_compatibility_tolerance, rtol=0.0,
            ):
                semantic_errors.append("applied_se3_gripper_scale_mismatch")
            applied_elbow = episode["applied_teleop_action"][:, 6:7]
            input_elbow = episode["teleop_action"][:, 6:7]
            elbow_valid = torch.isclose(
                applied_elbow, input_elbow,
                atol=self.teacher_action_compatibility_tolerance, rtol=0.0,
            ) | applied_elbow.abs().le(self.teacher_action_compatibility_tolerance)
            if not bool(elbow_valid.all()):
                semantic_errors.append("applied_elbow_not_identity_or_disabled")
            for action_name, valid_name in (
                ("expert_policy_action", "teacher_action_compatible"),
                ("student_policy_action", "student_action_valid"),
            ):
                valid_rows = episode[valid_name].to(torch.bool).reshape(-1)
                if bool(
                    (episode[action_name][valid_rows].abs() > 1.0 + 1.0e-6).any()
                ):
                    semantic_errors.append(
                        f"normalized_action_out_of_range:{action_name}"
                    )
            if bool(
                (
                    episode["applied_teleop_action"][keyboard_rows, 7].abs() - 1.0
                ).abs().gt(1.0e-6).any()
            ):
                semantic_errors.append("keyboard_gripper_not_binary")
            terminal = (
                episode["terminated"].to(torch.bool)
                | episode["truncated"].to(torch.bool)
            ).reshape(-1)
            if terminal.numel() and (
                bool(terminal[:-1].any()) or not bool(terminal[-1])
            ):
                semantic_errors.append("terminal_not_final_row")
            if terminal.numel() and bool(
                (outcome[:-1] != int(G2EpisodeOutcome.ONGOING)).any()
            ):
                semantic_errors.append("outcome_before_final_row")
            expected_compatible = torch.where(
                source == int(G2ExpertSource.TEACHER),
                episode["teacher_action_valid"].to(torch.bool)
                & (episode["expert_policy_action"].abs() <= 1.0 + 1.0e-6).all(
                    dim=-1, keepdim=True
                )
                & (
                    episode["expert_policy_action"][:, -1:].abs() - 1.0
                ).abs().le(1.0e-6),
                torch.abs(episode["applied_teleop_action"][:, 6:7])
                <= self.teacher_action_compatibility_tolerance,
            )
            if not torch.equal(
                episode["teacher_action_compatible"].to(torch.bool),
                expected_compatible,
            ):
                semantic_errors.append("teacher_action_compatibility_mismatch")
        return {
            "pass": (
                not missing
                and len(unique_lengths) == 1
                and teacher_shape == [59]
                and next_shape == [59]
                and not semantic_errors
            ),
            "missing_keys": missing,
            "lengths": lengths,
            "teacher_observation_shape": teacher_shape,
            "next_teacher_observation_shape": next_shape,
            "semantic_errors": semantic_errors,
        }

    def validate_hdf_episode(
        self, episode: Mapping[str, torch.Tensor]
    ) -> dict[str, object]:
        """Validate canonical tensors plus Isaac Lab's public ``actions`` key."""

        validation = self.validate_episode(episode)
        semantic_errors = list(validation["semantic_errors"])
        missing_public_keys = sorted(
            G2_KEYBOARD_TEACHER_PUBLIC_HDF_KEYS.difference(episode)
        )
        if not missing_public_keys and "applied_teleop_action" in episode:
            actions = episode["actions"]
            applied = episode["applied_teleop_action"]
            if not isinstance(actions, torch.Tensor) or actions.shape != applied.shape:
                semantic_errors.append("shape:actions")
            elif not bool(torch.isfinite(actions).all()):
                semantic_errors.append("nonfinite:actions")
            elif not torch.equal(actions, applied):
                semantic_errors.append("public_actions_applied_request_mismatch")
        result = dict(validation)
        result["missing_public_hdf_keys"] = missing_public_keys
        result["semantic_errors"] = semantic_errors
        result["pass"] = bool(
            validation["pass"] and not missing_public_keys and not semantic_errors
        )
        return result

    def extract_training_views(
        self, episode: Mapping[str, torch.Tensor]
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Expose unambiguous teacher-RL, teacher-BC and RGB-D student views.

        Rows with a non-zero redundancy command are deliberately masked out
        of seven-dimensional behavior cloning instead of silently dropping
        the elbow channel.
        """

        validation = self.validate_episode(episode)
        if not validation["pass"]:
            raise ValueError(f"invalid keyboard teacher episode: {validation}")
        done = episode["terminated"].to(torch.bool) | episode["truncated"].to(torch.bool)
        bc_mask = episode["teacher_action_compatible"].to(torch.bool).reshape(-1)
        teacher_slices = G2TeacherObservationContract().slices
        deployable_proprioception = G2StudentObservationContract().build(
            arm_hand_joint_position_relative_rad=episode["teacher_observation"][
                :, teacher_slices["controlled_joint_position_relative_rad"]
            ],
            arm_hand_joint_velocity_rad_s=episode["teacher_observation"][
                :, teacher_slices["controlled_joint_velocity_rad_s"]
            ],
            torso_joint_position_relative_rad=episode["torso_joint_position_relative_rad"],
            torso_joint_velocity_rad_s=episode["torso_joint_velocity_rad_s"],
            end_effector_pose_root_xyzw=episode["teacher_observation"][
                :, teacher_slices["end_effector_pose_root_xyzw"]
            ],
            goal_position_root_m=episode["teacher_observation"][
                :, teacher_slices["goal_position_root_m"]
            ],
            previous_action=episode["teacher_observation"][:, teacher_slices["previous_action"]],
            camera_frame_age_s=episode["camera_frame_age"],
        )
        contact_target = episode["teacher_observation"][
            :, teacher_slices["bilateral_contact_features"]
        ].clamp(0.0, 1.0)
        phase = episode["teacher_observation"][:, teacher_slices["curriculum_phase_features"]]
        failure_class_target = torch.zeros(
            phase.shape[0], dtype=torch.long, device=phase.device
        )
        failure_class_target = torch.where(phase[:, 1] > 0.5, 1, failure_class_target)
        slipping = (phase[:, 1] > 0.5) & (episode["slip_speed_m_s"].reshape(-1) >= 0.06)
        failure_class_target = torch.where(slipping, 3, failure_class_target)
        failure_class_target = torch.where(
            episode["stable_grasp"].to(torch.bool).reshape(-1),
            2,
            failure_class_target,
        )
        failure_class_target = torch.where(phase[:, 2] > 0.5, 5, failure_class_target)
        failure_class_target = torch.where(phase[:, 3] > 0.5, 6, failure_class_target)
        collision = episode["forbidden_collision"].to(torch.bool).reshape(-1)
        failure_class_target = torch.where(collision, 4, failure_class_target)
        failure_class_valid = episode["forbidden_collision_valid"].to(torch.bool).reshape(-1)
        depth_validity_target = torch.stack(
            (
                episode["head_depth_valid"].float().mean(dim=(1, 2, 3)),
                episode["right_wrist_depth_valid"].float().mean(dim=(1, 2, 3)),
            ),
            dim=-1,
        )
        return {
            "teacher_rl": {
                "observation": episode["teacher_observation"],
                "action": episode["canonical_policy_action"],
                "reward": episode["reward"],
                "next_observation": episode["next_teacher_observation"],
                "done": done,
                "forbidden_collision": episode["forbidden_collision"],
                "forbidden_collision_valid": episode["forbidden_collision_valid"],
                "safe_for_her_force": (
                    episode["forbidden_collision_valid"].to(torch.bool)
                    & ~episode["forbidden_collision"].to(torch.bool)
                ),
            },
            "teacher_bc": {
                "observation": episode["teacher_observation"][bc_mask],
                "action": episode["expert_policy_action"][bc_mask],
                "expert_source_code": episode["expert_source_code"][bc_mask],
                "expert_confidence": episode["expert_confidence"][bc_mask],
                "teacher_q_value": episode["teacher_q_value"][bc_mask],
                "teacher_value_valid": episode["teacher_value_valid"][bc_mask],
                "teacher_action_valid": episode["teacher_action_valid"][bc_mask],
                "compatible_mask": bc_mask,
            },
            "vision_student": {
                "head_rgb": episode["head_rgb"],
                "head_depth": episode["head_depth"],
                "head_depth_valid": episode["head_depth_valid"],
                "right_wrist_rgb": episode["right_wrist_rgb"],
                "right_wrist_depth": episode["right_wrist_depth"],
                "right_wrist_depth_valid": episode["right_wrist_depth_valid"],
                "deployable_proprioception": deployable_proprioception,
                "camera_timestamp": episode["camera_timestamp"],
                "camera_frame_age": episode["camera_frame_age"],
                "relative_pose_target": relative_pose_target_from_teacher_state(
                    episode["teacher_observation"]
                ),
                "contact_target": contact_target,
                # Preserve the two independently measured finger forces for
                # offline force-aware demonstration sampling.  These values
                # are critic/sampler metadata and never enter the deployable
                # actor observation.
                "contact_force_target_n": episode["contact_force_n"],
                "bilateral_contact_target": episode["bilateral_contact"].float(),
                "stable_grasp_target": episode["stable_grasp"].float(),
                "slip_speed_target_m_s": episode["slip_speed_m_s"],
                "push_before_grasp_trial": episode["push_before_grasp_trial"],
                "recovery_fail": episode["recovery_fail"],
                "depth_validity_target": depth_validity_target,
                "failure_class_target": failure_class_target,
                "failure_class_valid": failure_class_valid,
                "failure_class_semantics": torch.tensor(
                    [0, 1, 2, 3, 4, 5, 6], dtype=torch.long, device=phase.device
                ),
                "teacher_state_target": episode["teacher_observation"],
                "teacher_action_target": episode["teacher_policy_action"],
                "expert_action_target": episode["expert_policy_action"],
                "expert_source_code": episode["expert_source_code"],
                "remote_command_valid": episode["remote_command_valid"],
                "remote_sequence_id": episode["remote_sequence_id"],
                "remote_source_timestamp": episode["remote_source_timestamp"],
                "remote_receive_age": episode["remote_receive_age"],
                "expert_confidence": episode["expert_confidence"],
                "teacher_q_value": episode["teacher_q_value"],
                "teacher_value_valid": episode["teacher_value_valid"],
                "student_policy_action": episode["student_policy_action"],
                "episode_id": episode["episode_id"],
                "sequence_step": episode["sequence_step"],
                "teacher_action_compatible": episode["teacher_action_compatible"],
                "success": episode["success"],
                "outcome_code": episode["outcome_code"],
                "forbidden_collision": episode["forbidden_collision"],
                "forbidden_collision_valid": episode["forbidden_collision_valid"],
                "safe_for_her_force": (
                    episode["forbidden_collision_valid"].to(torch.bool)
                    & ~episode["forbidden_collision"].to(torch.bool)
                ),
            },
            "redundancy_control": {
                "keyboard_action": episode["teleop_action"],
                "applied_keyboard_action": episode["applied_teleop_action"],
                "applied_actuator_joint_target_rad": episode["applied_actuator_joint_target_rad"],
                "processed_arm_command": episode["processed_arm_command"],
                "nullspace_joint_delta": episode["nullspace_joint_delta"],
                "nullspace_task_leakage": episode["nullspace_task_leakage"],
            },
        }


@dataclass(frozen=True)
class G2StudentSequenceContract:
    """Create padded recurrent batches without crossing episode boundaries."""

    sequence_length: int = 48
    burn_in_steps: int = 12
    stride: int = 36

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if not 0 <= self.burn_in_steps < self.sequence_length:
            raise ValueError("burn_in_steps must be in [0, sequence_length)")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

    @staticmethod
    def _pad(value: torch.Tensor, start: int, stop: int, length: int) -> torch.Tensor:
        selected = value[start:stop]
        if selected.shape[0] == length:
            return selected
        padding = torch.zeros(
            (length - selected.shape[0], *selected.shape[1:]),
            dtype=selected.dtype,
            device=selected.device,
        )
        return torch.cat((selected, padding), dim=0)

    def build(
        self,
        episode: Mapping[str, torch.Tensor],
        *,
        transition_contract: G2KeyboardTeacherTransitionContract | None = None,
    ) -> dict[str, torch.Tensor | int]:
        contract = transition_contract or G2KeyboardTeacherTransitionContract()
        views = contract.extract_training_views(episode)
        student = views["vision_student"]
        count = int(episode["episode_id"].shape[0])
        if count <= 0:
            raise ValueError("cannot create a sequence from an empty episode")
        compact_rgbd = pack_rgbd(
            student["head_rgb"],
            student["head_depth"],
            student["right_wrist_rgb"],
            student["right_wrist_depth"],
        )
        starts = list(range(0, count, self.stride))
        tensor_names = (
            "head_rgb", "head_depth", "head_depth_valid",
            "right_wrist_rgb", "right_wrist_depth", "right_wrist_depth_valid",
            "deployable_proprioception", "expert_action_target",
            "teacher_state_target", "success",
            "expert_confidence", "teacher_q_value", "teacher_value_valid", "relative_pose_target",
            "remote_command_valid", "remote_sequence_id", "remote_source_timestamp",
            "remote_receive_age",
            "contact_target", "stable_grasp_target", "slip_speed_target_m_s",
            "contact_force_target_n",
            "bilateral_contact_target", "push_before_grasp_trial", "recovery_fail",
            "failure_class_target", "failure_class_valid", "forbidden_collision",
            "forbidden_collision_valid", "safe_for_her_force",
            "episode_id", "sequence_step", "camera_timestamp", "camera_frame_age",
        )
        result: dict[str, torch.Tensor | int] = {}
        for name in tensor_names:
            value = student[name]
            result[name] = torch.stack(
                [
                    self._pad(value, start, min(start + self.sequence_length, count), self.sequence_length)
                    for start in starts
                ],
                dim=0,
            )
        result["rgbd_u8"] = torch.stack(
            [
                self._pad(
                    compact_rgbd,
                    start,
                    min(start + self.sequence_length, count),
                    self.sequence_length,
                )
                for start in starts
            ],
            dim=0,
        )
        # Supervise exactly the validity mask seen by the network after
        # nearest-neighbour compaction, not a full-resolution ratio that can
        # differ at sparse depth boundaries.
        compact_depth_validity = compact_rgbd[:, :, 5].float().div(255.0).mean(
            dim=(-1, -2)
        )
        result["depth_validity_target"] = torch.stack(
            [
                self._pad(
                    compact_depth_validity,
                    start,
                    min(start + self.sequence_length, count),
                    self.sequence_length,
                )
                for start in starts
            ],
            dim=0,
        )
        result["episode_id"] = result["episode_id"].squeeze(-1)
        result["sequence_step"] = result["sequence_step"].squeeze(-1)
        lengths = torch.tensor(
            [min(self.sequence_length, count - start) for start in starts],
            dtype=torch.long,
            device=episode["episode_id"].device,
        )
        step_axis = torch.arange(
            self.sequence_length, device=lengths.device
        ).unsqueeze(0)
        padding_mask = step_axis < lengths.unsqueeze(1)
        hidden_reset_mask = torch.zeros_like(padding_mask)
        hidden_reset_mask[:, 0] = True
        # Incompatible keyboard redundancy rows remain in the audit dataset,
        # but cannot contribute to canonical teacher/student imitation.
        compatibility = torch.stack(
            [
                self._pad(
                    student["teacher_action_compatible"],
                    start,
                    min(start + self.sequence_length, count),
                    self.sequence_length,
                )
                for start in starts
            ],
            dim=0,
        ).to(torch.bool)
        result["expert_confidence"] = (
            result["expert_confidence"]
            * compatibility.to(result["expert_confidence"].dtype)
            # Never clone a command on a row whose full-body collision
            # authority is missing or reports a forbidden collision.  Such
            # rows remain available for failure/critic supervision.
            * result["forbidden_collision_valid"].to(
                result["expert_confidence"].dtype
            )
            * (~result["forbidden_collision"].to(torch.bool)).to(
                result["expert_confidence"].dtype
            )
        )
        result.update(
            {
                "padding_mask": padding_mask,
                "hidden_reset_mask": hidden_reset_mask,
                "sequence_lengths": lengths,
                "burn_in_steps": self.burn_in_steps,
            }
        )
        return result


def classify_episode_outcome(
    *,
    success: bool,
    emergency_stop: bool = False,
    forbidden_collision: bool = False,
    camera_failure: bool = False,
    object_dropped: bool = False,
    recovery_failed: bool = False,
    manual_failure: bool = False,
    half_success: bool = False,
    timed_out: bool = False,
) -> G2EpisodeOutcome:
    """Apply fail-closed outcome precedence; success cannot hide safety failures."""

    if emergency_stop:
        return G2EpisodeOutcome.EMERGENCY_STOP
    if forbidden_collision:
        return G2EpisodeOutcome.FORBIDDEN_COLLISION
    if camera_failure:
        return G2EpisodeOutcome.CAMERA_FAILURE
    if object_dropped:
        return G2EpisodeOutcome.OBJECT_DROP
    if recovery_failed:
        return G2EpisodeOutcome.RECOVERY_FAILURE
    if manual_failure:
        return G2EpisodeOutcome.MANUAL_FAILURE
    if success:
        return G2EpisodeOutcome.SUCCESS
    if half_success:
        return G2EpisodeOutcome.HALF_SUCCESS
    if timed_out:
        return G2EpisodeOutcome.TIMEOUT
    return G2EpisodeOutcome.ONGOING


def classify_operator_saved_outcome(
    *,
    ever_stably_grasped_and_lifted: bool,
    ever_stable_grasp: bool,
    ever_lifted: bool,
) -> G2EpisodeOutcome:
    """Classify an explicitly saved teleop episode without inventing success.

    ``SUCCESS`` is reserved for the runtime physical predicate in which a
    stable grasp and lift are true together.  A stable grasp before lift, or a
    lift whose grasp was subsequently lost, remains useful supervision but is
    only ``HALF_SUCCESS``.
    """

    if ever_stably_grasped_and_lifted:
        return G2EpisodeOutcome.SUCCESS
    if ever_stable_grasp or ever_lifted:
        return G2EpisodeOutcome.HALF_SUCCESS
    return G2EpisodeOutcome.MANUAL_FAILURE


__all__ = [
    "G2_CANONICAL_POLICY_ACTION_LABELS",
    "G2_CONTROLLED_JOINT_LABELS",
    "G2ExpertSource",
    "G2EpisodeOutcome",
    "G2_KEYBOARD_NORMALIZED_ACTION_LABELS",
    "G2_KEYBOARD_PHYSICAL_ACTION_LABELS",
    "G2_KEYBOARD_TEACHER_PUBLIC_HDF_KEYS",
    "G2KeyboardTeacherTransitionContract",
    "G2StudentSequenceContract",
    "classify_operator_saved_outcome",
    "keyboard_teacher_dataset_metadata",
    "teacher_compatible_keyboard_controller_action",
    "G2_KEYBOARD_TEACHER_DATASET_SCHEMA",
    "classify_episode_outcome",
]
