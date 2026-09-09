"""Shared runtime builder for the exact Milestone 7 teacher observation."""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

from .g2_lift_methodology import (
    G2LiftSandboxContract,
    RIGHT_ARM_JOINTS,
    RIGHT_EE_BODY,
    RIGHT_GRIPPER_MASTER,
)
from .g2_quaternion import (
    isaaclab_native_quaternion_order,
    quaternion_native_to_xyzw,
)
from .g2_teacher_sac import G2_TEACHER_ACTION_DIM, G2TeacherObservationContract
from .g2_lift_task_mdp import object_stably_grasped_and_lifted
from .g2_redundancy_teleop import tensor_value


class G2TeacherRuntimeObservationBuilder:
    """Build the same 59-D privileged input in SAC and keyboard collection."""

    def __init__(self, env) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        self.cube = env.scene["object"]
        self.inner_contact = env.scene["right_inner_finger_contact"]
        self.outer_contact = env.scene["right_outer_finger_contact"]
        controlled_names = (*RIGHT_ARM_JOINTS, RIGHT_GRIPPER_MASTER)
        self.joint_indices = torch.tensor(
            [self.robot.joint_names.index(name) for name in controlled_names],
            device=env.device,
            dtype=torch.long,
        )
        self.ee_index = self.robot.body_names.index(RIGHT_EE_BODY)
        self.contract = G2TeacherObservationContract()
        self.sandbox = G2LiftSandboxContract().validated()
        self.native_quaternion_order = isaaclab_native_quaternion_order()

    def build(self, previous_teacher_action: torch.Tensor) -> torch.Tensor:
        joint_position = tensor_value(self.robot.data.joint_pos)
        batch = joint_position.shape[0]
        if previous_teacher_action.shape != (batch, G2_TEACHER_ACTION_DIM):
            raise ValueError(
                "previous_teacher_action must have shape "
                f"{(batch, G2_TEACHER_ACTION_DIM)}"
            )
        q = joint_position.index_select(1, self.joint_indices)
        q_default = tensor_value(self.robot.data.default_joint_pos).index_select(
            1, self.joint_indices
        )
        qd = tensor_value(self.robot.data.joint_vel).index_select(1, self.joint_indices)
        root_position = tensor_value(self.robot.data.root_pos_w)
        root_quaternion = tensor_value(self.robot.data.root_quat_w)
        # Match the authority used by the official Lift MDP.  The frame sensor
        # also makes the source/target frame convention explicit at the USD
        # boundary, while body velocity remains the slip authority below.
        ee_frame = self.env.scene["ee_frame"].data
        ee_position, ee_quaternion = subtract_frame_transforms(
            root_position,
            root_quaternion,
            tensor_value(ee_frame.target_pos_w)[:, 0],
            tensor_value(ee_frame.target_quat_w)[:, 0],
        )
        ee_pose = torch.cat(
            (
                ee_position,
                quaternion_native_to_xyzw(
                    ee_quaternion, self.native_quaternion_order
                ),
            ),
            dim=-1,
        )
        cube_position, cube_quaternion = subtract_frame_transforms(
            root_position,
            root_quaternion,
            tensor_value(self.cube.data.root_pos_w),
            tensor_value(self.cube.data.root_quat_w),
        )
        cube_pose = torch.cat(
            (
                cube_position,
                quaternion_native_to_xyzw(
                    cube_quaternion, self.native_quaternion_order
                ),
            ),
            dim=-1,
        )
        # UniformPoseCommand is already expressed in the robot root frame.
        # Subtracting a world root position a second time corrupted vector-env
        # goals and W&B distances for every env except env_0.
        goal = self.env.command_manager.get_command("object_pose")[:, :3]
        inner_force = torch.linalg.vector_norm(
            tensor_value(self.inner_contact.data.force_matrix_w)[:, 0, 0], dim=-1
        )
        outer_force = torch.linalg.vector_norm(
            tensor_value(self.outer_contact.data.force_matrix_w)[:, 0, 0], dim=-1
        )
        contact_force = torch.stack((inner_force, outer_force), dim=-1)
        bilateral = (inner_force > 1.0) & (outer_force > 1.0)
        lifted = tensor_value(self.cube.data.root_pos_w)[:, 2] >= (
            self.sandbox.table_surface_height_m + self.sandbox.lift_height_above_table_m
        )
        success = object_stably_grasped_and_lifted(self.env)
        phase = torch.stack((~bilateral, bilateral, lifted, success), dim=-1).float()
        return self.contract.build(
            joint_position_relative_rad=q - q_default,
            joint_velocity_rad_s=qd,
            end_effector_pose_root_xyzw=ee_pose,
            cube_pose_root_xyzw=cube_pose,
            cube_linear_angular_velocity=torch.cat(
                (
                    quat_apply_inverse(root_quaternion, tensor_value(self.cube.data.root_lin_vel_w)),
                    quat_apply_inverse(root_quaternion, tensor_value(self.cube.data.root_ang_vel_w)),
                ),
                dim=-1,
            ),
            goal_position_root_m=goal,
            bilateral_contact_force_n=contact_force,
            curriculum_phase_features=phase,
            previous_action=previous_teacher_action,
        )


__all__ = ["G2TeacherRuntimeObservationBuilder"]
