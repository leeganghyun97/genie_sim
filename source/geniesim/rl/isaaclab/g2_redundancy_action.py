"""Isaac Lab action/device adapters for G2 7-DoF keyboard teleoperation."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch

from isaaclab.devices import Se3Keyboard
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass

from .g2_redundancy_teleop import (
    G2ElbowKeyState,
    damped_nullspace_project,
    ee_rotation_only_mask,
    tensor_value,
    wrist_only_rotation_jacobian,
)
from .g2_teleop_dataset import (
    G2RedundancyKeyboardTeleopContract,
    axis_isolated_translation_command_origin,
    clamp_outstanding_cartesian_position_target,
    reset_safe_gripper_open_mask,
    synchronized_rate_limit_position_target,
)


class _G2PositionTargetRateLimit:
    """Shared per-physics-step target rate limiter for G2 action terms."""

    def _initialize_target_rate_limit(self, width: int, env) -> None:
        speed = float(self.cfg.maximum_joint_target_speed_rad_s)
        if speed <= 0.0:
            raise ValueError("maximum_joint_target_speed_rad_s must be positive")
        acceleration = float(self.cfg.maximum_joint_target_acceleration_rad_s2)
        if acceleration <= 0.0:
            raise ValueError(
                "maximum_joint_target_acceleration_rad_s2 must be positive"
            )
        self._g2_physics_dt_s = float(env.physics_dt)
        self._g2_previous_target = torch.zeros(
            (self.num_envs, width), device=self.device
        )
        self._g2_target_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._g2_previous_target_velocity = torch.zeros(
            (self.num_envs, width), device=self.device
        )
        self._g2_environment_speed_limit_rad_s = torch.full(
            (self.num_envs, 1), speed, device=self.device
        )
        self._g2_maximum_emitted_target_speed_rad_s = torch.zeros(
            (), device=self.device
        )
        self._g2_maximum_emitted_target_acceleration_rad_s2 = torch.zeros(
            (), device=self.device
        )

    def _rate_limited_target(
        self, desired: torch.Tensor, measured: torch.Tensor
    ) -> torch.Tensor:
        uninitialized = ~self._g2_target_initialized
        if bool(uninitialized.any()):
            self._g2_previous_target[uninitialized] = measured[uninitialized]
            self._g2_previous_target_velocity[uninitialized] = 0.0
            self._g2_target_initialized[uninitialized] = True
        maximum_target_delta = (
            self._g2_environment_speed_limit_rad_s * self._g2_physics_dt_s
        )
        speed_limited = self._g2_previous_target + torch.clamp(
            desired - self._g2_previous_target,
            -maximum_target_delta,
            maximum_target_delta,
        )
        desired_target_velocity = (
            speed_limited - self._g2_previous_target
        ) / self._g2_physics_dt_s
        maximum_velocity_delta = (
            float(self.cfg.maximum_joint_target_acceleration_rad_s2)
            * self._g2_physics_dt_s
        )
        target_velocity = self._g2_previous_target_velocity + torch.clamp(
            desired_target_velocity - self._g2_previous_target_velocity,
            -maximum_velocity_delta,
            maximum_velocity_delta,
        )
        limited = (
            self._g2_previous_target
            + target_velocity * self._g2_physics_dt_s
        )
        emitted_speed = torch.max(
            torch.abs(limited - self._g2_previous_target)
        ) / self._g2_physics_dt_s
        emitted_acceleration = torch.max(
            torch.abs(target_velocity - self._g2_previous_target_velocity)
        ) / self._g2_physics_dt_s
        self._g2_maximum_emitted_target_speed_rad_s = torch.maximum(
            self._g2_maximum_emitted_target_speed_rad_s, emitted_speed
        )
        self._g2_maximum_emitted_target_acceleration_rad_s2 = torch.maximum(
            self._g2_maximum_emitted_target_acceleration_rad_s2,
            emitted_acceleration,
        )
        self._g2_previous_target.copy_(limited)
        self._g2_previous_target_velocity.copy_(target_velocity)
        return limited

    def _synchronized_rate_limited_target(
        self, desired: torch.Tensor, measured: torch.Tensor
    ) -> torch.Tensor:
        """Limit an IK target without destroying its coupled joint direction."""

        uninitialized = ~self._g2_target_initialized
        if bool(uninitialized.any()):
            self._g2_previous_target[uninitialized] = measured[uninitialized]
            self._g2_previous_target_velocity[uninitialized] = 0.0
            self._g2_target_initialized[uninitialized] = True
        previous_velocity = self._g2_previous_target_velocity.clone()
        limited, target_velocity = synchronized_rate_limit_position_target(
            self._g2_previous_target,
            desired,
            self._g2_previous_target_velocity,
            maximum_speed_rad_s=self._g2_environment_speed_limit_rad_s,
            maximum_acceleration_rad_s2=float(
                self.cfg.maximum_joint_target_acceleration_rad_s2
            ),
            physics_dt_s=self._g2_physics_dt_s,
        )
        emitted_speed = torch.amax(
            torch.abs(limited - self._g2_previous_target)
        ) / self._g2_physics_dt_s
        emitted_acceleration = torch.amax(
            torch.abs(target_velocity - previous_velocity)
        ) / self._g2_physics_dt_s
        self._g2_maximum_emitted_target_speed_rad_s = torch.maximum(
            self._g2_maximum_emitted_target_speed_rad_s, emitted_speed
        )
        self._g2_maximum_emitted_target_acceleration_rad_s2 = torch.maximum(
            self._g2_maximum_emitted_target_acceleration_rad_s2,
            emitted_acceleration,
        )
        self._g2_previous_target.copy_(limited)
        self._g2_previous_target_velocity.copy_(target_velocity)
        return limited

    def _reset_target_rate_limit(self, env_ids) -> None:
        self._g2_target_initialized[env_ids] = False
        self._g2_previous_target_velocity[env_ids] = 0.0
        self._g2_environment_speed_limit_rad_s[env_ids] = float(
            self.cfg.maximum_joint_target_speed_rad_s
        )

    def set_environment_speed_limit_rad_s(self, limit: torch.Tensor) -> None:
        """Set a per-environment joint-target speed cap.

        Values may only reduce the configured maximum; this method cannot
        raise actuator authority above the audited command limit.
        """

        if limit.shape != (self.num_envs,):
            raise ValueError("environment speed limit shape mismatch")
        limit = limit.to(self.device, dtype=self._g2_previous_target.dtype)
        if bool((limit <= 0.0).any()):
            raise ValueError("environment speed limits must be positive")
        self._g2_environment_speed_limit_rad_s[:, 0].copy_(
            torch.clamp(
                limit,
                max=float(self.cfg.maximum_joint_target_speed_rad_s),
            )
        )

    def synchronize_target_to_measured(self, env_ids: torch.Tensor) -> None:
        """Start a zero-velocity hold from the live measured joint state.

        A zero Cartesian action does not by itself cancel velocity retained by
        the acceleration-limited target generator.  Reset initialization uses
        this explicit hand-off only after the task-space endpoint has remained
        in tolerance.  It does not teleport articulation state or relax any
        velocity/acceleration limit.
        """

        if env_ids.ndim != 1:
            raise ValueError("env_ids must be a one-dimensional tensor")
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        measured = tensor_value(self._asset.data.joint_pos)[:, self._joint_ids]
        self._g2_previous_target[env_ids] = measured[env_ids]
        self._g2_previous_target_velocity[env_ids] = 0.0
        self._g2_target_initialized[env_ids] = True

    @property
    def current_target_velocity_rad_s(self) -> torch.Tensor:
        """Live velocity state of the acceleration-limited target generator."""

        return self._g2_previous_target_velocity

    @property
    def last_emitted_joint_position_target(self) -> torch.Tensor:
        """Final post-controller, post-rate-limit actuator target in radians."""

        return self._g2_previous_target

    @property
    def maximum_emitted_target_acceleration_rad_s2(self) -> torch.Tensor:
        """Peak commanded target acceleration; distinct from measured FD acceleration."""

        return self._g2_maximum_emitted_target_acceleration_rad_s2


class G2RateLimitedDifferentialIKAction(
    _G2PositionTargetRateLimit, DifferentialInverseKinematicsAction
):
    """Six-dimensional differential IK with a configurable joint-target speed."""

    cfg: "G2RateLimitedDifferentialIKActionCfg"

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._initialize_target_rate_limit(len(self._joint_ids), env)

    def apply_actions(self) -> None:
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = tensor_value(self._asset.data.joint_pos)[:, self._joint_ids]
        if bool(torch.linalg.vector_norm(ee_quat_curr, dim=-1).min() > 0.0):
            jacobian = self._compute_frame_jacobian()
            desired = self._ik_controller.compute(
                ee_pos_curr, ee_quat_curr, jacobian, joint_pos
            )
        else:
            desired = joint_pos.clone()
        # Differential IK can emit a setpoint beyond the articulation limit.
        # Rate limiting that target alone does not prevent the measured joint
        # from striking the hard stop (observed at joint7 == pi/2).  Clamp to
        # the audited soft limits with the same internal margin used by the
        # redundancy action; do not alter the URDF/USD limits themselves.
        soft_limits = tensor_value(self._asset.data.soft_joint_pos_limits)[
            :, self._joint_ids
        ]
        hard_limits = tensor_value(self._asset.data.joint_pos_limits)[
            :, self._joint_ids
        ]
        margin = float(self.cfg.joint_limit_margin_rad)
        # The physics hard limit is the final authority.  Keep the learning
        # soft region only where it is stricter.  In a 25-env run joint7 was
        # measured at its -1.571 rad hard stop while the emitted target was
        # -1.597 rad; the resulting constraint stop changed measured velocity
        # from -0.206 to 0 in one 20 ms control interval.  Intersecting both
        # sources fixes the target, without modifying either asset limit.
        limits = torch.stack(
            (
                torch.maximum(soft_limits[..., 0], hard_limits[..., 0]),
                torch.minimum(soft_limits[..., 1], hard_limits[..., 1]),
            ),
            dim=-1,
        )
        safe_lower = limits[..., 0] + margin
        safe_upper = limits[..., 1] - margin
        if bool((safe_lower >= safe_upper).any()):
            raise RuntimeError("G2_ARM_JOINT_LIMIT_MARGIN_EMPTY")
        # A stale internal target from an earlier IK request must not remain
        # beyond a hard stop while the desired target has already been
        # clamped.  Fresh episodes normally make this a no-op.
        bounded_previous = torch.clamp(
            self._g2_previous_target, safe_lower, safe_upper
        )
        previous_was_clamped = bounded_previous != self._g2_previous_target
        self._g2_previous_target.copy_(bounded_previous)
        self._g2_previous_target_velocity.masked_fill_(previous_was_clamped, 0.0)
        desired = torch.clamp(
            desired, safe_lower, safe_upper
        )
        desired = self._rate_limited_target(desired, joint_pos)
        desired = torch.clamp(desired, safe_lower, safe_upper)
        if bool(((desired < safe_lower) | (desired > safe_upper)).any()):
            raise RuntimeError("G2_ARM_EMITTED_TARGET_OUTSIDE_SAFE_LIMIT")
        self._g2_previous_target.copy_(desired)
        self._asset.set_joint_position_target_index(
            target=desired, joint_ids=self._joint_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._reset_target_rate_limit(env_ids)


@configclass
class G2RateLimitedDifferentialIKActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type = G2RateLimitedDifferentialIKAction
    maximum_joint_target_speed_rad_s: float = 0.6
    joint_limit_margin_rad: float = 0.02
    # A 25-env live attribution measured 13.046 rad/s^2 at joint6 from a
    # 5.0 rad/s^2 emitted-target limit without contact produced 13.046 rad/s^2
    # measured acceleration.  A later 25-env run still produced 11.899 at a
    # 3.0 target limit.  Retiming to 2.0 preserves the measured-motion hard
    # gate at 10 rad/s^2 and adds margin for close-range drive reversal.
    maximum_joint_target_acceleration_rad_s2: float = 2.0


class G2RateLimitedBinaryJointPositionAction(
    _G2PositionTargetRateLimit, BinaryJointPositionAction
):
    """Binary gripper goal with a bounded emitted master-joint target speed."""

    cfg: "G2RateLimitedBinaryJointPositionActionCfg"

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        if int(self.cfg.minimum_open_policy_steps_after_reset) < 0:
            raise ValueError("minimum_open_policy_steps_after_reset must be nonnegative")
        # Isaac Lab initializes BinaryJointAction._processed_actions to zero,
        # which is the G2 close target.  Seed it explicitly to OPEN so the
        # first apply and every reset cannot inherit a closed command from a
        # previous episode before the policy's next action is processed.
        self._raw_actions.fill_(1.0)
        self._processed_actions.copy_(
            self._open_command.unsqueeze(0).expand_as(self._processed_actions)
        )
        self._initialize_target_rate_limit(self._num_joints, env)
        self._last_rate_limited_target = self._processed_actions.clone()
        self._g2_external_hold_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._g2_open_hold_physics_steps = int(
            math.ceil(
                float(self.cfg.minimum_open_policy_steps_after_reset)
                * float(env.step_dt)
                / float(env.physics_dt)
            )
        )
        self._g2_open_hold_remaining = torch.full(
            (self.num_envs,),
            self._g2_open_hold_physics_steps,
            dtype=torch.long,
            device=self.device,
        )
        self._g2_close_command_armed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        """Reject inherited close state until this episode has requested OPEN."""

        super().process_actions(actions)
        protected, armed = reset_safe_gripper_open_mask(
            actions,
            self._g2_open_hold_remaining,
            self._g2_close_command_armed,
        )
        self._g2_close_command_armed.copy_(armed)
        self._processed_actions[protected] = self._open_command

    def set_external_hold_mask(self, mask: torch.Tensor) -> None:
        """Hold measured gripper position after bilateral object contact.

        The binary close intent remains in the replay action.  This actuator
        safety layer only prevents the position drive from continuing to
        compress an already contacted rigid object.
        """

        if mask.shape != (self.num_envs,):
            raise ValueError("gripper external hold mask shape mismatch")
        self._g2_external_hold_mask.copy_(mask.to(self.device, dtype=torch.bool))

    def apply_actions(self) -> None:
        measured = tensor_value(self._asset.data.joint_pos)[:, self._joint_ids]
        desired = self._processed_actions
        if bool(self._g2_external_hold_mask.any()):
            desired = desired.clone()
            desired[self._g2_external_hold_mask] = measured[
                self._g2_external_hold_mask
            ]
        target = self._rate_limited_target(desired, measured)
        self._last_rate_limited_target.copy_(target)
        self._asset.set_joint_position_target_index(
            target=target, joint_ids=self._joint_ids
        )
        self._g2_open_hold_remaining.sub_(1).clamp_(min=0)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        selection = slice(None) if env_ids is None else env_ids
        self._raw_actions[selection] = 1.0
        self._processed_actions[selection] = self._open_command
        self._last_rate_limited_target[selection] = self._open_command
        self._reset_target_rate_limit(env_ids)
        self._g2_external_hold_mask[selection] = False
        self._g2_open_hold_remaining[selection] = self._g2_open_hold_physics_steps
        self._g2_close_command_armed[selection] = False


@configclass
class G2RateLimitedBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    class_type: type = G2RateLimitedBinaryJointPositionAction
    maximum_joint_target_speed_rad_s: float = 0.8
    maximum_joint_target_acceleration_rad_s2: float = 10.0
    minimum_open_policy_steps_after_reset: int = 60


class G2RedundancyDifferentialIKAction(
    _G2PositionTargetRateLimit, DifferentialInverseKinematicsAction
):
    """Differential IK plus a bounded Jacobian-nullspace elbow request."""

    cfg: "G2RedundancyDifferentialIKActionCfg"

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._initialize_target_rate_limit(len(self._joint_ids), env)
        if len(self._joint_ids) != 7:
            raise ValueError("G2 wrist-only rotation requires seven ordered arm joints")
        if not 1 <= int(self.cfg.rotation_only_wrist_start_joint_index) < 7:
            raise ValueError(
                "rotation_only_wrist_start_joint_index must be in [1, 6]"
            )
        # Pure roll/pitch/yaw teleoperation holds the EE position captured at
        # the beginning of the rotation sequence.  Without this persistent
        # anchor, each relative command uses the slightly drifted measured
        # position as its next origin and converts IK tracking error into
        # cumulative XYZ motion.
        self._g2_rotation_position_lock = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self._g2_rotation_position_lock_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Relative Cartesian key presses are pose increments, not one-frame
        # velocity commands.  The joint target limiter generally needs many
        # physics steps to realize one increment.  Retain the IK endpoint on
        # neutral heartbeat packets; otherwise every heartbeat re-bases the
        # desired pose on a partially tracked measurement and silently
        # cancels the outstanding X/Y/Z or R/P/Y request.
        self._g2_pose_target_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._g2_active_translation_axis = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._g2_effective_elbow_command = torch.zeros(
            self.num_envs, device=self.device
        )
        self._g2_rotation_only_request = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    @property
    def action_dim(self) -> int:
        return 7

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 7):
            raise ValueError(f"G2 redundancy arm action must be (N, 7), got {tuple(actions.shape)}")
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        cartesian = self._processed_actions[:, :6]
        translation_requested = torch.any(
            torch.abs(cartesian[:, :3]) > 1.0e-8, dim=-1
        )
        elbow_requested = torch.abs(self._processed_actions[:, 6]) > 1.0e-8
        rotation_only = ee_rotation_only_mask(cartesian) & ~elbow_requested
        cartesian_requested = translation_requested | rotation_only
        previous_pose_target_valid = self._g2_pose_target_valid.clone()

        # An explicit XYZ request begins a new positional command and releases
        # the old rotation anchor.  Neutral packets intentionally keep a live
        # anchor so the controller corrects residual XYZ drift between key
        # presses instead of accepting it as the next origin.
        self._g2_rotation_position_lock_valid[
            translation_requested | elbow_requested
        ] = False
        capture = rotation_only & ~self._g2_rotation_position_lock_valid
        self._g2_rotation_position_lock[capture] = ee_pos_curr[capture]
        self._g2_rotation_position_lock_valid[capture] = True
        # New deltas accumulate from the last requested endpoint.  A neutral
        # packet is a zero delta about that same endpoint, so the controller
        # continues converging instead of accepting tracking lag as the next
        # origin.  Elbow null-space requests intentionally start from the
        # measured Cartesian pose because they are not Cartesian increments.
        command_origin_position, next_translation_axis = (
            axis_isolated_translation_command_origin(
                ee_pos_curr,
                self._ik_controller.ee_pos_des,
                previous_pose_target_valid,
                self._g2_active_translation_axis,
                cartesian[:, :3],
            )
        )
        command_origin_position[elbow_requested] = ee_pos_curr[elbow_requested]
        # Switching from translation to pure R/P/Y ends the outstanding
        # translation at the measured pose.  The wrist-only solver then owns
        # only orientation and cannot unexpectedly continue an old XYZ goal.
        command_origin_position = torch.where(
            rotation_only.unsqueeze(-1),
            self._g2_rotation_position_lock,
            command_origin_position,
        )
        keep_orientation_target = previous_pose_target_valid & ~(
            translation_requested | elbow_requested
        )
        command_origin_orientation = torch.where(
            keep_orientation_target.unsqueeze(-1),
            self._ik_controller.ee_quat_des,
            ee_quat_curr,
        )

        # Do not combine a pure EE orientation request with an elbow
        # null-space request.  The latter is a separate 7-DoF control and can
        # introduce small task-space leakage even though its projection is
        # bounded.
        # Keep wrist-only authority active on neutral heartbeat packets.  An
        # XYZ or explicit elbow request intentionally releases it.
        wrist_only_authority = self._g2_rotation_position_lock_valid & ~(
            translation_requested | elbow_requested
        )
        self._g2_effective_elbow_command.copy_(self._processed_actions[:, 6])
        self._g2_effective_elbow_command[wrist_only_authority] = 0.0
        self._g2_rotation_only_request.copy_(wrist_only_authority)
        self._ik_controller.set_command(
            cartesian, command_origin_position, command_origin_orientation
        )
        # Key-repeat may arrive faster than the physical arm can settle.  Keep
        # the outstanding endpoint local to the measured gripper so dozens of
        # repeated E/W/etc. packets cannot queue an unreachable target.  This
        # is an axis-preserving queue bound, not a workspace/tolerance bypass.
        self._ik_controller.ee_pos_des.copy_(
            clamp_outstanding_cartesian_position_target(
                ee_pos_curr,
                self._ik_controller.ee_pos_des,
                maximum_axis_error_m=float(
                    self.cfg.maximum_outstanding_translation_axis_error_m
                ),
            )
        )
        self._g2_pose_target_valid[cartesian_requested] = True
        self._g2_pose_target_valid[elbow_requested] = False
        self._g2_active_translation_axis[translation_requested] = (
            next_translation_axis[translation_requested]
        )
        self._g2_active_translation_axis[rotation_only | elbow_requested] = -1

    def cancel_pose_target_to_measured(
        self, env_ids: torch.Tensor | Sequence[int]
    ) -> None:
        """Cancel outstanding Cartesian motion at the measured EE pose.

        This is used only for explicit operator control-plane requests such as
        ``L`` (clear queued motion) and viewport changes.  Ordinary neutral
        transport heartbeats deliberately do *not* call it.
        """

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.ndim != 1:
            raise ValueError("env_ids must be a one-dimensional tensor")
        if env_ids.numel() == 0:
            return
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.ee_pos_des[env_ids] = ee_pos_curr[env_ids]
        self._ik_controller.ee_quat_des[env_ids] = ee_quat_curr[env_ids]
        self._g2_pose_target_valid[env_ids] = False
        self._g2_active_translation_axis[env_ids] = -1
        self._g2_rotation_position_lock[env_ids] = 0.0
        self._g2_rotation_position_lock_valid[env_ids] = False
        self._g2_effective_elbow_command[env_ids] = 0.0
        self._g2_rotation_only_request[env_ids] = False
        self.synchronize_target_to_measured(env_ids)

    def apply_actions(self) -> None:
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = tensor_value(self._asset.data.joint_pos)[:, self._joint_ids]
        jacobian = self._compute_frame_jacobian()
        if bool(torch.linalg.vector_norm(ee_quat_curr, dim=-1).min() > 0.0):
            desired = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
            if bool(self._g2_rotation_only_request.any()):
                wrist_jacobian = wrist_only_rotation_jacobian(
                    jacobian,
                    wrist_start_joint_index=(
                        self.cfg.rotation_only_wrist_start_joint_index
                    ),
                )
                wrist_desired = self._ik_controller.compute(
                    ee_pos_curr,
                    ee_quat_curr,
                    wrist_jacobian,
                    joint_pos,
                )
                wrist_start = int(self.cfg.rotation_only_wrist_start_joint_index)
                wrist_desired[:, :wrist_start] = joint_pos[:, :wrist_start]
                desired = torch.where(
                    self._g2_rotation_only_request.unsqueeze(-1),
                    wrist_desired,
                    desired,
                )
        else:
            desired = joint_pos.clone()

        projection = damped_nullspace_project(
            jacobian,
            self._g2_effective_elbow_command,
            seed_joint_index=self.cfg.redundancy_seed_joint_index,
            damping=self.cfg.nullspace_damping,
            maximum_joint_delta_rad=self.cfg.maximum_nullspace_joint_delta_rad_per_physics_step,
        )
        desired = desired + projection.joint_delta
        limits = tensor_value(self._asset.data.soft_joint_pos_limits)[:, self._joint_ids]
        margin = self.cfg.joint_limit_margin_rad
        desired = torch.clamp(desired, limits[..., 0] + margin, limits[..., 1] - margin)
        if bool(self._g2_rotation_only_request.any()):
            wrist_start = int(self.cfg.rotation_only_wrist_start_joint_index)
            rows = self._g2_rotation_only_request
            # Matching proximal targets to measured q cancels any outstanding
            # seven-joint target without introducing a drive-position jump.
            self._g2_previous_target[rows, :wrist_start] = joint_pos[
                rows, :wrist_start
            ]
            self._g2_previous_target_velocity[rows, :wrist_start] = 0.0
        # One common row scale preserves the 7-joint DLS solution.  Independent
        # clipping made joints 1/4/7 saturate at the same delta in live E-key
        # traces and turned a pure root-frame -Z request into +X/+Z motion.
        desired = self._synchronized_rate_limited_target(desired, joint_pos)
        self._last_nullspace_joint_delta = projection.joint_delta
        self._last_nullspace_task_leakage = projection.task_leakage
        self._last_nullspace_clipped = projection.clipped

        if hasattr(self._asset, "set_joint_position_target_index"):
            self._asset.set_joint_position_target_index(target=desired, joint_ids=self._joint_ids)
        else:
            self._asset.set_joint_position_target(desired, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._processed_actions[env_ids] = 0.0
        self._g2_rotation_position_lock[env_ids] = 0.0
        self._g2_rotation_position_lock_valid[env_ids] = False
        self._g2_pose_target_valid[env_ids] = False
        self._g2_active_translation_axis[env_ids] = -1
        self._g2_effective_elbow_command[env_ids] = 0.0
        self._g2_rotation_only_request[env_ids] = False
        self._reset_target_rate_limit(env_ids)

    @property
    def ee_rotation_position_lock_active(self) -> torch.Tensor:
        """Per-environment attestation for rotation-only EE position hold."""

        return self._g2_rotation_position_lock_valid

    @property
    def ee_rotation_only_request(self) -> torch.Tensor:
        """Rows whose latest command changes EE orientation only."""

        return self._g2_rotation_only_request

    @property
    def ee_pose_target_active(self) -> torch.Tensor:
        """Rows retaining an outstanding Cartesian endpoint across heartbeats."""

        return self._g2_pose_target_valid

    @property
    def active_translation_axis(self) -> torch.Tensor:
        """Root-frame XYZ axis retained by the current keyboard endpoint."""

        return self._g2_active_translation_axis

    @property
    def ee_desired_position(self) -> torch.Tensor:
        """Persistent gripper-center position target in robot-root axes."""

        return self._ik_controller.ee_pos_des

    @property
    def ee_desired_orientation(self) -> torch.Tensor:
        """Persistent gripper-center orientation target as root-frame WXYZ."""

        return self._ik_controller.ee_quat_des


@configclass
class G2RedundancyDifferentialIKActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type = G2RedundancyDifferentialIKAction
    redundancy_seed_joint_index: int = 2
    nullspace_damping: float = 0.05
    maximum_nullspace_joint_delta_rad_per_physics_step: float = 0.0002
    # idx61..idx64 are held for pure RPY; idx65..idx67 retain authority.
    rotation_only_wrist_start_joint_index: int = 4
    joint_limit_margin_rad: float = 0.02
    maximum_joint_target_speed_rad_s: float = 0.6
    maximum_joint_target_acceleration_rad_s2: float = 2.0
    # At the default 15 mm translation key step this permits two outstanding
    # increments while preventing unbounded key-repeat accumulation.
    maximum_outstanding_translation_axis_error_m: float = 0.03


class G2RedundancySe3Keyboard(Se3Keyboard):
    """Official SE(3) keyboard with bracket-key elbow swivel."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._g2_elbow_state = G2ElbowKeyState()
        self._g2_normalization = G2RedundancyKeyboardTeleopContract()

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "_g2_elbow_state"):
            self._g2_elbow_state.reset()

    @property
    def input_device_name(self) -> str:
        """Return the live Omniverse keyboard name for GUI attestation."""

        return str(self._input.get_keyboard_name(self._keyboard))

    def advance_physical(self) -> torch.Tensor:
        """Return physical SE(3), normalized elbow request, and gripper."""

        original = super().advance()
        elbow = torch.tensor(
            [self._g2_elbow_state.command], dtype=original.dtype, device=original.device
        )
        return torch.cat((original[:6], elbow, original[6:7]))

    def advance(self) -> torch.Tensor:
        """Return the normalized 8-D action expected by the environment."""

        return self.sample()[1]

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the keyboard once and return physical and normalized commands.

        A recorder must not call :meth:`advance_physical` and :meth:`advance`
        separately: that would read the device twice and could pair camera data
        with a different key state.  This method is the single sampling point.
        """

        physical = self.advance_physical()
        return physical, self._g2_normalization.normalize(physical)

    def _on_keyboard_event(self, event, *args, **kwargs):
        import carb

        pressed = event.type == carb.input.KeyboardEventType.KEY_PRESS
        released = event.type == carb.input.KeyboardEventType.KEY_RELEASE
        if pressed or released:
            self._g2_elbow_state.handle(event.input.name, pressed=pressed)
        return super()._on_keyboard_event(event, *args, **kwargs)


__all__ = [
    "G2RateLimitedBinaryJointPositionAction",
    "G2RateLimitedBinaryJointPositionActionCfg",
    "G2RateLimitedDifferentialIKAction",
    "G2RateLimitedDifferentialIKActionCfg",
    "G2RedundancyDifferentialIKAction",
    "G2RedundancyDifferentialIKActionCfg",
    "G2RedundancySe3Keyboard",
]
