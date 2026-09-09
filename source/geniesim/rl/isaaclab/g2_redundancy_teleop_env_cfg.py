"""Teleoperation-only G2 RGB-D environment with elbow redundancy control."""

from __future__ import annotations

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .g2_lift_methodology import RIGHT_ARM_JOINTS, RIGHT_EE_BODY
from .g2_lift_env_cfg import G2LiftReachableSandboxEnvCfg
from .g2_lift_rgbd_env_cfg import G2LiftRgbdEnvCfg
from .g2_redundancy_action import G2RedundancyDifferentialIKActionCfg
from .g2_teleop_dataset import (
    G2_ROTATION_ACTION_SCALE_RAD,
    G2_TRANSLATION_ACTION_SCALE_M,
)
from .g2_keyboard_pose import (
    G2_KEYBOARD_GRASP_ORIENTATION_ROOT_XYZW,
    apply_keyboard_collection_initial_pose,
    apply_keyboard_collection_task_geometry,
)


def _apply_keyboard_photo_pose(cfg) -> None:
    """Override only the keyboard environment's right-arm reset state."""

    apply_keyboard_collection_initial_pose(cfg, RIGHT_ARM_JOINTS)


def _apply_keyboard_task_geometry(cfg) -> None:
    """Apply the keyboard-only reachable, visible table/cube placement."""

    apply_keyboard_collection_task_geometry(cfg)


def _redundancy_arm_action() -> G2RedundancyDifferentialIKActionCfg:
    return G2RedundancyDifferentialIKActionCfg(
            asset_name="robot",
            joint_names=list(RIGHT_ARM_JOINTS),
            body_name=RIGHT_EE_BODY,
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=(
                G2_TRANSLATION_ACTION_SCALE_M,
                G2_TRANSLATION_ACTION_SCALE_M,
                G2_TRANSLATION_ACTION_SCALE_M,
                G2_ROTATION_ACTION_SCALE_RAD,
                G2_ROTATION_ACTION_SCALE_RAD,
                G2_ROTATION_ACTION_SCALE_RAD,
                1.0,
            ),
            redundancy_seed_joint_index=2,
            nullspace_damping=0.05,
            # apply_actions runs at every physics step. Ten applications per
            # policy step therefore bound this contribution to 0.002 rad.
            maximum_nullspace_joint_delta_rad_per_physics_step=0.0002,
            joint_limit_margin_rad=0.02,
            maximum_joint_target_speed_rad_s=0.6,
        )


@configclass
class G2RedundancyControlEnvCfg(G2LiftReachableSandboxEnvCfg):
    """Camera-free control smoke configuration."""

    def __post_init__(self):
        super().__post_init__()
        # The camera-free smoke still evaluates the shared bilateral-contact,
        # stable-grasp and recovery reward terms.  Those terms require the two
        # filtered pad sensors that the RGB-D subclass normally installs.
        # Omitting them made the first env.step fail before any control key was
        # exercised, hiding Cartesian-axis regressions behind a KeyError.
        self.scene.right_inner_finger_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_r_inner_link4",
            update_period=0.0,
            history_length=1,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        self.scene.right_outer_finger_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_r_outer_link4",
            update_period=0.0,
            history_length=1,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        self.actions.arm_action = _redundancy_arm_action()


@configclass
class G2RedundancyTeleopEnvCfg(G2LiftRgbdEnvCfg):
    """RGB-D data-collection configuration with the 8-D teleop action."""

    def __post_init__(self):
        super().__post_init__()
        _apply_keyboard_photo_pose(self)
        _apply_keyboard_task_geometry(self)
        self.actions.arm_action = _redundancy_arm_action()
        self.rewards.grasp_orientation.params.update(
            {
                "desired_orientation_root_xyzw": (
                    G2_KEYBOARD_GRASP_ORIENTATION_ROOT_XYZW
                )
            }
        )
        # Keyboard collection owns collision disposition: it must preserve the
        # post-action collision row, label it as a failed demonstration, flush
        # it, and only then reset to the audited open-hand initialization.
        # ManagerBasedRLEnv auto-reset would replace that post-action state
        # before the recorder can consume it, so disable only this duplicated
        # termination term.  The live full-body collision evaluator remains
        # active in run_g2_keyboard_teacher_collection.py.
        self.terminations.forbidden_collision = None
        # Collection likewise owns the camera-loss row and must flush it
        # before reset.  The collector applies the same recurrent-window grace
        # as the normal environment termination; leaving both enabled lets
        # ManagerBasedRLEnv auto-reset before that terminal row is recorded.
        self.terminations.precontact_camera_visibility = None


__all__ = ["G2RedundancyControlEnvCfg", "G2RedundancyTeleopEnvCfg"]
