"""Camera-free G2 Lift configuration for the privileged SAC teacher."""

from __future__ import annotations

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .g2_lift_env_cfg import G2LiftReachableSandboxEnvCfg
from .g2_lift_methodology import RIGHT_ARM_JOINTS
from .g2_keyboard_pose import (
    apply_keyboard_collection_initial_pose,
    apply_keyboard_collection_task_geometry,
)


@configclass
class G2LiftPrivilegedTeacherEnvCfg(G2LiftReachableSandboxEnvCfg):
    """Milestone 7 environment: state teacher plus physical finger contacts."""

    def __post_init__(self):
        super().__post_init__()
        apply_keyboard_collection_initial_pose(self, RIGHT_ARM_JOINTS)
        apply_keyboard_collection_task_geometry(self)
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


__all__ = ["G2LiftPrivilegedTeacherEnvCfg"]
