"""RGB-D G2 Lift environment with an adaptive reverse-reset event."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .g2_lift_rgbd_env_cfg import G2LiftRgbdEnvCfg
from .g2_lift_methodology import RIGHT_ARM_JOINTS
from .g2_keyboard_pose import (
    apply_keyboard_collection_initial_pose,
    apply_keyboard_collection_task_geometry,
)
from .g2_visual_curriculum import reset_right_arm_reverse_curriculum
from .g2_camera_visibility import precontact_camera_visibility_failure


@configclass
class G2LiftVisualSACEnvCfg(G2LiftRgbdEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Online visual Teacher rollouts must start from the same robot and
        # task distribution as the keyboard demonstrations used to initialize
        # its actor.  This is deliberately opt-in at the visual Teacher config
        # rather than changing the generic G2 Lift sandbox.
        apply_keyboard_collection_initial_pose(self, RIGHT_ARM_JOINTS)
        apply_keyboard_collection_task_geometry(self)
        self.events.reverse_curriculum_arm = EventTerm(
            func=reset_right_arm_reverse_curriculum,
            mode="reset",
            params={
                "initial_probability": 0.60,
                "minimum_probability": 0.20,
                "progress_min": 0.95,
                "progress_max": 1.00,
                "policy_near_probability": 0.25,
                "policy_progress_min": 0.40,
                "policy_progress_max": 0.55,
                "joint_perturbation_rad": 0.005,
            },
        )
        self.terminations.precontact_camera_visibility = DoneTerm(
            func=precontact_camera_visibility_failure,
        )


__all__ = ["G2LiftVisualSACEnvCfg"]
