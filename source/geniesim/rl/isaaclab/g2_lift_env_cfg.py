"""Version-audited Isaac Lab ManagerBased scaffold for the G2 Lift sandbox.

Franka-specific robot, joints, gains, frames and scales are not reused.  Only
the official Lift task's manager decomposition is inherited.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from .g2_lift_methodology import (
    FIXED_TORSO_Q,
    G2LiftSandboxContract,
    G2_USD_PATH,
    RIGHT_ARM_JOINTS,
    RIGHT_EE_BODY,
    RIGHT_GRIPPER_MASTER,
    TORSO_JOINTS,
    exhibition_fixed_torso_pose,
)
from .g2_redundancy_action import (
    G2RateLimitedBinaryJointPositionActionCfg,
    G2RateLimitedDifferentialIKActionCfg,
)
from .g2_teleop_dataset import (
    G2_ROTATION_ACTION_SCALE_RAD,
    G2_TRANSLATION_ACTION_SCALE_M,
)
from .g2_quaternion import native_identity_quaternion
from .g2_collision_authority import (
    install_g2_forbidden_collision_sensors,
    make_g2_selective_collision_spawn,
)
from . import g2_lift_task_mdp as g2_task_mdp


SANDBOX = G2LiftSandboxContract().validated()


def _exhibition_initial_joint_pose() -> dict[str, float]:
    pose = exhibition_fixed_torso_pose()
    if tuple(pose[name] for name in TORSO_JOINTS) != FIXED_TORSO_Q:
        raise RuntimeError("G2_LEVEL1_TORSO_IS_NOT_FIXED_ZERO")
    # Keep the OmniPicker at its URDF/USD constraint-consistent home.  The
    # authored master joint starts at q=0 (its URDF lower limit), while the
    # passive four-bar joints retain the state validated by the G2 asset
    # smoke.  Normal POSITION control opens only the outer master after reset;
    # PhysX owns the mimic/passive coordinates.
    if pose[RIGHT_GRIPPER_MASTER] != 0.0:
        raise RuntimeError("G2_URDF_GRIPPER_HOME_MASTER_NOT_ZERO")
    return pose


@configclass
class G2LiftReachableSandboxEnvCfg(LiftEnvCfg):
    """Milestone 4 scaffold; it is not yet approved for policy training."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                func=make_g2_selective_collision_spawn(),
                usd_path=G2_USD_PATH,
                activate_contact_sensors=True,
                # Session-layer overrides only: the checked-in asset remains
                # byte-for-byte unchanged.  The live collision evaluator is
                # still required before a zero-contact sample is considered
                # valid safety evidence.
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                # The two supported Isaac Lab generations use different
                # native quaternion orders. Persisted policy data remains
                # canonical XYZW; the asset boundary is version-audited.
                rot=native_identity_quaternion(),
                joint_pos=_exhibition_initial_joint_pose(),
                joint_vel={".*": 0.0},
            ),
            actuators={
                "usd_authored_drives": ImplicitActuatorCfg(
                    joint_names_expr=[".*"], stiffness=None, damping=None,
                    effort_limit_sim=None, velocity_limit_sim=None,
                )
            },
        )
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=SANDBOX.table_center_world_m),
            spawn=sim_utils.CuboidCfg(
                size=SANDBOX.table_size_m,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=SANDBOX.table_static_friction,
                    dynamic_friction=SANDBOX.table_dynamic_friction,
                    restitution=SANDBOX.restitution,
                    friction_combine_mode="average",
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.22, 0.14)),
            ),
        )
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=SANDBOX.cube_nominal_world_m,
                rot=native_identity_quaternion(),
            ),
            spawn=sim_utils.CuboidCfg(
                size=SANDBOX.cube_size_m,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=SANDBOX.cube_mass_kg),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=0.8,
                    restitution=SANDBOX.restitution,
                    friction_combine_mode="max",
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
            ),
        )
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{RIGHT_EE_BODY}",
                    name="end_effector",
                )
            ],
        )
        install_g2_forbidden_collision_sensors(self.scene)

        self.actions.arm_action = G2RateLimitedDifferentialIKActionCfg(
            asset_name="robot",
            joint_names=list(RIGHT_ARM_JOINTS),
            body_name=RIGHT_EE_BODY,
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            # Use the same physical normalization as keyboard demonstrations,
            # Teacher replay and Student deployment. Joint velocity and
            # acceleration safety limits are deliberately unchanged.
            scale=(
                G2_TRANSLATION_ACTION_SCALE_M,
                G2_TRANSLATION_ACTION_SCALE_M,
                G2_TRANSLATION_ACTION_SCALE_M,
                G2_ROTATION_ACTION_SCALE_RAD,
                G2_ROTATION_ACTION_SCALE_RAD,
                G2_ROTATION_ACTION_SCALE_RAD,
            ),
            maximum_joint_target_speed_rad_s=0.6,
        )
        self.actions.gripper_action = G2RateLimitedBinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[RIGHT_GRIPPER_MASTER],
            open_command_expr={RIGHT_GRIPPER_MASTER: 0.7853981633974483},
            close_command_expr={RIGHT_GRIPPER_MASTER: 0.0},
            maximum_joint_target_speed_rad_s=0.8,
        )
        controlled = set(self.actions.arm_action.joint_names) | set(
            self.actions.gripper_action.joint_names
        )
        if controlled.intersection(TORSO_JOINTS):
            raise RuntimeError("G2_FIXED_TORSO_LEAKED_INTO_ACTION_SPACE")
        self.terminations.forbidden_collision = DoneTerm(
            func=g2_task_mdp.forbidden_collision,
            params={"force_threshold_n": 1.0e-6},
        )
        self.terminations.fixed_torso_drift = DoneTerm(
            func=g2_task_mdp.fixed_torso_drift,
            params={"maximum_drift_rad": SANDBOX.fixed_torso_maximum_drift_rad},
        )
        self.commands.object_pose.body_name = RIGHT_EE_BODY
        self.commands.object_pose.ranges.pos_x = (0.49, 0.51)
        self.commands.object_pose.ranges.pos_y = (-0.06, -0.04)
        self.commands.object_pose.ranges.pos_z = (
            SANDBOX.goal_nominal_world_m[2], SANDBOX.goal_nominal_world_m[2]
        )
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)

        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )
        self.events.randomize_object_material = EventTerm(
            func=isaac_mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "static_friction_range": SANDBOX.cube_static_friction_range,
                "dynamic_friction_range": SANDBOX.cube_dynamic_friction_range,
                "restitution_range": (SANDBOX.restitution, SANDBOX.restitution),
                "num_buckets": 64,
                "make_consistent": True,
            },
        )
        self.events.reset_g2_task_progress = EventTerm(
            func=g2_task_mdp.reset_task_progress_state,
            mode="reset",
        )
        self.events.reset_fixed_torso_and_head_targets = EventTerm(
            func=g2_task_mdp.reset_fixed_torso_and_head_targets,
            mode="reset",
        )
        # Sparse lift/goal rewards remain the task authority.  These bounded
        # additions supply contact experience and recovery without treating a
        # first touch as failure or changing any hard motion limit.
        self.rewards.ee_cube_progress_pbrs = RewTerm(
            func=g2_task_mdp.ee_cube_progress_pbrs,
            params={"scale_m": 0.10, "gamma": 0.99},
            weight=2.0,
        )
        self.rewards.cube_goal_progress_pbrs = RewTerm(
            func=g2_task_mdp.cube_goal_progress_pbrs,
            params={"scale_m": 0.10, "gamma": 0.99},
            weight=4.0,
        )
        self.rewards.bilateral_contact = RewTerm(
            func=g2_task_mdp.bilateral_contact_reward,
            weight=2.0,
        )
        self.rewards.any_finger_touch = RewTerm(
            func=g2_task_mdp.any_finger_touch_reward,
            params={"force_threshold_n": 1.0},
            weight=0.25,
        )
        self.rewards.stable_grasp = RewTerm(
            func=g2_task_mdp.stable_grasp_reward,
            params={"slip_limit_m_s": 0.06, "stable_time_s": 0.10},
            weight=5.0,
        )
        self.rewards.partial_grasp_terminal_credit = RewTerm(
            func=g2_task_mdp.partial_grasp_terminal_credit,
            weight=2.0,
        )
        self.rewards.push_recovery_pbrs = RewTerm(
            func=g2_task_mdp.push_recovery_pbrs,
            params={"grace_s": SANDBOX.first_touch_recovery_grace_s, "scale_m": 0.02},
            weight=1.0,
        )
        self.rewards.persistent_push = RewTerm(
            func=g2_task_mdp.persistent_push_magnitude,
            params={"grace_s": SANDBOX.first_touch_recovery_grace_s, "free_motion_m": 0.003},
            weight=-1.0,
        )
        self.rewards.outside_certified_region = RewTerm(
            func=g2_task_mdp.certified_region_violation,
            weight=-2.0,
        )
        self.rewards.grasp_orientation = RewTerm(
            func=g2_task_mdp.grasp_orientation_excess,
            weight=-0.5,
        )
        self.rewards.controlled_joint_acceleration = RewTerm(
            func=g2_task_mdp.controlled_joint_acceleration_excess,
            params={"free_rad_s2": 5.0},
            weight=-0.05,
        )
        self.rewards.lifting_object = RewTerm(
            func=mdp.object_is_lifted,
            params={"minimal_height": SANDBOX.table_surface_height_m + SANDBOX.lift_height_above_table_m},
            weight=15.0,
        )
        self.rewards.object_goal_tracking.params["minimal_height"] = (
            SANDBOX.table_surface_height_m + SANDBOX.lift_height_above_table_m
        )
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = (
            SANDBOX.table_surface_height_m + SANDBOX.lift_height_above_table_m
        )
        self.terminations.object_dropping = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={
                "minimum_height": SANDBOX.table_surface_height_m - SANDBOX.drop_margin_below_table_m,
                "asset_cfg": SceneEntityCfg("object"),
            },
        )
        self.terminations.object_reached_goal = DoneTerm(
            func=g2_task_mdp.object_stably_grasped_and_lifted,
            params={
                "slip_limit_m_s": 0.06,
                "stable_time_s": 0.10,
            },
        )
        self.terminations.time_out = DoneTerm(
            func=g2_task_mdp.time_out_without_success,
            time_out=True,
            params={"success_threshold": SANDBOX.goal_tolerance_m},
        )
        self.terminations.cube_left_table_persistently = DoneTerm(
            func=g2_task_mdp.cube_left_table_persistently,
            params={"persistence_s": SANDBOX.ungraspable_persistence_s},
        )

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.0
        self.sim.dt = SANDBOX.physics_dt_s
        self.decimation = SANDBOX.control_decimation
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0
        # The Lift template otherwise resamples the goal after 5 s, halfway
        # through this 10 s episode.  That creates a discontinuous reward and
        # success target while the progress-history state still references the
        # old goal.  Reset remains the sole goal-resampling boundary.
        policy_dt_s = self.sim.dt * self.decimation
        goal_hold_s = self.episode_length_s + policy_dt_s
        self.commands.object_pose.resampling_time_range = (goal_hold_s, goal_hold_s)


__all__ = ["G2LiftReachableSandboxEnvCfg", "SANDBOX"]
