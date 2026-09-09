"""Milestone 4 G2 Lift methodology-sandbox contract.

The official Isaac Lab Lift task supplies the scene/reset/reward/termination
shape.  Every robot-specific value below remains G2-owned.  This module has no
Kit imports so configuration and reset acceptance can be tested before an
Isaac process is launched.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable

import numpy as np


RIGHT_ARM_JOINTS = tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8))
LEFT_ARM_JOINTS = tuple(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8))
TORSO_JOINTS = tuple(f"idx0{i}_body_joint{i}" for i in range(1, 6))
FIXED_TORSO_Q = (0.0, 0.0, 0.0, 0.0, 0.0)
RIGHT_GRIPPER_MASTER = "idx81_gripper_r_outer_joint1"
RIGHT_GRIPPER_MIMIC = "idx71_gripper_r_inner_joint1"
RIGHT_EE_BODY = "gripper_r_center_link"
G2_USD_PATH = str(
    Path(__file__).resolve().parents[2]
    / "assets/robot/G2_omnipicker/robot_fix.usda"
)
G2_USD_SHA256 = "7751a265b02b1c4f6d290a1555d5bb8f2750b7280e66cb6a94042954dc032132"

LEFT_ARM_HOME = (0.739033, -0.717023, -1.524419, -1.537612, 0.27811, -0.925845, -0.839257)
RIGHT_ARM_HOME = (-0.739033, -0.717023, 1.524419, -1.537612, -0.27811, -0.925845, 0.839257)
# Name-matched fixed-right/fixed-left G2 deployment configs both seal the
# third head joint at 0.174 rad.  Zeroing all three head joints was sufficient
# only while torso/head collision geometry was disabled; with the audited
# collision manifest live it places head_link3 through body_link5.
HEAD_HOME = (0.0, 0.0, 0.174)

# The exhibition task does not actuate or randomize the torso.  The head is
# nevertheless part of the fixed full-body geometry and its third joint is
# raised inside the audited joint limits so the table workspace is in view.
# Keep this separate from ``HEAD_HOME``: the latter is the asset smoke pose,
# while this is the camera-aware task pose used by keyboard, teacher and
# student environments.
EXHIBITION_HEAD_Q = (0.0, 0.0, 0.46)

# Fixed-torso, camera-aware L-shaped start pose shared by keyboard, privileged
# teacher and visual student environments.  It is solved against the corrected
# checked-in URDF (arm_r_link7->arm_r_end_link = 0.088 m).  This branch is the
# shoulder-material-correct full-L alternative of the V13 FK solution: the black
# arm_r_link2 surface is on robot-forward +X and the white surface is behind
# it.  The USD mesh, collision shape, inertia, gripper and camera mounts are
# not rotated or patched independently.  The physical elbow angle is measured
# from the upper-arm axis (joint3->joint4) to the complete forearm/wrist axis
# (joint4->arm_r_end_link), i.e. through the fixed joint that mounts the EE.
# This deliberately replaces the incomplete V13 joint4->joint6 measurement.
# In that URDF the
# gripper center's local +Z is the palm-to-fingertip/tool-forward direction and
# local +/-X is the finger-separation direction.  The upper arm remains within
# numerical precision of vertical-down, the full forearm/EE mounting axis
# within 0.03 degrees of horizontal, and the physical elbow angle is 90
# degrees.  The
# checked-in wrist optical extrinsic and the fixed head pose both see the full
# randomized cube box at reset.  This is a reset state, never a teacher action
# or imitation target.  Torso coordinates remain exactly zero.
EXHIBITION_RIGHT_ARM_INITIAL_Q = (
    -1.620750745749757,
    -1.6085981733983385,
    2.2180736694597014,
    -1.8037046976382127,
    -0.4504973901282091,
    0.06505922615127382,
    1.4047437203199062,
)
EXHIBITION_INITIAL_TCP_WORLD_M = (
    0.5223934070372415,
    -0.2991361981965176,
    0.9933938527444026,
)
EXHIBITION_ELBOW_ANGLE_ENDPOINT = "arm_r_end_link"
EXHIBITION_PHYSICAL_ELBOW_DEG = 89.97416300491238
EXHIBITION_UPPER_ARM_AXIS_WORLD = (
    2.2938678468197836e-10,
    8.446871363310014e-11,
    -1.0,
)
EXHIBITION_FOREARM_AXIS_WORLD = (
    0.9134243573953118,
    -0.4070082799768067,
    -0.0004509404426698042,
)
EXHIBITION_UPPER_ARM_DOWN_ALIGNMENT_ERROR_DEG = 0.0
EXHIBITION_FOREARM_HORIZONTAL_ALIGNMENT_ERROR_DEG = 0.025836985052387674
EXHIBITION_FOREARM_INWARD_YAW_DEG = -24.01704025755506
# URDF gripper opening axis.  The inner/short finger lies on +X and is up.
EXHIBITION_TCP_LOCAL_X_WORLD = (
    0.19206444063611458,
    0.39958101462562917,
    0.8963516404815072,
)
# URDF gripper tool-forward axis.  It points along robot-root +X.
EXHIBITION_TCP_LOCAL_Z_WORLD = (
    0.7221264837716854,
    0.5609556844557884,
    -0.4047987913918222,
)
EXHIBITION_TCP_TO_CUBE_UNIT_VECTOR = (
    -0.07279594286547146,
    0.6473471090063581,
    -0.7587110590754806,
)
EXHIBITION_TOOL_FORWARD_ALIGNMENT_ERROR_DEG = 43.76967289563921
EXHIBITION_GRIPPER_OPENING_UP_ALIGNMENT_ERROR_DEG = 26.317425003882214
EXHIBITION_TOOL_TO_CUBE_ALIGNMENT_ERROR_DEG = 51.85232330714634
EXHIBITION_INITIAL_EE_CUBE_DISTANCE_M = 0.30761888857769154
EXHIBITION_SHORT_FINGER_SIDE = "inner"
EXHIBITION_RIGHT_WRIST_CAMERA_WORLD_M = (
    0.522364786546452,
    -0.41091463548203866,
    1.0419351131129306,
)
EXHIBITION_RIGHT_WRIST_CAMERA_RELATIVE_TCP_WORLD_M = (
    -2.8620490789443842e-05,
    -0.11177843728552106,
    0.048541260368528016,
)
# Area-weighted centroids of the black and white arm_r_link2 USD material
# subsets after FK.  Robot-forward is world +X.  Keeping this signed evidence
# in the reset contract prevents a future IK branch from visually reversing
# the shoulder while still satisfying the same TCP target.
EXHIBITION_RIGHT_SHOULDER_BLACK_CENTROID_WORLD_M = (
    0.14584127649531442,
    -0.2431443996967036,
    1.330465916930075,
)
EXHIBITION_RIGHT_SHOULDER_WHITE_CENTROID_WORLD_M = (
    0.07790941288893528,
    -0.24089175193620094,
    1.279273714709294,
)
EXHIBITION_RIGHT_SHOULDER_BLACK_FORWARD_SEPARATION_M = 0.06793186360637914
EXHIBITION_POSE_PROFILE = "ARM_ONLY_FIXED_TORSO_DUAL_CAMERA_FULL_L_90_FRESH_V14"

# Static Pinocchio/checked-in USD optical projection evidence.  Margins are
# normalized image-edge distances after testing all cube reset-box corners and
# all eight physical 30-mm cube corners.  Live rendering is still checked at
# reset because projection cannot attest semantic occlusion.
EXHIBITION_HEAD_CUBE_CENTER_UV_NORMALIZED = (
    0.5557888746261597,
    0.7751542925834656,
)
EXHIBITION_WRIST_CUBE_CENTER_UV_NORMALIZED = (
    0.6535337567329407,
    0.5528174042701721,
)
EXHIBITION_HEAD_CUBE_DEPTH_M = 0.8695235848426819
EXHIBITION_WRIST_CUBE_DEPTH_M = 0.44097191095352173
EXHIBITION_HEAD_RESET_FRUSTUM_MIN_MARGIN = 0.20941752195358276
EXHIBITION_WRIST_RESET_FRUSTUM_MIN_MARGIN = 0.3168955445289612
_CLOSED_OMNIPICKER = {
    "inner_joint1": 0.0,
    "inner_joint3": -0.01951522007584572,
    "inner_joint4": 0.0024153252597898245,
    "inner_joint0": 0.015910323709249496,
    "outer_joint1": 0.0,
    "outer_joint3": 0.019584663212299347,
    "outer_joint4": -0.0025442452169954777,
    "outer_joint0": -0.015818988904356956,
}


def stable_original_home() -> dict[str, float]:
    """Return the exact pose used by the Milestone 2/3 live G2 smokes."""
    values = {f"idx0{i}_body_joint{i}": 0.0 for i in range(1, 6)}
    values.update(
        {f"idx1{i}_head_joint{i}": value for i, value in enumerate(HEAD_HOME, 1)}
    )
    values.update(dict(zip(LEFT_ARM_JOINTS, LEFT_ARM_HOME, strict=True)))
    values.update({f"idx6{i}_arm_r_joint{i}": value for i, value in enumerate(RIGHT_ARM_HOME, 1)})
    for prefix, side in ((3, "l"), (7, "r")):
        for suffix, digit in (("inner_joint1", 1), ("inner_joint3", 2), ("inner_joint4", 3), ("inner_joint0", 9)):
            values[f"idx{prefix}{digit}_gripper_{side}_{suffix}"] = _CLOSED_OMNIPICKER[suffix]
    for prefix, side in ((4, "l"), (8, "r")):
        for suffix, digit in (("outer_joint1", 1), ("outer_joint3", 2), ("outer_joint4", 3), ("outer_joint0", 9)):
            values[f"idx{prefix}{digit}_gripper_{side}_{suffix}"] = _CLOSED_OMNIPICKER[suffix]
    return values


def exhibition_fixed_torso_pose() -> dict[str, float]:
    """Return the shared task pose with an explicitly fixed zero torso."""

    values = stable_original_home()
    values.update(dict(zip(TORSO_JOINTS, FIXED_TORSO_Q, strict=True)))
    values.update(
        {f"idx1{i}_head_joint{i}": value for i, value in enumerate(EXHIBITION_HEAD_Q, 1)}
    )
    values.update(dict(zip(RIGHT_ARM_JOINTS, EXHIBITION_RIGHT_ARM_INITIAL_Q, strict=True)))
    return values


@dataclass(frozen=True)
class G2LiftState:
    ee_position_world_m: tuple[float, float, float]
    cube_position_world_m: tuple[float, float, float]
    goal_position_world_m: tuple[float, float, float]


@dataclass(frozen=True)
class G2LiftSandboxContract:
    """Frozen, right-arm-only Milestone 4 sandbox geometry and MDP contract."""

    # AMR center is x=0.  The 0.60 m table center produces x=[0.40,0.80]
    # and keeps the object in the independently validated right-arm region.
    table_center_world_m: tuple[float, float, float] = (0.60, -0.05, 0.705)
    table_size_m: tuple[float, float, float] = (0.40, 0.20, 0.05)
    table_surface_height_m: float = 0.73
    # Rectangular rigid workpiece: X/Y footprint 40 mm, Z height 60 mm.
    # Keep this as one shared geometry authority for keyboard collection,
    # privileged Teacher SAC and visual Student SAC.
    cube_size_m: tuple[float, float, float] = (0.04, 0.04, 0.06)
    cube_mass_kg: float = 1.200
    # The torso remains fixed.  The table and task were shifted 0.05 m toward
    # robot-right (-Y); cube y=-0.10 m remains inside the 0.20 m tabletop and
    # the measured right-arm top-down workspace; y=0 made the low grasp
    # endpoint kinematically unreachable despite reachable pre-grasp/lift
    # endpoints.
    cube_nominal_world_m: tuple[float, float, float] = (0.50, -0.10, 0.760)
    cube_xy_jitter_m: float = 0.010
    goal_nominal_world_m: tuple[float, float, float] = (0.50, -0.10, 0.85)
    goal_xy_jitter_m: float = 0.010
    goal_tolerance_m: float = 0.005
    # A 60 mm-tall cube rests with its centre at 0.760 m.  Future SUCCESS is
    # sealed at centre Z >= 0.780 m (20 mm actual upward displacement) while
    # stable grasp is simultaneously true.  Legacy accepted demonstrations
    # are migrated separately and never weaken this live runtime authority.
    lift_height_above_table_m: float = 0.050
    drop_margin_below_table_m: float = 0.050
    reach_reward_std_m: float = 0.10
    goal_reward_coarse_std_m: float = 0.30
    goal_reward_fine_std_m: float = 0.05
    maximum_reset_attempts: int = 120
    physics_dt_s: float = 0.002
    control_decimation: int = 10
    # The cube's static coefficient is the requested 1.00 +/- 0.05.  Dynamic
    # friction is sampled from a lower ordinary-contact band so every sampled
    # material obeys dynamic <= static without post-hoc clipping.
    cube_static_friction_range: tuple[float, float] = (0.95, 1.05)
    cube_dynamic_friction_range: tuple[float, float] = (0.75, 0.85)
    table_static_friction: float = 0.60
    table_dynamic_friction: float = 0.50
    restitution: float = 0.0
    # Offline Pinocchio-2.7 IK validated all 12 combinations of the shifted
    # box corners and grasp/near/pre-grasp heights with <=1.271 mm position
    # residual and >=48.90 mm conservative collision margin.
    reachable_cube_x_range_m: tuple[float, float] = (0.49, 0.51)
    reachable_cube_y_range_m: tuple[float, float] = (-0.11, -0.09)
    ordinary_initial_ee_cube_distance_m: float = EXHIBITION_INITIAL_EE_CUBE_DISTANCE_M
    ordinary_initial_distance_tolerance_m: float = 0.025
    reverse_initial_ee_cube_distance_m: float = 0.020
    reverse_initial_distance_tolerance_m: float = 0.015
    first_touch_recovery_grace_s: float = 0.10
    ungraspable_persistence_s: float = 0.20
    maximum_grasp_orientation_error_rad: float = math.radians(15.0)
    fixed_torso_q_rad: tuple[float, ...] = FIXED_TORSO_Q
    torso_action_enabled: bool = False
    fixed_torso_maximum_drift_rad: float = 0.01
    # Both sensors remain active actor inputs, but visibility is an OR gate:
    # any partial RGB-D evidence from head or wrist is sufficient.  Requiring
    # both cameras or all eight cube corners over-constrains valid grasps.
    # After bilateral contact the existing contact/proprioception authority
    # handles normal finger occlusion.
    reset_required_visible_cameras: int = 1
    precontact_minimum_visible_cameras: int = 1
    # The recurrent visual policy consumes 16 samples at the 50 Hz policy
    # boundary (0.32 s).  A shorter one-frame termination would discard the
    # exact transient occlusion/dropout sequences that the GRU is intended to
    # bridge.  Persistent loss remains fail-closed before bilateral contact.
    precontact_camera_loss_grace_s: float = 0.32
    camera_normalized_edge_margin: float = 0.08
    camera_minimum_depth_m: float = 0.03
    # The checked-in task cube has a red PreviewSurface.  At least one active
    # camera must expose this many colour-and-depth-consistent pixels.
    # Requiring the entire cube or only its projected centre ray incorrectly
    # rejects useful partial views around a fingertip occlusion.
    camera_minimum_cube_visible_pixels: int = 4

    @property
    def policy_dt_s(self) -> float:
        return self.physics_dt_s * self.control_decimation

    @property
    def controlled_joints(self) -> tuple[str, ...]:
        return (*RIGHT_ARM_JOINTS, RIGHT_GRIPPER_MASTER)

    @property
    def cube_half_extents_m(self) -> tuple[float, float, float]:
        return tuple(value / 2.0 for value in self.cube_size_m)

    @property
    def cube_edge_m(self) -> float:
        """Legacy horizontal edge accessor; new code must use cube_size_m."""

        return self.cube_size_m[0]

    @property
    def lift_success_cube_center_z_m(self) -> float:
        return self.table_surface_height_m + self.lift_height_above_table_m

    def validated(self) -> "G2LiftSandboxContract":
        if self.table_size_m != (0.40, 0.20, 0.05):
            raise ValueError("Milestone 4 table must remain 0.40 x 0.20 x 0.05 m")
        top = self.table_center_world_m[2] + self.table_size_m[2] / 2.0
        if not math.isclose(top, self.table_surface_height_m, abs_tol=1.0e-12):
            raise ValueError("table center/size do not produce the sealed surface height")
        if self.cube_size_m != (0.04, 0.04, 0.06):
            raise ValueError("task workpiece must remain 0.04 x 0.04 x 0.06 m")
        if any(value <= 0.0 for value in self.cube_size_m):
            raise ValueError("workpiece dimensions must be positive")
        cube_z = self.table_surface_height_m + self.cube_half_extents_m[2]
        if not math.isclose(self.cube_nominal_world_m[2], cube_z, abs_tol=1.0e-12):
            raise ValueError("cube must rest exactly on the table surface")
        if not math.isclose(self.cube_mass_kg, 1.200, abs_tol=1.0e-12):
            raise ValueError("exhibition task cube mass must remain 1.200 kg")
        if not math.isclose(
            self.lift_success_cube_center_z_m, 0.780, abs_tol=1.0e-12
        ):
            raise ValueError("live lift success must require cube center Z >= 0.780 m")
        if not (
            0.0 < self.cube_dynamic_friction_range[0]
            <= self.cube_dynamic_friction_range[1]
            <= self.cube_static_friction_range[0]
            <= self.cube_static_friction_range[1]
        ):
            raise ValueError("cube friction ranges must be positive and dynamic <= static")
        if not (0.0 < self.table_dynamic_friction <= self.table_static_friction):
            raise ValueError("table dynamic friction must not exceed static friction")
        if self.restitution != 0.0:
            raise ValueError("the rigid exhibition cube must remain non-bouncy")
        if self.first_touch_recovery_grace_s < 5 * self.policy_dt_s:
            raise ValueError("first-touch recovery grace must cover at least five policy steps")
        if self.goal_tolerance_m != 0.005:
            raise ValueError("goal tolerance must remain exactly 5 mm")
        if self.physics_dt_s <= 0.0 or self.control_decimation <= 0:
            raise ValueError("simulation timing must be positive")
        if self.fixed_torso_q_rad != FIXED_TORSO_Q or self.torso_action_enabled:
            raise ValueError("Level 1 torso must be fixed at zero and absent from actions")
        if self.fixed_torso_maximum_drift_rad != 0.01:
            raise ValueError("fixed-torso drift gate must remain 0.01 rad")
        if any(joint in self.controlled_joints for joint in TORSO_JOINTS):
            raise ValueError("fixed torso joints leaked into the controlled action group")
        if self.reset_required_visible_cameras != 1:
            raise ValueError("reset requires partial cube visibility in either asset camera")
        if self.precontact_minimum_visible_cameras != 1:
            raise ValueError("pre-contact requires at least one asset camera")
        if self.precontact_camera_loss_grace_s < self.policy_dt_s:
            raise ValueError("pre-contact camera loss grace must cover one policy step")
        if not 0.0 < self.camera_normalized_edge_margin < 0.5:
            raise ValueError("camera edge margin must be normalized and inside (0,0.5)")
        if self.camera_minimum_cube_visible_pixels < 1:
            raise ValueError("camera cube visibility requires at least one RGB-D pixel")
        return self

    def sample_reset(
        self,
        rng: np.random.Generator,
        *,
        reachability_gate: Callable[[np.ndarray, np.ndarray], bool],
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Sample a tabletop cube/goal pair and require external G2 IK approval."""
        self.validated()
        table_half = np.asarray(self.table_size_m[:2], dtype=np.float64) / 2.0
        table_center = np.asarray(self.table_center_world_m[:2], dtype=np.float64)
        footprint = np.asarray(self.cube_half_extents_m[:2]) + 0.010
        for rejected in range(self.maximum_reset_attempts):
            cube = np.asarray(self.cube_nominal_world_m, dtype=np.float64).copy()
            goal = np.asarray(self.goal_nominal_world_m, dtype=np.float64).copy()
            cube[:2] += rng.uniform(-self.cube_xy_jitter_m, self.cube_xy_jitter_m, 2)
            goal[:2] += rng.uniform(-self.goal_xy_jitter_m, self.goal_xy_jitter_m, 2)
            if np.any(np.abs(cube[:2] - table_center) + footprint > table_half):
                continue
            if np.any(np.abs(goal[:2] - table_center) + footprint > table_half):
                continue
            if not bool(reachability_gate(cube.copy(), goal.copy())):
                continue
            return cube, goal, rejected
        raise RuntimeError("G2_REACHABLE_RESET_EXHAUSTED")

    def cube_is_in_certified_reachable_region(self, cube_world_m: np.ndarray) -> bool:
        """Return whether one cube center is inside the offline-certified box."""

        cube = np.asarray(cube_world_m, dtype=np.float64)
        if cube.shape != (3,) or not np.all(np.isfinite(cube)):
            return False
        expected_z = self.table_surface_height_m + self.cube_half_extents_m[2]
        return bool(
            self.reachable_cube_x_range_m[0] <= cube[0] <= self.reachable_cube_x_range_m[1]
            and self.reachable_cube_y_range_m[0] <= cube[1] <= self.reachable_cube_y_range_m[1]
            and math.isclose(float(cube[2]), expected_z, abs_tol=1.0e-4)
        )

    def reward_terms(self, state: G2LiftState) -> dict[str, float]:
        """Official Lift-shaped terms with G2 geometry; no learning side effects."""
        ee = np.asarray(state.ee_position_world_m, dtype=np.float64)
        cube = np.asarray(state.cube_position_world_m, dtype=np.float64)
        goal = np.asarray(state.goal_position_world_m, dtype=np.float64)
        reach_distance = float(np.linalg.norm(cube - ee))
        goal_distance = float(np.linalg.norm(cube - goal))
        lifted = cube[2] >= self.table_surface_height_m + self.lift_height_above_table_m
        return {
            "reaching_object": 1.0 - math.tanh(reach_distance / self.reach_reward_std_m),
            "lifting_object": 15.0 if lifted else 0.0,
            "object_goal_tracking": 16.0 * (1.0 - math.tanh(goal_distance / self.goal_reward_coarse_std_m)) if lifted else 0.0,
            "object_goal_tracking_fine": 5.0 * (1.0 - math.tanh(goal_distance / self.goal_reward_fine_std_m)) if lifted else 0.0,
        }

    def termination(self, state: G2LiftState) -> dict[str, bool]:
        cube = np.asarray(state.cube_position_world_m, dtype=np.float64)
        goal = np.asarray(state.goal_position_world_m, dtype=np.float64)
        return {
            "success": bool(
                cube[2]
                >= self.table_surface_height_m + self.lift_height_above_table_m
                and np.linalg.norm(cube - goal) <= self.goal_tolerance_m
            ),
            "object_dropped": bool(cube[2] < self.table_surface_height_m - self.drop_margin_below_table_m),
        }


__all__ = [
    "G2LiftSandboxContract",
    "G2LiftState",
    "RIGHT_ARM_JOINTS",
    "LEFT_ARM_JOINTS",
    "TORSO_JOINTS",
    "FIXED_TORSO_Q",
    "RIGHT_EE_BODY",
    "RIGHT_GRIPPER_MASTER",
    "RIGHT_GRIPPER_MIMIC",
    "G2_USD_PATH",
    "G2_USD_SHA256",
    "EXHIBITION_INITIAL_TCP_WORLD_M",
    "EXHIBITION_INITIAL_EE_CUBE_DISTANCE_M",
    "EXHIBITION_PHYSICAL_ELBOW_DEG",
    "EXHIBITION_ELBOW_ANGLE_ENDPOINT",
    "EXHIBITION_UPPER_ARM_AXIS_WORLD",
    "EXHIBITION_FOREARM_AXIS_WORLD",
    "EXHIBITION_UPPER_ARM_DOWN_ALIGNMENT_ERROR_DEG",
    "EXHIBITION_FOREARM_HORIZONTAL_ALIGNMENT_ERROR_DEG",
    "EXHIBITION_FOREARM_INWARD_YAW_DEG",
    "EXHIBITION_POSE_PROFILE",
    "EXHIBITION_HEAD_Q",
    "EXHIBITION_RIGHT_ARM_INITIAL_Q",
    "EXHIBITION_TCP_LOCAL_X_WORLD",
    "EXHIBITION_TCP_LOCAL_Z_WORLD",
    "EXHIBITION_TCP_TO_CUBE_UNIT_VECTOR",
    "EXHIBITION_TOOL_TO_CUBE_ALIGNMENT_ERROR_DEG",
    "EXHIBITION_TOOL_FORWARD_ALIGNMENT_ERROR_DEG",
    "EXHIBITION_GRIPPER_OPENING_UP_ALIGNMENT_ERROR_DEG",
    "EXHIBITION_SHORT_FINGER_SIDE",
    "EXHIBITION_RIGHT_WRIST_CAMERA_WORLD_M",
    "EXHIBITION_RIGHT_WRIST_CAMERA_RELATIVE_TCP_WORLD_M",
    "EXHIBITION_RIGHT_SHOULDER_BLACK_CENTROID_WORLD_M",
    "EXHIBITION_RIGHT_SHOULDER_WHITE_CENTROID_WORLD_M",
    "EXHIBITION_RIGHT_SHOULDER_BLACK_FORWARD_SEPARATION_M",
    "EXHIBITION_HEAD_CUBE_CENTER_UV_NORMALIZED",
    "EXHIBITION_WRIST_CUBE_CENTER_UV_NORMALIZED",
    "EXHIBITION_HEAD_CUBE_DEPTH_M",
    "EXHIBITION_WRIST_CUBE_DEPTH_M",
    "EXHIBITION_HEAD_RESET_FRUSTUM_MIN_MARGIN",
    "EXHIBITION_WRIST_RESET_FRUSTUM_MIN_MARGIN",
    "HEAD_HOME",
    "exhibition_fixed_torso_pose",
    "stable_original_home",
]
