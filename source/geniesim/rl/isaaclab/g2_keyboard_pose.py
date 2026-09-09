"""Canonical reset contract shared by keyboard collection and its learners.

The constants remain pure data.  The two small application helpers at the end
are intentionally explicit: a learner must opt into the keyboard collection
distribution instead of silently inheriting a different exhibition reset.
"""

from __future__ import annotations


G2_KEYBOARD_PHOTO_POSE_PROFILE = (
    "KEYBOARD_ONLY_SIDE_PINCH_PERSISTENT_CARTESIAN_90_V8"
)
# Robot-right is world -Y. Keyboard collection, Teacher and Student opt into
# this same reset geometry so demonstrations and online rollouts cannot drift.
# The table/cube/goal remain shifted to robot-right (world -Y), while the cube
# stays in the validated X=0.50 m side-pinch workspace.
G2_KEYBOARD_TABLE_RIGHT_SHIFT_M = 0.18
G2_KEYBOARD_TASK_FORWARD_SHIFT_M = 0.0
G2_KEYBOARD_TABLE_CENTER_WORLD_M = (0.60, -0.23, 0.705)
G2_KEYBOARD_CUBE_CENTER_WORLD_M = (0.50, -0.23, 0.760)
G2_KEYBOARD_GOAL_CENTER_WORLD_M = (0.50, -0.23, 0.85)
G2_KEYBOARD_CUBE_X_RANGE_M = (0.49, 0.51)
G2_KEYBOARD_CUBE_Y_RANGE_M = (-0.24, -0.22)
G2_KEYBOARD_PHOTO_RIGHT_ARM_Q = (
    -2.4370370928846,
    -1.5895377476710915,
    1.548755252974733,
    -1.4860237851526188,
    -1.5876536154048206,
    0.02351333891149231,
    -0.9487143139039422,
)
G2_KEYBOARD_PHOTO_TCP_WORLD_M = (
    0.3000066245633075,
    -0.23001684450152932,
    0.8755820710604881,
)
G2_KEYBOARD_GRIPPER_BEND_JOINT_NAME = "COUPLED_WRIST_JOINT5_TO_JOINT7"
G2_KEYBOARD_GRIPPER_ALIGNMENT_JOINT_NAMES = (
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
)
G2_KEYBOARD_GRIPPER_BEND_TARGET_DEG = 90.0
G2_KEYBOARD_GRIPPER_BEND_DEG = 89.89146807
G2_KEYBOARD_GRIPPER_BEND_UPPER_LIMIT_MARGIN_RAD = 0.62208170
G2_KEYBOARD_PHOTO_PHYSICAL_ELBOW_DEG = 89.89072875
G2_KEYBOARD_PHOTO_UPPER_ARM_AXIS_WORLD = (
    -0.71971969,
    0.02008288,
    -0.69397424,
)
G2_KEYBOARD_PHOTO_FOREARM_AXIS_WORLD = (
    0.69289090,
    0.02085743,
    -0.72074071,
)
G2_KEYBOARD_PHOTO_FOREARM_HORIZONTAL_ERROR_DEG = 46.11566861
G2_KEYBOARD_PHOTO_UPPER_ARM_DOWN_ERROR_DEG = 46.05446619
G2_KEYBOARD_PHOTO_FOREARM_INWARD_YAW_DEG = 1.72419938
G2_KEYBOARD_PHOTO_TCP_LOCAL_X_WORLD = (
    -5.47942836e-05,
    0.9999999985,
    5.18774190e-08,
)
G2_KEYBOARD_PHOTO_TCP_LOCAL_Z_WORLD = (
    0.9999982059,
    5.47942836e-05,
    -0.0018934457,
)
G2_KEYBOARD_PHOTO_TCP_TO_CUBE_UNIT_VECTOR = (
    0.8658081635033992,
    0.00007292295008459088,
    -0.5003760772590098,
)
G2_KEYBOARD_PHOTO_TOOL_FORWARD_ALIGNMENT_ERROR_DEG = 0.10853193
G2_KEYBOARD_PHOTO_TOOL_TO_CUBE_ALIGNMENT_ERROR_DEG = 29.91639773
G2_KEYBOARD_PHOTO_GRIPPER_OPENING_UP_ERROR_DEG = 89.99999703
G2_KEYBOARD_PHOTO_GRIPPER_OPENING_HORIZONTAL_ERROR_DEG = 0.00000297
G2_KEYBOARD_PHOTO_EE_CUBE_DISTANCE_M = 0.23884938
G2_KEYBOARD_PHOTO_WRIST_CAMERA_WORLD_M = (
    0.21780401,
    -0.23118135,
    0.96553788,
)
G2_KEYBOARD_PHOTO_WRIST_CAMERA_RELATIVE_TCP_WORLD_M = (
    -0.08220262,
    -0.00116451,
    0.08995581,
)
# The camera is on the upper/back face of the horizontally aligned gripper.
G2_KEYBOARD_WRIST_CAMERA_OUTWARD_OFFSET_M = 0.00116451
G2_KEYBOARD_WRIST_CAMERA_ABOVE_TCP_M = 0.08995581
G2_KEYBOARD_WRIST_CAMERA_TOP_ALIGNMENT_ERROR_DEG = 42.42427806
# Projection through the camera prim's authored focal length/apertures.  Live
# RGB-D visibility remains the runtime authority; this is a reset diagnostic.
G2_KEYBOARD_PHOTO_WRIST_CUBE_CENTER_UV_NORMALIZED = (
    0.49832373,
    0.42711322,
)
G2_KEYBOARD_PHOTO_WRIST_CUBE_DEPTH_M = 0.35699624
G2_KEYBOARD_PHOTO_SELF_COLLISION_MARGIN_M = 0.07561799
G2_KEYBOARD_PHOTO_TABLE_COLLISION_MARGIN_M = 0.16850947
# Elbow-center displacement is not used as a body-clearance surrogate.  The
# actual collision-sphere surface distance below remains over 10 cm.
G2_KEYBOARD_FORMER_ELBOW_CENTER_WORLD_M = (
    0.10200000006596335,
    -0.24349999997570984,
    1.053936094059077,
)
G2_KEYBOARD_PHOTO_ELBOW_CENTER_WORLD_M = (
    -0.10496540,
    -0.23772489,
    1.14193806,
)
G2_KEYBOARD_ELBOW_OUTWARD_SHIFT_M = -0.00577511
# Surface clearance from torso collision spheres to freely moving
# arm_r_link2+ geometry. arm_r_link1 is excluded because it is the physical
# shoulder attachment. The camera-visible pose retains 13.61 cm of measured
# surface clearance while keeping torso joints fixed. This collision-sphere
# measurement, rather than elbow-center displacement, is the safety authority.
G2_KEYBOARD_NONSHOULDER_ARM_TORSO_CLEARANCE_M = 0.13607356
G2_KEYBOARD_MINIMUM_ARM_TORSO_CLEARANCE_M = 0.10
# Side-pinch frame: local +X is the horizontal jaw-opening axis, local +Y is
# world up, and local +Z approaches the cube horizontally along root +X.
G2_KEYBOARD_GRASP_ORIENTATION_ROOT_XYZW = (0.5, 0.5, 0.5, 0.5)
# Constraint-consistent open-hand seed used by the checked-in G2 Curobo
# contract.  Applying this at articulation creation prevents the first GUI
# frame from showing the URDF lower-limit (closed) pose before the ordinary
# bounded POSITION controller has completed its 60-step open-state settle.
# Only the master is actuated at runtime; inner joint1 is its -1 mimic and the
# joint3/4 values seed the passive four-bar at the matching open geometry.
# The live Isaac Sim 6 USD exposes joint4 limits as +/-0.035 rad; the legacy
# Curobo file's +/-0.35 entries are not valid runtime articulation positions.
G2_KEYBOARD_OPEN_GRIPPER_JOINT_POSITIONS_RAD = {
    "idx71_gripper_r_inner_joint1": -0.7853981633974483,
    "idx72_gripper_r_inner_joint3": 0.0,
    "idx73_gripper_r_inner_joint4": 0.0349,
    "idx81_gripper_r_outer_joint1": 0.7853981633974483,
    "idx82_gripper_r_outer_joint3": 0.01,
    "idx83_gripper_r_outer_joint4": -0.0349,
}
G2_KEYBOARD_RESET_TO_GRASP_ROTATION_DEG = 0.10853277
G2_KEYBOARD_ORIENTATION_RECOMMENDATION_DISTANCE_M = 0.050
G2_KEYBOARD_APPROACH_MODE = "SIDE_HORIZONTAL"
G2_KEYBOARD_APPROACH_DIRECTION_ROOT = (1.0, 0.0, 0.0)
G2_KEYBOARD_SIDE_PREGRASP_AXIAL_OFFSET_M = 0.100
G2_KEYBOARD_SIDE_GRASP_READY_AXIAL_OFFSET_M = 0.0
G2_KEYBOARD_SIDE_GRASP_CENTER_HEIGHT_OFFSET_M = 0.020


def keyboard_photo_joint_pose(
    base_pose: dict[str, float], right_arm_joint_names: tuple[str, ...]
) -> dict[str, float]:
    """Return a copy with the photo pose applied; never mutate shared config."""

    if len(right_arm_joint_names) != len(G2_KEYBOARD_PHOTO_RIGHT_ARM_Q):
        raise ValueError("keyboard photo pose requires exactly seven right-arm joints")
    result = dict(base_pose)
    result.update(
        dict(
            zip(
                right_arm_joint_names,
                G2_KEYBOARD_PHOTO_RIGHT_ARM_Q,
                strict=True,
            )
        )
    )
    return result


def apply_keyboard_collection_initial_pose(
    cfg, right_arm_joint_names: tuple[str, ...]
) -> None:
    """Apply the shared keyboard arm pose and open OmniPicker reset seed."""

    pose = keyboard_photo_joint_pose(
        cfg.scene.robot.init_state.joint_pos,
        right_arm_joint_names,
    )
    missing = set(G2_KEYBOARD_OPEN_GRIPPER_JOINT_POSITIONS_RAD).difference(pose)
    if missing:
        raise ValueError(
            "keyboard open-gripper joints are absent from the G2 pose: "
            + ",".join(sorted(missing))
        )
    pose.update(G2_KEYBOARD_OPEN_GRIPPER_JOINT_POSITIONS_RAD)
    cfg.scene.robot.init_state.joint_pos = pose


def apply_keyboard_collection_task_geometry(cfg) -> None:
    """Apply the shared keyboard/Teacher/Student task distribution."""

    # Import locally to keep the constant-only module safe for CLI startup.
    from .g2_lift_methodology import G2LiftSandboxContract

    sandbox = G2LiftSandboxContract().validated()

    cfg.scene.table.init_state.pos = G2_KEYBOARD_TABLE_CENTER_WORLD_M
    cfg.scene.object.init_state.pos = G2_KEYBOARD_CUBE_CENTER_WORLD_M
    cfg.scene.object.spawn.size = sandbox.cube_size_m
    cfg.commands.object_pose.ranges.pos_x = G2_KEYBOARD_CUBE_X_RANGE_M
    cfg.commands.object_pose.ranges.pos_y = G2_KEYBOARD_CUBE_Y_RANGE_M
    cfg.commands.object_pose.ranges.pos_z = (
        G2_KEYBOARD_GOAL_CENTER_WORLD_M[2],
        G2_KEYBOARD_GOAL_CENTER_WORLD_M[2],
    )
    cfg.rewards.outside_certified_region.params.update(
        {
            "reachable_x_range_m": G2_KEYBOARD_CUBE_X_RANGE_M,
            "reachable_y_range_m": G2_KEYBOARD_CUBE_Y_RANGE_M,
        }
    )


__all__ = [name for name in globals() if name.startswith("G2_KEYBOARD_")]
__all__.append("keyboard_photo_joint_pose")
__all__.extend(
    (
        "apply_keyboard_collection_initial_pose",
        "apply_keyboard_collection_task_geometry",
    )
)
