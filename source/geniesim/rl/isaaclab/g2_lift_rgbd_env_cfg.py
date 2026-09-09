"""Milestone 5 RGB-D extension of the Milestone 4 G2 Lift sandbox."""

from __future__ import annotations

from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import configclass

from .g2_lift_env_cfg import G2LiftReachableSandboxEnvCfg, SANDBOX
from .g2_asset_camera_pose import (
    CAMERA_POSE_AUTHORITY,
    CAMERA_RUNTIME_POSE_OVERRIDE,
    HEAD_CAMERA_RELATIVE_PRIM,
    RIGHT_WRIST_CAMERA_RELATIVE_PRIM,
)


CAMERA_RESOLUTION = (256, 192)
CAMERA_CAPTURE_INTERVAL_STEPS = 2


@configclass
class G2LiftRgbdEnvCfg(G2LiftReachableSandboxEnvCfg):
    """Bind the checked-in G2 camera prims without inventing extrinsics."""

    def __post_init__(self):
        super().__post_init__()
        update_period = SANDBOX.physics_dt_s * CAMERA_CAPTURE_INTERVAL_STEPS
        self.scene.head_camera = CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{HEAD_CAMERA_RELATIVE_PRIM}",
            update_period=update_period,
            height=CAMERA_RESOLUTION[1],
            width=CAMERA_RESOLUTION[0],
            data_types=["rgb", "distance_to_image_plane"],
            spawn=None,
            # Isaac Lab 3's attached-camera renderer already consumes the
            # live Fabric hierarchy.  Its generic FrameView pose refresh can
            # instead replay the authored initial world pose for nested G2
            # cameras, so do not let that stale readback override rendering.
            update_latest_camera_pose=False,
        )
        self.scene.right_wrist_camera = CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{RIGHT_WRIST_CAMERA_RELATIVE_PRIM}",
            update_period=update_period,
            height=CAMERA_RESOLUTION[1],
            width=CAMERA_RESOLUTION[0],
            data_types=["rgb", "distance_to_image_plane"],
            spawn=None,
            update_latest_camera_pose=False,
        )
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
        # Rendering every two physics steps is the camera contract.  It is
        # intentionally independent of the 10-substep policy decimation.
        self.sim.render_interval = CAMERA_CAPTURE_INTERVAL_STEPS


__all__ = [
    "CAMERA_CAPTURE_INTERVAL_STEPS",
    "CAMERA_POSE_AUTHORITY",
    "CAMERA_RESOLUTION",
    "CAMERA_RUNTIME_POSE_OVERRIDE",
    "G2LiftRgbdEnvCfg",
    "HEAD_CAMERA_RELATIVE_PRIM",
    "RIGHT_WRIST_CAMERA_RELATIVE_PRIM",
]
