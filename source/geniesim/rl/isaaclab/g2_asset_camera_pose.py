"""G2 asset-authored camera mounts composed with live articulation poses.

The runtime URDF describes the rigid links and the checked-in G2 USD carries
the calibrated optical Camera prims.  Isaac Lab 3's generic Camera FrameView
can return the authored (initial) world transform for a Camera nested below a
PhysX articulation link.  This module keeps the authored local transform as
the calibration authority and composes it with the measured parent-body pose;
it never writes or teleports a camera prim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .g2_quaternion import isaaclab_native_quaternion_order, quaternion_native_to_xyzw
from .g2_redundancy_teleop import tensor_value


CAMERA_POSE_AUTHORITY = "URDF_LINK_PLUS_CHECKED_IN_G2_USD_OPTICAL_EXTRINSIC"
CAMERA_RUNTIME_POSE_OVERRIDE = False
HEAD_CAMERA_RELATIVE_PRIM = "Robot/head_link3/head_front_Camera"
RIGHT_WRIST_CAMERA_RELATIVE_PRIM = "Robot/gripper_r_base_link/Right_Camera"


@dataclass(frozen=True)
class G2CameraMount:
    parent_body: str
    relative_prim: str


G2_CAMERA_MOUNTS = {
    "head": G2CameraMount("head_link3", HEAD_CAMERA_RELATIVE_PRIM),
    "right_wrist": G2CameraMount(
        "gripper_r_base_link", RIGHT_WRIST_CAMERA_RELATIVE_PRIM
    ),
}


class G2AssetCameraPoseResolver:
    """Resolve live camera poses without trusting stale Camera FrameView data."""

    def __init__(self, env) -> None:
        import omni.usd
        from pxr import UsdGeom

        self.env = env
        self.robot = env.scene["robot"]
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("G2_CAMERA_POSE_STAGE_UNAVAILABLE")
        self._parent_body_indices = {
            name: self.robot.body_names.index(mount.parent_body)
            for name, mount in G2_CAMERA_MOUNTS.items()
        }
        self._local: dict[str, torch.Tensor] = {}
        self._local_serializable: dict[str, list[list[float]]] = {}
        clone_in_fabric = bool(
            getattr(getattr(env.scene, "cfg", None), "clone_in_fabric", False)
        )
        for name, mount in G2_CAMERA_MOUNTS.items():
            matrices = []
            for env_id in range(env.num_envs):
                path = f"/World/envs/env_{env_id}/{mount.relative_prim}"
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid() and clone_in_fabric and env_id > 0:
                    # Fabric clones are not authored as duplicate USD prims.
                    # Their local mount is definitionally the env_0 source
                    # extrinsic and their live parent pose remains per-env.
                    if not matrices:
                        raise RuntimeError(f"G2_CAMERA_POSE_PRIM_MISSING:{path}")
                    matrices.append(matrices[0].copy())
                    continue
                if not prim.IsValid() or prim.GetTypeName() != "Camera":
                    raise RuntimeError(f"G2_CAMERA_POSE_PRIM_MISSING:{path}")
                # USD Gf matrices are row-vector based.  Transpose into the
                # column-vector homogeneous convention used by the controller.
                matrices.append(
                    np.asarray(
                        UsdGeom.Xformable(prim).GetLocalTransformation(),
                        dtype=np.float64,
                    ).T
                )
            first = matrices[0]
            if any(not np.allclose(first, matrix, atol=1.0e-12) for matrix in matrices[1:]):
                raise RuntimeError(f"G2_CAMERA_LOCAL_EXTRINSIC_CROSS_ENV_MISMATCH:{name}")
            self._local[name] = torch.as_tensor(
                np.stack(matrices), dtype=torch.float32, device=env.device
            )
            self._local_serializable[name] = first.tolist()

    def world_transform(self, name: str) -> torch.Tensor:
        if name not in G2_CAMERA_MOUNTS:
            raise KeyError(name)
        body_index = self._parent_body_indices[name]
        positions = tensor_value(self.robot.data.body_pos_w)[:, body_index]
        quaternions = tensor_value(self.robot.data.body_quat_w)[:, body_index]
        # This path is used by vector reset visibility checks.  Building one
        # Python object per environment made camera validation scale linearly
        # on the CPU, so compose all parent transforms on the GPU at once.
        q = quaternion_native_to_xyzw(
            quaternions, isaaclab_native_quaternion_order()
        )
        x, y, z, w = q.unbind(dim=-1)
        rotation = torch.stack(
            (
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ),
            dim=-1,
        ).reshape(-1, 3, 3)
        parent_world = torch.eye(
            4, dtype=positions.dtype, device=positions.device
        ).expand(self.env.num_envs, -1, -1).clone()
        parent_world[:, :3, :3] = rotation
        parent_world[:, :3, 3] = positions
        return torch.bmm(parent_world, self._local[name])

    def world_position(self, name: str) -> torch.Tensor:
        return self.world_transform(name)[:, :3, 3]

    def contract(self) -> dict[str, object]:
        return {
            "authority": CAMERA_POSE_AUTHORITY,
            "runtime_pose_override": CAMERA_RUNTIME_POSE_OVERRIDE,
            "mounts": {
                name: {
                    "parent_body": mount.parent_body,
                    "relative_prim": mount.relative_prim,
                    "parent_from_optical_4x4": self._local_serializable[name],
                }
                for name, mount in G2_CAMERA_MOUNTS.items()
            },
        }


__all__ = [
    "CAMERA_POSE_AUTHORITY",
    "CAMERA_RUNTIME_POSE_OVERRIDE",
    "G2AssetCameraPoseResolver",
    "G2_CAMERA_MOUNTS",
    "HEAD_CAMERA_RELATIVE_PRIM",
    "RIGHT_WRIST_CAMERA_RELATIVE_PRIM",
]
