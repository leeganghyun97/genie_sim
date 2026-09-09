"""GPU-native RGB-D back-projection used by the G2 Milestone 5/6 gates."""

from __future__ import annotations

import torch

from .g2_quaternion import (
    QuaternionOrder,
    canonicalize_quaternion_xyzw,
    isaaclab_native_quaternion_order,
    quaternion_native_to_xyzw,
)


def transform_from_pose_xyzw(position: torch.Tensor, quaternion_xyzw: torch.Tensor) -> torch.Tensor:
    """Build a homogeneous transform without leaving the tensor device."""

    if position.shape != (3,) or quaternion_xyzw.shape != (4,):
        raise ValueError("position/quaternion shapes must be 3 and 4")
    if not bool(torch.isfinite(position).all()):
        raise ValueError("position contains non-finite values")
    q = canonicalize_quaternion_xyzw(quaternion_xyzw)
    x, y, z, w = q.unbind()
    row0 = torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)))
    row1 = torch.stack((2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)))
    row2 = torch.stack((2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)))
    result = torch.eye(4, device=position.device, dtype=position.dtype)
    result[:3, :3] = torch.stack((row0, row1, row2))
    result[:3, 3] = position
    return result


def transform_from_pose_native(
    position: torch.Tensor,
    quaternion_native: torch.Tensor,
    *,
    native_order: QuaternionOrder | None = None,
) -> torch.Tensor:
    """Build a transform from the installed Isaac Lab native pose contract.

    Live camera/body tensors must cross this boundary. Persisted data and the
    geometry helpers themselves remain canonical XYZW.
    """

    order = native_order or isaaclab_native_quaternion_order()
    return transform_from_pose_xyzw(
        position,
        quaternion_native_to_xyzw(quaternion_native, order),
    )


def backproject_opengl_depth(
    depth_m: torch.Tensor,
    intrinsic_3x3: torch.Tensor,
    camera_to_world_4x4: torch.Tensor,
    *,
    stride: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Back-project metric depth from Isaac's OpenGL optical convention."""

    if depth_m.ndim == 3 and depth_m.shape[-1] == 1:
        depth_m = depth_m[..., 0]
    if depth_m.ndim != 2:
        raise ValueError(f"depth must be HxW or HxWx1, got {tuple(depth_m.shape)}")
    if intrinsic_3x3.shape != (3, 3) or camera_to_world_4x4.shape != (4, 4):
        raise ValueError("intrinsic and transform shapes must be 3x3 and 4x4")
    if stride < 1:
        raise ValueError("stride must be positive")
    depth = depth_m[::stride, ::stride]
    height, width = depth_m.shape
    rows = torch.arange(0, height, stride, device=depth.device, dtype=depth.dtype)
    cols = torch.arange(0, width, stride, device=depth.device, dtype=depth.dtype)
    vv, uu = torch.meshgrid(rows, cols, indexing="ij")
    valid = torch.isfinite(depth) & (depth > 0.0)
    x = (uu - intrinsic_3x3[0, 2]) * depth / intrinsic_3x3[0, 0]
    y = -(vv - intrinsic_3x3[1, 2]) * depth / intrinsic_3x3[1, 1]
    optical_h = torch.stack((x, y, -depth, torch.ones_like(depth)), dim=-1)
    world = torch.matmul(optical_h, camera_to_world_4x4.transpose(0, 1))[..., :3]
    return world[valid], valid


def fuse_workspace_clouds(
    clouds_world: list[torch.Tensor],
    *,
    workspace_min_world_m: torch.Tensor,
    workspace_max_world_m: torch.Tensor,
) -> torch.Tensor:
    """Concatenate finite camera clouds and clip to an explicit workspace."""

    if not clouds_world:
        raise ValueError("at least one cloud is required")
    device, dtype = clouds_world[0].device, clouds_world[0].dtype
    lower = workspace_min_world_m.to(device=device, dtype=dtype)
    upper = workspace_max_world_m.to(device=device, dtype=dtype)
    if lower.shape != (3,) or upper.shape != (3,) or not bool(torch.all(lower < upper)):
        raise ValueError("workspace bounds must be ordered xyz vectors")
    merged = torch.cat(clouds_world, dim=0)
    finite = torch.isfinite(merged).all(dim=-1)
    inside = ((merged >= lower) & (merged <= upper)).all(dim=-1)
    return merged[finite & inside]


__all__ = [
    "backproject_opengl_depth",
    "fuse_workspace_clouds",
    "transform_from_pose_native",
    "transform_from_pose_xyzw",
]
