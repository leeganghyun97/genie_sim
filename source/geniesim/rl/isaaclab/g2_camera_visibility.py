"""Fail-closed G2 cube-frustum checks for fixed-torso task resets.

The deployment actor still receives only RGB-D and proprioception.  Cube
ground truth is consumed here strictly as reset/safety geometry, just like a
joint-limit or reachability gate; it is never appended to the actor input.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .g2_asset_camera_pose import G2AssetCameraPoseResolver
from .g2_lift_methodology import G2LiftSandboxContract
from .g2_redundancy_teleop import tensor_value


CAMERA_NAMES = ("head", "right_wrist")


@dataclass(frozen=True)
class G2CameraFrustumResult:
    center_uv_normalized: torch.Tensor
    center_depth_m: torch.Tensor
    minimum_normalized_edge_margin: torch.Tensor
    all_cube_corners_in_frustum: torch.Tensor
    any_cube_part_in_frustum: torch.Tensor
    rendered_center_depth_m: torch.Tensor | None = None
    rendered_depth_consistent: torch.Tensor | None = None
    rendered_cube_visible_pixel_count: torch.Tensor | None = None
    rendered_cube_visible: torch.Tensor | None = None


@dataclass(frozen=True)
class G2DualCameraVisibilityResult:
    cameras: dict[str, G2CameraFrustumResult]
    visible_camera_count: torch.Tensor
    reset_pass: torch.Tensor
    precontact_pass: torch.Tensor

    def serializable(self) -> dict[str, object]:
        return {
            "camera_ground_truth_usage": "RESET_SAFETY_GATE_ONLY_NOT_ACTOR_OBSERVATION",
            "visible_camera_count": self.visible_camera_count.detach().cpu().tolist(),
            "reset_pass": self.reset_pass.detach().cpu().tolist(),
            "precontact_pass": self.precontact_pass.detach().cpu().tolist(),
            "cameras": {
                name: {
                    "center_uv_normalized": value.center_uv_normalized.detach().cpu().tolist(),
                    "center_depth_m": value.center_depth_m.detach().cpu().tolist(),
                    "minimum_normalized_edge_margin": (
                        value.minimum_normalized_edge_margin.detach().cpu().tolist()
                    ),
                    "all_cube_corners_in_frustum": (
                        value.all_cube_corners_in_frustum.detach().cpu().tolist()
                    ),
                    "any_cube_part_in_frustum": (
                        value.any_cube_part_in_frustum.detach().cpu().tolist()
                    ),
                    "rendered_center_depth_m": (
                        None
                        if value.rendered_center_depth_m is None
                        else value.rendered_center_depth_m.detach().cpu().tolist()
                    ),
                    "rendered_depth_consistent": (
                        None
                        if value.rendered_depth_consistent is None
                        else value.rendered_depth_consistent.detach().cpu().tolist()
                    ),
                    "rendered_cube_visible_pixel_count": (
                        None
                        if value.rendered_cube_visible_pixel_count is None
                        else value.rendered_cube_visible_pixel_count.detach().cpu().tolist()
                    ),
                    "rendered_cube_visible": (
                        None
                        if value.rendered_cube_visible is None
                        else value.rendered_cube_visible.detach().cpu().tolist()
                    ),
                }
                for name, value in self.cameras.items()
            },
        }


def cube_points_world(
    cube_center_world_m: torch.Tensor,
    *,
    cube_edge_m: float | None = None,
    cube_size_m: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """Return center followed by the eight physical cuboid corners.

    ``cube_edge_m`` remains for old diagnostic callers. Runtime task code uses
    the explicit XYZ size so a 40 x 40 x 60 mm object is never projected as a
    cube.
    """

    if cube_center_world_m.ndim != 2 or cube_center_world_m.shape[1] != 3:
        raise ValueError("cube center must have shape [N,3]")
    if not bool(torch.isfinite(cube_center_world_m).all()):
        raise ValueError("cube center contains non-finite values")
    if cube_size_m is None:
        if cube_edge_m is None or cube_edge_m <= 0.0:
            raise ValueError("cube edge must be positive")
        cube_size_m = (cube_edge_m, cube_edge_m, cube_edge_m)
    if len(cube_size_m) != 3 or any(value <= 0.0 for value in cube_size_m):
        raise ValueError("cube size must contain three positive dimensions")
    signs = torch.tensor(
        [
            (-1.0, -1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
            (1.0, 1.0, 1.0),
        ],
        dtype=cube_center_world_m.dtype,
        device=cube_center_world_m.device,
    )
    half_extents = torch.tensor(
        cube_size_m, dtype=cube_center_world_m.dtype,
        device=cube_center_world_m.device,
    ) / 2.0
    corners = cube_center_world_m[:, None, :] + signs[None] * half_extents
    return torch.cat((cube_center_world_m[:, None, :], corners), dim=1)


def project_opengl_points(
    points_world_m: torch.Tensor,
    camera_to_world_4x4: torch.Tensor,
    intrinsic_3x3: torch.Tensor,
    *,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world points using Isaac's OpenGL camera convention.

    Optical forward is local ``-Z`` and image ``v`` grows downward.  Returned
    image coordinates are normalized by ``width-1`` and ``height-1``.
    """

    if points_world_m.ndim != 3 or points_world_m.shape[-1] != 3:
        raise ValueError("points must have shape [N,P,3]")
    batch = points_world_m.shape[0]
    if camera_to_world_4x4.shape != (batch, 4, 4):
        raise ValueError("camera transforms must have shape [N,4,4]")
    if intrinsic_3x3.ndim == 2:
        intrinsic_3x3 = intrinsic_3x3.unsqueeze(0).expand(batch, -1, -1)
    if intrinsic_3x3.shape != (batch, 3, 3):
        raise ValueError("intrinsics must have shape [N,3,3] or [3,3]")
    if image_width < 2 or image_height < 2:
        raise ValueError("image dimensions must be at least two pixels")
    ones = torch.ones(
        (*points_world_m.shape[:-1], 1),
        dtype=points_world_m.dtype,
        device=points_world_m.device,
    )
    points_h = torch.cat((points_world_m, ones), dim=-1)
    rotation_t = camera_to_world_4x4[:, :3, :3].transpose(1, 2)
    translation = camera_to_world_4x4[:, :3, 3]
    world_to_camera = torch.eye(
        4, dtype=camera_to_world_4x4.dtype, device=camera_to_world_4x4.device
    ).expand(batch, -1, -1).clone()
    world_to_camera[:, :3, :3] = rotation_t
    world_to_camera[:, :3, 3] = -torch.bmm(
        rotation_t, translation.unsqueeze(-1)
    ).squeeze(-1)
    optical = torch.bmm(points_h, world_to_camera.transpose(1, 2))[..., :3]
    depth = -optical[..., 2]
    safe_depth = torch.where(depth.abs() > 1.0e-12, depth, torch.ones_like(depth))
    u_px = intrinsic_3x3[:, None, 0, 0] * optical[..., 0] / safe_depth
    u_px = u_px + intrinsic_3x3[:, None, 0, 2]
    v_px = intrinsic_3x3[:, None, 1, 2]
    v_px = v_px - intrinsic_3x3[:, None, 1, 1] * optical[..., 1] / safe_depth
    normalized = torch.stack(
        (u_px / float(image_width - 1), v_px / float(image_height - 1)), dim=-1
    )
    return normalized, depth


def evaluate_camera_frustum(
    points_world_m: torch.Tensor,
    camera_to_world_4x4: torch.Tensor,
    intrinsic_3x3: torch.Tensor,
    *,
    image_width: int,
    image_height: int,
    minimum_normalized_edge_margin: float,
    minimum_depth_m: float,
) -> G2CameraFrustumResult:
    uv, depth = project_opengl_points(
        points_world_m,
        camera_to_world_4x4,
        intrinsic_3x3,
        image_width=image_width,
        image_height=image_height,
    )
    corner_uv = uv[:, 1:]
    corner_depth = depth[:, 1:]
    edge_margin = torch.stack(
        (
            corner_uv[..., 0],
            corner_uv[..., 1],
            1.0 - corner_uv[..., 0],
            1.0 - corner_uv[..., 1],
        ),
        dim=-1,
    ).amin(dim=(-1, -2))
    visible = (
        torch.isfinite(corner_uv).all(dim=(-1, -2))
        & torch.isfinite(corner_depth).all(dim=-1)
        & (corner_depth >= minimum_depth_m).all(dim=-1)
        & (edge_margin >= minimum_normalized_edge_margin)
    )
    # Partial visibility is deliberately less strict than the diagnostic
    # all-corners margin.  Center plus eight corners is enough for this small
    # cube; live RGB-D evidence below remains the final visibility authority.
    finite_points = torch.isfinite(uv).all(dim=-1) & torch.isfinite(depth)
    point_inside = (
        finite_points
        & (depth >= minimum_depth_m)
        & (uv[..., 0] >= 0.0)
        & (uv[..., 0] <= 1.0)
        & (uv[..., 1] >= 0.0)
        & (uv[..., 1] <= 1.0)
    )
    return G2CameraFrustumResult(
        center_uv_normalized=uv[:, 0],
        center_depth_m=depth[:, 0],
        minimum_normalized_edge_margin=edge_margin,
        all_cube_corners_in_frustum=visible,
        any_cube_part_in_frustum=point_inside.any(dim=-1),
    )


def rendered_cube_rgbd_evidence(
    rgb: torch.Tensor,
    depth_m: torch.Tensor,
    center_uv_normalized: torch.Tensor,
    center_depth_m: torch.Tensor,
    intrinsic_3x3: torch.Tensor,
    *,
    cube_edge_m: float | None = None,
    cube_size_m: tuple[float, float, float] | None = None,
    minimum_pixel_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count projected cube-colour pixels whose metric depth is plausible.

    This is reset/safety telemetry, not an actor observation.  The projection
    restricts the test to a bounded ROI around the known cube geometry; the
    checked-in red material and metric depth jointly reject robot/table pixels.
    A partially occluded cube remains visible even when its exact centre ray is
    covered by a fingertip.
    """

    if rgb.ndim != 4 or rgb.shape[-1] < 3:
        raise RuntimeError(f"G2_CAMERA_RGB_SHAPE_INVALID:{tuple(rgb.shape)}")
    if depth_m.ndim == 4 and depth_m.shape[-1] == 1:
        depth_m = depth_m[..., 0]
    if depth_m.ndim != 3 or depth_m.shape[:3] != rgb.shape[:3]:
        raise RuntimeError(
            f"G2_CAMERA_RGBD_SHAPE_MISMATCH:{tuple(rgb.shape)}:{tuple(depth_m.shape)}"
        )
    batch, height, width = depth_m.shape
    if center_uv_normalized.shape != (batch, 2) or center_depth_m.shape != (batch,):
        raise RuntimeError("G2_CAMERA_PROJECTION_BATCH_MISMATCH")
    if intrinsic_3x3.ndim == 2:
        intrinsic_3x3 = intrinsic_3x3.unsqueeze(0).expand(batch, -1, -1)

    rgb_255 = rgb[..., :3].to(dtype=torch.float32)
    if rgb.dtype != torch.uint8:
        # Isaac Camera RGB is uint8 today; retaining this path makes a future
        # normalized float output obey the identical physical threshold.
        rgb_255 = rgb_255 * 255.0
    red = rgb_255[..., 0]
    green = rgb_255[..., 1]
    blue = rgb_255[..., 2]
    # A ratio is intentionally stricter than a small channel difference: the
    # beige tabletop can have R>G/B, whereas the checked-in cube remains at
    # least 20% red-dominant in the live RTX frames.
    cube_colour = (
        (red >= 64.0) & (red >= 1.20 * green) & (red >= 1.20 * blue)
    )

    rows = torch.arange(height, device=depth_m.device).view(1, height, 1)
    cols = torch.arange(width, device=depth_m.device).view(1, 1, width)
    center_u_px = center_uv_normalized[:, 0] * float(width - 1)
    center_v_px = center_uv_normalized[:, 1] * float(height - 1)
    focal_px = torch.maximum(intrinsic_3x3[:, 0, 0], intrinsic_3x3[:, 1, 1])
    if cube_size_m is None:
        if cube_edge_m is None or cube_edge_m <= 0.0:
            raise ValueError("cube edge must be positive")
        cube_size_m = (cube_edge_m, cube_edge_m, cube_edge_m)
    if len(cube_size_m) != 3 or any(value <= 0.0 for value in cube_size_m):
        raise ValueError("cube size must contain three positive dimensions")
    half_diagonal_m = math.sqrt(sum(value * value for value in cube_size_m)) / 2.0
    radius_px = torch.ceil(
        focal_px * half_diagonal_m / center_depth_m.clamp_min(1.0e-6) + 2.0
    )
    roi = (
        (torch.abs(cols - center_u_px[:, None, None]) <= radius_px[:, None, None])
        & (torch.abs(rows - center_v_px[:, None, None]) <= radius_px[:, None, None])
    )
    depth_tolerance_m = half_diagonal_m + 0.010
    depth_consistent = (
        torch.isfinite(depth_m)
        & (depth_m > 0.0)
        & (
            torch.abs(depth_m - center_depth_m[:, None, None])
            <= depth_tolerance_m
        )
    )
    count = (roi & cube_colour & depth_consistent).sum(dim=(-1, -2))
    return count, count >= int(minimum_pixel_count)


class G2DualCameraVisibilityEvaluator:
    """Evaluate the checked-in head/wrist optical mounts against the cube."""

    def __init__(
        self,
        env,
        *,
        sandbox: G2LiftSandboxContract | None = None,
        pose_resolver: G2AssetCameraPoseResolver | None = None,
    ) -> None:
        self.env = env
        self.sandbox = (sandbox or G2LiftSandboxContract()).validated()
        self.pose_resolver = pose_resolver or G2AssetCameraPoseResolver(env)
        self.sensors = {
            "head": env.scene["head_camera"],
            "right_wrist": env.scene["right_wrist_camera"],
        }

    def evaluate(
        self,
        cube_center_world_m: torch.Tensor | None = None,
        *,
        require_rendered_depth: bool = False,
    ) -> G2DualCameraVisibilityResult:
        if cube_center_world_m is None:
            cube_center_world_m = tensor_value(
                self.env.scene["object"].data.root_pos_w
            )[:, :3]
        points = cube_points_world(
            cube_center_world_m, cube_size_m=self.sandbox.cube_size_m
        )
        cameras: dict[str, G2CameraFrustumResult] = {}
        for name in CAMERA_NAMES:
            sensor = self.sensors[name]
            intrinsic = tensor_value(sensor.data.intrinsic_matrices).to(
                device=points.device, dtype=points.dtype
            )
            frustum = evaluate_camera_frustum(
                points,
                self.pose_resolver.world_transform(name).to(dtype=points.dtype),
                intrinsic,
                image_width=int(sensor.cfg.width),
                image_height=int(sensor.cfg.height),
                minimum_normalized_edge_margin=(
                    self.sandbox.camera_normalized_edge_margin
                ),
                minimum_depth_m=self.sandbox.camera_minimum_depth_m,
            )
            if require_rendered_depth:
                rendered = tensor_value(
                    sensor.data.output["distance_to_image_plane"]
                )
                if rendered.ndim == 4 and rendered.shape[-1] == 1:
                    rendered = rendered[..., 0]
                if rendered.ndim != 3 or rendered.shape[0] != points.shape[0]:
                    raise RuntimeError(
                        f"G2_CAMERA_DEPTH_SHAPE_INVALID:{name}:{tuple(rendered.shape)}"
                    )
                u = torch.round(
                    frustum.center_uv_normalized[:, 0] * float(sensor.cfg.width - 1)
                ).to(torch.long)
                v = torch.round(
                    frustum.center_uv_normalized[:, 1] * float(sensor.cfg.height - 1)
                ).to(torch.long)
                pixel_inside = (
                    (u >= 0)
                    & (u < int(sensor.cfg.width))
                    & (v >= 0)
                    & (v < int(sensor.cfg.height))
                )
                safe_u = u.clamp(0, int(sensor.cfg.width) - 1)
                safe_v = v.clamp(0, int(sensor.cfg.height) - 1)
                env_ids = torch.arange(points.shape[0], device=points.device)
                rendered_center_depth = rendered[env_ids, safe_v, safe_u]
                # The observed ray terminates at the cube surface, not its
                # center.  Allow half the cube diagonal plus 20 mm for pixel
                # quantization/render depth precision; a table/background hit
                # remains far outside this bounded interval.
                maximum_surface_offset_m = (
                    math.sqrt(
                        sum(value * value for value in self.sandbox.cube_size_m)
                    ) / 2.0 + 0.010
                )
                center_minus_rendered = (
                    frustum.center_depth_m - rendered_center_depth
                )
                depth_consistent = (
                    pixel_inside
                    & torch.isfinite(rendered_center_depth)
                    & (rendered_center_depth > 0.0)
                    # A visible cube surface must terminate the ray at or in
                    # front of its center.  A bare table hit behind the cube
                    # therefore cannot satisfy the test.  Two millimetres are
                    # reserved for renderer precision/pixel quantization.
                    & (center_minus_rendered >= -0.002)
                    & (center_minus_rendered <= maximum_surface_offset_m)
                )
                rgb = tensor_value(sensor.data.output["rgb"])
                visible_pixel_count, rendered_cube_visible = (
                    rendered_cube_rgbd_evidence(
                        rgb,
                        rendered,
                        frustum.center_uv_normalized,
                        frustum.center_depth_m,
                        intrinsic,
                        cube_size_m=self.sandbox.cube_size_m,
                        minimum_pixel_count=(
                            self.sandbox.camera_minimum_cube_visible_pixels
                        ),
                    )
                )
                frustum = G2CameraFrustumResult(
                    center_uv_normalized=frustum.center_uv_normalized,
                    center_depth_m=frustum.center_depth_m,
                    minimum_normalized_edge_margin=(
                        frustum.minimum_normalized_edge_margin
                    ),
                    # Keep geometric containment separate from rendered
                    # evidence.  The centre ray remains diagnostic only.
                    all_cube_corners_in_frustum=frustum.all_cube_corners_in_frustum,
                    any_cube_part_in_frustum=frustum.any_cube_part_in_frustum,
                    rendered_center_depth_m=rendered_center_depth,
                    rendered_depth_consistent=depth_consistent,
                    rendered_cube_visible_pixel_count=visible_pixel_count,
                    rendered_cube_visible=rendered_cube_visible,
                )
            cameras[name] = frustum
        visible_count = torch.stack(
            [
                cameras[name].any_cube_part_in_frustum
                & (
                    cameras[name].rendered_cube_visible
                    if require_rendered_depth
                    else torch.ones_like(cameras[name].all_cube_corners_in_frustum)
                )
                for name in CAMERA_NAMES
            ],
            dim=-1,
        ).sum(dim=-1)
        return G2DualCameraVisibilityResult(
            cameras=cameras,
            visible_camera_count=visible_count,
            reset_pass=visible_count >= self.sandbox.reset_required_visible_cameras,
            precontact_pass=(
                visible_count >= self.sandbox.precontact_minimum_visible_cameras
            ),
        )


def precontact_camera_visibility_failure(env) -> torch.Tensor:
    """Terminate only if both asset cameras lose all cube evidence before contact.

    The privileged cube center is used solely for this safety/observability
    gate.  It is not exposed in the visual actor observation.  Bilateral
    contact switches authority to contact and proprioception because normal
    finger occlusion is expected during a grasp.
    """

    # Reset settling/reverse-preroll occurs before an RL episode exists and
    # may legitimately observe repeated or not-yet-fresh camera frames.  The
    # runner performs an explicit dual-camera visibility acceptance check at
    # the commit boundary.  Suppress only this observability termination while
    # that bounded initialization transaction is active; physical/collision
    # terminations remain authoritative.
    if bool(getattr(env, "_g2_reset_initialization_active", False)):
        env._g2_precontact_camera_loss_age_s = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_g2_camera_visibility_evaluator"):
        env._g2_camera_visibility_evaluator = G2DualCameraVisibilityEvaluator(env)
    result = env._g2_camera_visibility_evaluator.evaluate(
        require_rendered_depth=True
    )
    from .g2_lift_task_mdp import contact_grasp_telemetry

    _, _, bilateral, _, _ = contact_grasp_telemetry(env)
    loss_now = (~result.precontact_pass) & (~bilateral)
    previous_age = getattr(env, "_g2_precontact_camera_loss_age_s", None)
    if previous_age is None or tuple(previous_age.shape) != (env.num_envs,):
        previous_age = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )
    age_s = torch.where(
        loss_now,
        previous_age + float(env.step_dt),
        torch.zeros_like(previous_age),
    )
    env._g2_precontact_camera_loss_age_s = age_s
    return loss_now & (age_s >= result_sandbox_camera_loss_grace_s(env))


def result_sandbox_camera_loss_grace_s(env) -> float:
    """Return the sealed recurrent occlusion grace used by the env gate."""

    evaluator = env._g2_camera_visibility_evaluator
    return float(evaluator.sandbox.precontact_camera_loss_grace_s)


__all__ = [
    "CAMERA_NAMES",
    "G2CameraFrustumResult",
    "G2DualCameraVisibilityEvaluator",
    "G2DualCameraVisibilityResult",
    "cube_points_world",
    "evaluate_camera_frustum",
    "project_opengl_points",
    "rendered_cube_rgbd_evidence",
    "precontact_camera_visibility_failure",
]
