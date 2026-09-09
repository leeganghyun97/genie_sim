"""Quaternion boundary contract for the migrated G2 Isaac Lab pipeline.

All persisted observations, datasets, replay and auxiliary targets use
``XYZW``.  Isaac Lab's native tensor order is an implementation detail: 2.x
uses WXYZ while 3.x uses XYZW.  Runtime code must convert exactly once at the
framework boundary instead of assuming that the two releases agree.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import torch


QuaternionOrder = Literal["xyzw", "wxyz"]
CANONICAL_QUATERNION_ORDER: QuaternionOrder = "xyzw"


def canonicalize_quaternion_xyzw(
    quaternion: torch.Tensor, *, minimum_norm: float = 1.0e-8
) -> torch.Tensor:
    """Normalize an XYZW quaternion and choose one deterministic sign.

    The scalar component is non-negative.  For an exact 180-degree rotation,
    where ``w == 0``, the largest-magnitude vector component is made positive.
    This removes the otherwise arbitrary ``q``/``-q`` discontinuity from
    observations and targets without changing the represented rotation.
    """

    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must have shape [...,4]")
    if not bool(torch.isfinite(quaternion).all()):
        raise ValueError("quaternion contains non-finite values")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if bool((norm < float(minimum_norm)).any()):
        raise ValueError("quaternion norm is zero or too small")
    normalized = quaternion / norm
    scalar = normalized[..., 3]
    vector = normalized[..., :3]
    dominant = torch.gather(
        vector, -1, vector.abs().argmax(dim=-1, keepdim=True)
    ).squeeze(-1)
    flip = torch.where(scalar.abs() > 1.0e-8, scalar < 0.0, dominant < 0.0)
    return torch.where(flip.unsqueeze(-1), -normalized, normalized)


def quaternion_native_to_xyzw(
    quaternion: torch.Tensor, native_order: QuaternionOrder
) -> torch.Tensor:
    """Convert one framework-native quaternion tensor to canonical XYZW."""

    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must have shape [...,4]")
    if native_order == "xyzw":
        converted = quaternion
    elif native_order == "wxyz":
        converted = quaternion[..., (1, 2, 3, 0)]
    else:
        raise ValueError(f"unsupported quaternion order: {native_order!r}")
    return canonicalize_quaternion_xyzw(converted)


def quaternion_xyzw_to_native(
    quaternion: torch.Tensor, native_order: QuaternionOrder
) -> torch.Tensor:
    """Convert canonical XYZW to a framework-native normalized quaternion."""

    canonical = canonicalize_quaternion_xyzw(quaternion)
    if native_order == "xyzw":
        return canonical
    if native_order == "wxyz":
        return canonical[..., (3, 0, 1, 2)]
    raise ValueError(f"unsupported quaternion order: {native_order!r}")


def quaternion_order_from_quat_unique_doc(docstring: str | None) -> QuaternionOrder:
    """Infer the installed Isaac Lab convention from its public API docs."""

    compact = " ".join((docstring or "").lower().split())
    if "(x, y, z, w)" in compact or "xyzw" in compact:
        return "xyzw"
    if "(w, x, y, z)" in compact or "wxyz" in compact:
        return "wxyz"
    raise RuntimeError(
        "cannot establish installed Isaac Lab quaternion order; refusing to "
        "guess at the runtime boundary"
    )


@lru_cache(maxsize=1)
def isaaclab_native_quaternion_order() -> QuaternionOrder:
    """Return the installed Isaac Lab quaternion convention, fail-closed."""

    from isaaclab.utils.math import quat_unique

    return quaternion_order_from_quat_unique_doc(quat_unique.__doc__)


def native_identity_quaternion() -> tuple[float, float, float, float]:
    """Identity quaternion in the installed Isaac Lab native order."""

    order = isaaclab_native_quaternion_order()
    return (0.0, 0.0, 0.0, 1.0) if order == "xyzw" else (1.0, 0.0, 0.0, 0.0)


__all__ = [
    "CANONICAL_QUATERNION_ORDER",
    "QuaternionOrder",
    "canonicalize_quaternion_xyzw",
    "isaaclab_native_quaternion_order",
    "native_identity_quaternion",
    "quaternion_native_to_xyzw",
    "quaternion_order_from_quat_unique_doc",
    "quaternion_xyzw_to_native",
]
