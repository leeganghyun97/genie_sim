"""Fail-closed readiness checks for long G2 visual training.

The checked-in runtime USD intentionally disables most arm/body colliders and
articulation self-collision.  A successful vector allocation is therefore not
evidence that a 50M-transition run has a live forbidden-collision authority.
This module keeps that distinction machine-readable for the supervisor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

from .g2_lift_methodology import G2_USD_PATH, G2_USD_SHA256


@dataclass(frozen=True)
class G2LongTrainingReadiness:
    usd_path: str
    usd_sha256_contract: str
    articulation_self_collision_enabled: bool
    right_arm_collision_links_enabled: tuple[str, ...]
    right_arm_collision_links_missing: tuple[str, ...]
    live_full_body_forbidden_collision_authority: bool
    long_training_approved: bool
    blocker: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _link_block(text: str, link: str) -> str:
    marker = f'over "{link}"'
    start = text.find(marker)
    if start < 0:
        return ""
    next_link = re.search(r'\n\s*over "arm_[lr]_link\d+"', text[start + len(marker):])
    if next_link is None:
        return text[start:]
    return text[start:start + len(marker) + next_link.start()]


def audit_g2_long_training_readiness(
    usd_path: str | Path = G2_USD_PATH,
) -> G2LongTrainingReadiness:
    """Audit only source-backed collision facts; never infer a PASS from motion."""

    path = Path(usd_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    self_collision = bool(
        re.search(r"physxArticulation:enabledSelfCollisions\s*=\s*1", text)
    )
    enabled: list[str] = []
    missing: list[str] = []
    for index in range(1, 8):
        link = f"arm_r_link{index}"
        block = _link_block(text, link)
        # The link is considered covered only when its authored convex collider
        # is enabled.  Gripper SDF colliders cannot attest an arm-link impact.
        has_enabled_collision = bool(
            re.search(
                rf'over "{re.escape(link)}_convex_\d+"\s*'
                r"(?:\([^)]*\)\s*)?\{[^}]*"
                r"?physics:collisionEnabled\s*=\s*1",
                block,
            )
        )
        (enabled if has_enabled_collision else missing).append(link)
    live_authority = self_collision and not missing
    blocker = None if live_authority else "G2_FULL_BODY_COLLISION_AUTHORITY_UNAVAILABLE"
    return G2LongTrainingReadiness(
        usd_path=str(path.resolve()),
        usd_sha256_contract=G2_USD_SHA256,
        articulation_self_collision_enabled=self_collision,
        right_arm_collision_links_enabled=tuple(enabled),
        right_arm_collision_links_missing=tuple(missing),
        live_full_body_forbidden_collision_authority=live_authority,
        long_training_approved=live_authority,
        blocker=blocker,
    )


__all__ = ["G2LongTrainingReadiness", "audit_g2_long_training_readiness"]
