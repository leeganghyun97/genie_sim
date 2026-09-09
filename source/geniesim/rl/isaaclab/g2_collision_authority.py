"""Runtime forbidden-collision authority for the G2 manipulation task.

The checked-in USD is left untouched.  Isaac Lab enables the already-authored
collision schemas in the session layer and these sensors separate the only
allowed task contact (right distal fingers against the cube) from every other
robot contact.  A zero value is valid only after every required sensor has
produced a finite tensor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

import torch


G2_RUNTIME_COLLISION_AUTHORITY_SCHEMA = (
    "g2_runtime_forbidden_collision_authority_v2"
)
G2_FORBIDDEN_COLLISION_FORCE_EPSILON_N = 1.0e-6
G2_SELECTIVE_COLLISION_OVERRIDE_SCHEMA = (
    "g2_closed_manifest_collision_and_filtered_pairs_override_v2"
)

# Multi-body, unfiltered sensors are valid in Isaac Lab.  They cover bodies
# for which *any* external contact is forbidden.  The two right distal pads
# are separate one-body filtered sensors because their cube contact is allowed.
G2_GENERAL_FORBIDDEN_CONTACT_SENSORS = (
    "forbidden_torso_contact",
    "forbidden_head_contact",
    "forbidden_arm_contact",
    "forbidden_left_gripper_contact",
    "forbidden_right_gripper_nonpad_contact",
)
G2_FILTERED_RIGHT_PAD_CONTACT_SENSORS = (
    "forbidden_right_inner_pad_contact",
    "forbidden_right_outer_pad_contact",
)
G2_REQUIRED_FORBIDDEN_CONTACT_SENSORS = (
    *G2_GENERAL_FORBIDDEN_CONTACT_SENSORS,
    *G2_FILTERED_RIGHT_PAD_CONTACT_SENSORS,
)

# Isaac's PhysX contact filter backend requires *each* filter expression to
# resolve to one collision shape per environment.  A rigid-body path is not a
# valid substitute: for example body_link1 expands to several shapes.  This
# tuple is therefore the audited set of effective collision-shape paths from
# the hash-pinned robot_fix.usda.  For bodies without an active shape in the
# source asset it names the authored ``collisions`` shape enabled by the
# selective session-layer spawner below.  Fixed frames without geometry are
# deliberately absent.
G2_COLLIDER_SHAPE_PATHS = (
    "body_link1/collisions/Cylinder/cylinder",
    "body_link1/collisions/Cylinder_001/cylinder",
    "body_link1/collisions/Cube/box",
    "body_link1/collisions/Cube_001/box",
    "body_link1/collisions/Cube_002/box",
    "body_link2/collisions/body_link2_convex_0/mesh",
    "body_link3/collisions/body_link3_convex_0/mesh",
    "body_link4/collisions/body_link4_convex_1/mesh",
    "body_link4/collisions/body_link4_convex_2/mesh",
    "body_link4/collisions/body_link4_convex_3/mesh",
    "body_link5/collisions/body_link5_convex_0/mesh",
    *(f"head_link{i}/collisions/head_link{i}_convex_0/mesh" for i in range(1, 4)),
    *(
        f"arm_{side}_link{i}/collisions/arm_{side}_link{i}_convex_0/mesh"
        for side in ("l", "r")
        for i in range(1, 8)
    ),
    "gripper_l_base_link/collisions/gripper_l_base_link_convex_0/mesh",
    *(
        f"gripper_{arm}_{side}_link{i}/visuals/{side}_link{i}/mesh"
        for arm in ("l", "r")
        for side in ("inner", "outer")
        for i in range(1, 5)
    ),
)


@dataclass(frozen=True)
class G2RuntimeCollisionAttestation:
    schema: str
    original_usd_modified: bool
    session_collision_override_required: bool
    session_self_collision_override_required: bool
    required_sensor_names: tuple[str, ...]
    initialized_sensor_names: tuple[str, ...]
    force_threshold_n: float
    session_self_collision_enabled: bool
    session_collision_shape_count: int
    session_enabled_collision_shape_count: int
    collision_candidate_body_count: int
    collision_covered_body_count: int
    collision_missing_body_names: tuple[str, ...]
    right_arm_collision_links_missing: tuple[str, ...]
    self_collision_filtered_pairs_valid: bool
    self_collision_filtered_pair_count: int
    all_sensor_tensors_finite: bool
    live_full_body_forbidden_collision_authority: bool
    allowed_contact: str
    right_pad_forbidden_contact_method: str
    gpu_pairwise_robot_shape_filter_supported: bool
    excluded_expected_support_bodies: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scene_sensor(env: object, name: str):
    try:
        return env.scene[name]
    except (KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError(
            f"G2_FORBIDDEN_COLLISION_SENSOR_MISSING:{name}"
        ) from exc


def _torch_view(value: object | None) -> torch.Tensor | None:
    """Return a zero-copy torch view for Isaac Lab 2/3 sensor storage."""

    if torch.is_tensor(value):
        return value
    proxy_torch = getattr(value, "torch", None)
    return proxy_torch if torch.is_tensor(proxy_torch) else None


def _force_tensor(sensor: object, *, filtered: bool) -> torch.Tensor:
    data = sensor.data
    if filtered:
        matrix = _torch_view(data.force_matrix_w)
        net = _torch_view(data.net_forces_w)
        if matrix is None or net is None:
            raise RuntimeError("G2_FORBIDDEN_COLLISION_FORCE_TENSOR_UNAVAILABLE")
        if matrix.ndim != 4 or matrix.shape[1] != 1 or matrix.shape[-1] != 3:
            raise RuntimeError("G2_FORBIDDEN_COLLISION_FILTER_SHAPE_INVALID")
        if net.shape != (matrix.shape[0], 1, 3):
            raise RuntimeError("G2_FORBIDDEN_COLLISION_NET_FORCE_SHAPE_INVALID")
        # Filter index zero is the only allowed pair: distal pad vs Object.
        # Every remaining target is forbidden and kept pairwise so opposing
        # forces cannot cancel.  The residual catches unfiltered global bodies
        # (for example a global ground plane) without calling a blocking raw
        # contact-pair query.
        forbidden_pairs = matrix[:, :, 1:, :]
        residual = net - matrix.sum(dim=2)
        value = torch.cat((forbidden_pairs, residual.unsqueeze(2)), dim=2)
    else:
        value = _torch_view(data.net_forces_w)
    if value is None:
        raise RuntimeError("G2_FORBIDDEN_COLLISION_FORCE_TENSOR_UNAVAILABLE")
    return value


def _is_safety_body_name(name: str) -> bool:
    return name.startswith(("body_", "head_", "arm_", "gripper_l_", "gripper_r_"))


def _body_collision_state(body_prim: object) -> tuple[int, int]:
    """Return authored collision candidates and enabled candidates for one body.

    Nested rigid bodies are never counted as descendants of their parent body.
    This prevents a collider on a child link from falsely covering a proxy-less
    fixed frame such as ``arm_r_end_link``.
    """

    from pxr import Usd, UsdPhysics

    candidates = 0
    enabled = 0
    iterator = iter(Usd.PrimRange(body_prim))
    next(iterator, None)
    for prim in iterator:
        if UsdPhysics.RigidBodyAPI(prim):
            iterator.PruneChildren()
            continue
        collision_api = UsdPhysics.CollisionAPI(prim)
        if not collision_api:
            continue
        candidates += 1
        attribute = collision_api.GetCollisionEnabledAttr()
        value = attribute.Get() if attribute.IsValid() else None
        enabled += int(True if value is None else bool(value))
    return candidates, enabled


def _audit_live_stage_collision_schema(
    env: object,
) -> dict[str, int | bool | tuple[str, ...]]:
    """Read the composed USD stage after spawn/session-layer overrides."""

    import omni.usd
    from pxr import Usd, UsdPhysics

    from geniesim.rl.physx.stage2_isaac_scene import (
        G2_DEVELOPMENT_RUNTIME_SELF_COLLISION_FILTER_PAIR_COUNT,
        G2_DEVELOPMENT_RUNTIME_SELF_COLLISION_FILTER_PAIRS,
    )

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("G2_COLLISION_STAGE_UNAVAILABLE")
    total = 0
    enabled_total = 0
    candidate_bodies: set[str] = set()
    covered_bodies: set[str] = set()
    missing_bodies: set[str] = set()
    missing_right_arm: set[str] = set()
    self_collision = True
    filtered_pairs_valid = True
    expected_relationships: dict[str, set[str]] = {}
    for owner, target in G2_DEVELOPMENT_RUNTIME_SELF_COLLISION_FILTER_PAIRS:
        expected_relationships.setdefault(owner, set()).add(target)
    clone_in_fabric = bool(
        getattr(getattr(env.scene, "cfg", None), "clone_in_fabric", False)
    )
    audited_roots = 0
    for env_id in range(env.num_envs):
        root = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot")
        if not root.IsValid():
            # Some Isaac Lab generations keep only the source environment in
            # USD when Fabric cloning is enabled.  Others (including the
            # installed v3 beta) still author all USD clone roots and use
            # Fabric for runtime state.  A missing non-source root is accepted
            # only in the former mode; whenever roots exist they are all
            # audited below.  Live contact tensors still have to cover every
            # environment, so this is not a zero-contact bypass.
            if clone_in_fabric and env_id > 0:
                continue
            raise RuntimeError(f"G2_COLLISION_ROBOT_PRIM_MISSING:{env_id}")
        audited_roots += 1
        attribute = root.GetAttribute("physxArticulation:enabledSelfCollisions")
        self_collision &= bool(attribute.IsValid() and attribute.Get())
        for owner, targets in expected_relationships.items():
            owner_prim = stage.GetPrimAtPath(f"{root.GetPath()}/{owner}")
            relationship = owner_prim.GetRelationship("physics:filteredPairs")
            measured = {
                str(path) for path in relationship.GetTargets()
            } if relationship.IsValid() else set()
            expected = {f"{root.GetPath()}/{target}" for target in targets}
            filtered_pairs_valid &= measured == expected
        for prim in Usd.PrimRange(root):
            if prim == root or not UsdPhysics.RigidBodyAPI(prim):
                continue
            name = prim.GetName()
            if not _is_safety_body_name(name):
                continue
            candidates, enabled = _body_collision_state(prim)
            total += candidates
            enabled_total += enabled
            # A rigid frame with no authored collision geometry is not a
            # collision candidate.  It is recorded by the asset contract, but
            # cannot be made safer by inventing a primitive at runtime.
            if candidates == 0:
                continue
            candidate_bodies.add(name)
            if enabled > 0:
                covered_bodies.add(name)
            else:
                missing_bodies.add(name)
                if name in {f"arm_r_link{i}" for i in range(1, 8)}:
                    missing_right_arm.add(name)
    allowed_root_counts = (
        {1, int(env.num_envs)} if clone_in_fabric else {int(env.num_envs)}
    )
    if audited_roots not in allowed_root_counts:
        raise RuntimeError(
            f"G2_COLLISION_SOURCE_ROOT_COUNT_INVALID:{audited_roots}"
        )
    return {
        "self_collision_enabled": self_collision,
        "collision_shape_count": total,
        "enabled_collision_shape_count": enabled_total,
        "collision_candidate_body_count": len(candidate_bodies),
        "collision_covered_body_count": len(covered_bodies),
        "collision_missing_body_names": tuple(sorted(missing_bodies)),
        "right_arm_collision_links_missing": tuple(sorted(missing_right_arm)),
        "self_collision_filtered_pairs_valid": filtered_pairs_valid,
        "self_collision_filtered_pair_count": len(
            G2_DEVELOPMENT_RUNTIME_SELF_COLLISION_FILTER_PAIRS
        ),
    }


def make_g2_selective_collision_spawn() -> Callable[..., object]:
    """Build an Isaac-Lab clone-aware G2 USD spawner.

    The original USD is never modified.  For each safety-relevant rigid body,
    the session layer is changed only when the body has authored collision
    candidates but none is enabled.  In that case only candidates below a
    ``collisions`` scope are enabled.  Bodies that already have a live visual
    or collision proxy are left byte-for-byte unchanged, avoiding duplicate
    overlapping shapes.

    Imports stay inside the factory so pure-Python contract tests do not need
    an Isaac Sim process.
    """

    from isaaclab.sim import schemas
    from isaaclab.sim.spawners.from_files import from_files
    from isaaclab.sim.utils import clone, get_current_stage
    from pxr import Sdf, UsdPhysics

    from geniesim.rl.physx.stage2_isaac_scene import (
        G2_ROBOT_COLLISION_ALLOWLIST,
        attest_g2_development_runtime_filter_policy,
    )
    from geniesim.rl.physx.stage2_simple_scene import ROBOT_PRIM_PATH

    @clone
    def _spawn(
        prim_path: str,
        cfg: object,
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        **kwargs: object,
    ) -> object:
        # Call the undecorated official implementation.  This outer clone
        # wrapper applies the audited session changes before cloning env_0.
        prim = from_files.spawn_from_usd.__wrapped__(
            prim_path, cfg, translation, orientation, **kwargs
        )
        stage = get_current_stage()
        enabled_paths: list[str] = []
        for entry in G2_ROBOT_COLLISION_ALLOWLIST:
            suffix = entry.prim_path.removeprefix(ROBOT_PRIM_PATH)
            path = f"{prim_path}{suffix}"
            collision_prim = stage.GetPrimAtPath(path)
            if not collision_prim.IsValid() or not UsdPhysics.CollisionAPI(collision_prim):
                raise RuntimeError(f"G2_COLLISION_MANIFEST_PRIM_MISSING:{path}")
            schemas.modify_collision_properties(
                path,
                schemas.CollisionPropertiesCfg(collision_enabled=True),
                stage=stage,
            )
            enabled_paths.append(path)

        filter_policy = attest_g2_development_runtime_filter_policy()
        filtered_pairs = tuple(
            tuple(str(value) for value in pair)
            for pair in filter_policy["filtered_link_pairs"]
        )
        expected_relationships: dict[str, list[str]] = {}
        for owner, target in filtered_pairs:
            expected_relationships.setdefault(owner, []).append(target)
        for owner, targets in expected_relationships.items():
            owner_prim = stage.GetPrimAtPath(f"{prim_path}/{owner}")
            if not owner_prim.IsValid() or not UsdPhysics.RigidBodyAPI(owner_prim):
                raise RuntimeError(f"G2_FILTERED_PAIR_OWNER_MISSING:{owner}")
            relation = UsdPhysics.FilteredPairsAPI.Apply(
                owner_prim
            ).CreateFilteredPairsRel()
            if not relation.SetTargets(
                [Sdf.Path(f"{prim_path}/{target}") for target in sorted(targets)]
            ):
                raise RuntimeError(f"G2_FILTERED_PAIR_WRITE_FAILED:{owner}")
        prim.SetCustomDataByKey(
            "geniesim:collisionAuthoritySchema",
            G2_SELECTIVE_COLLISION_OVERRIDE_SCHEMA,
        )
        prim.SetCustomDataByKey(
            # OpenUSD customData rejects a plain Python list on some Isaac
            # versions.  A deterministic newline-delimited scalar remains
            # portable and preserves the exact session-layer evidence.
            "geniesim:runtimeEnabledCollisionPaths", "\n".join(sorted(enabled_paths))
        )
        prim.SetCustomDataByKey(
            "geniesim:selfCollisionFilteredPairCount", len(filtered_pairs)
        )
        prim.SetCustomDataByKey(
            "geniesim:selfCollisionFilteredPairSha256",
            filter_policy["runtime_authoring"]["pair_sha256"],
        )
        return prim

    return _spawn


class G2ForbiddenCollisionEvaluator:
    """Fail-closed, batched task-collision evaluator.

    This authority deliberately excludes the AMR base/wheel support bodies,
    whose ground contact is expected.  It covers torso, head, both arms, the
    left gripper and every right-gripper body; only right distal-pad/object
    contact is omitted from the filtered pair list.
    """

    def __init__(
        self,
        env: object,
        *,
        force_threshold_n: float = G2_FORBIDDEN_COLLISION_FORCE_EPSILON_N,
        runtime_schema_probe: Mapping[str, object] | None = None,
    ) -> None:
        from geniesim.rl.physx.stage2_isaac_scene import (
            G2_DEVELOPMENT_RUNTIME_SELF_COLLISION_FILTER_PAIR_COUNT,
        )

        if not 0.0 <= float(force_threshold_n) < 1.0:
            raise ValueError("forbidden collision threshold must be in [0,1) N")
        self.env = env
        self.force_threshold_n = float(force_threshold_n)
        self._last_peak_forces_by_sensor: dict[str, torch.Tensor] = {}
        self._last_body_peak_forces_by_sensor: dict[
            str, tuple[tuple[str, ...], torch.Tensor]
        ] = {}
        schema_probe = dict(
            runtime_schema_probe
            if runtime_schema_probe is not None
            else _audit_live_stage_collision_schema(env)
        )
        self_collision_enabled = bool(schema_probe["self_collision_enabled"])
        collision_shape_count = int(schema_probe["collision_shape_count"])
        enabled_collision_shape_count = int(
            schema_probe["enabled_collision_shape_count"]
        )
        collision_candidate_body_count = int(
            schema_probe["collision_candidate_body_count"]
        )
        collision_covered_body_count = int(
            schema_probe["collision_covered_body_count"]
        )
        collision_missing_body_names = tuple(
            str(value) for value in schema_probe["collision_missing_body_names"]
        )
        right_arm_collision_links_missing = tuple(
            str(value)
            for value in schema_probe["right_arm_collision_links_missing"]
        )
        self_collision_filtered_pairs_valid = bool(
            schema_probe["self_collision_filtered_pairs_valid"]
        )
        self_collision_filtered_pair_count = int(
            schema_probe["self_collision_filtered_pair_count"]
        )
        initialized: list[str] = []
        finite = True
        for name in G2_REQUIRED_FORBIDDEN_CONTACT_SENSORS:
            sensor = _scene_sensor(env, name)
            force = _force_tensor(
                sensor, filtered=name in G2_FILTERED_RIGHT_PAD_CONTACT_SENSORS
            )
            if force.shape[0] != env.num_envs:
                raise RuntimeError(
                    f"G2_FORBIDDEN_COLLISION_SENSOR_BATCH_MISMATCH:{name}"
                )
            finite &= bool(torch.isfinite(force).all().item())
            initialized.append(name)
        self.attestation = G2RuntimeCollisionAttestation(
            schema=G2_RUNTIME_COLLISION_AUTHORITY_SCHEMA,
            original_usd_modified=False,
            session_collision_override_required=True,
            session_self_collision_override_required=True,
            required_sensor_names=G2_REQUIRED_FORBIDDEN_CONTACT_SENSORS,
            initialized_sensor_names=tuple(initialized),
            force_threshold_n=self.force_threshold_n,
            session_self_collision_enabled=self_collision_enabled,
            session_collision_shape_count=collision_shape_count,
            session_enabled_collision_shape_count=enabled_collision_shape_count,
            collision_candidate_body_count=collision_candidate_body_count,
            collision_covered_body_count=collision_covered_body_count,
            collision_missing_body_names=collision_missing_body_names,
            right_arm_collision_links_missing=right_arm_collision_links_missing,
            self_collision_filtered_pairs_valid=(
                self_collision_filtered_pairs_valid
            ),
            self_collision_filtered_pair_count=(
                self_collision_filtered_pair_count
            ),
            all_sensor_tensors_finite=finite,
            live_full_body_forbidden_collision_authority=(
                finite
                and self_collision_enabled
                and collision_shape_count > 0
                and enabled_collision_shape_count > 0
                and collision_candidate_body_count == collision_covered_body_count
                and not collision_missing_body_names
                and not right_arm_collision_links_missing
                and self_collision_filtered_pairs_valid
                and self_collision_filtered_pair_count
                == G2_DEVELOPMENT_RUNTIME_SELF_COLLISION_FILTER_PAIR_COUNT
                and tuple(initialized) == G2_REQUIRED_FORBIDDEN_CONTACT_SENSORS
            ),
            allowed_contact=(
                "right_inner_link4/right_outer_link4 against Object only"
            ),
            right_pad_forbidden_contact_method=(
                "NET_FORCE_MINUS_OBJECT_FILTERED_FORCE"
            ),
            gpu_pairwise_robot_shape_filter_supported=False,
            excluded_expected_support_bodies=(
                "base_link",
                "chassis_link",
                "chassis_lwheel_front_link1",
                "chassis_lwheel_front_link2",
                "chassis_lwheel_rear_link1",
                "chassis_lwheel_rear_link2",
                "chassis_rwheel_front_link1",
                "chassis_rwheel_front_link2",
                "chassis_rwheel_rear_link1",
                "chassis_rwheel_rear_link2",
            ),
        )
        if not self.attestation.live_full_body_forbidden_collision_authority:
            raise RuntimeError("G2_FORBIDDEN_COLLISION_AUTHORITY_NOT_LIVE")

    def __call__(self, env: object | None = None) -> torch.Tensor:
        if env is not None and env is not self.env:
            raise ValueError("collision evaluator is bound to another environment")
        collision = torch.zeros(
            self.env.num_envs,
            dtype=torch.bool,
            device=self.env.device,
        )
        for name in G2_REQUIRED_FORBIDDEN_CONTACT_SENSORS:
            force = _force_tensor(
                _scene_sensor(self.env, name),
                filtered=name in G2_FILTERED_RIGHT_PAD_CONTACT_SENSORS,
            )
            flattened = force.reshape(self.env.num_envs, -1)
            finite_per_env = torch.isfinite(flattened).all(dim=1)
            sanitized = torch.where(torch.isfinite(force), force, torch.zeros_like(force))
            per_env_peak = torch.linalg.vector_norm(sanitized, dim=-1).reshape(
                self.env.num_envs, -1
            ).amax(dim=1)
            self._last_peak_forces_by_sensor[name] = per_env_peak.detach().clone()
            if name in G2_GENERAL_FORBIDDEN_CONTACT_SENSORS and force.ndim == 3:
                sensor = _scene_sensor(self.env, name)
                body_names = tuple(
                    str(value) for value in getattr(sensor, "body_names", ())
                )
                if len(body_names) != force.shape[1]:
                    body_names = tuple(
                        f"body_index_{index}" for index in range(force.shape[1])
                    )
                self._last_body_peak_forces_by_sensor[name] = (
                    body_names,
                    torch.linalg.vector_norm(sanitized, dim=-1).detach().clone(),
                )
            # Non-finite contact telemetry is a safety failure, never a safe
            # zero; it terminates the affected environment through the same
            # fail-closed signal.
            collision |= (~finite_per_env) | (per_env_peak > self.force_threshold_n)
        return collision

    def sensor_peak_forces_n(self) -> dict[str, list[float]]:
        """Materialize per-env sensor peaks only for failure diagnostics."""

        peaks: dict[str, list[float]] = {}
        for name in G2_REQUIRED_FORBIDDEN_CONTACT_SENSORS:
            value = self._last_peak_forces_by_sensor.get(name)
            if value is None:
                force = _force_tensor(
                    _scene_sensor(self.env, name),
                    filtered=name in G2_FILTERED_RIGHT_PAD_CONTACT_SENSORS,
                )
                sanitized = torch.where(
                    torch.isfinite(force), force, torch.zeros_like(force)
                )
                value = (
                    torch.linalg.vector_norm(sanitized, dim=-1)
                    .reshape(self.env.num_envs, -1)
                    .amax(dim=1)
                )
            peaks[name] = [float(item) for item in value.detach().cpu().tolist()]
        return peaks

    def sensor_body_peak_forces_n(self) -> dict[str, dict[str, list[float]]]:
        """Return per-body unfiltered peaks for one fail-closed snapshot."""

        result: dict[str, dict[str, list[float]]] = {}
        for name in G2_GENERAL_FORBIDDEN_CONTACT_SENSORS:
            cached = self._last_body_peak_forces_by_sensor.get(name)
            if cached is None:
                sensor = _scene_sensor(self.env, name)
                force = _force_tensor(sensor, filtered=False)
                if force.ndim != 3 or force.shape[-1] != 3:
                    raise RuntimeError("G2_FORBIDDEN_COLLISION_BODY_FORCE_SHAPE_INVALID")
                body_names = tuple(
                    str(value) for value in getattr(sensor, "body_names", ())
                )
                if len(body_names) != force.shape[1]:
                    body_names = tuple(
                        f"body_index_{index}" for index in range(force.shape[1])
                    )
                magnitudes = torch.linalg.vector_norm(force, dim=-1)
            else:
                body_names, magnitudes = cached
            result[name] = {
                body_name: [
                    float(item)
                    for item in magnitudes[:, index].detach().cpu().tolist()
                ]
                for index, body_name in enumerate(body_names)
            }
        return result


def install_g2_forbidden_collision_sensors(scene: object) -> None:
    """Install the exact sensor partition on an Isaac Lab scene config."""

    from isaaclab.sensors import ContactSensorCfg

    common = {"update_period": 0.0, "history_length": 1}
    scene.forbidden_torso_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body_link.*", **common
    )
    scene.forbidden_head_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_.*", **common
    )
    scene.forbidden_arm_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm_.*", **common
    )
    scene.forbidden_left_gripper_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_l_.*", **common
    )
    scene.forbidden_right_gripper_nonpad_contact = ContactSensorCfg(
        prim_path=(
            "{ENV_REGEX_NS}/Robot/"
            "gripper_r_(base_link|center_link|inner_link[123]|outer_link[123])"
        ),
        **common,
    )
    # Isaac Sim 6 GPU contact views reject robot collision-shape paths as
    # pairwise filter targets.  The supported Object filter remains exact.
    # Subtracting that allowed vector from the pad's unfiltered net force
    # yields the forbidden-contact resultant without a raw blocking query.
    pad_filters = ["{ENV_REGEX_NS}/Object"]
    scene.forbidden_right_inner_pad_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_r_inner_link4",
        filter_prim_paths_expr=pad_filters,
        **common,
    )
    scene.forbidden_right_outer_pad_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_r_outer_link4",
        filter_prim_paths_expr=pad_filters,
        **common,
    )


__all__ = [
    "G2_FORBIDDEN_COLLISION_FORCE_EPSILON_N",
    "G2_GENERAL_FORBIDDEN_CONTACT_SENSORS",
    "G2_FILTERED_RIGHT_PAD_CONTACT_SENSORS",
    "G2_COLLIDER_SHAPE_PATHS",
    "G2_REQUIRED_FORBIDDEN_CONTACT_SENSORS",
    "G2_RUNTIME_COLLISION_AUTHORITY_SCHEMA",
    "G2_SELECTIVE_COLLISION_OVERRIDE_SCHEMA",
    "G2ForbiddenCollisionEvaluator",
    "G2RuntimeCollisionAttestation",
    "install_g2_forbidden_collision_sensors",
    "make_g2_selective_collision_spawn",
]
