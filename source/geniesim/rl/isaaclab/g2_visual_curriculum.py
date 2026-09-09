"""Reset event for the G2 visual reverse curriculum."""

from __future__ import annotations

import numpy as np
import torch

from .g2_lift_methodology import RIGHT_ARM_JOINTS
from .g2_visual_sac import G2ReverseCurriculum


# Collision-clear pre-grasp seed originally generated for the earlier object.
# Runtime reverse preroll recomputes its Cartesian endpoint from the live
# object center, so this remains only a collision-clear articulation seed.
# Position residual is 1.35 mm and the static collision margin is 66.4 mm.
# It is only a reset seed; it is never an imitation action or actor target.
EVENT_PREGRASP_RIGHT_ARM_Q = (
    -0.8752342027309461, -1.272325512843969, 0.38677689557687983,
    -1.1906268460861673, -1.0159472515610708, -0.7185662147123338,
    1.243663979353904,
)

# Reverse articulation seed. The live 40 x 40 x 60 mm workpiece endpoint is
# recomputed from the current rigid-body center rather than inferred here.
EVENT_GRASP_READY_RIGHT_ARM_Q = (
    -0.9108010300665879, -1.438278214780484, 0.36585662315835343,
    -0.7981900540332117, -1.3152808467532087, -0.7488714961638561,
    0.8930670507291367,
)


def reset_right_arm_reverse_curriculum(
    env,
    env_ids: torch.Tensor,
    *,
    initial_probability: float = 0.60,
    minimum_probability: float = 0.20,
    progress_min: float = 0.95,
    progress_max: float = 1.00,
    policy_near_probability: float = 0.25,
    policy_progress_min: float = 0.40,
    policy_progress_max: float = 0.55,
    joint_perturbation_rad: float = 0.005,
    pregrasp_right_arm_q: tuple[float, ...] | None = None,
    grasp_ready_right_arm_q: tuple[float, ...] | None = None,
) -> None:
    """Mix original-home and validated near-reference arm reset states."""

    if not 0.0 <= policy_near_probability <= 1.0:
        raise ValueError("policy-near probability must be in [0,1]")
    if not 0.0 <= policy_progress_min <= policy_progress_max <= 1.0:
        raise ValueError("invalid policy-near progress range")
    robot = env.scene["robot"]
    if not hasattr(env, "_g2_reverse_curriculum"):
        env._g2_reverse_curriculum = G2ReverseCurriculum(
            initial_probability=initial_probability,
            minimum_probability=minimum_probability,
        )
        env._g2_reverse_rng = np.random.default_rng(1_000_003)
        env._g2_reverse_reset_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._g2_reverse_last_near = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._g2_reverse_last_progress = torch.zeros(env.num_envs, device=env.device)
        env._g2_reverse_last_policy_near = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    ids_np = env_ids.detach().cpu().numpy()
    near_np = env._g2_reverse_curriculum.sample_near(env._g2_reverse_rng, len(ids_np))
    rho_np = env._g2_reverse_rng.uniform(progress_min, progress_max, len(ids_np))
    # Reserve a subset of reset episodes for policy-only near-contact
    # experience.  The flag is consumed at the same reset boundary by the
    # behavior scheduler, so a privileged reference cannot rescue the rollout.
    policy_near_np = env._g2_reverse_rng.random(len(ids_np)) < policy_near_probability
    near_np |= policy_near_np
    if bool(policy_near_np.any()):
        rho_np[policy_near_np] = env._g2_reverse_rng.uniform(
            policy_progress_min, policy_progress_max, int(policy_near_np.sum())
        )
    arm_ids = [robot.joint_names.index(name) for name in RIGHT_ARM_JOINTS]
    q = robot.data.default_joint_pos[env_ids].clone()
    qd = torch.zeros_like(q)
    pregrasp_values = (
        EVENT_PREGRASP_RIGHT_ARM_Q
        if pregrasp_right_arm_q is None
        else tuple(float(value) for value in pregrasp_right_arm_q)
    )
    if len(pregrasp_values) != len(RIGHT_ARM_JOINTS):
        raise ValueError("reverse pregrasp seed must contain seven joints")
    pregrasp = torch.tensor(pregrasp_values, dtype=q.dtype, device=q.device)
    rho = torch.as_tensor(rho_np, dtype=q.dtype, device=q.device).unsqueeze(-1)
    perturb = torch.as_tensor(
        env._g2_reverse_rng.uniform(-joint_perturbation_rad, joint_perturbation_rad, (len(ids_np), 7)),
        dtype=q.dtype, device=q.device,
    )
    # A joint-space interpolation between two measured demonstration poses is
    # not a collision-attested Cartesian path.  Use one of the exact recorded
    # endpoints as the reset seed and let the bounded runtime preroll align it
    # to the sampled cube.  This prevents a synthesized intermediate posture
    # from pushing the cube before the first policy transition.
    if grasp_ready_right_arm_q is None:
        proposed = pregrasp.unsqueeze(0).expand(len(ids_np), -1)
    else:
        grasp_ready_values = tuple(float(value) for value in grasp_ready_right_arm_q)
        if len(grasp_ready_values) != len(RIGHT_ARM_JOINTS):
            raise ValueError("reverse grasp-ready seed must contain seven joints")
        grasp_ready = torch.tensor(
            grasp_ready_values, dtype=q.dtype, device=q.device
        )
        proposed = grasp_ready.unsqueeze(0).expand(len(ids_np), -1)
    proposed = proposed + perturb
    near = torch.as_tensor(near_np, dtype=torch.bool, device=q.device)
    # The ordinary branch is the environment's configured reset authority.
    # Hard-coding the legacy exhibition arm here silently replaced the
    # keyboard pose even when the Teacher config was dataset-aligned.
    ordinary = robot.data.default_joint_pos[env_ids][:, arm_ids]
    q[:, arm_ids] = torch.where(near.unsqueeze(-1), proposed, ordinary)
    limits = robot.data.soft_joint_pos_limits[env_ids][:, arm_ids]
    if bool(((q[:, arm_ids] <= limits[..., 0]) | (q[:, arm_ids] >= limits[..., 1])).any()):
        raise RuntimeError("G2_REVERSE_CURRICULUM_JOINT_LIMIT_REJECTION")
    robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)
    robot.set_joint_position_target(q, env_ids=env_ids)
    env._g2_reverse_last_near[env_ids] = near
    env._g2_reverse_last_progress[env_ids] = torch.where(near, rho[:, 0], torch.zeros_like(rho[:, 0]))
    env._g2_reverse_last_policy_near[env_ids] = torch.as_tensor(
        policy_near_np, dtype=torch.bool, device=q.device
    )
    env._g2_reverse_reset_count[env_ids] += 1


__all__ = [
    "EVENT_GRASP_READY_RIGHT_ARM_Q",
    "EVENT_PREGRASP_RIGHT_ARM_Q",
    "reset_right_arm_reverse_curriculum",
]
