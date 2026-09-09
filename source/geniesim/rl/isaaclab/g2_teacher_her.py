"""Goal-safe replay adapters for Milestone 8 G2 teacher SAC.

Only achieved object positions may become hindsight goals.  Robot state,
physical object state, contacts, actions and termination causes remain intact.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import torch

from .g2_lift_methodology import G2LiftSandboxContract, G2LiftState
from .g2_teacher_sac import G2TeacherObservationContract


@dataclass(frozen=True)
class G2TeacherTransition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    sequence_step: int | None = None
    non_goal_terminated: bool | None = None


def replay_phase_label(
    next_observation: np.ndarray, *, stable_grasp: bool
) -> str:
    """Map immutable physical phase evidence to a replay stratum."""

    contract = G2TeacherObservationContract()
    value = np.asarray(next_observation, dtype=np.float32)
    contract.validate_flat_observation(torch.from_numpy(value))
    phase = value[contract.slices["curriculum_phase_features"]]
    if bool(phase[2] > 0.5 or phase[3] > 0.5):
        return "LIFT"
    if stable_grasp:
        return "STABLE_GRASP"
    if bool(phase[1] > 0.5):
        return "CONTACT"
    return "REACH"


def relabel_episode_with_future_goals(
    transitions: Sequence[G2TeacherTransition],
    *,
    future_goals_per_transition: int,
    rng: np.random.Generator,
    sandbox: G2LiftSandboxContract | None = None,
) -> list[G2TeacherTransition]:
    """Generate same-episode HER rows from real non-terminal future states.

    Auto-reset terminal next observations are never accepted as future states.
    Sampling is with replacement when fewer than ``k`` future rows exist,
    which keeps an exact per-source ratio without fabricating physical state.
    """

    if future_goals_per_transition <= 0:
        raise ValueError("future_goals_per_transition must be positive")
    episode = list(transitions)
    if any(item.terminated or item.truncated for item in episode):
        raise ValueError("live HER episode input must exclude auto-reset rows")
    result: list[G2TeacherTransition] = []
    for source_index, transition in enumerate(episode):
        candidate_indices = np.arange(source_index, len(episode), dtype=np.int64)
        if candidate_indices.size == 0:
            continue
        selected = rng.choice(
            candidate_indices,
            size=future_goals_per_transition,
            replace=candidate_indices.size < future_goals_per_transition,
        )
        for future_index in np.asarray(selected).reshape(-1):
            future_goal = achieved_goal(episode[int(future_index)].next_observation)
            result.append(
                relabel_with_actual_future_goal(
                    transition,
                    actual_future_achieved_goal_root_m=future_goal,
                    sandbox=sandbox,
                )
            )
    return result


def _cube_goal_progress_transition_reward(
    observation: np.ndarray,
    next_observation: np.ndarray,
    goal: np.ndarray,
    *,
    task: G2LiftSandboxContract,
    sequence_step: int | None,
) -> float:
    """Recompute the active bounded goal-progress shaping term.

    The live reward term is a weighted rate and RewardManager integrates it by
    ``policy_dt_s``.  Its history is uninitialized on sequence step zero, so
    that transition contributes exactly zero.
    """

    if sequence_step is None:
        raise ValueError("HER requires sequence_step to relabel goal-progress PBRS")
    if sequence_step < 0:
        raise ValueError("sequence_step cannot be negative")
    if sequence_step == 0:
        return 0.0
    current_cube = achieved_goal(observation)
    next_cube = achieved_goal(next_observation)
    lifted = next_cube[2] >= (
        task.table_surface_height_m + task.lift_height_above_table_m
    )
    if not lifted:
        return 0.0
    current_distance = float(np.linalg.norm(goal - current_cube))
    next_distance = float(np.linalg.norm(goal - next_cube))
    bounded_rate = float(
        np.clip((current_distance - 0.99 * next_distance) / 0.10, -1.0, 1.0)
    )
    return task.policy_dt_s * 4.0 * bounded_rate


def achieved_goal(observation: np.ndarray) -> np.ndarray:
    contract = G2TeacherObservationContract()
    value = np.asarray(observation, dtype=np.float32)
    if value.shape != (contract.observation_dim,) or not np.all(np.isfinite(value)):
        raise ValueError("teacher observation is invalid")
    contract.validate_flat_observation(torch.from_numpy(value))
    return value[contract.slices["cube_pose_root_xyzw"]][:3].copy()


def relabel_observation_goal(
    observation: np.ndarray, desired_goal_root_m: np.ndarray
) -> np.ndarray:
    """Change only goal-dependent teacher fields."""

    contract = G2TeacherObservationContract()
    source = np.asarray(observation, dtype=np.float32)
    goal = np.asarray(desired_goal_root_m, dtype=np.float32)
    if source.shape != (contract.observation_dim,) or not np.all(np.isfinite(source)):
        raise ValueError("teacher observation is invalid")
    if goal.shape != (3,) or not np.all(np.isfinite(goal)):
        raise ValueError("HER goal must contain three finite values")
    result = source.copy()
    cube = achieved_goal(source)
    result[contract.slices["goal_position_root_m"]] = goal
    result[contract.slices["cube_to_goal_m"]] = goal - cube
    return result


def relabel_with_actual_future_goal(
    transition: G2TeacherTransition,
    *,
    actual_future_achieved_goal_root_m: np.ndarray,
    sandbox: G2LiftSandboxContract | None = None,
) -> G2TeacherTransition:
    """Return one HER row with reward/success recomputed from a real future state."""

    task = (sandbox or G2LiftSandboxContract()).validated()
    if transition.terminated and transition.truncated:
        raise ValueError("transition cannot be both terminated and truncated")
    goal = np.asarray(actual_future_achieved_goal_root_m, dtype=np.float32)
    current = relabel_observation_goal(transition.observation, goal)
    following = relabel_observation_goal(transition.next_observation, goal)
    ee = following[G2TeacherObservationContract().slices["end_effector_pose_root_xyzw"]][:3]
    cube = achieved_goal(following)
    observation_contract = G2TeacherObservationContract()
    old_goal = transition.next_observation[
        observation_contract.slices["goal_position_root_m"]
    ]
    old_terms = task.reward_terms(G2LiftState(tuple(ee), tuple(cube), tuple(old_goal)))
    new_terms = task.reward_terms(G2LiftState(tuple(ee), tuple(cube), tuple(goal)))
    # HER changes only the desired goal.  Preserve every goal-independent term
    # already emitted by the live environment (contact, recovery, push,
    # orientation, action and safety shaping) and replace only the two
    # goal-dependent tracking components.  Rebuilding the whole reward from
    # the compact observation would silently discard those live-only terms.
    goal_term_names = ("object_goal_tracking", "object_goal_tracking_fine")
    old_pbrs = _cube_goal_progress_transition_reward(
        transition.observation,
        transition.next_observation,
        old_goal,
        task=task,
        sequence_step=transition.sequence_step,
    )
    new_pbrs = _cube_goal_progress_transition_reward(
        current,
        following,
        goal,
        task=task,
        sequence_step=transition.sequence_step,
    )
    reward = float(
        transition.reward
        - old_pbrs
        + new_pbrs
        + task.policy_dt_s
        * (
            -sum(old_terms[name] for name in goal_term_names)
            + sum(new_terms[name] for name in goal_term_names)
        )
    )
    if not math.isfinite(reward):
        raise ValueError("relabelled reward is non-finite")
    # A relabelled position must not synthesize a successful grasp/lift from a
    # tabletop REACH transition.  The physical phase fields are immutable HER
    # state, so the live lifted predicate remains a required success premise.
    phase = following[observation_contract.slices["curriculum_phase_features"]]
    # Phase[3] is the immutable live physical predicate
    # (stable grasp AND lift).  Phase[2] alone is merely height and must never
    # let HER promote a slip/throw into success.
    physically_lifted = bool(phase[3] > 0.5)
    old_success = bool(
        physically_lifted
        and np.linalg.norm(cube - old_goal) <= task.goal_tolerance_m
    )
    success = bool(
        physically_lifted and np.linalg.norm(cube - goal) <= task.goal_tolerance_m
    )
    non_goal_termination = (
        bool(transition.non_goal_terminated)
        if transition.non_goal_terminated is not None
        else bool(transition.terminated and not old_success)
    )
    return G2TeacherTransition(
        observation=current,
        action=np.asarray(transition.action, dtype=np.float32).copy(),
        reward=reward,
        next_observation=following,
        terminated=bool(non_goal_termination or success),
        truncated=bool(transition.truncated and not success),
        sequence_step=transition.sequence_step,
        non_goal_terminated=non_goal_termination,
    )


def relabel_recurrent_visual_batch_with_future_goals(
    batch: Mapping[str, np.ndarray],
    *,
    sequence_fraction: float,
    rng: np.random.Generator,
    sandbox: G2LiftSandboxContract | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Apply episode-safe, sequence-consistent HER to an online RGB-D batch.

    One actual achieved cube position from the end of a sampled sequence is
    used as the desired goal for every non-terminal row in that sequence.
    Images, actions, contact/force state and all other privileged state remain
    unchanged.  Terminal auto-reset next observations are never relabelled.
    """

    if not 0.0 <= sequence_fraction <= 1.0:
        raise ValueError("HER sequence fraction must be in [0,1]")
    required = (
        "privileged", "next_privileged", "proprio", "next_proprio",
        "actions", "rewards", "terminated", "truncated", "sequence_step",
    )
    if any(name not in batch for name in required):
        raise ValueError("recurrent visual HER batch is missing required fields")
    result = {
        name: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        for name, value in batch.items()
    }
    privileged = result["privileged"]
    if privileged.ndim != 3:
        raise ValueError("recurrent visual HER privileged state must be [B,T,D]")
    batch_size, steps = privileged.shape[:2]
    selected = rng.random(batch_size) < float(sequence_fraction)
    teacher = G2TeacherObservationContract()
    # Local import avoids making the HER module an import-time dependency of
    # the visual policy implementation.
    from .g2_visual_sac import G2StudentObservationContract

    student = G2StudentObservationContract()
    goal_slice = student.slices["goal_position_root_m"]
    applied = rejected = rows = 0
    for sequence_index in np.flatnonzero(selected):
        # The current state at the final replay row is pre-auto-reset and is
        # therefore valid achieved-goal evidence even when its next state is a
        # reset observation.
        future_goal = achieved_goal(privileged[sequence_index, -1])
        sequence_applied = False
        for step in range(steps):
            if bool(result["terminated"][sequence_index, step, 0]) or bool(
                result["truncated"][sequence_index, step, 0]
            ):
                continue
            transition = G2TeacherTransition(
                observation=privileged[sequence_index, step],
                action=result["actions"][sequence_index, step],
                reward=float(result["rewards"][sequence_index, step, 0]),
                next_observation=result["next_privileged"][sequence_index, step],
                terminated=False,
                truncated=False,
                sequence_step=int(result["sequence_step"][sequence_index, step]),
                non_goal_terminated=False,
            )
            try:
                relabelled = relabel_with_actual_future_goal(
                    transition,
                    actual_future_achieved_goal_root_m=future_goal,
                    sandbox=sandbox,
                )
            except ValueError:
                rejected += 1
                continue
            result["privileged"][sequence_index, step] = relabelled.observation
            result["next_privileged"][sequence_index, step] = relabelled.next_observation
            result["proprio"][sequence_index, step, goal_slice] = future_goal
            result["next_proprio"][sequence_index, step, goal_slice] = future_goal
            result["rewards"][sequence_index, step, 0] = relabelled.reward
            result["terminated"][sequence_index, step, 0] = float(relabelled.terminated)
            result["truncated"][sequence_index, step, 0] = float(relabelled.truncated)
            if "hindsight_relabel" in result:
                result["hindsight_relabel"][sequence_index, step] = True
            rows += 1
            sequence_applied = True
        applied += int(sequence_applied)
    return result, {
        "her/selected_sequences": float(selected.sum()),
        "her/applied_sequences": float(applied),
        "her/relabelled_rows": float(rows),
        "her/rejected_rows": float(rejected),
        "her/sequence_fraction": float(sequence_fraction),
        "her/physical_contact_state_preserved": 1.0,
    }


__all__ = [
    "G2TeacherTransition",
    "achieved_goal",
    "relabel_observation_goal",
    "relabel_episode_with_future_goals",
    "relabel_recurrent_visual_batch_with_future_goals",
    "relabel_with_actual_future_goal",
    "replay_phase_label",
]
