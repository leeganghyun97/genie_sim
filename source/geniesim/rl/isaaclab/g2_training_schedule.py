"""Sealed timing/update helpers for vectorized G2 off-policy training."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class G2OffPolicyUpdateSchedule:
    """Keep replay samples per collected transition stable across ``num_envs``.

    Counting one optimizer update per vector step makes the effective update
    intensity shrink by ``1 / num_envs``.  This schedule instead budgets
    gradient batches from the number of newly collected transitions.  The
    configured ratio counts replay samples, not optimizer calls.
    """

    learning_starts: int
    batch_size: int
    replay_samples_per_transition: float = 1.0
    maximum_updates_per_vector_step: int = 32

    def __post_init__(self) -> None:
        if self.learning_starts < 0:
            raise ValueError("learning_starts cannot be negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not math.isfinite(self.replay_samples_per_transition) or not (
            self.replay_samples_per_transition > 0.0
        ):
            raise ValueError("replay_samples_per_transition must be finite and positive")
        if self.maximum_updates_per_vector_step <= 0:
            raise ValueError("maximum_updates_per_vector_step must be positive")

    def target_update_count(self, transitions_collected: int) -> int:
        eligible = max(0, int(transitions_collected) - self.learning_starts)
        return int(
            math.floor(
                eligible * self.replay_samples_per_transition / self.batch_size
            )
        )

    def updates_due(self, transitions_collected: int, updates_completed: int) -> int:
        if updates_completed < 0:
            raise ValueError("updates_completed cannot be negative")
        outstanding = max(
            0, self.target_update_count(transitions_collected) - int(updates_completed)
        )
        return min(outstanding, self.maximum_updates_per_vector_step)

    def realized_replay_samples_per_transition(
        self, transitions_collected: int, updates_completed: int
    ) -> float:
        eligible = max(0, int(transitions_collected) - self.learning_starts)
        if eligible == 0:
            return 0.0
        return float(updates_completed * self.batch_size / eligible)


def descending_environment_candidates(
    start: int = 2000, minimum: int = 200, decrement: int = 200
) -> tuple[int, ...]:
    """Return the requested fail-closed capacity sequence, including minimum."""

    if start <= 0 or minimum <= 0 or decrement <= 0:
        raise ValueError("environment capacity values must be positive")
    if start < minimum:
        raise ValueError("start must be greater than or equal to minimum")
    values = list(range(start, minimum - 1, -decrement))
    if not values or values[-1] != minimum:
        values.append(minimum)
    return tuple(values)


__all__ = ["G2OffPolicyUpdateSchedule", "descending_environment_candidates"]
