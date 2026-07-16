"""Dynamic task mixing schedule for Metis training.

Implements linear annealing of task sampling weights across epochs.

Default schedule:
  - Task 0 (fact_recall):    25% -> 10%
  - Task 1 (memory_op):      35% -> 25%
  - Task 2 (long_term):      20% -> 30%
  - Task 3 (v2 task3):       10% -> 20%
  - Task 4 (v2 task4):       10% -> 15%

Weights are normalised to sum to 1.0 at each epoch. If some tasks are not
present in a dataset, the sampler normalises over the available tasks.
"""

from __future__ import annotations

class TaskScheduler:
    """Manages per-epoch task sampling weights with linear annealing."""

    def __init__(
        self,
        task0_weight_start: float = 0.25,
        task0_weight_end: float = 0.1,
        task1_weight_start: float = 0.35,
        task1_weight_end: float = 0.25,
        task2_weight_start: float = 0.2,
        task2_weight_end: float = 0.3,
        task3_weight_start: float = 0.1,
        task3_weight_end: float = 0.2,
        task4_weight_start: float = 0.1,
        task4_weight_end: float = 0.15,
    ):
        self._starts = {
            0: max(task0_weight_start, 0.0),
            1: max(task1_weight_start, 0.0),
            2: max(task2_weight_start, 0.0),
            3: max(task3_weight_start, 0.0),
            4: max(task4_weight_start, 0.0),
        }
        self._ends = {
            0: max(task0_weight_end, 0.0),
            1: max(task1_weight_end, 0.0),
            2: max(task2_weight_end, 0.0),
            3: max(task3_weight_end, 0.0),
            4: max(task4_weight_end, 0.0),
        }

    def get_weights(self, epoch: int, total_epochs: int) -> dict[int, float]:
        """Return normalised {task_id: weight} for the given epoch."""
        if total_epochs <= 1:
            progress = 0.0
        else:
            progress = min(epoch / max(total_epochs - 1, 1), 1.0)

        weights = {
            task_id: start + (self._ends[task_id] - start) * progress
            for task_id, start in self._starts.items()
        }
        weights = {task_id: max(weight, 0.0) for task_id, weight in weights.items()}

        # Normalise
        total = sum(weights.values())
        if total == 0:
            return {task_id: 1.0 / len(weights) for task_id in weights}

        return {task_id: weight / total for task_id, weight in weights.items()}

    @property
    def schedule_description(self) -> str:
        parts = (
            f"Task{task_id} {self._starts[task_id]:.1%}→{self._ends[task_id]:.1%}"
            for task_id in sorted(self._starts)
        )
        return "TaskScheduler: " + ", ".join(parts)
