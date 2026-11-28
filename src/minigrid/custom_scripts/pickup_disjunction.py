"""Three-ball tasks sharing the negation environment's rewards and pickup verifier."""

from .disjunction_tasks import (
    CONDITIONS, HELD_OUT_PAIRS, POSITIVE_TASKS, TASK_BY_ID, TRAIN_OR_TASKS, TRAIN_TASKS,
)
from .pickup_tasks import COLORS
from .pickup_negation import PickupNegation, TaskDesc, TaskPickupInstr


class PickupDisjunction(PickupNegation):
    task_family = "disjunction"
    task_by_id = TASK_BY_ID

    def __init__(self, scene_setup="three-balls", **kwargs):
        if scene_setup != "three-balls":
            raise ValueError("Disjunction uses scene_setup='three-balls'")
        super().__init__(**kwargs)
        self.scene_setup = scene_setup
        self.object_count = 3
        self.training_tasks = TRAIN_TASKS

    def reset(self, *, seed=None, options=None):
        options = options or {}
        self._condition = options.get("condition")
        self._paired = options.get("paired", False)
        if self._condition is not None and self._condition not in CONDITIONS:
            raise ValueError(f"Unknown disjunction condition: {self._condition}")
        if self._paired:
            task = self.task_by_id[options["task"]]
            if seed is None or task.category != "held_out_disjunction":
                raise ValueError("Paired scenes require a seed and a held-out disjunction")
            # Complementary variants balance each target color over seeds 0..19.
            self._paired_variant = (seed // 2 % 4) ^ (7 if seed % 2 else 0)
        return super().reset(seed=seed, options=options)

    def _sample_task(self):
        # Sample categories explicitly: 50% positive, 50% disjunction.
        return self._rand_elem(POSITIVE_TASKS if self._rand_int(0, 2) == 0 else TRAIN_OR_TASKS)

    def _gen_grid(self, width, height):
        # Keep the condition fixed through the parent's scene-rejection loop.
        self._episode_condition = self._condition or (None if self._paired else self._rand_elem(CONDITIONS))
        super()._gen_grid(width, height)

    def gen_mission(self):
        task = self.task
        if self._paired:
            # No task-dependent RNG calls: all three instructions recreate this scene.
            colors = [pair[(self._paired_variant >> i) & 1]
                      for i, pair in enumerate(HELD_OUT_PAIRS)]
            self.scene_condition = "first-only" if task.colors[0] in colors else "second-only"
        else:
            self.scene_condition = ("positive" if task.kind == "positive" else
                                    self._episode_condition)
            if self.scene_condition in ("positive", "first-only"):
                colors = [task.colors[0]]
            elif self.scene_condition == "second-only":
                colors = [task.colors[1]]
            else:
                colors = list(task.colors)
            while len(colors) < 3:
                colors.append(self._rand_elem([c for c in COLORS
                                               if c not in task.colors and c not in colors]))
            # Placement order must not identify targets or the wording order.
            colors = self._rand_subset(colors, len(colors))
        for color in colors:
            self.add_object(0, 0, "ball", color)
        self.place_agent(0, 0)
        self.check_objs_reachable()
        self.instrs = TaskPickupInstr(TaskDesc(task), strict=False)
