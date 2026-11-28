"""Color-pair split and instructions for the three-ball disjunction study."""

from dataclasses import dataclass
from itertools import combinations

from .pickup_tasks import COLORS, CONFIG as BASE_CONFIG

HELD_OUT_PAIRS = (("red", "green"), ("blue", "purple"), ("yellow", "grey"))
PAIRS = tuple(combinations(COLORS, 2))
CONDITIONS = ("first-only", "second-only", "both")
CONFIG = {**BASE_CONFIG, "scene_setup": "three-balls", "objects": 3,
          "train_negated_colors": [], "held_out_negated_colors": [],
          "held_out_pairs": [list(pair) for pair in HELD_OUT_PAIRS],
          "positive_probability": 0.5, "disjunction_conditions": list(CONDITIONS)}


@dataclass(frozen=True)
class DisjunctionTask:
    colors: tuple[str, ...]
    object_type: str = "ball"

    @property
    def kind(self):
        return "positive" if len(self.colors) == 1 else "disjunction"

    @property
    def color(self):
        return self.colors[0]

    @property
    def id(self):
        return f"{self.kind}:{':'.join(self.colors)}:ball"

    @property
    def mission(self):
        return "pick up " + " or ".join(f"a {c} ball" for c in self.colors)

    @property
    def pair(self):
        return tuple(sorted(self.colors, key=COLORS.index))

    @property
    def category(self):
        if self.kind == "positive":
            return "positive"
        return "held_out_disjunction" if self.pair in HELD_OUT_PAIRS else "familiar_disjunction"

    def matches(self, obj):
        return obj is not None and obj.type == "ball" and obj.color in self.colors


POSITIVE_TASKS = tuple(DisjunctionTask((c,)) for c in COLORS)
PAIR_TASKS = tuple(DisjunctionTask(pair) for pair in PAIRS)
TASKS = POSITIVE_TASKS + tuple(t for pair in PAIRS
                              for t in (DisjunctionTask(pair), DisjunctionTask(pair[::-1])))
TASK_BY_ID = {t.id: t for t in TASKS}
TRAIN_OR_TASKS = tuple(t for t in TASKS if t.category == "familiar_disjunction")
TRAIN_TASKS = POSITIVE_TASKS + TRAIN_OR_TASKS
TEST_TASKS = tuple(t for t in PAIR_TASKS if t.category == "held_out_disjunction")
SEEN_OR_TASKS = tuple(t for t in PAIR_TASKS if t.category == "familiar_disjunction")
# Evaluation uses one canonical pair; odd scene seeds reverse the wording.
EVALUATION_TASKS = TEST_TASKS + SEEN_OR_TASKS + POSITIVE_TASKS
INSTRUCTION_SETS = {"unseen": TEST_TASKS, "seen-disjunction": SEEN_OR_TASKS,
                    "seen-positive": POSITIVE_TASKS, "seen": SEEN_OR_TASKS + POSITIVE_TASKS}
