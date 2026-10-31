from dataclasses import dataclass

#object types and colors
TYPES = ("ball", "key", "box")
TRAIN_COLORS = ("red", "green", "blue")
HELD_OUT_COLORS = ("purple", "yellow", "grey")
COLORS = TRAIN_COLORS + HELD_OUT_COLORS

#training environment config
CONFIG = {"room_size": 7, "objects": 5, "max_steps": 500,
          "actions": ["left", "right", "forward", "pickup", "drop"],
          "train_negated_colors": list(TRAIN_COLORS),
          "held_out_negated_colors": list(HELD_OUT_COLORS),
          "reward": "fixed_correct_pickup_ends; penalized_wrong_pickup_continues; optional_step_cost"}

#reward config
REWARD_DEFAULTS = {"correct_reward": 10.0, "wrong_reward": -2.0, "step_cost": 0.0}


@dataclass(frozen=True)
# tasks represent a single instruction
# kind determines positive/negative instruction
# color determines instruction color
# object_type determines instruction object
class Task:
    kind: str
    color: str
    object_type: str

    # id = "kind:color:object_type"
    @property
    def id(self):
        return f"{self.kind}:{self.color}:{self.object_type}"

    # e.g. mission = "pick up a ball that is not red"
    @property
    def mission(self):
        if self.kind == "positive":
            return f"pick up a {self.color} {self.object_type}"
        return f"pick up a {self.object_type} that is not {self.color}"

    # positive instruction
    # familiar_negative: color appears in negative training instructions
    # held_out_negative: color is withheld from negative training instruction
    @property
    def category(self):
        if self.kind == "positive":
            return "positive"
        return "familiar_negative" if self.color in TRAIN_COLORS else "held_out_negative"

    # determines if an object is correct for the given instruction
    def matches(self, obj):
        return (obj is not None and obj.type == self.object_type
                and ((obj.color == self.color) if self.kind == "positive"
                     else (obj.color != self.color)))

# creates every instruction combination
TASKS = tuple(Task(kind, color, obj) for kind in ("positive", "negative")
              for color in COLORS for obj in TYPES)
# creates id dictionary lookup from id to a task created from above
TASK_BY_ID = {task.id: task for task in TASKS}
# builds the training instruction set (removes withheld instructions )
TRAIN_TASKS = tuple(task for task in TASKS if task.category != "held_out_negative")
