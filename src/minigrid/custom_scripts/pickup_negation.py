from gymnasium import spaces

from minigrid.envs.babyai.pickup import PickupDist
from minigrid.envs.babyai.core.verifier import ObjDesc, PickupInstr

from minigrid.custom_scripts.pickup_tasks import (
    CONFIG, COLORS, TYPES, TASK_BY_ID, REWARD_DEFAULTS, SCENE_OBJECT_COUNTS, get_training_tasks,
)

# inherits from ObjDesc class in babyai -> verifier.py
# ObjDesc already knows how to find objects by type/color with find_matching_objs()
class TaskDesc(ObjDesc):
    def __init__(self, task):
        super().__init__(task.object_type, task.color if task.kind == "positive" else None)
        self.task = task

    # override find_matching_objs to handle negative instruction types
    # if instruction is positive type, use original implementation
    # Otherwise filter candidates using the task predicate (negation or disjunction).
    def find_matching_objs(self, env, use_location=True):
        super().find_matching_objs(env, use_location)
        # ObjDesc handles type/positive color; task.matches handles other predicates.
        if self.task.kind != "positive" and use_location:
            # Keep matching objects and their corresponding positions together.
            pairs = [(obj, pos) for obj, pos in zip(self.obj_set, self.obj_poss)
                     if self.task.matches(obj)]
            
            #update the target object set and corresponding positions for the environment
            self.obj_set = [obj for obj, _ in pairs]
            self.obj_poss = [pos for _, pos in pairs]
        return self.obj_set, self.obj_poss

    def surface(self, env):
        self.find_matching_objs(env)
        # Task.mission from pickup_tasks also produces natural language representation of the object description
        return self.task.mission.removeprefix("pick up ")


class TaskPickupInstr(PickupInstr):
    def verify(self, action):
        # Complete task immediately when correct object is picked up
        # does not require the agent to determine it has completed the task by performing the done action
        return self.verify_action(action)


class PickupNegation(PickupDist):
    task_family = "negation"
    task_by_id = TASK_BY_ID

    def __init__(self, render_mode=None,
                 correct_reward=REWARD_DEFAULTS["correct_reward"],
                 wrong_reward=REWARD_DEFAULTS["wrong_reward"],
                 step_cost=REWARD_DEFAULTS["step_cost"],
                 first_pickup_reward_only=False, later_correct_reward=None,
                 terminate_on_wrong_pickup=False, scene_setup="mixed"):
        self.scene_setup = scene_setup
        self.object_count = SCENE_OBJECT_COUNTS[scene_setup]
        self.training_tasks = get_training_tasks(scene_setup)
        #reward config
        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.step_cost = step_cost
        self.first_pickup_reward_only = first_pickup_reward_only
        self.terminate_on_wrong_pickup = terminate_on_wrong_pickup
        if first_pickup_reward_only and later_correct_reward not in (None, 0):
            raise ValueError("First-pickup-only rewards require a zero later-correct reward")
        self.later_correct_reward = (0.0 if first_pickup_reward_only else
                                     correct_reward if later_correct_reward is None else later_correct_reward)

        #store reward settings
        super().__init__(max_steps=CONFIG["max_steps"], render_mode=render_mode)

        # converts actions into integers representations
        # 0 - left
        # 1 - right
        # 2 - forward
        # 3 - pickup
        # 4 - drop
        self.action_space = spaces.Discrete(len(CONFIG["actions"]))

        # determines the min and max reward possible for an action
        bonuses = (0.0, wrong_reward, correct_reward, self.later_correct_reward)
        self.reward_range = (min(bonuses) - step_cost, max(bonuses) - step_cost)

    def reset(self, *, seed=None, options=None):
        self.had_wrong_pickup = False
        # Evaluation can select a task; ordinary resets still sample training tasks.
        task_id = (options or {}).get("task")
        self._selected_task = self.task_by_id[task_id] if task_id is not None else None
        if self.scene_setup != "mixed" and self._selected_task is not None and self._selected_task.object_type != "ball":
            raise ValueError("Ball-only scenes require a ball instruction")
        return super().reset(seed=seed, options=options)

    # choose an instruction, then generate an environment to satisfy it
    def _gen_grid(self, width, height):
        # MiniGrid has seeded its RNG before calling this method. Select once,
        # outside RoomGridLevel's rejection loop, so retries cannot change the task.
        self.task = self._selected_task or self._sample_task()
        # calls our gen_mission override to create environments
        super()._gen_grid(width, height)

    def _sample_task(self):
        return self._rand_elem(self.training_tasks)


    def gen_mission(self):
        task = self.task
        if self.scene_setup != "mixed":
            # Include the named color and sample other colors without replacement.
            # Positive/negative counterparts use the same scene for a fixed seed.
            colors = [task.color]
            while len(colors) < self.object_count:
                colors.append(self._rand_elem([c for c in COLORS if c not in colors]))
            for color in colors:
                self.add_object(0, 0, "ball", color)
        else:
            self._gen_mixed_objects(task)
        self.place_agent(0, 0)
        self.check_objs_reachable()
        self.instrs = TaskPickupInstr(TaskDesc(task), strict=False)

    def _gen_mixed_objects(self, task):
        # gaurantee there will be at least one target object with incorrect target color
        other_color = self._rand_elem([c for c in COLORS if c != task.color])
        # gaurantee there will be at least one object that is the target color, but not the target object
        other_type = self._rand_elem([t for t in TYPES if t != task.object_type])
        # The two same-type objects supply a positive target/color distractor,
        # or a negative forbidden-color distractor/valid target respectively.
        required = [(task.object_type, task.color), (task.object_type, other_color),
                    (other_type, task.color if task.kind == "positive"
                     else self._rand_elem([c for c in COLORS if c != task.color]))]
        
        # adds in required objects
        for obj_type, color in required:
            self.add_object(0, 0, obj_type, color)
        
        #adds in two more random objects/color pairs
        self.add_distractors(0, 0, num_distractors=2, all_unique=True)


    def step(self, action):
        # check if valid action
        if not self.action_space.contains(action):
            raise ValueError("Only left, right, forward, pickup, and drop are available")
        # recall what the agent is currently carrying, if anything
        previously_carrying = self.carrying

        # agent performs an action
        # obs - updated observation
        # _ - original baby ai reward (discarded)
        # terminated - is task completed?
        # truncated - episode reached step limit? 
        # info - additional information
        obs, _, terminated, truncated, info = super().step(action)

        # pickup action performed and succeeded
        acquired = action == self.actions.pickup and previously_carrying is None and self.carrying is not None
        
        # determine if action picked up is correct
        correct = bool(acquired and self.task.matches(self.carrying))
        wrong = bool(acquired and not correct)

        # determine reward for action
        correct_bonus = self.later_correct_reward if self.had_wrong_pickup else self.correct_reward
        reward = (correct_bonus if correct else self.wrong_reward if wrong else 0.0) - self.step_cost
        self.had_wrong_pickup |= wrong
        terminated = terminated or (wrong and self.terminate_on_wrong_pickup)
        info.update(success=correct, wrong_pickup=wrong)
        return obs, reward, terminated, truncated, info
