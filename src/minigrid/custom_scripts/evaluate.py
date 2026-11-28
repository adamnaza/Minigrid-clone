import argparse
import csv
import torch
from pathlib import Path
from statistics import mean, pstdev

from minigrid.custom_scripts.pickup_negation import PickupNegation
from minigrid.custom_scripts.pickup_tasks import TASKS, REWARD_DEFAULTS
from minigrid.custom_scripts import disjunction_tasks as disjunction
from minigrid.custom_scripts import get_storage_dir, PROJECT_ROOT
from rl_starter_files import utils
from rl_starter_files.model import ACModel

# Keep unseen instructions first so their existing action seeds remain unchanged.
TEST_TASKS = tuple(t for t in TASKS if t.category == "held_out_negative")
SEEN_NEGATIVE_TASKS = tuple(t for t in TASKS if t.category == "familiar_negative")
SEEN_POSITIVE_TASKS = tuple(t for t in TASKS if t.kind == "positive")
EVALUATION_TASKS = TEST_TASKS + SEEN_NEGATIVE_TASKS + SEEN_POSITIVE_TASKS
INSTRUCTION_SETS = {"unseen": TEST_TASKS, "seen-negative": SEEN_NEGATIVE_TASKS,
                    "seen-positive": SEEN_POSITIVE_TASKS,
                    "seen": SEEN_NEGATIVE_TASKS + SEEN_POSITIVE_TASKS}

RESULTS_DIR_DEFAULT = PROJECT_ROOT / "results" / "evaluation"


def load_agent(checkpoint, device):
    status = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = status["training_config"]
    task_family = config.get("task_family", "negation")
    rewards = {name: config[name] for name in REWARD_DEFAULTS}
    rewards["first_pickup_reward_only"] = config.get("first_pickup_reward_only", False)
    rewards["later_correct_reward"] = config.get("later_correct_reward")
    rewards["terminate_on_wrong_pickup"] = config.get("terminate_on_wrong_pickup", False)
    rewards["scene_setup"] = config.get("scene_setup", "three-balls" if task_family == "disjunction" else "mixed")
    if task_family == "disjunction":
        from minigrid.custom_scripts.pickup_disjunction import PickupDisjunction
        env = PickupDisjunction(**rewards)
    else:
        env = PickupNegation(**rewards)
    memory = config.get("recurrence", 1) > 1
    agent = config.get("agent", "gru")
    if agent == "frozen-flan":
        from minigrid.custom_scripts.frozen_llm_model import FrozenLLMACModel
        from minigrid.custom_scripts.frozen_llm_preprocessor import get_obss_preprocessor

        name, revision = config["llm_model"], config.get("llm_revision", "main")
        obs_space, preprocess = get_obss_preprocessor(env.observation_space, name, revision)
        model = FrozenLLMACModel(obs_space, env.action_space, memory, name, revision)
    else:
        obs_space, preprocess = utils.get_obss_preprocessor(env.observation_space)
        preprocess.vocab.load_vocab(status["vocab"])
        model = ACModel(obs_space, env.action_space, memory, config.get("text", True))
    model.load_state_dict(status["model_state"])
    model.to(device).eval()
    metadata = {"checkpoint": str(checkpoint.resolve()), "training_steps": status["num_frames"],
                "agent": agent, "device": str(device), "task_family": task_family, **rewards}
    return env, model, preprocess, metadata


@torch.inference_mode()
def run_episode(env, model, preprocess, task, scene_seed, device,
                condition=None, paired=False, reverse_order=None):
    # Canonical task order keeps this mapping stable even when testing a subset.
    is_disjunction = isinstance(task, disjunction.DisjunctionTask)
    options = {"task": task.id}
    extra = {}
    if is_disjunction:
        index = disjunction.EVALUATION_TASKS.index(task)
        reverse = task.kind != "positive" and (bool(scene_seed % 2) if reverse_order is None else reverse_order)
        actual_task = disjunction.DisjunctionTask(task.colors[::-1]) if reverse else task
        options = {"task": actual_task.id, "condition": condition, "paired": paired}
        condition_index = disjunction.CONDITIONS.index(condition) if condition else 0
        action_seed = index * 120 + condition_index * 40 + int(reverse) * 20 + scene_seed
        task = actual_task
        extra = {"color_pair": ":".join(task.pair), "wording_order": "reversed" if reverse else "canonical",
                 "paired_scene": paired}
    else:
        action_seed = EVALUATION_TASKS.index(task) * 20 + scene_seed
    obs, _ = env.reset(seed=scene_seed, options=options)
    if is_disjunction:
        extra.update(scene_condition=env.scene_condition, first_picked_color="")
    torch.manual_seed(action_seed)
    memory = torch.zeros(1, model.memory_size, device=device)
    total, wrong_pickups, length = 0.0, 0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        dist, _, memory = model(preprocess([obs], device=device), memory)
        obs, reward, terminated, truncated, info = env.step(dist.sample().item())
        if is_disjunction and not extra["first_picked_color"] and (info["success"] or info["wrong_pickup"]):
            extra["first_picked_color"] = env.carrying.color
        total += reward
        wrong_pickups += int(info["wrong_pickup"])
        length += 1
    success = int(info["success"])
    return {"instruction_id": task.id, "instruction": task.mission,
            "instruction_category": task.category,
            "scene_seed": scene_seed, "action_seed": action_seed,
            "original_return": total, "common_return": 10 * success - 2 * wrong_pickups,
            "episode_length": length, "wrong_pickups": wrong_pickups,
            "success": success, "timeout": int(truncated and not terminated), **extra}


def summarize(rows):
    groups = {key: [r for r in rows if r["instruction_id"] == key]
              for key in dict.fromkeys(r["instruction_id"] for r in rows)}
    if "scene_condition" in rows[0]:
        for condition in dict.fromkeys(r["scene_condition"] for r in rows):
            groups["condition:" + condition] = [r for r in rows if r["scene_condition"] == condition]
        for category in dict.fromkeys(r["instruction_category"] for r in rows):
            groups["category:" + category] = [r for r in rows if r["instruction_category"] == category]
    groups["overall"] = rows
    summaries = []
    for instruction, episodes in groups.items():
        result = {"instruction_id": instruction, "episodes": len(episodes)}
        for field in ("original_return", "common_return"):
            values = [r[field] for r in episodes]
            result.update({field + "_mean": mean(values), field + "_std": pstdev(values)})
        result.update(episode_length_mean=mean(r["episode_length"] for r in episodes),
                      wrong_pickups_mean=mean(r["wrong_pickups"] for r in episodes),
                      success_rate=mean(r["success"] for r in episodes),
                      first_pickup_rate=mean(r["success"] == 1 and r["wrong_pickups"] == 0 for r in episodes),
                      timeout_rate=mean(r["timeout"] for r in episodes))
        summaries.append(result)
    return summaries


def evaluate(checkpoint, output_dir, tasks, seeds, device, *, task_ids=None, paired=False):
    env, model, preprocess, metadata = load_agent(checkpoint, device)
    try:
        family = metadata["task_family"]
        instruction_set = tasks if isinstance(tasks, str) else None
        if instruction_set is not None:
            sets = disjunction.INSTRUCTION_SETS if family == "disjunction" else INSTRUCTION_SETS
            if instruction_set not in sets:
                raise ValueError(f"{instruction_set} is not an instruction set for {family}")
            tasks = sets[instruction_set]
        if task_ids:
            if not set(task_ids).issubset({t.id for t in tasks}):
                raise ValueError("--tasks must belong to the selected instruction set")
            tasks = [t for t in tasks if t.id in task_ids]
        if paired and (family != "disjunction" or any(t.category != "held_out_disjunction" for t in tasks)):
            raise ValueError("--paired-scenes requires held-out disjunction tasks")
        if env.scene_setup != "mixed":
            tasks = [t for t in tasks if t.object_type == "ball"]
            if not tasks:
                raise ValueError("No ball instructions selected for this ball-only checkpoint")
        if output_dir is None:
            if family == "disjunction":
                label = "paired" if paired else instruction_set or "unseen"
                root = RESULTS_DIR_DEFAULT.with_name("evaluation_disjunction_" + label.replace("-", "_"))
            else:
                root = (RESULTS_DIR_DEFAULT if instruction_set in (None, "unseen") else
                        RESULTS_DIR_DEFAULT.with_name("evaluation_" + instruction_set.replace("-", "_")))
            output_dir = root / checkpoint.parent.name
        metadata["instruction_categories"] = ",".join(sorted({task.category for task in tasks}))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.csv").unlink(missing_ok=True)
        rows = []
        with (output_dir / "episodes.csv").open("w", newline="") as stream:
            for task in tasks:
                for seed in seeds:
                    conditions = disjunction.CONDITIONS if family == "disjunction" and task.kind != "positive" and not paired else (None,)
                    for condition in conditions:
                        for reverse in ((False, True) if paired else (None,)):
                            row = {**metadata, **run_episode(env, model, preprocess, task, seed, device,
                                                           condition=condition, paired=paired, reverse_order=reverse)}
                            if not rows:
                                writer = csv.DictWriter(stream, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            stream.flush()
                            rows.append(row)
                print(f"{checkpoint.parent.name}: {len(rows)} episodes evaluated", flush=True)
        summaries = [{**metadata, **row} for row in summarize(rows)]
        with (output_dir / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        overall = summaries[-1]
        print(f"Saved {output_dir}: original={overall['original_return_mean']:.2f}, "
              f"common={overall['common_return_mean']:.2f}", flush=True)
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    storage = get_storage_dir()
    parser.add_argument("--runs-dir", type=Path, default=storage / "reward_sweep")
    parser.add_argument("--checkpoint", type=Path, help="Evaluate one checkpoint instead of the sweep")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--instruction-set", choices=tuple(dict.fromkeys((*INSTRUCTION_SETS, *disjunction.INSTRUCTION_SETS))), default="unseen")
    parser.add_argument("--tasks", nargs="+", choices=sorted({t.id for t in (*EVALUATION_TASKS, *disjunction.EVALUATION_TASKS)}),
                        help="Optional subset of the selected instruction set")
    parser.add_argument("--paired-scenes", action="store_true",
                        help="Disjunction diagnostic: identical scenes across held-out pairs and both word orders")
    parser.add_argument("--scene-seeds", nargs="+", type=int, choices=range(20), default=list(range(20)))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    checkpoints = [args.checkpoint] if args.checkpoint else sorted(args.runs_dir.glob("*/status.pt"))
    if not checkpoints:
        parser.error(f"No status.pt checkpoints found under {args.runs_dir}")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    for checkpoint in checkpoints:
        output = args.output_dir / checkpoint.parent.name if args.output_dir else None
        evaluate(checkpoint, output, args.instruction_set, args.scene_seeds, device,
                 task_ids=args.tasks, paired=args.paired_scenes)


if __name__ == "__main__":
    main()
