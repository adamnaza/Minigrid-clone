import argparse
import csv
import os
from pathlib import Path
from statistics import mean, pstdev

import torch

from minigrid.custom_scripts.pickup_negation import PickupNegation
from minigrid.custom_scripts.pickup_tasks import TASKS, REWARD_DEFAULTS
from rl_starter_files import utils
from rl_starter_files.model import ACModel

# evaluate on the unseen negative instructions
TEST_TASKS = tuple(t for t in TASKS if t.category == "held_out_negative")


def load_agent(checkpoint, device):
    status = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = status["training_config"]
    rewards = {name: config[name] for name in REWARD_DEFAULTS}
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
                "agent": agent, "device": str(device), **rewards}
    return env, model, preprocess, metadata


@torch.inference_mode()
def run_episode(env, model, preprocess, task, scene_seed, device):
    # Canonical task order keeps this mapping stable even when testing a subset.
    action_seed = TEST_TASKS.index(task) * 20 + scene_seed
    obs, _ = env.reset(seed=scene_seed, options={"task": task.id})
    torch.manual_seed(action_seed)
    memory = torch.zeros(1, model.memory_size, device=device)
    total, wrong_pickups, length = 0.0, 0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        dist, _, memory = model(preprocess([obs], device=device), memory)
        obs, reward, terminated, truncated, info = env.step(dist.sample().item())
        total += reward
        wrong_pickups += int(info["wrong_pickup"])
        length += 1
    success = int(info["success"])
    return {"instruction_id": task.id, "instruction": task.mission,
            "scene_seed": scene_seed, "action_seed": action_seed,
            "original_return": total, "common_return": 10 * success - 2 * wrong_pickups,
            "episode_length": length, "wrong_pickups": wrong_pickups,
            "success": success, "timeout": int(truncated and not success)}


def summarize(rows):
    groups = {key: [r for r in rows if r["instruction_id"] == key]
              for key in dict.fromkeys(r["instruction_id"] for r in rows)}
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
                      timeout_rate=mean(r["timeout"] for r in episodes))
        summaries.append(result)
    return summaries


def evaluate(checkpoint, output_dir, tasks, seeds, device):
    env, model, preprocess, metadata = load_agent(checkpoint, device)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.csv").unlink(missing_ok=True)
        rows = []
        with (output_dir / "episodes.csv").open("w", newline="") as stream:
            for task in tasks:
                for seed in seeds:
                    row = {**metadata, **run_episode(env, model, preprocess, task, seed, device)}
                    if not rows:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    rows.append(row)
                print(f"{checkpoint.parent.name}: {len(rows)}/{len(tasks) * len(seeds)} episodes", flush=True)
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
    storage = Path(os.environ.get("RL_STORAGE", Path(__file__).parent / "runs"))
    parser.add_argument("--runs-dir", type=Path, default=storage / "reward_sweep")
    parser.add_argument("--checkpoint", type=Path, help="Evaluate one checkpoint instead of the sweep")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results" / "evaluation")
    parser.add_argument("--tasks", nargs="+", choices=[t.id for t in TEST_TASKS],
                        default=[t.id for t in TEST_TASKS])
    parser.add_argument("--scene-seeds", nargs="+", type=int, choices=range(20), default=list(range(20)))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    checkpoints = [args.checkpoint] if args.checkpoint else sorted(args.runs_dir.glob("*/status.pt"))
    if not checkpoints:
        parser.error(f"No status.pt checkpoints found under {args.runs_dir}")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    tasks = [t for t in TEST_TASKS if t.id in args.tasks]
    for checkpoint in checkpoints:
        evaluate(checkpoint, args.output_dir / checkpoint.parent.name, tasks, args.scene_seeds, device)


if __name__ == "__main__":
    main()
