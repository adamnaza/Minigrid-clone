"""Train GRU and frozen FLAN sequentially on the selected reward configurations."""

import argparse
import os
import shlex
import subprocess
import sys
from itertools import product
from pathlib import Path
from minigrid.custom_scripts import get_storage_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    storage = get_storage_dir()
    parser.add_argument("--reward-mode", choices=("original", "first-pickup", "large-correct", "stop-on-wrong"), default="original",
                        help="Select reward behavior and default sweep configurations")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--agents", nargs="+", choices=("gru", "frozen-flan"),
                        default=["gru", "frozen-flan"])
    parser.add_argument("--wrong-rewards", nargs="+", type=float, choices=(-2.0, -5.0, -6.0, -10.0, -14.0))
    parser.add_argument("--step-penalties", nargs="+", type=float,
                        choices=(0.0, -1.0, -0.1, -0.01),
                        help="Signed rewards per action; converted to positive --step-cost values")
    parser.add_argument("--frames", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--procs", type=int, default=16)
    parser.add_argument("--frames-per-proc", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--recurrence", type=int, default=1,
                        help="Backpropagation sequence length; values > 1 enable LSTM memory")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without starting training or creating output files")
    args = parser.parse_args(argv)

    first_pickup_reward_only = args.reward_mode == "first-pickup"
    terminate_on_wrong_pickup = args.reward_mode == "stop-on-wrong"
    sweep_name = "first_pickup_reward_sweep" if first_pickup_reward_only else "reward_sweep"
    # (first-correct bonus, later-correct bonus, first-pickup-only flag)
    reward_variants = [(10.0, None, first_pickup_reward_only)]
    if args.reward_mode == "large-correct":
        sweep_name = "large_correct_reward_sweep"
        reward_variants = [(30.0, 0.0, True), (30.0, 10.0, False), (30.0, 30.0, False)]
        if args.wrong_rewards is None:
            args.wrong_rewards = [-2.0, -5.0]
        if args.step_penalties is None:
            args.step_penalties = [-1.0]
    if terminate_on_wrong_pickup:
        sweep_name = "stop_on_wrong_reward_sweep"
        if args.wrong_rewards is None:
            args.wrong_rewards = [-10.0, -2.0]
        if args.step_penalties is None:
            args.step_penalties = [-1.0, -0.1, 0.0]
    if args.wrong_rewards is None:
        args.wrong_rewards = [-2.0, -6.0, -10.0, -14.0] if first_pickup_reward_only else [-2.0, -10.0]
    if args.step_penalties is None:
        args.step_penalties = [-1.0, -0.1] if first_pickup_reward_only else [0.0, -1.0, -0.1, -0.01]
    output_dir = (args.output_dir or storage / sweep_name).resolve()
    env = dict(os.environ, RL_STORAGE=str(output_dir))
    total = len(args.agents) * len(args.wrong_rewards) * len(args.step_penalties) * len(reward_variants)
    print(f"{total} runs, {args.frames:,} total steps per run; outputs: {output_dir}", flush=True)

    combinations = product(args.agents, args.wrong_rewards, args.step_penalties, reward_variants)
    for run, (agent, wrong_reward, step_penalty, variant) in enumerate(combinations, 1):
        correct_reward, later_reward, first_only = variant
        # The environment SUBTRACTS step_cost. Passing a negative cost
        # would reward each action instead of penalizing it.
        step_cost = abs(step_penalty)
        cost_label = f"{step_cost:g}".replace(".", "p")
        model_name = f"{agent}_wrong{abs(wrong_reward):g}_cost{cost_label}_seed{args.seed}"
        if args.reward_mode == "large-correct":
            model_name += f"_first{correct_reward:g}_later{later_reward:g}"
        elif first_only:
            model_name += "_first_pickup"
        if terminate_on_wrong_pickup:
            model_name += "_stop_on_wrong"
        if args.recurrence > 1:
            model_name += f"_recurrence{args.recurrence}"
        command = [sys.executable, "-m", "minigrid.custom_scripts.train",
                   "--agent", agent, "--model", model_name,
                   "--correct-reward", str(correct_reward), "--wrong-reward", str(wrong_reward),
                   "--step-cost", str(step_cost)]
        if first_only:
            command.append("--first-pickup-reward-only")
        if terminate_on_wrong_pickup:
            command.append("--terminate-on-wrong-pickup")
        if later_reward is not None:
            command.extend(["--later-correct-reward", str(later_reward)])
        for name in ("frames", "seed", "device", "procs", "frames_per_proc",
                     "batch_size", "epochs", "threads", "recurrence"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        print(f"\n[{run}/{total}] {model_name}\n{shlex.join(command)}", flush=True)
        if not args.dry_run:
            # Separate processes release model/GPU memory between runs.
            # Stop on failure; rerunning resumes the trainer's checkpoints.
            subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
