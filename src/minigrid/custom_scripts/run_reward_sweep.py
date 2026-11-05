"""Train both agents sequentially on the eight requested reward settings."""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from minigrid.custom_scripts import get_storage_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    storage = get_storage_dir()
    parser.add_argument("--reward-mode", choices=("original", "first-pickup"), default="original",
                        help="Select reward behavior and default sweep configurations")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--agents", nargs="+", choices=("gru", "frozen-flan"),
                        default=["gru", "frozen-flan"])
    parser.add_argument("--wrong-rewards", nargs="+", type=float, choices=(-2.0, -6.0, -10.0, -14.0))
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
    sweep_name = "first_pickup_reward_sweep" if first_pickup_reward_only else "reward_sweep"
    if args.wrong_rewards is None:
        args.wrong_rewards = [-2.0, -6.0, -10.0, -14.0] if first_pickup_reward_only else [-2.0, -10.0]
    if args.step_penalties is None:
        args.step_penalties = [-1.0, -0.1] if first_pickup_reward_only else [0.0, -1.0, -0.1, -0.01]
    output_dir = (args.output_dir or storage / sweep_name).resolve()
    env = dict(os.environ, RL_STORAGE=str(output_dir))
    total = len(args.agents) * len(args.wrong_rewards) * len(args.step_penalties)
    print(f"{total} runs, {args.frames:,} total steps per run; outputs: {output_dir}", flush=True)

    run = 0
    for agent in args.agents:
        for wrong_reward in args.wrong_rewards:
            for step_penalty in args.step_penalties:
                # The environment SUBTRACTS step_cost. Passing a negative cost
                # would reward each action instead of penalizing it.
                step_cost = abs(step_penalty)
                cost_label = f"{step_cost:g}".replace(".", "p")
                model_name = f"{agent}_wrong{abs(wrong_reward):g}_cost{cost_label}_seed{args.seed}"
                if first_pickup_reward_only:
                    model_name += "_first_pickup"
                if args.recurrence > 1:
                    model_name += f"_recurrence{args.recurrence}"
                command = [sys.executable, "-m", "minigrid.custom_scripts.train",
                           "--agent", agent, "--model", model_name,
                           "--correct-reward", "10.0", "--wrong-reward", str(wrong_reward),
                           "--step-cost", str(step_cost)]
                if first_pickup_reward_only:
                    command.append("--first-pickup-reward-only")
                for name in ("frames", "seed", "device", "procs", "frames_per_proc",
                             "batch_size", "epochs", "threads", "recurrence"):
                    command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
                run += 1
                print(f"\n[{run}/{total}] {model_name}\n{shlex.join(command)}", flush=True)
                if not args.dry_run:
                    # Separate processes release model/GPU memory between runs.
                    # Stop on failure; rerunning resumes the trainer's checkpoints.
                    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
