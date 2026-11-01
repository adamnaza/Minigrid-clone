"""Train both agents sequentially on the eight requested reward settings."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    storage = Path(os.environ.get("RL_STORAGE", Path(__file__).parent / "runs"))
    parser.add_argument("--output-dir", type=Path, default=storage / "reward_sweep")
    parser.add_argument("--agents", nargs="+", choices=("gru", "frozen-flan"),
                        default=["gru", "frozen-flan"])
    parser.add_argument("--wrong-rewards", nargs="+", type=float, choices=(-2.0, -10.0),
                        default=[-2.0, -10.0])
    parser.add_argument("--step-penalties", nargs="+", type=float,
                        choices=(0.0, -1.0, -0.1, -0.01), default=[0.0, -1.0, -0.1, -0.01],
                        help="Signed rewards per action; converted to positive --step-cost values")
    parser.add_argument("--frames", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--procs", type=int, default=16)
    parser.add_argument("--frames-per-proc", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without starting training or creating output files")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
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
                command = [sys.executable, "-m", "minigrid.custom_scripts.train",
                           "--agent", agent, "--model", model_name,
                           "--correct-reward", "10.0", "--wrong-reward", str(wrong_reward),
                           "--step-cost", str(step_cost)]
                for name in ("frames", "seed", "device", "procs", "frames_per_proc",
                             "batch_size", "epochs", "threads"):
                    command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
                run += 1
                print(f"\n[{run}/{total}] {model_name}\n{shlex.join(command)}", flush=True)
                if not args.dry_run:
                    # Separate processes release model/GPU memory between runs.
                    # Stop on failure; rerunning resumes the trainer's checkpoints.
                    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
