"""Train GRU and frozen FLAN sequentially on the selected reward configurations."""

import argparse
import os
import shlex
import subprocess
import sys
from itertools import product
from pathlib import Path
from minigrid.custom_scripts import get_storage_dir
from minigrid.custom_scripts.pickup_tasks import SCENE_OBJECT_COUNTS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    storage = get_storage_dir()
    parser.add_argument("--reward-mode", choices=("original", "first-pickup", "large-correct", "stop-on-wrong", "negation-study", "disjunction-study"), default="original",
                        help="Select reward behavior and default sweep configurations")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scene-setup", choices=(*SCENE_OBJECT_COUNTS, "three-balls"), default=None,
                        help="Defaults to mixed; study presets select two-balls (negation) or three-balls (disjunction)")
    parser.add_argument("--correct-reward", type=float, default=None,
                        help="Correct bonus for original, first-pickup, or stop-on-wrong modes (default: 10)")
    parser.add_argument("--later-correct-reward", type=float, default=None,
                        help="Override the correct bonus after a mistake in original mode")
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
    study = args.reward_mode == "negation-study"
    disjunction = args.reward_mode == "disjunction-study"
    args.scene_setup = args.scene_setup or ("three-balls" if disjunction else "two-balls" if study else "mixed")
    if args.scene_setup == "three-balls" and not disjunction:
        parser.error("three-balls requires --reward-mode disjunction-study")
    if study or disjunction:
        expected_scene = "three-balls" if disjunction else "two-balls"
        if args.scene_setup != expected_scene or args.recurrence != 1:
            parser.error(f"{args.reward_mode} uses {expected_scene} and no memory (--recurrence 1)")
        if any(v is not None for v in (args.correct_reward, args.later_correct_reward,
                                       args.wrong_rewards, args.step_penalties)):
            parser.error(f"{args.reward_mode} has a fixed reward matrix; omit reward overrides")
    if args.reward_mode == "large-correct" and args.correct_reward is not None:
        parser.error("large-correct uses its own fixed +30 bonus; omit --correct-reward")
    if args.later_correct_reward is not None and args.reward_mode != "original":
        parser.error("--later-correct-reward requires --reward-mode original")

    first_pickup_reward_only = args.reward_mode == "first-pickup"
    terminate_on_wrong_pickup = args.reward_mode == "stop-on-wrong"
    sweep_name = "first_pickup_reward_sweep" if first_pickup_reward_only else "reward_sweep"
    # (first-correct bonus, later-correct bonus, first-pickup-only flag)
    correct_bonus = 10.0 if args.correct_reward is None else args.correct_reward
    reward_variants = [(correct_bonus, args.later_correct_reward, first_pickup_reward_only)]
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
    # Each configuration contains wrong reward, signed step penalty, correct bonus,
    # later bonus, first-pickup-only flag, and terminate-on-wrong flag.
    configurations = [(wrong, step, *variant, terminate_on_wrong_pickup)
                      for wrong, step, variant in product(args.wrong_rewards, args.step_penalties, reward_variants)]
    if study:
        sweep_name = "negation_study"
        configurations = [
            (-2.0, -1.0, 10.0, None, False, False),
            (-2.0, -1.0, 30.0, None, False, False),
            (-2.0, -1.0, 10.0, None, True, False),
            (-2.0, -1.0, 30.0, None, True, False),
            (-2.0, -1.0, 30.0, 10.0, False, False),
            (-5.0, -1.0, 30.0, 10.0, False, False),
            (-2.0, -1.0, 10.0, None, False, True),
            (-2.0, -0.1, 10.0, None, False, True),
        ]
    elif disjunction:
        sweep_name = "disjunction_study"
        configurations = [(-2.0, step, 10.0, None, first_only, stop)
                          for first_only, stop in ((False, True), (True, False), (False, False))
                          for step in (-0.1, -1.0)]
    elif args.scene_setup != "mixed":
        sweep_name += "_" + args.scene_setup.replace("-", "_")
    output_dir = (args.output_dir or storage / sweep_name).resolve()
    env = dict(os.environ, RL_STORAGE=str(output_dir))
    total = len(args.agents) * len(configurations)
    print(f"{total} runs, {args.frames:,} total steps per run; outputs: {output_dir}", flush=True)

    combinations = product(args.agents, configurations)
    for run, (agent, configuration) in enumerate(combinations, 1):
        wrong_reward, step_penalty, correct_reward, later_reward, first_only, terminate_on_wrong_pickup = configuration
        # The environment SUBTRACTS step_cost. Passing a negative cost
        # would reward each action instead of penalizing it.
        step_cost = abs(step_penalty)
        cost_label = f"{step_cost:g}".replace(".", "p")
        model_name = f"{agent}_wrong{abs(wrong_reward):g}_cost{cost_label}_seed{args.seed}"
        if later_reward is not None:
            model_name += f"_first{correct_reward:g}_later{later_reward:g}"
        elif first_only:
            model_name += "_first_pickup"
        if later_reward is None and correct_reward != 10:
            model_name += "_correct" + f"{correct_reward:g}".replace(".", "p")
        if terminate_on_wrong_pickup:
            model_name += "_stop_on_wrong"
        if args.recurrence > 1:
            model_name += f"_recurrence{args.recurrence}"
        if args.scene_setup != "mixed":
            model_name += "_" + args.scene_setup.replace("-", "_")
        if disjunction:
            model_name += "_disjunction"
        command = [sys.executable, "-m", "minigrid.custom_scripts.train",
                   "--agent", agent, "--model", model_name,
                   "--correct-reward", str(correct_reward), "--wrong-reward", str(wrong_reward),
                   "--step-cost", str(step_cost)]
        if disjunction:
            command.extend(["--task-family", "disjunction"])
        if first_only:
            command.append("--first-pickup-reward-only")
        if terminate_on_wrong_pickup:
            command.append("--terminate-on-wrong-pickup")
        if later_reward is not None:
            command.extend(["--later-correct-reward", str(later_reward)])
        for name in ("frames", "seed", "device", "procs", "frames_per_proc",
                     "batch_size", "epochs", "threads", "recurrence", "scene_setup"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        print(f"\n[{run}/{total}] {model_name}\n{shlex.join(command)}", flush=True)
        if not args.dry_run:
            # Separate processes release model/GPU memory between runs.
            # Stop on failure; rerunning resumes the trainer's checkpoints.
            subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
