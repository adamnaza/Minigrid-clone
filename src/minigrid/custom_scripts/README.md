# Custom BabyAI training scripts

This folder currently implements training for a **GRU/CNN agent** on positive and
color-negation pickup instructions, using `BabyAI-PickupDist-v0` as the base.
Evaluation is deferred. **FLAN-T5-base is selected for the frozen-LLM agent, but
that agent is not implemented yet.**

Project choices are recorded in [PROJECT_DECISIONS.md](../PROJECT_DECISIONS.md);
the broader scope is in [COURSE_PROJECT.md](../COURSE_PROJECT.md).

## Files and responsibilities

| File | What it does |
|---|---|
| [pickup_tasks.py](pickup_tasks.py) | Defines instruction templates, colors, object types, correct-object matching, the training instruction list, and default environment/reward settings. |
| [pickup_negation.py](pickup_negation.py) | Extends BabyAI's `PickupDist`. Selects instructions, places objects, verifies pickups, and calculates rewards. |
| [__init__.py](__init__.py) | Registers the training environment as `BabyAI-PickupNegation-v0` when the package is imported. |
| [train.py](train.py) | Adapts the starter training script to create our environments, run PPO, log statistics, and save/resume checkpoints. |

The model comes from [rl_starter_files/model.py](../../rl_starter_files/model.py).
The trainer also reuses the starter's observation preprocessing and logging
utilities, and uses `torch_ac` for experience collection and PPO updates.

## Instructions and scene generation

Objects are balls, keys, and boxes. Colors are red, green, blue, purple, yellow,
and grey. The instruction templates are:

- Positive: `pick up a {color} {object_type}`.
- Negative: `pick up a {object_type} that is not {color}`.

There are 36 possible instructions. Training uniformly samples from 27:

| Instruction category | Colors named in the instruction | Count |
|---|---|---:|
| Positive | All six colors | 18 |
| Negative, included in training | Red, green, blue | 9 |
| Negative, withheld for later testing | Purple, yellow, grey | 9 |

Each allowed instruction has probability 1/27 at reset. This gives a **2:1
positive/negative episode ratio in expectation**, not exactly balanced counts in
finite runs. All six colors can appear as scene objects and in positive training
instructions. Only their combinations with negation are withheld.

Generation currently chooses the instruction first, then builds a scene for it:

1. Create one 7×7 room with three required objects.
2. Add two random objects, keeping all five type–color combinations distinct.
3. Randomize object positions and the agent's position and orientation.
4. Use BabyAI's reachability check and retry invalid scenes without changing the instruction.

For `pick up a red ball`, required objects could be a red ball, blue ball, and red
key. For `pick up a ball that is not red`, they could be a red ball, blue ball,
and green key. Extra random objects can also be valid targets for negation.
Scene-first instruction selection was discussed but has not been implemented.

The environment uses BabyAI's inherited `reset()`, always sampling from
`TRAIN_TASKS`. There is no train/test mode, explicit test-task selection, or saved
scene manifest. The seed controls the random sequence; each subsequent reset
produces another instruction and scene.

`CONFIG["max_steps"]` and the configured action count are used directly. The room
size comes from `PickupDist`, and the object count comes from the placement code;
changing the descriptive `room_size` or `objects` entries alone does not alter them.

## Actions, rewards, and observations

| Action ID | Action |
|---|---|
| 0 | Turn left |
| 1 | Turn right |
| 2 | Move forward |
| 3 | Pick up |
| 4 | Drop |

MiniGrid defines these action meanings. The agent can carry one object at a time,
so it must drop a wrong object before picking up another.

| Event | Default reward | Episode behavior |
|---|---:|---|
| Acquire an object satisfying the instruction | +10 | Ends successfully |
| Acquire an incorrect object | −2 | Continues |
| Move, turn, drop, or attempt an unsuccessful pickup | 0 | Continues |
| Reach 500 actions without success | No additional penalty | Truncates |

A penalty is charged only on an actual acquisition. Pressing pickup while already
carrying something does not charge another pickup penalty. Dropping and reacquiring
an incorrect object does. Correct pickup on action 500 still counts as success.

`--step-cost 0.01` subtracts 0.01 from every action, including invalid attempts and
terminal actions. It is optional and defaults to zero. With default pickup rewards:

```text
episode return = 10 × successful_completion − 2 × wrong_pickups − step_cost × actions
```

The agent receives the usual partial observation and mission text. The starter
preprocessor converts the observation image into a tensor and the mission into
word IDs. The model uses a CNN for the observation and a GRU for the instruction,
concatenates their features, and predicts action probabilities and a value estimate.
Temporal memory is disabled by default; the text GRU still processes the mission.
Internal target lists and task matching rules are not supplied to the policy.

## Run training

Run commands from the repository root (the directory containing `pyproject.toml`).
Install the project dependencies first; uv creates and manages `.venv`, so no
manual activation is needed:

```bash
uv sync
```

For installation instructions and automated test commands, see the
[uv run guide](../../../UV_GUIDE.md).

Run training with:

```bash
# Short smoke run in a separate directory.
uv run python -m minigrid.custom_scripts.train --model gru_smoke --frames 4096

# Fresh pilot with the default 3M total environment-step budget.
uv run python -m minigrid.custom_scripts.train --model gru_pilot --seed 1

# Continue the same run to 5M TOTAL steps, not another 5M.
uv run python -m minigrid.custom_scripts.train --model gru_pilot --seed 1 --frames 5000000 --checkpoint-frames 4000000 5000000

# Separate experiment with an action cost.
uv run python -m minigrid.custom_scripts.train --model gru_step_cost --seed 1 --step-cost 0.01
```

Use a new model name for a fresh run. Reusing a directory containing `status.pt`
loads its model, optimizer, vocabulary, and cumulative training counters.

| Setting | Default | Meaning |
|---|---|---|
| `--algo` | `ppo` | Learning algorithm |
| `--frames` | 3,000,000 | Total environment steps, including resumed steps |
| `--procs` | 16 | Environments collecting experience for one shared model |
| `--frames-per-proc` | 128 for PPO | Steps collected per environment before training updates |
| `--batch-size` | 256 | Transitions per PPO minibatch with the default recurrence of 1 |
| `--epochs` | 4 | Passes over each collected rollout |
| `--device` | `cpu` | Training device; use `cuda` with working GPU support |
| `--threads` | 4 | PyTorch intra-operation CPU threads |
| `--recurrence` | 1 | No temporal memory by default |
| `--correct-reward` / `--wrong-reward` / `--step-cost` | 10 / −2 / 0 | Environment reward settings |

With the defaults, each collection cycle gathers 16 × 128 = **2,048 steps** across
all environments. The budget may be exceeded slightly to finish a whole rollout.
Text processing is enabled by default in this custom trainer.

## Logs and checkpoints

Outputs default to `src/minigrid/custom_scripts/runs/<model>/`. An existing `RL_STORAGE`
environment variable overrides this location. The local `runs/` and `results/`
folders are ignored by Git.

- `log.txt`: command, settings, and progress messages.
- `log.csv`: training return/episode-length statistics, speed, and optimization metrics.
- TensorBoard event files: scalar training metrics.
- `config.json`: recorded training and environment settings.
- `status.pt`: latest saved model, optimizer, vocabulary, settings, and counters.
- `checkpoint_1000000.pt`, `checkpoint_2000000.pt`, `checkpoint_3000000.pt`:
  default milestone copies, saved after the rollout that crosses each target.

The recorded checkpoint step count is authoritative; a milestone filename names
the target rather than necessarily the exact step count. `--save-interval` controls
periodic saves in collection/update cycles. Setting it to zero disables only
periodic saves: milestone saves and the save after normal completion still occur.
An interrupted run does not automatically save its latest in-memory state.

Source hashes and custom configuration checks have been removed. **Use a fresh
model name when changing experiment settings.** Incompatible model shapes still
fail during weight loading. Resume restores the optimizer but resets environment
and random-generator state, so it is not an exact continuation of an uninterrupted
random sequence. The recorded configuration is for reference, not validation.

Training logs summarize recent episode returns and lengths. They are not held-out
evaluation results or a complete per-episode dataset.

## Verification performed

Local CPU checks passed for the current GRU/CNN pipeline:

- 1,000 generated episodes covered all 27 training instructions and excluded held-out negations.
- Scripted pickups checked recovery, reward values, the optional action cost, and 500-step timeouts.
- Training completed 32,768 steps with 16 environments and resumed to 40,960 steps.
- CNN, GRU, actor, and critic parameters changed and remained finite.
- Checkpoint restoration, CSV logging, and TensorBoard output worked.

The verification script, logs, checkpoints, and report are local artifacts under
ignored `misc/` and are not required for training. These checks establish that the
pipeline runs and updates parameters, not convergence or generalization.

## Planned frozen-LLM agent and evaluation

**Confirmed model choice:** FLAN-T5-base (`google/flan-t5-base`) for the frozen
instruction encoder. The current trainer still builds the GRU/CNN agent; there is
no frozen-LLM implementation or command yet.

The design under discussion is to use the frozen encoder, pool token vectors while
excluding padding, and train a 768-to-128 projection before concatenating with CNN
features. Pooling, projection, caching, precision, and implementation details remain
to be finalized. See D029 in [PROJECT_DECISIONS.md](../PROJECT_DECISIONS.md).

Evaluation will focus on return over repeated unseen-instruction episodes. Its scene
count, instruction-selection interface, and scripts are deferred. `or` and counting
tasks and the full-LLM policy remain later project work.
