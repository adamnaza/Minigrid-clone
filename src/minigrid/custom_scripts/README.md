# Custom BabyAI training scripts

This folder trains **GRU/CNN and frozen FLAN-T5/CNN agents** on positive and
color-negation pickup instructions, using `BabyAI-PickupDist-v0` as the base.
Both agents share the environment, rewards, PPO loop, and downstream architecture.
Held-out evaluation is available for both agents; the full-LLM agent remains deferred.

## Files and responsibilities

| File | What it does |
|---|---|
| [pickup_tasks.py](pickup_tasks.py) | Defines instruction templates, colors, object types, correct-object matching, the training instruction list, and default environment/reward settings. |
| [pickup_negation.py](pickup_negation.py) | Extends BabyAI's `PickupDist`. Selects instructions, places objects, verifies pickups, and calculates rewards. |
| [__init__.py](__init__.py) | Registers the training environment as `BabyAI-PickupNegation-v0` when the package is imported. |
| [frozen_llm_model.py](frozen_llm_model.py) | Reuses the starter actor-critic, replacing GRU text processing with frozen FLAN, masked mean pooling, and a learned linear projection. |
| [frozen_llm_preprocessor.py](frozen_llm_preprocessor.py) | Tokenizes missions with FLAN’s tokenizer, supplies attention masks, and reuses image preprocessing. |
| [train.py](train.py) | Selects the agent and adapts the starter training script to create our environments, run PPO, log statistics, and save/resume checkpoints. |
| [evaluate.py](evaluate.py) | Evaluates final checkpoints on shared held-out instructions/scenes and writes episode and summary CSVs with original and common returns. |
| [run_reward_sweep.py](run_reward_sweep.py) | Runs the eight reward configurations for both agents sequentially, with separate run directories. |

The GRU model comes from [rl_starter_files/model.py](../../rl_starter_files/model.py).
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

Ordinary `reset()` calls sample from `TRAIN_TASKS` as before. Evaluation can select
an instruction with `reset(seed=0, options={"task": "negative:purple:ball"})`.
This override applies only to that reset; the next reset without a task samples
training instructions again. No separate test environment or saved manifest is
required. The seed controls random scene generation.

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
loads its model, optimizer, and cumulative training counters (plus vocabulary for
the GRU). Resume with the same agent and model/tokenizer settings.

| Setting | Default | Meaning |
|---|---|---|
| `--agent` | `gru` | Choose `gru` or `frozen-flan` |
| `--llm-model` | `google/flan-t5-base` | Pretrained encoder/tokenizer for the frozen agent |
| `--llm-revision` | `main` | Shared Hugging Face revision; use a commit ID to pin an experiment |
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
- `status.pt`: latest saved model, optimizer, settings, and counters; GRU checkpoints also include their word vocabulary. FLAN checkpoints include frozen encoder weights, so they are much larger. The tokenizer is loaded from the configured model/revision.
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

Earlier local CPU checks passed for the GRU/CNN pipeline:

- 1,000 generated episodes covered all 27 training instructions and excluded held-out negations.
- Scripted pickups checked recovery, reward values, the optional action cost, and 500-step timeouts.
- Training completed 32,768 steps with 16 environments and resumed to 40,960 steps.
- CNN, GRU, actor, and critic parameters changed and remained finite.
- Checkpoint restoration, CSV logging, and TensorBoard output worked.

After the uv refactor and frozen-agent implementation, CPU checks also passed:

- The environment checks above were rerun against the new package paths.
- GRU training ran for 2,048 steps and resumed to 3,072 steps.
- Real FLAN-T5-base training ran for 128 steps and resumed to 1,024 steps.
- Saved FLAN encoder weights exactly matched the original pretrained encoder;
  the projection, CNN, actor, and critic changed, and optimizer counters continued.
- Two offline tests passed for pooling, padding, frozen weights, gradients, and PPO.
- Existing wrapper tests: 2,051 passed and 192 skipped.

These checks used PyTorch 2.15.0 with CUDA 13.2, whose CUDA build required a newer
driver than the local driver exposes, so they ran on CPU. The project now pins
PyTorch 2.11.0 with CUDA 12.8 on Linux/Windows for compatibility with that driver.
Two incomplete CUDA library installations were repaired before verification.
The new report is `misc/frozen_llm_verification/report.json`.

The verification script, logs, checkpoints, and report are local artifacts under
ignored `misc/` and are not required for training. These checks establish that the
pipeline runs and updates parameters, not convergence or generalization.

## Frozen FLAN-T5/CNN training

Install the optional dependencies and keep the extra on subsequent uv commands:

```bash
uv sync --extra llm

# Short CPU check, using a new model name.
uv run --extra llm python -m minigrid.custom_scripts.train \
  --agent frozen-flan --model flan_smoke --frames 128 \
  --procs 2 --frames-per-proc 16 --batch-size 16 --epochs 1 --device cpu

# Full pilot, once a short run works on the chosen device.
uv run --extra llm python -m minigrid.custom_scripts.train \
  --agent frozen-flan --model flan_pilot --frames 3000000 --device cuda
```

The first FLAN run downloads the public model/tokenizer from Hugging Face. Later
runs reuse downloaded files; **instruction embeddings are not cached**. CPU works
without CUDA. Use `--device cuda` only with a compatible PyTorch build and driver.
A small smoke run verifies execution, not learning quality or the full-run budget.

The model uses FLAN-T5-base's **encoder only**, in float32. It averages final token
vectors over non-padding positions, then applies one trainable `Linear(768, 128)`
without extra layers or activations. Concatenating with the 64 CNN features keeps
the actor/critic input at 192 features. The CNN, actor, critic, and forward method
are inherited from the starter `ACModel`; only text encoding is replaced.

The encoder has gradients disabled and remains in evaluation mode even when PPO
calls `train()`. Its output is computed under `no_grad()`, but the projection is
outside that context and learns normally. Starter initialization runs before FLAN
is attached, preserving pretrained weights. The model's `parameters()` yields only
trainable parameters because torch-ac's gradient-norm logging assumes gradients
exist; `named_parameters()` and `state_dict()` still include the encoder.

The preprocessor returns image tensors plus a nested `DictList` containing token
IDs and attention masks. PPO can slice these together into minibatches. FLAN uses
its pretrained tokenizer, not the GRU's learned word-ID vocabulary.

Use the same `--agent`, `--llm-model`, and `--llm-revision` when resuming a FLAN run.
Use a distinct model directory for each agent. The starter evaluator/visualizer
constructs the GRU architecture and **cannot load FLAN checkpoints**; evaluation for both agents is provided by `minigrid.custom_scripts.evaluate`.

Offline model tests use a tiny local T5 and require no model download:

```bash
uv run --extra llm --extra testing pytest tests/test_frozen_llm.py -q
```

## Reward sweep: both agents, eight configurations each

Preview all commands without starting training:

```bash
uv run --extra llm python -m minigrid.custom_scripts.run_reward_sweep --dry-run
```

Start the full sweep:

```bash
uv run --extra llm python -m minigrid.custom_scripts.run_reward_sweep
```

This runs **16 separate training jobs sequentially** (GRU first, then frozen FLAN),
each with 3 million total environment steps, seed 0, and `--device cuda`.
The other exposed training settings match the existing trainer defaults. The full
sweep requests 48 million steps, with each run rounded up to a complete rollout.

Each agent uses all combinations below, always with correct-pickup reward +10:

| Wrong-pickup reward | Per-action reward | Trainer's `--step-cost` |
|---:|---:|---:|
| −2 | 0 | 0 |
| −2 | −1 | 1 |
| −2 | −0.1 | 0.1 |
| −2 | −0.01 | 0.01 |
| −10 | 0 | 0 |
| −10 | −1 | 1 |
| −10 | −0.1 | 0.1 |
| −10 | −0.01 | 0.01 |

The launcher accepts **signed** `--step-penalties` and converts them to positive
costs because the environment subtracts `step_cost`. The cost applies to every
action, including pickups: with cost 1, a correct pickup gives +9 and a wrong
pickup gives −3 or −11. Wrong pickups still allow the episode to continue.
The environment's default rewards are not changed by this script.

Outputs go under `src/minigrid/custom_scripts/runs/reward_sweep/` by default, or
`$RL_STORAGE/reward_sweep/` if set. Override with `--output-dir`. Each agent/reward/
seed combination has its own directory, for example
`frozen-flan_wrong10_cost0p01_seed0/`, containing the trainer's usual logs and
checkpoints. Training output also remains visible in the terminal.

The sweep stops if a job fails. Rerun the same command to resume from saved
checkpoints; runs already at the target budget perform no additional PPO updates.
Use a new output directory for a fresh sweep, and keep the other training settings
unchanged when resuming. To extend the existing runs, pass `--frames 5000000`.

For a short execution check in a **separate** directory:

```bash
uv run --extra llm python -m minigrid.custom_scripts.run_reward_sweep \
  --output-dir misc/reward_sweep_smoke --frames 32 --procs 2 \
  --frames-per-proc 16 --batch-size 16 --epochs 1 \
  --wrong-rewards -2 --step-penalties -0.01 --device cpu
```

This checks both agents on one reward configuration. Select one agent using
`--agents gru` or `--agents frozen-flan`. Other PPO settings, FLAN model selection,
logging, and checkpoint intervals retain the shared trainer's defaults.
Compare agents within each reward configuration; raw returns across different
reward scales are not directly comparable measures of task performance.

## Evaluate trained agents

Evaluate every `*/status.pt` under the reward-sweep directory:

```bash
uv run --extra llm python -m minigrid.custom_scripts.evaluate
```

The default input directory matches the sweep: `src/minigrid/custom_scripts/runs/reward_sweep/`,
or `$RL_STORAGE/reward_sweep/` if set. Use `--runs-dir` for another sweep directory.
The evaluator reads model type, original rewards, and FLAN model/revision from each
checkpoint. It loads GRU vocabularies rather than building new ones. Model weights
stay fixed, dropout is disabled, and actions are sampled from the policy.

Defaults are CUDA, four CPU threads, all nine unseen negations, and scene seeds
0–19 for each instruction: **180 episodes per checkpoint**, or **2,880 for 16 runs**.
The existing 500-action limit, five-action set, partial view, and recovery after
wrong pickups are retained. Final checkpoints currently contain 3,000,320 steps.

Scene generation is identical across agents for the same instruction/scene seed.
Action-sampling seeds are `20 × instruction_index + scene_seed`, giving 0–179.
Instruction order is purple, yellow, grey, with ball, key, box within each color.
Subset checks keep the same mapping. Different policies can still choose different
actions; repeatability assumes the same software and device. Different scene seeds
do not guarantee distinct layouts.

To evaluate one checkpoint, or perform a small execution check:

```bash
uv run --extra llm python -m minigrid.custom_scripts.evaluate \
  --checkpoint src/minigrid/custom_scripts/runs/reward_sweep/gru_wrong2_cost1_seed0/status.pt \
  --tasks negative:purple:ball --scene-seeds 0 1 \
  --output-dir misc/evaluation_smoke --device cpu
```

Each episode produces both scores from the **same trajectory**:

- **Original return:** sum of environment rewards, using the checkpoint's training
  reward settings, including its per-action cost.
- **Common return:** `10 × completion − 2 × wrong_pickups`, with no action cost.

Outputs default to `src/minigrid/custom_scripts/results/evaluation/`; override
with `--output-dir`. There is one subdirectory per training run:

```text
results/evaluation/
├── gru_wrong2_cost0_seed0/
│   ├── episodes.csv
│   └── summary.csv
└── frozen-flan_wrong2_cost0_seed0/
    ├── episodes.csv
    └── summary.csv
```

`episodes.csv` records the checkpoint path, training step count, agent, device,
original reward settings, instruction, both seeds, both returns, episode length,
wrong pickups, success, and timeout. A correct pickup on the last allowed step
counts as success rather than a timeout. Rows are saved after every episode.

`summary.csv` contains one row per instruction plus an overall row: episode count,
mean and population standard deviation of both returns, mean length and wrong
pickups, and diagnostic success/timeout rates. Summary values use individual
episodes, not training-log windows. No plots are generated automatically.

Rerunning evaluation replaces that run's CSVs; it does not resume partial episode
lists. Use separate output directories for earlier checkpoints or different test
settings. The training checkpoints are read only. Summary files appear only after
that checkpoint's evaluation finishes successfully.

Focused offline checks:

```bash
uv run --extra llm --extra testing pytest tests/test_evaluate.py tests/test_frozen_llm.py -q
```

`or`, counting tasks, and the full-LLM policy remain later project work.
