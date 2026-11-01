# Running this project with uv

Run the commands below from the repository root (the directory containing
`pyproject.toml`). Shell examples use Bash on Linux/macOS.

## 1. Install uv and get the code

If uv is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal if `uv` is not found. See the official
[uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
for other platforms and installation methods.

For a fresh checkout:

```bash
git clone https://github.com/adamnaza/Minigrid-clone.git
cd Minigrid-clone
```

If you already have the repository, change into your existing checkout instead.
The checkout must include the uv migration changes described by this guide.

## 2. Create the environment and install dependencies

Use Python 3.12 for these examples:

```bash
uv python install 3.12
uv sync --python 3.12 --extra testing --extra wfc
```

This creates `.venv`, installs both local packages (`minigrid` and
`rl_starter_files`) in editable mode, and installs the test and WFC dependencies.
WFC dependencies are included because the repository contains WFC tests.
No separate `pip install`, `setup.py` invocation, or environment activation is
needed. In your IDE, select `.venv/bin/python` as the interpreter.

`pyproject.toml` declares dependencies; `uv.lock` records resolved versions.
PyTorch is pinned to 2.11.0, using the CUDA 12.8 index on Linux/Windows and
PyPI on other platforms. This CUDA build supports the local RTX 4090 driver
576.88; no separate CUDA toolkit installation is needed for these training scripts.
Commit both files when dependency changes update the lockfile. For a checkout
whose lockfile is already current, add `--locked` to `uv sync` to require it to
remain unchanged. See [uv synchronization](https://docs.astral.sh/uv/concepts/projects/sync/).

Check that imports resolve to this checkout:

```bash
uv run python -c "import minigrid, rl_starter_files; print(minigrid.__file__); print(rl_starter_files.__file__)"
uv run python -m minigrid.custom_scripts.train --help
uv run python -m rl_starter_files.scripts.train --help
```

## 3. Run the automated tests

```bash
uv run --extra testing --extra wfc pytest tests
```

For a smaller check:

```bash
uv run --extra testing --extra wfc pytest tests/test_wrappers.py -q
```

On a machine without a display, use SDL's dummy video driver:

```bash
SDL_VIDEODRIVER=dummy uv run --extra testing --extra wfc pytest tests
```

Keep the extras on test commands: a later uv synchronization may otherwise omit
optional dependencies. Automated tests check the software; evaluating a trained
agent, below, measures model performance.

## 4. Train the custom pickup-negation agent

Choose a shared output directory so training and evaluation find the same models:

```bash
export RL_STORAGE="$PWD/runs"
```

Set this in each new terminal used for training/evaluation. The scripts create
the model directories. Without this override, custom training writes under
`src/minigrid/custom_scripts/runs/`, while starter scripts use `./storage/`.

First try a short run to exercise the training pipeline:

```bash
uv run python -m minigrid.custom_scripts.train \
  --model pickup-smoke --frames 2048 --procs 1 \
  --frames-per-proc 128 --batch-size 128 --epochs 1 \
  --device cpu --threads 2 --save-interval 1
```

This checks execution; it is too short to establish learning performance.
For the configured full training run:

```bash
uv run python -m minigrid.custom_scripts.train \
  --model pickup-ppo --frames 3000000 --seed 1 --device cpu
```

Custom training enables text processing by default. Reusing a model name loads
its existing checkpoint; `--frames` is the target cumulative frame count, not
additional frames. Use a new model name for a fresh experiment.
Outputs under `$RL_STORAGE/<model>/` include checkpoints, `config.json`, CSV/text
logs, and TensorBoard events. Training does not write plots into `figures/`.

## 5. Evaluate the custom agent on its training distribution

The starter evaluator needs the custom environments registered in its process.
Use this wrapper to register them before invoking its command-line interface:

```bash
uv run python -c 'import minigrid.custom_scripts; import runpy; runpy.run_module("rl_starter_files.scripts.evaluate", run_name="__main__")' \
  --env BabyAI-PickupNegation-v0 --model pickup-ppo \
  --episodes 100 --procs 1 --text --argmax
```

Use `--model pickup-smoke` to evaluate the short run instead. Match the model's
architecture: custom training uses text by default, so evaluation needs `--text`;
if training used `--recurrence` greater than 1, also pass `--memory`.

This measures performance on the training distribution, not held-out negation
generalization. Only `BabyAI-PickupNegation-v0` is registered; the custom evaluator explicitly
selects held-out instructions through reset options. The wrapper above
is for the GRU/CNN agent, not the frozen FLAN model.

For frozen FLAN training, install and retain the LLM extra:

```bash
uv sync --extra llm
uv run --extra llm python -m minigrid.custom_scripts.train \
  --agent frozen-flan --model flan-smoke --frames 128 --procs 2 \
  --frames-per-proc 16 --batch-size 16 --epochs 1 --device cpu
```

The first run downloads FLAN-T5-base; it stays frozen while the projection, CNN,
actor, and critic train. See the [custom scripts README](src/minigrid/custom_scripts/README.md)
for full training and resuming. Evaluate both architectures on held-out instructions with:

```bash
uv run --extra llm python -m minigrid.custom_scripts.evaluate
```

This reads final checkpoints from `$RL_STORAGE/reward_sweep/` when `RL_STORAGE` is
set, otherwise from `src/minigrid/custom_scripts/runs/reward_sweep/`. Use `--runs-dir`
to specify another directory. Default results go under
`src/minigrid/custom_scripts/results/evaluation/`.

## 6. Train, evaluate, and watch a standard MiniGrid agent

These commands use the same `RL_STORAGE` setting as above:

```bash
uv run python -m rl_starter_files.scripts.train \
  --algo ppo --env MiniGrid-DoorKey-5x5-v0 \
  --model DoorKey --save-interval 10 --frames 80000

uv run python -m rl_starter_files.scripts.evaluate \
  --env MiniGrid-DoorKey-5x5-v0 --model DoorKey \
  --episodes 100 --procs 1 --argmax

uv run python -m rl_starter_files.scripts.visualize \
  --env MiniGrid-DoorKey-5x5-v0 --model DoorKey \
  --episodes 5 --argmax
```

Visualization opens a window and requires a graphical display. If a model was
trained with `--text` or recurrent memory, pass the matching `--text` and/or
`--memory` flags when evaluating or visualizing it.

## Common issues

- **Module not found:** run from the repository root and rerun the sync command
  in step 2. Use `uv run`, and select the project interpreter in your IDE.
- **Checkpoint missing:** check the model name and `RL_STORAGE` in the current
  terminal. Train the model before evaluating it.
- **Model shape mismatch:** match text/memory flags to training and use the same
  environment observation/action spaces.
- **Custom environment not registered:** use the registration wrapper in step 5.
- **Held-out evaluation:** use `minigrid.custom_scripts.evaluate`; the starter
  evaluator samples training instructions and supports only the GRU model.
- **FLAN dependencies:** retain `--extra llm` on uv commands used for this agent.
- **CUDA unavailable:** the installed PyTorch CUDA build must be supported by the
  NVIDIA driver. CPU smoke tests can run even when CUDA is unavailable.
- **Generated files:** `runs/` contains experiment output; keep large checkpoints
  and logs out of commits. `figures/` contains existing documentation assets,
  and `tests/` contains automated tests. Both remain at the repository root.
