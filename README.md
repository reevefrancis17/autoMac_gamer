# autoMac_gamer

`autoMac_gamer` is a public, autoresearch-inspired framework for running local game training experiments on Apple Silicon and other consumer hardware. It keeps the spirit of short, repeatable experiment loops from [karpathy/autoresearch](https://github.com/karpathy/autoresearch), but targets game emulators instead of LLM pretraining.

This repository is a fork for lineage/provenance. The original upstream snapshot is preserved in:
- branch: `upstream-snapshot`
- tag: `upstream-autoresearch-2026-03-10`

## What this repo does

- Defines backend-agnostic interfaces for game adapters and trainers.
- Implements a PyBoy backend and a Game Boy Tetris adapter.
- Provides a trainable baseline (DQN) with checkpointing and evaluation.
- Supports accelerated headless training and a separate live watcher process.
- Uses TOML experiment configs with CLI overrides.

## What this repo does not do

- It does not include or distribute ROMs.
- It does not include a self-modifying coding agent loop.
- It does not require OpenAI credits for training once setup is complete.

## Quick Start

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If needed in your current shell session:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Install dependencies

```bash
uv sync --extra dev
```

### 3. Configure ROM path

Set either:
- `AUTOMAC_GAMER_TETRIS_ROM=/absolute/path/to/tetris.gb`
- or `rom_path` in `configs/tetris_baseline.toml`

### 4. Smoke test

```bash
uv run automac-gamer smoke --config configs/tetris_baseline.toml
```

### 5. Train (headless, accelerated)

```bash
uv run automac-gamer train --config configs/tetris_baseline.toml
```

### 6. Watch a model play (separate process)

```bash
uv run automac-gamer watch --config configs/tetris_baseline.toml --checkpoint latest
```

## CLI

```bash
automac-gamer train --config <path> [--rom-path <path>] [--resume-from <ckpt>] [--run-dir <dir>]
automac-gamer eval --config <path> --checkpoint <path|latest> [--episodes N]
automac-gamer watch --config <path> [--checkpoint <path|latest>] [--refresh-seconds N]
automac-gamer smoke --config <path> [--watch]
```

## Project Structure

```text
src/automac_gamer/
  cli.py
  config.py
  core/interfaces.py
  backends/pyboy/session.py
  games/tetris/adapter.py
  rl/dqn.py
  trainer.py
  watcher.py
configs/
  tetris_baseline.toml
tests/
```

## License

MIT (see [LICENSE](LICENSE)).
