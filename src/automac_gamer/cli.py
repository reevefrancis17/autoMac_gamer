from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from automac_gamer.config import load_config
from automac_gamer.games.tetris.adapter import TetrisAdapter
from automac_gamer.trainer import TetrisDQNTrainer, find_latest_checkpoint
from automac_gamer.watcher import TetrisWatcher, WatcherOptions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="automac-gamer",
        description="Autoresearch-inspired local game training framework (PyBoy-first).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True, help="Path to TOML experiment config")
    common.add_argument("--rom-path", default=None, help="Override ROM path")
    common.add_argument("--run-dir", default=None, help="Override run directory")

    train_p = sub.add_parser("train", parents=[common], help="Run headless accelerated training")
    train_p.add_argument("--resume-from", default=None, help="Checkpoint path to resume from")

    eval_p = sub.add_parser("eval", parents=[common], help="Evaluate a saved checkpoint")
    eval_p.add_argument("--checkpoint", required=True, help="Checkpoint path or 'latest'")
    eval_p.add_argument("--episodes", type=int, default=5, help="Evaluation episodes")

    watch_p = sub.add_parser("watch", parents=[common], help="Watch a model play in a visible emulator")
    watch_p.add_argument("--checkpoint", default="latest", help="Checkpoint path or 'latest'")
    watch_p.add_argument(
        "--refresh-seconds",
        type=float,
        default=None,
        help="Poll interval for latest checkpoint refresh between episodes",
    )

    smoke_p = sub.add_parser("smoke", parents=[common], help="Smoke-test ROM and environment setup")
    smoke_p.add_argument("--watch", action="store_true", help="Also run a short visible SDL2 check")

    return parser


def _load(args: argparse.Namespace):
    return load_config(
        args.config,
        rom_path=args.rom_path,
        run_dir=args.run_dir,
        resume_from=getattr(args, "resume_from", None),
    )


def _resolve_checkpoint(checkpoint: str, run_dir: str) -> Path:
    if checkpoint == "latest":
        return find_latest_checkpoint(Path(run_dir))
    resolved = Path(checkpoint)
    if not resolved.exists():
        raise FileNotFoundError(f"checkpoint not found: {resolved}")
    return resolved


def cmd_train(args: argparse.Namespace) -> int:
    cfg = _load(args)
    trainer = TetrisDQNTrainer(cfg, training_mode=True)
    latest = trainer.train()
    print(f"[train] complete latest_checkpoint={latest}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = _load(args)
    trainer = TetrisDQNTrainer(cfg, training_mode=False)
    checkpoint = _resolve_checkpoint(args.checkpoint, cfg.run.run_dir)
    metrics = trainer.evaluate(checkpoint=checkpoint, episodes=args.episodes)
    print(
        "[eval] "
        f"episodes={int(metrics['episodes'])} "
        f"avg_reward={metrics['avg_reward']:.3f} "
        f"avg_score={metrics['avg_score']:.1f} "
        f"avg_lines={metrics['avg_lines']:.2f}"
    )
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    cfg = _load(args)
    watcher = TetrisWatcher(
        cfg,
        WatcherOptions(checkpoint=args.checkpoint, refresh_seconds=args.refresh_seconds),
    )
    watcher.run()
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    cfg = _load(args)
    env = TetrisAdapter(cfg.env, render_mode="headless", seed=cfg.run.seed)
    try:
        obs_a = env.reset(seed=123)
        obs_b = env.reset(seed=123)
        if not np.array_equal(obs_a["board"], obs_b["board"]):
            raise RuntimeError("deterministic reset check failed for same seed")

        result = env.step(0)
        if "board" not in result.observation:
            raise RuntimeError("missing board observation after step")
        if not isinstance(result.reward, float):
            raise RuntimeError("reward is not float")
        print("[smoke] headless check passed")
    finally:
        env.close()

    if args.watch:
        watch_env = TetrisAdapter(cfg.env, render_mode="human", seed=cfg.run.seed)
        try:
            watch_env.reset(seed=cfg.run.seed)
            for _ in range(20):
                watch_env.step(0)
            print("[smoke] watch check passed")
        finally:
            watch_env.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "train": cmd_train,
        "eval": cmd_eval,
        "watch": cmd_watch,
        "smoke": cmd_smoke,
    }
    try:
        return handlers[args.command](args)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
