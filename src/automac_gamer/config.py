from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import tomllib


@dataclass(slots=True)
class RewardConfig:
    survival_reward: float = 0.01
    line_clear_reward: float = 1.0
    score_delta_scale: float = 0.01
    game_over_penalty: float = -5.0


@dataclass(slots=True)
class EnvConfig:
    rom_path: str | None = None
    action_repeat: int = 4
    frame_skip: int = 1
    max_episode_steps: int = 3000
    headless_speed: int = 0
    watch_speed: int = 1
    reset_seed_modulo: int = 256
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass(slots=True)
class ModelConfig:
    hidden_dim: int = 256
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 128
    replay_capacity: int = 100_000
    learning_starts: int = 2_000
    train_frequency: int = 4
    target_update_interval: int = 2_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000


@dataclass(slots=True)
class DeviceConfig:
    preference: str = "mps"


@dataclass(slots=True)
class RunConfig:
    seed: int = 7
    total_steps: int = 250_000
    log_interval: int = 500
    checkpoint_interval: int = 5_000
    run_dir: str = "runs/tetris_baseline"
    resume_from: str | None = None


@dataclass(slots=True)
class WatchConfig:
    checkpoint_poll_seconds: float = 5.0
    refresh_between_episodes: bool = True
    watch_episodes: int = 0


@dataclass(slots=True)
class ExperimentConfig:
    env: EnvConfig
    model: ModelConfig
    device: DeviceConfig
    run: RunConfig
    watch: WatchConfig

    @property
    def rom_path(self) -> str:
        if not self.env.rom_path:
            raise ValueError(
                "ROM path missing. Set env.rom_path in config or AUTOMAC_GAMER_TETRIS_ROM."
            )
        return self.env.rom_path

    @property
    def reward(self) -> RewardConfig:
        return self.env.reward


def _build_dataclass(cls, values: dict | None):
    if values is None:
        return cls()
    return cls(**values)


def _merge_env_overrides(
    cfg: ExperimentConfig,
    rom_path: str | None = None,
    run_dir: str | None = None,
    resume_from: str | None = None,
) -> ExperimentConfig:
    env_rom = rom_path or cfg.env.rom_path or os.getenv("AUTOMAC_GAMER_TETRIS_ROM")
    if env_rom == "":
        env_rom = None
    cfg.env.rom_path = env_rom
    if run_dir:
        cfg.run.run_dir = run_dir
    if resume_from == "":
        resume_from = None
    if resume_from:
        cfg.run.resume_from = resume_from
    if cfg.run.resume_from == "":
        cfg.run.resume_from = None
    return cfg


def load_config(
    path: str | Path,
    *,
    rom_path: str | None = None,
    run_dir: str | None = None,
    resume_from: str | None = None,
) -> ExperimentConfig:
    config_path = Path(path)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))

    reward_cfg = _build_dataclass(RewardConfig, raw.get("reward"))
    env_raw = raw.get("env") or {}
    env_raw["reward"] = reward_cfg

    cfg = ExperimentConfig(
        env=_build_dataclass(EnvConfig, env_raw),
        model=_build_dataclass(ModelConfig, raw.get("model")),
        device=_build_dataclass(DeviceConfig, raw.get("device")),
        run=_build_dataclass(RunConfig, raw.get("run")),
        watch=_build_dataclass(WatchConfig, raw.get("watch")),
    )
    return _merge_env_overrides(cfg, rom_path=rom_path, run_dir=run_dir, resume_from=resume_from)
