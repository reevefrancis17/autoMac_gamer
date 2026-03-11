from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from automac_gamer.config import ExperimentConfig
from automac_gamer.core.interfaces import Watcher
from automac_gamer.games.tetris.adapter import TetrisAdapter
from automac_gamer.rl.dqn import DQNAgent, flatten_observation
from automac_gamer.trainer import find_latest_checkpoint, resolve_device


@dataclass(slots=True)
class WatcherOptions:
    checkpoint: str = "latest"
    refresh_seconds: float | None = None


class TetrisWatcher(Watcher):
    def __init__(self, config: ExperimentConfig, options: WatcherOptions) -> None:
        self.config = config
        self.options = options
        self.device = resolve_device(config.device.preference)
        self.base_run_dir = Path(config.run.run_dir)
        self._loaded_checkpoint: Path | None = None
        self._loaded_mtime: float = 0.0

        self.env = TetrisAdapter(config.env, render_mode="human", seed=config.run.seed)
        initial_obs = self.env.reset(seed=config.run.seed)
        obs_dim = flatten_observation(initial_obs).shape[0]
        self.agent = DQNAgent(
            observation_dim=obs_dim,
            action_dim=self.env.action_space_n,
            model_cfg=config.model,
            device=self.device,
            seed=config.run.seed,
        )

    def _resolve_checkpoint(self) -> Path:
        if self.options.checkpoint != "latest":
            ckpt = Path(self.options.checkpoint)
            if not ckpt.exists():
                raise FileNotFoundError(f"checkpoint not found: {ckpt}")
            return ckpt
        return find_latest_checkpoint(self.base_run_dir)

    def _maybe_reload(self, force: bool = False) -> None:
        checkpoint = self._resolve_checkpoint()
        mtime = checkpoint.stat().st_mtime
        if force or self._loaded_checkpoint != checkpoint or mtime > self._loaded_mtime:
            self.agent.load(checkpoint, map_location=self.device)
            self._loaded_checkpoint = checkpoint
            self._loaded_mtime = mtime
            print(f"[watch] loaded checkpoint: {checkpoint}")

    def run(self) -> None:
        refresh = self.options.refresh_seconds
        if refresh is None:
            refresh = self.config.watch.checkpoint_poll_seconds

        max_episodes = self.config.watch.watch_episodes
        episode = 0

        try:
            self._maybe_reload(force=True)
            while max_episodes == 0 or episode < max_episodes:
                obs = self.env.reset(seed=self.config.run.seed + episode)
                obs_vec = flatten_observation(obs)
                done = False
                reward_total = 0.0
                score = 0
                lines = 0
                while not done:
                    action = self.agent.act(obs_vec, eval_mode=True)
                    result = self.env.step(action)
                    obs_vec = flatten_observation(result.observation)
                    reward_total += result.reward
                    score = int(result.info["score"])
                    lines = int(result.info["lines"])
                    done = result.done
                episode += 1
                print(
                    f"[watch] episode={episode} score={score} lines={lines} "
                    f"reward={reward_total:.2f}"
                )
                if self.options.checkpoint == "latest" and self.config.watch.refresh_between_episodes:
                    time.sleep(max(0.0, refresh))
                    self._maybe_reload(force=False)
        finally:
            self.env.close()
