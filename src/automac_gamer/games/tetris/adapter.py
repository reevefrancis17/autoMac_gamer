from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from automac_gamer.backends.pyboy.session import PyBoySession
from automac_gamer.config import EnvConfig, RewardConfig
from automac_gamer.core.interfaces import GameAdapter, Observation, StepResult

BLANK_TILE: Final[int] = 47
LOSE_TILE: Final[int] = 135
TETROMINOES: Final[tuple[str, ...]] = ("I", "O", "T", "L", "J", "S", "Z")
ACTION_MAP: Final[tuple[str | None, ...]] = (None, "left", "right", "a", "down")


def _next_piece_one_hot(piece: str) -> np.ndarray:
    one_hot = np.zeros(len(TETROMINOES), dtype=np.float32)
    if piece in TETROMINOES:
        one_hot[TETROMINOES.index(piece)] = 1.0
    return one_hot


def _board_to_minimal(board: np.ndarray) -> np.ndarray:
    minimal = np.ones_like(board, dtype=np.float32)
    minimal[board == BLANK_TILE] = 0.0
    minimal[board == LOSE_TILE] = 2.0
    return minimal


@dataclass(slots=True)
class EpisodeStats:
    score: int = 0
    lines: int = 0
    level: int = 0
    steps: int = 0


class TetrisAdapter(GameAdapter):
    def __init__(
        self,
        env_config: EnvConfig,
        *,
        render_mode: str = "headless",
        seed: int | None = None,
    ) -> None:
        if not env_config.rom_path:
            raise ValueError(
                "ROM path missing. Set env.rom_path in config or AUTOMAC_GAMER_TETRIS_ROM."
            )
        if render_mode not in {"headless", "human"}:
            raise ValueError("render_mode must be 'headless' or 'human'")

        self.env_config = env_config
        self.reward_config: RewardConfig = env_config.reward
        self.render_mode = render_mode
        self.seed = seed

        speed = env_config.headless_speed if render_mode == "headless" else env_config.watch_speed
        window = "null" if render_mode == "headless" else "SDL2"
        self._session = PyBoySession(
            rom_path=Path(env_config.rom_path),
            window=window,
            scale=3,
            speed=speed,
            debug=False,
        )

        self._started = False
        self._last_stats = EpisodeStats()
        self._episode_steps = 0

    @property
    def action_space_n(self) -> int:
        return len(ACTION_MAP)

    def _timer_seed(self, seed: int | None) -> int | None:
        if seed is None:
            return None
        return seed % self.env_config.reset_seed_modulo

    def _read_stats(self) -> EpisodeStats:
        return EpisodeStats(
            score=self._session.score,
            lines=self._session.lines,
            level=self._session.level,
            steps=self._episode_steps,
        )

    def _observation(self) -> Observation:
        board_raw = self._session.game_area()
        board = _board_to_minimal(board_raw)
        stats = self._read_stats()
        next_piece = _next_piece_one_hot(self._session.next_tetromino())

        return {
            "board": board,
            "next_piece": next_piece,
            "score": float(stats.score),
            "lines": float(stats.lines),
            "level": float(stats.level),
            "steps": float(stats.steps),
        }

    def _reward(self, stats: EpisodeStats, done: bool) -> float:
        line_delta = max(0, stats.lines - self._last_stats.lines)
        score_delta = max(0, stats.score - self._last_stats.score)
        reward = (
            self.reward_config.survival_reward
            + line_delta * self.reward_config.line_clear_reward
            + score_delta * self.reward_config.score_delta_scale
        )
        if done:
            reward += self.reward_config.game_over_penalty
        return float(reward)

    def reset(self, seed: int | None = None) -> Observation:
        if not self._started:
            self._session.start_game(timer_div=self._timer_seed(seed if seed is not None else self.seed))
            self._started = True
        else:
            self._session.reset_game(timer_div=self._timer_seed(seed if seed is not None else self.seed))
        self._episode_steps = 0
        self._last_stats = self._read_stats()
        return self._observation()

    def step(self, action: int) -> StepResult:
        if action < 0 or action >= self.action_space_n:
            raise ValueError(f"invalid action {action}")

        button = ACTION_MAP[action]
        if button:
            self._session.press(button)

        ticks = max(1, self.env_config.frame_skip * self.env_config.action_repeat)
        render = self.render_mode == "human"
        self._session.tick(ticks=ticks, render=render)

        self._episode_steps += 1
        done = self._session.game_over() or self._episode_steps >= self.env_config.max_episode_steps
        stats = self._read_stats()
        reward = self._reward(stats, done)
        self._last_stats = stats

        observation = self._observation()
        info = {
            "score": stats.score,
            "lines": stats.lines,
            "level": stats.level,
            "steps": self._episode_steps,
        }
        return StepResult(observation=observation, reward=reward, done=done, info=info)

    def close(self) -> None:
        self._session.close()
