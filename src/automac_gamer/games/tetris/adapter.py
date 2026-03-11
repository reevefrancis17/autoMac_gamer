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
ACTION_MAP_CLASSIC: Final[tuple[str | None, ...]] = (None, "left", "right", "a", "down")
ACTION_MAP_DMGT: Final[tuple[str | None, ...]] = (
    None,
    "left",
    "right",
    "a",
    "up",
)

DMGT_WFIELD_START: Final[int] = 0xC8A3
DMGT_WFIELD_LEN: Final[int] = 24 * 10
DMGT_HLEVEL: Final[int] = 0xFF8E
DMGT_HLINECLEARCT: Final[int] = 0xFFB0
DMGT_HSCORE: Final[int] = 0xFFB8
DMGT_HNEXTPIECE: Final[int] = 0xFFD2
DMGT_HCURRENTPIECEX: Final[int] = 0xFFE0
DMGT_HMODE: Final[int] = 0xFFE5
DMGT_HGAMESTATE: Final[int] = 0xFFFE

DMGT_TILE_FIELD_EMPTY: Final[int] = 108
DMGT_WRNGMODESTATE: Final[int] = 0xCF38
DMGT_WDROPMODESTATE: Final[int] = 0xCF3A
DMGT_WSPEEDCURVESTATE: Final[int] = 0xCF3B
DMGT_STATE_GAMEPLAY: Final[int] = 3
DMGT_STATE_GAMEPLAY_BIG: Final[int] = 6
DMGT_MODE_GAME_OVER: Final[int] = 21
DMGT_MODE_PRE_GAME_OVER: Final[int] = 24
DMGT_RNG_MODE_TGM3: Final[int] = 2
DMGT_DROP_MODE_HARD: Final[int] = 2
DMGT_SCURVE_CHILL: Final[int] = 5


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
        self._session.open()
        title = self._session.cartridge_title.upper().strip()
        self._rom_kind = "classic" if title == "TETRIS" else "dmgtris"

        self._started = False
        self._snapshot: bytes | None = None
        self._last_stats = EpisodeStats()
        self._episode_steps = 0
        self._last_line_clear_ct = 0
        self._estimated_lines = 0
        self._last_board_metrics: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._guided_plan: list[int] = []

    @property
    def action_space_n(self) -> int:
        if self._rom_kind == "classic":
            return len(ACTION_MAP_CLASSIC)
        return len(ACTION_MAP_DMGT)

    def _timer_seed(self, seed: int | None) -> int | None:
        if seed is None:
            return None
        return seed % self.env_config.reset_seed_modulo

    def _read_stats(self) -> EpisodeStats:
        if self._rom_kind != "classic":
            return self._read_stats_dmgtris()
        return EpisodeStats(
            score=self._session.score,
            lines=self._session.lines,
            level=self._session.level,
            steps=self._episode_steps,
        )

    def _observation(self) -> Observation:
        if self._rom_kind != "classic":
            return self._observation_dmgtris()
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

    def _decode_dmgtris_score(self) -> int:
        digits = self._session.read_mem_slice(DMGT_HSCORE, 8).astype(np.int32)
        digits = np.clip(digits, 0, 9)
        value = 0
        for d in digits:
            value = value * 10 + int(d)
        return int(value)

    def _read_stats_dmgtris(self) -> EpisodeStats:
        level_lo = self._session.read_mem(DMGT_HLEVEL)
        level_hi = self._session.read_mem(DMGT_HLEVEL + 1)
        level = int(level_lo + (level_hi << 8))
        score = self._decode_dmgtris_score()
        return EpisodeStats(
            score=score,
            lines=self._estimated_lines,
            level=level,
            steps=self._episode_steps,
        )

    def _dmgtris_next_piece(self) -> np.ndarray:
        piece_idx = self._session.read_mem(DMGT_HNEXTPIECE)
        one_hot = np.zeros(len(TETROMINOES), dtype=np.float32)
        if 0 <= piece_idx < len(TETROMINOES):
            one_hot[piece_idx] = 1.0
        return one_hot

    def _dmgtris_board_occupancy(self) -> np.ndarray:
        board_raw = self._session.read_mem_slice(DMGT_WFIELD_START, DMGT_WFIELD_LEN).reshape(24, 10)
        board_visible = board_raw[4:, :]
        board = np.ones_like(board_visible, dtype=np.float32)
        board[board_visible == DMGT_TILE_FIELD_EMPTY] = 0.0
        return board

    @staticmethod
    def _dmgtris_board_metrics(board: np.ndarray) -> tuple[int, int, int, int]:
        filled = int(board.sum())
        heights: list[int] = []
        holes = 0
        for col in range(board.shape[1]):
            col_data = board[:, col]
            filled_cells = np.where(col_data > 0.0)[0]
            if filled_cells.size == 0:
                heights.append(0)
                continue
            top = int(filled_cells[0])
            heights.append(board.shape[0] - top)
            holes += int(np.sum(col_data[top:] == 0.0))
        max_height = max(heights) if heights else 0
        bumpiness = int(sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1)))
        return filled, max_height, holes, bumpiness

    def _observation_dmgtris(self, board: np.ndarray | None = None) -> Observation:
        if board is None:
            board = self._dmgtris_board_occupancy()
        next_piece = self._dmgtris_next_piece()
        stats = self._read_stats_dmgtris()
        return {
            "board": board,
            "next_piece": next_piece,
            "score": float(stats.score),
            "lines": float(stats.lines),
            "level": float(stats.level),
            "steps": float(stats.steps),
        }

    def _reward_dmgtris(
        self,
        stats: EpisodeStats,
        done: bool,
        current_board_metrics: tuple[int, int, int, int],
    ) -> float:
        line_delta = max(0, stats.lines - self._last_stats.lines)
        score_delta = max(0, stats.score - self._last_stats.score)
        prev_filled, prev_height, prev_holes, prev_bump = self._last_board_metrics
        filled, max_height, holes, bumpiness = current_board_metrics

        reward = (
            self.reward_config.survival_reward
            + line_delta * self.reward_config.line_clear_reward
            + score_delta * self.reward_config.score_delta_scale
        )
        reward += 0.02 * float(prev_filled - filled)
        reward -= 0.01 * float(max(0, holes - prev_holes))
        reward -= 0.003 * float(max(0, bumpiness - prev_bump))
        reward -= 0.01 * float(max(0, max_height - prev_height))

        if done:
            reward += self.reward_config.game_over_penalty
        return float(reward)

    def guided_action(self, global_step: int) -> int | None:
        if self._rom_kind != "dmgtris":
            return None
        # Fast shaping policy for early exploration.
        board = self._dmgtris_board_occupancy()
        heights = []
        for col in range(board.shape[1]):
            col_data = board[:, col]
            filled_cells = np.where(col_data > 0.0)[0]
            if filled_cells.size == 0:
                heights.append(0)
            else:
                heights.append(board.shape[0] - int(filled_cells[0]))
        target_col = int(np.argmin(np.asarray(heights, dtype=np.int32)))
        current_x = int(self._session.read_mem(DMGT_HCURRENTPIECEX))
        if current_x > target_col + 1:
            return 1  # left
        if current_x < target_col:
            return 2  # right
        if global_step % 8 == 0:
            return 3  # rotate sometimes
        return 4  # hard drop

    def _simulate_dmgtris_action(self, action: int) -> None:
        button = ACTION_MAP_DMGT[action]
        if button:
            self._session.press(button)
        self._session.tick(1, False)

    def _score_dmgtris_candidate(self) -> float:
        board = self._dmgtris_board_occupancy()
        filled, max_height, holes, bumpiness = self._dmgtris_board_metrics(board)
        line_clear_ct = int(self._session.read_mem(DMGT_HLINECLEARCT))
        score = self._decode_dmgtris_score()
        mode = int(self._session.read_mem(DMGT_HMODE))
        done = mode in (DMGT_MODE_GAME_OVER, DMGT_MODE_PRE_GAME_OVER)
        heuristic = (
            line_clear_ct * 220.0
            + score * 0.05
            - holes * 7.0
            - bumpiness * 1.5
            - max_height * 5.0
            - filled * 0.2
        )
        if done:
            heuristic -= 800.0
        return float(heuristic)

    def _compute_dmgtris_guided_plan(self) -> list[int]:
        snapshot = self._session.save_state_bytes()
        best_score = -1e18
        best_plan: list[int] = [4]

        candidates: tuple[tuple[int, ...], ...] = (
            (4,),
            (1, 4),
            (2, 4),
            (1, 1, 4),
            (2, 2, 4),
            (3, 4),
            (3, 1, 4),
            (3, 2, 4),
        )

        for candidate in candidates:
            self._session.load_state_bytes(snapshot)
            for action in candidate:
                self._simulate_dmgtris_action(action)

            for _ in range(12):
                mode = int(self._session.read_mem(DMGT_HMODE))
                if mode != 15:
                    break
                self._session.tick(1, False)

            heuristic = self._score_dmgtris_candidate()
            if heuristic > best_score:
                best_score = heuristic
                best_plan = list(candidate)

        self._session.load_state_bytes(snapshot)
        return best_plan

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

    def _bootstrap_dmgtris(self) -> None:
        # Force easier settings before entering gameplay.
        self._session.write_mem(DMGT_WRNGMODESTATE, DMGT_RNG_MODE_TGM3)
        self._session.write_mem(DMGT_WDROPMODESTATE, DMGT_DROP_MODE_HARD)
        self._session.write_mem(DMGT_WSPEEDCURVESTATE, DMGT_SCURVE_CHILL)

        # Walk menus until gameplay state is active.
        for frame in range(5000):
            state = self._session.read_mem(DMGT_HGAMESTATE)
            if state in (DMGT_STATE_GAMEPLAY, DMGT_STATE_GAMEPLAY_BIG):
                break
            if frame % 20 == 0:
                self._session.press("start")
            if frame % 60 == 0:
                self._session.press("a")
            self._session.tick(1, False)
        else:
            raise RuntimeError("failed to reach gameplay state for Pandora's Blocks")

        # Let READY/GO settle.
        self._session.tick(180, False)
        self._snapshot = self._session.save_state_bytes()

    def _reset_dmgtris(self, seed: int | None) -> None:
        if self._snapshot is None:
            self._bootstrap_dmgtris()
        assert self._snapshot is not None
        self._session.load_state_bytes(self._snapshot)
        if seed is not None:
            jitter = int(seed % 61)
            if jitter:
                self._session.tick(jitter, False)
        self._last_line_clear_ct = 0
        self._estimated_lines = 0

    def reset(self, seed: int | None = None) -> Observation:
        if self._rom_kind == "classic":
            if not self._started:
                self._session.start_game(timer_div=self._timer_seed(seed if seed is not None else self.seed))
                self._started = True
            else:
                self._session.reset_game(timer_div=self._timer_seed(seed if seed is not None else self.seed))
        else:
            self._reset_dmgtris(seed if seed is not None else self.seed)
        self._episode_steps = 0
        self._guided_plan.clear()
        self._last_stats = self._read_stats()
        if self._rom_kind == "classic":
            return self._observation()
        board = self._dmgtris_board_occupancy()
        self._last_board_metrics = self._dmgtris_board_metrics(board)
        return self._observation_dmgtris(board)

    def step(self, action: int) -> StepResult:
        if action < 0 or action >= self.action_space_n:
            raise ValueError(f"invalid action {action}")

        action_map = ACTION_MAP_CLASSIC if self._rom_kind == "classic" else ACTION_MAP_DMGT
        button = action_map[action]
        if button:
            self._session.press(button)

        ticks = max(1, self.env_config.frame_skip * self.env_config.action_repeat)
        render = self.render_mode == "human"
        self._session.tick(ticks=ticks, render=render)

        self._episode_steps += 1
        if self._rom_kind == "classic":
            done = self._session.game_over() or self._episode_steps >= self.env_config.max_episode_steps
        else:
            line_clear_ct = self._session.read_mem(DMGT_HLINECLEARCT)
            if line_clear_ct > 0 and self._last_line_clear_ct == 0:
                self._estimated_lines += int(line_clear_ct)
            self._last_line_clear_ct = int(line_clear_ct)
            mode = self._session.read_mem(DMGT_HMODE)
            state = self._session.read_mem(DMGT_HGAMESTATE)
            done = (
                mode in (DMGT_MODE_GAME_OVER, DMGT_MODE_PRE_GAME_OVER)
                or state not in (DMGT_STATE_GAMEPLAY, DMGT_STATE_GAMEPLAY_BIG)
                or self._episode_steps >= self.env_config.max_episode_steps
            )
        stats = self._read_stats()
        if self._rom_kind == "classic":
            reward = self._reward(stats, done)
            observation = self._observation()
        else:
            board = self._dmgtris_board_occupancy()
            metrics = self._dmgtris_board_metrics(board)
            reward = self._reward_dmgtris(stats, done, metrics)
            self._last_board_metrics = metrics
            observation = self._observation_dmgtris(board)
        self._last_stats = stats

        info = {
            "score": stats.score,
            "lines": stats.lines,
            "level": stats.level,
            "steps": self._episode_steps,
        }
        return StepResult(observation=observation, reward=reward, done=done, info=info)

    def close(self) -> None:
        self._session.close()
