from __future__ import annotations

import os

import numpy as np
import pytest

from automac_gamer.config import EnvConfig, RewardConfig
from automac_gamer.games.tetris.adapter import TetrisAdapter


ROM_ENV = "AUTOMAC_GAMER_TETRIS_ROM"


@pytest.mark.skipif(not os.getenv(ROM_ENV), reason="Set AUTOMAC_GAMER_TETRIS_ROM to run integration test")
def test_tetris_headless_step() -> None:
    cfg = EnvConfig(
        rom_path=os.environ[ROM_ENV],
        action_repeat=2,
        frame_skip=1,
        max_episode_steps=50,
        headless_speed=0,
        watch_speed=1,
        reward=RewardConfig(),
    )
    env = TetrisAdapter(cfg, render_mode="headless", seed=42)
    try:
        obs_a = env.reset(seed=11)
        obs_b = env.reset(seed=11)
        assert np.array_equal(obs_a["board"], obs_b["board"])

        result = env.step(0)
        assert isinstance(result.reward, float)
        assert "board" in result.observation
        assert "score" in result.info
    finally:
        env.close()
