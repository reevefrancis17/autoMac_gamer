from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from automac_gamer.config import ModelConfig
from automac_gamer.rl.dqn import DQNAgent


def test_dqn_checkpoint_roundtrip(tmp_path: Path) -> None:
    model_cfg = ModelConfig(
        hidden_dim=32,
        learning_rate=1e-3,
        gamma=0.99,
        batch_size=4,
        replay_capacity=128,
        learning_starts=4,
        train_frequency=1,
        target_update_interval=8,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=100,
    )
    device = torch.device("cpu")
    agent = DQNAgent(observation_dim=32, action_dim=5, model_cfg=model_cfg, device=device, seed=123)

    for i in range(16):
        obs = np.random.randn(32).astype(np.float32)
        nxt = np.random.randn(32).astype(np.float32)
        agent.buffer.add(obs, action=i % 5, reward=0.1, next_obs=nxt, done=(i % 7 == 0))
        agent.state.global_step += 1
        agent.optimize()

    checkpoint = tmp_path / "checkpoints" / "step_00000016.pt"
    agent.save(checkpoint, extra={"test": True})
    assert checkpoint.exists()

    clone = DQNAgent(observation_dim=32, action_dim=5, model_cfg=model_cfg, device=device, seed=123)
    payload = clone.load(checkpoint, map_location=device)
    assert payload["extra"]["test"] is True
    assert clone.state.global_step == agent.state.global_step
