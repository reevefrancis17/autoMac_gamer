from __future__ import annotations

from pathlib import Path

from automac_gamer.config import load_config


def test_load_config_with_overrides(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[env]
rom_path = ""
action_repeat = 2
frame_skip = 1
max_episode_steps = 100
headless_speed = 0
watch_speed = 1
reset_seed_modulo = 256

[reward]
survival_reward = 0.1
line_clear_reward = 2.0
score_delta_scale = 0.5
game_over_penalty = -1.0

[model]
hidden_dim = 64
learning_rate = 0.001
gamma = 0.99
batch_size = 8
replay_capacity = 1024
learning_starts = 8
train_frequency = 1
target_update_interval = 16
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay_steps = 100

[device]
preference = "cpu"

[run]
seed = 1
total_steps = 10
log_interval = 2
checkpoint_interval = 5
run_dir = "runs/local"
resume_from = ""

[watch]
checkpoint_poll_seconds = 1.0
refresh_between_episodes = true
watch_episodes = 1
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("AUTOMAC_GAMER_TETRIS_ROM", "/tmp/tetris.gb")
    cfg = load_config(config_path, run_dir="runs/override")
    assert cfg.rom_path == "/tmp/tetris.gb"
    assert cfg.run.run_dir == "runs/override"
    assert cfg.env.action_repeat == 2
    assert cfg.reward.line_clear_reward == 2.0
