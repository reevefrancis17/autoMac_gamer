from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
import json
import random

import numpy as np
import torch
from torch import nn
from torch import optim

from automac_gamer.config import ModelConfig
from automac_gamer.core.interfaces import Observation


def flatten_observation(observation: Observation) -> np.ndarray:
    board = np.asarray(observation["board"], dtype=np.float32).reshape(-1)
    next_piece = np.asarray(observation["next_piece"], dtype=np.float32).reshape(-1)
    stats = np.asarray(
        [
            float(observation["score"]) / 10_000.0,
            float(observation["lines"]) / 100.0,
            float(observation["level"]) / 20.0,
            float(observation["steps"]) / 1_000.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([board, next_piece, stats], dtype=np.float32)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.index = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.index
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": self.obs[indices],
            "next_obs": self.next_obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
        }


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


@dataclass(slots=True)
class DQNState:
    global_step: int
    episode: int
    epsilon: float


class DQNAgent:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        model_cfg: ModelConfig,
        device: torch.device,
        seed: int,
    ) -> None:
        self.model_cfg = model_cfg
        self.device = device
        self.action_dim = action_dim
        self.state = DQNState(global_step=0, episode=0, epsilon=model_cfg.epsilon_start)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

        self.q_net = QNetwork(observation_dim, action_dim, model_cfg.hidden_dim).to(device)
        self.target_net = QNetwork(observation_dim, action_dim, model_cfg.hidden_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=model_cfg.learning_rate)
        self.buffer = ReplayBuffer(model_cfg.replay_capacity, observation_dim)

    def _epsilon_at_step(self, step: int) -> float:
        fraction = min(1.0, step / float(max(1, self.model_cfg.epsilon_decay_steps)))
        return self.model_cfg.epsilon_start + fraction * (
            self.model_cfg.epsilon_end - self.model_cfg.epsilon_start
        )

    def act(self, obs_vec: np.ndarray, eval_mode: bool = False) -> int:
        epsilon = 0.0 if eval_mode else self._epsilon_at_step(self.state.global_step)
        self.state.epsilon = epsilon
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            obs_t = torch.tensor(obs_vec, device=self.device).unsqueeze(0)
            q_values = self.q_net(obs_t)
            return int(torch.argmax(q_values, dim=1).item())

    def optimize(self) -> float | None:
        cfg = self.model_cfg
        if self.buffer.size < cfg.learning_starts or self.buffer.size < cfg.batch_size:
            return None
        if self.state.global_step % cfg.train_frequency != 0:
            return None

        batch = self.buffer.sample(cfg.batch_size)
        obs_t = torch.tensor(batch["obs"], device=self.device)
        next_obs_t = torch.tensor(batch["next_obs"], device=self.device)
        action_t = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
        reward_t = torch.tensor(batch["rewards"], device=self.device)
        done_t = torch.tensor(batch["dones"], device=self.device)

        q_values = self.q_net(obs_t)
        q_selected = q_values.gather(1, action_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_obs_t).max(dim=1).values
            td_target = reward_t + (1.0 - done_t) * cfg.gamma * next_q

        loss = nn.functional.smooth_l1_loss(q_selected, td_target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.state.global_step % cfg.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())

    def save(self, path: Path, extra: dict | None = None) -> None:
        payload = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "state": {
                "global_step": self.state.global_step,
                "episode": self.state.episode,
                "epsilon": self.state.epsilon,
            },
            "model_cfg": asdict(self.model_cfg),
            "extra": extra or {},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load(self, path: Path, map_location: torch.device) -> dict:
        payload = torch.load(path, map_location=map_location)
        self.q_net.load_state_dict(payload["q_net"])
        self.target_net.load_state_dict(payload["target_net"])
        self.optimizer.load_state_dict(payload["optimizer"])
        state = payload["state"]
        self.state = DQNState(
            global_step=int(state["global_step"]),
            episode=int(state["episode"]),
            epsilon=float(state["epsilon"]),
        )
        return payload


def append_metrics(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
