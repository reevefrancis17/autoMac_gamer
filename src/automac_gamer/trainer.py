from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json
import random
import shutil

import torch
from torch import nn

from automac_gamer.config import ExperimentConfig
from automac_gamer.core.interfaces import Trainer
from automac_gamer.games.tetris.adapter import TetrisAdapter
from automac_gamer.rl.dqn import DQNAgent, append_metrics, flatten_observation


def resolve_device(preference: str) -> torch.device:
    pref = preference.lower()
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_latest_checkpoint(run_dir: Path) -> Path:
    direct_latest = run_dir / "checkpoints" / "latest.pt"
    if direct_latest.exists():
        return direct_latest

    nested_latest = sorted(run_dir.glob("*/checkpoints/latest.pt"), key=lambda p: p.stat().st_mtime)
    if nested_latest:
        return nested_latest[-1]

    step_ckpts = sorted(run_dir.glob("**/checkpoints/step_*.pt"), key=lambda p: p.stat().st_mtime)
    if step_ckpts:
        return step_ckpts[-1]

    raise FileNotFoundError(f"no checkpoint found in {run_dir}")


class TetrisDQNTrainer(Trainer):
    def __init__(self, config: ExperimentConfig, *, training_mode: bool = True) -> None:
        self.config = config
        self.training_mode = training_mode
        self.device = resolve_device(config.device.preference)
        self.run_dir = self._resolve_run_dir(config, training_mode=training_mode)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.jsonl"

        self._train_env = TetrisAdapter(config.env, render_mode="headless", seed=config.run.seed)
        initial_obs = self._train_env.reset(seed=config.run.seed)
        observation_dim = flatten_observation(initial_obs).shape[0]
        self.agent = DQNAgent(
            observation_dim=observation_dim,
            action_dim=self._train_env.action_space_n,
            model_cfg=config.model,
            device=self.device,
            seed=config.run.seed,
        )

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if training_mode:
            self._write_resolved_config()

        if config.run.resume_from:
            self._resume(Path(config.run.resume_from))

    def _resolve_run_dir(self, config: ExperimentConfig, *, training_mode: bool) -> Path:
        base = Path(config.run.run_dir)
        if not training_mode:
            return base
        if config.run.resume_from:
            resume_path = Path(config.run.resume_from)
            if resume_path.name == "latest.pt" and resume_path.parent.name == "checkpoints":
                return resume_path.parent.parent
            if resume_path.name.startswith("step_") and resume_path.parent.name == "checkpoints":
                return resume_path.parent.parent
            return base
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return base / stamp

    def _write_resolved_config(self) -> None:
        payload = {
            "env": asdict(self.config.env),
            "model": asdict(self.config.model),
            "device": asdict(self.config.device),
            "run": asdict(self.config.run),
            "watch": asdict(self.config.watch),
            "resolved_device": str(self.device),
        }
        path = self.run_dir / "resolved_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _resume(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        self.agent.load(checkpoint_path, map_location=self.device)

    def _save_checkpoint(self, step: int, *, reason: str) -> Path:
        checkpoint_path = self.checkpoint_dir / f"step_{step:08d}.pt"
        self.agent.save(
            checkpoint_path,
            extra={"reason": reason, "device": str(self.device), "run_dir": str(self.run_dir)},
        )
        latest_path = self.checkpoint_dir / "latest.pt"
        shutil.copy2(checkpoint_path, latest_path)
        return checkpoint_path

    def _imitation_warmup(self, steps: int = 6000, epochs: int = 3) -> None:
        observations: list[torch.Tensor] = []
        actions: list[int] = []

        obs = self._train_env.reset(seed=self.config.run.seed)
        obs_vec = flatten_observation(obs)

        for step in range(steps):
            guided = self._train_env.guided_action(step)
            if guided is None:
                guided = random.randrange(self._train_env.action_space_n)

            result = self._train_env.step(guided)
            next_obs_vec = flatten_observation(result.observation)

            self.agent.buffer.add(
                obs=obs_vec,
                action=guided,
                reward=result.reward,
                next_obs=next_obs_vec,
                done=result.done,
            )
            observations.append(torch.tensor(obs_vec, dtype=torch.float32))
            actions.append(int(guided))
            obs_vec = next_obs_vec

            if result.done:
                obs = self._train_env.reset(seed=self.config.run.seed + step + 1)
                obs_vec = flatten_observation(obs)

        if not observations:
            return

        x = torch.stack(observations).to(self.device)
        y = torch.tensor(actions, dtype=torch.long, device=self.device)

        batch_size = 256
        self.agent.q_net.train()
        for _ in range(epochs):
            perm = torch.randperm(x.size(0), device=self.device)
            for start in range(0, x.size(0), batch_size):
                idx = perm[start : start + batch_size]
                logits = self.agent.q_net(x[idx])
                loss = nn.functional.cross_entropy(logits, y[idx])
                self.agent.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.q_net.parameters(), max_norm=10.0)
                self.agent.optimizer.step()
        self.agent.target_net.load_state_dict(self.agent.q_net.state_dict())
        self.agent.state.global_step = max(self.agent.state.global_step, steps)
        append_metrics(
            self.metrics_path,
            {
                "kind": "imitation_warmup",
                "global_step": self.agent.state.global_step,
                "samples": int(x.size(0)),
                "epochs": epochs,
            },
        )

    def train(self) -> Path:
        cfg = self.config
        obs = self._train_env.reset(seed=cfg.run.seed)
        obs_vec = flatten_observation(obs)
        episode_reward = 0.0
        episode_lines = 0
        episode_score = 0

        last_loss = None
        try:
            self._imitation_warmup()
            obs = self._train_env.reset(seed=cfg.run.seed + 100_000)
            obs_vec = flatten_observation(obs)
            for _ in range(cfg.run.total_steps):
                guided_action = self._train_env.guided_action(self.agent.state.global_step)
                guidance_prob = max(0.15, 0.8 - (self.agent.state.global_step / 80_000.0))
                if guided_action is not None and random.random() < guidance_prob:
                    action = guided_action
                else:
                    action = self.agent.act(obs_vec, eval_mode=False)
                result = self._train_env.step(action)
                next_obs_vec = flatten_observation(result.observation)

                self.agent.buffer.add(
                    obs=obs_vec,
                    action=action,
                    reward=result.reward,
                    next_obs=next_obs_vec,
                    done=result.done,
                )

                self.agent.state.global_step += 1
                loss = self.agent.optimize()
                if loss is not None:
                    last_loss = loss

                obs_vec = next_obs_vec
                episode_reward += result.reward
                episode_lines = int(result.info["lines"])
                episode_score = int(result.info["score"])

                if result.done:
                    append_metrics(
                        self.metrics_path,
                        {
                            "kind": "episode",
                            "episode": self.agent.state.episode,
                            "global_step": self.agent.state.global_step,
                            "reward": episode_reward,
                            "score": episode_score,
                            "lines": episode_lines,
                            "epsilon": self.agent.state.epsilon,
                        },
                    )
                    self.agent.state.episode += 1
                    episode_reward = 0.0
                    obs = self._train_env.reset(seed=cfg.run.seed + self.agent.state.episode)
                    obs_vec = flatten_observation(obs)

                if self.agent.state.global_step % cfg.run.log_interval == 0:
                    append_metrics(
                        self.metrics_path,
                        {
                            "kind": "train_log",
                            "global_step": self.agent.state.global_step,
                            "episode": self.agent.state.episode,
                            "buffer_size": self.agent.buffer.size,
                            "epsilon": self.agent.state.epsilon,
                            "loss": last_loss,
                            "score": episode_score,
                            "lines": episode_lines,
                        },
                    )

                if self.agent.state.global_step % cfg.run.checkpoint_interval == 0:
                    self._save_checkpoint(self.agent.state.global_step, reason="interval")
        finally:
            self._train_env.close()

        final = self._save_checkpoint(self.agent.state.global_step, reason="final")
        return final

    def evaluate(self, checkpoint: Path | None = None, episodes: int = 5) -> dict[str, float]:
        checkpoint_path = checkpoint or find_latest_checkpoint(self.run_dir)
        self.agent.load(checkpoint_path, map_location=self.device)

        eval_env = TetrisAdapter(self.config.env, render_mode="headless", seed=self.config.run.seed)
        rewards = []
        scores = []
        lines = []

        for episode in range(episodes):
            obs = eval_env.reset(seed=self.config.run.seed + episode)
            obs_vec = flatten_observation(obs)
            done = False
            episode_reward = 0.0
            final_score = 0
            final_lines = 0
            while not done:
                action = self.agent.act(obs_vec, eval_mode=True)
                result = eval_env.step(action)
                obs_vec = flatten_observation(result.observation)
                episode_reward += result.reward
                final_score = int(result.info["score"])
                final_lines = int(result.info["lines"])
                done = result.done
            rewards.append(episode_reward)
            scores.append(final_score)
            lines.append(final_lines)

        eval_env.close()
        self._train_env.close()
        metrics = {
            "episodes": float(episodes),
            "avg_reward": float(sum(rewards) / max(1, len(rewards))),
            "avg_score": float(sum(scores) / max(1, len(scores))),
            "avg_lines": float(sum(lines) / max(1, len(lines))),
        }
        append_metrics(
            self.metrics_path,
            {
                "kind": "evaluation",
                "global_step": self.agent.state.global_step,
                **metrics,
                "checkpoint": str(checkpoint_path),
            },
        )
        return metrics
