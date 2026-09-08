"""
Checkpoint system for crash-safe, resumable PPO training.

Saves three kinds of checkpoints under <models_dir>:
  - latest.pth        overwritten every checkpoint_every_episodes
  - best_model.pth     overwritten only when eval reward improves
  - ckpt_ep<N>.pth      rolling history, capped at keep_last_n_checkpoints

A checkpoint captures everything needed to resume bit-for-bit-ish:
policy weights, optimizer state, scheduler state, episode/step counters,
best reward so far, RNG states, and the hyperparameters used to produce it.
"""

from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass
class TrainingState:
    """Everything about training progress that must survive a restart."""
    episode: int = 0
    total_steps: int = 0
    best_eval_reward: float = float("-inf")


class CheckpointManager:
    def __init__(self, models_dir: str, keep_last_n: int = 5) -> None:
        self.models_dir = models_dir
        self.keep_last_n = keep_last_n
        os.makedirs(models_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------
    def _build_payload(
        self,
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        state: TrainingState,
        config_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "episode": state.episode,
            "total_steps": state.total_steps,
            "best_eval_reward": state.best_eval_reward,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "config": config_dict,
        }

    def save_latest(
        self,
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        state: TrainingState,
        config_dict: Dict[str, Any],
    ) -> str:
        payload = self._build_payload(policy, optimizer, scheduler, state, config_dict)
        path = os.path.join(self.models_dir, "latest.pth")
        torch.save(payload, path)
        return path

    def save_best(
        self,
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        state: TrainingState,
        config_dict: Dict[str, Any],
    ) -> str:
        payload = self._build_payload(policy, optimizer, scheduler, state, config_dict)
        path = os.path.join(self.models_dir, "best_model.pth")
        torch.save(payload, path)
        return path

    def save_rolling(
        self,
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        state: TrainingState,
        config_dict: Dict[str, Any],
    ) -> str:
        payload = self._build_payload(policy, optimizer, scheduler, state, config_dict)
        path = os.path.join(self.models_dir, f"ckpt_ep{state.episode}.pth")
        torch.save(payload, path)
        self._prune_old_checkpoints()
        return path

    def _prune_old_checkpoints(self) -> None:
        pattern = os.path.join(self.models_dir, "ckpt_ep*.pth")
        ckpts = sorted(
            glob.glob(pattern),
            key=lambda p: int(p.split("ckpt_ep")[-1].split(".pth")[0]),
        )
        excess = len(ckpts) - self.keep_last_n
        for old_ckpt in ckpts[:max(excess, 0)]:
            os.remove(old_ckpt)

    # ------------------------------------------------------------
    # Loading / resume
    # ------------------------------------------------------------
    def find_latest_checkpoint(self) -> Optional[str]:
        path = os.path.join(self.models_dir, "latest.pth")
        return path if os.path.exists(path) else None

    def load(
        self,
        path: str,
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        device: torch.device,
        strict: bool = True,
    ) -> TrainingState:
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        policy.load_state_dict(checkpoint["policy_state_dict"], strict=strict)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        rng = checkpoint.get("rng_state")
        if rng is not None:
            torch.set_rng_state(rng["torch"])
            if rng.get("torch_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])

        return TrainingState(
            episode=checkpoint.get("episode", 0),
            total_steps=checkpoint.get("total_steps", 0),
            best_eval_reward=checkpoint.get("best_eval_reward", float("-inf")),
        )

    def try_resume(
        self,
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        device: torch.device,
    ) -> TrainingState:
        """Auto-detects latest.pth and resumes if present, else returns a fresh state."""
        ckpt_path = self.find_latest_checkpoint()
        if ckpt_path is None:
            return TrainingState()
        return self.load(ckpt_path, policy, optimizer, scheduler, device)