"""
Logging utilities: CSV writer, TensorBoard writer, console summary, and a
running mean/std tracker used for reward normalization.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TENSORBOARD = True
except ImportError:  # pragma: no cover
    _HAS_TENSORBOARD = False


@dataclass
class EpisodeRecord:
    episode: int
    reward: float
    avg_reward: float
    actor_loss: float
    critic_loss: float
    entropy: float
    episode_length: int
    success: int
    collision: int
    timeout: int
    learning_rate: float
    avg_value_estimate: float


@dataclass
class EvalRecord:
    episode: int
    avg_reward: float
    avg_episode_length: float
    success_rate: float
    collision_rate: float
    avg_final_dist_to_goal: float


class RunningNormalizer:
    """Welford-style running mean/std tracker, used to normalize rewards on the fly."""

    def __init__(self, clip: float = 10.0, eps: float = 1e-8) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = eps
        self.clip = clip
        self.eps = eps

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += delta * delta2

    def normalize(self, x: float) -> float:
        std = np.sqrt(self.var / max(self.count, 1.0)) + self.eps
        normed = (x - self.mean) / std
        return float(np.clip(normed, -self.clip, self.clip))

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]


class CSVLogger:
    def __init__(self, path: str, record_type: type) -> None:
        self.path = path
        self.fieldnames = [f.name for f in fields(record_type)]
        is_new = not os.path.exists(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        if is_new:
            self._writer.writeheader()
            self._file.flush()

    def write(self, record) -> None:
        self._writer.writerow({f: getattr(record, f) for f in self.fieldnames})
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class TrainingLogger:
    """Wraps CSV + TensorBoard + console output behind one interface."""

    def __init__(self, logs_dir: str, tensorboard_dir: str, run_name: str = "ppo_nav") -> None:
        os.makedirs(logs_dir, exist_ok=True)
        self.train_csv = CSVLogger(os.path.join(logs_dir, "training.csv"), EpisodeRecord)
        self.eval_csv = CSVLogger(os.path.join(logs_dir, "evaluation.csv"), EvalRecord)

        self.tb_writer: Optional["SummaryWriter"] = None
        if _HAS_TENSORBOARD:
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, run_name))

        self._recent_rewards: list = []
        self._recent_success: list = []
        self._recent_collision: list = []
        self._recent_timeout: list = []
        self._window = 50

    def log_episode(self, record: EpisodeRecord) -> None:
        self.train_csv.write(record)

        self._recent_rewards.append(record.reward)
        self._recent_success.append(record.success)
        self._recent_collision.append(record.collision)
        self._recent_timeout.append(record.timeout)
        for buf in (self._recent_rewards, self._recent_success,
                    self._recent_collision, self._recent_timeout):
            if len(buf) > self._window:
                buf.pop(0)

        success_rate = 100.0 * np.mean(self._recent_success)
        collision_rate = 100.0 * np.mean(self._recent_collision)
        timeout_rate = 100.0 * np.mean(self._recent_timeout)

        print(
            f"\nEpisode {record.episode}\n"
            f"Reward           : {record.reward:.2f}\n"
            f"Average Reward   : {record.avg_reward:.2f}\n"
            f"Success Rate     : {success_rate:.0f}%\n"
            f"Collision Rate   : {collision_rate:.0f}%\n"
            f"Timeout Rate     : {timeout_rate:.0f}%\n"
            f"Actor Loss       : {record.actor_loss:.4f}\n"
            f"Critic Loss      : {record.critic_loss:.4f}\n"
            f"Entropy          : {record.entropy:.4f}\n"
            f"Learning Rate    : {record.learning_rate:.6f}"
        )

        if self.tb_writer is not None:
            step = record.episode
            self.tb_writer.add_scalar("train/reward", record.reward, step)
            self.tb_writer.add_scalar("train/avg_reward", record.avg_reward, step)
            self.tb_writer.add_scalar("train/success_rate", success_rate, step)
            self.tb_writer.add_scalar("train/collision_rate", collision_rate, step)
            self.tb_writer.add_scalar("train/timeout_rate", timeout_rate, step)
            self.tb_writer.add_scalar("loss/actor", record.actor_loss, step)
            self.tb_writer.add_scalar("loss/critic", record.critic_loss, step)
            self.tb_writer.add_scalar("loss/policy_entropy", record.entropy, step)
            self.tb_writer.add_scalar("train/learning_rate", record.learning_rate, step)
            self.tb_writer.add_scalar("train/avg_value_estimate", record.avg_value_estimate, step)
            self.tb_writer.add_scalar("train/episode_length", record.episode_length, step)

    def log_eval(self, record: EvalRecord) -> None:
        self.eval_csv.write(record)
        print(
            f"\n[EVAL @ episode {record.episode}]\n"
            f"Avg Reward       : {record.avg_reward:.2f}\n"
            f"Avg Ep Length    : {record.avg_episode_length:.1f}\n"
            f"Success Rate     : {record.success_rate:.0f}%\n"
            f"Collision Rate   : {record.collision_rate:.0f}%\n"
            f"Avg Final Dist   : {record.avg_final_dist_to_goal:.2f} m"
        )
        if self.tb_writer is not None:
            step = record.episode
            self.tb_writer.add_scalar("eval/avg_reward", record.avg_reward, step)
            self.tb_writer.add_scalar("eval/avg_episode_length", record.avg_episode_length, step)
            self.tb_writer.add_scalar("eval/success_rate", record.success_rate, step)
            self.tb_writer.add_scalar("eval/collision_rate", record.collision_rate, step)
            self.tb_writer.add_scalar("eval/avg_final_dist_to_goal", record.avg_final_dist_to_goal, step)

    def close(self) -> None:
        self.train_csv.close()
        self.eval_csv.close()
        if self.tb_writer is not None:
            self.tb_writer.close()