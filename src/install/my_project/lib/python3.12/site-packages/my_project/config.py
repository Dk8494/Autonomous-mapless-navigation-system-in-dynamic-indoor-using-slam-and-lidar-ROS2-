"""
Central configuration for the PPO navigation trainer.

All hyperparameters, paths, and toggles live here so the rest of the
codebase never hardcodes a magic number. Import PPOConfig and either use
the defaults or override fields at construction time.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List


@dataclass
class PPOConfig:
    # ---------------------------------------------------------------
    # Environment / state
    # ---------------------------------------------------------------
    state_dim: int = 38                  # 36 LiDAR beams + dist + angle
    action_dim: int = 2                  # [linear_vel, angular_vel]
    num_lidar_beams: int = 36
    max_lin: float = 0.22                # m/s, TurtleBot3 Burger max
    max_ang: float = 1.0                 # rad/s
    lidar_max_range: float = 3.5         # used for normalization
    goal_dist_norm: float = 8.0          # used for normalization

    target_goal: List[float] = field(default_factory=lambda: [5.0, -3.0])
    goal_threshold: float = 0.3
    collision_threshold: float = 0.2

    # FIX: spawn (-5.0, -3.0) -> goal (5.0, -3.0) is a straight-line
    # distance of 10.0 m. At max_lin=0.22 m/s, even a perfect policy
    # driving dead straight the whole way needs ~45.5s (~455 steps at
    # control_hz=10). The old max_steps_per_episode=200 (20s) made the
    # goal geometrically unreachable regardless of policy quality, which
    # is why success rate was stuck at 0% even as reward improved. 600
    # steps (~60s) gives ~1.3x the pure-travel-time minimum as margin
    # for turning/correction. Lower again later only if you shrink the
    # spawn-goal distance instead.
    max_steps_per_episode: int = 600

    # ---------------------------------------------------------------
    # PPO core hyperparameters
    # ---------------------------------------------------------------
    lr: float = 3e-4
    lr_min: float = 1e-5
    lr_schedule: str = "linear"          # "linear" | "cosine" | "none"
    # FIX: 2,000,000 was tuned for a much longer/harder task. On a simple
    # empty-room single-goal environment, decaying LR over a horizon this
    # long means the schedule barely moves for thousands of episodes.
    # Shortened to match the actual scale of this task.
    lr_total_steps: int = 300_000        # horizon used for schedule decay

    gamma: float = 0.99
    lam: float = 0.95                    # GAE lambda
    clip_eps: float = 0.2
    epochs: int = 10                     # PPO epochs per update
    batch_size: int = 64
    update_every: int = 1024             # env steps collected per update
    max_grad_norm: float = 0.5

    value_coef: float = 0.5
    entropy_coef_start: float = 0.01
    entropy_coef_end: float = 0.001
    # FIX: same issue as lr_total_steps — 1,000,000 meant entropy barely
    # decayed at all across the first several hundred episodes on a task
    # this simple. Shortened so exploration actually tightens on a
    # timescale matching the environment.
    entropy_coef_decay_steps: int = 150_000

    # ---------------------------------------------------------------
    # Reward shaping
    # ---------------------------------------------------------------
    reward_goal: float = 100.0
    reward_collision: float = -20.0
    reward_step: float = -0.05
    reward_progress_scale: float = 5.0
    reward_heading_scale: float = 0.3    # penalize facing away from goal
    reward_spin_penalty: float = 0.05    # penalize high |angular_vel|
    reward_smoothness_scale: float = 0.02  # penalize jerky action deltas

    normalize_rewards: bool = True
    reward_norm_clip: float = 10.0

    # Waypoint shaping (additive only — does not alter calculate_reward).
    waypoint_shaping_enabled: bool = True
    num_waypoints: int = 4          # checkpoints between spawn and goal
    waypoint_radius: float = 0.4    # meters, how close counts as "reached"
    waypoint_bonus: float = 10.0    # one-time bonus per checkpoint per episode

    # ---------------------------------------------------------------
    # Checkpointing
    # ---------------------------------------------------------------
    checkpoint_every_episodes: int = 10
    keep_last_n_checkpoints: int = 5

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------
    eval_every_episodes: int = 25
    eval_episodes: int = 5
    # FIX: same 10m-distance math applies to eval episodes — 500 steps
    # (50s) was still short of the ~455-step pure-travel-time minimum,
    # with zero margin for turning. Raised past max_steps_per_episode
    # since eval runs deterministically and should get at least as much
    # room to succeed as training does.
    eval_max_steps: int = 700

    # ---------------------------------------------------------------
    # Misc / reproducibility
    # ---------------------------------------------------------------
    seed: int = 42
    control_hz: float = 10.0             # timer frequency

    # ---------------------------------------------------------------
    # Paths (created automatically by DirectoryManager)
    # ---------------------------------------------------------------
    workspace_dir: str = os.path.expanduser("my_project/MODEL")

    def __post_init__(self) -> None:
        self.models_dir = os.path.join(self.workspace_dir, "models")
        self.logs_dir = os.path.join(self.workspace_dir, "logs")
        self.tensorboard_dir = os.path.join(self.workspace_dir, "tensorboard")
        self.videos_dir = os.path.join(self.workspace_dir, "videos")
        self.configs_dir = os.path.join(self.workspace_dir, "configs")

    # ------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PPOConfig":
        with open(path, "r") as f:
            data = json.load(f)
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def create_directories(self) -> None:
        for d in (self.models_dir, self.logs_dir, self.tensorboard_dir,
                  self.videos_dir, self.configs_dir):
            os.makedirs(d, exist_ok=True)