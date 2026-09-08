"""
Config for the PyBullet PPO navigation trainer.

Deliberately self-contained (no dependency on the ROS2 package's config.py)
so this whole pipeline can run with just `pip install pybullet torch numpy`.
All the reward/PPO hyperparameters mirror the tuned values from the
ROS/Gazebo version (config.py) since those fixes (goal_dist_norm=12.0,
lr/entropy decay horizons, max_steps_per_episode=600, etc.) are still
correct here — only the sim/robot fields are new.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List


@dataclass
class PyBulletPPOConfig:
    # ---------------------------------------------------------------
    # Robot / URDF — YOU MUST SET robot_urdf_path
    # ---------------------------------------------------------------
    robot_urdf_path: str = ""   # <-- point this at your robot's .urdf file

    # Wheel joint names, one list per side (matches gz-sim-diff-drive-system
    # which also accepts multiple <left_joint>/<right_joint> entries for
    # 4/6-wheel robots). Left blank -> auto-detected by scanning ALL joint
    # names for "left"/"right" + "wheel" (case-insensitive) and grouping
    # every match per side. If your URDF uses different names, set these
    # explicitly — the env prints all joint names at startup if auto-detect
    # fails, so you can copy exact names from there.
    wheel_joints_left: List[str] = field(default_factory=lambda: [
        "front_left_wheel_joint", "rear_left_wheel_joint",
    ])
    wheel_joints_right: List[str] = field(default_factory=lambda: [
        "front_right_wheel_joint", "rear_right_wheel_joint",
    ])

    # From this robot's <gz-sim-diff-drive-system> plugin block directly.
    wheel_radius: float = 0.1
    wheel_separation: float = 0.35

    # Lidar mount offset relative to base_link (from the URDF's lidar_joint:
    # <origin xyz="0.2 0 0.2"/>) — x is forward-offset in the robot's own
    # frame (rotated into world frame using current yaw), z is height above
    # base_link origin.
    lidar_x_offset: float = 0.2
    lidar_z_offset: float = 0.2

    # ---------------------------------------------------------------
    # Environment / state  (unchanged from the ROS version)
    # ---------------------------------------------------------------
    state_dim: int = 38
    action_dim: int = 2
    num_lidar_beams: int = 36
    max_lin: float = 0.22
    max_ang: float = 1.0
    lidar_max_range: float = 3.5
    goal_dist_norm: float = 12.0

    spawn_x: float = -5.0
    spawn_y: float = -3.0
    spawn_z: float = 0.1

    target_goal: List[float] = field(default_factory=lambda: [5.0, -3.0])
    goal_threshold: float = 0.3
    collision_threshold: float = 0.2   # kept for reference; collisions are
                                        # now detected via real contact
                                        # points, not a lidar-min threshold

    max_steps_per_episode: int = 600

    # ---------------------------------------------------------------
    # Corridor obstacles — set False to train in the empty room first,
    # then flip True once the agent reliably reaches the goal.
    # ---------------------------------------------------------------
    enable_obstacles: bool = True
    dynamic_obstacle_enabled: bool = True
    dynamic_obstacle_period_s: float = 8.0   # full oscillation period

    # ---------------------------------------------------------------
    # PPO core hyperparameters (unchanged)
    # ---------------------------------------------------------------
    lr: float = 3e-4
    lr_min: float = 1e-5
    lr_schedule: str = "linear"
    lr_total_steps: int = 300_000

    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 10
    batch_size: int = 64
    update_every: int = 1024
    max_grad_norm: float = 0.5

    value_coef: float = 0.5
    entropy_coef_start: float = 0.01
    entropy_coef_end: float = 0.001
    entropy_coef_decay_steps: int = 150_000

    # ---------------------------------------------------------------
    # Reward shaping (unchanged)
    # ---------------------------------------------------------------
    reward_goal: float = 100.0
    reward_collision: float = -20.0
    reward_step: float = -0.05
    reward_progress_scale: float = 5.0
    reward_heading_scale: float = 0.3
    reward_spin_penalty: float = 0.05
    reward_smoothness_scale: float = 0.02

    normalize_rewards: bool = True
    reward_norm_clip: float = 10.0

    waypoint_shaping_enabled: bool = True
    num_waypoints: int = 4
    waypoint_radius: float = 0.4
    waypoint_bonus: float = 10.0

    # ---------------------------------------------------------------
    # Checkpointing / eval
    # ---------------------------------------------------------------
    checkpoint_every_episodes: int = 10
    keep_last_n_checkpoints: int = 5
    eval_every_episodes: int = 25
    eval_episodes: int = 5
    eval_max_steps: int = 700

    # ---------------------------------------------------------------
    # Sim / rendering
    # ---------------------------------------------------------------
    seed: int = 42
    control_hz: float = 10.0          # policy decision rate
    physics_hz: float = 240.0         # pybullet substep rate
    gui: bool = True
    render_realtime: bool = True      # sleep to ~match wall-clock while GUI is open

    # ---------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------
    workspace_dir: str = os.path.expanduser("~/pybullet_nav/MODEL")

    def __post_init__(self) -> None:
        self.models_dir = os.path.join(self.workspace_dir, "models")
        self.logs_dir = os.path.join(self.workspace_dir, "logs")
        self.tensorboard_dir = os.path.join(self.workspace_dir, "tensorboard")
        self.configs_dir = os.path.join(self.workspace_dir, "configs")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PyBulletPPOConfig":
        with open(path, "r") as f:
            data = json.load(f)
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def create_directories(self) -> None:
        for d in (self.models_dir, self.logs_dir, self.tensorboard_dir, self.configs_dir):
            os.makedirs(d, exist_ok=True)
