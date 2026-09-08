"""
PPO Trainer node for differential-drive autonomous navigation.

This file improves the TRAINING PIPELINE ONLY. The ROS2 architecture
(topics, subscribers, publishers, services) and the network architecture
(PPOActorCritic) are unchanged.

FIX (this revision): calculate_reward() previously used hardcoded
w1/w2/w3 constants and a hardcoded progress multiplier, and never read
config.py's reward_heading_scale / reward_spin_penalty /
reward_smoothness_scale fields at all. That meant there was zero penalty
on spinning in place — a policy holding a small forward velocity with a
large constant angular velocity (driving in a circle) scored almost the
same as one driving straight at the goal, since only net distance
progress was rewarded. This wires up the terms config.py already
declares: a heading-error penalty (facing away from the goal), a direct
angular-velocity penalty, and an action-smoothness penalty, without
changing the meaning or magnitude convention of the goal/collision terms.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import rclpy
import torch
import torch.optim as optim
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import LaserScan

from .checkpoint_manager import CheckpointManager, TrainingState
from .config import PPOConfig
from .logger import EpisodeRecord, EvalRecord, RunningNormalizer, TrainingLogger
from .ppo_model import PPOActorCritic


# =====================================================================
# Rollout buffer
# =====================================================================
class PPOBuffer:
    """On-policy rollout storage. Cleared after every PPO update."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.values: List[float] = []

    def push(self, s, a, lp, r, d, v) -> None:
        self.states.append(s)
        self.actions.append(a)
        self.log_probs.append(lp)
        self.rewards.append(r)
        self.dones.append(d)
        self.values.append(v)

    def size(self) -> int:
        return len(self.states)

    def compute_gae(self, gamma: float, lam: float) -> Tuple[List[float], List[float]]:
        returns: List[float] = []
        advantages: List[float] = []
        gae = 0.0
        next_value = 0.0
        for r, d, v in zip(reversed(self.rewards), reversed(self.dones), reversed(self.values)):
            delta = r + gamma * next_value * (1.0 - d) - v
            gae = delta + gamma * lam * (1.0 - d) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + v)
            next_value = v
        return returns, advantages


# =====================================================================
# Per-episode bookkeeping (for logging only, does not affect training)
# =====================================================================
@dataclass
class EpisodeOutcome:
    success: bool = False
    collision: bool = False
    timeout: bool = False


class PPOTrainerNode(Node):
    def __init__(self, config: Optional[PPOConfig] = None) -> None:
        super().__init__('ppo_trainer')

        self.cfg = config or PPOConfig()
        self.cfg.create_directories()
        self.cfg.save(os.path.join(self.cfg.configs_dir, 'config.json'))

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = PPOActorCritic(self.cfg.state_dim, self.cfg.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
        self.scheduler = self._build_scheduler(self.optimizer)

        self.ckpt_manager = CheckpointManager(
            self.cfg.models_dir, keep_last_n=self.cfg.keep_last_n_checkpoints)
        self.train_state: TrainingState = self.ckpt_manager.try_resume(
            self.policy, self.optimizer, self.scheduler, self.device)

        resumed = self.train_state.episode > 0 or self.train_state.total_steps > 0
        if resumed:
            self.get_logger().info(
                f'Resumed from checkpoint: episode={self.train_state.episode}, '
                f'total_steps={self.train_state.total_steps}, '
                f'best_eval_reward={self.train_state.best_eval_reward:.2f}')
        else:
            self.get_logger().info('No checkpoint found — starting fresh training run.')

        self.logger = TrainingLogger(self.cfg.logs_dir, self.cfg.tensorboard_dir)
        self.reward_normalizer = RunningNormalizer(clip=self.cfg.reward_norm_clip)
        self._recent_rewards_for_avg: List[float] = []

        self.buffer = PPOBuffer()
        self.episode = self.train_state.episode
        self.total_steps = self.train_state.total_steps
        self.best_eval_reward = self.train_state.best_eval_reward

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.reset_client = self.create_client(SetEntityPose, '/world/default/set_pose')
        self.entity_name = getattr(self.cfg, 'entity_name', 'my_robot')
        self.spawn_x = getattr(self.cfg, 'spawn_x', -5.0)
        self.spawn_y = getattr(self.cfg, 'spawn_y', -3.0)
        self.spawn_z = getattr(self.cfg, 'spawn_z', 0.1)

        self.current_scan: Optional[np.ndarray] = None
        self.current_pose = None
        self.prev_dist: Optional[float] = None
        self.step_count = 0
        self.episode_reward = 0.0
        self.is_resetting = False
        self.outcome = EpisodeOutcome()

        self._waypoints: List[Tuple[float, float]] = []
        self._waypoints_hit: set = set()
        self._waypoints_initialized = False

        # Smoothness-shaping state — previous step's angular command, used
        # to penalize jerky frame-to-frame angular changes.
        self._prev_ang: float = 0.0

        self._last_actor_loss = 0.0
        self._last_critic_loss = 0.0
        self._last_entropy = 0.0
        self._last_avg_value = 0.0

        self.eval_mode = False
        self.eval_results: List[dict] = []
        self._pending_eval_episodes = 0

        self.get_logger().info(f'PPO Trainer started on device: {self.device}')
        self.timer = self.create_timer(1.0 / self.cfg.control_hz, self.training_step)

    # =================================================================
    # Scheduler
    # =================================================================
    def _build_scheduler(self, optimizer: optim.Optimizer):
        if self.cfg.lr_schedule == 'none':
            return None
        if self.cfg.lr_schedule == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.cfg.lr_total_steps, eta_min=self.cfg.lr_min)
        def _linear_fn(step: int) -> float:
            progress = min(step / max(self.cfg.lr_total_steps, 1), 1.0)
            floor_ratio = self.cfg.lr_min / self.cfg.lr
            return 1.0 - progress * (1.0 - floor_ratio)
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_linear_fn)

    def _current_entropy_coef(self) -> float:
        progress = min(self.total_steps / max(self.cfg.entropy_coef_decay_steps, 1), 1.0)
        start, end = self.cfg.entropy_coef_start, self.cfg.entropy_coef_end
        return start + progress * (end - start)

    # =================================================================
    # ROS2 callbacks
    # =================================================================
    def scan_cb(self, msg: LaserScan) -> None:
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=msg.range_max, neginf=0.0)

        # Roll the array so index 0 always corresponds to angle 0 (straight
        # ahead) in the robot's own frame, regardless of angle_min.
        n = len(ranges)
        if n > 0 and msg.angle_increment != 0:
            zero_angle_idx = int(round(-msg.angle_min / msg.angle_increment)) % n
            ranges = np.roll(ranges, -zero_angle_idx)

        self.current_scan = ranges[::10][:self.cfg.num_lidar_beams]

    def odom_cb(self, msg: Odometry) -> None:
        self.current_pose = msg.pose.pose

    # =================================================================
    # Geometry / state helpers
    # =================================================================
    def get_yaw(self, q) -> float:
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def get_dist_to_goal(self) -> float:
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        return math.sqrt(
            (self.cfg.target_goal[0] - x) ** 2 + (self.cfg.target_goal[1] - y) ** 2)

    def get_heading_error(self) -> float:
        """Signed angle (radians, [-pi, pi]) between the robot's current
        heading and the direction straight toward the goal. 0 = facing
        the goal exactly; +-pi = facing directly away from it."""
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        yaw = self.get_yaw(self.current_pose.orientation)
        dx = self.cfg.target_goal[0] - x
        dy = self.cfg.target_goal[1] - y
        return math.atan2(
            math.sin(math.atan2(dy, dx) - yaw),
            math.cos(math.atan2(dy, dx) - yaw))

    def get_state(self) -> np.ndarray:
        dist = self.get_dist_to_goal()
        angle = self.get_heading_error()
        return np.concatenate((self.current_scan, [dist, angle])).astype(np.float32)

    def check_collision(self) -> bool:
        return bool(np.min(self.current_scan) < self.cfg.collision_threshold)

    def check_goal(self) -> bool:
        return self.get_dist_to_goal() < self.cfg.goal_threshold

    # =================================================================
    # Reward function (see module docstring for what changed and why)
    # =================================================================
    def calculate_reward(self, collision: bool, goal_reached: bool,
                          prev_dist: float, curr_dist: float,
                          heading_error: float, ang_cmd: float) -> float:
        reward = self.cfg.reward_step
        reward += (prev_dist - curr_dist) * self.cfg.reward_progress_scale

        # Penalize facing away from the goal: 0 at heading_error=0,
        # max at heading_error=+-pi.
        reward += -self.cfg.reward_heading_scale * (abs(heading_error) / math.pi)

        # Direct anti-spin term — this was missing entirely before, which
        # is why circling cost the policy almost nothing.
        norm_ang = abs(ang_cmd) / max(self.cfg.max_ang, 1e-6)
        reward += -self.cfg.reward_spin_penalty * norm_ang

        # Penalize jerky angular command deltas frame-to-frame.
        ang_delta = abs(ang_cmd - self._prev_ang) / max(2.0 * self.cfg.max_ang, 1e-6)
        reward += -self.cfg.reward_smoothness_scale * ang_delta

        if collision:
            reward += self.cfg.reward_collision
        if goal_reached:
            reward += self.cfg.reward_goal
        return reward

    # =================================================================
    # Waypoint shaping bonus — ADDITIVE only
    # =================================================================
    def _init_waypoints(self) -> None:
        if self.current_pose is None:
            return
        sx, sy = self.current_pose.position.x, self.current_pose.position.y
        gx, gy = self.cfg.target_goal[0], self.cfg.target_goal[1]
        n = max(self.cfg.num_waypoints, 1)
        self._waypoints = [
            (sx + (gx - sx) * (i / n), sy + (gy - sy) * (i / n))
            for i in range(1, n + 1)
        ]

    def _waypoint_bonus(self, x: float, y: float) -> float:
        if not self.cfg.waypoint_shaping_enabled or not self._waypoints:
            return 0.0
        bonus = 0.0
        for i, (wx, wy) in enumerate(self._waypoints):
            if i in self._waypoints_hit:
                continue
            if math.sqrt((wx - x) ** 2 + (wy - y) ** 2) < self.cfg.waypoint_radius:
                self._waypoints_hit.add(i)
                bonus += self.cfg.waypoint_bonus
        return bonus

    # =================================================================
    # Gazebo reset (Gazebo Harmonic / ros_gz — teleport via SetEntityPose,
    # non-blocking)
    # =================================================================
    def reset_simulation(self) -> None:
        self.is_resetting = True
        self.cmd_pub.publish(Twist())

        if self.reset_client.service_is_ready():
            req = SetEntityPose.Request()
            req.entity = Entity(name=self.entity_name)
            req.pose.position.x = self.spawn_x
            req.pose.position.y = self.spawn_y
            req.pose.position.z = self.spawn_z
            req.pose.orientation.w = 1.0
            future = self.reset_client.call_async(req)
            future.add_done_callback(self._on_reset_done)
        else:
            self.get_logger().warn(
                '/world/default/set_pose service not available — '
                'episode boundary only, robot not teleported')
            self._on_reset_done(None)

    def _on_reset_done(self, future) -> None:
        self.step_count = 0
        self.episode_reward = 0.0
        self.prev_dist = None
        self.outcome = EpisodeOutcome()
        self._waypoints_hit = set()
        self._prev_ang = 0.0
        self.episode += 1
        self.is_resetting = False

    # =================================================================
    # Action selection: training (stochastic) vs evaluation (deterministic)
    # =================================================================
    def _select_action(self, state_tensor: torch.Tensor):
        with torch.no_grad():
            dist_obj, value = self.policy(state_tensor)
            if self.eval_mode:
                action = torch.clamp(dist_obj.mean, -1.0, 1.0)
                return action, None, None
            raw_action = dist_obj.sample()
            action = torch.clamp(raw_action, -1.0, 1.0)
            log_prob = dist_obj.log_prob(action).sum(-1)
            return action, log_prob, value.squeeze(-1)

    # =================================================================
    # PPO update
    # =================================================================
    def ppo_update(self) -> None:
        if self.buffer.size() < self.cfg.batch_size:
            self.get_logger().warn(f'Buffer too small ({self.buffer.size()}), skipping update')
            return

        returns, advantages = self.buffer.compute_gae(self.cfg.gamma, self.cfg.lam)

        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs)).unsqueeze(1).to(self.device)
        returns_t = torch.FloatTensor(returns).unsqueeze(1).to(self.device)

        advantages_t = torch.FloatTensor(advantages).to(self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        entropy_coef = self._current_entropy_coef()
        dataset_size = self.buffer.size()

        actor_losses, critic_losses, entropies, value_estimates = [], [], [], []

        for _ in range(self.cfg.epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.cfg.batch_size):
                idx = indices[start:start + self.cfg.batch_size]
                s, a = states[idx], actions[idx]
                olp, ret, adv = old_log_probs[idx], returns_t[idx], advantages_t[idx]

                log_probs, values, entropy = self.policy.evaluate(s, a)
                ratio = torch.exp(log_probs - olp)

                surr1 = ratio * adv.unsqueeze(1)
                surr2 = torch.clamp(
                    ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv.unsqueeze(1)

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.cfg.value_coef * (ret - values).pow(2).mean()
                entropy_loss = -entropy_coef * entropy.mean()
                loss = actor_loss + critic_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())
                value_estimates.append(values.mean().item())

        if self.scheduler is not None:
            self.scheduler.step()

        self._last_actor_loss = float(np.mean(actor_losses))
        self._last_critic_loss = float(np.mean(critic_losses))
        self._last_entropy = float(np.mean(entropies))
        self._last_avg_value = float(np.mean(value_estimates))

        self.get_logger().info(
            f'PPO update | episode={self.episode} | steps={self.total_steps} '
            f'| actor_loss={self._last_actor_loss:.4f} '
            f'| critic_loss={self._last_critic_loss:.4f} '
            f'| entropy={self._last_entropy:.4f}')

        self.buffer.clear()

    # =================================================================
    # Checkpointing
    # =================================================================
    def _current_train_state(self) -> TrainingState:
        return TrainingState(
            episode=self.episode,
            total_steps=self.total_steps,
            best_eval_reward=self.best_eval_reward,
        )

    def _save_latest(self) -> None:
        self.ckpt_manager.save_latest(
            self.policy, self.optimizer, self.scheduler,
            self._current_train_state(), self.cfg.to_dict())

    def _save_rolling(self) -> None:
        self.ckpt_manager.save_rolling(
            self.policy, self.optimizer, self.scheduler,
            self._current_train_state(), self.cfg.to_dict())

    def _save_best(self) -> None:
        self.ckpt_manager.save_best(
            self.policy, self.optimizer, self.scheduler,
            self._current_train_state(), self.cfg.to_dict())
        self.get_logger().info(
            f'New best model saved (eval reward={self.best_eval_reward:.2f})')

    # =================================================================
    # Evaluation mode
    # =================================================================
    def _start_evaluation(self) -> None:
        self.eval_mode = True
        self.eval_results = []
        self._pending_eval_episodes = self.cfg.eval_episodes
        self.get_logger().info(f'Starting evaluation ({self.cfg.eval_episodes} episodes)...')
        self.step_count = 0
        self.episode_reward = 0.0
        self.prev_dist = None
        self.outcome = EpisodeOutcome()
        self._prev_ang = 0.0

    def _finish_eval_episode(self, final_dist: float) -> None:
        self.eval_results.append({
            'reward': self.episode_reward,
            'length': self.step_count,
            'success': self.outcome.success,
            'collision': self.outcome.collision,
            'final_dist': final_dist,
        })
        self._pending_eval_episodes -= 1

        self.step_count = 0
        self.episode_reward = 0.0
        self.prev_dist = None
        self.outcome = EpisodeOutcome()
        self._prev_ang = 0.0

        if self._pending_eval_episodes <= 0:
            self._end_evaluation()

    def _end_evaluation(self) -> None:
        rewards = [r['reward'] for r in self.eval_results]
        lengths = [r['length'] for r in self.eval_results]
        successes = [r['success'] for r in self.eval_results]
        collisions = [r['collision'] for r in self.eval_results]
        final_dists = [r['final_dist'] for r in self.eval_results]

        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        record = EvalRecord(
            episode=self.episode,
            avg_reward=avg_reward,
            avg_episode_length=float(np.mean(lengths)) if lengths else 0.0,
            success_rate=100.0 * float(np.mean(successes)) if successes else 0.0,
            collision_rate=100.0 * float(np.mean(collisions)) if collisions else 0.0,
            avg_final_dist_to_goal=float(np.mean(final_dists)) if final_dists else 0.0,
        )
        self.logger.log_eval(record)

        if avg_reward > self.best_eval_reward:
            self.best_eval_reward = avg_reward
            self._save_best()

        self.eval_mode = False
        self.eval_results = []
        self.reset_simulation()

    # =================================================================
    # Main control loop
    # =================================================================
    def training_step(self) -> None:
        if self.current_scan is None or self.current_pose is None or self.is_resetting:
            return

        if self.prev_dist is None:
            self.prev_dist = self.get_dist_to_goal()
            if not self._waypoints_initialized:
                self._init_waypoints()
                self._waypoints_initialized = True
            return

        state = self.get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        action, log_prob, value = self._select_action(state_tensor)

        lin = float(action[0][0]) * self.cfg.max_lin
        ang = float(action[0][1]) * self.cfg.max_ang
        twist = Twist()
        twist.linear.x = lin
        twist.angular.z = ang
        self.cmd_pub.publish(twist)

        prev_dist = self.prev_dist
        curr_dist = self.get_dist_to_goal()
        self.prev_dist = curr_dist

        heading_error = self.get_heading_error()
        collision = self.check_collision()
        goal_reached = self.check_goal()
        reward = self.calculate_reward(
            collision, goal_reached, prev_dist, curr_dist, heading_error, ang)
        self._prev_ang = ang

        if not self.eval_mode:
            x = self.current_pose.position.x
            y = self.current_pose.position.y
            reward += self._waypoint_bonus(x, y)

        max_steps = self.cfg.eval_max_steps if self.eval_mode else self.cfg.max_steps_per_episode
        timeout = self.step_count >= max_steps
        done = collision or goal_reached or timeout

        self.outcome.success = self.outcome.success or goal_reached
        self.outcome.collision = self.outcome.collision or collision
        self.outcome.timeout = self.outcome.timeout or (timeout and not collision and not goal_reached)

        if self.eval_mode:
            self.episode_reward += reward
            self.step_count += 1
            if done:
                self._finish_eval_episode(final_dist=curr_dist)
            return

        stored_reward = (
            self.reward_normalizer.normalize(reward)
            if self.cfg.normalize_rewards else reward
        )
        if self.cfg.normalize_rewards:
            self.reward_normalizer.update(reward)

        self.buffer.push(
            state,
            action.cpu().numpy()[0],
            log_prob.cpu().item(),
            stored_reward,
            float(done),
            value.cpu().item(),
        )

        self.episode_reward += reward
        self.step_count += 1
        self.total_steps += 1

        if done:
            self._recent_rewards_for_avg.append(self.episode_reward)
            if len(self._recent_rewards_for_avg) > 50:
                self._recent_rewards_for_avg.pop(0)
            avg_reward = float(np.mean(self._recent_rewards_for_avg))

            current_lr = self.optimizer.param_groups[0]['lr']
            record = EpisodeRecord(
                episode=self.episode,
                reward=self.episode_reward,
                avg_reward=avg_reward,
                actor_loss=self._last_actor_loss,
                critic_loss=self._last_critic_loss,
                entropy=self._last_entropy,
                episode_length=self.step_count,
                success=int(self.outcome.success),
                collision=int(self.outcome.collision),
                timeout=int(self.outcome.timeout),
                learning_rate=current_lr,
                avg_value_estimate=self._last_avg_value,
            )
            self.logger.log_episode(record)

            self.get_logger().info(
                f'Episode {self.episode} | Steps: {self.step_count} | '
                f'Reward: {self.episode_reward:.2f} | '
                f'Collision: {self.outcome.collision} | Goal: {self.outcome.success}')

            self.reset_simulation()

        if self.total_steps % self.cfg.update_every == 0 and self.total_steps > 0:
            self.ppo_update()
            self._save_latest()

        if self.episode > 0 and self.episode % self.cfg.checkpoint_every_episodes == 0 and done:
            self._save_rolling()

        if (self.episode > 0 and self.episode % self.cfg.eval_every_episodes == 0
                and done and not self.eval_mode):
            self._start_evaluation()

    def destroy_node(self) -> bool:
        self._save_latest()
        self.logger.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PPOTrainerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()