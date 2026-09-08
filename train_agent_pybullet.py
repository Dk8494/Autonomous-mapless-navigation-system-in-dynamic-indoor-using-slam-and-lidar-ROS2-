"""
PPO training entry point for the PyBullet navigation environment.

Run with:
    python train_agent_pybullet.py              # GUI, real-time (for watching)
    python train_agent_pybullet.py --fast        # GUI, but sim runs as fast as possible
    python train_agent_pybullet.py --headless    # no window at all — fastest, use this
                                                  # for the actual multi-hour training run

Set cfg.robot_urdf_path (and wheel_joints_left/right if auto-detect fails)
in pybullet_config.py, or edit `build_config()` below, before running.

This mirrors the training logic from the fixed ROS train_agent.py:
  - state/reward normalization
  - GAE bootstrapped with the critic's own value estimate on truncated
    (non-terminal) buffer flushes, not a hardcoded 0.0
  - dense reward normalized on its own running stats; sparse
    goal/collision/waypoint bonuses added back unnormalized
  - evaluation episodes are genuinely independent resets (trivial here
    since reset() is synchronous — no eval-continues-from-crash-site bug
    was even possible to introduce in this version)
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Normal

from checkpoint_manager import CheckpointManager, TrainingState
from logger import EpisodeRecord, EvalRecord, RunningNormalizer, TrainingLogger
from ppo_model import PPOActorCritic
from pybullet_config import PyBulletPPOConfig
from pybullet_nav_env import PyBulletNavEnv


class PPOBuffer:
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

    def compute_gae(self, gamma: float, lam: float,
                     last_value: float = 0.0) -> Tuple[List[float], List[float]]:
        returns: List[float] = []
        advantages: List[float] = []
        gae = 0.0
        next_value = last_value
        for r, d, v in zip(reversed(self.rewards), reversed(self.dones), reversed(self.values)):
            delta = r + gamma * next_value * (1.0 - d) - v
            gae = delta + gamma * lam * (1.0 - d) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + v)
            next_value = v
        return returns, advantages


def build_config() -> PyBulletPPOConfig:
    cfg = PyBulletPPOConfig()
    # Points at robot.urdf shipped alongside this script. Move/rename the
    # URDF and update this path if you relocate it.
    cfg.robot_urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot.urdf")
    # wheel_joints_left/right, wheel_radius, wheel_separation, and
    # lidar_x_offset/lidar_z_offset in pybullet_config.py already default
    # to the values from your uploaded URDF (4-wheel diff-drive, front+rear
    # pairs, wheel_radius=0.1, wheel_separation=0.35, lidar at +0.2/+0.2).
    # Only override below if you swap in a different robot.
    # cfg.wheel_joints_left = ["your_left_joint_1", "your_left_joint_2"]
    # cfg.wheel_joints_right = ["your_right_joint_1", "your_right_joint_2"]
    return cfg


class Trainer:
    def __init__(self, cfg: PyBulletPPOConfig):
        self.cfg = cfg
        cfg.create_directories()
        cfg.save(os.path.join(cfg.configs_dir, "config.json"))

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            # Apple Silicon (M1/M2/M3/M4) GPU backend. Falls back to CPU
            # automatically if unavailable — this network is small enough
            # that CPU is also perfectly workable, MPS just gets you a bit
            # more headroom.
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.policy = PPOActorCritic(cfg.state_dim, cfg.action_dim).to(self.device)
        print(f"Training on device: {self.device}")
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.scheduler = self._build_scheduler(self.optimizer)

        self.ckpt_manager = CheckpointManager(cfg.models_dir, keep_last_n=cfg.keep_last_n_checkpoints)
        self.train_state: TrainingState = self.ckpt_manager.try_resume(
            self.policy, self.optimizer, self.scheduler, self.device)

        self.logger = TrainingLogger(cfg.logs_dir, cfg.tensorboard_dir)
        self.reward_normalizer = RunningNormalizer(clip=cfg.reward_norm_clip)
        self._recent_rewards: List[float] = []

        self.buffer = PPOBuffer()
        self.episode = self.train_state.episode
        self.total_steps = self.train_state.total_steps
        self.best_eval_reward = self.train_state.best_eval_reward

        self.env = PyBulletNavEnv(cfg)

    def _build_scheduler(self, optimizer):
        cfg = self.cfg
        if cfg.lr_schedule == "none":
            return None

        def lr_lambda(step):
            frac = min(step / max(cfg.lr_total_steps, 1), 1.0)
            if cfg.lr_schedule == "cosine":
                scale = 0.5 * (1 + np.cos(np.pi * frac))
            else:  # linear
                scale = 1.0 - frac
            floor = cfg.lr_min / cfg.lr
            return floor + (1.0 - floor) * scale

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _entropy_coef(self) -> float:
        cfg = self.cfg
        frac = min(self.total_steps / max(cfg.entropy_coef_decay_steps, 1), 1.0)
        return cfg.entropy_coef_start + frac * (cfg.entropy_coef_end - cfg.entropy_coef_start)

    def _select_action(self, state: np.ndarray, deterministic: bool = False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, value = self.policy(state_t)
            action = dist.mean if deterministic else dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        return action.cpu().numpy()[0], log_prob, value.squeeze(-1)

    def ppo_update(self, last_value: float = 0.0) -> None:
        cfg = self.cfg
        if self.buffer.size() < cfg.batch_size:
            return

        returns, advantages = self.buffer.compute_gae(cfg.gamma, cfg.lam, last_value=last_value)

        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n = states.size(0)
        entropy_coef = self._entropy_coef()
        actor_losses, critic_losses, entropies = [], [], []

        for _ in range(cfg.epochs):
            idx = torch.randperm(n)
            for start in range(0, n, cfg.batch_size):
                b = idx[start:start + cfg.batch_size]
                log_probs, values, entropy = self.policy.evaluate(states[b], actions[b])
                log_probs = log_probs.squeeze(-1)
                values = values.squeeze(-1)
                entropy = entropy.squeeze(-1)

                ratio = torch.exp(log_probs - old_log_probs[b])
                surr1 = ratio * advantages_t[b]
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages_t[b]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = torch.nn.functional.mse_loss(values, returns_t[b])
                loss = actor_loss + cfg.value_coef * critic_loss - entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())

        if self.scheduler is not None:
            self.scheduler.step()

        self._last_actor_loss = float(np.mean(actor_losses))
        self._last_critic_loss = float(np.mean(critic_losses))
        self._last_entropy = float(np.mean(entropies))
        self._last_avg_value = float(torch.stack(
            [torch.tensor(v) for v in self.buffer.values]).mean().item())

        self.buffer.clear()

    def run_episode(self, eval_mode: bool = False) -> dict:
        cfg = self.cfg
        state = self.env.reset()
        episode_reward = 0.0
        outcome = {"success": False, "collision": False, "timeout": False}
        max_steps = cfg.eval_max_steps if eval_mode else cfg.max_steps_per_episode

        last_value = 0.0
        for _ in range(max_steps):
            action, log_prob, value = self._select_action(state, deterministic=eval_mode)
            next_state, dense_reward, sparse_bonus, _, done, info = self.env.step(action)
            reward = dense_reward + sparse_bonus

            if not eval_mode:
                if cfg.normalize_rewards:
                    stored_reward = self.reward_normalizer.normalize(dense_reward) + sparse_bonus
                    self.reward_normalizer.update(dense_reward)
                else:
                    stored_reward = reward

                self.buffer.push(state, action, log_prob.cpu().item(), stored_reward,
                                  float(done), value.cpu().item())
                self.total_steps += 1

            episode_reward += reward
            outcome["success"] = outcome["success"] or info["goal_reached"]
            outcome["collision"] = outcome["collision"] or info["collision"]
            outcome["timeout"] = outcome["timeout"] or info["timeout"]
            state = next_state
            last_value = value.cpu().item()

            if not eval_mode and self.total_steps % cfg.update_every == 0:
                lv = 0.0 if done else last_value
                self.ppo_update(last_value=lv)
                self._save_latest()

            if done:
                break

        return {
            "reward": episode_reward,
            "length": self.env.step_count,
            "final_dist": info["dist_to_goal"],
            **outcome,
        }

    def evaluate(self) -> None:
        cfg = self.cfg
        results = [self.run_episode(eval_mode=True) for _ in range(cfg.eval_episodes)]
        avg_reward = float(np.mean([r["reward"] for r in results]))
        record = EvalRecord(
            episode=self.episode,
            avg_reward=avg_reward,
            avg_episode_length=float(np.mean([r["length"] for r in results])),
            success_rate=100.0 * float(np.mean([r["success"] for r in results])),
            collision_rate=100.0 * float(np.mean([r["collision"] for r in results])),
            avg_final_dist_to_goal=float(np.mean([r["final_dist"] for r in results])),
        )
        self.logger.log_eval(record)
        if avg_reward > self.best_eval_reward:
            self.best_eval_reward = avg_reward
            self._save_best()

    def _save_latest(self) -> None:
        state = TrainingState(self.episode, self.total_steps, self.best_eval_reward)
        self.ckpt_manager.save_latest(self.policy, self.optimizer, self.scheduler,
                                       state, self.cfg.to_dict())

    def _save_best(self) -> None:
        state = TrainingState(self.episode, self.total_steps, self.best_eval_reward)
        self.ckpt_manager.save_best(self.policy, self.optimizer, self.scheduler,
                                     state, self.cfg.to_dict())

    def _save_rolling(self) -> None:
        state = TrainingState(self.episode, self.total_steps, self.best_eval_reward)
        self.ckpt_manager.save_rolling(self.policy, self.optimizer, self.scheduler,
                                        state, self.cfg.to_dict())

    def train(self, num_episodes: int = 100_000) -> None:
        cfg = self.cfg
        for _ in range(num_episodes):
            self.episode += 1
            result = self.run_episode(eval_mode=False)

            self._recent_rewards.append(result["reward"])
            if len(self._recent_rewards) > 50:
                self._recent_rewards.pop(0)

            record = EpisodeRecord(
                episode=self.episode,
                reward=result["reward"],
                avg_reward=float(np.mean(self._recent_rewards)),
                actor_loss=getattr(self, "_last_actor_loss", 0.0),
                critic_loss=getattr(self, "_last_critic_loss", 0.0),
                entropy=getattr(self, "_last_entropy", 0.0),
                episode_length=result["length"],
                success=int(result["success"]),
                collision=int(result["collision"]),
                timeout=int(result["timeout"]),
                learning_rate=self.optimizer.param_groups[0]["lr"],
                avg_value_estimate=getattr(self, "_last_avg_value", 0.0),
            )
            self.logger.log_episode(record)

            print(f"Episode {self.episode} | Steps: {result['length']} | "
                  f"Reward: {result['reward']:.2f} | "
                  f"Collision: {result['collision']} | Goal: {result['success']}")

            if self.episode % cfg.checkpoint_every_episodes == 0:
                self._save_rolling()
                self._save_latest()

            if self.episode % cfg.eval_every_episodes == 0:
                self.evaluate()

        self.env.close()
        self.logger.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true",
                         help="No GUI window at all — fastest option, use for real training runs.")
    parser.add_argument("--fast", action="store_true",
                         help="Keep the GUI window open but don't throttle to real-time.")
    args = parser.parse_args()

    cfg = build_config()
    if args.headless:
        cfg.gui = False
        cfg.render_realtime = False
    elif args.fast:
        cfg.render_realtime = False

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
