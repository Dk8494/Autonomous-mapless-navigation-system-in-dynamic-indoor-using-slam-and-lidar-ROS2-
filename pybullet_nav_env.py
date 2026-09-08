"""
PyBullet navigation environment.

Recreates the geometry from dynamic_corridor.world (12x8m room, two inner
walls, three static obstacles, one moving obstacle) and exposes a simple
synchronous reset()/step() interface. State/reward semantics deliberately
mirror the fixed ROS/Gazebo train_agent.py:
  - state = 36 normalized lidar beams + normalized [dist, angle]  (38-dim)
  - dense_reward / terminal_bonus kept separate (see reward normalization
    note in train_agent_pybullet.py) — same split as the ROS fix
  - collision is detected via real PyBullet contact points, not a
    lidar-min threshold, which sidesteps the NaN/zero-reading ambiguity
    that caused false collisions in the ROS version entirely.

Because this is synchronous, reset() can teleport position AND zero
velocity AND hand back a fully fresh state in one call — none of the
async-reset settle-window/race-condition workarounds needed for Gazebo
apply here.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data

from pybullet_config import PyBulletPPOConfig


class PyBulletNavEnv:
    def __init__(self, cfg: PyBulletPPOConfig):
        self.cfg = cfg
        self._client = p.connect(p.GUI if cfg.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / cfg.physics_hz)

        self._substeps_per_control = max(int(cfg.physics_hz / cfg.control_hz), 1)

        self.plane_id = p.loadURDF("plane.urdf")
        self._static_ids: List[int] = [self.plane_id]
        self._build_corridor()

        if not cfg.robot_urdf_path:
            raise ValueError(
                "cfg.robot_urdf_path is empty — set it to your robot's .urdf "
                "file before creating PyBulletNavEnv."
            )
        self.robot_id = p.loadURDF(
            cfg.robot_urdf_path,
            basePosition=[cfg.spawn_x, cfg.spawn_y, cfg.spawn_z],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        )
        self._left_wheels, self._right_wheels = self._resolve_wheel_joints()

        self._t = 0.0
        self._dyn_obstacle_id: Optional[int] = None
        self._dyn_obstacle_base_pos: Optional[List[float]] = None
        if cfg.enable_obstacles and cfg.dynamic_obstacle_enabled:
            # mass=0 (kinematic): motion is fully scripted via
            # resetBasePositionAndOrientation in _update_dynamic_obstacle,
            # not physics-driven. A non-zero mass would let gravity pull it
            # down between our per-tick resets, fighting the script.
            self._dyn_obstacle_id = self._make_box(
                half_extents=[0.3, 0.3, 0.5], pos=[0.0, -2.5, 0.5],
                rgba=[1, 1, 0, 1], static=True,
            )
            self._dyn_obstacle_base_pos = [0.0, -2.5, 0.5]

        self._waypoints: List[Tuple[float, float]] = []
        self._waypoints_hit: set = set()
        self._prev_ang = 0.0
        self.step_count = 0
        self.prev_dist: Optional[float] = None

    # =================================================================
    # World construction — geometry copied from dynamic_corridor.world
    # =================================================================
    def _make_box(self, half_extents, pos, rgba, static: bool, yaw: float = 0.0):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
        mass = 0.0 if static else 1.0
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
        )
        if static:
            self._static_ids.append(body_id)
        return body_id

    def _make_cylinder(self, radius, height, pos, rgba, static: bool = True):
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=rgba)
        body_id = p.createMultiBody(
            baseMass=0.0, baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis, basePosition=pos,
        )
        if static:
            self._static_ids.append(body_id)
        return body_id

    def _build_corridor(self) -> None:
        # Outer walls — pose/size copied directly from dynamic_corridor.world
        self._make_box([6, 0.1, 0.5], [0, 4, 0.5], [0.7, 0.7, 0.7, 1], static=True)     # wall_north
        self._make_box([6, 0.1, 0.5], [0, -4, 0.5], [0.7, 0.7, 0.7, 1], static=True)    # wall_south
        self._make_box([0.1, 4, 0.5], [6, 0, 0.5], [0.7, 0.7, 0.7, 1], static=True)     # wall_east
        self._make_box([0.1, 4, 0.5], [-6, 0, 0.5], [0.7, 0.7, 0.7, 1], static=True)    # wall_west

        if self.cfg.enable_obstacles:
            # Inner walls (form a chicane)
            self._make_box([3, 0.1, 0.5], [-1, 1, 0.5], [0.6, 0.6, 0.6, 1], static=True)
            self._make_box([3, 0.1, 0.5], [2, -1.5, 0.5], [0.6, 0.6, 0.6, 1], static=True)

            # Static obstacles
            self._make_box([0.5, 0.5, 0.5], [3, 0, 0.5], [1, 0, 0, 1], static=True)
            self._make_cylinder(0.4, 1.0, [-3, -2, 0.5], [0, 0, 1, 1], static=True)
            self._make_box([0.4, 0.4, 0.5], [4.5, 2.5, 0.5], [1, 0.5, 0, 1],
                            static=True, yaw=0.5)

        # Goal marker — visual only, no collision, matches the Gazebo world
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=0.05,
                                   rgbaColor=[0, 1, 0, 1])
        p.createMultiBody(
            baseMass=0.0, baseVisualShapeIndex=vis,
            basePosition=[self.cfg.target_goal[0], self.cfg.target_goal[1], 0.05],
        )

    # =================================================================
    # Wheel joint resolution
    # =================================================================
    def _resolve_wheel_joints(self) -> Tuple[List[int], List[int]]:
        n = p.getNumJoints(self.robot_id)
        joint_names = {}
        for i in range(n):
            info = p.getJointInfo(self.robot_id, i)
            joint_names[info[1].decode("utf-8")] = i

        left_ids = [joint_names[name] for name in self.cfg.wheel_joints_left if name in joint_names]
        right_ids = [joint_names[name] for name in self.cfg.wheel_joints_right if name in joint_names]

        if not left_ids or not right_ids:
            left_ids, right_ids = [], []
            for name, idx in joint_names.items():
                lname = name.lower()
                if "wheel" in lname and "left" in lname:
                    left_ids.append(idx)
                if "wheel" in lname and "right" in lname:
                    right_ids.append(idx)

        if not left_ids or not right_ids:
            print("Could not auto-detect wheel joints. Joints found in URDF:")
            for name, idx in joint_names.items():
                print(f"  [{idx}] {name}")
            raise ValueError(
                "Set cfg.wheel_joints_left / cfg.wheel_joints_right explicitly "
                "to the exact joint names printed above."
            )
        return left_ids, right_ids

    # =================================================================
    # Reset
    # =================================================================
    def reset(self) -> np.ndarray:
        p.resetBasePositionAndOrientation(
            self.robot_id,
            [self.cfg.spawn_x, self.cfg.spawn_y, self.cfg.spawn_z],
            p.getQuaternionFromEuler([0, 0, 0]),
        )
        # Synchronous sim -> we can zero velocity directly, no async
        # teleport-then-hope workaround needed.
        p.resetBaseVelocity(self.robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        for j in self._left_wheels + self._right_wheels:
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, targetVelocity=0)

        if self._dyn_obstacle_id is not None:
            self._t = 0.0
            p.resetBasePositionAndOrientation(
                self._dyn_obstacle_id, self._dyn_obstacle_base_pos,
                p.getQuaternionFromEuler([0, 0, 0]),
            )

        self._waypoints_hit = set()
        self._prev_ang = 0.0
        self.step_count = 0
        self._init_waypoints()

        # a couple of physics substeps so contact/lidar queries reflect the
        # settled reset state
        for _ in range(3):
            p.stepSimulation()

        self.prev_dist = self._dist_to_goal()
        return self._get_state()

    def _init_waypoints(self) -> None:
        sx, sy = self.cfg.spawn_x, self.cfg.spawn_y
        gx, gy = self.cfg.target_goal
        n = max(self.cfg.num_waypoints, 1)
        self._waypoints = [
            (sx + (gx - sx) * (i / n), sy + (gy - sy) * (i / n))
            for i in range(1, n + 1)
        ]

    # =================================================================
    # Step
    # =================================================================
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, float, float, bool, dict]:
        lin = float(np.clip(action[0], -1.0, 1.0)) * self.cfg.max_lin
        ang = float(np.clip(action[1], -1.0, 1.0)) * self.cfg.max_ang
        self._apply_wheel_velocities(lin, ang)

        if self._dyn_obstacle_id is not None and self.cfg.dynamic_obstacle_enabled:
            self._update_dynamic_obstacle()

        for _ in range(self._substeps_per_control):
            p.stepSimulation()
            if self.cfg.gui and self.cfg.render_realtime:
                time.sleep(1.0 / self.cfg.physics_hz)

        prev_dist = self.prev_dist
        curr_dist = self._dist_to_goal()
        self.prev_dist = curr_dist
        heading_error = self._heading_error()

        collision = self._check_collision()
        goal_reached = curr_dist < self.cfg.goal_threshold

        dense_reward, terminal_bonus = self._calculate_reward(
            collision, goal_reached, prev_dist, curr_dist, heading_error, ang)
        self._prev_ang = ang

        x, y, _ = self._get_pose()
        waypoint_bonus = self._waypoint_bonus(x, y)

        self.step_count += 1
        timeout = self.step_count >= self.cfg.max_steps_per_episode
        done = collision or goal_reached or timeout

        info = {
            "collision": collision,
            "goal_reached": goal_reached,
            "timeout": timeout and not collision and not goal_reached,
            "dist_to_goal": curr_dist,
        }
        state = self._get_state()
        return state, dense_reward, terminal_bonus + waypoint_bonus, waypoint_bonus, done, info

    def _apply_wheel_velocities(self, lin: float, ang: float) -> None:
        r = self.cfg.wheel_radius
        L = self.cfg.wheel_separation
        v_left = (lin - ang * L / 2.0) / r
        v_right = (lin + ang * L / 2.0) / r
        for j in self._left_wheels:
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL,
                                     targetVelocity=v_left, force=10.0)
        for j in self._right_wheels:
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL,
                                     targetVelocity=v_right, force=10.0)

    def _update_dynamic_obstacle(self) -> None:
        self._t += 1.0 / self.cfg.control_hz
        period = self.cfg.dynamic_obstacle_period_s
        y = -2.5 + 1.0 * math.sin(2 * math.pi * self._t / period)
        p.resetBasePositionAndOrientation(
            self._dyn_obstacle_id, [0.0, y, 0.5], p.getQuaternionFromEuler([0, 0, 0]))

    # =================================================================
    # Sensing
    # =================================================================
    def _get_pose(self) -> Tuple[float, float, float]:
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        return pos[0], pos[1], yaw

    def _dist_to_goal(self) -> float:
        x, y, _ = self._get_pose()
        gx, gy = self.cfg.target_goal
        return math.sqrt((gx - x) ** 2 + (gy - y) ** 2)

    def _heading_error(self) -> float:
        x, y, yaw = self._get_pose()
        gx, gy = self.cfg.target_goal
        angle = math.atan2(gy - y, gx - x) - yaw
        return math.atan2(math.sin(angle), math.cos(angle))

    def _get_lidar(self) -> np.ndarray:
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        x_base, y_base, z_base = pos
        yaw = p.getEulerFromQuaternion(orn)[2]
        # Rotate the URDF's lidar_joint forward offset (x=0.2 in the robot's
        # own frame) into world frame using current yaw.
        x = x_base + self.cfg.lidar_x_offset * math.cos(yaw)
        y = y_base + self.cfg.lidar_x_offset * math.sin(yaw)
        z = z_base + self.cfg.lidar_z_offset
        n = self.cfg.num_lidar_beams
        max_range = self.cfg.lidar_max_range

        from_points, to_points = [], []
        for i in range(n):
            beam_angle = yaw + 2 * math.pi * i / n
            dx, dy = math.cos(beam_angle), math.sin(beam_angle)
            from_points.append([x, y, z])
            to_points.append([x + dx * max_range, y + dy * max_range, z])

        results = p.rayTestBatch(from_points, to_points)
        ranges = np.full(n, max_range, dtype=np.float32)
        for i, res in enumerate(results):
            hit_fraction = res[2]
            if res[0] != -1:   # -1 = no hit
                ranges[i] = hit_fraction * max_range
        return ranges

    def _check_collision(self) -> bool:
        # Real contact points, excluding the ground plane the robot always
        # rests on — this sidesteps the NaN/zero-reading false-collision
        # bug entirely since it doesn't rely on lidar values at all.
        contacts = p.getContactPoints(bodyA=self.robot_id)
        for c in contacts:
            other_body = c[2]
            if other_body == self.plane_id:
                continue
            return True
        return False

    def _get_state(self) -> np.ndarray:
        scan = self._get_lidar()
        scan_norm = np.clip(scan, 0.0, self.cfg.lidar_max_range) / self.cfg.lidar_max_range
        dist_norm = min(self._dist_to_goal() / self.cfg.goal_dist_norm, 1.0)
        angle_norm = self._heading_error() / math.pi
        return np.concatenate((scan_norm, [dist_norm, angle_norm])).astype(np.float32)

    # =================================================================
    # Reward (identical formula to the fixed ROS train_agent.py)
    # =================================================================
    def _calculate_reward(self, collision: bool, goal_reached: bool,
                           prev_dist: float, curr_dist: float,
                           heading_error: float, ang_cmd: float) -> Tuple[float, float]:
        dense = self.cfg.reward_step
        dense += (prev_dist - curr_dist) * self.cfg.reward_progress_scale
        dense += -self.cfg.reward_heading_scale * (abs(heading_error) / math.pi)

        norm_ang = abs(ang_cmd) / max(self.cfg.max_ang, 1e-6)
        dense += -self.cfg.reward_spin_penalty * norm_ang

        ang_delta = abs(ang_cmd - self._prev_ang) / max(2.0 * self.cfg.max_ang, 1e-6)
        dense += -self.cfg.reward_smoothness_scale * ang_delta

        terminal_bonus = 0.0
        if collision:
            terminal_bonus += self.cfg.reward_collision
        if goal_reached:
            terminal_bonus += self.cfg.reward_goal
        return dense, terminal_bonus

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

    def close(self) -> None:
        p.disconnect(self._client)
