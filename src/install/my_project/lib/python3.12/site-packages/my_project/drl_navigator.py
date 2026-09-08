import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import torch
import numpy as np
import math
import os
from .ppo_model import PPOActorCritic


class DRLNavigatorNode(Node):
    def __init__(self):
        super().__init__('drl_navigator')

        self.declare_parameter('model_path', '')
        self.declare_parameter('goal_x', 5.0)
        self.declare_parameter('goal_y', -3.0)
        self.declare_parameter('max_linear_vel', 0.22)
        self.declare_parameter('max_angular_vel', 1.0)

        model_path = self.get_parameter('model_path').value
        self.target_goal = [
            self.get_parameter('goal_x').value,
            self.get_parameter('goal_y').value
        ]
        self.max_lin = self.get_parameter('max_linear_vel').value
        self.max_ang = self.get_parameter('max_angular_vel').value

        # State: 36 downsampled LiDAR rays + [distance, angle] = 38
        self.state_dim = 38
        self.action_dim = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy = PPOActorCritic(self.state_dim, self.action_dim).to(self.device)

        if model_path and os.path.exists(model_path):
            self.policy.load_state_dict(
                torch.load(model_path, map_location=self.device))
            self.get_logger().info(f'Loaded model from {model_path}')
        else:
            self.get_logger().warn('No model loaded — running random policy')

        self.policy.eval()

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.current_scan = None
        self.current_pose = None

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('DRL Navigator Node started.')

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=0.0,
                               posinf=msg.range_max, neginf=0.0)
        self.current_scan = ranges[::10][:36]

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_relative_target(self):
        if self.current_pose is None:
            return [0.0, 0.0]
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)
        dx = self.target_goal[0] - x
        dy = self.target_goal[1] - y
        dist = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx) - yaw
        angle = math.atan2(math.sin(angle), math.cos(angle))
        return [dist, angle]

    def control_loop(self):
        if self.current_scan is None or self.current_pose is None:
            return

        rel = self.get_relative_target()
        state = np.concatenate((self.current_scan, rel)).astype(np.float32)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist, _ = self.policy(state_tensor)
            action = dist.mean  # deterministic inference

        lin = float(torch.clamp(action[0][0], -1.0, 1.0)) * self.max_lin
        ang = float(torch.clamp(action[0][1], -1.0, 1.0)) * self.max_ang

        twist = Twist()
        twist.linear.x = lin
        twist.angular.z = ang
        self.cmd_pub.publish(twist)

        if rel[0] < 0.3:
            self.get_logger().info('Goal reached!')
            twist = Twist()
            self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = DRLNavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()