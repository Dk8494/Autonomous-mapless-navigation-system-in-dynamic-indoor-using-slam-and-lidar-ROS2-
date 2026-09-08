#!/usr/bin/env python3

import sys
import tty
import termios
import rclpy

from rclpy.node import Node
from geometry_msgs.msg import Twist

LINEAR_SPEED = 0.5
ANGULAR_SPEED = 1.0


def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        ch1 = sys.stdin.read(1)

        if ch1 == '\x1b':
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            return ch1 + ch2 + ch3

        return ch1

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class Teleop(Node):

    def __init__(self):
        super().__init__('teleop_arrow')

        self.pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.get_logger().info(
            "Arrow Teleop Ready\n"
            "↑ Forward\n"
            "↓ Backward\n"
            "← Rotate Left\n"
            "→ Rotate Right\n"
            "Space = Stop\n"
            "q = Quit"
        )

    def run(self):

        while rclpy.ok():

            key = get_key()

            msg = Twist()

            if key == '\x1b[A':        # up
                msg.linear.x = LINEAR_SPEED

            elif key == '\x1b[B':      # down
                msg.linear.x = -LINEAR_SPEED

            elif key == '\x1b[D':      # left
                msg.angular.z = ANGULAR_SPEED

            elif key == '\x1b[C':      # right
                msg.angular.z = -ANGULAR_SPEED

            elif key == ' ':
                pass

            elif key == 'q':
                break

            self.pub.publish(msg)


def main():

    rclpy.init()

    node = Teleop()

    try:
        node.run()

    except KeyboardInterrupt:
        pass

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
