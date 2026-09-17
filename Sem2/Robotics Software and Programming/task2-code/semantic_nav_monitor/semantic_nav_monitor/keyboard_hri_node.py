#!/usr/bin/env python3
"""
keyboard_hri_node.py

Simple human-robot interaction (HRI) node for Task 2, satisfying the
assignment's "simple HRI, e.g. via keyboard" requirement literally: it reads
single key presses from the terminal it is run in (no Enter key needed) and
publishes the corresponding mission command on /hri_command, which
mission_controller.py already subscribes to.

Keys:
    s  -> start the mission
    p  -> pause the mission (robot stops in place, state machine holds)
    x  -> stop the mission (robot stops, transitions to MISSION_COMPLETE)
    q  -> quit this HRI node (does not stop the mission)

Run this in its own terminal, alongside semantic_nav_monitor.launch.py.
"""

import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

KEY_TO_COMMAND = {
    "s": "start",
    "p": "pause",
    "x": "stop",
}


def read_single_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key


class KeyboardHRINode(Node):
    def __init__(self):
        super().__init__("keyboard_hri_node")
        self.hri_pub = self.create_publisher(String, "/hri_command", 10)
        self.get_logger().info(
            "keyboard_hri_node ready. Press 's' start / 'p' pause / 'x' stop / 'q' quit."
        )

    def publish_command(self, command):
        msg = String()
        msg.data = command
        self.hri_pub.publish(msg)
        self.get_logger().info(f"Published HRI command: '{command}'")


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardHRINode()
    try:
        while rclpy.ok():
            key = read_single_key().lower()
            if key == "q":
                node.get_logger().info("Quitting keyboard_hri_node.")
                break
            command = KEY_TO_COMMAND.get(key)
            if command is not None:
                node.publish_command(command)
            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
