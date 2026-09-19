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
    v  -> toggle the planned-path line on/off in the Gazebo Sim client
          (publishes std_msgs/Bool on /show_planned_path; the plan itself
          never changes - this only shows/hides the drawn line)
    q  -> quit this HRI node (does not stop the mission)

Run this in its own terminal, alongside semantic_nav_monitor.launch.py.
"""

import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

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
        self.show_path_pub = self.create_publisher(Bool, "/show_planned_path", 10)
        self.path_visible = False
        self._wait_for_subscriber(timeout_sec=5.0)
        self.get_logger().info(
            "keyboard_hri_node ready. Press 's' start / 'p' pause / 'x' stop / "
            "'v' toggle planned-path line / 'q' quit."
        )

    def _wait_for_subscriber(self, timeout_sec):
        start = time.monotonic()
        while rclpy.ok() and self.hri_pub.get_subscription_count() == 0:
            if time.monotonic() - start >= timeout_sec:
                self.get_logger().warn(
                    "No subscriber on /hri_command after "
                    f"{timeout_sec:.0f}s (is mission_controller running?). "
                    "Commands sent now may be missed until it connects."
                )
                return
            rclpy.spin_once(self, timeout_sec=0.1)

    def publish_command(self, command):
        msg = String()
        msg.data = command
        self.hri_pub.publish(msg)
        self.get_logger().info(f"Published HRI command: '{command}'")

    def toggle_path_visibility(self):
        self.path_visible = not self.path_visible
        msg = Bool()
        msg.data = self.path_visible
        self.show_path_pub.publish(msg)
        state = "ON" if self.path_visible else "OFF"
        self.get_logger().info(f"Planned-path line: {state}")


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardHRINode()
    try:
        while rclpy.ok():
            key = read_single_key().lower()
            if key == "q":
                node.get_logger().info("Quitting keyboard_hri_node.")
                break
            if key == "v":
                node.toggle_path_visibility()
                rclpy.spin_once(node, timeout_sec=0.0)
                continue
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