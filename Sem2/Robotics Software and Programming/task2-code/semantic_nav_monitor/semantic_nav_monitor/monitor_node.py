#!/usr/bin/env python3
"""
monitor_node.py

Passive monitoring node for the semantic_nav_monitor mission.

Subscribes to /mission_state and /odom, logs every state transition with a
timestamp, and computes simple performance metrics: total mission time, time
spent in each state, approximate distance travelled (via odometry
integration), and a count of obstacle encounters (every entry into the
AVOID_OBSTACLE state), used as the assignment's "collisions" metric since
the robot is designed to avoid rather than actually collide with obstacles.
"""

import math
import time

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from nav_msgs.msg import Odometry


class MonitorNode(Node):
    def __init__(self):
        super().__init__("monitor_node")

        self.state_sub = self.create_subscription(
            String, "/mission_state", self.state_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )

        self.current_state = None
        self.state_entry_time = None
        self.mission_start_time = None
        self.time_in_state = {}  # state name -> accumulated seconds

        self.last_position = None
        self.total_distance_m = 0.0
        self.obstacle_encounter_count = 0

        # Periodically print a metrics summary so progress is visible even
        # without additional state transitions.
        self.summary_timer = self.create_timer(5.0, self.print_summary)

        self.get_logger().info("monitor_node started. Waiting for mission_state updates.")

    def state_callback(self, msg: String):
        new_state = msg.data
        now = time.monotonic()

        if self.mission_start_time is None:
            self.mission_start_time = now

        if self.current_state is not None and self.state_entry_time is not None:
            elapsed = now - self.state_entry_time
            self.time_in_state[self.current_state] = (
                self.time_in_state.get(self.current_state, 0.0) + elapsed
            )

        self.get_logger().info(f"[STATE CHANGE] {self.current_state} -> {new_state}")

        if new_state == "AVOID_OBSTACLE" and self.current_state != "AVOID_OBSTACLE":
            self.obstacle_encounter_count += 1
            self.get_logger().info("Obstacle detected -> switching to AvoidObstacle")

        self.current_state = new_state
        self.state_entry_time = now

        if new_state == "MISSION_COMPLETE":
            self.print_summary(final=True)

    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        current = (pos.x, pos.y)

        if self.last_position is not None:
            dx = current[0] - self.last_position[0]
            dy = current[1] - self.last_position[1]
            self.total_distance_m += math.hypot(dx, dy)

        self.last_position = current

    def print_summary(self, final: bool = False):
        if self.mission_start_time is None:
            return

        total_time = time.monotonic() - self.mission_start_time
        header = "FINAL MISSION SUMMARY" if final else "Mission summary (in progress)"

        self.get_logger().info(f"--- {header} ---")
        self.get_logger().info(f"  Total mission time  : {total_time:.1f} s")
        self.get_logger().info(f"  Distance travelled  : {self.total_distance_m:.2f} m")
        self.get_logger().info(f"  Obstacle encounters : {self.obstacle_encounter_count}")
        self.get_logger().info("  Time spent per state:")
        for state, seconds in sorted(self.time_in_state.items()):
            self.get_logger().info(f"    {state:<18s}: {seconds:.1f} s")
        self.get_logger().info("-" * 40)


def main(args=None):
    rclpy.init(args=args)
    node = MonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()