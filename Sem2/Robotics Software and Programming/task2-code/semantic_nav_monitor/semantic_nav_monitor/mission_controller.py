#!/usr/bin/env python3
"""
mission_controller.py

Goal-oriented mission controller for a TurtleBot3 robot.

Implements a five-state finite state machine (Idle, Navigate, AvoidObstacle,
Replan, MissionComplete), subscribes to /scan and /odom, publishes velocity
commands on /cmd_vel, publishes state transitions on /mission_state, and
accepts simple human-robot-interaction commands on /hri_command
("start", "pause", "stop").
"""

import math
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class MissionState(Enum):
    IDLE = "IDLE"
    NAVIGATE = "NAVIGATE"
    AVOID_OBSTACLE = "AVOID_OBSTACLE"
    REPLAN = "REPLAN"
    MISSION_COMPLETE = "MISSION_COMPLETE"


# --- Tunable mission parameters -------------------------------------------
WAYPOINTS = [
    (1.5, 0.0),
    (1.5, 1.5),
    (0.0, 1.5),
    (0.0, 0.0),
]
WAYPOINT_TOLERANCE_M = 0.15
OBSTACLE_SAFETY_RANGE_M = 0.45
FORWARD_SECTOR_DEG = 30  # +/- degrees either side of the robot's forward axis
LINEAR_SPEED_MPS = 0.15
ANGULAR_GAIN = 1.2
MAX_ANGULAR_SPEED_RADPS = 1.0
CONTROL_PERIOD_S = 0.1
CONSECUTIVE_DETECTIONS_REQUIRED = 3  # debounce for obstacle detection


def yaw_from_quaternion(q):
    """Extract yaw (rotation about Z) from a geometry_msgs Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class MissionController(Node):
    def __init__(self):
        super().__init__("mission_controller")

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscriptions
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, sensor_qos
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )
        self.hri_sub = self.create_subscription(
            String, "/hri_command", self.hri_callback, 10
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.state_pub = self.create_publisher(String, "/mission_state", 10)

        # Mission / sensor state
        self.state = MissionState.IDLE
        self.mission_running = False
        self.current_pose = None          # (x, y, yaw)
        self.latest_scan = None
        self.waypoint_index = 0
        self.obstacle_hit_count = 0

        self.publish_state(self.state)  # announce initial state

        self.control_timer = self.create_timer(
            CONTROL_PERIOD_S, self.control_loop
        )

        self.get_logger().info("mission_controller node started in IDLE state.")

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.current_pose = (pos.x, pos.y, yaw)

    def hri_callback(self, msg: String):
        command = msg.data.strip().lower()
        self.get_logger().info(f"Received HRI command: '{command}'")

        if command == "start":
            if self.state == MissionState.IDLE:
                self.mission_running = True
                self.waypoint_index = 0
                self.transition_to(MissionState.NAVIGATE)
        elif command == "pause":
            self.mission_running = False
            self.publish_zero_velocity()
        elif command == "stop":
            self.mission_running = False
            self.publish_zero_velocity()
            self.transition_to(MissionState.MISSION_COMPLETE)
        else:
            self.get_logger().warn(f"Unknown HRI command ignored: '{command}'")

    # ------------------------------------------------------------------ #
    # State machine
    # ------------------------------------------------------------------ #
    def transition_to(self, new_state: MissionState):
        if new_state != self.state:
            self.get_logger().info(f"State transition: {self.state.value} -> {new_state.value}")
            self.state = new_state
            self.publish_state(new_state)

    def publish_state(self, state: MissionState):
        msg = String()
        msg.data = state.value
        self.state_pub.publish(msg)

    def publish_zero_velocity(self):
        self.cmd_vel_pub.publish(Twist())

    def forward_obstacle_distance(self):
        """Return the minimum valid LIDAR range within the forward sector,
        or None if no scan has been received yet."""
        if self.latest_scan is None:
            return None

        scan = self.latest_scan
        n = len(scan.ranges)
        if n == 0:
            return None

        sector_rad = math.radians(FORWARD_SECTOR_DEG)
        min_range = float("inf")
        found = False

        for i, r in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment
            angle = normalize_angle(angle)
            if abs(angle) <= sector_rad:
                if scan.range_min <= r <= scan.range_max:
                    found = True
                    min_range = min(min_range, r)

        return min_range if found else None

    def control_loop(self):
        if self.state == MissionState.IDLE:
            self.publish_zero_velocity()
            return

        if not self.mission_running:
            self.publish_zero_velocity()
            return

        if self.current_pose is None:
            return  # wait for first odometry message

        obstacle_range = self.forward_obstacle_distance()
        obstacle_detected = (
            obstacle_range is not None and obstacle_range < OBSTACLE_SAFETY_RANGE_M
        )

        if obstacle_detected:
            self.obstacle_hit_count += 1
        else:
            self.obstacle_hit_count = 0

        if self.obstacle_hit_count >= CONSECUTIVE_DETECTIONS_REQUIRED:
            if self.state != MissionState.AVOID_OBSTACLE:
                self.transition_to(MissionState.AVOID_OBSTACLE)
            self.run_avoid_obstacle()
            return

        if self.state == MissionState.AVOID_OBSTACLE:
            # obstacle has cleared: re-orient towards the goal before resuming
            self.transition_to(MissionState.REPLAN)

        if self.state == MissionState.REPLAN:
            self.run_replan()
            return

        if self.state in (MissionState.NAVIGATE, MissionState.REPLAN):
            self.run_navigate()
            return

    # ------------------------------------------------------------------ #
    # Behaviours
    # ------------------------------------------------------------------ #
    def run_navigate(self):
        if self.waypoint_index >= len(WAYPOINTS):
            self.publish_zero_velocity()
            self.mission_running = False
            self.transition_to(MissionState.MISSION_COMPLETE)
            return

        goal_x, goal_y = WAYPOINTS[self.waypoint_index]
        x, y, yaw = self.current_pose

        dx = goal_x - x
        dy = goal_y - y
        distance = math.hypot(dx, dy)

        if distance < WAYPOINT_TOLERANCE_M:
            self.get_logger().info(
                f"Reached waypoint {self.waypoint_index}: ({goal_x:.2f}, {goal_y:.2f})"
            )
            self.waypoint_index += 1
            return

        target_heading = math.atan2(dy, dx)
        heading_error = normalize_angle(target_heading - yaw)

        cmd = Twist()
        cmd.linear.x = LINEAR_SPEED_MPS
        cmd.angular.z = max(
            -MAX_ANGULAR_SPEED_RADPS,
            min(MAX_ANGULAR_SPEED_RADPS, ANGULAR_GAIN * heading_error),
        )
        self.cmd_vel_pub.publish(cmd)

        if self.state != MissionState.NAVIGATE:
            self.transition_to(MissionState.NAVIGATE)

    def run_avoid_obstacle(self):
        # Stop forward motion and rotate away from the obstacle.
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = MAX_ANGULAR_SPEED_RADPS
        self.cmd_vel_pub.publish(cmd)

    def run_replan(self):
        # Re-evaluate heading towards the current waypoint, then hand back to
        # Navigate on the next control cycle.
        self.publish_zero_velocity()
        self.transition_to(MissionState.NAVIGATE)


def main(args=None):
    rclpy.init(args=args)
    node = MissionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_zero_velocity()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
