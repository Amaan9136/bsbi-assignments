#!/usr/bin/env python3
"""
mission_controller.py

Goal-oriented mission controller for a TurtleBot3 robot.

TASK 2 USE CASE: "Warehouse Inspection Patrol Robot"
The robot's mission is to patrol four inspection checkpoints laid out in the
custom warehouse_inspection.sdf (see the worlds/ folder), reporting its
own behaviour throughout. Eight pallet stacks (two per patrol leg) plus three
small centerline obstacles sit near the route so a real run exercises every
state repeatedly, not just NAVIGATE.

Implements the assignment's five-state finite state machine (Idle, Navigate,
AvoidObstacle, Replan, MissionComplete), subscribes to /scan and /odom,
publishes velocity commands on /cmd_vel, publishes state transitions on
/mission_state, and accepts simple human-robot-interaction commands on
/hri_command ("start", "pause", "stop").

AVOID_OBSTACLE / REPLAN previously turned in place until a clear reading came
back, then handed straight back to NAVIGATE facing the original goal - which
often meant driving straight back into the same obstacle and oscillating in
front of it. The two states now split the work differently:
  - AVOID_OBSTACLE: backs off only as far as needed (REVERSE), then rotates
    to a heading it has actively confirmed is clear by probing the scan at
    several candidate angles - weighting nearby walls exactly like obstacles,
    since a wall shows up as a short range reading in the same sectors
    (ORIENT). This is what lets it steer away from a wall in a corner
    instead of into it.
  - REPLAN: once a clear heading is confirmed, the robot drives forward
    along that heading for a committed distance (not just a moment) before
    handing back to NAVIGATE, so the waypoint-seeking controller only
    resumes once the robot has actually cleared the obstacle. If a new
    obstacle appears while REPLAN is committing, control goes back to
    AVOID_OBSTACLE to pick a fresh heading rather than stalling in place.

NOTE (Jazzy/TB3 port): the authoritative ros_gz_bridge started by
turtlebot3_gazebo's launch files subscribes to /cmd_vel as
geometry_msgs/msg/TwistStamped (not plain Twist). We publish TwistStamped
here so velocity commands actually reach Gazebo.
"""

import math
import time
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import String


class MissionState(Enum):
    IDLE = "IDLE"
    NAVIGATE = "NAVIGATE"
    AVOID_OBSTACLE = "AVOID_OBSTACLE"
    REPLAN = "REPLAN"
    MISSION_COMPLETE = "MISSION_COMPLETE"


WAYPOINTS = [
    (3.0, 0.0),
    (3.0, 3.0),
    (0.0, 3.0),
    (0.0, 0.0),
]
WAYPOINT_TOLERANCE_M = 0.15
OBSTACLE_SAFETY_RANGE_M = 0.45
SIDE_SECTOR_DEG = 150
FORWARD_SECTOR_DEG = 30
ROBOT_HALF_WIDTH_M = 0.11
OBSTACLE_LATERAL_MARGIN_M = 0.12
LINEAR_SPEED_MPS = 0.15
ANGULAR_GAIN = 1.2
MAX_ANGULAR_SPEED_RADPS = 1.0
CONTROL_PERIOD_S = 0.1
CONSECUTIVE_DETECTIONS_REQUIRED = 3
REVERSE_TRIGGER_RANGE_M = 0.30
REVERSE_DISTANCE_M = 0.12
REVERSE_SPEED_MPS = -0.1
REVERSE_MAX_DURATION_S = 2.0
ESCAPE_PROBE_ANGLES_DEG = [45, 65, 85, 105, 125]
ESCAPE_PROBE_HALFWIDTH_DEG = 10
ESCAPE_CLEARANCE_MARGIN_M = 0.35
ORIENT_YAW_TOLERANCE_RAD = 0.08
ORIENT_MAX_DURATION_S = 4.0
REPLAN_LINEAR_SPEED_MPS = 0.12
REPLAN_DISTANCE_M = 0.5
REPLAN_MAX_DURATION_S = 6.0
MAX_AVOID_ATTEMPTS = 6
DIAGNOSTIC_THROTTLE_S = 3.0
STALL_CHECK_DURATION_S = 1.0
STALL_DISTANCE_THRESHOLD_M = 0.05
SCAN_TIMEOUT_WARN_S = 3.0


def yaw_from_quaternion(q):
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

        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, sensor_qos
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )
        self.hri_sub = self.create_subscription(
            String, "/hri_command", self.hri_callback, 10
        )

        self.cmd_vel_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.state_pub = self.create_publisher(String, "/mission_state", 10)

        self.state = MissionState.IDLE
        self.mission_running = False
        self.current_pose = None
        self.latest_scan = None
        self.waypoint_index = 0
        self.obstacle_hit_count = 0
        self.avoid_attempts = 0
        self.avoid_phase = "REVERSE"
        self.avoid_phase_ticks = 0
        self.avoid_turn_direction = 1.0
        self.avoid_target_yaw = 0.0
        self.avoid_reverse_origin = None
        self.avoid_commit_origin = None

        self._odom_received_at_least_once = False
        self._scan_received_at_least_once = False
        self._logged_no_odom_since_start = False
        self._logged_no_scan_since_start = False
        self._last_scan_wall_time = None
        self.stall_origin = None
        self.stall_origin_time = None

        self.publish_state(self.state)

        self.control_timer = self.create_timer(
            CONTROL_PERIOD_S, self.control_loop
        )

        self.get_logger().info("mission_controller node started in IDLE state.")
        self.get_logger().info(
            "Subscribed to /scan and /odom, publishing /cmd_vel and /mission_state, "
            "listening for /hri_command."
        )

    def scan_callback(self, msg: LaserScan):
        self._last_scan_wall_time = time.monotonic()
        if not self._scan_received_at_least_once:
            self._scan_received_at_least_once = True
            self.get_logger().info(
                "First /scan message received - LIDAR data is flowing. "
                f"ranges={len(msg.ranges)} angle_min={msg.angle_min:.3f} "
                f"angle_max={msg.angle_max:.3f} angle_increment={msg.angle_increment:.5f} "
                f"range_min={msg.range_min:.3f} range_max={msg.range_max:.3f}"
            )
        self.latest_scan = msg

    def odom_callback(self, msg: Odometry):
        if not self._odom_received_at_least_once:
            self._odom_received_at_least_once = True
            self.get_logger().info("First /odom message received - odometry is flowing.")
        pos = msg.pose.pose.position
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.current_pose = (pos.x, pos.y, yaw)

    def hri_callback(self, msg: String):
        command = msg.data.strip().lower()
        self.get_logger().info(f"Received HRI command: '{command}'")

        if command == "start":
            if not self._odom_received_at_least_once:
                self.get_logger().warn(
                    "'start' received, but NO /odom message has ever arrived. "
                    "The robot will stay stationary until odometry is published. "
                    "Check 'ros2 topic hz /odom' and your ros_gz_bridge / spawn setup."
                )
            if self.state == MissionState.IDLE:
                self.mission_running = True
                self.waypoint_index = 0
                self.obstacle_hit_count = 0
                self.avoid_attempts = 0
                self._logged_no_odom_since_start = False
                self.stall_origin = None
                self.stall_origin_time = None
                self.transition_to(MissionState.NAVIGATE)
            else:
                self.get_logger().warn(
                    f"'start' ignored: mission is not IDLE (current state: {self.state.value})."
                )
        elif command == "pause":
            self.mission_running = False
            self.publish_zero_velocity()
        elif command == "stop":
            self.mission_running = False
            self.publish_zero_velocity()
            self.transition_to(MissionState.MISSION_COMPLETE)
        else:
            self.get_logger().warn(f"Unknown HRI command ignored: '{command}'")

    def transition_to(self, new_state: MissionState):
        if new_state != self.state:
            self.get_logger().info(f"State transition: {self.state.value} -> {new_state.value}")
            self.state = new_state
            self.publish_state(new_state)

    def publish_state(self, state: MissionState):
        msg = String()
        msg.data = state.value
        self.state_pub.publish(msg)

    def _make_stamped_twist(self, linear_x=0.0, angular_z=0.0):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.twist.linear.x = linear_x
        cmd.twist.angular.z = angular_z
        return cmd

    def publish_zero_velocity(self):
        self.cmd_vel_pub.publish(self._make_stamped_twist())

    def forward_obstacle_distance(self):
        if self.latest_scan is None:
            return None

        scan = self.latest_scan
        n = len(scan.ranges)
        if n == 0:
            return None

        corridor_half_width = ROBOT_HALF_WIDTH_M + OBSTACLE_LATERAL_MARGIN_M
        min_forward = float("inf")
        found = False

        for i, r in enumerate(scan.ranges):
            if not (0.0 < r <= scan.range_max):
                continue
            angle = normalize_angle(scan.angle_min + i * scan.angle_increment)
            if abs(angle) >= math.pi / 2.0:
                continue
            forward = r * math.cos(angle)
            lateral = r * math.sin(angle)
            if forward <= 0.0:
                continue
            if abs(lateral) <= corridor_half_width:
                found = True
                min_forward = min(min_forward, forward)

        return min_forward if found else None

    def side_clearance(self):
        if self.latest_scan is None:
            return None, None

        scan = self.latest_scan
        if len(scan.ranges) == 0:
            return None, None

        side_rad = math.radians(SIDE_SECTOR_DEG)
        forward_rad = math.radians(FORWARD_SECTOR_DEG)
        left_min = float("inf")
        right_min = float("inf")
        left_found = False
        right_found = False

        for i, r in enumerate(scan.ranges):
            angle = normalize_angle(scan.angle_min + i * scan.angle_increment)
            if not (0.0 < r <= scan.range_max):
                continue
            if forward_rad < angle <= side_rad:
                left_found = True
                left_min = min(left_min, r)
            elif -side_rad <= angle < -forward_rad:
                right_found = True
                right_min = min(right_min, r)

        left = left_min if left_found else None
        right = right_min if right_found else None
        return left, right

    def probe_heading_clearance(self, relative_angle_rad):
        if self.latest_scan is None:
            return None

        scan = self.latest_scan
        if len(scan.ranges) == 0:
            return None

        halfwidth = math.radians(ESCAPE_PROBE_HALFWIDTH_DEG)
        min_range = float("inf")
        found = False

        for i, r in enumerate(scan.ranges):
            if not (0.0 < r <= scan.range_max):
                continue
            angle = normalize_angle(scan.angle_min + i * scan.angle_increment)
            if abs(normalize_angle(angle - relative_angle_rad)) <= halfwidth:
                found = True
                min_range = min(min_range, r)

        return min_range if found else None

    def compute_escape_angle(self, direction_sign):
        required_clearance = OBSTACLE_SAFETY_RANGE_M + ESCAPE_CLEARANCE_MARGIN_M
        for probe_deg in ESCAPE_PROBE_ANGLES_DEG:
            relative_angle = math.radians(probe_deg) * direction_sign
            clearance = self.probe_heading_clearance(relative_angle)
            if clearance is None or clearance >= required_clearance:
                return relative_angle
        return math.radians(ESCAPE_PROBE_ANGLES_DEG[-1]) * direction_sign

    def control_loop(self):
        if self.state == MissionState.IDLE:
            self.publish_zero_velocity()
            return

        if not self.mission_running:
            self.publish_zero_velocity()
            return

        if self.current_pose is None:
            self.get_logger().warn(
                "Mission is running but no /odom has been received yet - "
                "cannot navigate without a pose. Waiting...",
                throttle_duration_sec=DIAGNOSTIC_THROTTLE_S,
            )
            return

        if not self._scan_received_at_least_once:
            self.get_logger().warn(
                "Mission is running but no /scan message has EVER been received. "
                "Obstacle avoidance is BLIND until this is fixed - the robot will "
                "drive straight through obstacles. Check 'ros2 topic hz /scan' and "
                "'ros2 topic info /scan' (QoS) in another terminal.",
                throttle_duration_sec=DIAGNOSTIC_THROTTLE_S,
            )
        elif (
            self._last_scan_wall_time is not None
            and time.monotonic() - self._last_scan_wall_time > SCAN_TIMEOUT_WARN_S
        ):
            self.get_logger().warn(
                f"No /scan message received in over {SCAN_TIMEOUT_WARN_S:.0f}s "
                "(was flowing before, now stopped) - obstacle avoidance is stale.",
                throttle_duration_sec=DIAGNOSTIC_THROTTLE_S,
            )

        obstacle_range = self.forward_obstacle_distance()
        obstacle_detected = (
            obstacle_range is not None and obstacle_range < OBSTACLE_SAFETY_RANGE_M
        )

        if obstacle_detected:
            self.obstacle_hit_count += 1
        else:
            self.obstacle_hit_count = 0

        if self.state == MissionState.NAVIGATE:
            if self.obstacle_hit_count >= CONSECUTIVE_DETECTIONS_REQUIRED:
                self.start_avoidance_attempt(first=True)
                self.step_avoid_obstacle(obstacle_detected)
                return
            self.run_navigate()
            return

        if self.state in (MissionState.AVOID_OBSTACLE, MissionState.REPLAN):
            self.step_avoid_obstacle(obstacle_detected)
            return

    def start_avoidance_attempt(self, first):
        self.avoid_attempts = 1 if first else self.avoid_attempts + 1
        self.stall_origin = None
        self.stall_origin_time = None

        obstacle_range = self.forward_obstacle_distance()
        if obstacle_range is not None and obstacle_range < REVERSE_TRIGGER_RANGE_M:
            self.avoid_phase = "REVERSE"
            self.avoid_reverse_origin = self.current_pose[:2] if self.current_pose else None
            self.avoid_phase_ticks = 0
        else:
            self.enter_orient_phase()

        self.transition_to(MissionState.AVOID_OBSTACLE)

    def enter_orient_phase(self):
        left, right = self.side_clearance()
        if left is not None and right is not None:
            self.avoid_turn_direction = 1.0 if left >= right else -1.0
        elif left is not None:
            self.avoid_turn_direction = 1.0
        elif right is not None:
            self.avoid_turn_direction = -1.0
        else:
            self.avoid_turn_direction = 1.0

        escape_angle = self.compute_escape_angle(self.avoid_turn_direction)
        _, _, yaw = self.current_pose
        self.avoid_target_yaw = normalize_angle(yaw + escape_angle)
        self.avoid_phase = "ORIENT"
        self.avoid_phase_ticks = 0
        self.transition_to(MissionState.AVOID_OBSTACLE)

    def enter_replan_phase(self):
        self.avoid_commit_origin = self.current_pose[:2] if self.current_pose else None
        self.avoid_phase = "COMMIT"
        self.avoid_phase_ticks = 0
        self.transition_to(MissionState.REPLAN)

    def step_avoid_obstacle(self, obstacle_detected):
        if self.avoid_attempts > MAX_AVOID_ATTEMPTS:
            self.get_logger().error(
                f"Unable to clear obstacle after {MAX_AVOID_ATTEMPTS} attempts. "
                "Holding position and awaiting operator intervention.",
                throttle_duration_sec=DIAGNOSTIC_THROTTLE_S,
            )
            self.publish_zero_velocity()
            return

        if self.avoid_phase == "REVERSE":
            self.run_avoid_reverse()
            return

        if self.avoid_phase == "ORIENT":
            self.run_avoid_orient()
            return

        if self.avoid_phase == "COMMIT":
            self.run_avoid_commit(obstacle_detected)
            return

    def run_avoid_reverse(self):
        self.cmd_vel_pub.publish(
            self._make_stamped_twist(linear_x=REVERSE_SPEED_MPS, angular_z=0.0)
        )
        self.avoid_phase_ticks += 1

        displaced = 0.0
        if self.avoid_reverse_origin is not None and self.current_pose is not None:
            x, y, _ = self.current_pose
            displaced = math.hypot(
                x - self.avoid_reverse_origin[0], y - self.avoid_reverse_origin[1]
            )

        timed_out = self.avoid_phase_ticks * CONTROL_PERIOD_S >= REVERSE_MAX_DURATION_S
        if displaced >= REVERSE_DISTANCE_M or timed_out:
            self.enter_orient_phase()

    def run_avoid_orient(self):
        x, y, yaw = self.current_pose
        heading_error = normalize_angle(self.avoid_target_yaw - yaw)
        self.avoid_phase_ticks += 1

        if abs(heading_error) <= ORIENT_YAW_TOLERANCE_RAD:
            self.enter_replan_phase()
            return

        if self.avoid_phase_ticks * CONTROL_PERIOD_S >= ORIENT_MAX_DURATION_S:
            self.get_logger().warn(
                f"Avoidance attempt {self.avoid_attempts} timed out while orienting; retrying."
            )
            self.start_avoidance_attempt(first=False)
            return

        angular_z = max(
            -MAX_ANGULAR_SPEED_RADPS,
            min(MAX_ANGULAR_SPEED_RADPS, ANGULAR_GAIN * heading_error),
        )
        self.cmd_vel_pub.publish(
            self._make_stamped_twist(linear_x=0.0, angular_z=angular_z)
        )

    def run_avoid_commit(self, obstacle_detected):
        if obstacle_detected:
            self.start_avoidance_attempt(first=False)
            return

        x, y, yaw = self.current_pose
        heading_error = normalize_angle(self.avoid_target_yaw - yaw)
        angular_z = max(
            -MAX_ANGULAR_SPEED_RADPS,
            min(MAX_ANGULAR_SPEED_RADPS, ANGULAR_GAIN * heading_error),
        )
        self.cmd_vel_pub.publish(
            self._make_stamped_twist(linear_x=REPLAN_LINEAR_SPEED_MPS, angular_z=angular_z)
        )
        self.avoid_phase_ticks += 1

        displaced = 0.0
        if self.avoid_commit_origin is not None:
            displaced = math.hypot(
                x - self.avoid_commit_origin[0], y - self.avoid_commit_origin[1]
            )

        timed_out = self.avoid_phase_ticks * CONTROL_PERIOD_S >= REPLAN_MAX_DURATION_S
        if displaced >= REPLAN_DISTANCE_M or timed_out:
            self.obstacle_hit_count = 0
            self.avoid_attempts = 0
            self.stall_origin = None
            self.stall_origin_time = None
            self.transition_to(MissionState.NAVIGATE)

    def is_stalled(self, x, y):
        """Odometry-based fallback obstacle detector, independent of LIDAR.

        If we've been commanding forward motion for STALL_CHECK_DURATION_S
        seconds but the robot's actual position has barely changed, it is
        physically blocked by something - regardless of whether /scan is
        working, correctly configured, or detecting it. This guarantees the
        robot can never indefinitely push against an obstacle even if the
        LIDAR-based detection path is broken for some environment-specific
        reason.
        """
        now = time.monotonic()
        if self.stall_origin is None:
            self.stall_origin = (x, y)
            self.stall_origin_time = now
            return False

        elapsed = now - self.stall_origin_time
        if elapsed < STALL_CHECK_DURATION_S:
            return False

        displacement = math.hypot(x - self.stall_origin[0], y - self.stall_origin[1])
        self.stall_origin = (x, y)
        self.stall_origin_time = now

        if displacement < STALL_DISTANCE_THRESHOLD_M:
            self.get_logger().warn(
                f"STALL DETECTED (odometry): commanded forward motion for "
                f"{elapsed:.1f}s but moved only {displacement:.3f}m. Treating as "
                "a blocked/pushed obstacle regardless of LIDAR state and "
                "triggering avoidance."
            )
            return True
        return False

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
            self.stall_origin = None
            self.stall_origin_time = None
            return

        if self.is_stalled(x, y):
            self.start_avoidance_attempt(first=True)
            self.step_avoid_obstacle(False)
            return

        target_heading = math.atan2(dy, dx)
        heading_error = normalize_angle(target_heading - yaw)

        angular_z = max(
            -MAX_ANGULAR_SPEED_RADPS,
            min(MAX_ANGULAR_SPEED_RADPS, ANGULAR_GAIN * heading_error),
        )
        self.cmd_vel_pub.publish(
            self._make_stamped_twist(linear_x=LINEAR_SPEED_MPS, angular_z=angular_z)
        )

        if self.state != MissionState.NAVIGATE:
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