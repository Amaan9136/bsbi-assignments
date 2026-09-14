#!/usr/bin/env python3
"""
navigation_controller.py

ROS 2 node wrapper around control_core.compute_velocity(). Subscribes
to /target_offset and /target_visible, calls the pure control logic in
control_core.py, and publishes the result as a geometry_msgs/Twist on
/cmd_vel. All decision logic lives in control_core.py so it can be
tested and debugged independently of ROS; this file only handles ROS
plumbing, timing and the safety watchdog.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import Bool, Float32

from turtlebot3_vision_nav.control_core import compute_velocity

# If no perception message arrives within this many seconds, stop the
# robot rather than act on stale data, e.g. if vision_detector dies or
# the camera topic stops publishing.
WATCHDOG_TIMEOUT = 1.0


class NavigationController(Node):
    def __init__(self) -> None:
        super().__init__("navigation_controller")

        self.target_offset = 0.0
        self.target_visible = False
        self.last_message_time = self.get_clock().now()

        self.cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel", 1)

        self.create_subscription(Float32, "/target_offset", self.offset_callback, 1)
        self.create_subscription(Bool, "/target_visible", self.visible_callback, 1)

        self.declare_parameter("control_rate", 10.0)
        control_rate_hz = self.get_parameter("control_rate").get_parameter_value().double_value

        self.timer = self.create_timer(1.0 / control_rate_hz, self.control_loop)

        self.get_logger().info("navigation_controller node started")

    def offset_callback(self, msg: Float32) -> None:
        self.target_offset = msg.data
        self.last_message_time = self.get_clock().now()

    def visible_callback(self, msg: Bool) -> None:
        self.target_visible = msg.data
        self.last_message_time = self.get_clock().now()

    def compute_command(self) -> Twist:
        """Applies the watchdog check, then delegates to the pure
        compute_velocity() function for the actual control decision."""
        twist = Twist()

        time_since_message = (self.get_clock().now() - self.last_message_time).nanoseconds / 1e9
        if time_since_message > WATCHDOG_TIMEOUT:
            return twist  # zero velocity: perception feed is stale

        velocity = compute_velocity(
            target_visible=self.target_visible,
            target_offset=self.target_offset,
        )
        twist.linear.x = velocity.linear_x
        twist.angular.z = velocity.angular_z
        return twist

    def control_loop(self) -> None:
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.twist = self.compute_command()
        self.cmd_pub.publish(stamped)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NavigationController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()