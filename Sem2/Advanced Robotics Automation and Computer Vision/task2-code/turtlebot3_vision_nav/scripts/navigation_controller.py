#!/usr/bin/env python3
"""
navigation_controller.py

ROS node wrapper around control_core.compute_velocity(). Subscribes to
/target_offset and /target_visible, calls the pure control logic in
control_core.py, and publishes the result as a geometry_msgs/Twist on
/cmd_vel. All decision logic lives in control_core.py so it can be
tested and debugged independently of ROS; this file only handles ROS
plumbing, timing and the safety watchdog.
"""

from __future__ import annotations

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32

from control_core import compute_velocity

# If no perception message arrives within this many seconds, stop the
# robot rather than act on stale data, e.g. if vision_detector dies or
# the camera topic stops publishing.
WATCHDOG_TIMEOUT = 1.0


class NavigationController:
    def __init__(self) -> None:
        rospy.init_node("navigation_controller", anonymous=False)

        self.target_offset = 0.0
        self.target_visible = False
        self.last_message_time = rospy.Time.now()

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        rospy.Subscriber("/target_offset", Float32, self.offset_callback, queue_size=1)
        rospy.Subscriber("/target_visible", Bool, self.visible_callback, queue_size=1)

        control_rate_hz = rospy.get_param("~control_rate", 10)
        self.rate = rospy.Rate(control_rate_hz)

        rospy.loginfo("navigation_controller node started")

    def offset_callback(self, msg: Float32) -> None:
        self.target_offset = msg.data
        self.last_message_time = rospy.Time.now()

    def visible_callback(self, msg: Bool) -> None:
        self.target_visible = msg.data
        self.last_message_time = rospy.Time.now()

    def compute_command(self) -> Twist:
        """Applies the watchdog check, then delegates to the pure
        compute_velocity() function for the actual control decision."""
        twist = Twist()

        time_since_message = (rospy.Time.now() - self.last_message_time).to_sec()
        if time_since_message > WATCHDOG_TIMEOUT:
            return twist  # zero velocity: perception feed is stale

        velocity = compute_velocity(
            target_visible=self.target_visible,
            target_offset=self.target_offset,
        )
        twist.linear.x = velocity.linear_x
        twist.angular.z = velocity.angular_z
        return twist

    def run(self) -> None:
        while not rospy.is_shutdown():
            self.cmd_pub.publish(self.compute_command())
            self.rate.sleep()


if __name__ == "__main__":
    try:
        NavigationController().run()
    except rospy.ROSInterruptException:
        pass
