#!/usr/bin/env python3
"""
vision_detector.py

ROS 2 node wrapper around vision_core.detect_marker(). Subscribes to
the robot's RGB camera topic, converts each frame via cv_bridge, calls
the pure detection logic in vision_core.py, and publishes the result.
All image-processing logic lives in vision_core.py so it can be tested
and debugged independently of ROS; this file only handles ROS plumbing.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32

from turtlebot3_vision_nav.vision_core import MIN_CONTOUR_AREA, detect_marker


class VisionDetector(Node):
    def __init__(self) -> None:
        super().__init__("vision_detector")

        self.bridge = CvBridge()
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("min_contour_area", float(MIN_CONTOUR_AREA))

        camera_topic = self.get_parameter("camera_topic").get_parameter_value().string_value
        self.min_area = self.get_parameter("min_contour_area").get_parameter_value().double_value

        self.offset_pub = self.create_publisher(Float32, "/target_offset", 1)
        self.visible_pub = self.create_publisher(Bool, "/target_visible", 1)
        self.debug_pub = self.create_publisher(Image, "/vision_detector/debug_image", 1)

        self.image_sub = self.create_subscription(Image, camera_topic, self.image_callback, 1)

        self.get_logger().info(f"vision_detector node started, subscribing to {camera_topic}")

    def image_callback(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().error(f"cv_bridge conversion failed: {exc}")
            return

        result = detect_marker(cv_image, min_area=self.min_area)

        self.offset_pub.publish(Float32(data=result.offset))
        self.visible_pub.publish(Bool(data=result.visible))

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(result.debug_frame, encoding="bgr8")
            self.debug_pub.publish(debug_msg)
        except CvBridgeError as exc:
            self.get_logger().warning(f"Failed to publish debug image: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()