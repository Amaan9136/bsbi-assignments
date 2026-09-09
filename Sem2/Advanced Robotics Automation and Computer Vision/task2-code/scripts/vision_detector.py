#!/usr/bin/env python3
"""
vision_detector.py

ROS node wrapper around vision_core.detect_marker(). Subscribes to the
robot's RGB camera topic, converts each frame via cv_bridge, calls the
pure detection logic in vision_core.py, and publishes the result. All
image-processing logic lives in vision_core.py so it can be tested and
debugged independently of ROS; this file only handles ROS plumbing.
"""

from __future__ import annotations

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32

from vision_core import MIN_CONTOUR_AREA, detect_marker


class VisionDetector:
    def __init__(self) -> None:
        rospy.init_node("vision_detector", anonymous=False)

        self.bridge = CvBridge()
        camera_topic = rospy.get_param("~camera_topic", "/camera/rgb/image_raw")
        self.min_area = rospy.get_param("~min_contour_area", MIN_CONTOUR_AREA)

        self.offset_pub = rospy.Publisher("/target_offset", Float32, queue_size=1)
        self.visible_pub = rospy.Publisher("/target_visible", Bool, queue_size=1)
        self.debug_pub = rospy.Publisher("/vision_detector/debug_image", Image, queue_size=1)

        self.image_sub = rospy.Subscriber(camera_topic, Image, self.image_callback, queue_size=1)

        rospy.loginfo("vision_detector node started, subscribing to %s", camera_topic)

    def image_callback(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr("cv_bridge conversion failed: %s", exc)
            return

        result = detect_marker(cv_image, min_area=self.min_area)

        self.offset_pub.publish(Float32(result.offset))
        self.visible_pub.publish(Bool(result.visible))

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(result.debug_frame, encoding="bgr8")
            self.debug_pub.publish(debug_msg)
        except CvBridgeError as exc:
            rospy.logwarn("Failed to publish debug image: %s", exc)


if __name__ == "__main__":
    try:
        VisionDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
