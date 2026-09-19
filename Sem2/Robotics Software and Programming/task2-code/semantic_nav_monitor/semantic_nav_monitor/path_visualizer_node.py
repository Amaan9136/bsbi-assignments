#!/usr/bin/env python3
"""
path_visualizer_node.py

Draws the mission_controller's planned checkpoint route as a visible line in
the Gazebo Sim client window. Gazebo Sim has no built-in "Path" GUI plugin,
so this uses its Marker service instead: a LINE_STRIP gz.msgs.Marker sent
via `gz service -s /marker`, the Gazebo Sim interface for adding/removing
markers.

Subscribes to /planned_path (TRANSIENT_LOCAL, matching mission_controller's
latched publisher) to learn the route once, and to /show_planned_path
(std_msgs/Bool, toggled by pressing 'v' in keyboard_hri_node) to know
whether the line should currently be visible. Toggling this only adds or
removes the marker - it never touches /planned_path or mission_controller's
actual behaviour.
"""

import subprocess

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Path
from std_msgs.msg import Bool

MARKER_NAMESPACE = "semantic_nav_monitor"
MARKER_ID = 1
MARKER_Z_M = 0.05
MARKER_LINE_WIDTH_M = 0.05
GZ_TOPIC_TIMEOUT_S = 3.0


class PathVisualizerNode(Node):
    def __init__(self):
        super().__init__("path_visualizer_node")

        path_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.path_sub = self.create_subscription(
            Path, "/planned_path", self.path_callback, path_qos
        )
        self.show_sub = self.create_subscription(
            Bool, "/show_planned_path", self.show_callback, 10
        )

        self.latest_path = None
        self.marker_visible = False

        self.get_logger().info(
            "path_visualizer_node ready. Waiting for /planned_path and "
            "/show_planned_path (toggled by 'v' in keyboard_hri_node)."
        )

    def path_callback(self, msg: Path):
        self.latest_path = msg
        if self.marker_visible:
            self.draw_marker(msg)

    def show_callback(self, msg: Bool):
        self.marker_visible = msg.data
        if not self.marker_visible:
            self.clear_marker()
            return

        if self.latest_path is None:
            self.get_logger().warn(
                "'v' pressed but no /planned_path received yet - press 's' "
                "to start the mission first, then 'v' again."
            )
            return

        self.draw_marker(self.latest_path)

    def draw_marker(self, path: Path):
        if len(path.poses) < 2:
            return

        points = "".join(
            f"point: {{x: {pose.pose.position.x} y: {pose.pose.position.y} "
            f"z: {MARKER_Z_M}}} "
            for pose in path.poses
        )
        proto_text = (
            "action: ADD_MODIFY type: LINE_STRIP "
            f"id: {MARKER_ID} ns: \"{MARKER_NAMESPACE}\" "
            f"scale: {{x: {MARKER_LINE_WIDTH_M} y: {MARKER_LINE_WIDTH_M} "
            f"z: {MARKER_LINE_WIDTH_M}}} "
            "material: {ambient: {r: 1.0 g: 0.55 b: 0.0 a: 1.0} "
            "diffuse: {r: 1.0 g: 0.55 b: 0.0 a: 1.0}} "
            f"{points}"
        )
        self._publish_marker(proto_text, "draw")

    def clear_marker(self):
        proto_text = (
            f"action: DELETE_MARKER id: {MARKER_ID} ns: \"{MARKER_NAMESPACE}\""
        )
        self._publish_marker(proto_text, "clear")

    def _publish_marker(self, proto_text, action_label):
        cmd = [
            "gz",
            "service",
            "-s",
            "/marker",
            "--reqtype",
            "gz.msgs.Marker",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "2000",
            "--req",
            proto_text,
        ]
        try:
            result = subprocess.run(
                cmd, timeout=GZ_TOPIC_TIMEOUT_S, capture_output=True, text=True
            )
            if result.returncode != 0:
                self.get_logger().warn(
                    f"'gz service -s /marker' ({action_label}) exited "
                    f"{result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
                )
            elif "true" not in result.stdout.lower():
                self.get_logger().warn(
                    f"'gz service -s /marker' ({action_label}) did not report "
                    f"success: {result.stdout.strip()!r}"
                )
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            self.get_logger().warn(
                f"Could not {action_label} planned-path marker via 'gz service': {exc}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = PathVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
