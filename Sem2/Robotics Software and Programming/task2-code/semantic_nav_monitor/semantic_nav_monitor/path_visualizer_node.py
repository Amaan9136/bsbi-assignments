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

ABOUT THE /marker SERVICE'S REQUEST/RESPONSE TYPES
----------------------------------------------------
Gazebo's "/marker" service is a Gazebo Transport *oneway* service: the
server (gz-sim's MarkerManager) is advertised with a callback that takes
only a request and never actually sends a reply back
(gz-sim's MarkerManagerPrivate::OnMarkerMsg has no response parameter).
Calling it from the CLI with a reply type it doesn't really send (e.g.
gz.msgs.Boolean, which is what an earlier version of this file used) means
the request is never routed to a matching provider at all - the marker is
silently never drawn, and the CLI reports something like
"did not report success: ''" every single time.

Rather than hardcode a second guess (gz.msgs.Empty, matching how this
service is advertised in the versions we've checked), this node ASKS
Gazebo what request/response types "/marker" actually advertises at
runtime, via `gz service -s /marker -i`, and uses exactly that. This is
deliberately self-healing across Gazebo point releases and also gives a
much clearer failure signal: if `-i` reports no providers at all, the
problem isn't the type names, it's that this node cannot see the Gazebo
GUI's marker service yet (GUI window not open, or a discovery/networking
mismatch between terminals) - both are logged explicitly below instead of
just "did not report success".
"""

import re
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
DEFAULT_MARKER_REQTYPE = "gz.msgs.Marker"
DEFAULT_MARKER_REPTYPE = "gz.msgs.Empty"


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
        # Not discovered yet - discover_marker_types() fills these in lazily
        # on first use (not here in __init__) because this node typically
        # starts at the same time as the Gazebo GUI client, and the
        # "/marker" service isn't advertised until the GUI's 3D Scene has
        # actually finished initialising. Discovering too early would just
        # cache a false "no providers" result forever.
        self.marker_reqtype = None
        self.marker_reptype = None

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
            "action: ADD_MODIFY type: LINE_STRIP visibility: ALL "
            f"id: {MARKER_ID} ns: \"{MARKER_NAMESPACE}\" "
            f"scale: {{x: {MARKER_LINE_WIDTH_M} y: {MARKER_LINE_WIDTH_M} "
            f"z: {MARKER_LINE_WIDTH_M}}} "
            "material: {ambient: {r: 1.0 g: 0.85 b: 0.0 a: 1.0} "
            "diffuse: {r: 1.0 g: 0.85 b: 0.0 a: 1.0}} "
            f"{points}"
        )
        self._publish_marker(proto_text, "draw")

    def clear_marker(self):
        proto_text = (
            f"action: DELETE_MARKER id: {MARKER_ID} ns: \"{MARKER_NAMESPACE}\""
        )
        self._publish_marker(proto_text, "clear")

    def discover_marker_types(self):
        """Query `gz service -s /marker -i` to find the request/response
        type names Gazebo is actually using for this service right now.
        Returns (reqtype, reptype), falling back to the documented defaults
        if discovery fails or nothing is providing the service yet."""
        try:
            result = subprocess.run(
                ["gz", "service", "-s", "/marker", "-i"],
                timeout=GZ_TOPIC_TIMEOUT_S,
                capture_output=True,
                text=True,
            )
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            self.get_logger().warn(
                f"Could not run 'gz service -s /marker -i' to discover the "
                f"marker service's real types ({exc}). Falling back to "
                f"{DEFAULT_MARKER_REQTYPE}/{DEFAULT_MARKER_REPTYPE}."
            )
            return DEFAULT_MARKER_REQTYPE, DEFAULT_MARKER_REPTYPE

        stdout = result.stdout.strip()
        match = re.search(r"[\w.]*://\S+,\s*([\w.]+),\s*([\w.]+)", stdout)
        if match:
            reqtype, reptype = match.group(1), match.group(2)
            self.get_logger().info(
                "Discovered live /marker service types via "
                f"'gz service -i': reqtype={reqtype} reptype={reptype}"
            )
            return reqtype, reptype

        # No provider line found at all - this means the Gazebo Sim GUI
        # client's marker service isn't visible to THIS terminal/process
        # yet. That's a different, more fundamental problem than a wrong
        # type name: either the Gazebo Sim GUI window hasn't finished
        # opening yet, or this node's terminal is on a different Gazebo
        # Transport discovery domain (GZ_PARTITION/GZ_IP) than the running
        # gz sim server+client. Say so explicitly instead of just retrying
        # silently forever.
        self.get_logger().warn(
            "'gz service -s /marker -i' reported no service providers "
            f"(raw output: {stdout!r}). This means Gazebo's marker service "
            "isn't visible from this terminal yet - most likely the Gazebo "
            "Sim GUI window hasn't fully opened, or this node's shell has "
            "different GZ_PARTITION/GZ_IP environment variables than the "
            "terminal(s) running 'gz sim'. Falling back to "
            f"{DEFAULT_MARKER_REQTYPE}/{DEFAULT_MARKER_REPTYPE} for now and "
            "will re-check next time a marker needs to be drawn/cleared."
        )
        return DEFAULT_MARKER_REQTYPE, DEFAULT_MARKER_REPTYPE

    def _publish_marker(self, proto_text, action_label):
        # Re-discover whenever we don't have a confirmed-good type pair yet
        # (see discover_marker_types()'s docstring for why this is lazy
        # rather than done once in __init__).
        if self.marker_reqtype is None or self.marker_reptype is None:
            self.marker_reqtype, self.marker_reptype = self.discover_marker_types()

        cmd = [
            "gz",
            "service",
            "-s",
            "/marker",
            "--reqtype",
            self.marker_reqtype,
            "--reptype",
            self.marker_reptype,
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
                    f"{result.returncode}: "
                    f"{result.stderr.strip() or result.stdout.strip()}. "
                    "Will re-discover the service's types on the next "
                    "draw/clear in case Gazebo wasn't fully up yet."
                )
                # Don't trust the cached types after a failure - the GUI may
                # not have been ready when we first discovered them.
                self.marker_reqtype = None
                self.marker_reptype = None
            else:
                self.get_logger().info(
                    f"Planned-path marker {action_label} sent "
                    f"(reqtype={self.marker_reqtype} "
                    f"reptype={self.marker_reptype})."
                )
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            self.get_logger().warn(
                f"Could not {action_label} planned-path marker via 'gz service': {exc}"
            )
            self.marker_reqtype = None
            self.marker_reptype = None


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