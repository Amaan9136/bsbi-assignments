#!/usr/bin/env python3
"""
path_visualizer_node.py
Draws the mission_controller's planned checkpoint route as a visible yellow
line in the Gazebo Sim client window, and lets it be toggled on/off either
from the keyboard ('v' in keyboard_hri_node) or from a real clickable button
in the Gazebo GUI itself (see NATIVE GUI BUTTON below).
WHY THIS DOES NOT USE THE "/marker" SERVICE
----------------------------------------------
An earlier version of this file drew the path using Gazebo's "/marker"
service (a LINE_STRIP gz.msgs.Marker). That service is a Gazebo Transport
*oneway* service that only exists once the GUI client's 3D Scene has been
created, and getting its exact reply type right proved unreliable. Instead,
this spawns the path as REAL, STATIC MODEL ENTITIES (thin yellow boxes, one
per route segment) using the exact same `/world/<world>/create` and
`/world/<world>/remove` services this project's own launch file already
uses (via `ros_gz_sim create`) to spawn the robot itself. These are:
  - Ordinary two-way services with a real gz.msgs.Boolean reply (unlike
    "/marker"), so there is no reply-type guessing involved.
  - Provided by the SERVER itself (the "UserCommands" system, already
    loaded by warehouse_inspection.sdf), not the GUI client - so they exist
    as soon as the simulation starts, with no GUI-initialisation race.
  - Rendered through the exact same pipeline as every pallet/pylon/
    checkpoint marker already visible in the world, which is proof they
    render correctly in this setup.
The path segments are <static>true</static>, so they add no physics cost.
They sit only LINE_Z_M off the floor, well below the 2D LIDAR's single
horizontal scan plane, so they never show up as obstacles even though (to
match the structure of every confirmed-working `/create` example) each one
also has a small <collision> box alongside its <visual>.
NATIVE GUI BUTTON (no compiling anything)
-------------------------------------------
Gazebo Sim ships a built-in "Publisher" GUI plugin (no custom C++ needed)
that shows a text-configurable Topic/Message-type/Message-data box with a
literal "Publish" toggle button. `config/gui_lidar_on.config` and
`config/gui_lidar_off.config` add one of these, titled "Toggle Planned
Path". The Publisher plugin does not support presetting its fields from
the config file, so the very first time you use it you fill in 3 fields
once (see RUN.md) - after that, just click "Publish" to show the path and
un-toggle it to hide it. This node listens for that toggle by running
`gz topic -e -t /gui/show_planned_path` in the background (the plain `gz`
command-line tool, so there's no extra Python package to install) - NOT
via ROS - since the Publisher plugin speaks native Gazebo Transport, not
ROS 2. The keyboard 'v' toggle (via /show_planned_path over ROS) still
works exactly as before; both controls drive the same on/off state.
"""
import math
import subprocess
import threading
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Path
from std_msgs.msg import Bool
WORLD_NAME = "warehouse_inspection"
GUI_TOGGLE_TOPIC = "/gui/show_planned_path"
SEGMENT_NAME_PREFIX = "planned_path_seg_"
LINE_Z_M = 0.02
LINE_WIDTH_M = 0.06
LINE_HEIGHT_M = 0.015
GZ_SERVICE_TIMEOUT_S = 3.0
def _segment_model_sdf(name, x1, y1, x2, y2):
    """Build a one-line SDF snippet for a thin static box spanning
    (x1,y1)->(x2,y2), used as one visible segment of the planned-path line.
    Matches the structure of every confirmed-working `/world/.../create`
    example in Gazebo's own docs (an explicit XML prolog, and a
    <collision> alongside the <visual>) rather than a more minimal
    visual-only version, since that's the pattern consistently used for
    entities spawned at runtime through this service - as opposed to
    models loaded from the world file at startup, which do tolerate
    visual-only links (this world's checkpoint markers are an example of
    that, but they go through a different loading path). The collision
    box costs nothing here: the model is static and sits at LINE_Z_M,
    well below the LIDAR's single horizontal scan plane, so it's never
    detected as an obstacle.
    All XML attribute values use single quotes so the whole snippet can be
    embedded inside a double-quoted protobuf text field with no escaping
    needed.
    """
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1e-6:
        return None
    yaw = math.atan2(y2 - y1, x2 - x1)
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    geometry = (
        f"<geometry><box><size>{length:.4f} {LINE_WIDTH_M} "
        f"{LINE_HEIGHT_M}</size></box></geometry>"
    )
    return (
        "<?xml version='1.0'?>"
        "<sdf version='1.9'>"
        f"<model name='{name}'>"
        "<static>true</static>"
        f"<pose>{mx:.4f} {my:.4f} {LINE_Z_M} 0 0 {yaw:.6f}</pose>"
        "<link name='link'>"
        f"<visual name='visual'>{geometry}"
        "<material>"
        "<ambient>1 0.85 0 1</ambient>"
        "<diffuse>1 0.85 0 1</diffuse>"
        "<emissive>0.6 0.5 0 1</emissive>"
        "</material>"
        "</visual>"
        f"<collision name='collision'>{geometry}</collision>"
        "</link>"
        "</model>"
        "</sdf>"
    )
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
            Bool, "/show_planned_path", self.ros_show_callback, 10
        )
        self.latest_path = None
        self.marker_visible = False
        self.spawned_segment_names = []
        self._lock = threading.Lock()
        # NOTE: earlier versions of this node tried to defensively remove a
        # block of possible leftover segment names at startup, "just in
        # case" a previous crashed run left some behind. That produced a
        # guaranteed [Err] "Entity ... not found, so not removed" line from
        # Gazebo's own server log on every single normal launch (since on a
        # clean start nothing exists yet to remove) - harmless noise, but
        # confusing and easy to mistake for a real problem. It's removed:
        # a full relaunch already restarts gzserver from scratch (nothing
        # can persist across that), and within one running session this
        # node's own destroy_node() cleans up its own spawned segments.
        self._gui_toggle_thread = threading.Thread(
            target=self._gui_toggle_listener_loop, daemon=True
        )
        self._gui_toggle_thread.start()
        self.get_logger().info(
            "path_visualizer_node ready. Waiting for /planned_path. Toggle "
            "the route line with 'v' in keyboard_hri_node, or with the "
            "'Toggle Planned Path' Publisher panel in the Gazebo GUI "
            "(see RUN.md for the one-time setup)."
        )
    # ---------------------------------------------------------------
    # Inputs: ROS path + two independent show/hide sources
    # ---------------------------------------------------------------
    def path_callback(self, msg: Path):
        with self._lock:
            self.latest_path = msg
            was_visible = self.marker_visible
        if was_visible:
            self._redraw()
    def ros_show_callback(self, msg: Bool):
        self.set_visibility(msg.data, source="keyboard 'v'")
    def _gui_toggle_listener_loop(self):
        """Subscribe to the native Gazebo Transport toggle topic by running
        `gz topic -e` in the background - plain CLI tool, no extra Python
        package required. Retries if Gazebo isn't up yet or restarts."""
        while rclpy.ok():
            try:
                proc = subprocess.Popen(
                    ["gz", "topic", "-e", "-t", GUI_TOGGLE_TOPIC],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
            except FileNotFoundError:
                self.get_logger().warn(
                    "'gz' executable not found - the native Gazebo GUI "
                    "toggle button for the path won't be available (the "
                    "keyboard 'v' toggle still works)."
                )
                return
            self.get_logger().info(
                f"Listening for the native Gazebo GUI toggle on "
                f"'{GUI_TOGGLE_TOPIC}' (the 'Toggle Planned Path' Publisher "
                "panel in the Gazebo window publishes here)."
            )
            try:
                for line in proc.stdout:
                    text = line.strip().lower()
                    if "true" in text:
                        self.set_visibility(True, source="GUI button")
                    elif "false" in text:
                        self.set_visibility(False, source="GUI button")
            except Exception as exc:  # pragma: no cover - defensive
                self.get_logger().warn(f"GUI toggle listener error: {exc}")
            finally:
                proc.terminate()
            # gz topic echo can exit if Gazebo restarts - wait and reconnect.
            time.sleep(2.0)
    def set_visibility(self, visible, source=""):
        with self._lock:
            if visible == self.marker_visible:
                return
            self.marker_visible = visible
            path = self.latest_path
        if not visible:
            self.get_logger().info(f"Hiding planned path ({source}).")
            self._clear_all_segments()
            return
        if path is None:
            self.get_logger().warn(
                f"Path visibility toggled on ({source}) but no /planned_path "
                "received yet - press 's' to start the mission first."
            )
            return
        self.get_logger().info(f"Showing planned path ({source}).")
        self._draw_segments(path)
    def _redraw(self):
        with self._lock:
            path = self.latest_path
        self._clear_all_segments()
        if path is not None:
            self._draw_segments(path)
    # ---------------------------------------------------------------
    # Drawing: spawn/remove real static model entities via EntityFactory
    # ---------------------------------------------------------------
    def _draw_segments(self, path: Path):
        if len(path.poses) < 2:
            self.get_logger().warn(
                f"/planned_path only has {len(path.poses)} pose(s) - need "
                "at least 2 to draw a line. Nothing drawn."
            )
            return
        for i in range(len(path.poses) - 1):
            p1 = path.poses[i].pose.position
            p2 = path.poses[i + 1].pose.position
            name = f"{SEGMENT_NAME_PREFIX}{i}"
            sdf = _segment_model_sdf(name, p1.x, p1.y, p2.x, p2.y)
            if sdf is None:
                continue
            if self._spawn_entity(name, sdf):
                self.spawned_segment_names.append(name)
        self.get_logger().info(
            f"Planned path drawn: {len(self.spawned_segment_names)} of "
            f"{len(path.poses) - 1} segment(s) spawned successfully."
        )
    def _clear_all_segments(self):
        for name in self.spawned_segment_names:
            self._remove_entity(name)
        self.spawned_segment_names = []
    def _spawn_entity(self, name, sdf_text):
        proto_text = f'sdf: "{sdf_text}" name: "{name}" allow_renaming: false'
        cmd = [
            "gz", "service", "-s", f"/world/{WORLD_NAME}/create",
            "--reqtype", "gz.msgs.EntityFactory",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "3000",
            "--req", proto_text,
        ]
        try:
            result = subprocess.run(
                cmd, timeout=GZ_SERVICE_TIMEOUT_S, capture_output=True, text=True
            )
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            self.get_logger().warn(f"Could not spawn path segment '{name}': {exc}")
            return False
        if result.returncode != 0:
            self.get_logger().warn(
                f"Spawning path segment '{name}' exited {result.returncode}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
            return False
        if "true" not in result.stdout.lower():
            self.get_logger().warn(
                f"Spawning path segment '{name}' returned "
                f"{result.stdout.strip()!r} (expected 'data: true'). It may "
                "already exist from an earlier draw in this session - try "
                "toggling the path off and back on."
            )
            return False
        self.get_logger().debug(f"Spawned path segment '{name}'.")
        return True
    def _remove_entity(self, name):
        proto_text = f'name: "{name}" type: MODEL'
        cmd = [
            "gz", "service", "-s", f"/world/{WORLD_NAME}/remove",
            "--reqtype", "gz.msgs.Entity",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "3000",
            "--req", proto_text,
        ]
        try:
            subprocess.run(
                cmd, timeout=GZ_SERVICE_TIMEOUT_S, capture_output=True, text=True
            )
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            self.get_logger().warn(f"Could not remove path segment '{name}': {exc}")
    def destroy_node(self):
        self._clear_all_segments()
        super().destroy_node()
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