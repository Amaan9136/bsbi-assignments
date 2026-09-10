# turtlebot3_vision_nav

Vision guided autonomous navigation for a simulated TurtleBot3, built for
TheConstruct.ai. The robot uses HSV colour thresholding on its RGB camera
feed to detect a coloured marker, then drives toward it using a simple
rule based controller.

**ROSject name: vVGb1L3fxD**

Generate this string yourself (for example using random.org's string
generator, 10 characters, alphanumeric) before creating the ROSject, and
record it both here and in the final report, as required by the
assignment brief.

## Package structure

```
turtlebot3_vision_nav/
├── CMakeLists.txt
├── package.xml
├── launch/
│   └── vision_nav.launch
└── scripts/
    ├── vision_core.py            # pure detection logic, no ROS dependency
    ├── control_core.py           # pure control logic, no ROS dependency
    ├── vision_detector.py        # ROS node: wraps vision_core
    └── navigation_controller.py  # ROS node: wraps control_core
```

The perception and control logic is deliberately split from the ROS
plumbing. `vision_core.py` and `control_core.py` contain plain Python
functions with no `rospy` import, only `opencv-python` and `numpy` as
dependencies. This means:

- **You can develop and debug the actual algorithm in plain VS Code**,
  with no ROS installation, no Gazebo, and no TheConstruct.ai session
  running, before ever touching the simulator.
- The ROS node files (`vision_detector.py`, `navigation_controller.py`)
  are thin: they only handle subscribing, publishing and timing, and
  delegate every actual decision to the core modules.

### Developing the core logic locally in VS Code

1. Open the `turtlebot3_vision_nav/` folder in VS Code.
2. Create a virtual environment and install the two non-ROS dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install opencv-python numpy
   ```
3. Run either core module directly; each has a small built-in smoke test
   under `if __name__ == "__main__":` that exercises it with synthetic
   input, no camera or ROS topic required:
   ```
   python scripts/vision_core.py
   python scripts/control_core.py
   ```
4. Edit the thresholds or control constants at the top of each file
   (`LOWER_RED_1`, `CENTER_THRESHOLD`, `MAX_LINEAR_SPEED`, etc.), rerun
   the smoke test, and iterate, all locally, before copying the updated
   `scripts/` folder into your ROSject on TheConstruct.ai to test against
   the real simulated camera and robot.

## What this package does

- `scripts/vision_detector.py` (ROS node) subscribes to the robot's
  camera topic, calls `vision_core.detect_marker()` on each frame to
  locate a target coloured marker via HSV thresholding, and publishes:
  - `/target_offset` (`std_msgs/Float32`): normalised offset in [-1, 1]
  - `/target_visible` (`std_msgs/Bool`): whether the marker is currently detected
  - `/vision_detector/debug_image` (`sensor_msgs/Image`): annotated frame for
    visual debugging in `rqt_image_view`

- `scripts/navigation_controller.py` (ROS node) subscribes to those two
  topics, calls `control_core.compute_velocity()` to decide what to do,
  and publishes `geometry_msgs/Twist` messages on `/cmd_vel`:
  - if no marker is visible, the robot rotates slowly to search
  - if the marker is visible but off-centre, the robot rotates toward it
  - if the marker is visible and centred, the robot drives forward
  - if no perception message has arrived recently (watchdog timeout), the
    robot stops rather than act on stale data

## Running this in app.theconstruct.ai

1. Open the ROSject in TheConstruct.ai web IDE.
2. Open a shell and build the workspace:
   ```
   cd ~/catkin_ws
   catkin_make
   source devel/setup.bash
   ```
3. Launch the simulation and both nodes together:
   ```
   roslaunch turtlebot3_vision_nav vision_nav.launch
   ```
   This starts the TurtleBot3 Gazebo world and both the perception and
   navigation nodes in one command.
4. Open Gazebo from the Simulations menu to view the robot and world.

## Inspecting topics and camera images

In a second shell:
```
rostopic list
rostopic echo /target_offset
rostopic echo /target_visible
```

To visually confirm what the perception node is detecting:
```
rqt_image_view
```
then select `/vision_detector/debug_image` from the topic dropdown. The
debug view draws the detected contour, its centroid, and the frame's
vertical centre line, which makes it straightforward to judge whether the
HSV thresholds are correctly tuned for the marker colour and the world's
lighting.

## Placing the coloured marker

Add a coloured cylinder or box model to the Gazebo world (Insert panel in
Gazebo, or edit the world file directly) and place it in the robot's field
of view at varying distances and lateral positions across trials. The
default HSV thresholds in `vision_detector.py` are tuned for a red marker;
if a different coloured marker is used, the `LOWER_RED_*` / `UPPER_RED_*`
constants at the top of that file must be updated to match, and it is
worth renaming them accordingly for clarity.

## Suggested test scenarios

Record observations and screenshots (from both Gazebo and
`rqt_image_view`) for each of the following, for inclusion in the
Execution and Testing subsection of the report:

- Marker directly ahead of the robot, at a moderate distance
- Marker offset to the left edge of the frame
- Marker offset to the right edge of the frame
- Marker close to the robot versus far from the robot
- No marker present, to confirm the search behaviour engages correctly
- A second, differently coloured object present in the frame alongside the
  marker, to check the detector does not falsely trigger on it

For each scenario, note the values published on `/target_offset` and
`/target_visible`, and describe the resulting robot behaviour observed in
Gazebo.
