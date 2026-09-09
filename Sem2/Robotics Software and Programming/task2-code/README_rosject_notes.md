# Rosject notes: semantic_nav_monitor

Copy the `semantic_nav_monitor` folder into the `~/ros2_ws/src/` directory of
your Rosject on TheConstruct.ai, then run the following commands from the
Rosject's shell (Shell #1).

## 1. Build the package

```bash
cd ~/ros2_ws
colcon build --packages-select semantic_nav_monitor
source install/setup.bash
```

## 2. Set the TurtleBot3 model

```bash
export TURTLEBOT3_MODEL=burger
```

## 3. Launch the simulation, mission controller and monitor node

```bash
ros2 launch semantic_nav_monitor semantic_nav_monitor.launch.py
```

This brings up the TurtleBot3 Gazebo world together with the
`mission_controller` and `monitor_node` nodes. The robot starts in the
`IDLE` state and will not move until it receives a start command.

## 4. Start the mission (from a second shell, Shell #2)

```bash
source ~/ros2_ws/install/setup.bash
ros2 topic pub --once /hri_command std_msgs/String "data: 'start'"
```

## 5. Pause or stop the mission at any time

```bash
ros2 topic pub --once /hri_command std_msgs/String "data: 'pause'"
ros2 topic pub --once /hri_command std_msgs/String "data: 'stop'"
```

## 6. Watch the mission state and monitoring output

```bash
ros2 topic echo /mission_state
```

The `monitor_node` also prints state transitions and periodic mission
summaries (total time, distance travelled, time per state) directly to its
own console output in Shell #1.

## 7. Inspect the camera feed (optional, if a camera is enabled on the model)

```bash
ros2 run rqt_image_view rqt_image_view
```

## 8. Rosject naming requirement

Per the assignment brief, the Rosject itself must be renamed to a unique
10-character alphanumeric string (for example, generated via random.org) and
that name must be quoted in Section 4.1 of the accompanying report, replacing
the `PLACEHOLDER_ROSJECT_NAME` marker.

---

The `rosject_notes` string below is provided for convenience, in case the
same content is wanted directly inside a Python file or notebook comment
inside the Rosject.
