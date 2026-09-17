# Rosject notes: semantic_nav_monitor

**Use case:** Warehouse Inspection Patrol Robot - the TurtleBot3 patrols
four checkpoints in a custom warehouse bay (`worlds/warehouse_inspection.world`),
avoiding two pallet obstacles placed on its route, while a monitor node logs
its state transitions and performance metrics.

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

## 3. Launch the custom world, mission controller and monitor node

```bash
ros2 launch semantic_nav_monitor semantic_nav_monitor.launch.py
```

This loads the custom `warehouse_inspection.world` (walls, two pallet
obstacles, four coloured checkpoint markers, a green charging-dock marker),
spawns the TurtleBot3 at the dock (0,0), and starts the
`mission_controller` and `monitor_node` nodes. The robot starts in the
`IDLE` state and will not move until it receives a start command.

## 4. Give the robot commands (from a second shell, Shell #2)

Preferred (literal keyboard HRI, no Enter key needed):

```bash
source ~/ros2_ws/install/setup.bash
ros2 run semantic_nav_monitor keyboard_hri_node
```

Then press:
- `s` to start the mission
- `p` to pause it
- `x` to stop it (mission is marked complete)
- `q` to quit the keyboard node (does not stop the mission)

Alternative (single one-shot commands, no live keypresses):

```bash
source ~/ros2_ws/install/setup.bash
ros2 topic pub --once /hri_command std_msgs/String "data: 'start'"
ros2 topic pub --once /hri_command std_msgs/String "data: 'pause'"
ros2 topic pub --once /hri_command std_msgs/String "data: 'stop'"
```

## 5. Watch the mission state and monitoring output

```bash
ros2 topic echo /mission_state
```

The `monitor_node` also prints state transitions and periodic mission
summaries (total time, distance travelled, obstacle-encounter count, time
per state) directly to its own console output in Shell #1.

## 6. Inspect the camera feed (optional, if a camera is enabled on the model)

```bash
ros2 run rqt_image_view rqt_image_view
```

## 7. What to capture for the report

- A Gazebo screenshot showing the warehouse bay, checkpoints, obstacles and
  the robot mid-patrol.
- Terminal screenshots of `mission_controller`/`monitor_node` output showing
  at least one full IDLE -> NAVIGATE -> AVOID_OBSTACLE -> REPLAN ->
  NAVIGATE -> MISSION_COMPLETE cycle.
- The final `monitor_node` summary block (total time, distance, obstacle
  encounters, time-per-state).

## 8. Rosject naming requirement

Per the assignment brief, the Rosject itself must be renamed to a unique
10-character alphanumeric string (for example, generated via random.org) and
that name must be quoted in Section 4.1 of the accompanying report, replacing
the `PLACEHOLDER_ROSJECT_NAME` marker.

---

The `rosject_notes` string below is provided for convenience, in case the
same content is wanted directly inside a Python file or notebook comment
inside the Rosject.
