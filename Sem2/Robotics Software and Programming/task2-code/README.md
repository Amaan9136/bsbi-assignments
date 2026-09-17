# Task 2: Goal-Oriented Autonomous Robot with Monitoring (ROS2)

ROS2 ament_python package implementing a five-state mission
controller and a monitoring node for a TurtleBot3 robot, per the assignment
brief's Task 2 requirements (state machine, sensor integration, obstacle
avoidance/goal navigation, simple HRI, monitoring/logging).

## Use case scenario

**Warehouse Inspection Patrol Robot.** The TurtleBot3 patrols a closed loop
of four checkpoints around a small warehouse bay defined in the custom
Gazebo world `worlds/warehouse_inspection.world`. Eight pallet obstacles (two per patrol leg) sit near
the route, so a normal run forces the robot through every FSM state (not
just NAVIGATE) repeatedly, and the monitor node reports how the mission
went.

## Folder structure

```
task2_ros2_package/
├── README.md                       <- this file
├── README_rosject_notes.md         <- step-by-step run instructions
├── .gitignore
└── semantic_nav_monitor/           <- the actual ROS2 package
    ├── package.xml
    ├── setup.py
    ├── setup.cfg
    ├── resource/semantic_nav_monitor
    ├── worlds/
    │   └── warehouse_inspection.world  <- custom warehouse bay: walls, 2 pallet
    │                                      obstacles, 4 checkpoint markers, dock
    ├── launch/
    │   └── semantic_nav_monitor.launch.py  <- loads the custom world + spawns robot
    └── semantic_nav_monitor/
        ├── __init__.py
        ├── mission_controller.py   <- 5-state FSM, /scan, /odom, /cmd_vel, /hri_command
        ├── monitor_node.py         <- logs /mission_state, metrics from /odom
        └── keyboard_hri_node.py    <- live keyboard HRI: s=start, p=pause, x=stop, q=quit
```

## Running on TheConstruct.ai (Rosject)

Copy the `semantic_nav_monitor/` folder into `~/ros2_ws/src/` inside your
Rosject, then follow `README_rosject_notes.md` (build, export
`TURTLEBOT3_MODEL`, launch, then either run `keyboard_hri_node` or publish
`start`/`pause`/`stop` on `/hri_command` from a second shell).

## Running locally in VS Code (Linux, with ROS2 installed)

ROS2 + Gazebo + TurtleBot3 requires a Linux environment (native Ubuntu 22.04,
WSL2, or a dev container) - it will not run natively on Windows/macOS. If you
have ROS2 and the TurtleBot3 packages installed locally:

1. Open this folder in VS Code (the official **ROS** extension by
   Microsoft is recommended for syntax highlighting and `colcon` tasks).
2. Copy `semantic_nav_monitor/` into your local `~/ros2_ws/src/`.
3. In the VS Code integrated terminal:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select semantic_nav_monitor
   source install/setup.bash
   export TURTLEBOT3_MODEL=burger
   ros2 launch semantic_nav_monitor semantic_nav_monitor.launch.py
   ```
4. From a second VS Code terminal (also sourced), start the mission:
   ```bash
   ros2 topic pub --once /hri_command std_msgs/String "data: 'start'"
   ```

If you don't have ROS2 installed locally, use the Rosject workflow above -
that is the officially supported route for this assignment.

## Package-name reminder

The Rosject itself must be renamed to a unique 10-character alphanumeric
string and that name quoted in Section 4.1 of the report, replacing
`PLACEHOLDER_ROSJECT_NAME`.