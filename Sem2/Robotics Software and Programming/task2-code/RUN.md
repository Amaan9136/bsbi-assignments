## COMMANDS TO RUN:

SHELL 1:

cd ~/ros2_ws
colcon build --packages-select semantic_nav_monitor
source install/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch semantic_nav_monitor semantic_nav_monitor.launch.py
______________________
SHELL 2:

cd ~/ros2_ws
source ~/ros2_ws/install/setup.bash
ros2 run semantic_nav_monitor keyboard_hri_node
______________________

## Gazebo screenshots to include in the report (Task 2)
- World overview at launch (IDLE), showing the full bay, 8 pallets, 4 checkpoint markers, green charging dock, and the robot at the dock — your uploaded screenshot is a good example of this one.
- Mid-mission, robot visibly navigating along a leg, mid-transit between checkpoints.
- Robot in the act of avoiding a pallet obstacle (close to one, angled away) — pair with the terminal log showing AVOID_OBSTACLE.
- Robot at the final checkpoint / back at the dock with MISSION_COMPLETE visible in the terminal.
- Terminal screenshot of Shell 1 showing a full state-transition sequence (IDLE→NAVIGATE→AVOID_OBSTACLE→REPLAN→NAVIGATE→MISSION_COMPLETE).
- Terminal screenshot of the final monitor_node summary block (time, distance, obstacle-encounter count, time-per-state).
- Terminal screenshot of Shell 2 (keyboard_hri_node) showing the s/p/x commands being published.