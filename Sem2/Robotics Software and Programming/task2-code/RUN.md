## Apply changes to the ~/ros2_ws workspace
SRC="/mnt/d/0 AMAAN MAIN/Documents/GERMANY/BSBI/BSBI-Assignments/Sem2/Robotics Software and Programming/task2-code"
DST=~/ros2_ws/src/semantic_nav_monitor

cp "$SRC/semantic_nav_monitor/semantic_nav_monitor/mission_controller.py" "$DST/semantic_nav_monitor/mission_controller.py"
cp "$SRC/semantic_nav_monitor/semantic_nav_monitor/keyboard_hri_node.py"  "$DST/semantic_nav_monitor/keyboard_hri_node.py"
cp "$SRC/semantic_nav_monitor/launch/semantic_nav_monitor.launch.py"     "$DST/launch/semantic_nav_monitor.launch.py"
cp "$SRC/semantic_nav_monitor/setup.py"                                   "$DST/setup.py"
cp "$SRC/semantic_nav_monitor/worlds/warehouse_inspection.sdf"   "$DST/worlds/warehouse_inspection.sdf"
______________________

## COMMANDS TO RUN:

### SHELL 1:

### Rebuild and source the package:

cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
rm -rf build/semantic_nav_monitor install/semantic_nav_monitor log
colcon build --symlink-install --packages-select semantic_nav_monitor
source install/setup.bash

### Run
clear
cd ~/ros2_ws
export TURTLEBOT3_MODEL=burger
ros2 launch semantic_nav_monitor semantic_nav_monitor.launch.py
______________________
### SHELL 2:

cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run semantic_nav_monitor keyboard_hri_node

or

which keyboard_hri_node
cd ~/ros2_ws
ros2 run semantic_nav_monitor keyboard_hri_node
______________________

check these topics from another shell:

ros2 topic echo /hricommand
ros2 topic echo /mission_state
ros2 topic echo /odom
ros2 topic echo /scan
ros2 topic echo /cmd_vel

check that the controller is running:
ros2 node list
______________________
## THINGS TO MENTION IN REPORT

## Gazebo screenshots to include in the report (Task 2)
- World overview at launch (IDLE), showing the full bay, 8 pallets, 4 checkpoint markers, green charging dock, and the robot at the dock — your uploaded screenshot is a good example of this one.
- Mid-mission, robot visibly navigating along a leg, mid-transit between checkpoints.
- Robot in the act of avoiding a pallet obstacle (close to one, angled away) — pair with the terminal log showing AVOID_OBSTACLE.
- Robot at the final checkpoint / back at the dock with MISSION_COMPLETE visible in the terminal.
- Terminal screenshot of Shell 1 showing a full state-transition sequence (IDLE→NAVIGATE→AVOID_OBSTACLE→REPLAN→NAVIGATE→MISSION_COMPLETE).
- Terminal screenshot of the final monitor_node summary block (time, distance, obstacle-encounter count, time-per-state).
- Terminal screenshot of Shell 2 (keyboard_hri_node) showing the s/p/x commands being published.

a Cartesian check, not an angular one. also tightened the odometry-based stall fallback so it catches a true block faster (2.0s → 1.0s) rather than relying solely on LIDAR.