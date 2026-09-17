SHELL 1:

cd ~/ros2_ws
ros2 launch semantic_nav_monitor semantic_nav_monitor.launch.py
______________________
SHELL 2:

cd ~/ros2_ws
source ~/ros2_ws/install/setup.bash
ros2 run semantic_nav_monitor keyboard_hri_node
______________________
not sure if need more shells to complete the goal. add if needed