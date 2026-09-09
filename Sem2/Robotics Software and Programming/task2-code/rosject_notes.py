"""
rosject_notes.py

Standalone module exposing the Rosject build/run instructions as a plain
Python string, for pasting into a notebook cell inside the Rosject if
preferred over the README.
"""

rosject_notes = """
Rosject setup and run instructions: semantic_nav_monitor
==========================================================

1. Build the package
   cd ~/ros2_ws
   colcon build --packages-select semantic_nav_monitor
   source install/setup.bash

2. Set the TurtleBot3 model
   export TURTLEBOT3_MODEL=burger

3. Launch the simulation, mission controller and monitor node
   ros2 launch semantic_nav_monitor semantic_nav_monitor.launch.py

4. Start the mission (second shell)
   source ~/ros2_ws/install/setup.bash
   ros2 topic pub --once /hri_command std_msgs/String "data: 'start'"

5. Pause or stop the mission
   ros2 topic pub --once /hri_command std_msgs/String "data: 'pause'"
   ros2 topic pub --once /hri_command std_msgs/String "data: 'stop'"

6. Watch mission state
   ros2 topic echo /mission_state

7. Optional camera view
   ros2 run rqt_image_view rqt_image_view

8. Rename this Rosject to a unique 10-character alphanumeric string and
   quote that name in Section 4.1 of the report.
"""

if __name__ == "__main__":
    print(rosject_notes)
