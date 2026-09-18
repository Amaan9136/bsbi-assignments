tasks:

fix: devcontainers@Amaan-Ideapad-3:~/ros2_ws$ ros2 launch semantic_nav_monitor semantic_nav_monitor.launch.py
[INFO] [launch]: All log files can be found below /home/devcontainers/.ros/log/2026-09-18-12-54-59-240730-Amaan-Ideapad-3-4856
[INFO] [launch]: Default logging verbosity is set to INFO
[ERROR] [launch]: Caught exception in launch (see debug for traceback): Caught multiple exceptions when trying to load file of format [py]:
 - NameError: name 'CAMERA_BRIDGE_TOPIC' is not defined
 - InvalidFrontendLaunchFileError: The launch file may have a syntax error, or its format is unknown
devcontainers@Amaan-Ideapad-3:~/ros2_ws$ 

i want to properly show the image display filename="ImageDisplay" name="Image Display so fix it to since its not visible in the gazebo sim. i have shared an image of that