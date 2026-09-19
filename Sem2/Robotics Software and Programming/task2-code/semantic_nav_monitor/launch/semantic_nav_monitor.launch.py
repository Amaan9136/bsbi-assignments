#!/usr/bin/env python3
"""
semantic_nav_monitor.launch.py

Launches the custom "warehouse_inspection" Gazebo world bundled with this
package, spawns a TurtleBot3 into it using turtlebot3_gazebo's launch files,
and starts the mission_controller, monitor_node and path_visualizer_node
nodes (the last one draws/clears the planned-path line in the Gazebo Sim
client when 'v' is pressed in keyboard_hri_node).

USE CASE (Task 2): Warehouse Inspection Patrol Robot.

NOTE (Jazzy port): the turtlebot3_gazebo launch files already start their
own ros_gz_bridge (odom/scan/imu/tf/clock/cmd_vel, with cmd_vel as
TwistStamped). We do NOT start a second bridge covering those same topics
here - a second bridge subscribing to /cmd_vel as plain Twist was the
actual cause of the robot not moving (type mismatch meant the
authoritative bridge never received mission_controller's velocity
commands). The optional camera sensor is enabled by switching
TURTLEBOT3_MODEL to "burger_cam".

NOTE (split server/GUI vs. combined process): by default this launches
Gazebo as two separate processes - a headless server (`-s`) and a GUI
client (`-g`) - which is the standard ros_gz_sim pattern. Some Gazebo GUI
plugins that look up an entity's live pose to position themselves (the
"Visualize Lidar" ray fan is one of these) rely on the GUI client staying
continuously in sync with the server's ECM over the network transport, and
can occasionally latch onto a stale/initial pose (typically wherever the
robot spawned) if that sync hiccups - the lidar rays then look "stuck" at
the spawn point even though the robot body itself keeps moving correctly
and the mission logic is unaffected. If you see that, try the
`combined_gz_process:=true` launch argument below, which runs the server
and GUI as a single `gz sim` process instead (no split, no cross-process
scene sync) - this is a legitimate, supported way to run Gazebo Sim and is
worth trying specifically as a workaround for that plugin's pose lookups.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg_semantic_nav_monitor = get_package_share_directory("semantic_nav_monitor")
    pkg_turtlebot3_gazebo = get_package_share_directory("turtlebot3_gazebo")
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")

    default_world = os.path.join(
        pkg_semantic_nav_monitor, "worlds", "warehouse_inspection.sdf"
    )
    gui_config_lidar_on = os.path.join(
        pkg_semantic_nav_monitor, "config", "gui_lidar_on.config"
    )
    gui_config_lidar_off = os.path.join(
        pkg_semantic_nav_monitor, "config", "gui_lidar_off.config"
    )

    world_arg = DeclareLaunchArgument(
        "world",
        default_value=default_world,
        description="Full path to the Gazebo world file to load.",
    )
    x_pose_arg = DeclareLaunchArgument("x_pose", default_value="0.0")
    y_pose_arg = DeclareLaunchArgument("y_pose", default_value="0.0")
    show_lidar_arg = DeclareLaunchArgument(
        "show_lidar",
        default_value="true",
        description="Show the LIDAR scan (Visualize Lidar GUI plugin) in the Gazebo client.",
    )
    combined_gz_process_arg = DeclareLaunchArgument(
        "combined_gz_process",
        default_value="false",
        description=(
            "Run Gazebo server+GUI as ONE 'gz sim' process instead of the "
            "default split server(-s)/client(-g) processes. Try 'true' if "
            "GUI-side visuals (e.g. the Visualize Lidar rays) appear stuck "
            "at the robot's spawn pose instead of tracking it."
        ),
    )

    turtlebot3_model_env = SetEnvironmentVariable(
        name="TURTLEBOT3_MODEL",
        value=os.environ.get("TURTLEBOT3_MODEL", "burger_cam"),
    )

    gui_config_path = PythonExpression([
        "'", gui_config_lidar_on, "' if '",
        LaunchConfiguration("show_lidar"),
        "' == 'true' else '", gui_config_lidar_off, "'",
    ])

    # Default path: split server(-s)/client(-g) processes, the standard
    # ros_gz_sim pattern.
    gz_sim_server_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": [
                "-r -s -v2 ",
                LaunchConfiguration("world"),
            ],
            "on_exit_shutdown": "true",
        }.items(),
        condition=UnlessCondition(LaunchConfiguration("combined_gz_process")),
    )

    gz_sim_client_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": ["-g -v2 --gui-config ", gui_config_path],
        }.items(),
        condition=UnlessCondition(LaunchConfiguration("combined_gz_process")),
    )

    # Workaround path: one combined 'gz sim' process (server+GUI together).
    # See the "combined_gz_process" argument/docstring note above for why
    # you'd want this - it removes the cross-process GUI/server scene sync
    # that some GUI plugins (e.g. Visualize Lidar) can occasionally lose
    # track of.
    gz_sim_combined_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": [
                "-r -v2 --gui-config ",
                gui_config_path,
                " ",
                LaunchConfiguration("world"),
            ],
            "on_exit_shutdown": "true",
        }.items(),
        condition=IfCondition(LaunchConfiguration("combined_gz_process")),
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_turtlebot3_gazebo, "launch", "robot_state_publisher.launch.py"
            )
        ),
        launch_arguments={"use_sim_time": "true"}.items(),
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_turtlebot3_gazebo, "launch", "spawn_turtlebot3.launch.py")
        ),
        launch_arguments={
            "x_pose": LaunchConfiguration("x_pose"),
            "y_pose": LaunchConfiguration("y_pose"),
        }.items(),
    )

    mission_controller_node = Node(
        package="semantic_nav_monitor",
        executable="mission_controller",
        name="mission_controller",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    monitor_node = Node(
        package="semantic_nav_monitor",
        executable="monitor_node",
        name="monitor_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    path_visualizer_node = Node(
        package="semantic_nav_monitor",
        executable="path_visualizer_node",
        name="path_visualizer_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription([
        world_arg,
        x_pose_arg,
        y_pose_arg,
        show_lidar_arg,
        combined_gz_process_arg,
        turtlebot3_model_env,
        gz_sim_server_cmd,
        gz_sim_client_cmd,
        gz_sim_combined_cmd,
        robot_state_publisher_cmd,
        spawn_turtlebot_cmd,
        mission_controller_node,
        monitor_node,
        path_visualizer_node,
    ])