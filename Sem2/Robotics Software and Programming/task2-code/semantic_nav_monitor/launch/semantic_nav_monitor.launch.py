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
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
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

    turtlebot3_model_env = SetEnvironmentVariable(
        name="TURTLEBOT3_MODEL",
        value=os.environ.get("TURTLEBOT3_MODEL", "burger_cam"),
    )

    gui_config_path = PythonExpression([
        "'", gui_config_lidar_on, "' if '",
        LaunchConfiguration("show_lidar"),
        "' == 'true' else '", gui_config_lidar_off, "'",
    ])

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
    )

    gz_sim_client_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": ["-g -v2 --gui-config ", gui_config_path],
        }.items(),
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
        turtlebot3_model_env,
        gz_sim_server_cmd,
        gz_sim_client_cmd,
        robot_state_publisher_cmd,
        spawn_turtlebot_cmd,
        mission_controller_node,
        monitor_node,
        path_visualizer_node,
    ])