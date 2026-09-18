#!/usr/bin/env python3
"""
semantic_nav_monitor.launch.py

Launches the custom "warehouse_inspection" Gazebo world bundled with this
package, spawns a TurtleBot3 into it using turtlebot3_gazebo's launch files,
and starts the mission_controller and monitor_node nodes.

USE CASE (Task 2): Warehouse Inspection Patrol Robot.

NOTE (Jazzy port): the turtlebot3_gazebo launch files already start their
own ros_gz_bridge (odom/scan/imu/tf/clock/cmd_vel, with cmd_vel as
TwistStamped). We do NOT start a second bridge here - a second bridge
subscribing to /cmd_vel as plain Twist was the actual cause of the robot
not moving (type mismatch meant the authoritative bridge never received
mission_controller's velocity commands).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_semantic_nav_monitor = get_package_share_directory("semantic_nav_monitor")
    pkg_turtlebot3_gazebo = get_package_share_directory("turtlebot3_gazebo")
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")

    default_world = os.path.join(
        pkg_semantic_nav_monitor, "worlds", "warehouse_inspection.sdf"
    )

    world_arg = DeclareLaunchArgument(
        "world",
        default_value=default_world,
        description="Full path to the Gazebo world file to load.",
    )
    x_pose_arg = DeclareLaunchArgument("x_pose", default_value="0.0")
    y_pose_arg = DeclareLaunchArgument("y_pose", default_value="0.0")

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
        launch_arguments={"gz_args": "-g -v2 "}.items(),
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

    return LaunchDescription([
        world_arg,
        x_pose_arg,
        y_pose_arg,
        gz_sim_server_cmd,
        gz_sim_client_cmd,
        robot_state_publisher_cmd,
        spawn_turtlebot_cmd,
        mission_controller_node,
        monitor_node,
    ])