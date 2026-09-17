#!/usr/bin/env python3
"""
semantic_nav_monitor.launch.py

Launches the custom "warehouse_inspection" Gazebo world bundled with this
package (see worlds/warehouse_inspection.world), spawns a TurtleBot3 into it
using the standard turtlebot3_gazebo robot_state_publisher/spawn launch
files, and then starts the mission_controller and monitor_node nodes.

USE CASE (Task 2): Warehouse Inspection Patrol Robot.
The robot patrols four checkpoints around a small warehouse bay, encounters
two pallet obstacles placed directly on its route, and reports its state
and performance metrics throughout the mission.

Assumes the TURTLEBOT3_MODEL environment variable has already been exported
(for example: export TURTLEBOT3_MODEL=burger) before running this launch
file, and that the turtlebot3_gazebo and gazebo_ros packages are available
on the ROS2 package path (both are preinstalled on TheConstruct.ai ROS2 + TurtleBot3 Rosjects).
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
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")

    default_world = os.path.join(
        pkg_semantic_nav_monitor, "worlds", "warehouse_inspection.world"
    )

    world_arg = DeclareLaunchArgument(
        "world",
        default_value=default_world,
        description="Full path to the Gazebo world file to load.",
    )
    x_pose_arg = DeclareLaunchArgument("x_pose", default_value="0.0")
    y_pose_arg = DeclareLaunchArgument("y_pose", default_value="0.0")

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzserver.launch.py")
        ),
        launch_arguments={"world": LaunchConfiguration("world")}.items(),
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzclient.launch.py")
        )
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
    )

    monitor_node = Node(
        package="semantic_nav_monitor",
        executable="monitor_node",
        name="monitor_node",
        output="screen",
    )

    return LaunchDescription([
        world_arg,
        x_pose_arg,
        y_pose_arg,
        gzserver_cmd,
        gzclient_cmd,
        robot_state_publisher_cmd,
        spawn_turtlebot_cmd,
        mission_controller_node,
        monitor_node,
    ])
