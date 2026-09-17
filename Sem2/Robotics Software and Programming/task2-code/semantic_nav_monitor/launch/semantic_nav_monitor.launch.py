#!/usr/bin/env python3
"""
semantic_nav_monitor.launch.py

Launches the standard TurtleBot3 Gazebo simulation world, then starts the
mission_controller and monitor_node nodes from this package.

Assumes the TURTLEBOT3_MODEL environment variable has already been exported
(for example: export TURTLEBOT3_MODEL=burger) before running this launch file,
and that the turtlebot3_gazebo package is available on the ROS2 package path.
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    turtlebot3_gazebo_dir = get_package_share_directory("turtlebot3_gazebo")

    world_launch_arg = DeclareLaunchArgument(
        "world",
        default_value="turtlebot3_world.launch.py",
        description="TurtleBot3 Gazebo launch file to include.",
    )

    turtlebot3_world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [turtlebot3_gazebo_dir, "launch", LaunchConfiguration("world")]
            )
        )
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
        world_launch_arg,
        turtlebot3_world_launch,
        mission_controller_node,
        monitor_node,
    ])