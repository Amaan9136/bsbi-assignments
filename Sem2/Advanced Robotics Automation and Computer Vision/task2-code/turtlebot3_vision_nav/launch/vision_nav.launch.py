import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    turtlebot3_model_env = SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle_pi")

    camera_topic_arg = DeclareLaunchArgument(
        "camera_topic",
        default_value="/camera/image_raw",
        description="Camera topic the perception node subscribes to.",
    )

    vision_detector_node = Node(
        package="turtlebot3_vision_nav",
        executable="vision_detector",
        name="vision_detector",
        output="screen",
        parameters=[{"camera_topic": LaunchConfiguration("camera_topic")}],
    )

    navigation_controller_node = Node(
        package="turtlebot3_vision_nav",
        executable="navigation_controller",
        name="navigation_controller",
        output="screen",
        parameters=[{"control_rate": 10.0}],
    )

    ld = LaunchDescription([turtlebot3_model_env, camera_topic_arg])

    try:
        turtlebot3_gazebo_share = get_package_share_directory("turtlebot3_gazebo")
        world_launch_path = os.path.join(turtlebot3_gazebo_share, "launch", "turtlebot3_world.launch.py")
        if os.path.exists(world_launch_path):
            ld.add_action(
                IncludeLaunchDescription(PythonLaunchDescriptionSource(world_launch_path))
            )
    except Exception:
        # turtlebot3_gazebo not found on this system; bring up only the
        # perception/control nodes and let the user launch the sim
        # separately with their installed TurtleBot3 launch files.
        pass

    ld.add_action(vision_detector_node)
    ld.add_action(navigation_controller_node)
    return ld