cmake_minimum_required(VERSION 3.0.2)
project(turtlebot3_vision_nav)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  image_transport
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs geometry_msgs std_msgs cv_bridge image_transport
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/vision_detector.py
  scripts/navigation_controller.py
  scripts/vision_core.py
  scripts/control_core.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
