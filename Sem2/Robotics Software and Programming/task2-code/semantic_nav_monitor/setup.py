import os
from glob import glob
from setuptools import find_packages, setup

package_name = "semantic_nav_monitor"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*.launch.py"))),
        (os.path.join("share", package_name, "worlds"),
            glob(os.path.join("worlds", "*.world"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="MSc AI Student",
    maintainer_email="student@bsbi-example.edu",
    description=(
        "Goal-oriented autonomous mission controller and monitoring node "
        "for a TurtleBot3 robot in ROS2."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mission_controller = semantic_nav_monitor.mission_controller:main",
            "monitor_node = semantic_nav_monitor.monitor_node:main",
            "keyboard_hri_node = semantic_nav_monitor.keyboard_hri_node:main",
        ],
    },
)