import os
from glob import glob

from setuptools import find_packages, setup

package_name = "turtlebot3_vision_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Student",
    maintainer_email="student@example.com",
    description=(
        "Vision guided navigation package for TurtleBot3. Detects a "
        "coloured marker using an RGB camera and OpenCV colour "
        "thresholding, then drives the robot toward the marker using a "
        "simple rule based controller."
    ),
    license="BSD",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vision_detector = turtlebot3_vision_nav.vision_detector:main",
            "navigation_controller = turtlebot3_vision_nav.navigation_controller:main",
        ],
    },
)