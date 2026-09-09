"""
control_core.py

Pure control logic for the navigation controller, with no dependency on
rospy or any ROS message types. Separated from navigation_controller.py
(the ROS node) so this module can be imported and exercised in a plain
Python environment, VS Code included, without a ROS installation.

navigation_controller.py is a thin wrapper that calls compute_velocity()
from this module and publishes the result as a geometry_msgs/Twist.
"""

from __future__ import annotations

from dataclasses import dataclass

CENTER_THRESHOLD = 0.15
MAX_LINEAR_SPEED = 0.15
MAX_ANGULAR_SPEED = 0.6
SEARCH_ANGULAR_SPEED = 0.3
ANGULAR_GAIN = 0.8


@dataclass
class Velocity:
    linear_x: float = 0.0
    angular_z: float = 0.0


def compute_velocity(
    target_visible: bool,
    target_offset: float,
    center_threshold: float = CENTER_THRESHOLD,
    max_linear_speed: float = MAX_LINEAR_SPEED,
    max_angular_speed: float = MAX_ANGULAR_SPEED,
    search_angular_speed: float = SEARCH_ANGULAR_SPEED,
    angular_gain: float = ANGULAR_GAIN,
) -> Velocity:
    """Pure function implementing the rule based navigation logic:

    - Target not visible: rotate slowly to search.
    - Target visible, off-centre beyond threshold: rotate toward it,
      proportional to offset magnitude, clamped to max_angular_speed.
    - Target visible and centred: drive forward at max_linear_speed.

    Takes no ROS types in or out, so it can be called directly from a
    plain Python script or the VS Code debugger with synthetic inputs,
    e.g. compute_velocity(target_visible=True, target_offset=0.4), to
    verify the control logic before ever running it against a live
    camera feed or the simulator.
    """
    if not target_visible:
        return Velocity(linear_x=0.0, angular_z=search_angular_speed)

    if abs(target_offset) > center_threshold:
        # Positive offset means the target is right of centre; a
        # negative angular velocity turns the robot toward it under
        # ROS's right-hand convention about the z-axis.
        angular_z = -angular_gain * target_offset
        angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))
        return Velocity(linear_x=0.0, angular_z=angular_z)

    return Velocity(linear_x=max_linear_speed, angular_z=0.0)


if __name__ == "__main__":
    # Quick local smoke test covering all three control branches. Run
    # directly in VS Code with: python src/control_core.py
    import sys

    cases = [
        (False, 0.0, "search"),
        (True, 0.4, "turn"),
        (True, 0.05, "forward"),
    ]

    all_passed = True
    for visible, offset, label in cases:
        v = compute_velocity(target_visible=visible, target_offset=offset)
        print(f"[{label}] visible={visible} offset={offset} -> linear_x={v.linear_x:.3f} angular_z={v.angular_z:.3f}")

        if label == "search" and not (v.angular_z > 0 and v.linear_x == 0):
            all_passed = False
        if label == "turn" and not (v.linear_x == 0 and v.angular_z != 0):
            all_passed = False
        if label == "forward" and not (v.linear_x > 0 and v.angular_z == 0):
            all_passed = False

    if not all_passed:
        print("Smoke test FAILED.", file=sys.stderr)
        sys.exit(1)
    print("Smoke test passed.")
