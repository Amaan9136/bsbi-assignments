"""
vision_core.py

Pure image-processing logic for marker detection, with no dependency on
rospy or any ROS message types. Separated from vision_detector.py (the
ROS node) so this module can be imported, run, and edited in a plain
Python environment, VS Code included, without a ROS installation.

vision_detector.py is a thin wrapper that calls detect_marker() from this
module and publishes the result on ROS topics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# HSV threshold range for the target marker colour. Defaults detect a red
# marker; red spans the wraparound point of the HSV hue circle, so two
# ranges are combined. For a different colour, e.g. blue, typical bounds
# are H: 100-130, S: 100-255, V: 50-255, with no wraparound needed.
LOWER_RED_1 = np.array([0, 120, 70])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 120, 70])
UPPER_RED_2 = np.array([180, 255, 255])

MIN_CONTOUR_AREA = 500


@dataclass
class DetectionResult:
    offset: float          # normalised horizontal offset in [-1, 1]
    visible: bool
    centroid: Optional[Tuple[int, int]]
    contour_area: float
    debug_frame: np.ndarray


def build_red_mask(hsv_frame: np.ndarray) -> np.ndarray:
    """Builds a binary mask isolating red pixels in an HSV frame,
    combining both ends of the hue wraparound, then cleans it with
    morphological opening and closing to remove small noise blobs."""
    mask1 = cv2.inRange(hsv_frame, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv_frame, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_marker(frame_bgr: np.ndarray, min_area: float = MIN_CONTOUR_AREA) -> DetectionResult:
    """Applies HSV thresholding to an OpenCV BGR frame, finds the largest
    matching contour, and returns its centroid, normalised horizontal
    offset from the frame centre, and an annotated debug frame.

    This function has no ROS dependency and can be exercised directly
    from a plain Python script or the VS Code debugger, e.g. against a
    static test image or a laptop webcam frame, before ever touching a
    ROS topic.
    """
    height, width = frame_bgr.shape[:2]
    frame_center_x = width / 2.0

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = build_red_mask(hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug_frame = frame_bgr.copy()

    offset = 0.0
    visible = False
    centroid = None
    area = 0.0

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area >= min_area:
            moments = cv2.moments(largest_contour)
            if moments["m00"] != 0:
                centroid_x = moments["m10"] / moments["m00"]
                centroid_y = moments["m01"] / moments["m00"]
                centroid = (int(centroid_x), int(centroid_y))

                offset = (centroid_x - frame_center_x) / frame_center_x
                visible = True

                cv2.drawContours(debug_frame, [largest_contour], -1, (0, 255, 0), 2)
                cv2.circle(debug_frame, centroid, 6, (255, 0, 0), -1)

    cv2.line(debug_frame, (width // 2, 0), (width // 2, height), (0, 255, 255), 1)
    cv2.putText(
        debug_frame, f"visible={visible} offset={offset:.2f}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )

    return DetectionResult(
        offset=offset, visible=visible, centroid=centroid,
        contour_area=area, debug_frame=debug_frame,
    )


if __name__ == "__main__":
    # Quick local smoke test: synthesise a frame with a red square and
    # confirm detection works, entirely outside of ROS. Run directly in
    # VS Code with: python src/vision_core.py
    import sys

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (400, 200), (460, 260), (0, 0, 255), -1)  # BGR red square

    result = detect_marker(frame)
    print(f"visible={result.visible} offset={result.offset:.3f} centroid={result.centroid}")

    if not result.visible:
        print("Smoke test FAILED: expected marker was not detected.", file=sys.stderr)
        sys.exit(1)
    print("Smoke test passed.")
