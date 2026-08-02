"""ArUco fiducial marker: the calibration target.

Calibration needs a physical point the camera can find reliably. An ArUco
marker (4x4 dictionary, id 0) is unambiguous, works from a phone or monitor
screen, and detects in one call at any reasonable distance and lighting.
"""

from pathlib import Path

import cv2
import numpy as np

_DICT = cv2.aruco.DICT_4X4_50
MARKER_ID = 0


def generate_target(path: str | Path, size: int = 900) -> Path:
    """Write the calibration target PNG: the marker with a generous white
    quiet zone (required for detection) and a center cross for the eye."""
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(_DICT), MARKER_ID, size
    )
    border = size // 5
    canvas = np.full((size + 2 * border, size + 2 * border), 255, dtype=np.uint8)
    canvas[border : border + size, border : border + size] = marker
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"could not write {path}")
    return path


def detect_target_center(image_bgr: np.ndarray) -> tuple[float, float] | None:
    """Normalized center of the marker in the frame, or None if not found."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(_DICT), cv2.aruco.DetectorParameters()
    )
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    h, w = gray.shape[:2]
    for marker_corners, marker_id in zip(corners, ids.flatten(), strict=True):
        if marker_id == MARKER_ID:
            center = marker_corners.reshape(-1, 2).mean(axis=0)
            return (float(center[0]) / w, float(center[1]) / h)
    return None
