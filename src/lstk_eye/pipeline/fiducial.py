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

# Detection accepts markers from the families people actually end up
# displaying: users grab "a calibration marker" from whatever generator they
# find, and AprilTag/other-ArUco images look identical to the naked eye.
# Any marker id is accepted - only the fiducial's center matters.
_ACCEPTED_DICTS = (
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_4X4_1000,
    cv2.aruco.DICT_5X5_1000,
    cv2.aruco.DICT_6X6_1000,
    cv2.aruco.DICT_7X7_1000,
    cv2.aruco.DICT_ARUCO_ORIGINAL,
    cv2.aruco.DICT_APRILTAG_16H5,
    cv2.aruco.DICT_APRILTAG_36H11,
)


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


_last_hit_dict: int | None = None


def _detect_any(gray: np.ndarray) -> np.ndarray | None:
    """Largest fiducial quad (pixel corners) from the first family that
    matches. The last successful dictionary is tried first and the scan stops
    on a hit - one physical marker is in play, so cross-family comparison
    buys nothing and the preview stream calls this at 2-3 fps."""
    global _last_hit_dict
    params = cv2.aruco.DetectorParameters()
    order = list(_ACCEPTED_DICTS)
    if _last_hit_dict in order:
        order.remove(_last_hit_dict)
        order.insert(0, _last_hit_dict)
    for dict_id in order:
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(dict_id), params
        )
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue
        best: np.ndarray | None = None
        best_side = 0.0
        for marker_corners in corners:
            quad = marker_corners.reshape(-1, 2)
            side = float(max(np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)))
            if side > best_side:
                best, best_side = quad, side
        _last_hit_dict = dict_id
        return best
    return None


def detect_target(
    image_bgr: np.ndarray,
) -> tuple[tuple[float, float], float, np.ndarray] | None:
    """Find the marker: (normalized center, normalized size, corner pixels).

    ``size`` is the longest quad side as a fraction of the frame width - the
    calibration flow uses it to tell "marker too small / too far" apart from
    "marker not in view". Returns None when no marker is detected.

    Small markers (a few dozen pixels) fall below the detectors' native
    working range, so a miss retries on a 2x upscale before giving up."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    h, w = gray.shape[:2]
    quad = _detect_any(gray)
    if quad is None:
        doubled = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        quad = _detect_any(doubled)
        if quad is None:
            return None
        quad = quad / 2.0
    center = quad.mean(axis=0)
    side = float(max(np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)))
    return ((float(center[0]) / w, float(center[1]) / h), side / w, quad)


def detect_target_center(image_bgr: np.ndarray) -> tuple[float, float] | None:
    """Normalized center of the marker in the frame, or None if not found."""
    found = detect_target(image_bgr)
    return found[0] if found is not None else None


def annotate_detection(
    image_bgr: np.ndarray, found: tuple[tuple[float, float], float, np.ndarray] | None
) -> np.ndarray:
    """Debug render: the detected quad and center cross, or a NOT FOUND
    banner. Saved with every calibration click so misses are inspectable."""
    out = image_bgr.copy()
    if found is None:
        cv2.putText(
            out, "MARKER NOT FOUND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3
        )
        return out
    (cx, cy), _, quad = found
    h, w = out.shape[:2]
    cv2.polylines(out, [quad.astype(np.int32)], True, (0, 255, 0), 3)
    px, py = int(cx * w), int(cy * h)
    cv2.drawMarker(out, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
    return out
