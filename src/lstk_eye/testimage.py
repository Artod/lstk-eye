"""Synthetic test scene for demos and the simulator.

A stylized multimeter on a desk: solid, high-contrast shapes so both the
classical mock segmenter and real models find distinct regions (screen, dial,
probe jacks). Deterministic - no RNG - so tests can assert on it.
"""

import cv2
import numpy as np


def make_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    img = np.full((height, width, 3), 190, dtype=np.uint8)  # desk

    # Multimeter body
    bx0, by0, bx1, by1 = int(width * 0.28), int(height * 0.08), int(width * 0.72), int(height * 0.92)
    cv2.rectangle(img, (bx0, by0), (bx1, by1), (60, 60, 70), -1)
    cv2.rectangle(img, (bx0, by0), (bx1, by1), (30, 30, 35), 4)

    # LCD screen
    sx0, sy0 = int(width * 0.33), int(height * 0.12)
    sx1, sy1 = int(width * 0.67), int(height * 0.28)
    cv2.rectangle(img, (sx0, sy0), (sx1, sy1), (170, 230, 200), -1)
    cv2.putText(img, "0.00", (sx0 + 20, sy1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (40, 60, 45), 3)

    # Rotary dial
    dc = (int(width * 0.5), int(height * 0.52))
    cv2.circle(img, dc, int(height * 0.14), (200, 200, 210), -1)
    cv2.circle(img, dc, int(height * 0.14), (30, 30, 35), 3)
    cv2.line(img, dc, (dc[0] + 20, dc[1] - int(height * 0.12)), (30, 30, 35), 5)

    # Probe jacks: VOmA, COM, 10A
    jy = int(height * 0.82)
    for i, (cx, color) in enumerate(
        [
            (int(width * 0.38), (40, 40, 200)),  # red-ish (BGR)
            (int(width * 0.50), (40, 40, 40)),  # black
            (int(width * 0.62), (40, 40, 200)),
        ]
    ):
        cv2.circle(img, (cx, jy), 16, color, -1)
        cv2.circle(img, (cx, jy), 16, (230, 230, 230), 2)
        cv2.circle(img, (cx, jy), 6, (10, 10, 10), -1)
        _ = i

    return img
