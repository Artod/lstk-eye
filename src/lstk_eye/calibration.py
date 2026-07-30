"""Camera->display calibration.

The display shows a rectangular window into the camera frame: display center
(64, 32) corresponds to camera point ``(center_x, center_y)`` and the full
128x64 panel spans ``(window_w, window_h)`` of the frame, all in normalized
camera coordinates. The four parameters come from config, or from measured
point pairs via :meth:`WindowCalibration.fit` (the physical calibration
procedure: show a dot on the OLED, point the camera at a marker, record where
the marker sits in the frame when it lines up with the dot).
"""

from __future__ import annotations

import numpy as np

from lstk_eye.config import CalibrationConfig
from lstk_eye.errors import ConfigError
from lstk_eye.pipeline.interfaces import Calibration
from lstk_eye.protocol.messages import DISPLAY_H, DISPLAY_W

# Per-axis (span, half): display extent in px and the pixel of the window center.
_AXES = ((DISPLAY_W, DISPLAY_W // 2), (DISPLAY_H, DISPLAY_H // 2))


class WindowCalibration(Calibration):
    """Affine per-axis window model; see module docstring."""

    def __init__(self, cfg: CalibrationConfig):
        self.center_x = cfg.center_x
        self.center_y = cfg.center_y
        self.window_w = cfg.window_w
        self.window_h = cfg.window_h

    def to_display(self, pt: tuple[float, float]) -> tuple[int, int]:
        x, y = pt
        dx = round((x - self.center_x) / self.window_w * DISPLAY_W + DISPLAY_W // 2)
        dy = round((y - self.center_y) / self.window_h * DISPLAY_H + DISPLAY_H // 2)
        return int(dx), int(dy)

    def in_window(self, pt: tuple[float, float], margin_px: int = 0) -> bool:
        dx, dy = self.to_display(pt)
        return (
            margin_px <= dx <= DISPLAY_W - 1 - margin_px
            and margin_px <= dy <= DISPLAY_H - 1 - margin_px
        )

    def edge_for(self, pt: tuple[float, float]) -> str:
        dx, dy = self.to_display(pt)
        # Overshoot beyond the panel on each axis, normalized by the axis
        # extent so a corner target picks the axis it is proportionally
        # further out on. Ties prefer the horizontal chevrons.
        over_x = max(-dx, dx - (DISPLAY_W - 1), 0)
        over_y = max(-dy, dy - (DISPLAY_H - 1), 0)
        if over_x / DISPLAY_W >= over_y / DISPLAY_H:
            return "left" if dx < 0 else "right"
        return "up" if dy < 0 else "down"

    @classmethod
    def fit(
        cls, pairs: list[tuple[tuple[float, float], tuple[int, int]]]
    ) -> WindowCalibration:
        """Least-squares fit of the four parameters from measured
        ``(camera_pt, display_px)`` pairs.

        The model is separable and linear per axis: ``d = a*c + b`` with
        ``a = span/window`` and ``b = half - a*center``, so each axis is an
        ordinary 1-D linear regression. Needs at least two pairs with spread
        on both axes; raises ConfigError otherwise.
        """
        if len(pairs) < 2:
            raise ConfigError(f"calibration fit needs >= 2 point pairs, got {len(pairs)}")
        cam = np.asarray([p[0] for p in pairs], dtype=np.float64)
        disp = np.asarray([p[1] for p in pairs], dtype=np.float64)
        centers: list[float] = []
        windows: list[float] = []
        for axis, (span, half) in enumerate(_AXES):
            c = cam[:, axis] - cam[:, axis].mean()
            d = disp[:, axis] - disp[:, axis].mean()
            denom = float(c @ c)
            if denom < 1e-12:
                raise ConfigError(f"calibration fit: no spread on axis {axis} camera values")
            a = float(c @ d) / denom
            if abs(a) < 1e-9:
                raise ConfigError(f"calibration fit: degenerate slope on axis {axis}")
            b = float(disp[:, axis].mean() - a * cam[:, axis].mean())
            windows.append(span / a)
            centers.append((half - b) / a)
        return cls(
            CalibrationConfig(
                center_x=centers[0],
                center_y=centers[1],
                window_w=windows[0],
                window_h=windows[1],
            )
        )
