"""Scene composition: pure geometry and layout, no I/O.

Every DisplayScene the device ever shows is produced here from session state
plus the camera->display calibration. Layout constants assume the 128x64
SSD1306 with the GFX 5x7 font: a size-1 character advances 6 px.

The optics crop the panel edges, so all layout happens inside the visible
rectangle defined by DisplayConfig pads - text, counters, arrows, target
brackets, and compass chevrons never touch the cropped border.
"""

from __future__ import annotations

import math
import textwrap
from typing import Literal

from lstk_eye.config import DisplayConfig
from lstk_eye.pipeline.interfaces import Calibration
from lstk_eye.pipeline.types import Slide
from lstk_eye.protocol.messages import (
    DISPLAY_H,
    DISPLAY_W,
    ArrowEl,
    ChevronEl,
    DisplayElement,
    DisplayScene,
    TargetEl,
    TextEl,
)

CHAR_W = 6  # size-1 glyph advance in px
LINE_H = 9  # label line pitch: 8 px glyph + 1 px gap
GLYPH_H = 8
ARROW_LENGTH = 12
ERROR_MAX_LINES = 4
TARGET_R_MIN = 6
TARGET_R_MAX = 22


def wrap_text(text: str, width: int, max_lines: int = 2) -> list[str]:
    """Word-wrap ``text`` into at most ``max_lines`` lines of at most
    ``width`` characters. Overflow truncates the last line with a ``..``
    suffix; words longer than a line are hard-split."""
    lines = textwrap.wrap(text, width=width)
    if len(lines) <= max_lines:
        return lines
    lines = lines[:max_lines]
    lines[-1] = lines[-1][: width - 2].rstrip() + ".."
    return lines


def _snap45(dx: int, dy: int) -> int:
    """Direction of vector (dx, dy) in screen degrees (0 = right, 90 = down)
    snapped to the nearest multiple of 45."""
    return round(math.degrees(math.atan2(dy, dx)) / 45) * 45 % 360


class SceneComposer:
    """Builds DisplayScene frames; owns all layout decisions."""

    def __init__(self, calibration: Calibration, display: DisplayConfig | None = None):
        self._calibration = calibration
        cfg = display or DisplayConfig()
        self.x0 = cfg.pad_x
        self.x1 = DISPLAY_W - 1 - cfg.pad_x
        self.y0 = cfg.pad_y
        self.y1 = DISPLAY_H - 1 - cfg.pad_y
        self.chars_per_line = max(4, (self.x1 - self.x0 + 1) // CHAR_W)
        self.status_y = self.y1 - GLYPH_H + 1  # bottom text row inside the pad
        self.center = ((self.x0 + self.x1) // 2, (self.y0 + self.y1) // 2)
        # Arrow tips stay clear of the two label rows and the status row.
        self.arrow_x = (self.x0 + 2, self.x1 - 2)
        self.arrow_y = (self.y0 + 2 * LINE_H, self.status_y - 3)

    # --- shared helpers ---

    def _visible(self, px: tuple[int, int], margin: int = 0) -> bool:
        x, y = px
        return (
            self.x0 + margin <= x <= self.x1 - margin
            and self.y0 + margin <= y <= self.y1 - margin
        )

    def _clamp(self, px: tuple[int, int], margin: int = 2) -> tuple[int, int]:
        x, y = px
        return (
            min(max(x, self.x0 + margin), self.x1 - margin),
            min(max(y, self.y0 + margin), self.y1 - margin),
        )

    def _centered(self, text: str, y: int, size: int = 1) -> TextEl:
        width = self.x1 - self.x0 + 1
        x = self.x0 + max(0, (width - len(text) * CHAR_W * size) // 2)
        return TextEl(x=x, y=y, size=size, text=text)

    def _label_els(self, label: str) -> list[DisplayElement]:
        return [
            TextEl(x=self.x0, y=self.y0 + i * LINE_H, text=line)
            for i, line in enumerate(wrap_text(label, self.chars_per_line))
        ]

    def _compass(self, anchor: tuple[float, float]) -> ChevronEl:
        """Chevron just inside the visible border, pointing toward an
        off-window target: turn your head this way."""
        tip = self._clamp(self._calibration.to_display(anchor), margin=3)
        return ChevronEl(edge=self._calibration.edge_for(anchor), x=tip[0], y=tip[1])

    # --- scenes ---

    def slide(
        self,
        slide: Slide,
        seq: int,
        anchored: bool,
        anchor: tuple[float, float] | None,
        style: Literal["arrow", "target"] = "arrow",
    ) -> DisplayScene:
        """One instruction slide.

        ``slide.anchor`` is the target from the capture frame (None for
        text-only steps); ``anchor`` is the current target position from
        relocalization; ``anchored`` is the debounced found/lost state.
        ``style`` picks the marker: an instruction arrow, or the object
        highlight brackets used by find-this sessions.
        """
        label_els = self._label_els(slide.label)
        els: list[DisplayElement] = list(label_els)
        # The arrow tip must clear the rows the label actually occupies, not a
        # fixed two-line reservation - a one-line label frees a row.
        label_bottom = self.y0 + len(label_els) * LINE_H
        if not anchored and slide.anchor is not None:
            els.append(TextEl(x=self.x0, y=self.status_y, text="look back"))
        if slide.total > 1:
            counter = f"{slide.index + 1}/{slide.total}"
            els.append(
                TextEl(x=self.x1 + 1 - len(counter) * CHAR_W, y=self.status_y, text=counter)
            )
        if anchor is not None and anchored:
            raw = self._calibration.to_display(anchor)
            if self._visible(raw):
                if style == "target":
                    els.append(self._target_el(raw, slide.size, label_bottom))
                else:
                    tip_x = min(max(raw[0], self.arrow_x[0]), self.arrow_x[1])
                    tip_y = min(max(raw[1], label_bottom + 3), self.arrow_y[1])
                    angle = _snap45(tip_x - self.center[0], tip_y - self.center[1])
                    els.append(ArrowEl(x=tip_x, y=tip_y, angle=angle, length=ARROW_LENGTH))
            else:
                els.append(self._compass(anchor))
        return DisplayScene(seq=seq, els=els)

    def _target_el(
        self, px: tuple[int, int], size: tuple[float, float] | None, label_bottom: int
    ) -> TargetEl:
        """Corner brackets sized to the object's apparent extent, kept inside
        the visible area and clear of the label rows."""
        x, y = px
        r = TARGET_R_MIN
        if size is not None:
            # Half-extent of the object's bbox in display pixels, via the
            # calibration's scale (mapping is affine, so a delta maps to a
            # delta).
            zero = self._calibration.to_display((0.0, 0.0))
            span = self._calibration.to_display((size[0] / 2, size[1] / 2))
            r = max(abs(span[0] - zero[0]), abs(span[1] - zero[1]))
        r = min(max(r, TARGET_R_MIN), TARGET_R_MAX)
        # Shrink rather than shift: the brackets must frame the object, so the
        # center stays put and r gives way at the border and under the label.
        limits = [r, x - self.x0, self.x1 - x, y - self.y0, self.y1 - y]
        if y > label_bottom:
            limits.append(y - label_bottom - 1)
        r = min(limits)
        return TargetEl(x=x, y=y, r=max(r, 3))

    def calibration_point(
        self, index: int, total: int, px: tuple[int, int], seq: int, hint: str | None = None
    ) -> DisplayScene:
        """One calibration crosshair: small brackets at ``px`` plus the step
        instruction on the status row (kept away from the crosshair)."""
        instruction = hint or f"aim {index + 1}/{total} + click"
        text_y = self.status_y if px[1] < self.center[1] else self.y0
        return DisplayScene(
            seq=seq,
            els=[
                TargetEl(x=px[0], y=px[1], r=5),
                TextEl(x=self.x0, y=text_y, text=instruction),
            ],
        )

    def photo_count(self, count: int, seq: int) -> DisplayScene:
        return DisplayScene(
            seq=seq,
            els=[
                self._centered(f"[{count}]", y=14, size=2),
                self._centered("photo saved", y=38),
            ],
        )

    def thinking(self, seq: int) -> DisplayScene:
        return DisplayScene(seq=seq, els=[self._centered("thinking...", y=26)])

    def status(self, text: str, seq: int, sub: str | None = None) -> DisplayScene:
        els: list[DisplayElement] = [self._centered(text, y=22)]
        if sub is not None:
            els.append(self._centered(sub, y=34))
        return DisplayScene(seq=seq, els=els)

    def error(self, msg: str, seq: int) -> DisplayScene:
        lines = wrap_text(f"! {msg}", self.chars_per_line, max_lines=ERROR_MAX_LINES)
        els: list[DisplayElement] = [
            TextEl(x=self.x0, y=self.y0 + 6 + i * LINE_H, text=line)
            for i, line in enumerate(lines)
        ]
        return DisplayScene(seq=seq, els=els)

    def blank(self, seq: int) -> DisplayScene:
        return DisplayScene(seq=seq, els=[])
