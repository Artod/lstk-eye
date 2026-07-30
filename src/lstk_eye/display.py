"""Scene composition: pure geometry and layout, no I/O.

Every DisplayScene the device ever shows is produced here from session state
plus the camera->display calibration. Layout constants assume the 128x64
SSD1306 with the GFX 5x7 font: a size-1 character advances 6 px, so a line
holds 21 characters.
"""

from __future__ import annotations

import math
import textwrap

from lstk_eye.pipeline.interfaces import Calibration
from lstk_eye.pipeline.types import Slide
from lstk_eye.protocol.messages import (
    CHARS_PER_LINE,
    DISPLAY_H,
    DISPLAY_W,
    ArrowEl,
    ChevronEl,
    DisplayElement,
    DisplayScene,
    TextEl,
)

CHAR_W = 6  # size-1 glyph advance in px
LINE_H = 9  # label line pitch: 8 px glyph + 1 px gap
STATUS_Y = 56  # bottom status line (56 + 8 = panel height)
CENTER_X = DISPLAY_W // 2
CENTER_Y = DISPLAY_H // 2
ARROW_LENGTH = 12
# Clamp band keeping the whole arrow glyph clear of the label and status rows.
ARROW_X_MIN, ARROW_X_MAX = 2, 125
ARROW_Y_MIN, ARROW_Y_MAX = 18, 53
ERROR_MAX_LINES = 4


def wrap_text(text: str, width: int = CHARS_PER_LINE, max_lines: int = 2) -> list[str]:
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


def _centered(text: str, y: int, size: int = 1) -> TextEl:
    x = max(0, (DISPLAY_W - len(text) * CHAR_W * size) // 2)
    return TextEl(x=x, y=y, size=size, text=text)


class SceneComposer:
    """Builds DisplayScene frames; owns all layout decisions."""

    def __init__(self, calibration: Calibration):
        self._calibration = calibration

    def slide(
        self, slide: Slide, seq: int, anchored: bool, anchor: tuple[float, float] | None
    ) -> DisplayScene:
        """One instruction slide.

        ``slide.anchor`` is the target from the capture frame (None for
        text-only steps); ``anchor`` is the current target position from
        relocalization; ``anchored`` is the debounced found/lost state. Arrows
        render only from a live, found anchor - a lost target degrades to the
        "look back" hint.
        """
        els: list[DisplayElement] = []
        for i, line in enumerate(wrap_text(slide.label)):
            els.append(TextEl(x=0, y=i * LINE_H, text=line))
        if not anchored and slide.anchor is not None:
            els.append(TextEl(x=0, y=STATUS_Y, text="look back"))
        counter = f"{slide.index + 1}/{slide.total}"
        els.append(TextEl(x=DISPLAY_W - len(counter) * CHAR_W, y=STATUS_Y, text=counter))
        if anchor is not None and anchored:
            if self._calibration.in_window(anchor):
                raw_x, raw_y = self._calibration.to_display(anchor)
                tip_x = min(max(raw_x, ARROW_X_MIN), ARROW_X_MAX)
                tip_y = min(max(raw_y, ARROW_Y_MIN), ARROW_Y_MAX)
                angle = _snap45(tip_x - CENTER_X, tip_y - CENTER_Y)
                els.append(ArrowEl(x=tip_x, y=tip_y, angle=angle, length=ARROW_LENGTH))
            else:
                els.append(ChevronEl(edge=self._calibration.edge_for(anchor)))
        return DisplayScene(seq=seq, els=els)

    def photo_count(self, count: int, seq: int) -> DisplayScene:
        return DisplayScene(
            seq=seq,
            els=[
                _centered(f"[{count}]", y=16, size=2),
                _centered("photo saved", y=40),
            ],
        )

    def thinking(self, seq: int) -> DisplayScene:
        return DisplayScene(seq=seq, els=[_centered("thinking...", y=28)])

    def status(self, text: str, seq: int, sub: str | None = None) -> DisplayScene:
        els: list[DisplayElement] = [_centered(text, y=24)]
        if sub is not None:
            els.append(_centered(sub, y=36))
        return DisplayScene(seq=seq, els=els)

    def error(self, msg: str, seq: int) -> DisplayScene:
        lines = wrap_text(f"! {msg}", max_lines=ERROR_MAX_LINES)
        els: list[DisplayElement] = [
            TextEl(x=0, y=8 + i * LINE_H, text=line) for i, line in enumerate(lines)
        ]
        return DisplayScene(seq=seq, els=els)

    def blank(self, seq: int) -> DisplayScene:
        return DisplayScene(seq=seq, els=[])
