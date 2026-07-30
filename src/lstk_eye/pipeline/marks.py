"""Set-of-Marks overlay and debug rendering.

``overlay_marks`` produces the numbered image the planner sees: the numbers
are the only handle the VLM gets, so badges must be unambiguous and legible
at the model's input resolution. ``draw_slides`` is a laptop-side debug view
of a finished plan; the device renders its own scene from DisplayScene.
"""

import cv2
import numpy as np

from lstk_eye.errors import PipelineError
from lstk_eye.pipeline.types import SegMask, Slide

_BADGE_RADIUS = 11
_BADGE_FILL = (139, 60, 0)  # dark blue, BGR
_BADGE_TEXT = (255, 255, 255)
_CONTOUR_COLOR = (180, 140, 60)  # muted blue, BGR
_SLIDE_COLOR = (0, 200, 255)  # amber, BGR
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def overlay_marks(image_bgr: np.ndarray, masks: list[SegMask]) -> np.ndarray:
    """Copy of the image with each region contoured and a numbered badge at
    its centroid. Badges near the border are nudged fully inside the frame."""
    out = image_bgr.copy()
    h, w = out.shape[:2]

    for m in masks:
        if m.mask is not None:
            contours, _ = cv2.findContours(
                m.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out, contours, -1, _CONTOUR_COLOR, 1, lineType=cv2.LINE_AA)
        else:
            x, y, bw, bh = m.bbox
            cv2.rectangle(
                out,
                (round(x * w), round(y * h)),
                (round((x + bw) * w), round((y + bh) * h)),
                _CONTOUR_COLOR,
                1,
            )

        cx = _clamp(round(m.centroid[0] * w), _BADGE_RADIUS + 1, w - _BADGE_RADIUS - 2)
        cy = _clamp(round(m.centroid[1] * h), _BADGE_RADIUS + 1, h - _BADGE_RADIUS - 2)
        cv2.circle(out, (cx, cy), _BADGE_RADIUS, _BADGE_FILL, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (cx, cy), _BADGE_RADIUS, _BADGE_TEXT, 1, lineType=cv2.LINE_AA)
        label = str(m.mark_id)
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.45, 1)
        cv2.putText(
            out,
            label,
            (cx - tw // 2, cy + th // 2),
            _FONT,
            0.45,
            _BADGE_TEXT,
            1,
            lineType=cv2.LINE_AA,
        )
    return out


def encode_png(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise PipelineError("PNG encoding failed")
    return buf.tobytes()


def encode_jpeg(image_bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise PipelineError("JPEG encoding failed")
    return buf.tobytes()


def draw_slides(image_bgr: np.ndarray, slides: list[Slide]) -> np.ndarray:
    """Debug render for the CLI demo: anchored slides get an arrow into their
    anchor point with a numbered label; text-only slides are listed top-left."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    corner_y = 22

    for s in slides:
        caption = f"{s.index + 1}: {s.label}"
        if s.anchor is None:
            _outlined_text(out, caption, (8, corner_y))
            corner_y += 22
            continue

        tip = (round(s.anchor[0] * w), round(s.anchor[1] * h))
        # Approach from the side that keeps the tail inside the frame.
        dx = -45 if tip[0] > w // 2 else 45
        dy = -35 if tip[1] > h // 2 else 35
        tail = (_clamp(tip[0] + dx, 0, w - 1), _clamp(tip[1] + dy, 0, h - 1))
        cv2.arrowedLine(out, tail, tip, _SLIDE_COLOR, 2, line_type=cv2.LINE_AA, tipLength=0.3)
        text_pos = (_clamp(tail[0] - 8, 4, w - 40), _clamp(tail[1] + (18 if dy > 0 else -8), 14, h - 6))
        _outlined_text(out, caption, text_pos)
    return out


def _outlined_text(image: np.ndarray, text: str, pos: tuple[int, int]) -> None:
    cv2.putText(image, text, pos, _FONT, 0.55, (0, 0, 0), 3, lineType=cv2.LINE_AA)
    cv2.putText(image, text, pos, _FONT, 0.55, _SLIDE_COLOR, 1, lineType=cv2.LINE_AA)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))
