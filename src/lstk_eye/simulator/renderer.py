"""Offline renderer for :class:`DisplayScene`, mirroring the OLED firmware.

Draws at native 128x64 into an 8-bit grayscale image (white on black, like
the SSD1306 behind the beamsplitter), then nearest-neighbor upscales so the
output stays pixel-faithful. Fonts differ slightly from the GFX 5x7 ROM font
but positions, sizes and primitives match the firmware drawing code.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from lstk_eye.protocol import (
    DISPLAY_H,
    DISPLAY_W,
    ArrowEl,
    ChevronEl,
    DisplayScene,
    TargetEl,
    TextEl,
)

_ARROW_HEAD_LEN = 5
_ARROW_HEAD_DEG = 45
_CHEVRON_ARM = 4  # px each arm extends back and sideways from the apex
_CHEVRON_STEP = 5  # px between the two apexes of the ">>"
_EDGE_MARGIN = 2  # px between the panel edge and the outer apex
_LABEL_GAP = 3  # px between the inner chevron extent and the label

# Outward unit direction for each display edge (screen convention: +y is down).
_EDGE_DIR = {
    "right": (1, 0),
    "left": (-1, 0),
    "up": (0, -1),
    "down": (0, 1),
}

_EDGE_CENTER = {
    "right": (DISPLAY_W - 1 - _EDGE_MARGIN, DISPLAY_H // 2),
    "left": (_EDGE_MARGIN, DISPLAY_H // 2),
    "up": (DISPLAY_W // 2, _EDGE_MARGIN),
    "down": (DISPLAY_W // 2, DISPLAY_H - 1 - _EDGE_MARGIN),
}


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """GFX text scale -> PIL font. Size 1 is the built-in bitmap font; larger
    sizes use the scalable default (8 px per GFX size unit, like a 6x8 cell)."""
    if size <= 1:
        return ImageFont.load_default()
    return ImageFont.load_default(size=8 * size)


def _draw_text(draw: ImageDraw.ImageDraw, el: TextEl) -> None:
    draw.text((el.x, el.y), el.text, fill=255, font=_font(el.size))


def _draw_arrow(draw: ImageDraw.ImageDraw, el: ArrowEl) -> None:
    tip = (el.x, el.y)
    shaft = math.radians(el.angle)
    tail = (el.x - el.length * math.cos(shaft), el.y - el.length * math.sin(shaft))
    draw.line([tip, tail], fill=255)
    for off in (-_ARROW_HEAD_DEG, _ARROW_HEAD_DEG):
        back = math.radians(el.angle + 180 + off)
        end = (
            el.x + _ARROW_HEAD_LEN * math.cos(back),
            el.y + _ARROW_HEAD_LEN * math.sin(back),
        )
        draw.line([tip, end], fill=255)


def _draw_target(draw: ImageDraw.ImageDraw, el: TargetEl) -> None:
    """Four corner brackets framing a square of half-size r around (x, y)."""
    arm = max(3, el.r // 2)
    x0, y0 = el.x - el.r, el.y - el.r
    x1, y1 = el.x + el.r, el.y + el.r
    for cx, cy, ax, ay in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        draw.line([(cx, cy), (cx + ax * arm, cy)], fill=255)
        draw.line([(cx, cy), (cx, cy + ay * arm)], fill=255)


def _draw_chevron(draw: ImageDraw.ImageDraw, el: ChevronEl) -> None:
    dx, dy = _EDGE_DIR[el.edge]
    px, py = -dy, dx  # perpendicular, along the edge
    # Explicit tip position wins (the server insets it when the optics crop
    # the panel); -1/-1 falls back to the physical edge.
    cx, cy = (el.x, el.y) if el.x >= 0 and el.y >= 0 else _EDGE_CENTER[el.edge]
    # Two solid triangles pointing outward, apexes 9 px apart (matches the
    # firmware's fillTriangle rendering).
    for i in range(2):
        ax = cx - i * 9 * dx
        ay = cy - i * 9 * dy
        base1 = (ax - 6 * dx + 5 * px, ay - 6 * dy + 5 * py)
        base2 = (ax - 6 * dx - 5 * px, ay - 6 * dy - 5 * py)
        draw.polygon([(ax, ay), base1, base2], fill=255)
    if not el.label:
        return
    font = _font(1)
    w = draw.textlength(el.label, font=font)
    bbox = font.getbbox(el.label)
    h = bbox[3] - bbox[1]
    # Innermost point of the chevron cluster, then a small gap.
    ix = cx - (_CHEVRON_STEP + _CHEVRON_ARM + _LABEL_GAP) * dx
    iy = cy - (_CHEVRON_STEP + _CHEVRON_ARM + _LABEL_GAP) * dy
    if dx > 0:
        pos = (ix - w, iy - h / 2)
    elif dx < 0:
        pos = (ix, iy - h / 2)
    elif dy < 0:
        pos = (ix - w / 2, iy)
    else:
        pos = (ix - w / 2, iy - h)
    draw.text(pos, el.label, fill=255, font=font)


def render_scene(scene: DisplayScene, scale: int = 4) -> Image.Image:
    """Render a scene to a grayscale image of size (128*scale, 64*scale)."""
    img = Image.new("L", (DISPLAY_W, DISPLAY_H), 0)
    draw = ImageDraw.Draw(img)
    for el in scene.els:
        if isinstance(el, TextEl):
            _draw_text(draw, el)
        elif isinstance(el, ArrowEl):
            _draw_arrow(draw, el)
        elif isinstance(el, ChevronEl):
            _draw_chevron(draw, el)
        elif isinstance(el, TargetEl):
            _draw_target(draw, el)
    if scale != 1:
        img = img.resize((DISPLAY_W * scale, DISPLAY_H * scale), Image.Resampling.NEAREST)
    return img


def save_scene(scene: DisplayScene, path: str | Path, scale: int = 4) -> Path:
    """Render a scene and write it as PNG, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    render_scene(scene, scale=scale).save(path)
    return path
