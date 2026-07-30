"""Tests for the Set-of-Marks overlay and debug rendering."""

import cv2
import numpy as np
import pytest

from lstk_eye.config import SegmenterConfig
from lstk_eye.pipeline.marks import draw_slides, encode_jpeg, encode_png, overlay_marks
from lstk_eye.pipeline.segmentation.mock import MockSegmenter
from lstk_eye.pipeline.types import SegMask, Slide

W, H = 640, 480


@pytest.fixture
def test_image() -> np.ndarray:
    img = np.full((H, W, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (80, 80), (180, 180), (40, 40, 40), -1)
    cv2.rectangle(img, (400, 100), (560, 200), (40, 40, 40), -1)
    cv2.circle(img, (320, 360), 50, (40, 40, 40), -1)
    return img


@pytest.fixture
def masks(test_image) -> list[SegMask]:
    cfg = SegmenterConfig(backend="mock", min_area=0.0008, max_masks=25)
    result = MockSegmenter(cfg).segment(test_image)
    assert len(result) >= 3
    return result


def patch_around(image: np.ndarray, centroid: tuple[float, float], size: int = 24) -> np.ndarray:
    h, w = image.shape[:2]
    cx, cy = round(centroid[0] * w), round(centroid[1] * h)
    return image[max(0, cy - size) : cy + size, max(0, cx - size) : cx + size]


def test_overlay_marks_draws_badges(test_image, masks):
    before = test_image.copy()
    out = overlay_marks(test_image, masks)

    assert out.shape == test_image.shape
    assert out.dtype == test_image.dtype
    assert np.array_equal(test_image, before)  # input untouched
    for m in masks:
        assert not np.array_equal(patch_around(out, m.centroid), patch_around(test_image, m.centroid))


def test_overlay_marks_nudges_border_badges(test_image):
    corner = SegMask(mark_id=1, bbox=(0.0, 0.0, 0.1, 0.1), centroid=(0.0, 0.0), area=0.01)
    out = overlay_marks(test_image, [corner])
    assert out.shape == test_image.shape
    # Badge is drawn fully inside the frame, so the corner region changed.
    assert not np.array_equal(out[:30, :30], test_image[:30, :30])


def test_overlay_marks_without_mask_array(test_image):
    m = SegMask(mark_id=1, bbox=(0.2, 0.2, 0.3, 0.3), centroid=(0.35, 0.35), area=0.09, mask=None)
    out = overlay_marks(test_image, [m])
    assert not np.array_equal(out, test_image)


def test_encode_png_roundtrip(test_image, masks):
    marked = overlay_marks(test_image, masks)
    png = encode_png(marked)
    assert isinstance(png, bytes) and len(png) > 0
    decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == marked.shape
    assert np.array_equal(decoded, marked)  # PNG is lossless


def test_encode_jpeg_roundtrip(test_image):
    jpg = encode_jpeg(test_image, quality=85)
    assert isinstance(jpg, bytes) and len(jpg) > 0
    decoded = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == test_image.shape


def test_draw_slides_smoke(test_image):
    slides = [
        Slide(index=0, total=3, label="red probe here", anchor=(0.2, 0.27), mark_id=1),
        Slide(index=1, total=3, label="black probe here", anchor=(0.75, 0.31), mark_id=2),
        Slide(index=2, total=3, label="wait 30 seconds", anchor=None, mark_id=None),
    ]
    out = draw_slides(test_image, slides)
    assert out.shape == test_image.shape
    assert not np.array_equal(out, test_image)


def test_draw_slides_empty(test_image):
    out = draw_slides(test_image, [])
    assert np.array_equal(out, test_image)
    assert out is not test_image
