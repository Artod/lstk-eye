"""Tests for the segmentation backends."""

import math

import cv2
import numpy as np
import pytest

from lstk_eye.config import SegmenterConfig
from lstk_eye.pipeline.segmentation.mock import MockSegmenter
from lstk_eye.pipeline.types import SegMask

W, H = 640, 480

# (draw function args) -> expected normalized centroid
SHAPE_CENTERS = [
    (130 / W, 130 / H),  # rectangle 1
    (480 / W, 150 / H),  # rectangle 2
    (320 / W, 360 / H),  # circle
    (130 / W, 380 / H),  # rectangle 3 (small)
]


def make_test_image() -> np.ndarray:
    """Plain gray background with four solid dark shapes at known centers."""
    img = np.full((H, W, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (80, 80), (180, 180), (40, 40, 40), -1)
    cv2.rectangle(img, (400, 100), (560, 200), (40, 40, 40), -1)
    cv2.circle(img, (320, 360), 50, (40, 40, 40), -1)
    cv2.rectangle(img, (100, 350), (160, 410), (40, 40, 40), -1)
    return img


@pytest.fixture
def test_image() -> np.ndarray:
    return make_test_image()


@pytest.fixture
def cfg() -> SegmenterConfig:
    return SegmenterConfig(backend="mock", min_area=0.0008, max_masks=25)


def assert_valid_masks(masks: list[SegMask], shape: tuple[int, int]) -> None:
    h, w = shape
    assert [m.mark_id for m in masks] == list(range(1, len(masks) + 1))
    areas = [m.area for m in masks]
    assert areas == sorted(areas, reverse=True)
    for m in masks:
        x, y, bw, bh = m.bbox
        assert 0.0 <= x and x + bw <= 1.0 + 1e-6
        assert 0.0 <= y and y + bh <= 1.0 + 1e-6
        assert 0.0 <= m.centroid[0] <= 1.0 and 0.0 <= m.centroid[1] <= 1.0
        assert 0.0 < m.area <= 1.0
        if m.mask is not None:
            assert m.mask.dtype == bool
            assert m.mask.shape == (h, w)
            ys, xs = np.nonzero(m.mask)
            assert len(xs) > 0
            # All mask pixels fall inside the reported bbox (1 px slack for rounding).
            assert xs.min() >= x * w - 1 and xs.max() <= (x + bw) * w + 1
            assert ys.min() >= y * h - 1 and ys.max() <= (y + bh) * h + 1


def test_mock_finds_known_shapes(test_image, cfg):
    masks = MockSegmenter(cfg).segment(test_image)

    assert len(masks) >= 3
    assert_valid_masks(masks, (H, W))
    assert all(m.mask is not None for m in masks)

    matched = 0
    for ex, ey in SHAPE_CENTERS:
        if any(math.dist((ex, ey), m.centroid) < 0.05 for m in masks):
            matched += 1
    assert matched >= 3


def test_mock_respects_max_masks(test_image):
    cfg = SegmenterConfig(backend="mock", min_area=0.0008, max_masks=2)
    masks = MockSegmenter(cfg).segment(test_image)
    assert len(masks) == 2
    assert [m.mark_id for m in masks] == [1, 2]


def test_mock_min_area_filters_small_shapes(test_image):
    # The smallest shape is ~61x61 px = ~1.2% of the frame; a 2% floor drops it.
    cfg = SegmenterConfig(backend="mock", min_area=0.02, max_masks=25)
    masks = MockSegmenter(cfg).segment(test_image)
    assert all(m.area >= 0.02 for m in masks)
    assert len(masks) == 3


def test_mock_empty_on_uniform_image(cfg):
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    assert MockSegmenter(cfg).segment(img) == []


def test_mock_handles_inverted_polarity(cfg):
    """Light shapes on a dark background segment just as well."""
    img = np.full((H, W, 3), 50, dtype=np.uint8)
    cv2.rectangle(img, (80, 80), (180, 180), (220, 220, 220), -1)
    cv2.circle(img, (450, 300), 60, (220, 220, 220), -1)
    masks = MockSegmenter(cfg).segment(img)
    assert len(masks) == 2
    assert_valid_masks(masks, (H, W))


@pytest.mark.slow
def test_fastsam_on_synthetic_image(test_image):
    pytest.importorskip("ultralytics")
    from lstk_eye.pipeline.segmentation.fastsam import FastSAMSegmenter

    cfg = SegmenterConfig(backend="fastsam", device="cpu")
    masks = FastSAMSegmenter(cfg).segment(test_image)

    assert isinstance(masks, list)
    assert all(isinstance(m, SegMask) for m in masks)
    assert_valid_masks(masks, (H, W))
    assert len(masks) <= cfg.max_masks
