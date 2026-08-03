"""Tests for template-matching relocalization.

Synthetic scene: a high-contrast checkerboard patch with a unique glyph on a
textured noise background. The capture frame (640x480) defines the template;
preview frames (320x240) render the same patch at known offsets and scales.
"""

import cv2
import numpy as np
import pytest

from lstk_eye.config import RelocConfig
from lstk_eye.pipeline.reloc.template import TemplateRelocalizerFactory

CAPTURE_W, CAPTURE_H = 640, 480
PREVIEW_W, PREVIEW_H = 320, 240

# Patch footprint in normalized coordinates (same on every frame size).
PATCH_W, PATCH_H = 0.25, 0.25
CAPTURE_CENTER = (0.5, 0.5)
BBOX = (
    CAPTURE_CENTER[0] - PATCH_W / 2,
    CAPTURE_CENTER[1] - PATCH_H / 2,
    PATCH_W,
    PATCH_H,
)


def _make_patch(w: int = 160, h: int = 120) -> np.ndarray:
    """Checkerboard plus an asymmetric glyph: high contrast, aperiodic, so the
    correlation peak is unique (a bare checkerboard repeats every period)."""
    ys, xs = np.mgrid[0:h, 0:w]
    patch = np.where(((xs // 14) + (ys // 14)) % 2 == 0, 230, 25).astype(np.uint8)
    cv2.circle(patch, (w // 3, h // 2), h // 4, 255, -1)
    cv2.circle(patch, (w // 3, h // 2), h // 8, 0, -1)
    cv2.line(patch, (0, h - 1), (w - 1, 0), 0, 7)
    tri = np.array([[w - 12, 12], [w - 60, 20], [w - 20, 56]])
    cv2.fillPoly(patch, [tri], 90)
    return patch


_PATCH = _make_patch()


def _background(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """Low-contrast smooth noise blobs (upscaled coarse noise)."""
    small = rng.integers(70, 130, size=(h // 16, w // 16), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _render(
    w: int,
    h: int,
    center: tuple[float, float],
    rng: np.random.Generator,
    scale: float = 1.0,
) -> np.ndarray:
    """Grayscale frame with the patch centered at ``center`` (normalized)."""
    frame = _background(w, h, rng)
    pw = int(round(PATCH_W * scale * w))
    ph = int(round(PATCH_H * scale * h))
    patch = cv2.resize(_PATCH, (pw, ph), interpolation=cv2.INTER_AREA)
    x0 = int(round(center[0] * w - pw / 2))
    y0 = int(round(center[1] * h - ph / 2))
    assert 0 <= x0 and 0 <= y0 and x0 + pw <= w and y0 + ph <= h, "patch off-frame"
    frame[y0 : y0 + ph, x0 : x0 + pw] = patch
    return frame


def _capture_bgr(rng: np.random.Generator) -> np.ndarray:
    gray = _render(CAPTURE_W, CAPTURE_H, CAPTURE_CENTER, rng)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _preview(center: tuple[float, float], rng: np.random.Generator, scale: float = 1.0):
    return _render(PREVIEW_W, PREVIEW_H, center, rng, scale=scale)


@pytest.fixture
def cfg() -> RelocConfig:
    return RelocConfig(
        appear_conf=0.60,
        disappear_conf=0.45,
        miss_hide=4,
        ema=0.4,
        scales=[0.85, 1.0, 1.18],
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _new_tracker(cfg: RelocConfig, rng: np.random.Generator):
    return TemplateRelocalizerFactory(cfg).create(_capture_bgr(rng), BBOX)


@pytest.mark.parametrize(
    "center",
    [(0.5, 0.5), (0.6, 0.55), (0.38, 0.58), (0.55, 0.4), (0.3, 0.35)],
)
def test_finds_patch_at_translations(cfg, rng, center):
    tracker = _new_tracker(cfg, rng)
    res = tracker.update(_preview(center, rng))
    assert res.found
    assert res.confidence >= cfg.appear_conf
    assert res.center is not None
    assert abs(res.center[0] - center[0]) < 0.04
    assert abs(res.center[1] - center[1]) < 0.04


def test_found_persists_across_translated_sequence(cfg, rng):
    tracker = _new_tracker(cfg, rng)
    xs = np.linspace(0.35, 0.65, 8)
    for x in xs:
        res = tracker.update(_preview((float(x), 0.5), rng))
        assert res.found


def test_disappear_is_lazy_exactly_miss_hide(cfg, rng):
    tracker = _new_tracker(cfg, rng)
    res = tracker.update(_preview((0.5, 0.5), rng))
    assert res.found
    for i in range(1, cfg.miss_hide + 1):
        res = tracker.update(_background(PREVIEW_W, PREVIEW_H, rng))
        assert res.confidence < cfg.disappear_conf, "noise frame matched too well"
        expected = i < cfg.miss_hide
        assert res.found is expected, f"after {i} misses expected found={expected}"


def test_reappear_is_immediate_for_textured_targets(cfg, rng):
    """The structure gate makes a confident match trustworthy on sight, so a
    textured target re-shows on the first verified frame - recovery speed
    matters to the eyes. (Textureless templates, where the gate is off, fall
    back to two-frame temporal consistency instead.)"""
    tracker = _new_tracker(cfg, rng)
    tracker.update(_preview((0.5, 0.5), rng))
    for _ in range(cfg.miss_hide):
        tracker.update(_background(PREVIEW_W, PREVIEW_H, rng))
    res = tracker.update(_preview((0.45, 0.55), rng))
    assert res.found, "verified reappearance must be immediate"
    assert res.confidence >= cfg.appear_conf


def test_visible_teleport_counts_as_miss(cfg, rng):
    """While visible, a confident match that teleports far from the smoothed
    center is an outlier: the marker holds instead of jumping."""
    tracker = _new_tracker(cfg, rng)
    for _ in range(4):
        res = tracker.update(_preview((0.35, 0.5), rng))
    settled = res.center
    res = tracker.update(_preview((0.80, 0.5), rng))  # 0.45 jump > outlier gate
    assert res.found, "one outlier only spends miss budget"
    assert res.center is not None
    assert abs(res.center[0] - settled[0]) < 0.02, "the marker must not teleport"


def test_center_none_until_first_confident_match(cfg, rng):
    tracker = _new_tracker(cfg, rng)
    res = tracker.update(_background(PREVIEW_W, PREVIEW_H, rng))
    # The tracker starts visible (it was built from a frame containing the
    # target), so one weak frame only spends miss budget - but no position is
    # known yet.
    assert res.found
    assert res.center is None
    res = tracker.update(_preview((0.5, 0.5), rng))
    assert res.center is not None


def test_ema_center_moves_monotonically_after_jump(cfg, rng):
    tracker = _new_tracker(cfg, rng)
    a, b = (0.40, 0.5), (0.55, 0.5)  # jump below the outlier gate
    for _ in range(6):
        tracker.update(_preview(a, rng))
    dists = []
    for _ in range(5):
        res = tracker.update(_preview(b, rng))
        assert res.found
        assert res.center is not None
        dists.append(float(np.hypot(res.center[0] - b[0], res.center[1] - b[1])))
    for prev, cur in zip(dists, dists[1:], strict=False):
        assert cur < prev, f"center did not approach target monotonically: {dists}"


@pytest.mark.parametrize("scale", [0.85, 1.18])
def test_scale_robustness(cfg, rng, scale):
    tracker = _new_tracker(cfg, rng)
    res = tracker.update(_preview((0.5, 0.5), rng, scale=scale))
    assert res.found
    assert res.center is not None
    assert abs(res.center[0] - 0.5) < 0.04
    assert abs(res.center[1] - 0.5) < 0.04


def test_flat_impostor_rejected_by_edge_check(cfg, rng):
    """A smooth blob with matching brightness (clothing) must not count as
    the textured target: the Laplacian correlation gate rejects it."""
    tracker = _new_tracker(cfg, rng)
    tracker.update(_preview((0.5, 0.5), rng))  # lock onto the real target

    # Frame with a flat bright rectangle instead of the patterned patch.
    flat = _background(PREVIEW_W, PREVIEW_H, rng)
    ph, pw = int(0.2 * PREVIEW_H), int(0.2 * PREVIEW_W)
    flat[20 : 20 + ph, 30 : 30 + pw] = 200
    for _ in range(cfg.miss_hide):
        res = tracker.update(flat)
    assert not res.found, "flat impostor kept the target visible"
