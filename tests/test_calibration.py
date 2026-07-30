"""WindowCalibration: forward mapping, window tests, edge picking, fitting.

Hand-computed expectations use the default config (center 0.5/0.5, window
0.4/0.4), where the forward map is dx = (x-0.5)/0.4*128+64 and
dy = (y-0.5)/0.4*64+32, with inverse x = (dx-64)/128*0.4+0.5 and
y = (dy-32)/64*0.4+0.5.
"""

import pytest

from lstk_eye.calibration import WindowCalibration
from lstk_eye.config import CalibrationConfig
from lstk_eye.errors import ConfigError


@pytest.fixture
def cal() -> WindowCalibration:
    return WindowCalibration(CalibrationConfig())


def test_center_maps_to_display_center(cal):
    assert cal.to_display((0.5, 0.5)) == (64, 32)


@pytest.mark.parametrize(
    ("pt", "expected"),
    [
        ((0.6, 0.6), (96, 48)),  # +0.1 in camera = +32 px x, +16 px y
        ((0.30625, 0.675), (2, 60)),  # inverse of (2, 60)
        ((0.3, 0.3), (0, 0)),  # top-left panel corner
        ((0.696875, 0.69375), (127, 63)),  # bottom-right panel corner
        ((1.0, 0.5), (224, 32)),  # unclamped: far outside the panel
        ((0.0, 0.0), (-96, -48)),  # unclamped, negative
    ],
)
def test_known_points(cal, pt, expected):
    assert cal.to_display(pt) == expected


def test_asymmetric_config():
    cal = WindowCalibration(
        CalibrationConfig(center_x=0.55, center_y=0.45, window_w=0.5, window_h=0.25)
    )
    assert cal.to_display((0.55, 0.45)) == (64, 32)
    # dx = (0.125/0.5)*128+64 = 96; dy = (0.05/0.25)*64+32 = 44.8 -> 45
    assert cal.to_display((0.675, 0.5)) == (96, 45)


def test_in_window_at_panel_corners(cal):
    assert cal.in_window((0.3, 0.3))  # maps to (0, 0)
    assert cal.in_window((0.696875, 0.69375))  # maps to (127, 63)
    assert not cal.in_window((0.7, 0.5))  # maps to (128, 32): one px out


def test_in_window_margin(cal):
    assert not cal.in_window((0.3, 0.3), margin_px=1)
    assert not cal.in_window((0.696875, 0.69375), margin_px=1)
    # Center survives the largest margin the y axis allows (32 <= 63-31)...
    assert cal.in_window((0.5, 0.5), margin_px=31)
    # ...and fails once the margin band on y becomes empty.
    assert not cal.in_window((0.5, 0.5), margin_px=32)


@pytest.mark.parametrize(
    ("pt", "edge"),
    [
        ((0.0, 0.5), "left"),  # (-96, 32)
        ((1.0, 0.5), "right"),  # (224, 32)
        ((0.5, 0.0), "up"),  # (64, -48)
        ((0.5, 1.0), "down"),  # (64, 112)
    ],
)
def test_edge_for_sides(cal, pt, edge):
    assert cal.edge_for(pt) == edge


@pytest.mark.parametrize(
    ("pt", "edge"),
    [
        # (-96, -48): overshoots 96/128 = 48/64 = 0.75 tie -> prefer horizontal.
        ((0.0, 0.0), "left"),
        # (224, -48): 97/128 > 48/64 -> right wins.
        ((1.0, 0.0), "right"),
        # (-96, 112): 49/64 > 96/128 -> the bottom edge wins.
        ((0.0, 1.0), "down"),
        # (224, 112): 49/64 > 97/128 -> down again.
        ((1.0, 1.0), "down"),
        # (-64, -48): 64/128 = 0.5 < 48/64 = 0.75 -> vertical dominates.
        ((0.1, 0.0), "up"),
        # (-96, 24): out only horizontally.
        ((0.0, 0.45), "left"),
    ],
)
def test_edge_for_corners_and_dominance(cal, pt, edge):
    assert cal.edge_for(pt) == edge


def _pairs_for(cx, cy, ww, wh, pixels):
    """Synthesize exact (camera_pt, display_px) pairs by inverting the model."""
    pairs = []
    for dx, dy in pixels:
        cam_x = (dx - 64) / 128 * ww + cx
        cam_y = (dy - 32) / 64 * wh + cy
        pairs.append(((cam_x, cam_y), (dx, dy)))
    return pairs


def test_fit_recovers_parameters():
    cx, cy, ww, wh = 0.52, 0.47, 0.36, 0.44
    pairs = _pairs_for(cx, cy, ww, wh, [(0, 5), (40, 30), (90, 50), (127, 63)])
    cal = WindowCalibration.fit(pairs)
    assert cal.center_x == pytest.approx(cx, abs=1e-6)
    assert cal.center_y == pytest.approx(cy, abs=1e-6)
    assert cal.window_w == pytest.approx(ww, abs=1e-6)
    assert cal.window_h == pytest.approx(wh, abs=1e-6)
    for cam_pt, disp_px in pairs:
        assert cal.to_display(cam_pt) == disp_px


def test_fit_two_pairs_minimum():
    pairs = _pairs_for(0.5, 0.5, 0.4, 0.4, [(10, 12), (100, 55)])
    cal = WindowCalibration.fit(pairs)
    assert cal.window_w == pytest.approx(0.4, abs=1e-6)
    assert cal.window_h == pytest.approx(0.4, abs=1e-6)


@pytest.mark.parametrize("pairs", [[], [((0.5, 0.5), (64, 32))]])
def test_fit_insufficient_pairs(pairs):
    with pytest.raises(ConfigError):
        WindowCalibration.fit(pairs)


def test_fit_degenerate_axis():
    # Same camera x in both pairs: the x axis cannot be fitted.
    pairs = [((0.4, 0.4), (10, 10)), ((0.4, 0.6), (10, 42))]
    with pytest.raises(ConfigError):
        WindowCalibration.fit(pairs)
