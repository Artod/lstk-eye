"""Chat-model semantics (history, reset) and the calibration flow."""

import cv2
import httpx
import numpy as np
import pytest

from lstk_eye.config import load_config
from lstk_eye.server import create_app
from lstk_eye.simulator import SimulatedGlasses
from lstk_eye.testimage import make_test_image


@pytest.fixture()
def cfg(tmp_path):
    return load_config(
        None,
        profile="mock",
        storage={"dir": str(tmp_path / "runs")},
        server={"zeroconf": False},
        config_path=str(tmp_path / "lstk-eye.toml"),
    )


@pytest.fixture()
def rig(cfg):
    app = create_app(cfg)
    glasses = SimulatedGlasses(
        base_url="http://test", transport=httpx.ASGITransport(app=app)
    )
    return glasses, app


def _session(app):
    return app.state.sessions.get("sim")


def _texts(scene):
    return [e.text for e in scene.els if e.t == "text"]


# --- chat history and reset ---


def test_history_accumulates_across_asks(rig):
    glasses, app = rig
    glasses.capture(make_test_image())
    glasses.ask(text="how do I check battery voltage")
    glasses.capture(make_test_image())
    glasses.ask(text="what about the current reading instead")
    history = _session(app)._history
    assert len(history) == 2
    assert history[0][0] == "how do I check battery voltage"
    assert history[1][0] == "what about the current reading instead"
    assert all(answer for _, answer in history)


def test_history_reaches_planner(rig):
    glasses, app = rig
    seen = []
    planner = app.state.runtime.planner
    original = planner.plan

    def spy(marked_png, question, marks, history=None):
        seen.append(list(history or []))
        return original(marked_png, question, marks, history=history)

    planner.plan = spy
    glasses.capture(make_test_image())
    glasses.ask(text="how do I check battery voltage")
    glasses.capture(make_test_image())
    glasses.ask(text="and what about resistance")
    assert seen[0] == []
    assert len(seen[1]) == 1
    assert seen[1][0][0] == "how do I check battery voltage"


def test_double_click_reset_clears_everything(rig):
    glasses, app = rig
    glasses.capture(make_test_image())
    resp = glasses.ask(text="how do I check battery voltage")
    assert resp.active

    done = glasses.reset()
    assert not done.active
    assert "done" in _texts(done.scene)
    sess = _session(app)
    assert sess._history == []
    assert sess._photos == []
    assert not sess.active

    # A fresh ask after reset starts from a clean slate: no photo yet.
    fresh = glasses.ask(text="what is this thing here")
    assert not fresh.active
    assert any("no photo" in t for t in _texts(fresh.scene))


def test_reset_works_while_idle(rig):
    glasses, _ = rig
    done = glasses.reset()
    assert not done.active
    assert done.scene is not None


# --- calibration flow ---


def _frame_with_marker(cx: float, cy: float, size: int = 120) -> np.ndarray:
    """Synthetic camera frame: white background, ArUco id 0 centered at the
    normalized (cx, cy)."""
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, size
    )
    x0 = int(cx * 640) - size // 2
    y0 = int(cy * 480) - size // 2
    frame[y0 : y0 + size, x0 : x0 + size] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return frame


def test_calibration_flow_end_to_end(rig, cfg):
    glasses, app = rig

    start = glasses.ask(text="calibrate")
    assert start.active
    assert any(e.t == "target" for e in start.scene.els), "crosshair expected"

    # The wearer aligns the physical marker with each crosshair and clicks.
    # Simulated: the marker appears at a distinct camera point per click.
    cam_points = [(0.55, 0.5), (0.75, 0.5), (0.55, 0.35)]
    for i, (cx, cy) in enumerate(cam_points):
        ack = glasses.capture(_frame_with_marker(cx, cy))
        assert ack.scene is not None
        if i < len(cam_points) - 1:
            assert any(e.t == "target" for e in ack.scene.els), "next crosshair"
    assert "calibrated" in _texts(ack.scene)

    # The shared calibration now maps each collected camera point back to the
    # display point it was aligned with (composer center etc. for default pads).
    cal = app.state.runtime.calibration
    comp = app.state.runtime.composer
    expected = [comp.center, (comp.x1 - 6, comp.center[1]), (comp.center[0], comp.y0 + 6)]
    for cam, disp in zip(cam_points, expected, strict=True):
        got = cal.to_display(cam)
        assert abs(got[0] - disp[0]) <= 1 and abs(got[1] - disp[1]) <= 1

    # Persisted to the config file for the next server start.
    saved = cfg.config_path.read_text()
    assert "[calibration]" in saved and "center_x" in saved


def test_calibration_retries_on_missing_marker(rig):
    glasses, _ = rig
    glasses.ask(text="calibrate")
    blank = np.full((480, 640, 3), 255, dtype=np.uint8)
    ack = glasses.capture(blank)
    assert any("no marker" in t for t in _texts(ack.scene))
    # Still on point 1; a good capture proceeds.
    ack = glasses.capture(_frame_with_marker(0.5, 0.5))
    assert any(e.t == "target" for e in ack.scene.els)


def test_calibration_cancelled_by_reset(rig, cfg):
    glasses, app = rig
    glasses.ask(text="калибровка")
    assert _session(app)._calib is not None
    glasses.reset()
    assert _session(app)._calib is None
