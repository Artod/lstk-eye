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
    # Hermetic config: pin an explicit (empty) TOML so a developer's local
    # ./lstk-eye.toml (pads, camera rotation) can never leak into tests.
    cfg_file = tmp_path / "lstk-eye.toml"
    cfg_file.write_text("")
    return load_config(
        cfg_file,
        profile="mock",
        storage={"dir": str(tmp_path / "runs")},
        server={"zeroconf": False},
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
    from lstk_eye.display import CROSSHAIR_R

    inset = CROSSHAIR_R + 2
    expected = [
        comp.center,
        (comp.x1 - inset, comp.center[1]),
        (comp.center[0], comp.y0 + inset),
    ]
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


def test_calibration_live_preview_feedback(rig):
    glasses, app = rig
    glasses.ask(text="calibrate")
    blank = np.full((480, 640, 3), 255, dtype=np.uint8)

    resp = glasses.preview(blank, last_seq=0)
    assert resp.active
    assert any("no marker" in t for t in _texts(resp.scene))

    resp = glasses.preview(_frame_with_marker(0.5, 0.5), last_seq=resp.scene.seq)
    assert any("marker OK" in t for t in _texts(resp.scene))

    # Unchanged state does not churn the scene.
    again = glasses.preview(_frame_with_marker(0.5, 0.5), last_seq=resp.scene.seq)
    assert again.scene is None


def test_calibration_rejects_small_and_edge_markers(rig):
    glasses, app = rig
    glasses.ask(text="calibrate")

    tiny = glasses.capture(_frame_with_marker(0.5, 0.5, size=20))  # 20/640 < 0.045
    assert any("closer" in t for t in _texts(tiny.scene))
    assert _session(app)._calib.idx == 0, "rejected click must not advance"

    edge = glasses.capture(_frame_with_marker(0.055, 0.5, size=60))
    assert any("near edge" in t for t in _texts(edge.scene))
    assert _session(app)._calib.idx == 0

    good = glasses.capture(_frame_with_marker(0.5, 0.5))
    assert _session(app)._calib.idx == 1
    assert any(e.t == "target" for e in good.scene.els)


def test_calibration_frames_saved_for_debugging(rig, cfg):
    glasses, _ = rig
    glasses.ask(text="calibrate")
    glasses.capture(np.full((480, 640, 3), 255, dtype=np.uint8))  # miss
    glasses.capture(_frame_with_marker(0.5, 0.5))  # hit
    calib_dir = cfg.storage.dir / "calibration"
    names = sorted(p.name for p in calib_dir.iterdir())
    assert any("miss" in n for n in names)
    assert any("_ok" in n for n in names)
    assert any("detect" in n for n in names)


def test_small_photo_rejected_from_buffer(rig):
    glasses, app = rig
    qvga = cv2.resize(make_test_image(), (320, 240))
    ack = glasses.capture(qvga)
    assert ack.count == 0, "preview-sized frames must not enter the photo buffer"
    assert any("bad photo" in t for t in _texts(ack.scene))

    # The planner never sees it: an ask still reports no photo.
    resp = glasses.ask(text="what is this object here")
    assert any("no photo" in t for t in _texts(resp.scene))


def test_error_scene_names_the_exit(rig, cfg):
    from lstk_eye.server import create_app as _make

    comp_scene = _make(cfg).state.runtime.composer.error("boom", seq=1)
    texts = [e.text for e in comp_scene.els if e.t == "text"]
    assert any("2click = reset" in t for t in texts)


def test_question_during_calibration_never_runs_pipeline(rig):
    glasses, app = rig
    called = []
    planner = app.state.runtime.planner
    original = planner.plan
    planner.plan = lambda *a, **k: (called.append(1), original(*a, **k))[1]

    glasses.ask(text="calibrate")
    resp = glasses.ask(text="почему калибровка не работает")
    assert resp.active, "calibration must survive an unrelated question"
    assert not called, "the chat pipeline must not run during calibration"
    assert any("2click" in t for t in _texts(resp.scene))
    assert _session(app)._calib is not None


def test_idle_voice_intent_gets_a_pointer(rig):
    glasses, _ = rig
    resp = glasses.ask(text="repeat")
    assert not resp.active
    texts = _texts(resp.scene)
    assert any("no session" in t for t in texts)
    assert any("hold btn" in t for t in texts)


def test_camera_rotation_normalizes_frames(tmp_path):
    cfg_file = tmp_path / "lstk-eye.toml"
    cfg_file.write_text("")
    cfg = load_config(
        cfg_file,
        profile="mock",
        storage={"dir": str(tmp_path / "runs")},
        server={"zeroconf": False},
        camera={"rotation": 270},
    )
    app = create_app(cfg)
    glasses = SimulatedGlasses(base_url="http://test", transport=httpx.ASGITransport(app=app))

    glasses.ask(text="calibrate")
    upright = _frame_with_marker(0.70, 0.50)
    # A camera mounted 90 deg CCW delivers the upright scene rotated CW; the
    # server's rotation=270 must undo that before detection.
    delivered = cv2.rotate(upright, cv2.ROTATE_90_CLOCKWISE)
    ack = glasses.capture(delivered)
    sess = app.state.sessions.get("sim")
    assert sess._calib.idx == 1, _texts(ack.scene)
    (cx, cy), _ = sess._calib.pairs[0][0], sess._calib.pairs[0][1]
    assert abs(cx - 0.70) < 0.03 and abs(cy - 0.50) < 0.03


def test_foreign_marker_families_accepted(rig):
    glasses, app = rig
    glasses.ask(text="calibrate")
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    tag = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5), 0, 120
    )
    frame[180:300, 260:380] = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)
    glasses.capture(frame)
    assert app.state.sessions.get("sim")._calib.idx == 1, "AprilTag must be accepted"


def test_too_close_marker_rejected(rig):
    glasses, app = rig
    glasses.ask(text="calibrate")
    huge = glasses.capture(_frame_with_marker(0.5, 0.5, size=260))  # 260/640 > 0.35
    assert any("further" in t for t in _texts(huge.scene))
    assert _session(app)._calib.idx == 0


def test_voice_cancel_of_calibration_fully_deactivates(rig):
    """Regression: 'cancel' during calibration left active=True with zero
    slides; a following 'repeat'/'back' crashed to an internal-error scene."""
    glasses, app = rig
    glasses.ask(text="calibrate")
    off = glasses.ask(text="cancel")
    assert not off.active
    sess = _session(app)
    assert sess._calib is None and not sess.active

    polled = glasses.poll_scene(-1)
    assert not polled.active, "GET /scene must agree the session ended"

    for word in ("repeat", "back", "next"):
        resp = glasses.ask(text=word)
        assert not resp.active
        assert not any("internal error" in t for t in _texts(resp.scene) if t)


def test_stop_calibration_phrase_cancels_not_restarts(rig):
    glasses, app = rig
    glasses.ask(text="calibrate")
    glasses.capture(_frame_with_marker(0.5, 0.5))
    assert _session(app)._calib.idx == 1
    off = glasses.ask(text="стоп калибровка")
    assert not off.active
    assert _session(app)._calib is None, "cancel must win over restart"


def test_empty_transcript_during_calibration_keeps_crosshair(rig):
    glasses, app = rig
    glasses.ask(text="calibrate")
    resp = glasses.ask(text="   ")
    assert resp.active
    assert any(e.t == "target" for e in resp.scene.els), "crosshair must survive"
    assert any("didn't catch" in t for t in _texts(resp.scene))
    assert _session(app)._calib is not None


def test_punctuation_only_question_treated_as_silence(rig):
    glasses, app = rig
    called = []
    planner = app.state.runtime.planner
    original = planner.plan
    planner.plan = lambda *a, **k: (called.append(1), original(*a, **k))[1]
    glasses.capture(make_test_image())
    resp = glasses.ask(text="?!.")
    assert not called, "punctuation-only transcript must not reach the planner"
    assert any("didn't catch" in t for t in _texts(resp.scene))
    assert resp.session_id == ""


def test_prompt_screen_auto_restores_slide(rig, monkeypatch):
    glasses, app = rig
    glasses.capture(make_test_image())
    start = glasses.ask(text="find the phone here")
    assert start.active

    # An empty transcript mid-session raises a transient prompt over the
    # live task (follow-up questions now reuse the capture, so "no photo"
    # no longer occurs mid-chat).
    prompt = glasses.ask(text="   ")
    assert prompt.active
    assert any("didn't catch" in t for t in _texts(prompt.scene))

    # Before the prompt deadline: previews keep the prompt.
    kept = glasses.preview(make_test_image(), last_seq=prompt.scene.seq)
    texts_now = _texts(kept.scene) if kept.scene else _texts(prompt.scene)
    assert any("didn't catch" in t for t in texts_now)

    # After the deadline the preview stream restores the slide.
    sess = _session(app)
    monkeypatch.setattr("lstk_eye.session.PROMPT_SECONDS", 0.0)
    sess._prompt_until = 0.0
    restored = glasses.preview(make_test_image(), last_seq=prompt.scene.seq)
    assert restored.scene is not None
    assert any(e.t == "target" for e in restored.scene.els), "slide must return"


def test_reset_during_calibration_says_calibration_off(rig):
    glasses, _ = rig
    glasses.ask(text="calibrate")
    done = glasses.reset()
    assert not done.active
    assert any("calibration off" in t for t in _texts(done.scene))
    idle_done = glasses.reset()
    assert any("done" in t for t in _texts(idle_done.scene))


def test_calibration_accepts_flipped_camera(rig, cfg):
    """A camera whose rotation config is 180 degrees off produces a mirrored
    mapping (negative windows). That is a valid affine model - calibration
    must succeed and map the collected points correctly, not reject with
    'calib failed'."""
    glasses, app = rig
    glasses.ask(text="calibrate")
    comp = app.state.runtime.composer
    from lstk_eye.display import CROSSHAIR_R

    inset = CROSSHAIR_R + 2
    disp = [comp.center, (comp.x1 - inset, comp.center[1]), (comp.center[0], comp.y0 + inset)]
    # Display moves right/up while the camera sees left/down: inverted axes.
    cam = [(0.6, 0.55), (0.45, 0.55), (0.6, 0.65)]
    for cx, cy in cam:
        ack = glasses.capture(_frame_with_marker(cx, cy))
    assert "calibrated" in _texts(ack.scene)
    cal = app.state.runtime.calibration
    assert cal.window_w < 0 and cal.window_h < 0
    for c, d in zip(cam, disp, strict=True):
        got = cal.to_display(c)
        assert abs(got[0] - d[0]) <= 1 and abs(got[1] - d[1]) <= 1


def test_crosshair_brackets_fully_visible(rig, cfg):
    glasses, app = rig
    comp = app.state.runtime.composer
    from lstk_eye.display import CROSSHAIR_R

    glasses.ask(text="calibrate")
    sess = app.state.sessions.get("sim")
    for x, y in sess._calib.points:
        assert x - CROSSHAIR_R >= comp.x0 and x + CROSSHAIR_R <= comp.x1
        assert y - CROSSHAIR_R >= comp.y0 and y + CROSSHAIR_R <= comp.y1


def test_calibration_success_names_next_action(rig):
    glasses, _ = rig
    glasses.ask(text="calibrate")
    for c in [(0.55, 0.5), (0.75, 0.5), (0.55, 0.35)]:
        ack = glasses.capture(_frame_with_marker(*c))
    texts = _texts(ack.scene)
    assert "calibrated" in texts
    assert any("hold btn" in t for t in texts), "the end must state what to do next"


def test_refinement_without_photo_reuses_active_capture(rig):
    """'No, the RED tulip' with no new click must rerun the pipeline on the
    capture the current task is built on, not demand a new photo."""
    glasses, app = rig
    called = []
    planner = app.state.runtime.planner
    original = planner.plan
    planner.plan = lambda *a, **k: (called.append(1), original(*a, **k))[1]

    glasses.capture(make_test_image())
    first = glasses.ask(text="find the tulip in the scene")
    assert first.active and len(called) == 1

    followup = glasses.ask(text="find the red tulip instead please")
    assert followup.active
    assert len(called) == 2, "the pipeline must rerun on the retained capture"
    assert not any("no photo" in t for t in _texts(followup.scene))

    # A fresh click still takes priority as the new context.
    glasses.capture(make_test_image())
    third = glasses.ask(text="find the vase next to everything")
    assert len(called) == 3
