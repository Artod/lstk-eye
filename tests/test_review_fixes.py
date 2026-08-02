"""Regression tests for defects found in the initial code review."""

import cv2
import httpx
import numpy as np
import pytest

from lstk_eye.config import RelocConfig, SttConfig, load_config
from lstk_eye.pipeline.reloc.template import TemplateRelocalizerFactory
from lstk_eye.server import create_app
from lstk_eye.session import MAX_DEVICES, Runtime, SessionManager
from lstk_eye.simulator import SimulatedGlasses
from lstk_eye.testimage import make_test_image


@pytest.fixture()
def cfg(tmp_path):
    return load_config(
        None,
        profile="mock",
        storage={"dir": str(tmp_path / "runs")},
        server={"zeroconf": False},
    )


@pytest.fixture()
def glasses(cfg):
    app = create_app(cfg)
    return SimulatedGlasses(base_url="http://test", transport=httpx.ASGITransport(app=app))


def test_mid_session_ask_without_photo_keeps_session_alive(glasses):
    """A follow-up question with no fresh capture must not tell the device the
    session ended while the server keeps it running (active-flag desync)."""
    glasses.capture(make_test_image())
    start = glasses.ask(text="how do I check battery voltage")
    assert start.active

    # The photo buffer was consumed by the ask; a follow-up question (>3
    # words, so not a voice intent) finds no photo.
    followup = glasses.ask(text="what does the display mean here")
    assert followup.active, "device and server must agree the session is still live"

    polled = glasses.poll_scene(-1)
    assert polled.active


def test_photo_during_session_does_not_clobber_slide(glasses):
    """Scenes replace each other and nothing restores the slide, so a
    mid-session capture must not push a photo-counter scene."""
    glasses.capture(make_test_image())
    start = glasses.ask(text="how do I check battery voltage")
    slide_seq = start.scene.seq

    ack = glasses.capture(make_test_image())
    assert ack.count == 1
    # The current scene survives with a photo-count badge appended - the
    # answer is never wiped by a full photo-counter screen.
    assert ack.scene is not None
    slide_texts = {e.text for e in start.scene.els if e.t == "text"}
    ack_texts = {e.text for e in ack.scene.els if e.t == "text"}
    assert "[1]" in ack_texts
    assert slide_texts <= ack_texts, "the answer content must survive the capture"
    assert ack.scene.seq > slide_seq


def test_stt_empty_language_means_autodetect():
    """TOML has no null; language='' in a config file must not reach
    faster-whisper (which rejects it)."""
    assert SttConfig(language="").language is None
    assert SttConfig(language=None).language is None
    assert SttConfig(language="en").language == "en"


def test_tracker_starts_visible():
    """The tracker is built from a frame where the target is present, so one
    weak preview frame (motion blur) must not hide the arrows - hiding costs
    the full miss_hide budget from the start."""
    rng = np.random.default_rng(7)
    capture = rng.integers(0, 60, size=(480, 640), dtype=np.uint8)
    patch = np.kron(rng.integers(0, 2, (8, 8)), np.ones((12, 12))) * 255
    capture[100:196, 200:296] = patch.astype(np.uint8)
    capture_bgr = cv2.cvtColor(capture, cv2.COLOR_GRAY2BGR)

    cfg = RelocConfig()
    tracker = TemplateRelocalizerFactory(cfg).create(
        capture_bgr, (200 / 640, 100 / 480, 96 / 640, 96 / 480)
    )

    noise = rng.integers(0, 60, size=(240, 320), dtype=np.uint8)
    for i in range(cfg.miss_hide - 1):
        res = tracker.update(noise)
        assert res.found, f"hidden after only {i + 1} misses (budget is {cfg.miss_hide})"
    assert not tracker.update(noise).found, "must hide once the miss budget is exhausted"


async def test_oversized_body_rejected(cfg):
    app = create_app(cfg)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/api/v1/photos",
            content=b"x" * 9_000_000,
            headers={"content-type": "image/jpeg"},
        )
        assert r.status_code == 413


def test_session_manager_evicts_unbounded_device_ids(cfg):
    """device_id is an unauthenticated query param; the session map must stay
    bounded when a LAN host fabricates ids."""
    mgr = SessionManager(Runtime(cfg))
    for i in range(MAX_DEVICES + 10):
        mgr.get(f"device-{i}")
    assert len(mgr._sessions) <= MAX_DEVICES
