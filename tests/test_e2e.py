"""End-to-end: the FastAPI server (mock profile) driven by the simulator.

Exercises the whole nervous system in-process - capture, ask, slide
navigation, voice intents, previews - with zero models and zero network.
"""

import httpx
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
    )


@pytest.fixture()
def glasses(cfg):
    app = create_app(cfg)
    transport = httpx.ASGITransport(app=app)
    return SimulatedGlasses(base_url="http://test", transport=transport)


def test_health(glasses):
    h = glasses.health()
    assert h.status == "ok"
    assert h.protocol == 1


def test_full_session_loop(glasses):
    frame = make_test_image()

    ack = glasses.capture(frame)
    assert ack.count == 1
    assert ack.scene is not None and ack.scene.els

    resp = glasses.ask(text="how do I check battery voltage")
    assert resp.active
    assert resp.session_id
    first = resp.scene
    assert first.els, "first slide must not be empty"

    # A slide session with the mock planner has up to 3 steps; walk to the end.
    seqs = [first.seq]
    active = True
    for _ in range(10):
        step = glasses.next()
        if step.scene is not None:
            assert step.scene.seq > seqs[-1]
            seqs.append(step.scene.seq)
        if not step.active:
            active = False
            break
    assert not active, "session should end after stepping past the last slide"

    idle = glasses.poll_scene(-1)
    assert not idle.active


def test_ask_without_photo(glasses):
    resp = glasses.ask(text="what is this")
    assert not resp.active
    assert resp.scene.els


def test_voice_intents_during_session(glasses):
    glasses.capture(make_test_image())
    start = glasses.ask(text="how do I check battery voltage")
    assert start.active

    # "repeat" re-pushes the current slide; "cancel" ends the session.
    rep = glasses.ask(text="repeat")
    assert rep.active
    assert rep.scene.seq > start.scene.seq

    cancel = glasses.ask(text="cancel")
    assert not cancel.active


def test_preview_lifecycle(glasses):
    frame = make_test_image()
    # Idle: previews are refused politely with active=False.
    idle = glasses.preview(frame, last_seq=-1)
    assert not idle.active

    glasses.capture(frame)
    resp = glasses.ask(text="how do I check battery voltage")
    assert resp.active

    live = glasses.preview(frame, last_seq=resp.scene.seq)
    assert live.active


def test_scene_polling_dedup(glasses):
    glasses.capture(make_test_image())
    resp = glasses.ask(text="how do I check battery voltage")
    current = glasses.poll_scene(-1)
    assert current.scene is not None
    unchanged = glasses.poll_scene(current.scene.seq)
    assert unchanged.scene is None
    assert resp.active


async def test_text_ask_forbidden_without_debug_flag(cfg):
    hardened = cfg.model_copy(
        update={"debug": cfg.debug.model_copy(update={"allow_text_ask": False})}
    )
    app = create_app(hardened)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/api/v1/ask", params={"text": "hi"}, content=b"")
        assert r.status_code == 403
