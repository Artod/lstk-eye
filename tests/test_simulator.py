"""Tests for the device simulator: renderer, HTTP client, scripted scenario.

The server side is an inline FastAPI stub that implements the wire protocol
with canned responses and records request properties for assertions.
"""

from __future__ import annotations

import cv2
import httpx
import numpy as np
import pytest
from fastapi import FastAPI, Request, Response
from PIL import Image

from lstk_eye.errors import LstkError
from lstk_eye.protocol import (
    ArrowEl,
    AskResponse,
    ChevronEl,
    DisplayScene,
    EventRequest,
    HealthResponse,
    PhotoAck,
    SceneResponse,
    TextEl,
)
from lstk_eye.simulator import SimulatedGlasses, make_mock_wav, render_scene, run_scenario

# --- server stub ---


def _scene(seq: int, label: str) -> DisplayScene:
    return DisplayScene(seq=seq, els=[TextEl(x=0, y=0, text=label)])


def make_stub_app(n_steps: int = 3) -> tuple[FastAPI, dict]:
    """Minimal protocol server: buffers photos, serves ``n_steps`` slides after
    an ask, then ends the session. Records every call in ``state['calls']``."""
    app = FastAPI()
    state: dict = {
        "photos": 0,
        "step": 0,
        "seq": 0,
        "active": False,
        "scene": None,
        "calls": [],
    }

    def new_scene(label: str) -> DisplayScene:
        state["seq"] += 1
        state["scene"] = _scene(state["seq"], label)
        return state["scene"]

    def scene_since(last_seq: int) -> SceneResponse:
        unchanged = state["scene"] is None or state["scene"].seq == last_seq
        return SceneResponse(
            scene=None if unchanged else state["scene"], active=state["active"]
        )

    @app.get("/api/v1/health")
    def health() -> HealthResponse:
        return HealthResponse(version="stub")

    @app.post("/api/v1/photos")
    async def photos(request: Request, device_id: str = "glasses") -> PhotoAck:
        body = await request.body()
        decoded = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        state["photos"] += 1
        state["calls"].append(
            {
                "op": "photos",
                "device_id": device_id,
                "content_type": request.headers.get("content-type"),
                "shape": None if decoded is None else decoded.shape,
            }
        )
        return PhotoAck(count=state["photos"], scene=new_scene(f"[{state['photos']}]"))

    @app.post("/api/v1/ask")
    async def ask(
        request: Request, device_id: str = "glasses", text: str | None = None
    ) -> AskResponse:
        state["calls"].append(
            {"op": "ask", "device_id": device_id, "text": text, "body": await request.body()}
        )
        state["active"] = True
        state["step"] = 1
        return AskResponse(session_id="sess-1", scene=new_scene("step 1"), active=True)

    @app.post("/api/v1/event")
    def event(req: EventRequest, device_id: str = "glasses") -> SceneResponse:
        state["calls"].append({"op": "event", "device_id": device_id, "type": req.type})
        if req.type == "cancel" or not state["active"]:
            state["active"] = False
            return SceneResponse(scene=None, active=False)
        if req.type == "next":
            if state["step"] >= n_steps:
                state["active"] = False
                return SceneResponse(scene=None, active=False)
            state["step"] += 1
            return SceneResponse(scene=new_scene(f"step {state['step']}"), active=True)
        if req.type == "prev":
            state["step"] = max(1, state["step"] - 1)
            return SceneResponse(scene=new_scene(f"step {state['step']}"), active=True)
        return SceneResponse(scene=state["scene"], active=True)  # repeat

    @app.post("/api/v1/preview")
    async def preview(
        request: Request, last_seq: int, device_id: str = "glasses"
    ) -> SceneResponse:
        body = await request.body()
        decoded = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        state["calls"].append(
            {
                "op": "preview",
                "device_id": device_id,
                "last_seq": last_seq,
                "content_type": request.headers.get("content-type"),
                "shape": None if decoded is None else decoded.shape,
            }
        )
        return scene_since(last_seq)

    @app.get("/api/v1/scene")
    def scene(last_seq: int, device_id: str = "glasses") -> SceneResponse:
        state["calls"].append({"op": "scene", "device_id": device_id, "last_seq": last_seq})
        return scene_since(last_seq)

    return app, state


@pytest.fixture()
def stub() -> tuple[FastAPI, dict]:
    return make_stub_app()


@pytest.fixture()
def glasses(stub):
    app, _ = stub
    with SimulatedGlasses(base_url="http://test", transport=httpx.ASGITransport(app=app)) as g:
        yield g


def _frame() -> np.ndarray:
    img = np.zeros((240, 320, 3), np.uint8)
    img[100:140, 150:170] = 255
    return img


# --- renderer ---


def test_render_scene_sizes():
    scene = DisplayScene(seq=1, els=[])
    assert render_scene(scene, scale=1).size == (128, 64)
    img = render_scene(scene, scale=4)
    assert img.size == (512, 256)
    assert img.mode == "L"


def test_render_text_pixels_in_expected_area():
    scene = DisplayScene(seq=1, els=[TextEl(x=4, y=4, size=1, text="HI")])
    arr = np.array(render_scene(scene, scale=1))
    assert arr[0:20, 0:25].any()  # glyphs near the requested origin
    assert not arr[:, 40:].any()  # nothing far to the right
    assert not arr[30:, :].any()  # nothing in the lower half


def test_render_text_size_scales_up():
    small = np.array(render_scene(DisplayScene(seq=1, els=[TextEl(text="A", size=1)]), scale=1))
    big = np.array(render_scene(DisplayScene(seq=1, els=[TextEl(text="A", size=2)]), scale=1))
    assert np.count_nonzero(big) > np.count_nonzero(small)


def test_render_arrow_shaft_and_head():
    # Tip at (64, 32) pointing right => shaft extends left from the tip.
    scene = DisplayScene(seq=1, els=[ArrowEl(x=64, y=32, angle=0, length=14)])
    arr = np.array(render_scene(scene, scale=1))
    assert arr[32, 64]  # tip
    assert arr[32, 50:64].all()  # shaft along -x
    assert arr[28:32, 59:64].any()  # head stroke above the shaft
    assert arr[33:37, 59:64].any()  # head stroke below the shaft
    assert not arr[32, 66:].any()  # nothing ahead of the tip


def test_render_chevron_hugs_edge():
    scene = DisplayScene(seq=1, els=[ChevronEl(edge="right", label="far")])
    arr = np.array(render_scene(scene, scale=1))
    assert arr[27:38, 114:128].any()  # chevron marks near the right edge
    assert arr[26:40, 90:114].any()  # label just inward of the marks
    assert not arr[:, :64].any()  # left half untouched


def test_render_chevron_up_edge():
    scene = DisplayScene(seq=1, els=[ChevronEl(edge="up")])
    arr = np.array(render_scene(scene, scale=1))
    assert arr[0:8, 58:71].any()  # marks at the top center
    assert not arr[16:, :].any()


# --- device client ---


def test_make_mock_wav():
    assert make_mock_wav("check voltage") == b"MOCKTEXT:check voltage"


def test_health(glasses):
    resp = glasses.health()
    assert resp.status == "ok"
    assert resp.version == "stub"
    assert resp.protocol == 1


def test_capture_posts_jpeg(stub, glasses):
    _, state = stub
    ack = glasses.capture(_frame())
    assert ack.count == 1
    call = state["calls"][-1]
    assert call["op"] == "photos"
    assert call["content_type"] == "image/jpeg"
    assert call["device_id"] == "sim"
    assert call["shape"] == (240, 320, 3)
    assert glasses.last_seq == ack.scene.seq


def test_capture_from_path(stub, glasses, tmp_path):
    _, state = stub
    path = tmp_path / "frame.png"
    assert cv2.imwrite(str(path), _frame())
    ack = glasses.capture(path)
    assert ack.count == 1
    assert state["calls"][-1]["shape"] == (240, 320, 3)


def test_ask_with_text_sends_query_param_and_mock_wav(stub, glasses):
    _, state = stub
    resp = glasses.ask(text="how do I check battery voltage")
    assert resp.session_id == "sess-1"
    assert resp.active is True
    call = state["calls"][-1]
    assert call["op"] == "ask"
    assert call["text"] == "how do I check battery voltage"
    assert call["body"] == b"MOCKTEXT:how do I check battery voltage"
    assert glasses.last_seq == resp.scene.seq


def test_ask_with_raw_wav(stub, glasses):
    _, state = stub
    glasses.ask(wav=b"RIFFxxxx")
    call = state["calls"][-1]
    assert call["text"] is None
    assert call["body"] == b"RIFFxxxx"


def test_ask_requires_exactly_one_input(glasses):
    with pytest.raises(ValueError):
        glasses.ask()
    with pytest.raises(ValueError):
        glasses.ask(text="hi", wav=b"RIFF")


@pytest.mark.parametrize("gesture", ["next", "prev", "cancel", "repeat"])
def test_events_send_json_body(stub, glasses, gesture):
    _, state = stub
    glasses.ask(text="q")
    resp = getattr(glasses, gesture)()
    assert isinstance(resp, SceneResponse)
    call = state["calls"][-1]
    assert call["op"] == "event"
    assert call["type"] == gesture  # EventRequest parsed from the JSON body


def test_preview_posts_qvga_jpeg_and_tracks_seq(stub, glasses):
    _, state = stub
    glasses.ask(text="q")
    seq_after_ask = glasses.last_seq

    unchanged = glasses.preview(np.zeros((480, 640, 3), np.uint8))
    assert unchanged.scene is None
    assert unchanged.active is True
    call = state["calls"][-1]
    assert call["content_type"] == "image/jpeg"
    assert call["shape"] == (240, 320, 3)  # resized to QVGA regardless of input
    assert call["last_seq"] == seq_after_ask

    stale = glasses.preview(_frame(), last_seq=0)
    assert stale.scene is not None
    assert stale.scene.seq == seq_after_ask
    assert glasses.last_seq == seq_after_ask


def test_poll_scene_tracks_last_seq(stub, glasses):
    glasses.ask(text="q")
    seq = glasses.last_seq
    assert glasses.poll_scene().scene is None  # up to date
    resp = glasses.poll_scene(last_seq=-1)
    assert resp.scene is not None and resp.scene.seq == seq
    assert glasses.last_seq == seq


def test_non_2xx_raises_clear_error():
    app = FastAPI()

    @app.get("/api/v1/health")
    def health() -> Response:
        return Response(status_code=503, content="overloaded")

    with SimulatedGlasses(base_url="http://test", transport=httpx.ASGITransport(app=app)) as g:
        with pytest.raises(LstkError, match="503"):
            g.health()


# --- scenario ---


def test_run_scenario_saves_frames(tmp_path):
    app, state = make_stub_app(n_steps=3)
    with SimulatedGlasses(base_url="http://test", transport=httpx.ASGITransport(app=app)) as g:
        paths = run_scenario(g, _frame(), "how do I check battery voltage", tmp_path)

    # capture counter + 3 slides (ask serves slide 1); the last next() ends
    # the session, so no cancel is needed
    assert [p.name for p in paths] == [f"frame_{i:03d}.png" for i in range(4)]
    for p in paths:
        with Image.open(p) as im:
            im.verify()
    with Image.open(paths[0]) as im:
        assert im.size == (512, 256)
        assert im.mode == "L"
    events = [c["type"] for c in state["calls"] if c["op"] == "event"]
    assert events == ["next", "next", "next"]
    assert state["active"] is False
