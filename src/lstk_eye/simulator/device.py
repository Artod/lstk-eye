"""HTTP client that impersonates the glasses firmware.

Mirrors the device gesture contract one method per gesture: ``capture`` =
single click, ``ask`` = long press release, ``next`` = double click,
``preview`` = the 2-3 fps relocalization stream. Every response is parsed
with the pydantic models from :mod:`lstk_eye.protocol`, so the simulator
doubles as a wire-protocol conformance check.

The client is synchronous. Tests may pass ``transport=httpx.ASGITransport(
app=...)`` to talk to an in-process FastAPI app; async transports are driven
through a private event loop per request, so this class must not be used from
inside a running event loop.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import httpx
import numpy as np

from lstk_eye.errors import LstkError
from lstk_eye.protocol import (
    AskResponse,
    DisplayScene,
    HealthResponse,
    PhotoAck,
    SceneResponse,
)

MOCK_WAV_PREFIX = b"MOCKTEXT:"
PREVIEW_SIZE = (320, 240)  # QVGA (width, height), matches the device preview stream

_DEFAULT_BASE_URL = "http://127.0.0.1:8321"
_JPEG_QUALITY = 85


def make_mock_wav(text: str) -> bytes:
    """Bytes the mock STT backend decodes verbatim: ASCII prefix + UTF-8 text."""
    return MOCK_WAV_PREFIX + text.encode("utf-8")


class DeviceHttpError(LstkError):
    """The server answered a device request with a non-2xx status."""

    def __init__(self, method: str, url: str, status_code: int, body: str):
        super().__init__(f"{method} {url} -> HTTP {status_code}: {body}")
        self.status_code = status_code


class _AsyncTransportAdapter(httpx.BaseTransport):
    """Drive an async httpx transport (e.g. ASGITransport) from a sync Client.

    Each request runs to completion in its own event loop and the body is
    re-wrapped in a sync response, because ``httpx.Response.close`` refuses
    async streams.
    """

    def __init__(self, transport: httpx.AsyncBaseTransport):
        self._transport = transport

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        async def _run() -> httpx.Response:
            response = await self._transport.handle_async_request(request)
            try:
                body = await response.aread()
            finally:
                await response.aclose()
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                content=body,
                request=request,
                extensions=response.extensions,
            )

        return asyncio.run(_run())

    def close(self) -> None:
        asyncio.run(self._transport.aclose())


def _load_bgr(image: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise LstkError(f"cannot read image file: {image}")
    return bgr


def _encode_jpeg(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not ok:
        raise LstkError("JPEG encoding failed")
    return buf.tobytes()


class SimulatedGlasses:
    """Software stand-in for the glasses, one HTTP call per device action.

    ``last_seq`` tracks the newest scene sequence number seen in any response;
    it starts at -1 so a first scene with ``seq == 0`` still counts as new.
    """

    def __init__(
        self,
        base_url: str | None = None,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
        device_id: str = "sim",
    ):
        if isinstance(transport, httpx.AsyncBaseTransport):
            transport = _AsyncTransportAdapter(transport)
        self.device_id = device_id
        self.last_seq = -1
        self._client = httpx.Client(
            base_url=base_url or _DEFAULT_BASE_URL, transport=transport, timeout=30.0
        )

    # --- device actions ---

    def health(self) -> HealthResponse:
        r = self._request("GET", "/api/v1/health")
        return HealthResponse.model_validate(r.json())

    def capture(self, image: np.ndarray | str | Path) -> PhotoAck:
        """Single click: encode the frame as JPEG and buffer it on the server."""
        jpeg = _encode_jpeg(_load_bgr(image))
        r = self._request(
            "POST",
            "/api/v1/photos",
            params={"device_id": self.device_id},
            content=jpeg,
            headers={"Content-Type": "image/jpeg"},
        )
        ack = PhotoAck.model_validate(r.json())
        self._track(ack.scene)
        return ack

    def ask(self, text: str | None = None, wav: bytes | None = None) -> AskResponse:
        """Long press release: send the question, blocking until the plan is ready.

        Exactly one of ``text`` (sent as a query param plus a mock-STT WAV body)
        or ``wav`` (raw WAV bytes) must be given.
        """
        if (text is None) == (wav is None):
            raise ValueError("ask() takes exactly one of text= or wav=")
        params: dict[str, str] = {"device_id": self.device_id}
        if text is not None:
            params["text"] = text
            body = make_mock_wav(text)
        else:
            body = wav
        r = self._request(
            "POST",
            "/api/v1/ask",
            params=params,
            content=body,
            headers={"Content-Type": "audio/wav"},
        )
        resp = AskResponse.model_validate(r.json())
        self._track(resp.scene)
        return resp

    def next(self) -> SceneResponse:
        """Double click: advance to the next slide."""
        return self._event("next")

    def prev(self) -> SceneResponse:
        return self._event("prev")

    def cancel(self) -> SceneResponse:
        return self._event("cancel")

    def repeat(self) -> SceneResponse:
        return self._event("repeat")

    def reset(self) -> SceneResponse:
        """Double click on the device: end the chat, clear everything."""
        return self._event("reset")

    def preview(self, image: np.ndarray | str | Path, last_seq: int | None = None) -> SceneResponse:
        """One relocalization frame: QVGA JPEG in, latest scene (or null) out."""
        qvga = cv2.resize(_load_bgr(image), PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
        r = self._request(
            "POST",
            "/api/v1/preview",
            params={"device_id": self.device_id, "last_seq": self._seq_arg(last_seq)},
            content=_encode_jpeg(qvga),
            headers={"Content-Type": "image/jpeg"},
        )
        return self._scene_response(r)

    def poll_scene(self, last_seq: int | None = None) -> SceneResponse:
        r = self._request(
            "GET",
            "/api/v1/scene",
            params={"device_id": self.device_id, "last_seq": self._seq_arg(last_seq)},
        )
        return self._scene_response(r)

    # --- lifecycle ---

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SimulatedGlasses:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- internals ---

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        resp = self._client.request(method, path, **kwargs)
        if not resp.is_success:
            raise DeviceHttpError(method, str(resp.request.url), resp.status_code, resp.text[:200])
        return resp

    def _event(self, kind: str) -> SceneResponse:
        r = self._request(
            "POST",
            "/api/v1/event",
            params={"device_id": self.device_id},
            json={"type": kind},
        )
        return self._scene_response(r)

    def _scene_response(self, r: httpx.Response) -> SceneResponse:
        resp = SceneResponse.model_validate(r.json())
        self._track(resp.scene)
        return resp

    def _seq_arg(self, last_seq: int | None) -> int:
        return self.last_seq if last_seq is None else last_seq

    def _track(self, scene: DisplayScene | None) -> None:
        if scene is not None:
            self.last_seq = scene.seq
