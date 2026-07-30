"""FastAPI application - the transport layer.

Deliberately thin: parse the request, hand bytes to the session, return the
pydantic response. All state and logic live in lstk_eye.session. Heavy work
runs in the threadpool so preview frames from other devices are not starved
while one request is planning.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from lstk_eye import __version__
from lstk_eye.config import AppConfig, load_config
from lstk_eye.discovery import ZeroconfAdvertiser
from lstk_eye.protocol.messages import (
    PROTOCOL_VERSION,
    AskResponse,
    EventRequest,
    HealthResponse,
    PhotoAck,
    SceneResponse,
)
from lstk_eye.session import Runtime, SessionManager

# Hard caps on request bodies: the server is reachable by anything on the
# LAN, and every body is buffered in RAM. Real payloads are far smaller
# (XGA JPEG ~150 KB, QVGA preview ~15 KB, 20 s WAV ~640 KB).
MAX_PHOTO_BYTES = 8_000_000
MAX_PREVIEW_BYTES = 2_000_000
MAX_AUDIO_BYTES = 4_000_000


async def _read_capped(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code=413, detail="body too large")
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail="body too large")
        chunks.append(chunk)
    return b"".join(chunks)


def create_app(cfg: AppConfig | None = None) -> FastAPI:
    if cfg is None:
        cfg = load_config()
    runtime = Runtime(cfg)
    sessions = SessionManager(runtime)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        advertiser = ZeroconfAdvertiser(cfg.server.port) if cfg.server.zeroconf else None
        if advertiser:
            await run_in_threadpool(advertiser.start)
        yield
        if advertiser:
            await run_in_threadpool(advertiser.stop)

    app = FastAPI(title="lstk-eye", version=__version__, lifespan=lifespan)
    app.state.cfg = cfg
    app.state.runtime = runtime
    app.state.sessions = sessions

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(version=__version__, protocol=PROTOCOL_VERSION)

    @app.post("/api/v1/photos", response_model=PhotoAck)
    async def photos(request: Request, device_id: str = Query("glasses")) -> PhotoAck:
        body = await _read_capped(request, MAX_PHOTO_BYTES)
        if not body:
            raise HTTPException(status_code=400, detail="empty photo body")
        return await run_in_threadpool(sessions.get(device_id).add_photo, body)

    @app.post("/api/v1/ask", response_model=AskResponse)
    async def ask(
        request: Request,
        device_id: str = Query("glasses"),
        text: str | None = Query(None),
    ) -> AskResponse:
        if text is not None and not cfg.debug.allow_text_ask:
            raise HTTPException(
                status_code=403,
                detail="text asks are disabled (set debug.allow_text_ask or use --profile mock)",
            )
        body = await _read_capped(request, MAX_AUDIO_BYTES)
        if text is None and not body:
            raise HTTPException(status_code=400, detail="empty audio body")
        return await run_in_threadpool(sessions.get(device_id).ask, body, text)

    @app.post("/api/v1/event", response_model=SceneResponse)
    async def event(body: EventRequest, device_id: str = Query("glasses")) -> SceneResponse:
        return await run_in_threadpool(sessions.get(device_id).event, body.type)

    @app.post("/api/v1/preview", response_model=SceneResponse)
    async def preview(
        request: Request,
        device_id: str = Query("glasses"),
        last_seq: int = Query(-1),
    ) -> SceneResponse:
        body = await _read_capped(request, MAX_PREVIEW_BYTES)
        if not body:
            raise HTTPException(status_code=400, detail="empty preview body")
        return await run_in_threadpool(sessions.get(device_id).preview, body, last_seq)

    @app.get("/api/v1/scene", response_model=SceneResponse)
    async def scene(device_id: str = Query("glasses"), last_seq: int = Query(-1)) -> SceneResponse:
        return await run_in_threadpool(sessions.get(device_id).scene_since, last_seq)

    return app
