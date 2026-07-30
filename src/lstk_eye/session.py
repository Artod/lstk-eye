"""Per-device session state machine - the server-side brain.

One DeviceSession per device_id. The device is stateless beyond the last
rendered scene, so everything lives here:

- a small photo buffer (single click on the device uploads immediately),
- the active slide session (plan steps resolved to anchors),
- the relocalization tracker for the current step,
- a monotonically increasing scene ``seq`` so the device only redraws on
  change.

All public methods are safe to call from FastAPI's threadpool: a per-session
lock serializes them. Pipeline work (STT, segmentation, planning) happens
inside ``ask`` while the device blocks on the HTTP response - that is the
3-6 s "thinking" budget by design.
"""

import logging
import threading
import time
import uuid

import cv2
import numpy as np

from lstk_eye.calibration import WindowCalibration
from lstk_eye.config import AppConfig
from lstk_eye.display import SceneComposer
from lstk_eye.errors import LstkError, PipelineError
from lstk_eye.intents import match_intent
from lstk_eye.pipeline import factories
from lstk_eye.pipeline.interfaces import TargetTracker
from lstk_eye.pipeline.marks import encode_png, overlay_marks
from lstk_eye.pipeline.types import SegMask, Slide
from lstk_eye.protocol.messages import AskResponse, DisplayScene, PhotoAck, SceneResponse
from lstk_eye.storage import RunStore

log = logging.getLogger(__name__)

PHOTO_TTL_S = 120.0
MAX_PHOTOS = 5
# Minimum normalized anchor movement that triggers a scene update.
ANCHOR_EPS = 0.012


class Runtime:
    """Pipeline stages shared by all sessions, built once from config."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.stt = factories.create_stt(cfg)
        self.segmenter = factories.create_segmenter(cfg)
        self.planner = factories.create_planner(cfg)
        self.reloc = factories.create_relocalizer(cfg)
        self.calibration = WindowCalibration(cfg.calibration)
        self.composer = SceneComposer(self.calibration)
        self.store = RunStore(cfg.storage)


def _decode_bgr(jpeg: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise PipelineError("could not decode image")
    return img


class DeviceSession:
    def __init__(self, device_id: str, rt: Runtime):
        self.device_id = device_id
        self._rt = rt
        self._lock = threading.Lock()
        self.last_access = time.monotonic()
        self._seq = 0
        self._photos: list[tuple[bytes, float]] = []
        self._scene: DisplayScene = rt.composer.status("ready", self._next_seq())
        # Slide session state
        self.session_id: str = ""
        self.active = False
        self._slides: list[Slide] = []
        self._mask_by_id: dict[int, SegMask] = {}
        self._capture_bgr: np.ndarray | None = None
        self._step = 0
        self._tracker: TargetTracker | None = None
        self._anchored = False
        self._anchor: tuple[float, float] | None = None

    # --- scene bookkeeping ---

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _set_scene(self, scene: DisplayScene) -> DisplayScene:
        self._scene = scene
        return scene

    def _scene_if_newer(self, last_seq: int) -> DisplayScene | None:
        return self._scene if self._scene.seq > last_seq else None

    # --- public API (called from transport) ---

    def add_photo(self, jpeg: bytes) -> PhotoAck:
        with self._lock:
            now = time.monotonic()
            self._photos = [(b, t) for b, t in self._photos if now - t < PHOTO_TTL_S]
            self._photos.append((jpeg, now))
            self._photos = self._photos[-MAX_PHOTOS:]
            count = len(self._photos)
            if self.active:
                # Scenes replace each other, and nothing would restore the
                # slide afterwards - so mid-session captures are acknowledged
                # in the count only, without touching the display.
                return PhotoAck(count=count, scene=None)
            scene = self._set_scene(self._rt.composer.photo_count(count, self._next_seq()))
            return PhotoAck(count=count, scene=scene)

    def ask(self, wav: bytes, text: str | None = None) -> AskResponse:
        with self._lock:
            try:
                return self._ask_locked(wav, text)
            except LstkError as e:
                log.warning("ask failed: %s", e)
                scene = self._set_scene(self._rt.composer.error(str(e), self._next_seq()))
                return AskResponse(session_id=self.session_id, scene=scene, active=self.active)
            except Exception:
                log.exception("unexpected error in ask")
                scene = self._set_scene(
                    self._rt.composer.error("internal error", self._next_seq())
                )
                return AskResponse(session_id=self.session_id, scene=scene, active=self.active)

    def _ask_locked(self, wav: bytes, text: str | None) -> AskResponse:
        question = text if text is not None else self._rt.stt.transcribe(wav).text
        question = question.strip()
        if not question:
            scene = self._set_scene(
                self._rt.composer.status("didn't catch that", self._next_seq(), "try again")
            )
            return AskResponse(session_id=self.session_id, scene=scene, active=self.active)

        # Voice commands during an active session ("back", "repeat", "cancel")
        # never start a new pipeline run.
        if self.active:
            intent = match_intent(question)
            if intent is not None:
                resp = self._event_locked(intent)
                scene = resp.scene if resp.scene is not None else self._scene
                return AskResponse(session_id=self.session_id, scene=scene, active=resp.active)

        now = time.monotonic()
        fresh = [(b, t) for b, t in self._photos if now - t < PHOTO_TTL_S]
        if not fresh:
            # active must reflect the real session state: a follow-up question
            # without a fresh capture must not tell the device the session
            # ended while the server keeps it alive ("repeat" restores the
            # slide). See the protocol contract on the active flag.
            scene = self._set_scene(
                self._rt.composer.status("no photo", self._next_seq(), "click to capture")
            )
            return AskResponse(session_id=self.session_id, scene=scene, active=self.active)

        capture_jpeg = fresh[-1][0]
        capture = _decode_bgr(capture_jpeg)

        masks = self._rt.segmenter.segment(capture)
        marked = overlay_marks(capture, masks)
        marked_png = encode_png(marked)
        plan = self._rt.planner.plan(marked_png, question, masks)

        mask_by_id = {m.mark_id: m for m in masks}
        slides = [
            Slide(
                index=i,
                total=len(plan.steps),
                label=step.label,
                anchor=(mask_by_id[step.mark_id].centroid if step.mark_id in mask_by_id else None),
                mark_id=step.mark_id if step.mark_id in mask_by_id else None,
            )
            for i, step in enumerate(plan.steps)
        ]

        self.session_id = uuid.uuid4().hex[:8]
        self.active = True
        self._slides = slides
        self._mask_by_id = mask_by_id
        self._capture_bgr = capture
        self._photos = []
        self._rt.store.save_request(
            self.session_id, capture_jpeg, marked_png, question, plan, slides
        )
        scene = self._enter_step(0)
        return AskResponse(session_id=self.session_id, scene=scene, active=True)

    def event(self, event_type: str) -> SceneResponse:
        with self._lock:
            return self._event_locked(event_type)

    def _event_locked(self, event_type: str) -> SceneResponse:
        if not self.active:
            return SceneResponse(scene=None, active=False)
        if event_type == "next":
            if self._step + 1 >= len(self._slides):
                return SceneResponse(scene=self._finish("done"), active=False)
            return SceneResponse(scene=self._enter_step(self._step + 1), active=True)
        if event_type == "prev":
            return SceneResponse(scene=self._enter_step(max(0, self._step - 1)), active=True)
        if event_type == "repeat":
            slide = self._slides[self._step]
            scene = self._set_scene(
                self._rt.composer.slide(slide, self._next_seq(), self._anchored, self._anchor)
            )
            return SceneResponse(scene=scene, active=True)
        if event_type == "cancel":
            return SceneResponse(scene=self._finish("cancelled"), active=False)
        return SceneResponse(scene=None, active=self.active)

    def preview(self, jpeg: bytes, last_seq: int) -> SceneResponse:
        # Never queue behind a long-running ask: a preview that cannot get the
        # lock is worthless by the time it would (2-3 fps stream), and blocked
        # requests would pile up holding threadpool tokens, starving every
        # other device. Drop the frame instead.
        if not self._lock.acquire(blocking=False):
            return SceneResponse(scene=None, active=self.active)
        try:
            if not self.active:
                return SceneResponse(scene=self._scene_if_newer(last_seq), active=False)
            if self._tracker is not None:
                try:
                    self._update_anchor(jpeg)
                except PipelineError:
                    log.debug("preview frame rejected", exc_info=True)
            return SceneResponse(scene=self._scene_if_newer(last_seq), active=True)
        finally:
            self._lock.release()

    def scene_since(self, last_seq: int) -> SceneResponse:
        # Same non-blocking policy as preview: polling must not convoy behind
        # the pipeline.
        if not self._lock.acquire(blocking=False):
            return SceneResponse(scene=None, active=self.active)
        try:
            return SceneResponse(scene=self._scene_if_newer(last_seq), active=self.active)
        finally:
            self._lock.release()

    # --- internals ---

    def _enter_step(self, index: int) -> DisplayScene:
        self._step = index
        slide = self._slides[index]
        self._tracker = None
        self._anchored = slide.anchor is not None
        self._anchor = slide.anchor
        if (
            slide.anchor is not None
            and slide.mark_id is not None
            and self._capture_bgr is not None
        ):
            try:
                mask = self._mask_by_id[slide.mark_id]
                self._tracker = self._rt.reloc.create(self._capture_bgr, mask.bbox)
            except Exception:
                log.warning("relocalizer setup failed for step %d", index, exc_info=True)
        return self._set_scene(
            self._rt.composer.slide(slide, self._next_seq(), self._anchored, self._anchor)
        )

    def _update_anchor(self, jpeg: bytes) -> None:
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise PipelineError("could not decode preview frame")
        assert self._tracker is not None
        res = self._tracker.update(frame)
        new_anchor = res.center if res.found and res.center is not None else self._anchor
        moved = (
            res.found
            and new_anchor is not None
            and self._anchor is not None
            and abs(new_anchor[0] - self._anchor[0]) + abs(new_anchor[1] - self._anchor[1])
            > ANCHOR_EPS
        )
        if res.found != self._anchored or moved:
            self._anchored = res.found
            self._anchor = new_anchor
            slide = self._slides[self._step]
            self._set_scene(
                self._rt.composer.slide(slide, self._next_seq(), self._anchored, self._anchor)
            )

    def _finish(self, message: str) -> DisplayScene:
        self.active = False
        self._slides = []
        self._mask_by_id = {}
        self._tracker = None
        self._capture_bgr = None
        return self._set_scene(self._rt.composer.status(message, self._next_seq()))


# Session eviction: device_id is an unauthenticated query parameter, so the
# session map must not grow without bound on a LAN-exposed server.
MAX_DEVICES = 32
IDLE_EVICT_S = 1800.0


class SessionManager:
    def __init__(self, rt: Runtime):
        self._rt = rt
        self._lock = threading.Lock()
        self._sessions: dict[str, DeviceSession] = {}

    def get(self, device_id: str) -> DeviceSession:
        with self._lock:
            now = time.monotonic()
            session = self._sessions.get(device_id)
            if session is None:
                self._evict(now)
                session = DeviceSession(device_id, self._rt)
                self._sessions[device_id] = session
            session.last_access = now
            return session

    def _evict(self, now: float) -> None:
        for did, sess in list(self._sessions.items()):
            if not sess.active and now - sess.last_access > IDLE_EVICT_S:
                del self._sessions[did]
        while len(self._sessions) >= MAX_DEVICES:
            # Drop the least recently used session, preferring inactive ones.
            candidates = sorted(
                self._sessions.items(), key=lambda kv: (kv[1].active, kv[1].last_access)
            )
            del self._sessions[candidates[0][0]]
