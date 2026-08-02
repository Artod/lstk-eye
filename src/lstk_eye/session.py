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
from lstk_eye.config import AppConfig, CalibrationConfig, save_calibration
from lstk_eye.display import CHAR_W, CROSSHAIR_R, SceneComposer
from lstk_eye.errors import ConfigError, LstkError, PipelineError
from lstk_eye.intents import (
    is_calibration_request,
    is_locate_request,
    match_intent,
    normalize_words,
)
from lstk_eye.pipeline import factories
from lstk_eye.pipeline.fiducial import annotate_detection, detect_target
from lstk_eye.pipeline.interfaces import TargetTracker
from lstk_eye.pipeline.marks import encode_png, overlay_marks
from lstk_eye.pipeline.types import SegMask, Slide
from lstk_eye.protocol.messages import (
    AskResponse,
    DisplayScene,
    PhotoAck,
    SceneResponse,
    TextEl,
)
from lstk_eye.storage import RunStore

log = logging.getLogger(__name__)

PHOTO_TTL_S = 120.0
MAX_PHOTOS = 5
# Minimum normalized anchor movement that triggers a scene update.
ANCHOR_EPS = 0.004
# Chat history kept per device (question/answer pairs); the planner sees the
# most recent slice of it.
MAX_HISTORY = 12
# Transient prompts ("no photo", "didn't catch that") shown mid-session are
# restored back to the slide by the preview stream after this long.
PROMPT_SECONDS = 4.0
# Lost-target staging: after miss_hide the marker disappears (blank screen);
# "cant see" appears only after this many consecutive misses - fast head
# motion blurs a few frames and must not scream "cant see" at the wearer.
CANT_SEE_MISSES = 12


# A photo whose decoded width is below this is not a capture (preview-sized
# frames must never reach the planner).
MIN_PHOTO_WIDTH = 600
# A find-answer anchored to a mask covering more than this fraction of the
# frame is too coarse to point at anything ("the whole picture is the
# clitoris"): a second, zoomed segmentation+planning pass refines it.
REFINE_AREA = 0.15
# Synthetic mark ids for refined (second-pass) masks - never collide with
# the segmenter's 1..max_masks range.
REFINE_ID_BASE = 1000

# Calibration marker acceptance: apparent size window (fraction of frame
# width) and distance from the frame border. The lower bound sits just above
# the detector's reliable floor; the upper bound rejects "marker fills the
# frame" setups where aiming precision collapses.
CALIB_MIN_MARKER = 0.045
CALIB_MAX_MARKER = 0.35
CALIB_EDGE_MARGIN = 0.06


class _CalibState:
    """In-flight calibration: crosshair points to show, pairs collected, and
    the last live-feedback line shown (to redraw only on change)."""

    def __init__(self, points: list[tuple[int, int]]):
        self.points = points
        self.pairs: list[tuple[tuple[float, float], tuple[int, int]]] = []
        self.idx = 0
        self.status = "looking for marker"


def _marker_status(found) -> tuple[str, bool]:
    """(feedback line, click would be accepted) for a detection result."""
    if found is None:
        return "no marker seen", False
    (cx, cy), size, _ = found
    if size < CALIB_MIN_MARKER:
        return "closer to marker", False
    if size > CALIB_MAX_MARKER:
        return "further from marker", False
    if not (
        CALIB_EDGE_MARGIN < cx < 1 - CALIB_EDGE_MARGIN
        and CALIB_EDGE_MARGIN < cy < 1 - CALIB_EDGE_MARGIN
    ):
        return "marker near edge", False
    return "marker OK - click", True


class Runtime:
    """Pipeline stages shared by all sessions, built once from config."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.stt = factories.create_stt(cfg)
        self.segmenter = factories.create_segmenter(cfg)
        self.planner = factories.create_planner(cfg)
        self.reloc = factories.create_relocalizer(cfg)
        self.calibration = WindowCalibration(cfg.calibration)
        self.composer = SceneComposer(self.calibration, cfg.display)
        self.store = RunStore(cfg.storage)


_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _decode(jpeg: bytes, cam, gray: bool = False) -> np.ndarray:
    """Decode a device frame and normalize it upright per the camera config
    (rotation, then mirror flags).

    Every frame - capture or preview, chat or calibration - passes through
    here, so downstream code (segmentation, tracking, calibration, display
    mapping) always works in upright coordinates."""
    flags = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), flags)
    if img is None:
        raise PipelineError("could not decode image")
    if cam.rotation:
        img = cv2.rotate(img, _ROTATE_CODES[cam.rotation])
    if cam.flip_v:
        img = cv2.flip(img, 0)
    if cam.flip_h:
        img = cv2.flip(img, 1)
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
        self._capture_jpeg: bytes | None = None
        self._step = 0
        self._tracker: TargetTracker | None = None
        self._anchored = False
        self._anchor: tuple[float, float] | None = None
        self._track_scale = 1.0
        self._lost_stage = 0
        # "target" for find-this sessions (a single anchored step renders as
        # highlight brackets); "arrow" for multi-step instructions.
        self._style = "arrow"  # type: str
        # Ongoing chat: (question, answer) pairs, cleared by reset.
        self._history: list[tuple[str, str]] = []
        self._calib: _CalibState | None = None
        # When set, the current scene is a transient prompt over a live
        # session; previews restore the slide after the deadline.
        self._prompt_until: float | None = None

    def _decode(self, jpeg: bytes, gray: bool = False) -> np.ndarray:
        return _decode(jpeg, self._rt.cfg.camera, gray=gray)

    def _restore_slide_after_prompt(self) -> None:
        self._prompt_until = time.monotonic() + PROMPT_SECONDS

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
            if self._calib is not None:
                return PhotoAck(count=len(self._photos), scene=self._calib_capture(jpeg))
            # Guard the buffer: a preview-sized or undecodable frame must
            # never become the capture the planner sees (the camera can emit
            # a stale low-res frame right after a resolution switch).
            try:
                frame = self._decode(jpeg)
                width_ok = max(frame.shape[:2]) >= MIN_PHOTO_WIDTH
            except PipelineError:
                width_ok = False
            if not width_ok:
                log.warning("rejected photo (undecodable or below %d px wide)", MIN_PHOTO_WIDTH)
                count = len(self._photos)
                if self.active:
                    return PhotoAck(count=count, scene=self._badge_scene("[!photo]"))
                scene = self._set_scene(
                    self._rt.composer.status("bad photo", self._next_seq(), "click again")
                )
                return PhotoAck(count=count, scene=scene)
            now = time.monotonic()
            self._photos = [(b, t) for b, t in self._photos if now - t < PHOTO_TTL_S]
            self._photos.append((jpeg, now))
            self._photos = self._photos[-MAX_PHOTOS:]
            count = len(self._photos)
            if self.active:
                # A new capture means the wearer moved on: stop the tracking
                # view (chat history survives) and show the photo state.
                self._slides = []
                self._tracker = None
                self.active = False
                self._prompt_until = None
            scene = self._set_scene(self._rt.composer.photo_count(count, self._next_seq()))
            return PhotoAck(count=count, scene=scene)

    def _badge_scene(self, badge: str) -> DisplayScene:
        """Re-issue the current scene with a short bracketed badge on the
        status row, replacing any previous badge."""
        comp = self._rt.composer
        els = [
            e
            for e in self._scene.els
            if not (
                isinstance(e, TextEl)
                and e.y == comp.status_y
                and e.text.startswith("[")
                and e.text.endswith("]")
            )
        ]
        x = comp.x1 + 1 - len(badge) * CHAR_W
        # Step counters ("2/5") also live on the right of the status row.
        for e in els:
            if isinstance(e, TextEl) and e.y == comp.status_y and "/" in e.text:
                x = e.x - len(badge) * CHAR_W - CHAR_W
        els.append(TextEl(x=max(comp.x0, x), y=comp.status_y, text=badge))
        return self._set_scene(DisplayScene(seq=self._next_seq(), els=els))

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
        # Punctuation- or emoji-only transcripts are as empty as silence.
        empty = not question.strip() or not normalize_words(question)

        # Calibration owns the interaction while it is running: every
        # utterance is handled here and nothing falls through into the chat
        # pipeline. Cancel phrases win over everything ("stop calibration"
        # must cancel, not restart).
        if self._calib is not None:
            if not empty and match_intent(question) == "cancel":
                self._calib = None
                self.active = False
                scene = self._set_scene(
                    self._rt.composer.status("calibration off", self._next_seq())
                )
                return AskResponse(session_id=self.session_id, scene=scene, active=False)
            if not empty and is_calibration_request(question):
                return self._start_calibration()  # restart from point 1
            status = "didn't catch that" if empty else "busy: 2click exits"
            return AskResponse(
                session_id=self.session_id, scene=self._crosshair(status), active=True
            )

        if empty:
            scene = self._set_scene(
                self._rt.composer.status("didn't catch that", self._next_seq(), "try again")
            )
            if self.active:
                self._restore_slide_after_prompt()
            return AskResponse(session_id=self.session_id, scene=scene, active=self.active)

        question = question.strip()
        if is_calibration_request(question):
            return self._start_calibration()

        # Voice commands during an active session ("back", "repeat", "cancel")
        # never start a new pipeline run.
        if self.active:
            intent = match_intent(question)
            if intent is not None:
                resp = self._event_locked(intent)
                scene = resp.scene if resp.scene is not None else self._scene
                return AskResponse(session_id=self.session_id, scene=scene, active=resp.active)

        if not self.active and match_intent(question) is not None:
            scene = self._set_scene(
                self._rt.composer.status("no session", self._next_seq(), "hold btn + ask")
            )
            return AskResponse(session_id=self.session_id, scene=scene, active=False)

        now = time.monotonic()
        fresh = [(b, t) for b, t in self._photos if now - t < PHOTO_TTL_S]
        if fresh:
            capture_jpeg = fresh[-1][0]
            capture = self._decode(capture_jpeg)
        elif self.active and self._capture_bgr is not None:
            # Refinement without a new photo ("no, the RED tulip"): rerun the
            # pipeline on the capture the current task is built on.
            capture = self._capture_bgr
            capture_jpeg = self._capture_jpeg or b""
        else:
            headline = "photos expired" if self._photos else "no photo"
            scene = self._set_scene(
                self._rt.composer.status(headline, self._next_seq(), "click, then ask")
            )
            self._photos = []
            if self.active:
                self._restore_slide_after_prompt()
            return AskResponse(session_id=self.session_id, scene=scene, active=self.active)

        if is_locate_request(question):
            loc = self._rt.planner.locate(capture, question, history=self._history)
            if loc is not None:
                return self._answer_locate(question, capture, capture_jpeg, loc)

        masks = self._rt.segmenter.segment(capture)
        marked = overlay_marks(capture, masks)
        marked_png = encode_png(marked)
        plan = self._rt.planner.plan(marked_png, question, masks, history=self._history)
        plan, masks = self._refine_plan(capture, question, plan, masks)

        mask_by_id = {m.mark_id: m for m in masks}
        slides = [
            Slide(
                index=i,
                total=len(plan.steps),
                label=step.label,
                anchor=(mask_by_id[step.mark_id].centroid if step.mark_id in mask_by_id else None),
                mark_id=step.mark_id if step.mark_id in mask_by_id else None,
                size=(
                    mask_by_id[step.mark_id].bbox[2:4] if step.mark_id in mask_by_id else None
                ),
            )
            for i, step in enumerate(plan.steps)
        ]

        self.session_id = self.session_id or uuid.uuid4().hex[:8]
        self.active = True
        self._slides = slides
        self._style = (
            "target" if len(slides) == 1 and slides[0].anchor is not None else "arrow"
        )
        answer = plan.summary or "; ".join(s.label for s in plan.steps)
        log.info(
            "[%s] ask %r -> %d steps: %s (style=%s, marks=%d)",
            self.device_id, question, len(slides),
            "; ".join(s.label for s in slides), self._style, len(masks),
        )
        self._history.append((question, answer))
        self._history = self._history[-MAX_HISTORY:]
        self._mask_by_id = mask_by_id
        self._capture_bgr = capture
        self._capture_jpeg = capture_jpeg
        self._photos = []
        self._rt.store.save_request(
            self.session_id, capture_jpeg, marked_png, question, plan, slides
        )
        scene = self._enter_step(0)
        return AskResponse(session_id=self.session_id, scene=scene, active=True)

    def event(self, event_type: str) -> SceneResponse:
        with self._lock:
            try:
                return self._event_locked(event_type)
            except Exception:
                log.exception("unexpected error in event %s", event_type)
                scene = self._set_scene(
                    self._rt.composer.error("internal error", self._next_seq())
                )
                return SceneResponse(scene=scene, active=self.active)

    def _event_locked(self, event_type: str) -> SceneResponse:
        log.info("[%s] event %s (active=%s, calib=%s)",
                 self.device_id, event_type, self.active, self._calib is not None)
        if event_type == "reset":
            # Double click: the chat is over - session, history, photo
            # buffer, and any in-flight calibration all go. Aborting a
            # calibration must not read like a successful one.
            message = "calibration off" if self._calib is not None else "done"
            self._calib = None
            self._history = []
            self._photos = []
            return SceneResponse(scene=self._finish(message), active=False)
        if self._calib is not None:
            if event_type == "cancel":
                # Same exit as the voice cancel phrase.
                self._calib = None
                self.active = False
                scene = self._set_scene(
                    self._rt.composer.status("calibration off", self._next_seq())
                )
                return SceneResponse(scene=scene, active=False)
            # Calibration owns the screen; other events are no-ops.
            return SceneResponse(scene=None, active=True)
        if not self.active:
            return SceneResponse(scene=None, active=False)
        if event_type == "next":
            if not self._slides or self._step + 1 >= len(self._slides):
                return SceneResponse(scene=self._finish("done"), active=False)
            return SceneResponse(scene=self._enter_step(self._step + 1), active=True)
        if event_type == "prev":
            if not self._slides:
                return SceneResponse(scene=self._finish("done"), active=False)
            return SceneResponse(scene=self._enter_step(max(0, self._step - 1)), active=True)
        if event_type == "repeat":
            if not self._slides:
                return SceneResponse(scene=self._finish("done"), active=False)
            slide = self._slides[self._step]
            scene = self._set_scene(
                self._rt.composer.slide(
                    slide, self._next_seq(), self._anchored, self._anchor, style=self._style
                )
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
            if self._calib is not None:
                # Live calibration feedback: every preview frame updates the
                # "does the camera see the marker" line, so the wearer knows
                # whether a click will land before clicking.
                try:
                    self._calib_preview(jpeg)
                except PipelineError:
                    log.debug("calibration preview rejected", exc_info=True)
                return SceneResponse(scene=self._scene_if_newer(last_seq), active=True)
            if not self.active:
                return SceneResponse(scene=self._scene_if_newer(last_seq), active=False)
            if self._prompt_until is not None:
                # A transient prompt owns the screen: hold tracker-driven
                # redraws until the wearer has had time to read it, then
                # bring the slide back.
                if time.monotonic() < self._prompt_until:
                    return SceneResponse(scene=self._scene_if_newer(last_seq), active=True)
                self._prompt_until = None
                if self._slides:
                    slide = self._slides[self._step]
                    self._set_scene(
                        self._rt.composer.slide(
                            slide, self._next_seq(), self._anchored, self._anchor,
                            style=self._style,
                        )
                    )
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
        self._track_scale = 1.0
        self._lost_stage = 0
        bbox = None
        if slide.mark_id is not None and slide.mark_id in self._mask_by_id:
            bbox = self._mask_by_id[slide.mark_id].bbox
        elif slide.anchor is not None and slide.size is not None:
            bw, bh = slide.size
            bbox = (
                min(max(slide.anchor[0] - bw / 2, 0.0), 1.0 - bw),
                min(max(slide.anchor[1] - bh / 2, 0.0), 1.0 - bh),
                bw,
                bh,
            )
        if slide.anchor is not None and bbox is not None and self._capture_bgr is not None:
            try:
                self._tracker = self._rt.reloc.create(self._capture_bgr, bbox)
            except Exception:
                log.warning("relocalizer setup failed for step %d", index, exc_info=True)
        return self._set_scene(
            self._rt.composer.slide(
                slide, self._next_seq(), self._anchored, self._anchor, style=self._style
            )
        )

    def _update_anchor(self, jpeg: bytes) -> None:
        frame = self._decode(jpeg, gray=True)
        assert self._tracker is not None
        res = self._tracker.update(frame)
        if res.found:
            self._track_scale = res.scale
        new_anchor = res.center if res.found and res.center is not None else self._anchor
        moved = (
            res.found
            and new_anchor is not None
            and self._anchor is not None
            and abs(new_anchor[0] - self._anchor[0]) + abs(new_anchor[1] - self._anchor[1])
            > ANCHOR_EPS
        )
        # Staged lost UI: 0 = marker shown, 1 = blank (just lost, could be
        # motion blur), 2 = "cant see" (confidently gone, the last resort).
        stage = 0 if res.found else (1 if res.misses < CANT_SEE_MISSES else 2)
        if stage != self._lost_stage or (res.found and moved) or res.found != self._anchored:
            self._lost_stage = stage
            self._anchored = res.found
            self._anchor = new_anchor
            if stage == 1:
                self._set_scene(self._rt.composer.blank(self._next_seq()))
            else:
                self._set_scene(
                    self._rt.composer.slide(
                        self._scaled_slide(), self._next_seq(), self._anchored,
                        self._anchor, style=self._style,
                    )
                )

    # --- calibration flow ---
    #
    # Triggered by voice ("calibrate"). The HUD shows a small crosshair; the
    # wearer moves their head until the physical ArUco target (shown on a
    # phone/monitor, see `lstk-eye target`) sits under the crosshair, then
    # clicks. Each click yields one (camera point, display point) pair; three
    # points with horizontal and vertical spread solve the window model.

    def _start_calibration(self) -> AskResponse:
        comp = self._rt.composer
        cx, cy = comp.center
        # Points sit a full bracket inside the visible area - a crosshair
        # with clipped corners cannot be aimed at.
        inset = CROSSHAIR_R + 2
        points = [(cx, cy), (comp.x1 - inset, cy), (cx, comp.y0 + inset)]
        was_chatting = self.active and bool(self._slides)
        self._calib = _CalibState(points)
        if was_chatting:
            self._calib.status = "chat ended"
        self._slides = []
        self._mask_by_id = {}
        self._tracker = None
        self._capture_bgr = None
        self._prompt_until = None
        self.active = True
        scene = self._set_scene(
            comp.calibration_point(
                0, len(points), points[0], self._next_seq(), self._calib.status
            )
        )
        return AskResponse(session_id=self.session_id, scene=scene, active=True)

    def _crosshair(self, status: str | None = None) -> DisplayScene:
        calib = self._calib
        assert calib is not None
        if status is not None:
            calib.status = status
        return self._set_scene(
            self._rt.composer.calibration_point(
                calib.idx, len(calib.points), calib.points[calib.idx],
                self._next_seq(), calib.status,
            )
        )

    def _calib_preview(self, jpeg: bytes) -> None:
        """Update the live marker-feedback line from a preview frame; redraw
        only when the feedback actually changes."""
        calib = self._calib
        assert calib is not None
        frame = self._decode(jpeg, gray=True)
        status, _ = _marker_status(detect_target(frame))
        if status != calib.status:
            self._crosshair(status)

    def _calib_capture(self, jpeg: bytes) -> DisplayScene:
        calib = self._calib
        assert calib is not None

        try:
            frame = self._decode(jpeg)
        except PipelineError:
            return self._crosshair("bad photo - again")
        found = detect_target(frame)
        status, ok = _marker_status(found)
        self._rt.store.save_calibration_frame(
            calib.idx, jpeg, annotate_detection(frame, found), ok
        )
        if not ok:
            # The live line explains the problem; the click is simply not
            # counted. Saved frames under runs/calibration/ show what the
            # camera actually captured.
            return self._crosshair(status)
        calib.pairs.append((found[0], calib.points[calib.idx]))
        calib.idx += 1
        if calib.idx < len(calib.points):
            return self._crosshair("marker OK - click")

        try:
            fitted = WindowCalibration.fit(calib.pairs)
            if not (
                0.05 < abs(fitted.window_w) < 2.0
                and 0.05 < abs(fitted.window_h) < 2.0
                and -0.5 < fitted.center_x < 1.5
                and -0.5 < fitted.center_y < 1.5
            ):
                raise ConfigError(
                    f"implausible fit: center=({fitted.center_x:.2f}, {fitted.center_y:.2f}) "
                    f"window=({fitted.window_w:.2f}, {fitted.window_h:.2f})"
                )
        except ConfigError as e:
            log.warning("calibration fit rejected: %s", e)
            calib.pairs.clear()
            calib.idx = 0
            return self._crosshair("calib failed-redo 1")

        # Mutate the shared calibration in place: the composer and every
        # session hold references to this object.
        cal = self._rt.calibration
        cal.center_x, cal.center_y = fitted.center_x, fitted.center_y
        cal.window_w, cal.window_h = fitted.window_w, fitted.window_h
        self._calib = None
        self.active = False
        log.info(
            "calibrated: center=(%.3f, %.3f) window=(%.3f, %.3f)",
            cal.center_x, cal.center_y, cal.window_w, cal.window_h,
        )
        note = "saved"
        cfg_path = self._rt.cfg.config_path
        if cfg_path is None:
            note = "not saved"
        else:
            try:
                save_calibration(
                    CalibrationConfig(
                        center_x=cal.center_x,
                        center_y=cal.center_y,
                        window_w=cal.window_w,
                        window_h=cal.window_h,
                    ),
                    cfg_path,
                )
            except OSError:
                log.warning("could not persist calibration", exc_info=True)
                note = "save failed"
        if abs(cal.window_w) > 1.0 or abs(cal.window_h) > 1.0:
            log.warning("calibration window suspiciously large - verify camera.rotation")
        return self._set_scene(
            self._rt.composer.status("calibrated", self._next_seq(), "hold btn + ask")
        )

    def _answer_locate(self, question: str, capture, capture_jpeg: bytes, loc) -> AskResponse:
        """Answer a find-question from a direct-pointing result."""
        from lstk_eye.pipeline.types import Plan, PlanStep

        label = (loc.label or "target").strip()
        log.info("[%s] locate %r -> found=%s at %s", self.device_id, question,
                 loc.found, loc.point)
        self._history.append((question, label if loc.found else f"{label} not visible"))
        self._history = self._history[-MAX_HISTORY:]
        if not loc.found:
            scene = self._set_scene(
                self._rt.composer.status(label, self._next_seq(), "not in view")
            )
            return AskResponse(session_id=self.session_id, scene=scene, active=self.active)

        size = (loc.bbox[2], loc.bbox[3]) if loc.bbox else (0.12, 0.12)
        slide = Slide(
            index=0, total=1, label=label, anchor=loc.point, mark_id=None, size=size
        )
        self.session_id = self.session_id or uuid.uuid4().hex[:8]
        self.active = True
        self._slides = [slide]
        self._mask_by_id = {}
        self._style = "target"
        self._capture_bgr = capture
        self._capture_jpeg = capture_jpeg
        self._photos = []
        self._rt.store.save_request(
            self.session_id,
            capture_jpeg,
            encode_png(capture),
            question,
            Plan(steps=[PlanStep(label=label, mark_id=None)], summary=label),
            self._slides,
        )
        scene = self._enter_step(0)
        return AskResponse(session_id=self.session_id, scene=scene, active=True)

    def _refine_plan(self, capture, question: str, plan, masks):
        """Coarse-to-fine pass: when a find-answer anchors to a huge mask,
        crop that region, re-segment it at its native resolution (small parts
        get their own masks there), and ask the planner to point again inside
        the zoom. Falls back to the coarse answer on any failure."""
        from lstk_eye.pipeline.types import Plan, PlanStep

        if len(plan.steps) != 1 or plan.steps[0].mark_id is None:
            return plan, masks
        coarse = next((m for m in masks if m.mark_id == plan.steps[0].mark_id), None)
        if coarse is None or coarse.area < REFINE_AREA:
            return plan, masks
        x, y, w, h = coarse.bbox
        pad = 0.08
        cx0, cy0 = max(0.0, x - pad * w), max(0.0, y - pad * h)
        cx1, cy1 = min(1.0, x + w * (1 + pad)), min(1.0, y + h * (1 + pad))
        fh, fw = capture.shape[:2]
        crop = capture[int(cy0 * fh) : int(cy1 * fh), int(cx0 * fw) : int(cx1 * fw)]
        if crop.size == 0:
            return plan, masks
        try:
            sub_masks = self._rt.segmenter.segment(crop)
        except LstkError:
            return plan, masks
        if len(sub_masks) < 2:
            return plan, masks  # the zoom found no finer structure
        sub_png = encode_png(overlay_marks(crop, sub_masks))
        zoom_q = (
            f"{question}\n(This is a ZOOMED view of the relevant area - "
            f"point at the precise spot now.)"
        )
        try:
            sub_plan = self._rt.planner.plan(
                sub_png, zoom_q, sub_masks, history=self._history
            )
        except LstkError:
            log.info("refine pass failed; keeping the coarse answer", exc_info=True)
            return plan, masks
        if len(sub_plan.steps) != 1 or sub_plan.steps[0].mark_id is None:
            return plan, masks
        chosen = next(
            (m for m in sub_masks if m.mark_id == sub_plan.steps[0].mark_id), None
        )
        if chosen is None:
            return plan, masks
        # Map the crop-space mask back into full-frame normalized coordinates.
        sw, sh = cx1 - cx0, cy1 - cy0
        bx, by, bw, bh = chosen.bbox
        mapped = SegMask(
            mark_id=REFINE_ID_BASE + chosen.mark_id,
            bbox=(cx0 + bx * sw, cy0 + by * sh, bw * sw, bh * sh),
            centroid=(cx0 + chosen.centroid[0] * sw, cy0 + chosen.centroid[1] * sh),
            area=chosen.area * sw * sh,
            mask=None,
        )
        log.info(
            "refined %r: coarse area %.2f -> %.3f at (%.2f, %.2f)",
            question, coarse.area, mapped.area, *mapped.centroid,
        )
        step = PlanStep(
            label=sub_plan.steps[0].label or plan.steps[0].label,
            mark_id=mapped.mark_id,
        )
        return Plan(steps=[step], summary=plan.summary), [*masks, mapped]

    def _scaled_slide(self) -> Slide:
        """Current slide with its bbox size scaled by the tracker's apparent
        scale, so the highlight brackets grow and shrink with distance."""
        import dataclasses

        slide = self._slides[self._step]
        if slide.size is None or abs(self._track_scale - 1.0) < 0.02:
            return slide
        return dataclasses.replace(
            slide,
            size=(slide.size[0] * self._track_scale, slide.size[1] * self._track_scale),
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
