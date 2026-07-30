"""Template-matching relocalization.

The capture frame yields a grayscale template of the active step's target;
every preview frame is scanned with multi-scale normalized cross-correlation
(``cv2.matchTemplate`` TM_CCOEFF_NORMED). All geometry travels as normalized
coordinates, so the capture (e.g. 1024x768) and the preview stream (e.g.
320x240) never need to share a resolution: the template is resized per frame
to the pixel footprint the target should have at that frame's size.

The tracker owns the user-facing hysteresis: a single confident frame shows
the target immediately, while hiding requires ``miss_hide`` consecutive
low-confidence frames. Confidences between the two thresholds hold the
current state. The reported center is EMA-smoothed to keep the HUD arrow
steady against per-frame match jitter.
"""

import cv2
import numpy as np

from lstk_eye.config import RelocConfig
from lstk_eye.errors import PipelineError
from lstk_eye.pipeline.interfaces import RelocalizerFactory, TargetTracker
from lstk_eye.pipeline.types import MatchResult

# Padding added around the target bbox when cropping the template, as a
# fraction of the bbox size per side. A little context ring makes the
# correlation peak sharper than the bare object alone.
_PAD = 0.15

# Scales are skipped when the resized template would drop below this many
# pixels on either axis (too little structure to correlate).
_MIN_TEMPLATE_PX = 8


class TemplateTracker(TargetTracker):
    """Tracks one target across preview frames via template matching."""

    def __init__(
        self,
        cfg: RelocConfig,
        template_gray: np.ndarray,
        norm_size: tuple[float, float],
    ) -> None:
        self._cfg = cfg
        self._template = template_gray
        # Normalized (w, h) of the template region on its source frame; used
        # to compute the template's expected pixel size on any preview frame.
        self._norm_w, self._norm_h = norm_size
        # The tracker is constructed from a frame where the target is known
        # present, so the debounced state starts visible: hiding must cost
        # miss_hide consecutive misses even on the very first preview frames
        # (which are often motion-blurred right after the button release).
        self._visible = True
        self._misses = 0
        self._center: tuple[float, float] | None = None

    def update(self, frame_gray: np.ndarray) -> MatchResult:
        frame_h, frame_w = frame_gray.shape[:2]
        best_conf: float | None = None
        best_center: tuple[float, float] | None = None

        for s in self._cfg.scales:
            tpl_w = round(self._norm_w * frame_w * s)
            tpl_h = round(self._norm_h * frame_h * s)
            if tpl_w < _MIN_TEMPLATE_PX or tpl_h < _MIN_TEMPLATE_PX:
                continue
            if tpl_w > frame_w or tpl_h > frame_h:
                continue
            interp = cv2.INTER_AREA if tpl_w < self._template.shape[1] else cv2.INTER_LINEAR
            tpl = cv2.resize(self._template, (tpl_w, tpl_h), interpolation=interp)
            scores = cv2.matchTemplate(frame_gray, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(scores)
            if best_conf is None or max_val > best_conf:
                best_conf = float(max_val)
                best_center = (
                    (max_loc[0] + tpl_w / 2.0) / frame_w,
                    (max_loc[1] + tpl_h / 2.0) / frame_h,
                )

        conf = 0.0 if best_conf is None else best_conf

        # Anchor position: EMA-smoothed, updated only on frames confident
        # enough that the peak is plausibly the target.
        if best_center is not None and conf >= self._cfg.disappear_conf:
            if self._center is None:
                self._center = best_center
            else:
                e = self._cfg.ema
                self._center = (
                    e * best_center[0] + (1.0 - e) * self._center[0],
                    e * best_center[1] + (1.0 - e) * self._center[1],
                )

        # Hysteresis: appear fast, disappear lazily.
        if conf >= self._cfg.appear_conf:
            self._visible = True
            self._misses = 0
        elif conf < self._cfg.disappear_conf:
            self._misses += 1
            if self._misses >= self._cfg.miss_hide:
                self._visible = False
        # Dead band between the thresholds: hold state, count nothing.

        return MatchResult(found=self._visible, center=self._center, confidence=conf)


class TemplateRelocalizerFactory(RelocalizerFactory):
    def __init__(self, cfg: RelocConfig) -> None:
        self._cfg = cfg

    def create(
        self, capture_bgr: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> TargetTracker:
        gray = (
            cv2.cvtColor(capture_bgr, cv2.COLOR_BGR2GRAY)
            if capture_bgr.ndim == 3
            else capture_bgr
        )
        frame_h, frame_w = gray.shape[:2]

        x, y, w, h = bbox
        x0 = max(0.0, x - _PAD * w)
        y0 = max(0.0, y - _PAD * h)
        x1 = min(1.0, x + w + _PAD * w)
        y1 = min(1.0, y + h + _PAD * h)

        px0 = int(round(x0 * frame_w))
        py0 = int(round(y0 * frame_h))
        px1 = int(round(x1 * frame_w))
        py1 = int(round(y1 * frame_h))
        template = gray[py0:py1, px0:px1]
        if template.size == 0:
            raise PipelineError(f"relocalization template crop is empty for bbox {bbox}")

        norm_size = ((px1 - px0) / frame_w, (py1 - py0) / frame_h)
        return TemplateTracker(self._cfg, template, norm_size)
