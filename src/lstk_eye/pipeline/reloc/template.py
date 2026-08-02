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

# Temporal consistency: after the target was lost, re-appearing requires two
# consecutive confident matches whose centers agree within this normalized
# distance - a single high-scoring random blob (same-colored clothing) jumps
# around and never passes. While visible, a confident match further than
# _OUTLIER_EPS from the smoothed center in one frame is treated as a miss
# instead of teleporting the marker.
_CONSIST_EPS = 0.08
_OUTLIER_EPS = 0.20

# Structure verification: intensity correlation alone latches onto flat
# same-brightness blobs (clothing). A candidate match must also correlate on
# the Laplacian (edge structure) - smooth impostors score near zero there.
# Skipped for genuinely textureless templates.
_EDGE_CONF = 0.25
_EDGE_MIN_STD = 4.0


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
        lap = cv2.Laplacian(template_gray, cv2.CV_32F, ksize=3)
        # Textureless templates cannot be edge-verified; disable the check.
        self._edge_check = float(lap.std()) >= _EDGE_MIN_STD
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
        self._scale = 1.0
        # Last confident-but-unconfirmed candidate while hidden (temporal
        # consistency check for re-appearing).
        self._pending: tuple[float, float] | None = None

    def update(self, frame_gray: np.ndarray) -> MatchResult:
        frame_h, frame_w = frame_gray.shape[:2]
        best_conf: float | None = None
        best_center: tuple[float, float] | None = None
        best_scale = 1.0
        best_roi: np.ndarray | None = None
        best_tpl: np.ndarray | None = None

        # Replicate-pad the frame so a target half-out of the camera frame
        # (the wearer turning away) still matches while any of it is visible;
        # without this, edge targets are lost the moment the template no
        # longer fits inside the frame.
        pad_w = max(1, round(self._norm_w * frame_w * max(self._cfg.scales) / 2))
        pad_h = max(1, round(self._norm_h * frame_h * max(self._cfg.scales) / 2))
        padded = cv2.copyMakeBorder(
            frame_gray, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE
        )

        for s in self._cfg.scales:
            tpl_w = round(self._norm_w * frame_w * s)
            tpl_h = round(self._norm_h * frame_h * s)
            if tpl_w < _MIN_TEMPLATE_PX or tpl_h < _MIN_TEMPLATE_PX:
                continue
            if tpl_w > padded.shape[1] or tpl_h > padded.shape[0]:
                continue
            interp = cv2.INTER_AREA if tpl_w < self._template.shape[1] else cv2.INTER_LINEAR
            tpl = cv2.resize(self._template, (tpl_w, tpl_h), interpolation=interp)
            scores = cv2.matchTemplate(padded, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(scores)
            if best_conf is None or max_val > best_conf:
                best_conf = float(max_val)
                best_center = (
                    (max_loc[0] - pad_w + tpl_w / 2.0) / frame_w,
                    (max_loc[1] - pad_h + tpl_h / 2.0) / frame_h,
                )
                best_scale = s
                best_roi = padded[
                    max_loc[1] : max_loc[1] + tpl_h, max_loc[0] : max_loc[0] + tpl_w
                ]
                best_tpl = tpl

        conf = 0.0 if best_conf is None else best_conf

        # Structure check: the matched patch must share edge structure with
        # the template, not just brightness distribution.
        if (
            conf >= self._cfg.disappear_conf
            and self._edge_check
            and best_roi is not None
            and best_tpl is not None
        ):
            lap_roi = cv2.Laplacian(best_roi, cv2.CV_32F, ksize=3)
            lap_tpl = cv2.Laplacian(best_tpl, cv2.CV_32F, ksize=3)
            edge_scores = cv2.matchTemplate(lap_roi, lap_tpl, cv2.TM_CCOEFF_NORMED)
            if float(edge_scores.max()) < _EDGE_CONF:
                conf = 0.0  # impostor: bright-similar but structurally flat
        confident = best_center is not None and conf >= self._cfg.appear_conf
        plausible = best_center is not None and conf >= self._cfg.disappear_conf

        outlier = (
            plausible
            and self._center is not None
            and abs(best_center[0] - self._center[0]) + abs(best_center[1] - self._center[1])
            > _OUTLIER_EPS
        )

        if self._visible:
            if confident and not outlier:
                self._misses = 0
                self._update_center(best_center)
                self._scale = 0.3 * best_scale + 0.7 * self._scale
            elif plausible and not outlier:
                # Dead band: hold state; still track the position gently.
                self._update_center(best_center)
            else:
                # Low confidence OR a teleporting "match": both are misses.
                self._misses += 1
                if self._misses >= self._cfg.miss_hide:
                    self._visible = False
                    self._pending = None
        else:
            if confident and self._edge_check:
                # Structure-verified matches are trustworthy on sight:
                # re-show immediately (recovery speed matters to the eyes).
                self._visible = True
                self._misses = 0
                self._center = best_center
                self._scale = best_scale
                self._pending = None
            elif confident:
                # Textureless template, no structure gate: fall back to
                # temporal consistency - two consecutive agreeing matches.
                if (
                    self._pending is not None
                    and abs(best_center[0] - self._pending[0])
                    + abs(best_center[1] - self._pending[1])
                    < _CONSIST_EPS
                ):
                    self._visible = True
                    self._misses = 0
                    self._center = best_center
                    self._scale = best_scale
                    self._pending = None
                else:
                    self._pending = best_center
            else:
                self._pending = None

        return MatchResult(
            found=self._visible,
            center=self._center,
            confidence=conf,
            scale=self._scale,
            misses=self._misses,
        )

    def _update_center(self, measured: tuple[float, float]) -> None:
        if self._center is None:
            self._center = measured
            return
        e = self._cfg.ema
        self._center = (
            e * measured[0] + (1.0 - e) * self._center[0],
            e * measured[1] + (1.0 - e) * self._center[1],
        )


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
