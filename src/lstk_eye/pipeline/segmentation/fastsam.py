"""FastSAM segment-everything backend via ultralytics.

Imported lazily by the factory; a missing ultralytics install surfaces there
as DependencyError with the install hint.
"""

import threading

import cv2
import numpy as np
from ultralytics import FastSAM

from lstk_eye.config import SegmenterConfig
from lstk_eye.pipeline.interfaces import Segmenter
from lstk_eye.pipeline.types import SegMask

# A mask covering nearly the whole frame is the tabletop/background, useless
# as an anchor target.
_MAX_MASK_FRACTION = 0.960


class FastSAMSegmenter(Segmenter):
    def __init__(self, cfg: SegmenterConfig):
        self._cfg = cfg
        self._model: FastSAM | None = None
        # The stage instance is shared across devices; guard the lazy init so
        # two first calls racing in the threadpool cannot load the model twice.
        self._init_lock = threading.Lock()

    def segment(self, image_bgr: np.ndarray) -> list[SegMask]:
        if self._model is None:
            # Deferred so constructing the pipeline stays cheap; the first
            # segment() call pays for the model load (and download).
            with self._init_lock:
                if self._model is None:
                    self._model = FastSAM(self._cfg.model)

        results = self._model(
            image_bgr,
            retina_masks=True,
            conf=self._cfg.conf,
            device=self._cfg.device,
            verbose=False,
        )
        result = results[0]
        if result.masks is None:
            return []

        h, w = image_bgr.shape[:2]
        frame_area = float(h * w)
        raw = result.masks.data.cpu().numpy()

        candidates: list[tuple[np.ndarray, float]] = []
        for plane in raw:
            mask = plane > 0.5
            if mask.shape != (h, w):
                # retina_masks normally matches the input; resize defensively
                # if the model returned its internal resolution.
                mask = (
                    cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                )
            area_px = float(np.count_nonzero(mask))
            fraction = area_px / frame_area
            if fraction < self._cfg.min_area or fraction > _MAX_MASK_FRACTION:
                continue
            candidates.append((mask, area_px))

        candidates.sort(key=lambda c: c[1], reverse=True)
        candidates = candidates[: self._cfg.max_masks]

        masks: list[SegMask] = []
        for i, (mask, area_px) in enumerate(candidates, start=1):
            ys, xs = np.nonzero(mask)
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            masks.append(
                SegMask(
                    mark_id=i,
                    bbox=(x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h),
                    centroid=(float(xs.mean()) / w, float(ys.mean()) / h),
                    area=area_px / frame_area,
                    mask=mask,
                )
            )
        return masks
