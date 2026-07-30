"""Deterministic classical-CV segmenter for tests and the mock profile.

Otsu threshold (both polarities, the one yielding more usable regions wins),
morphological cleanup, then connected components. Reliable for solid distinct
shapes on a plain background; makes no promises on natural scenes.
"""

import cv2
import numpy as np

from lstk_eye.config import SegmenterConfig
from lstk_eye.pipeline.interfaces import Segmenter
from lstk_eye.pipeline.types import SegMask

# A single region covering more than this fraction of the frame is treated as
# background leaking through the threshold, not an object.
_MAX_REGION_FRACTION = 0.60

_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


class MockSegmenter(Segmenter):
    def __init__(self, cfg: SegmenterConfig):
        self._cfg = cfg

    def segment(self, image_bgr: np.ndarray) -> list[SegMask]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        frame_area = float(h * w)

        candidates = [
            self._regions(gray, cv2.THRESH_BINARY, frame_area),
            self._regions(gray, cv2.THRESH_BINARY_INV, frame_area),
        ]
        # More valid regions wins; on a tie prefer the polarity with less total
        # foreground (the one that kept the background as background).
        regions = max(candidates, key=lambda r: (len(r), -sum(a for _, a, *_ in r)))

        regions.sort(key=lambda r: r[1], reverse=True)
        regions = regions[: self._cfg.max_masks]

        masks: list[SegMask] = []
        for i, (mask, area_px, bbox_px, centroid_px) in enumerate(regions, start=1):
            x, y, bw, bh = bbox_px
            cx, cy = centroid_px
            masks.append(
                SegMask(
                    mark_id=i,
                    bbox=(x / w, y / h, bw / w, bh / h),
                    centroid=(cx / w, cy / h),
                    area=area_px / frame_area,
                    mask=mask,
                )
            )
        return masks

    def _regions(
        self, gray: np.ndarray, polarity: int, frame_area: float
    ) -> list[tuple[np.ndarray, float, tuple[int, int, int, int], tuple[float, float]]]:
        """Threshold with one Otsu polarity and extract filtered components as
        (bool mask, area_px, pixel bbox, pixel centroid) tuples."""
        _, binary = cv2.threshold(gray, 0, 255, polarity + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _KERNEL)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _KERNEL)

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        out = []
        for label in range(1, n):  # label 0 is background
            area_px = float(stats[label, cv2.CC_STAT_AREA])
            fraction = area_px / frame_area
            if fraction < self._cfg.min_area or fraction > _MAX_REGION_FRACTION:
                continue
            bbox_px = (
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
            )
            centroid_px = (float(centroids[label, 0]), float(centroids[label, 1]))
            out.append((labels == label, area_px, bbox_px, centroid_px))
        return out
