"""Data types flowing between pipeline stages.

Coordinate convention: everything in the pipeline uses NORMALIZED image
coordinates - floats in [0, 1] relative to the frame the value was measured
on. This makes values resolution-independent, which matters because capture
frames (e.g. 1024x768) and relocalization preview frames (e.g. 320x240) have
different sizes. Conversion to display pixels happens exactly once, in the
calibration/composition step.

Images are numpy arrays in OpenCV BGR order unless a parameter name says
otherwise (``gray`` = single channel, ``png``/``jpeg`` = encoded bytes).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SegMask:
    """One segmented region. ``mark_id`` is the number drawn on the
    Set-of-Marks overlay and the only identifier the planner ever sees."""

    mark_id: int
    bbox: tuple[float, float, float, float]  # x, y, w, h, normalized
    centroid: tuple[float, float]  # normalized
    area: float  # fraction of frame area, in [0, 1]
    mask: np.ndarray | None = None  # bool array at source resolution, optional

    def __post_init__(self) -> None:
        x, y, w, h = self.bbox
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"bbox origin out of [0,1]: {self.bbox}")
        if w <= 0.0 or h <= 0.0 or x + w > 1.0 + 1e-6 or y + h > 1.0 + 1e-6:
            raise ValueError(f"bbox size invalid: {self.bbox}")


@dataclass
class Transcript:
    text: str
    language: str | None = None


@dataclass
class PlanStep:
    """One instruction slide. ``mark_id`` is None for steps with no visual
    anchor (pure text, e.g. "wait 30 seconds")."""

    label: str  # short instruction shown on the HUD, ~2 lines max
    mark_id: int | None = None
    detail: str | None = None  # longer wording, kept in session logs only


@dataclass
class Plan:
    steps: list[PlanStep] = field(default_factory=list)
    summary: str | None = None


@dataclass
class Slide:
    """A plan step resolved against the capture frame: anchor is the mask
    centroid in normalized camera coordinates, or None for text-only steps.
    ``size`` is the normalized (w, h) of the target's bbox, used to scale the
    highlight brackets in find-this sessions."""

    index: int  # 0-based
    total: int
    label: str
    anchor: tuple[float, float] | None = None
    mark_id: int | None = None
    size: tuple[float, float] | None = None


@dataclass
class MatchResult:
    """Output of one relocalization update against a preview frame."""

    found: bool
    center: tuple[float, float] | None = None  # normalized, in preview frame
    confidence: float = 0.0
