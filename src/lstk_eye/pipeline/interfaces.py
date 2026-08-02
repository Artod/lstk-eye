"""Abstract interfaces for every swappable pipeline stage.

Each stage has at least two implementations: a real one (possibly behind an
optional dependency) and a mock used for tests and the ``mock`` profile.
Implementations are constructed via ``lstk_eye.pipeline.factories`` from the
app config; nothing outside a stage's own module may import its heavy
dependencies at module level.
"""

from abc import ABC, abstractmethod

import numpy as np

from lstk_eye.pipeline.types import LocateResult, MatchResult, Plan, SegMask, Transcript


class SpeechToText(ABC):
    """WAV bytes in, transcript out. Input is 16-bit PCM mono WAV."""

    @abstractmethod
    def transcribe(self, wav: bytes) -> Transcript: ...


class Segmenter(ABC):
    """Segment-everything over a BGR frame. Returns masks sorted by mark_id,
    starting at 1, already filtered (min area, max count) per config."""

    @abstractmethod
    def segment(self, image_bgr: np.ndarray) -> list[SegMask]: ...


class Planner(ABC):
    """Turns (marked image, question, available marks) into a step plan.

    ``marked_png`` is the Set-of-Marks overlay as encoded PNG. Implementations
    must return steps whose mark_id values are drawn from ``marks`` (or None),
    and must raise PlanningError rather than return out-of-range marks.

    ``history`` is the ongoing chat: (question, answer) pairs from earlier
    turns of the same session, oldest first. The current photo supersedes the
    ones from earlier turns; history exists so follow-ups ("what about now?",
    "the other one") resolve against what was already said.
    """

    @abstractmethod
    def plan(
        self,
        marked_png: bytes,
        question: str,
        marks: list[SegMask],
        history: list[tuple[str, str]] | None = None,
    ) -> Plan: ...

    def locate(
        self,
        image_bgr: np.ndarray,
        question: str,
        history: list[tuple[str, str]] | None = None,
    ) -> LocateResult | None:
        """Direct pointing for find-this questions: the model returns the
        target's normalized center/box itself, zooming into crops as needed -
        works at any object scale without segmentation in the critical path.
        Default: unsupported (None) - the caller falls back to Set-of-Marks."""
        return None


class TargetTracker(ABC):
    """Tracks one slide's target across preview frames (relocalization).

    Created per step from the capture frame; ``update`` is called for every
    preview frame while that step is active. Implementations own their
    hysteresis: ``found`` in the result is the debounced, user-facing state
    (appear fast, disappear lazily), not the raw per-frame match.
    """

    @abstractmethod
    def update(self, frame_gray: np.ndarray) -> MatchResult: ...


class RelocalizerFactory(ABC):
    @abstractmethod
    def create(
        self, capture_bgr: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> TargetTracker:
        """Build a tracker for the target inside ``bbox`` (normalized) of the
        full-resolution capture frame."""
        ...


class Calibration(ABC):
    """Maps normalized camera coordinates to OLED pixels.

    The display shows a window into the camera image: display center
    corresponds to camera point (center_x, center_y) and the display spans
    (window_w, window_h) of the frame. Measured once per build at working
    distance; defaults assume a centered window.
    """

    @abstractmethod
    def to_display(self, pt: tuple[float, float]) -> tuple[int, int]:
        """Unclamped display pixel position (may fall outside the panel)."""
        ...

    @abstractmethod
    def in_window(self, pt: tuple[float, float], margin_px: int = 0) -> bool: ...

    @abstractmethod
    def edge_for(self, pt: tuple[float, float]) -> str:
        """Which display edge ('left'/'right'/'up'/'down') is nearest to an
        out-of-window target - used to pick the chevron."""
        ...
