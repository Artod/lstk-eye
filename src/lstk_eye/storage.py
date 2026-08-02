"""Best-effort persistence of pipeline runs for debugging and calibration.

Every ask request gets a directory under the configured runs dir with the
capture, the marked overlay, the transcript, and the plan. Failures to save
never fail the request.
"""

import dataclasses
import json
import logging
import time

from lstk_eye.config import StorageConfig
from lstk_eye.pipeline.types import Plan, Slide

log = logging.getLogger(__name__)


class RunStore:
    def __init__(self, cfg: StorageConfig):
        self._cfg = cfg

    def save_request(
        self,
        session_id: str,
        capture_jpeg: bytes,
        marked_png: bytes,
        question: str,
        plan: Plan,
        slides: list[Slide],
    ) -> None:
        if not self._cfg.save_sessions:
            return
        try:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            run_dir = self._cfg.dir / f"{stamp}_{session_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "capture.jpg").write_bytes(capture_jpeg)
            (run_dir / "marked.png").write_bytes(marked_png)
            (run_dir / "question.txt").write_text(question, encoding="utf-8")
            (run_dir / "plan.json").write_text(
                json.dumps(
                    {
                        "plan": dataclasses.asdict(plan),
                        "slides": [dataclasses.asdict(s) for s in slides],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            log.warning("failed to save run for session %s", session_id, exc_info=True)

    def save_calibration_frame(
        self, point_idx: int, jpeg: bytes, annotated_bgr, ok: bool
    ) -> None:
        """Persist every calibration click (raw + detection overlay) under
        runs/calibration/ so 'no marker' failures are inspectable."""
        if not self._cfg.save_sessions:
            return
        try:
            import cv2

            calib_dir = self._cfg.dir / "calibration"
            calib_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            tag = "ok" if ok else "miss"
            (calib_dir / f"{stamp}_p{point_idx + 1}_{tag}.jpg").write_bytes(jpeg)
            cv2.imwrite(str(calib_dir / f"{stamp}_p{point_idx + 1}_{tag}_detect.jpg"), annotated_bgr)
        except OSError:
            log.warning("failed to save calibration frame", exc_info=True)
