"""Whisper on Apple-silicon GPU via mlx-whisper.

An order of magnitude faster than CPU faster-whisper on M-series Macs
(large-v3-turbo transcribes 4 s of speech in ~0.3 s) at large-v3 accuracy.
macOS-only; install with the ``mlx`` extra.
"""

import logging

import mlx_whisper
import numpy as np

from lstk_eye.audio import resample_linear, wav_to_float32
from lstk_eye.config import SttConfig
from lstk_eye.pipeline.interfaces import SpeechToText
from lstk_eye.pipeline.types import Transcript

log = logging.getLogger(__name__)

_TARGET_SR = 16000


class MlxWhisperSpeechToText(SpeechToText):
    def __init__(self, cfg: SttConfig):
        self._cfg = cfg
        self._repo = (
            cfg.model
            if "/" in cfg.model
            else "mlx-community/whisper-large-v3-turbo"
        )
        # Warm start: load weights now (a few seconds from cache) so the
        # first voice request is not the one that pays for it.
        try:
            mlx_whisper.transcribe(
                np.zeros(_TARGET_SR // 2, dtype=np.float32), path_or_hf_repo=self._repo
            )
            log.info("mlx-whisper %s warmed up", self._repo)
        except Exception:
            log.warning("mlx-whisper warmup failed", exc_info=True)

    def transcribe(self, wav: bytes) -> Transcript:
        audio, sr = wav_to_float32(wav)
        if sr != _TARGET_SR:
            audio = resample_linear(audio, sr, _TARGET_SR)
        kwargs = {}
        if self._cfg.language:
            kwargs["language"] = self._cfg.language
        out = mlx_whisper.transcribe(audio, path_or_hf_repo=self._repo, **kwargs)
        return Transcript(
            text=str(out.get("text", "")).strip(),
            language=out.get("language"),
        )
