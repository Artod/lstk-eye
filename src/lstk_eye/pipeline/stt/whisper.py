"""faster-whisper STT backend.

The model is loaded lazily on the first transcribe call so that building the
pipeline stays fast and a misconfigured model name fails at use time with a
clear traceback, not at server startup.
"""

from faster_whisper import WhisperModel

from lstk_eye.audio import resample_linear, wav_to_float32
from lstk_eye.config import SttConfig
from lstk_eye.pipeline.interfaces import SpeechToText
from lstk_eye.pipeline.types import Transcript

WHISPER_SAMPLE_RATE = 16_000


class WhisperSpeechToText(SpeechToText):
    def __init__(self, cfg: SttConfig):
        self._cfg = cfg
        self._model: WhisperModel | None = None

    def transcribe(self, wav: bytes) -> Transcript:
        if self._model is None:
            self._model = WhisperModel(
                self._cfg.model,
                device=self._cfg.device,
                compute_type=self._cfg.compute_type,
            )
        x, sr = wav_to_float32(wav)
        if sr != WHISPER_SAMPLE_RATE:
            x = resample_linear(x, sr, WHISPER_SAMPLE_RATE)
        segments, info = self._model.transcribe(x, language=self._cfg.language)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return Transcript(text=text, language=info.language)
