"""Tests for the STT backends."""

import numpy as np
import pytest

from lstk_eye.audio import float32_to_wav
from lstk_eye.pipeline.stt.mock import MockSpeechToText
from lstk_eye.pipeline.types import Transcript


def test_mock_prefix_returns_embedded_text() -> None:
    stt = MockSpeechToText()
    result = stt.transcribe(b"MOCKTEXT:" + "как проверить напряжение батареи".encode())
    assert isinstance(result, Transcript)
    assert result.text == "как проверить напряжение батареи"


def test_mock_plain_wav_returns_canned_transcript() -> None:
    stt = MockSpeechToText()
    wav = float32_to_wav(np.zeros(1600, dtype=np.float32), 16000)
    result = stt.transcribe(wav)
    assert isinstance(result, Transcript)
    assert result.text == "how do I check battery voltage"


@pytest.mark.slow
def test_whisper_transcribes_silence() -> None:
    pytest.importorskip("faster_whisper")
    from lstk_eye.config import SttConfig
    from lstk_eye.pipeline.stt.whisper import WhisperSpeechToText

    cfg = SttConfig(backend="whisper", model="tiny", device="cpu", compute_type="int8")
    stt = WhisperSpeechToText(cfg)
    wav = float32_to_wav(np.zeros(16000, dtype=np.float32), 16000)
    result = stt.transcribe(wav)
    assert isinstance(result, Transcript)
    assert isinstance(result.text, str)
