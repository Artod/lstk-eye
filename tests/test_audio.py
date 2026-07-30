"""Tests for lstk_eye.audio WAV helpers."""

import io
import wave

import numpy as np
import pytest

from lstk_eye.audio import float32_to_wav, resample_linear, wav_to_float32
from lstk_eye.errors import PipelineError


def encode_wav(samples: np.ndarray, sr: int, channels: int = 1, sampwidth: int = 2) -> bytes:
    """Encode raw integer samples (interleaved if multichannel) as WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def sine_int16(freq: float, sr: int, seconds: float = 0.25, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (np.sin(2 * np.pi * freq * t) * amp * 32767).astype("<i2")


@pytest.mark.parametrize("sr", [8000, 16000])
def test_mono_sine_decodes(sr: int) -> None:
    pcm = sine_int16(440.0, sr)
    x, got_sr = wav_to_float32(encode_wav(pcm, sr))
    assert got_sr == sr
    assert x.dtype == np.float32
    assert x.shape == (pcm.size,)
    assert np.max(np.abs(x)) <= 1.0
    assert np.allclose(x, pcm.astype(np.float32) / 32768.0, atol=1e-6)


@pytest.mark.parametrize("sr", [8000, 16000])
def test_mono_silence_decodes_to_zeros(sr: int) -> None:
    pcm = np.zeros(sr // 2, dtype="<i2")
    x, got_sr = wav_to_float32(encode_wav(pcm, sr))
    assert got_sr == sr
    assert x.shape == (pcm.size,)
    assert np.all(x == 0.0)


def test_stereo_averages_channels() -> None:
    sr = 16000
    left = sine_int16(440.0, sr)
    right = np.zeros_like(left)
    interleaved = np.column_stack([left, right]).reshape(-1)
    x, got_sr = wav_to_float32(encode_wav(interleaved, sr, channels=2))
    assert got_sr == sr
    assert x.shape == (left.size,)
    assert np.allclose(x, left.astype(np.float32) / 32768.0 / 2.0, atol=1e-6)


def test_rejects_8bit_samples() -> None:
    pcm = np.full(1000, 128, dtype=np.uint8)
    with pytest.raises(PipelineError, match="sample width"):
        wav_to_float32(encode_wav(pcm, 8000, sampwidth=1))


def test_rejects_three_channels() -> None:
    pcm = np.zeros(999, dtype="<i2")
    with pytest.raises(PipelineError, match="channel count"):
        wav_to_float32(encode_wav(pcm, 16000, channels=3))


def test_rejects_non_wav_bytes() -> None:
    with pytest.raises(PipelineError, match="not a valid WAV"):
        wav_to_float32(b"definitely not RIFF data")


def test_roundtrip_float32_to_wav() -> None:
    sr = 16000
    t = np.arange(sr // 4) / sr
    x = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    decoded, got_sr = wav_to_float32(float32_to_wav(x, sr))
    assert got_sr == sr
    assert decoded.shape == x.shape
    # One LSB of 16-bit quantization error is 1/32768.
    assert np.allclose(decoded, x, atol=2.0 / 32768.0)


def test_resample_upsamples_to_expected_length() -> None:
    x = np.zeros(8000, dtype=np.float32)
    y = resample_linear(x, 8000, 16000)
    assert y.shape == (16000,)
    assert y.dtype == np.float32


def test_resample_downsamples_to_expected_length() -> None:
    x = np.zeros(16000, dtype=np.float32)
    y = resample_linear(x, 16000, 8000)
    assert y.shape == (8000,)


def test_resample_same_rate_is_identity() -> None:
    x = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
    y = resample_linear(x, 16000, 16000)
    assert np.array_equal(y, x)


def test_resample_preserves_constant_signal() -> None:
    x = np.full(4000, 0.25, dtype=np.float32)
    y = resample_linear(x, 8000, 16000)
    assert np.allclose(y, 0.25, atol=1e-6)
