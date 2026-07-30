"""WAV encode/decode helpers built on stdlib ``wave`` plus numpy.

The device sends 16-bit PCM WAV; whisper wants float32 mono 16 kHz. These
helpers cover that path plus the reverse direction for tests and tools.
"""

import io
import wave

import numpy as np

from lstk_eye.errors import PipelineError


def wav_to_float32(wav: bytes) -> tuple[np.ndarray, int]:
    """Decode 16-bit PCM WAV bytes to ``(samples, sample_rate)``.

    ``samples`` is mono float32 in [-1, 1]; stereo input is downmixed by
    averaging the channels. Any other layout (non-PCM, 8/24/32-bit, more than
    two channels) raises :class:`PipelineError`.
    """
    try:
        with wave.open(io.BytesIO(wav), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError) as e:
        raise PipelineError(f"not a valid WAV stream: {e}") from e
    if sampwidth != 2:
        raise PipelineError(
            f"unsupported WAV sample width: {sampwidth * 8}-bit (need 16-bit PCM)"
        )
    if n_channels not in (1, 2):
        raise PipelineError(
            f"unsupported WAV channel count: {n_channels} (need mono or stereo)"
        )
    x = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if n_channels == 2:
        x = x.reshape(-1, 2).mean(axis=1)
    return x, sample_rate


def resample_linear(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample by linear interpolation - crude but adequate for speech STT."""
    if sr == target_sr or x.size == 0:
        return x.astype(np.float32, copy=False)
    n_out = max(1, round(x.size * target_sr / sr))
    positions = np.linspace(0.0, x.size - 1, n_out)
    return np.interp(positions, np.arange(x.size), x).astype(np.float32)


def float32_to_wav(x: np.ndarray, sr: int) -> bytes:
    """Encode mono float32 samples in [-1, 1] as 16-bit PCM WAV bytes."""
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()
