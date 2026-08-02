"""Construct pipeline stages from config.

Heavy dependencies (faster-whisper, ultralytics) are imported lazily inside
the matching branch, so the mock profile and the test suite never touch them.
A missing optional dependency surfaces as DependencyError with the exact
install command.
"""

from lstk_eye.config import AppConfig
from lstk_eye.errors import ConfigError, DependencyError
from lstk_eye.pipeline.interfaces import Planner, RelocalizerFactory, Segmenter, SpeechToText


def create_stt(cfg: AppConfig) -> SpeechToText:
    backend = cfg.stt.backend
    if backend == "mock":
        from lstk_eye.pipeline.stt.mock import MockSpeechToText

        return MockSpeechToText()
    if backend == "mlx":
        try:
            from lstk_eye.pipeline.stt.mlx import MlxWhisperSpeechToText
        except ImportError as e:
            raise DependencyError(
                "mlx-whisper is not installed (needed for stt.backend=mlx, macOS only)",
                install_hint='pip install "lstk-eye[mlx]"',
            ) from e
        return MlxWhisperSpeechToText(cfg.stt)
    if backend == "whisper":
        try:
            from lstk_eye.pipeline.stt.whisper import WhisperSpeechToText
        except ImportError as e:
            raise DependencyError(
                "faster-whisper is not installed (needed for stt.backend=whisper)",
                install_hint='pip install "lstk-eye[stt]"',
            ) from e
        return WhisperSpeechToText(cfg.stt)
    raise ConfigError(f"unknown stt backend: {backend}")


def create_segmenter(cfg: AppConfig) -> Segmenter:
    backend = cfg.segmenter.backend
    if backend == "mock":
        from lstk_eye.pipeline.segmentation.mock import MockSegmenter

        return MockSegmenter(cfg.segmenter)
    if backend == "fastsam":
        try:
            from lstk_eye.pipeline.segmentation.fastsam import FastSAMSegmenter
        except ImportError as e:
            raise DependencyError(
                "ultralytics is not installed (needed for segmenter.backend=fastsam)",
                install_hint='pip install "lstk-eye[seg]"',
            ) from e
        return FastSAMSegmenter(cfg.segmenter)
    raise ConfigError(f"unknown segmenter backend: {backend}")


def create_planner(cfg: AppConfig) -> Planner:
    backend = cfg.planner.backend
    if backend == "mock":
        from lstk_eye.pipeline.planner.mock import MockPlanner

        return MockPlanner()
    if backend == "anthropic":
        from lstk_eye.pipeline.planner.anthropic_planner import AnthropicPlanner

        return AnthropicPlanner(cfg.planner)
    if backend == "claude-cli":
        from lstk_eye.pipeline.planner.claude_cli import ClaudeCliPlanner

        return ClaudeCliPlanner(cfg.planner)
    raise ConfigError(f"unknown planner backend: {backend}")


def create_relocalizer(cfg: AppConfig) -> RelocalizerFactory:
    from lstk_eye.pipeline.reloc.template import TemplateRelocalizerFactory

    return TemplateRelocalizerFactory(cfg.reloc)
