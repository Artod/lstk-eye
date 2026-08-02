"""Application configuration.

Precedence (highest first): constructor kwargs (CLI flags) > environment
variables > TOML file. Environment variables use the ``LSTK_`` prefix with
``__`` as the nesting delimiter, e.g. ``LSTK_PLANNER__MODEL=claude-opus-5``.

TOML is looked up at ``./lstk-eye.toml`` then ``~/.config/lstk-eye/config.toml``
unless an explicit path is given to :func:`load_config`.

The ``profile`` switch exists so the whole system runs with zero heavy
dependencies and zero API keys: ``--profile mock`` swaps every pipeline stage
for its mock and allows text-based asks (no microphone needed).
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

DEFAULT_CONFIG_PATHS = (
    Path("lstk-eye.toml"),
    Path.home() / ".config" / "lstk-eye" / "config.toml",
)

# Explicit --config path; consulted by settings_customise_sources at
# construction time. Set only via load_config().
_toml_path_override: Path | None = None


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8321
    # Advertise _lstk-eye._tcp via mDNS so the glasses can find the laptop
    # without a hardcoded IP.
    zeroconf: bool = True


class SttConfig(BaseModel):
    backend: Literal["whisper", "mock"] = "whisper"
    model: str = "small"  # faster-whisper model size
    language: str | None = None  # None or "" = autodetect
    device: str = "auto"
    compute_type: str = "int8"

    @field_validator("language")
    @classmethod
    def _empty_means_autodetect(cls, v: str | None) -> str | None:
        # TOML has no null, so the natural way to spell "autodetect" in a
        # config file is an empty string - which faster-whisper rejects.
        return v or None


class CameraConfig(BaseModel):
    """Physical camera mount. ``rotation`` is how many degrees CLOCKWISE the
    incoming frames must be rotated to become upright - set it when the
    module is mounted sideways. Applied to every photo and preview at
    ingestion, so the whole pipeline works in upright coordinates."""

    rotation: Literal[0, 90, 180, 270] = 0
    # Sensor mirror flags, applied AFTER rotation. Diagnose from calibration:
    # a negative fitted window on exactly one axis means that axis is
    # mirrored (flip_v for height, flip_h for width).
    flip_v: bool = False
    flip_h: bool = False


class SegmenterConfig(BaseModel):
    backend: Literal["fastsam", "mock"] = "fastsam"
    model: str = "FastSAM-s.pt"  # auto-downloaded by ultralytics
    device: str = "cpu"
    max_masks: int = 25
    min_area: float = 0.0008  # fraction of frame area
    conf: float = 0.4


class PlannerConfig(BaseModel):
    backend: Literal["anthropic", "claude-cli", "mock"] = "anthropic"
    model: str = "claude-opus-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_tokens: int = 8000
    max_steps: int = 7
    # Must stay below the firmware's blocking /ask timeout (30 s) so the
    # device sees an error scene instead of timing out on its own.
    timeout_s: float = 25.0


class DisplayConfig(BaseModel):
    """Usable area of the panel. The optics crop the panel edges, so all
    content is laid out inside [pad_x, 127-pad_x] x [pad_y, 63-pad_y].
    Tune to the physical build: shrink pads until content reaches the visible
    border, grow until nothing is cut off."""

    pad_x: int = 16
    pad_y: int = 2


class CalibrationConfig(BaseModel):
    """Camera->display mapping, measured at working distance (~60 cm).

    Display center shows camera point (center_x, center_y); the full 128x64
    panel spans (window_w, window_h) of the camera frame. Defaults assume the
    HUD window is centered and covers ~40% of the frame width.
    """

    center_x: float = 0.5
    center_y: float = 0.5
    window_w: float = 0.40
    window_h: float = 0.40


class RelocConfig(BaseModel):
    # Raised after field testing: at 0.60 the matcher latched onto
    # similar-colored blobs (a yellow jacket "found" as a tulip). A wrong
    # marker is worse than an honest "look back".
    appear_conf: float = 0.70
    disappear_conf: float = 0.55
    miss_hide: int = 3  # consecutive misses before the marker hides
    ema: float = 0.65  # smoothing factor for anchor position, 1.0 = no smoothing
    scales: list[float] = Field(default_factory=lambda: [0.85, 1.0, 1.18])


class StorageConfig(BaseModel):
    dir: Path = Path("runs")
    save_sessions: bool = True


class DebugConfig(BaseModel):
    # Allow POST /api/v1/ask?text=... to bypass STT. Forced on by the mock
    # profile; useful for the simulator and tests.
    allow_text_ask: bool = False


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LSTK_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    profile: Literal["real", "mock"] = "real"
    # The TOML file this config was loaded from (or the default location to
    # persist to); set by load_config, used by calibration to save results.
    config_path: Path | None = None
    server: ServerConfig = Field(default_factory=ServerConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    segmenter: SegmenterConfig = Field(default_factory=SegmenterConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    reloc: RelocConfig = Field(default_factory=RelocConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        toml_file = _toml_path_override or next(
            (p for p in DEFAULT_CONFIG_PATHS if p.is_file()), None
        )
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        if toml_file is not None:
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file))
        return tuple(sources)

    def resolved(self) -> "AppConfig":
        """Apply the profile. ``mock`` swaps every backend for its mock and
        enables text asks so the system runs with no models and no API key."""
        if self.profile != "mock":
            return self
        return self.model_copy(
            update={
                "stt": self.stt.model_copy(update={"backend": "mock"}),
                "segmenter": self.segmenter.model_copy(update={"backend": "mock"}),
                "planner": self.planner.model_copy(update={"backend": "mock"}),
                "debug": self.debug.model_copy(update={"allow_text_ask": True}),
            }
        )


def load_config(path: Path | None = None, **overrides) -> AppConfig:
    """Build the config with CLI ``overrides`` > env > TOML, then apply the
    profile. ``path`` pins the TOML file explicitly (error if missing)."""
    global _toml_path_override
    if path is not None and not path.is_file():
        from lstk_eye.errors import ConfigError

        raise ConfigError(f"config file not found: {path}")
    _toml_path_override = path
    try:
        cfg = AppConfig(**{k: v for k, v in overrides.items() if v is not None})
    finally:
        _toml_path_override = None
    if cfg.config_path is None:
        used = path or next((p for p in DEFAULT_CONFIG_PATHS if p.is_file()), None)
        cfg = cfg.model_copy(update={"config_path": used or DEFAULT_CONFIG_PATHS[0]})
    return cfg.resolved()


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return '"' + str(v).replace('\\', '\\\\').replace('"', '\\"') + '"'


def save_calibration(cal: CalibrationConfig, path: Path) -> Path:
    """Merge fitted calibration values into the TOML config at ``path``
    (created if absent). Rewrites the file; comments are not preserved."""
    import tomllib

    data: dict = {}
    if path.is_file():
        data = tomllib.loads(path.read_text())
    data["calibration"] = {
        "center_x": round(cal.center_x, 5),
        "center_y": round(cal.center_y, 5),
        "window_w": round(cal.window_w, 5),
        "window_h": round(cal.window_h, 5),
    }
    lines: list[str] = []
    for key, value in data.items():
        if not isinstance(value, dict):
            lines.append(f"{key} = {_toml_value(value)}")
    for section, table in data.items():
        if isinstance(table, dict):
            lines.append("")
            lines.append(f"[{section}]")
            for key, value in table.items():
                lines.append(f"{key} = {_toml_value(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
