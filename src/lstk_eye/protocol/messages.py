"""Wire protocol between the glasses (or simulator) and the server.

Design constraints:

- The device is an ESP32-S3 parsing JSON with ArduinoJson into a small static
  buffer, so field names are short and every message is flat and bounded.
- Transport is plain HTTP request/response, always device-initiated. There is
  no server push: while a session is active the device streams preview frames
  at 2-3 fps and each response carries the latest scene, which bounds anchor
  update latency by the preview rate. WebSocket support can be layered on
  later without changing these schemas.
- All display coordinates are integer pixels in the 128x64 OLED space. The
  server owns the camera->display mapping; the device never sees camera
  coordinates.

A scene fully replaces whatever the device is currently showing ("replace"
semantics). The device redraws only when ``seq`` changes.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

PROTOCOL_VERSION = 1

DISPLAY_W = 128
DISPLAY_H = 64

# 5x7 font at size 1 => 6 px advance per character, 21 chars per line.
CHARS_PER_LINE = DISPLAY_W // 6


class TextEl(BaseModel):
    """A text run. ``x, y`` is the top-left corner; ``size`` is the GFX text
    scale (1 => 6x8 px cell, 2 => 12x16)."""

    t: Literal["text"] = "text"
    x: int = 0
    y: int = 0
    size: int = 1
    text: str


class ArrowEl(BaseModel):
    """An arrow whose tip is at ``x, y`` pointing in direction ``angle``.

    ``angle`` is in degrees, screen convention: 0 points right (+x), 90 points
    down (+y). The shaft extends ``length`` px backward from the tip.
    """

    t: Literal["arrow"] = "arrow"
    x: int
    y: int
    angle: int = 225
    length: int = 14


class ChevronEl(BaseModel):
    """A compass indicator for a target outside the display window: a double
    chevron pointing along ``edge`` with an optional short label next to it.

    ``x, y`` position the chevron tip; -1 (the default) lets the device place
    it at the physical panel edge. The server sets explicit coordinates when
    the optics crop the panel and the visible area is inset."""

    t: Literal["chevron"] = "chevron"
    edge: Literal["left", "right", "up", "down"]
    label: str = ""
    x: int = -1
    y: int = -1


class TargetEl(BaseModel):
    """Object highlight: four corner brackets framing a square of half-size
    ``r`` around ``x, y`` - a game-style target marker."""

    t: Literal["target"] = "target"
    x: int
    y: int
    r: int = 12


DisplayElement = Annotated[TextEl | ArrowEl | ChevronEl | TargetEl, Field(discriminator="t")]


class DisplayScene(BaseModel):
    """Full frame content. Replaces the previous scene atomically."""

    v: int = PROTOCOL_VERSION
    seq: int = 0
    els: list[DisplayElement] = Field(default_factory=list)


# --- HTTP response bodies (device-initiated request/response only) ---


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    protocol: int = PROTOCOL_VERSION


class PhotoAck(BaseModel):
    """Response to ``POST /api/v1/photos`` (raw JPEG body). ``count`` is the
    number of photos currently buffered; ``scene`` shows the capture counter."""

    count: int
    scene: DisplayScene | None = None


class AskResponse(BaseModel):
    """Response to ``POST /api/v1/ask`` (raw WAV body). Blocking: the server
    runs the full pipeline and returns the first slide (or an error scene).

    ``active`` tells the device whether a slide session is now running - while
    true it streams preview frames at 2-3 fps for relocalization."""

    session_id: str
    scene: DisplayScene
    active: bool = True


class EventRequest(BaseModel):
    """Body of ``POST /api/v1/event`` - button gestures beyond capture/ask.

    ``reset`` (double click) ends the chat: session, history, and photo
    buffer are cleared. The others arrive as voice commands."""

    type: Literal["next", "prev", "cancel", "repeat", "reset"]


class SceneResponse(BaseModel):
    """Response to ``POST /api/v1/preview`` and ``POST /api/v1/event`` (and
    ``GET /api/v1/scene``). ``scene`` is null when nothing changed since the
    ``last_seq`` the device sent - the device skips the redraw. ``active``
    false tells the device the session ended: stop streaming previews."""

    scene: DisplayScene | None = None
    active: bool = True
