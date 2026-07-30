# Device protocol

Wire protocol between the glasses (or the simulator) and the server. Schemas live in
`src/lstk_eye/protocol/messages.py`; this document describes protocol version **1**
(`PROTOCOL_VERSION`).

Transport is plain HTTP, always device-initiated request/response - there is no server
push. The server listens on port 8321 by default and advertises `_lstk-eye._tcp` via
mDNS (`server.zeroconf`) so the device can find the laptop without a hardcoded IP.

Design constraints baked into the schemas:

- The device parses JSON with ArduinoJson into a small static buffer: field names are
  short, every message is flat and bounded.
- All display coordinates are integer pixels in the 128x64 OLED space
  (`DISPLAY_W` x `DISPLAY_H`). The server owns the camera-to-display mapping; the
  device never sees camera coordinates.

## Endpoints

All endpoints are under `/api/v1`. `device_id` is a query parameter defaulting to
`"glasses"`.

| Method | Path | Request body | Response model |
|---|---|---|---|
| GET | `/api/v1/health` | - | `HealthResponse` |
| POST | `/api/v1/photos?device_id=glasses` | raw JPEG (`image/jpeg`) | `PhotoAck` |
| POST | `/api/v1/ask?device_id=glasses[&text=...]` | raw WAV (16-bit PCM mono) | `AskResponse` |
| POST | `/api/v1/event?device_id=glasses` | JSON `EventRequest` | `SceneResponse` |
| POST | `/api/v1/preview?device_id=glasses&last_seq=N` | raw JPEG (QVGA) | `SceneResponse` |
| GET | `/api/v1/scene?device_id=glasses&last_seq=N` | - | `SceneResponse` |

### GET /api/v1/health

```json
{"status": "ok", "version": "0.1.0", "protocol": 1}
```

### POST /api/v1/photos

Buffers one capture frame. `count` is the number of photos currently buffered for the
next ask; `scene` shows the capture counter on the HUD.

```json
{
  "count": 2,
  "scene": {
    "v": 1,
    "seq": 3,
    "els": [{"t": "text", "x": 0, "y": 0, "size": 1, "text": "photo 2 buffered"}]
  }
}
```

### POST /api/v1/ask

Body is the recorded WAV. Blocking: the server runs the full pipeline (STT ->
segmentation -> Set-of-Marks -> planner) and returns the first slide, or an error
scene. The body may be empty when the `text=` query parameter is given and the server
has `debug.allow_text_ask` enabled (forced on by the mock profile) - this is how the
simulator and tests ask without a microphone.

```json
{
  "session_id": "a1b2",
  "scene": {
    "v": 1,
    "seq": 1,
    "els": [
      {"t": "text", "x": 0, "y": 0, "size": 1, "text": "Red probe here"},
      {"t": "arrow", "x": 88, "y": 37, "angle": 225, "length": 14}
    ]
  },
  "active": true
}
```

`active: true` means a slide session is now running: the device starts streaming
preview frames at 2-3 fps.

### POST /api/v1/event

Button gestures beyond capture/ask. Request body:

```json
{"type": "next"}
```

`type` is one of `"next"`, `"prev"`, `"cancel"`, `"repeat"`.

### POST /api/v1/preview and GET /api/v1/scene

`/preview` uploads a small frame for relocalization; `/scene` just polls. Both take
`last_seq` (the `seq` of the scene the device currently shows) and return a
`SceneResponse`:

```json
{"scene": null, "active": true}
```

`scene` is `null` when nothing changed since `last_seq` - the device skips the redraw.
When the anchor moved or the step changed, `scene` carries the full new frame content.

## Scene semantics

- A `DisplayScene` fully **replaces** whatever the device is showing. There are no
  partial updates.
- The device redraws only when `seq` changes; `seq` increases with every server-side
  scene change.
- `active: false` in any response means the session ended: the device stops streaming
  previews and returns to idle.

## Display elements

A scene is `{"v": 1, "seq": N, "els": [...]}` where each element is discriminated by
`t`:

**text** - a text run. `x, y` is the top-left corner in pixels; `size` is the GFX
text scale (1 = 6x8 px per character, 2 = 12x16). At size 1 the panel fits 21
characters per line (`CHARS_PER_LINE`).

```json
{"t": "text", "x": 0, "y": 0, "size": 1, "text": "step 2/5"}
```

**arrow** - tip at `x, y`, pointing in direction `angle`. `angle` is in degrees,
screen convention: 0 points right (+x), 90 points down (+y). The shaft extends
`length` px backward from the tip.

```json
{"t": "arrow", "x": 88, "y": 37, "angle": 225, "length": 14}
```

**chevron** - edge indicator for a target outside the display window: a double
chevron hugging `edge` (`"left"`, `"right"`, `"up"`, `"down"`) with an optional short
label next to it.

```json
{"t": "chevron", "edge": "right", "label": "further right"}
```

## Device gesture contract

| Gesture | Device action |
|---|---|
| single click | capture photo -> `POST /photos` |
| long press | record mic; on release -> `POST /ask` with the WAV |
| double click | `POST /event {"type": "next"}` |
| (session active) | stream `POST /preview` at 2-3 fps until `active: false` |

Everything else - back, repeat, cancel - is voice, not clicks: rare commands are
spoken, frequent ones are clicked.
