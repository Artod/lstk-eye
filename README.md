# lstk-eye

**An AI laser pointer for the real world.** DIY glasses attachment: a tiny camera and a
see-through HUD, working as a thin client to your laptop. Press a button, ask a question
by voice — the answer comes back as step-by-step slides with arrows anchored to the
physical objects you're looking at.

Canonical use case: look at a multimeter, ask "how do I check battery voltage" →
three slides: arrow at the VΩmA jack ("red probe here"), arrow at COM ("black probe
here"), arrow at the dial ("set V⎓ 20"). Double-click to advance as you complete
each step.

> Status: early development. Software runs end-to-end against a device simulator;
> hardware bring-up is in progress.

## How it works

```
 glasses (ESP32-S3)                       laptop (Python server)
┌─────────────────────┐   JPEG + WAV    ┌──────────────────────────────┐
│ camera ── button ── │ ──── HTTP ────▶ │ Whisper STT                  │
│ mic                 │                 │ SAM segmentation             │
│                     │                 │ Set-of-Marks overlay         │
│ 128x64 OLED ◀────── │ ◀─── scenes ─── │ Claude vision → step plan    │
│ (beamsplitter HUD)  │   (JSON draw    │ calibration → display coords │
└─────────────────────┘    primitives)  │ relocalization (2-3 fps)     │
                                        └──────────────────────────────┘
```

The glasses are a dumb terminal: camera in, pixels out, zero intelligence onboard.
The server owns all state and thinking:

1. **Capture** — single click buffers a photo; long press records voice; on release
   everything flies to the server as one request.
2. **Segment** — SAM-style segmentation finds every object region; the server overlays
   numbered marks on the photo (*Set-of-Marks*).
3. **Plan** — Claude (vision) gets the marked photo + transcribed question and returns a
   structured plan that references objects **only by mark number** — never raw
   coordinates, which kills VLM coordinate hallucination.
4. **Anchor** — mark numbers resolve to mask centroids, map through a one-time
   camera↔display calibration, and render as arrows + text on the OLED.
5. **Relocalize** — while a session is active the glasses stream small preview frames;
   the server re-finds the current step's target (template matching) so arrows follow
   the object at 1–2 Hz. Look away and arrows hide; look back and they return in ~0.5 s.
   Targets outside the display window get edge chevrons ("further right →").

Latency honesty: plan generation takes 3–6 s once per request. Slide advance is instant
(steps are precomputed). Anchor refresh is 1–2 Hz by the physics of the pipeline; text
rendering never degrades.

## Hardware

Full build guide: [docs/HARDWARE.md](docs/HARDWARE.md). Bill of materials (~$40 on top
of your own glasses):

| Part | Notes |
|---|---|
| Seeed XIAO ESP32S3 Sense | the only computer and radio on the device |
| OV5640 5 MP autofocus camera | swapped into the Sense camera socket |
| SSD1306 0.96" OLED, 128×64, I2C | the image source for the HUD |
| 25 mm dia / 45 mm FL biconvex lens | collimates the OLED (Google Cardboard spec) |
| 50/50 beamsplitter, ~30×25 mm | folds the image into your eye |
| LiPo 502535 450 mAh + slide switch | ~1.5 h streaming, much more at normal duty cycle |
| Panel-mount push button | single/double/long-press gestures |
| MPU6050 IMU (optional, planned) | gyro smoothing between anchor updates |

Everything mounts on the left temple of regular glasses. Hot glue is the official
assembly method — dielectric, reversible with isopropyl.

## Getting started (software only, no hardware needed)

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or plain pip).

```bash
git clone https://github.com/lstk-eye/lstk-eye
cd lstk-eye
uv sync
```

Run the server in **mock mode** (no ML models, no API key) and drive it with the
built-in device simulator:

```bash
uv run lstk-eye serve --profile mock
```

```bash
uv run lstk-eye simulate --ask "how do I check battery voltage"
```

The simulator plays the role of the glasses: it "captures" a photo, sends the question,
renders the returned scenes as PNG frames, and advances slides — the full nervous
system without a soldering iron.

For the real pipeline install the ML extras and provide an Anthropic API key:

```bash
uv sync --extra full
export ANTHROPIC_API_KEY=sk-ant-...
uv run lstk-eye serve
```

Check your environment anytime with:

```bash
uv run lstk-eye doctor
```

## Firmware

Arduino sketch for the XIAO ESP32S3 Sense lives in
[firmware/lstk_eye_glasses](firmware/lstk_eye_glasses/). Board setup, library list, and
flashing instructions: [firmware/README.md](firmware/README.md).

## Project layout

```
src/lstk_eye/          server package
  protocol/            device<->server wire schemas (pydantic)
  pipeline/            STT, segmentation, planner, relocalization (all pluggable)
  simulator/           software stand-in for the glasses
firmware/              Arduino sketch for the glasses
docs/                  architecture, protocol, hardware guide
tests/                 pytest suite (runs fully offline on mocks)
```

Every pipeline stage is an interface with at least two implementations (real + mock);
adding a backend (different STT, a new segmenter, another LLM) means implementing one
class and registering it in `pipeline/factories.py`. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Contributing

Issues and PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The test suite runs
offline in seconds (`uv run pytest -m "not slow"`); CI is just ruff + pytest.

## License

[MIT](LICENSE)
