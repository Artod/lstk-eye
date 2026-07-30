# Development

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/lstk-eye/lstk-eye
cd lstk-eye
uv sync
```

`uv sync` installs the core package plus the dev group (pytest, ruff). The ML
backends are optional extras, kept out of the default install on purpose:

```bash
uv sync --extra stt    # faster-whisper
uv sync --extra seg    # ultralytics (FastSAM)
uv sync --extra full   # both
```

The real planner needs an Anthropic API key: `export ANTHROPIC_API_KEY=sk-ant-...`.

## Running tests

```bash
uv run pytest -m "not slow" -q   # offline suite: mocks only, no models, no keys
uv run pytest -m slow            # real ML backends; needs --extra full, downloads models
uv run ruff check src tests      # lint
```

The offline suite is the CI gate and must stay green with zero optional dependencies
installed. Tests touching heavy backends use `pytest.importorskip` and are marked
`@pytest.mark.slow`.

## Mock profile and the simulator loop

The whole system runs with no ML models and no API key:

```bash
uv run lstk-eye serve --profile mock
```

`--profile mock` swaps every pipeline stage for its mock (see
`AppConfig.resolved()`) and enables `debug.allow_text_ask`, so
`POST /api/v1/ask?text=...` works with an empty body - no microphone needed.

In a second terminal, drive the server with the device simulator:

```bash
uv run lstk-eye simulate --ask "how do I check battery voltage"
```

The simulator plays the glasses: it sends a capture, asks the question, renders the
returned scenes, and advances slides - the full request loop without hardware.
`uv run lstk-eye doctor` checks which optional dependencies and keys are available.

## Configuration reference

Precedence (highest first): CLI flags > environment variables > TOML file. Environment
variables use the `LSTK_` prefix with `__` as the nesting delimiter. TOML is looked up
at `./lstk-eye.toml`, then `~/.config/lstk-eye/config.toml`, unless a path is given
explicitly.

| Field | Env var | Default | Meaning |
|---|---|---|---|
| `profile` | `LSTK_PROFILE` | `real` | `mock` swaps every backend for its mock and enables text asks |
| `server.host` | `LSTK_SERVER__HOST` | `0.0.0.0` | bind address |
| `server.port` | `LSTK_SERVER__PORT` | `8321` | bind port |
| `server.zeroconf` | `LSTK_SERVER__ZEROCONF` | `true` | advertise `_lstk-eye._tcp` via mDNS |
| `stt.backend` | `LSTK_STT__BACKEND` | `whisper` | `whisper` or `mock` |
| `stt.model` | `LSTK_STT__MODEL` | `small` | faster-whisper model size |
| `stt.language` | `LSTK_STT__LANGUAGE` | `null` | `null` = autodetect |
| `stt.device` | `LSTK_STT__DEVICE` | `auto` | faster-whisper device |
| `stt.compute_type` | `LSTK_STT__COMPUTE_TYPE` | `int8` | faster-whisper compute type |
| `segmenter.backend` | `LSTK_SEGMENTER__BACKEND` | `fastsam` | `fastsam` or `mock` |
| `segmenter.model` | `LSTK_SEGMENTER__MODEL` | `FastSAM-s.pt` | auto-downloaded by ultralytics |
| `segmenter.device` | `LSTK_SEGMENTER__DEVICE` | `cpu` | inference device |
| `segmenter.max_masks` | `LSTK_SEGMENTER__MAX_MASKS` | `25` | cap on marks per frame |
| `segmenter.min_area` | `LSTK_SEGMENTER__MIN_AREA` | `0.0008` | min mask area, fraction of frame |
| `segmenter.conf` | `LSTK_SEGMENTER__CONF` | `0.4` | detection confidence threshold |
| `planner.backend` | `LSTK_PLANNER__BACKEND` | `anthropic` | `anthropic`, `claude-cli`, or `mock` |
| `planner.model` | `LSTK_PLANNER__MODEL` | `claude-opus-5` | model id |
| `planner.effort` | `LSTK_PLANNER__EFFORT` | `medium` | `low`/`medium`/`high`/`xhigh`/`max` |
| `planner.max_tokens` | `LSTK_PLANNER__MAX_TOKENS` | `8000` | response token cap |
| `planner.max_steps` | `LSTK_PLANNER__MAX_STEPS` | `7` | max slides per plan |
| `planner.timeout_s` | `LSTK_PLANNER__TIMEOUT_S` | `60.0` | API timeout |
| `calibration.center_x` | `LSTK_CALIBRATION__CENTER_X` | `0.5` | camera point shown at display center |
| `calibration.center_y` | `LSTK_CALIBRATION__CENTER_Y` | `0.5` | camera point shown at display center |
| `calibration.window_w` | `LSTK_CALIBRATION__WINDOW_W` | `0.40` | fraction of frame width the panel spans |
| `calibration.window_h` | `LSTK_CALIBRATION__WINDOW_H` | `0.40` | fraction of frame height the panel spans |
| `reloc.appear_conf` | `LSTK_RELOC__APPEAR_CONF` | `0.60` | match confidence to show arrows |
| `reloc.disappear_conf` | `LSTK_RELOC__DISAPPEAR_CONF` | `0.45` | confidence below which a frame counts as a miss |
| `reloc.miss_hide` | `LSTK_RELOC__MISS_HIDE` | `4` | consecutive misses before arrows hide |
| `reloc.ema` | `LSTK_RELOC__EMA` | `0.4` | anchor smoothing factor, 1.0 = no smoothing |
| `reloc.scales` | `LSTK_RELOC__SCALES` | `[0.85, 1.0, 1.18]` | template match scale pyramid |
| `storage.dir` | `LSTK_STORAGE__DIR` | `runs` | session log directory |
| `storage.save_sessions` | `LSTK_STORAGE__SAVE_SESSIONS` | `true` | persist session artifacts |
| `debug.allow_text_ask` | `LSTK_DEBUG__ALLOW_TEXT_ASK` | `false` | allow `POST /ask?text=...` to bypass STT; forced on by the mock profile |

`ANTHROPIC_API_KEY` is read by the Anthropic client itself, not by this config.

Calibration is measured once per build at working distance (~60 cm): the display shows
a window into the camera image centered at `(center_x, center_y)` spanning
`(window_w, window_h)` of the frame.

## Adding a backend

Every pipeline stage is an ABC in `src/lstk_eye/pipeline/interfaces.py`:
`SpeechToText`, `Segmenter`, `Planner`, `RelocalizerFactory`/`TargetTracker`,
`Calibration`. To add one (a new STT engine, another segmenter, a different LLM):

1. Implement the ABC in a new module under `src/lstk_eye/pipeline/<stage>/`. If the
   backend needs a heavy optional dependency, import it at the top of that module and
   nowhere else - the factory catches the `ImportError` and raises `DependencyError`
   with the exact install hint.
2. Add the backend name to the stage's `backend` Literal in `src/lstk_eye/config.py`,
   plus any new knobs the backend needs.
3. Add a branch in the matching `create_*` function in
   `src/lstk_eye/pipeline/factories.py`.
4. Add tests. Offline behavior (config wiring, error paths) goes in the normal suite;
   anything importing the heavy dependency uses `pytest.importorskip` and
   `@pytest.mark.slow`.

Contract notes worth keeping: segmenters return masks sorted by `mark_id` starting at
1, already filtered per config; planners must raise `PlanningError` rather than return
out-of-range marks; trackers own their hysteresis (debounced `found`, appear fast,
disappear lazily); all pipeline coordinates are normalized floats in [0, 1] - display
pixels exist only inside `Calibration`.

## Repo layout

```
src/lstk_eye/          server package
  protocol/            device<->server wire schemas (pydantic)
  pipeline/            interfaces, types, factories, and stage implementations
  simulator/           software stand-in for the glasses
firmware/              Arduino sketch for the glasses
docs/                  architecture, protocol, hardware guide
tests/                 pytest suite (offline on mocks; slow marks for real models)
```
