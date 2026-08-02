# Camera-to-display calibration

The camera sits a few centimeters away from your eye, so what the display
center points at and what the camera sees are different directions - and the
difference depends on distance (parallax). Calibration measures that mapping
with three aligned points and stores it in the config TOML. Without it,
highlight markers land off-target.

## Camera mounting first

If the camera module is mounted sideways, set the rotation once in
`lstk-eye.toml` - every frame is normalized upright at ingestion:

```toml
[camera]
rotation = 270   # degrees CLOCKWISE to rotate incoming frames upright
```

Check with any saved capture under `runs/`: if the world looks rotated,
adjust this until it is upright. Calibration and anchoring are wrong until
this is right.

## Setup

1. Get a marker: `uv run lstk-eye target` writes `calibration_target.png`.
   Any square fiducial from the common ArUco/AprilTag families also works -
   detection accepts them all, any id. **Do not draw anything on or over the
   marker** - the pattern is data, a dot or cursor on top can corrupt it.
2. Show it on a **phone at full width** (~6-7 cm wide) - this is the size the
   HUD brackets are designed around. A larger marker on a laptop works too;
   the live feedback will tell you if it is too big or too small.
3. Place the screen at your **normal working distance** (50-70 cm). The
   correction is distance-dependent: calibrate at the distance you will
   actually use.

## Procedure

1. Hold the button, say **"calibrate"** (or «калибровка»), release.
2. The HUD shows bracket crosshairs, `point 1/3 - click`, and a live
   camera-feedback line updating ~2x per second.
3. Move your **head** until the marker (seen through the glass) sits
   **inside the brackets** and the feedback reads `marker OK - click`, then
   single-click. Repeat for points 2/3 (right) and 3/3 (top).
4. `calibrated / saved`: applied immediately, persisted to `lstk-eye.toml`.

## The feedback line

| Line | Meaning | What to do |
|---|---|---|
| `marker OK - click` | Camera sees the marker well | Align brackets, click |
| `no marker seen` | Marker not detected in the camera frame | Turn your head toward the marker (the camera looks from the temple, not from your eye); if it persists, move closer |
| `closer to marker` | Detected but too small | Move closer or enlarge on screen |
| `further from marker` | Marker fills too much of the frame | Step back or shrink it |
| `marker near edge` | About to leave the camera frame | Re-center your head on the marker |
| `busy: 2click exits` | You asked something mid-calibration | Finish or double-click out |
| `bad photo - again` | Corrupt capture | Click again |
| `calib failed-redo 1` | The three points did not form a sane solution | Flow restarted; redo all three |

A click only counts on `marker OK` - clicking on any other line keeps you on
the same point. Saying "calibrate" again restarts from point 1; double click
exits at any moment.

## Debugging

Every calibration click is saved under `runs/calibration/`:
`<time>_p<point>_<ok|miss>.jpg` (raw, already rotation-normalized) and
`..._detect.jpg` (detection overlay: green quad + red cross, or a NOT FOUND
banner). If clicks keep failing, look at the latest `_miss_detect.jpg` -
it shows exactly what the camera framed.

Recalibrate whenever the camera, display, or optics are remounted, or when
your working distance changes a lot. The procedure takes ~20 seconds.
