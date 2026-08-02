# Camera-to-display calibration

The camera sits a few centimeters away from your eye, so what the display
center points at and what the camera center sees are different directions -
and the difference depends on distance (parallax). Calibration measures that
mapping with three aligned points and stores it in the config TOML. Without
it, highlight markers land off-target.

## Setup

1. Generate the target once: `uv run lstk-eye target` -> `calibration_target.png`.
2. Open it **fullscreen on a laptop or tablet** (a phone works from close
   range). The white border around the marker is required - do not crop it.
3. Sit at your **normal working distance** (50-70 cm for desk work). The
   correction is distance-dependent: calibrate at the distance you will
   actually use. Recalibrate if you switch to a very different range.
4. Normal room lighting. Avoid heavy glare on the screen showing the marker.

The marker must appear at least ~6% of the camera frame wide to be accepted;
a fullscreen laptop marker satisfies this from ~40 cm out to beyond 1.5 m.

## Procedure

1. Hold the button, say **"calibrate"** (or «калибровка»), release.
2. The HUD shows a small bracket crosshair, `point 1/3 - click`, and a live
   camera-feedback line that updates ~2 times per second.
3. Move your **head** until the physical marker (seen through the glass)
   sits under the crosshair **and** the feedback line reads `marker OK - click`,
   then single-click. The crosshair jumps to the next point; repeat for 2/3
   (right) and 3/3 (top).
4. `calibrated / saved` means the mapping is applied immediately and written
   to `lstk-eye.toml` for every future run.

## The feedback line

| Line | Meaning | What to do |
|---|---|---|
| `marker OK - click` | Camera sees the marker well | Align crosshair, click |
| `no marker seen` | Marker not in the camera frame | Turn your head toward the marker; remember the camera looks from the temple, not from your eye |
| `closer to marker` | Marker too small in frame | Move closer or enlarge it on screen |
| `marker near edge` | Marker about to leave the frame | Re-center your head on the marker |
| `bad photo - again` | Corrupt capture | Just click again |
| `odd data - redo 1/3` | The three points did not form a sane solution (e.g. head moved between align and click) | The flow restarted; redo all three points |

A click is only counted when the marker check passes - clicking on a red
line does nothing except keep you on the same point.

## Debugging

Every calibration click is saved under `runs/calibration/`:
`<time>_p<point>_<ok|miss>.jpg` is the raw capture and `..._detect.jpg` has
the detection overlay (green quad + red cross, or a NOT FOUND banner). If
you keep getting `no marker seen`, open the latest `_miss_detect.jpg` and
look at what the camera actually framed.

Exit any time with a double click. Recalibrate whenever the camera, display,
or optics are remounted - the procedure takes about 20 seconds.
