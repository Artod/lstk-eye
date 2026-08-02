"""SceneComposer layout: wrapping, padding, status line, arrows, targets,
chevrons, fallbacks.

Display expectations are hand-computed against the default calibration
(center 0.5/0.5, window 0.4/0.4): dx = (x-0.5)/0.4*128+64,
dy = (y-0.5)/0.4*64+32 - and the default visible-area pads (pad_x=16,
pad_y=2): x in [16, 111], y in [2, 61], 16 chars per line, status row y=54,
visible center (63, 31), arrow bands x [18, 109] / y [20, 51].
"""

import pytest

from lstk_eye.calibration import WindowCalibration
from lstk_eye.config import CalibrationConfig, DisplayConfig
from lstk_eye.display import SceneComposer, wrap_text
from lstk_eye.pipeline.types import Slide
from lstk_eye.protocol.messages import ArrowEl, ChevronEl, DisplayScene, TargetEl, TextEl


@pytest.fixture
def composer() -> SceneComposer:
    return SceneComposer(WindowCalibration(CalibrationConfig()), DisplayConfig())


def texts(scene):
    return [e for e in scene.els if isinstance(e, TextEl)]


def arrows(scene):
    return [e for e in scene.els if isinstance(e, ArrowEl)]


def chevrons(scene):
    return [e for e in scene.els if isinstance(e, ChevronEl)]


def targets(scene):
    return [e for e in scene.els if isinstance(e, TargetEl)]


def make_slide(label="red probe here", anchor=(0.6, 0.6), index=1, total=3, size=None):
    return Slide(index=index, total=total, label=label, anchor=anchor, size=size)


# --- wrap_text ---


def test_wrap_short_text_single_line():
    assert wrap_text("red probe here", 16) == ["red probe here"]


def test_wrap_two_lines_at_word_boundary():
    assert wrap_text("set dial to V 20 then read", 16) == [
        "set dial to V 20",  # exactly 16 chars
        "then read",
    ]


def test_wrap_truncates_overflow_with_dots():
    lines = wrap_text("press and hold the red button until the screen shows zero", 16)
    assert len(lines) == 2
    assert lines[-1].endswith("..")
    assert all(len(line) <= 16 for line in lines)


def test_wrap_hard_splits_long_word():
    assert wrap_text("x" * 30, 16) == ["x" * 16, "x" * 14]


def test_wrap_empty():
    assert wrap_text("", 16) == []


# --- every method returns a valid scene with the right seq ---


def test_all_methods_carry_seq(composer):
    scenes = [
        composer.slide(make_slide(), seq=11, anchored=True, anchor=(0.6, 0.6)),
        composer.photo_count(2, seq=12),
        composer.thinking(seq=13),
        composer.status("session over", seq=14),
        composer.error("no plan", seq=15),
        composer.blank(seq=16),
    ]
    for scene, seq in zip(scenes, range(11, 17), strict=True):
        assert isinstance(scene, DisplayScene)
        assert scene.seq == seq
        # Round-trip through the wire format must survive for every scene.
        assert DisplayScene.model_validate_json(scene.model_dump_json()) == scene


# --- padding invariant: nothing renders outside the visible area ---


@pytest.mark.parametrize(
    "anchor",
    [(0.5, 0.5), (0.6, 0.6), (0.3, 0.4), (0.0, 0.5), (1.0, 1.0), None],
)
def test_everything_inside_visible_area(composer, anchor):
    slide = make_slide(anchor=anchor, size=(0.3, 0.3))
    for style in ("arrow", "target"):
        scene = composer.slide(slide, seq=1, anchored=True, anchor=anchor, style=style)
        for e in texts(scene):
            assert e.x >= composer.x0
            assert e.y >= composer.y0
            assert e.y + 8 * e.size - 1 <= composer.y1
        for e in arrows(scene):
            assert composer.x0 <= e.x <= composer.x1
            assert composer.y0 <= e.y <= composer.y1
        for e in chevrons(scene):
            assert composer.x0 <= e.x <= composer.x1
            assert composer.y0 <= e.y <= composer.y1
        for e in targets(scene):
            assert e.x - e.r >= composer.x0 - 3  # bracket may poke 3px on min-r floor
            assert e.x + e.r <= composer.x1 + 3


# --- slide layout ---


def test_slide_label_lines_positions(composer):
    scene = composer.slide(
        make_slide(label="set dial to V 20 then read"),
        seq=1,
        anchored=True,
        anchor=None,
    )
    lines = [e for e in texts(scene) if e.y in (2, 11)]
    assert [(e.x, e.y, e.text) for e in lines] == [
        (16, 2, "set dial to V 20"),
        (16, 11, "then read"),
    ]


@pytest.mark.parametrize(
    ("index", "total", "text", "x"),
    [
        (2, 5, "3/5", 112 - 3 * 6),  # 94
        (9, 12, "10/12", 112 - 5 * 6),  # 82
    ],
)
def test_slide_counter_right_aligned(composer, index, total, text, x):
    scene = composer.slide(
        make_slide(index=index, total=total), seq=1, anchored=True, anchor=None
    )
    counter = [e for e in texts(scene) if e.y == 54 and "/" in e.text]
    assert len(counter) == 1
    assert counter[0].text == text
    assert counter[0].x == x


def test_single_slide_hides_counter(composer):
    scene = composer.slide(
        make_slide(index=0, total=1), seq=1, anchored=True, anchor=(0.6, 0.6)
    )
    assert not any("/" in e.text for e in texts(scene))


@pytest.mark.parametrize(
    ("anchor", "tip", "angle"),
    [
        ((0.6, 0.6), (96, 48), 45),  # down-right: d=(33,17) -> 27.3 -> 45
        ((0.42, 0.42), (38, 19), 225),  # up-left (1-line label frees the row)
        ((0.55, 0.42), (80, 19), 315),  # up-right: d=(17,-12) -> -35.2 -> 315
        ((0.45, 0.58), (48, 45), 135),  # down-left: d=(-15,14) -> 137 -> 135
        ((0.6, 0.5), (96, 32), 0),  # due right: d=(33,1) -> 1.7 -> 0
        ((0.5, 0.6), (64, 48), 90),  # due down: d=(1,17) -> 86.6 -> 90
    ],
)
def test_slide_arrow_quadrants(composer, anchor, tip, angle):
    scene = composer.slide(make_slide(anchor=anchor), seq=1, anchored=True, anchor=anchor)
    (arrow,) = arrows(scene)
    assert (arrow.x, arrow.y) == tip
    assert arrow.angle == angle
    assert arrow.length == 12
    assert not chevrons(scene)


@pytest.mark.parametrize(
    ("anchor", "tip", "angle"),
    [
        # Raw (110, 58): visible, outside the arrow band -> clamped (109, 51).
        ((0.64375, 0.6625), (109, 51), 45),
        # Raw (17, 10): visible, clamped to (18, 14) - the one-line label
        # occupies rows up to y=11, and the arrow clears it by 3 px.
        ((0.353125, 0.3625), (18, 14), 180),
    ],
)
def test_slide_arrow_clamped_at_borders(composer, anchor, tip, angle):
    scene = composer.slide(make_slide(anchor=anchor), seq=1, anchored=True, anchor=anchor)
    (arrow,) = arrows(scene)
    assert (arrow.x, arrow.y) == tip
    assert arrow.angle == angle


@pytest.mark.parametrize(
    ("anchor", "edge"),
    [
        ((0.0, 0.5), "left"),
        ((1.0, 0.5), "right"),
        ((0.5, 0.0), "up"),
        ((0.5, 1.0), "down"),
    ],
)
def test_slide_chevron_out_of_window(composer, anchor, edge):
    scene = composer.slide(make_slide(anchor=anchor), seq=1, anchored=True, anchor=anchor)
    assert not arrows(scene)
    (chevron,) = chevrons(scene)
    assert chevron.edge == edge
    # Compass chevrons carry an explicit inset position.
    assert composer.x0 <= chevron.x <= composer.x1
    assert composer.y0 <= chevron.y <= composer.y1


def test_slide_look_back_when_not_anchored(composer):
    scene = composer.slide(make_slide(), seq=1, anchored=False, anchor=None)
    hints = [e for e in texts(scene) if e.text == "look back"]
    assert len(hints) == 1
    assert (hints[0].x, hints[0].y) == (16, 54)
    assert not arrows(scene) and not chevrons(scene)


def test_slide_not_anchored_suppresses_arrow_even_with_position(composer):
    # A live position with anchored=False (still below the appear threshold)
    # must not draw an arrow.
    scene = composer.slide(make_slide(), seq=1, anchored=False, anchor=(0.6, 0.6))
    assert not arrows(scene) and not chevrons(scene)
    assert any(e.text == "look back" for e in texts(scene))


def test_slide_anchored_hides_look_back(composer):
    scene = composer.slide(make_slide(), seq=1, anchored=True, anchor=(0.6, 0.6))
    assert not any(e.text == "look back" for e in texts(scene))


def test_slide_text_only_no_arrow_no_look_back(composer):
    scene = composer.slide(
        make_slide(label="wait 30 seconds", anchor=None), seq=1, anchored=False, anchor=None
    )
    assert not arrows(scene) and not chevrons(scene)
    assert not any(e.text == "look back" for e in texts(scene))


# --- target style (find-this sessions) ---


def test_target_style_renders_brackets(composer):
    slide = make_slide(label="phone", anchor=(0.5, 0.5), index=0, total=1, size=(0.2, 0.1))
    scene = composer.slide(slide, seq=1, anchored=True, anchor=(0.5, 0.5), style="target")
    assert not arrows(scene)
    (target,) = targets(scene)
    # Raw center (64, 32); r from the bbox half-extent (32 px) capped at 22,
    # then shrunk to clear the one-line label (rows up to y=11): 32-11-1 = 20.
    assert (target.x, target.y) == (64, 32)
    assert target.r == 20


def test_target_shrinks_near_border(composer):
    # Anchor mapping near the visible border: r gives way, center stays put.
    anchor = (0.36, 0.5)  # raw x = 19.2 -> 19
    slide = make_slide(label="phone", anchor=anchor, index=0, total=1, size=(0.3, 0.3))
    scene = composer.slide(slide, seq=1, anchored=True, anchor=anchor, style="target")
    (target,) = targets(scene)
    assert target.x - target.r >= composer.x0 - 3
    assert target.r >= 3


def test_target_out_of_window_uses_compass(composer):
    slide = make_slide(label="phone", anchor=(1.0, 0.5), index=0, total=1, size=(0.2, 0.2))
    scene = composer.slide(slide, seq=1, anchored=True, anchor=(1.0, 0.5), style="target")
    assert not targets(scene)
    (chevron,) = chevrons(scene)
    assert chevron.edge == "right"


def test_target_without_size_uses_min_radius(composer):
    slide = make_slide(label="phone", anchor=(0.5, 0.5), index=0, total=1, size=None)
    scene = composer.slide(slide, seq=1, anchored=True, anchor=(0.5, 0.5), style="target")
    (target,) = targets(scene)
    assert target.r == 6


# --- other scenes ---


def test_photo_count_layout(composer):
    scene = composer.photo_count(3, seq=5)
    big = [e for e in texts(scene) if e.size == 2]
    assert len(big) == 1
    assert big[0].text == "[3]"
    assert big[0].x == 16 + (96 - 3 * 12) // 2  # 46: centered in the visible area
    sub = [e for e in texts(scene) if e.text == "photo saved"]
    assert len(sub) == 1
    assert sub[0].x == 16 + (96 - 11 * 6) // 2  # 31
    assert sub[0].y > big[0].y


def test_thinking_centered(composer):
    scene = composer.thinking(seq=2)
    (el,) = texts(scene)
    assert el.text == "thinking..."
    assert el.x == 16 + (96 - 11 * 6) // 2  # 31


def test_status_with_and_without_sub(composer):
    scene = composer.status("step 2/5", seq=3, sub="look back")
    els = texts(scene)
    assert [e.text for e in els] == ["step 2/5", "look back"]
    assert els[0].x == 16 + (96 - 8 * 6) // 2  # 40
    assert els[1].x == 16 + (96 - 9 * 6) // 2  # 37
    assert els[0].y < els[1].y
    assert len(texts(composer.status("done", seq=4))) == 1


def test_error_prefix_and_wrap(composer):
    scene = composer.error("camera timeout", seq=6)
    lines = texts(scene)
    assert lines[0].text.startswith("! ")
    long = composer.error(
        "planner returned marks outside the segmented set, session aborted, retry the question",
        seq=7,
    )
    lines = texts(long)
    assert lines[0].text.startswith("! ")
    assert 1 < len(lines) <= 4
    assert all(len(e.text) <= composer.chars_per_line for e in lines)
    assert lines[-1].text.endswith("..")


def test_blank_empty(composer):
    scene = composer.blank(seq=9)
    assert scene.els == []
