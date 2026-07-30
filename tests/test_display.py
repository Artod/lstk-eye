"""SceneComposer layout: wrapping, status line, arrows, chevrons, fallbacks.

All display expectations are hand-computed against the default calibration
(center 0.5/0.5, window 0.4/0.4): dx = (x-0.5)/0.4*128+64,
dy = (y-0.5)/0.4*64+32.
"""

import pytest

from lstk_eye.calibration import WindowCalibration
from lstk_eye.config import CalibrationConfig
from lstk_eye.display import SceneComposer, wrap_text
from lstk_eye.pipeline.types import Slide
from lstk_eye.protocol.messages import ArrowEl, ChevronEl, DisplayScene, TextEl


@pytest.fixture
def composer() -> SceneComposer:
    return SceneComposer(WindowCalibration(CalibrationConfig()))


def texts(scene):
    return [e for e in scene.els if isinstance(e, TextEl)]


def arrows(scene):
    return [e for e in scene.els if isinstance(e, ArrowEl)]


def chevrons(scene):
    return [e for e in scene.els if isinstance(e, ChevronEl)]


def make_slide(label="red probe here", anchor=(0.6, 0.6), index=1, total=3):
    return Slide(index=index, total=total, label=label, anchor=anchor)


# --- wrap_text ---


def test_wrap_short_text_single_line():
    assert wrap_text("red probe here") == ["red probe here"]


def test_wrap_two_lines_at_word_boundary():
    assert wrap_text("set dial to V 20 then read the value") == [
        "set dial to V 20 then",  # exactly 21 chars
        "read the value",
    ]


def test_wrap_truncates_overflow_with_dots():
    lines = wrap_text("press and hold the red button until the screen shows zero")
    assert lines == ["press and hold the", "red button until th.."]
    assert all(len(line) <= 21 for line in lines)


def test_wrap_hard_splits_long_word():
    lines = wrap_text("x" * 30)
    assert lines == ["x" * 21, "x" * 9]


def test_wrap_empty():
    assert wrap_text("") == []


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


# --- slide layout ---


def test_slide_label_lines_positions(composer):
    scene = composer.slide(
        make_slide(label="set dial to V 20 then read the value"),
        seq=1,
        anchored=True,
        anchor=None,
    )
    lines = [e for e in texts(scene) if e.y in (0, 9)]
    assert [(e.x, e.y, e.text) for e in lines] == [
        (0, 0, "set dial to V 20 then"),
        (0, 9, "read the value"),
    ]


@pytest.mark.parametrize(
    ("index", "total", "text", "x"),
    [
        (2, 5, "3/5", 128 - 3 * 6),  # 110
        (9, 12, "10/12", 128 - 5 * 6),  # 98
    ],
)
def test_slide_counter_right_aligned(composer, index, total, text, x):
    scene = composer.slide(
        make_slide(index=index, total=total), seq=1, anchored=True, anchor=None
    )
    counter = [e for e in texts(scene) if e.y == 56 and "/" in e.text]
    assert len(counter) == 1
    assert counter[0].text == text
    assert counter[0].x == x


@pytest.mark.parametrize(
    ("anchor", "tip", "angle"),
    [
        ((0.6, 0.6), (96, 48), 45),  # down-right quadrant
        ((0.42, 0.42), (38, 19), 225),  # up-left: d=(-26,-13) -> -153.4 -> 225
        ((0.58, 0.42), (90, 19), 315),  # up-right: d=(26,-13) -> -26.6 -> 315
        ((0.42, 0.58), (38, 45), 135),  # down-left: d=(-26,13) -> 153.4 -> 135
        ((0.6, 0.5), (96, 32), 0),  # due right
        ((0.5, 0.6), (64, 48), 90),  # due down
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
        # Raw (126, 60): inside the panel, outside the clamp band on both axes.
        ((0.69375, 0.675), (125, 53), 0),
        # Raw (0, 17) -> clamped (2, 18): d=(-62,-14) -> -167.3 -> 180.
        ((0.3, 0.40625), (2, 18), 180),
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


def test_slide_look_back_when_not_anchored(composer):
    scene = composer.slide(make_slide(), seq=1, anchored=False, anchor=None)
    hints = [e for e in texts(scene) if e.text == "look back"]
    assert len(hints) == 1
    assert (hints[0].x, hints[0].y) == (0, 56)
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


# --- other scenes ---


def test_photo_count_layout(composer):
    scene = composer.photo_count(3, seq=5)
    big = [e for e in texts(scene) if e.size == 2]
    assert len(big) == 1
    assert big[0].text == "[3]"
    assert big[0].x == (128 - 3 * 12) // 2  # 46: centered at size 2
    sub = [e for e in texts(scene) if e.text == "photo saved"]
    assert len(sub) == 1
    assert sub[0].x == (128 - 11 * 6) // 2  # 31
    assert sub[0].y > big[0].y


def test_thinking_centered(composer):
    scene = composer.thinking(seq=2)
    (el,) = texts(scene)
    assert el.text == "thinking..."
    assert el.x == (128 - 11 * 6) // 2  # 31


def test_status_with_and_without_sub(composer):
    scene = composer.status("step 2/5", seq=3, sub="look back at device")
    els = texts(scene)
    assert [e.text for e in els] == ["step 2/5", "look back at device"]
    assert els[0].x == (128 - 8 * 6) // 2  # 40
    assert els[1].x == (128 - 19 * 6) // 2  # 7
    assert els[0].y < els[1].y
    assert len(texts(composer.status("done", seq=4))) == 1


def test_error_prefix_and_wrap(composer):
    scene = composer.error("camera timeout", seq=6)
    (el,) = texts(scene)
    assert el.text == "! camera timeout"
    long = composer.error(
        "planner returned marks outside the segmented set, session aborted, retry the question",
        seq=7,
    )
    lines = texts(long)
    assert lines[0].text.startswith("! ")
    assert 1 < len(lines) <= 4
    assert all(len(e.text) <= 21 for e in lines)
    assert lines[-1].text.endswith("..")


def test_blank_empty(composer):
    scene = composer.blank(seq=9)
    assert scene.els == []
