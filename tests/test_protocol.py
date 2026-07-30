"""Wire protocol: DisplayScene JSON round-trip, discriminated union, short keys."""

import json

import pytest
from pydantic import ValidationError

from lstk_eye.protocol.messages import (
    PROTOCOL_VERSION,
    ArrowEl,
    ChevronEl,
    DisplayScene,
    TextEl,
)


def test_scene_json_round_trip_preserves_element_types():
    scene = DisplayScene(
        seq=7,
        els=[
            TextEl(x=0, y=9, text="red probe here"),
            ArrowEl(x=96, y=48, angle=45, length=12),
            ChevronEl(edge="right", label="dial"),
        ],
    )
    parsed = DisplayScene.model_validate_json(scene.model_dump_json())
    assert parsed == scene
    assert [type(e) for e in parsed.els] == [TextEl, ArrowEl, ChevronEl]
    assert parsed.v == PROTOCOL_VERSION


def test_unknown_element_type_rejected():
    payload = '{"seq": 1, "els": [{"t": "circle", "x": 3, "y": 4}]}'
    with pytest.raises(ValidationError):
        DisplayScene.model_validate_json(payload)


def test_missing_discriminator_rejected():
    with pytest.raises(ValidationError):
        DisplayScene.model_validate_json('{"seq": 1, "els": [{"x": 3, "y": 4}]}')


def test_wire_keys_are_short_names():
    scene = DisplayScene(
        seq=3,
        els=[TextEl(text="hi"), ArrowEl(x=1, y=2), ChevronEl(edge="up")],
    )
    data = json.loads(scene.model_dump_json())
    assert set(data) == {"v", "seq", "els"}
    text_el, arrow_el, chevron_el = data["els"]
    assert set(text_el) == {"t", "x", "y", "size", "text"}
    assert text_el["t"] == "text"
    assert set(arrow_el) == {"t", "x", "y", "angle", "length"}
    assert arrow_el["t"] == "arrow"
    assert set(chevron_el) == {"t", "edge", "label"}
    assert chevron_el["t"] == "chevron"


def test_element_defaults():
    assert TextEl(text="a").size == 1
    assert ArrowEl(x=0, y=0).length == 14
    assert ChevronEl(edge="left").label == ""
    assert DisplayScene().els == []
