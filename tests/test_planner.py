"""Unit tests for the three planner backends (mock, anthropic, claude-cli).

No network, no CLI binary: the Anthropic client is injected as a stub, the
CLI subprocess is monkeypatched.
"""

import base64
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from lstk_eye.config import PlannerConfig
from lstk_eye.errors import PlanningError
from lstk_eye.pipeline.planner.anthropic_planner import AnthropicPlanner, _PlanOut, _StepOut
from lstk_eye.pipeline.planner.claude_cli import ClaudeCliPlanner
from lstk_eye.pipeline.planner.mock import MockPlanner
from lstk_eye.pipeline.types import SegMask

PNG = b"\x89PNG\r\n\x1a\nfake-payload"
QUESTION = "how do I check battery voltage"


def make_mask(mark_id: int, area: float) -> SegMask:
    side = min(area**0.5, 1.0)
    return SegMask(
        mark_id=mark_id,
        bbox=(0.0, 0.0, side, side),
        centroid=(side / 2, side / 2),
        area=area,
    )


# --- MockPlanner ---


def test_mock_planner_picks_three_largest_marks():
    marks = [
        make_mask(1, 0.05),
        make_mask(2, 0.30),
        make_mask(3, 0.10),
        make_mask(4, 0.22),
        make_mask(5, 0.01),
    ]
    plan = MockPlanner().plan(PNG, QUESTION, marks)

    assert len(plan.steps) == 3
    assert [s.mark_id for s in plan.steps] == [2, 4, 3]  # descending area
    valid_ids = {m.mark_id for m in marks}
    assert all(s.mark_id in valid_ids for s in plan.steps)
    assert all(s.label.strip() for s in plan.steps)
    assert plan.summary


def test_mock_planner_no_marks_gives_text_only_step():
    plan = MockPlanner().plan(PNG, QUESTION, [])

    assert len(plan.steps) == 1
    assert plan.steps[0].mark_id is None
    assert plan.steps[0].label == "No object detected"


# --- AnthropicPlanner (injected fake client) ---


class FakeClient:
    """Stub matching the client.messages.parse(**kwargs) surface."""

    def __init__(self, response):
        self.parse_kwargs = None

        def parse(**kwargs):
            self.parse_kwargs = kwargs
            return response

        self.messages = SimpleNamespace(parse=parse)


def make_response(steps, summary="check complete", stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        parsed_output=_PlanOut(steps=steps, summary=summary),
    )


def make_planner(response, max_steps=7):
    cfg = PlannerConfig(backend="anthropic", max_steps=max_steps)
    return AnthropicPlanner(cfg, client=FakeClient(response))


def test_anthropic_happy_path_preserves_order_labels_details():
    marks = [make_mask(1, 0.1), make_mask(2, 0.2), make_mask(3, 0.3)]
    response = make_response(
        [
            _StepOut(label="Red probe here", mark_id=2, detail="VOmA jack"),
            _StepOut(label="Black probe here", mark_id=1, detail=None),
            _StepOut(label="Set dial to V= 20", mark_id=None, detail="rotary dial"),
        ]
    )
    plan = make_planner(response).plan(PNG, QUESTION, marks)

    assert [s.label for s in plan.steps] == [
        "Red probe here",
        "Black probe here",
        "Set dial to V= 20",
    ]
    assert [s.mark_id for s in plan.steps] == [2, 1, None]
    assert [s.detail for s in plan.steps] == ["VOmA jack", None, "rotary dial"]
    assert plan.summary == "check complete"


def test_anthropic_parse_kwargs_carry_image_and_question():
    marks = [make_mask(1, 0.1)]
    planner = make_planner(make_response([_StepOut(label="ok", mark_id=1)]))
    planner.plan(PNG, QUESTION, marks)

    kwargs = planner._client.parse_kwargs
    content = kwargs["messages"][0]["content"]
    image_block = content[0]
    assert image_block["type"] == "image"
    assert image_block["source"]["media_type"] == "image/png"
    assert image_block["source"]["data"] == base64.standard_b64encode(PNG).decode("ascii")
    text_block = content[1]
    assert text_block["type"] == "text"
    assert QUESTION in text_block["text"]


def test_anthropic_out_of_range_mark_id_becomes_none():
    marks = [make_mask(1, 0.1), make_mask(2, 0.2)]
    response = make_response([_StepOut(label="Probe here", mark_id=99)])
    plan = make_planner(response).plan(PNG, QUESTION, marks)

    assert plan.steps[0].mark_id is None


def test_anthropic_truncates_to_max_steps():
    marks = [make_mask(1, 0.1)]
    response = make_response(
        [_StepOut(label=f"step {i}", mark_id=1) for i in range(5)]
    )
    plan = make_planner(response, max_steps=2).plan(PNG, QUESTION, marks)

    assert [s.label for s in plan.steps] == ["step 0", "step 1"]


def test_anthropic_drops_blank_labels():
    marks = [make_mask(1, 0.1)]
    response = make_response(
        [
            _StepOut(label="   ", mark_id=1),
            _StepOut(label="Real step", mark_id=1),
        ]
    )
    plan = make_planner(response).plan(PNG, QUESTION, marks)

    assert [s.label for s in plan.steps] == ["Real step"]


def test_anthropic_all_blank_labels_raises():
    marks = [make_mask(1, 0.1)]
    response = make_response([_StepOut(label="  ", mark_id=1)])
    with pytest.raises(PlanningError):
        make_planner(response).plan(PNG, QUESTION, marks)


def test_anthropic_parsed_output_none_raises():
    response = SimpleNamespace(stop_reason="end_turn", parsed_output=None)
    with pytest.raises(PlanningError):
        make_planner(response).plan(PNG, QUESTION, [make_mask(1, 0.1)])


def test_anthropic_refusal_raises():
    response = make_response([_StepOut(label="ok", mark_id=1)], stop_reason="refusal")
    with pytest.raises(PlanningError):
        make_planner(response).plan(PNG, QUESTION, [make_mask(1, 0.1)])


# --- ClaudeCliPlanner (monkeypatched subprocess.run) ---


def cli_planner(timeout_s=30.0, max_steps=7):
    return ClaudeCliPlanner(
        PlannerConfig(backend="claude-cli", timeout_s=timeout_s, max_steps=max_steps)
    )


def envelope_with(inner: dict, fenced: bool = True) -> str:
    body = json.dumps(inner)
    if fenced:
        body = f"```json\n{body}\n```"
    return json.dumps({"result": body})


def test_claude_cli_happy_path(monkeypatch):
    marks = [make_mask(1, 0.1), make_mask(2, 0.2)]
    inner = {
        "steps": [
            {"label": "Red probe here", "mark_id": 2, "detail": "jack"},
            {"label": "Black probe here", "mark_id": 7, "detail": None},
        ],
        "summary": "done",
    }
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        prompt = cmd[2]
        path = Path(re.search(r"(/\S+\.png)", prompt).group(1))
        seen["tmp_path"] = path
        seen["tmp_bytes"] = path.read_bytes()
        return subprocess.CompletedProcess(cmd, 0, stdout=envelope_with(inner), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    plan = cli_planner(timeout_s=12.5).plan(PNG, QUESTION, marks)

    assert [s.label for s in plan.steps] == ["Red probe here", "Black probe here"]
    assert [s.mark_id for s in plan.steps] == [2, None]  # 7 is out of range
    assert plan.summary == "done"

    cmd = seen["cmd"]
    assert cmd[0] == "claude"
    assert cmd[1] == "-p"
    assert cmd[3:] == ["--output-format", "json", "--allowedTools", "Read"]
    assert QUESTION in cmd[2]
    assert seen["kwargs"]["timeout"] == 12.5
    assert seen["tmp_bytes"] == PNG  # image existed on disk during the call
    assert not seen["tmp_path"].exists()  # and was deleted afterwards


def test_claude_cli_unfenced_result_also_parses(monkeypatch):
    inner = {"steps": [{"label": "ok", "mark_id": 1, "detail": None}], "summary": None}

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout=envelope_with(inner, fenced=False), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    plan = cli_planner().plan(PNG, QUESTION, [make_mask(1, 0.1)])

    assert plan.steps[0].label == "ok"
    assert plan.steps[0].mark_id == 1


@pytest.mark.parametrize(
    "stdout",
    [
        "not json at all",
        json.dumps({"no_result": True}),
        json.dumps({"result": "```json\n{broken\n```"}),
        json.dumps({"result": '{"steps": "not-a-list"}'}),
    ],
)
def test_claude_cli_bad_output_raises(monkeypatch, stdout):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(PlanningError):
        cli_planner().plan(PNG, QUESTION, [make_mask(1, 0.1)])


def test_claude_cli_nonzero_exit_raises(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="login required")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(PlanningError, match="exited with code 1"):
        cli_planner().plan(PNG, QUESTION, [make_mask(1, 0.1)])


def test_claude_cli_timeout_raises(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(PlanningError, match="timed out"):
        cli_planner(timeout_s=5.0).plan(PNG, QUESTION, [make_mask(1, 0.1)])


def test_claude_cli_missing_binary_raises(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("claude")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(PlanningError, match="not found"):
        cli_planner().plan(PNG, QUESTION, [make_mask(1, 0.1)])
