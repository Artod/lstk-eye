"""EXPERIMENTAL planner backend that shells out to the Claude Code CLI.

Reuses a local ``claude`` subscription instead of an API key: the marked
photo is written to a temp file, the CLI is told to read it (Read tool only)
and answer in strict JSON, and the answer is validated exactly like the API
backend's. Slower and less robust than the ``anthropic`` backend - useful
when only a Claude Code login is available.
"""

import json
import subprocess
import tempfile
from pathlib import Path

from pydantic import ValidationError

from lstk_eye.config import PlannerConfig
from lstk_eye.errors import PlanningError
from lstk_eye.pipeline.interfaces import Planner
from lstk_eye.pipeline.planner.anthropic_planner import _PlanOut
from lstk_eye.pipeline.planner.prompts import SYSTEM_PROMPT, build_user_text
from lstk_eye.pipeline.types import Plan, PlanStep, SegMask

_SCHEMA = (
    '{"steps": [{"label": str, "mark_id": int | null, "detail": str | null}],'
    ' "summary": str | null}'
)


def _strip_fences(text: str) -> str:
    """Drop an optional ```/```json markdown fence around a JSON body."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.strip()
        if body.endswith("```"):
            body = body[:-3]
    return body.strip()


class ClaudeCliPlanner(Planner):
    def __init__(self, cfg: PlannerConfig):
        self._cfg = cfg

    def plan(
        self,
        marked_png: bytes,
        question: str,
        marks: list[SegMask],
        history: list[tuple[str, str]] | None = None,
    ) -> Plan:
        # History is included as plain dialogue lines ahead of the question.
        if history:
            dialogue = "\n".join(
                f"Wearer: {q}\nYou answered: {a}" for q, a in history[-6:]
            )
            question = f"Earlier in this conversation:\n{dialogue}\n\nNow: {question}"
        mark_ids = [m.mark_id for m in marks]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(marked_png)
            image_path = Path(f.name)
        try:
            prompt = self._build_prompt(image_path, question, mark_ids)
            stdout = self._run_cli(prompt)
            parsed = self._parse(stdout)
        finally:
            image_path.unlink(missing_ok=True)
        return self._validate(parsed, set(mark_ids))

    def _build_prompt(self, image_path: Path, question: str, mark_ids: list[int]) -> str:
        return (
            f"First read the photo at {image_path} with the Read tool, then answer.\n\n"
            f"{SYSTEM_PROMPT}\n"
            f"{build_user_text(question, mark_ids, self._cfg.max_steps)}\n\n"
            "Output STRICT JSON only - no markdown fences, no text before or after -\n"
            f"matching exactly this schema:\n{_SCHEMA}"
        )

    def _run_cli(self, prompt: str) -> str:
        cmd = ["claude", "-p", prompt, "--output-format", "json", "--allowedTools", "Read"]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self._cfg.timeout_s
            )
        except FileNotFoundError as e:
            raise PlanningError(
                "claude CLI not found on PATH - install Claude Code "
                "or switch to planner.backend=anthropic"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise PlanningError(f"claude CLI timed out after {self._cfg.timeout_s:g}s") from e
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()[:500]
            raise PlanningError(f"claude CLI exited with code {proc.returncode}: {stderr}")
        return proc.stdout

    def _parse(self, stdout: str) -> _PlanOut:
        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise PlanningError(f"claude CLI stdout is not valid JSON: {e}") from e
        result = envelope.get("result") if isinstance(envelope, dict) else None
        if not isinstance(result, str):
            raise PlanningError("claude CLI output has no string 'result' field")
        try:
            data = json.loads(_strip_fences(result))
        except json.JSONDecodeError as e:
            raise PlanningError(f"claude CLI result is not valid JSON: {e}") from e
        try:
            return _PlanOut.model_validate(data)
        except ValidationError as e:
            raise PlanningError(f"claude CLI result does not match the plan schema: {e}") from e

    def _validate(self, out: _PlanOut, valid_ids: set[int]) -> Plan:
        # Mirrors AnthropicPlanner._validate: same rules, same failure mode.
        steps: list[PlanStep] = []
        for s in out.steps[: self._cfg.max_steps]:
            label = s.label.strip()
            if not label:
                continue
            mark_id = s.mark_id if s.mark_id in valid_ids else None
            steps.append(PlanStep(label=label, mark_id=mark_id, detail=s.detail))
        if not steps:
            raise PlanningError("planner returned no usable steps")
        return Plan(steps=steps, summary=out.summary)
