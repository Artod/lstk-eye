"""Planner backed by the Anthropic API (structured outputs + vision)."""

import base64

from pydantic import BaseModel, Field

from lstk_eye.config import PlannerConfig
from lstk_eye.errors import PlanningError
from lstk_eye.pipeline.interfaces import Planner
from lstk_eye.pipeline.planner.prompts import SYSTEM_PROMPT, build_user_text
from lstk_eye.pipeline.types import Plan, PlanStep, SegMask


class _StepOut(BaseModel):
    label: str = Field(description="Instruction shown on the HUD, ASCII, max ~42 chars")
    mark_id: int | None = Field(
        description="Mark number the arrow points at, or null for text-only steps"
    )
    detail: str | None = Field(default=None, description="Optional longer wording, not displayed")


class _PlanOut(BaseModel):
    steps: list[_StepOut]
    summary: str | None = Field(default=None, description="One-line summary of the whole answer")


class AnthropicPlanner(Planner):
    def __init__(self, cfg: PlannerConfig, client=None):
        self._cfg = cfg
        self._client = client  # injectable for tests; real client built lazily

    def _get_client(self):
        if self._client is None:
            import anthropic

            # Zero-arg constructor resolves ANTHROPIC_API_KEY / auth profile.
            self._client = anthropic.Anthropic(timeout=self._cfg.timeout_s)
        return self._client

    def plan(
        self,
        marked_png: bytes,
        question: str,
        marks: list[SegMask],
        history: list[tuple[str, str]] | None = None,
    ) -> Plan:
        import anthropic

        mark_ids = [m.mark_id for m in marks]
        image_b64 = base64.standard_b64encode(marked_png).decode("ascii")
        # Earlier turns as plain text: their photos are gone, but the wearer's
        # follow-ups build on what was already said.
        messages: list[dict] = []
        for past_q, past_a in (history or [])[-6:]:
            messages.append({"role": "user", "content": past_q})
            messages.append({"role": "assistant", "content": past_a})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": build_user_text(question, mark_ids, self._cfg.max_steps),
                    },
                ],
            }
        )
        try:
            response = self._get_client().messages.parse(
                model=self._cfg.model,
                max_tokens=self._cfg.max_tokens,
                output_config={"effort": self._cfg.effort},
                system=SYSTEM_PROMPT,
                messages=messages,
                output_format=_PlanOut,
            )
        except anthropic.AuthenticationError as e:
            raise PlanningError(
                "Anthropic API authentication failed - set ANTHROPIC_API_KEY "
                "or run with --profile mock"
            ) from e
        except anthropic.APIError as e:
            raise PlanningError(f"Anthropic API error: {e}") from e

        if response.stop_reason == "refusal":
            raise PlanningError("model declined the request (stop_reason=refusal)")
        parsed = response.parsed_output
        if parsed is None:
            raise PlanningError("model output did not match the plan schema")
        return self._validate(parsed, set(mark_ids))

    def _validate(self, out: _PlanOut, valid_ids: set[int]) -> Plan:
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
