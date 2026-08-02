"""Planner backed by the Anthropic API (structured outputs + vision)."""

import base64
from typing import Literal

from pydantic import BaseModel, Field

from lstk_eye.config import PlannerConfig
from lstk_eye.errors import PlanningError
from lstk_eye.pipeline.interfaces import Planner
from lstk_eye.pipeline.planner.prompts import LOCATE_PROMPT, SYSTEM_PROMPT, build_user_text
from lstk_eye.pipeline.types import LocateResult, Plan, PlanStep, SegMask


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

    def locate(self, image_bgr, question, history=None):
        """Direct pointing with a self-directed zoom loop: Claude localizes
        the target itself and requests crops when the detail is too small,
        so any object scale (a resistor on a board) resolves precisely."""
        import anthropic
        import cv2

        region = (0.0, 0.0, 1.0, 1.0)  # absolute region the current image shows
        current = image_bgr
        note = ""
        label = "target"
        for _ in range(3):
            ok, jpg = cv2.imencode(".jpg", current, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise PlanningError("could not encode frame for the model")
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
                                "media_type": "image/jpeg",
                                "data": base64.standard_b64encode(jpg.tobytes()).decode(),
                            },
                        },
                        {"type": "text", "text": f"{note}{question}"},
                    ],
                }
            )
            try:
                response = self._get_client().messages.parse(
                    model=self._cfg.model,
                    max_tokens=self._cfg.max_tokens,
                    output_config={"effort": self._cfg.effort},
                    system=LOCATE_PROMPT,
                    messages=messages,
                    output_format=_LocateOut,
                )
            except anthropic.AuthenticationError as e:
                raise PlanningError(
                    "Anthropic API authentication failed - set ANTHROPIC_API_KEY"
                ) from e
            except anthropic.APIError as e:
                raise PlanningError(f"Anthropic API error: {e}") from e
            if response.stop_reason == "refusal":
                raise PlanningError("model declined the request")
            out = response.parsed_output
            if out is None:
                raise PlanningError("model output did not match the locate schema")
            label = (out.label or label).strip() or "target"

            rx, ry, rw, rh = region
            if out.status == "not_visible":
                return LocateResult(found=False, label=label)
            if out.status == "found" and out.x is not None and out.y is not None:
                px = rx + min(max(out.x, 0.0), 1.0) * rw
                py = ry + min(max(out.y, 0.0), 1.0) * rh
                bw = min(max(out.box_w or 0.12, 0.01), 1.0) * rw
                bh = min(max(out.box_h or 0.12, 0.01), 1.0) * rh
                bx = min(max(px - bw / 2, 0.0), 1.0 - bw)
                by = min(max(py - bh / 2, 0.0), 1.0 - bh)
                return LocateResult(
                    found=True, label=label, point=(px, py), bbox=(bx, by, bw, bh)
                )
            if out.status == "zoom" and out.zoom_w and out.zoom_h:
                zx = rx + min(max(out.zoom_x or 0.0, 0.0), 1.0) * rw
                zy = ry + min(max(out.zoom_y or 0.0, 0.0), 1.0) * rh
                zw = max(min(out.zoom_w, 1.0) * rw, 0.05)
                zh = max(min(out.zoom_h, 1.0) * rh, 0.05)
                zx = min(max(zx, 0.0), 1.0 - zw)
                zy = min(max(zy, 0.0), 1.0 - zh)
                fh, fw = image_bgr.shape[:2]
                crop = image_bgr[
                    int(zy * fh) : int((zy + zh) * fh), int(zx * fw) : int((zx + zw) * fw)
                ]
                if crop.size == 0:
                    break
                current = crop
                region = (zx, zy, zw, zh)
                note = "(Zoomed view of the region you requested.) "
                continue
            break
        return LocateResult(found=False, label=label)

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


class _LocateOut(BaseModel):
    status: Literal["found", "not_visible", "zoom"]
    label: str = Field(description="ONE short word naming the target object")
    x: float | None = Field(default=None, description="normalized center x in THIS image")
    y: float | None = Field(default=None, description="normalized center y in THIS image")
    box_w: float | None = Field(default=None, description="tight box width, normalized")
    box_h: float | None = Field(default=None, description="tight box height, normalized")
    zoom_x: float | None = Field(default=None, description="zoom region left, normalized")
    zoom_y: float | None = Field(default=None, description="zoom region top, normalized")
    zoom_w: float | None = Field(default=None, description="zoom region width, normalized")
    zoom_h: float | None = Field(default=None, description="zoom region height, normalized")
