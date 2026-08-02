"""Deterministic planner for tests and the ``mock`` profile.

Ignores the image entirely; the question only selects the plan shape.
Find-this questions ("find my phone", "найди телефон") return a single
anchored step - the shape the session renders as an object highlight -
while everything else gets canned multimeter-style instruction steps on the
largest marks. No API calls, always a stable, valid plan.
"""

from lstk_eye.pipeline.interfaces import Planner
from lstk_eye.pipeline.types import LocateResult, Plan, PlanStep, SegMask

_LABELS = ("Red probe here", "Black probe here", "Set dial to V= 20")
_SUMMARY = "mock plan"
_LOCATE_WORDS = ("find", "locate", "where", "show me", "найди", "найти", "где", "покажи")


class MockPlanner(Planner):
    def plan(
        self,
        marked_png: bytes,
        question: str,
        marks: list[SegMask],
        history: list[tuple[str, str]] | None = None,
    ) -> Plan:
        if not marks:
            return Plan(
                steps=[PlanStep(label="No object detected", mark_id=None)],
                summary=_SUMMARY,
            )
        largest = sorted(marks, key=lambda m: m.area, reverse=True)
        q = question.lower()
        if any(w in q for w in _LOCATE_WORDS):
            return Plan(
                steps=[PlanStep(label="Here", mark_id=largest[0].mark_id)],
                summary=_SUMMARY,
            )
        steps = [
            PlanStep(label=label, mark_id=mask.mark_id)
            for mask, label in zip(largest, _LABELS, strict=False)
        ]
        return Plan(steps=steps, summary=_SUMMARY)

    def locate(self, image_bgr, question, history=None):
        """Deterministic direct pointing: center of the frame unless the
        question mentions "nothing" (simulates an honest not-visible)."""
        if "nothing" in question.lower():
            return LocateResult(found=False, label="nothing")
        return LocateResult(
            found=True, label="Here", point=(0.5, 0.5), bbox=(0.4, 0.4, 0.2, 0.2)
        )
