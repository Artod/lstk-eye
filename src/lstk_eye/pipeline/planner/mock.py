"""Deterministic planner for tests and the ``mock`` profile.

Ignores the image and the question entirely; anchors canned multimeter-style
labels to the largest marks so downstream stages always receive a stable,
valid plan without any API call.
"""

from lstk_eye.pipeline.interfaces import Planner
from lstk_eye.pipeline.types import Plan, PlanStep, SegMask

_LABELS = ("Red probe here", "Black probe here", "Set dial to V= 20")
_SUMMARY = "mock plan"


class MockPlanner(Planner):
    def plan(self, marked_png: bytes, question: str, marks: list[SegMask]) -> Plan:
        if not marks:
            return Plan(
                steps=[PlanStep(label="No object detected", mark_id=None)],
                summary=_SUMMARY,
            )
        largest = sorted(marks, key=lambda m: m.area, reverse=True)
        steps = [
            PlanStep(label=label, mark_id=mask.mark_id)
            for mask, label in zip(largest, _LABELS, strict=False)
        ]
        return Plan(steps=steps, summary=_SUMMARY)
