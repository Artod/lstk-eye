"""Scripted end-to-end run: capture -> ask -> step through slides -> cancel.

Saves every distinct scene the server sends as a numbered PNG, giving a
storyboard of what the wearer would see on the HUD for one question.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lstk_eye.protocol import DisplayScene
from lstk_eye.simulator.device import SimulatedGlasses
from lstk_eye.simulator.renderer import save_scene


def run_scenario(
    glasses: SimulatedGlasses,
    image: np.ndarray | str | Path,
    question: str,
    out_dir: Path,
    max_steps: int = 12,
) -> list[Path]:
    """Drive one full session and return the saved frame paths in order.

    Advances slides with ``next()`` until the server ends the session, the
    scene stops changing, or ``max_steps`` advances have been sent; if the
    session is still active after that, it is cancelled.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    last_saved_seq: int | None = None

    def save(scene: DisplayScene | None, label: str) -> bool:
        nonlocal last_saved_seq
        if scene is None or scene.seq == last_saved_seq:
            return False
        path = save_scene(scene, out_dir / f"frame_{len(saved):03d}.png")
        saved.append(path)
        last_saved_seq = scene.seq
        print(f"[scenario] {label}: seq={scene.seq} -> {path.name}")
        return True

    ack = glasses.capture(image)
    save(ack.scene, "capture")

    resp = glasses.ask(text=question)
    save(resp.scene, "ask")
    active = resp.active

    for _ in range(max_steps):
        if not active:
            break
        step = glasses.next()
        active = step.active
        if not save(step.scene, "next"):
            break

    if active:
        end = glasses.cancel()
        save(end.scene, "cancel")

    return saved
