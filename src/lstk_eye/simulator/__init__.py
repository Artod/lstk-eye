"""Software stand-in for the glasses: HTTP device client, offline OLED
renderer, and scripted end-to-end scenarios. Lets the whole system run and be
tested with zero hardware."""

from lstk_eye.simulator.device import SimulatedGlasses, make_mock_wav
from lstk_eye.simulator.renderer import render_scene, save_scene
from lstk_eye.simulator.scenario import run_scenario

__all__ = [
    "SimulatedGlasses",
    "make_mock_wav",
    "render_scene",
    "run_scenario",
    "save_scene",
]
