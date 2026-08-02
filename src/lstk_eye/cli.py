"""Command-line interface: serve, simulate, demo, doctor."""

import json
import os
import shutil
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from lstk_eye import __version__
from lstk_eye.config import AppConfig, load_config

app = typer.Typer(help="lstk-eye: HUD glasses server and tools", no_args_is_help=True)
console = Console()

ProfileOpt = typer.Option(None, "--profile", help="Config profile: real or mock")
ConfigOpt = typer.Option(None, "--config", help="Path to a TOML config file")


def _load(config: Path | None, profile: str | None, **overrides) -> AppConfig:
    return load_config(config, profile=profile, **overrides)


@app.command()
def serve(
    profile: str | None = ProfileOpt,
    config: Path | None = ConfigOpt,
    host: str | None = typer.Option(None, help="Bind address (default from config)"),
    port: int | None = typer.Option(None, help="Port (default from config)"),
):
    """Run the glasses server."""
    import uvicorn

    from lstk_eye.server import create_app

    cfg = _load(config, profile)
    if host or port:
        cfg = cfg.model_copy(
            update={
                "server": cfg.server.model_copy(
                    update={k: v for k, v in {"host": host, "port": port}.items() if v}
                )
            }
        )
    console.print(f"[bold]lstk-eye[/] v{__version__}  profile=[cyan]{cfg.profile}[/]")
    console.print(
        f"backends: stt=[cyan]{cfg.stt.backend}[/] segmenter=[cyan]{cfg.segmenter.backend}[/] "
        f"planner=[cyan]{cfg.planner.backend}[/] ({cfg.planner.model})"
    )
    console.print(f"listening on http://{cfg.server.host}:{cfg.server.port}")
    uvicorn.run(create_app(cfg), host=cfg.server.host, port=cfg.server.port, log_level="info")


@app.command()
def simulate(
    ask: str = typer.Option("how do I check battery voltage", help="Question to send"),
    image: Path | None = typer.Option(None, help="Photo to capture (default: synthetic scene)"),
    server: str = typer.Option("http://127.0.0.1:8321", help="Server base URL"),
    out: Path | None = typer.Option(None, help="Output dir for rendered frames"),
):
    """Play the role of the glasses against a running server: capture, ask,
    step through slides, render every scene to PNG."""
    import cv2

    from lstk_eye.simulator import SimulatedGlasses, run_scenario
    from lstk_eye.testimage import make_test_image

    if image is not None:
        frame = cv2.imread(str(image))
        if frame is None:
            console.print(f"[red]cannot read image: {image}[/]")
            raise typer.Exit(1)
    else:
        frame = make_test_image()

    out_dir = out or Path("runs") / f"sim_{time.strftime('%Y%m%d_%H%M%S')}"
    glasses = SimulatedGlasses(base_url=server)
    try:
        glasses.health()
    except Exception as e:
        console.print(f"[red]server not reachable at {server}[/]: {e}")
        console.print("start it with: uv run lstk-eye serve --profile mock")
        raise typer.Exit(1) from None
    frames = run_scenario(glasses, frame, ask, out_dir)
    console.print(f"[green]{len(frames)} frames rendered to {out_dir}[/]")


@app.command()
def demo(
    question: str = typer.Argument(..., help="Question about the image"),
    image: Path | None = typer.Option(None, help="Input photo (default: synthetic scene)"),
    profile: str | None = ProfileOpt,
    config: Path | None = ConfigOpt,
    out: Path | None = typer.Option(None, help="Output directory"),
):
    """Run the Set-of-Marks pipeline standalone on one image - no server, no
    device: segmentation -> marks overlay -> plan -> annotated result."""
    import dataclasses

    import cv2

    from lstk_eye.pipeline import factories
    from lstk_eye.pipeline.marks import draw_slides, encode_png, overlay_marks
    from lstk_eye.pipeline.types import Slide
    from lstk_eye.testimage import make_test_image

    cfg = _load(config, profile)
    if image is not None:
        frame = cv2.imread(str(image))
        if frame is None:
            console.print(f"[red]cannot read image: {image}[/]")
            raise typer.Exit(1)
    else:
        frame = make_test_image()

    out_dir = out or Path("runs") / f"demo_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"segmenting with [cyan]{cfg.segmenter.backend}[/]...")
    segmenter = factories.create_segmenter(cfg)
    masks = segmenter.segment(frame)
    console.print(f"  {len(masks)} regions")
    marked = overlay_marks(frame, masks)
    (out_dir / "marked.png").write_bytes(encode_png(marked))

    console.print(f"planning with [cyan]{cfg.planner.backend}[/] ({cfg.planner.model})...")
    planner = factories.create_planner(cfg)
    plan = planner.plan(encode_png(marked), question, masks)

    by_id = {m.mark_id: m for m in masks}
    slides = [
        Slide(
            index=i,
            total=len(plan.steps),
            label=s.label,
            anchor=by_id[s.mark_id].centroid if s.mark_id in by_id else None,
            mark_id=s.mark_id if s.mark_id in by_id else None,
        )
        for i, s in enumerate(plan.steps)
    ]
    annotated = draw_slides(frame, slides)
    (out_dir / "slides.png").write_bytes(encode_png(annotated))
    (out_dir / "plan.json").write_text(
        json.dumps(dataclasses.asdict(plan), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for s in slides:
        target = f"mark {s.mark_id}" if s.mark_id is not None else "no anchor"
        console.print(f"  [bold]{s.index + 1}/{s.total}[/] {s.label}  [dim]({target})[/]")
    console.print(f"[green]outputs in {out_dir}[/]: marked.png, slides.png, plan.json")


@app.command()
def target(
    out: Path = typer.Option(Path("calibration_target.png"), help="Output PNG path"),
    size: int = typer.Option(900, help="Marker size in pixels (plus quiet zone)"),
):
    """Generate the calibration target - show it fullscreen on a phone or
    monitor, then say "calibrate" to the glasses and follow the crosshairs."""
    from lstk_eye.pipeline.fiducial import generate_target

    path = generate_target(out, size=size)
    console.print(f"[green]calibration target written to {path}[/]")
    console.print("display it on a screen, then say 'calibrate' to the glasses")


@app.command()
def doctor(
    profile: str | None = ProfileOpt,
    config: Path | None = ConfigOpt,
):
    """Check the environment: dependencies, backends, credentials."""
    checks: list[tuple[str, bool, str]] = []

    import sys

    checks.append(
        (
            "python >= 3.11",
            sys.version_info >= (3, 11),
            f"{sys.version_info.major}.{sys.version_info.minor}",
        )
    )

    for mod, hint in [("cv2", "opencv-python"), ("numpy", "numpy"), ("fastapi", "fastapi")]:
        try:
            __import__(mod)
            checks.append((f"import {mod}", True, "ok"))
        except ImportError:
            checks.append((f"import {mod}", False, f"pip install {hint}"))

    try:
        cfg = _load(config, profile)
        checks.append(("config", True, f"profile={cfg.profile}"))
    except Exception as e:
        checks.append(("config", False, str(e)))
        cfg = None

    try:
        import faster_whisper  # noqa: F401

        checks.append(("stt: faster-whisper", True, "installed"))
    except ImportError:
        checks.append(
            ("stt: faster-whisper", False, 'pip install "lstk-eye\\[stt]" (or mock profile)')
        )

    try:
        import ultralytics  # noqa: F401

        checks.append(("segmenter: ultralytics", True, "installed"))
    except ImportError:
        checks.append(
            ("segmenter: ultralytics", False, 'pip install "lstk-eye\\[seg]" (or mock profile)')
        )

    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_cli = shutil.which("claude") is not None
    checks.append(
        (
            "planner credentials",
            has_key or has_cli,
            "ANTHROPIC_API_KEY set"
            if has_key
            else ("claude CLI found (use planner.backend=claude-cli)" if has_cli else
                  "set ANTHROPIC_API_KEY, or install Claude Code for claude-cli, or use mock"),
        )
    )

    if cfg is not None:
        try:
            cfg.storage.dir.mkdir(parents=True, exist_ok=True)
            probe = cfg.storage.dir / ".doctor"
            probe.write_text("ok")
            probe.unlink()
            checks.append(("runs dir writable", True, str(cfg.storage.dir)))
        except OSError as e:
            checks.append(("runs dir writable", False, str(e)))

    table = Table(title=f"lstk-eye doctor v{__version__}")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    ok_all = True
    for name, ok, detail in checks:
        table.add_row(name, "[green]ok[/]" if ok else "[red]FAIL[/]", detail)
        ok_all = ok_all and ok
    console.print(table)
    if not ok_all:
        console.print("[yellow]some checks failed - mock profile works regardless:[/] "
                      "uv run lstk-eye serve --profile mock")
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
