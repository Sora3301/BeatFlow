"""``python3 main.py <command>`` - the main-program CLI."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from beatflow_input import Injector, ModelStream
from beatflow_input.backends import BackendUnavailable, DryRunBackend, make_backend

from .config import TAIL_FRAMES, RunConfig
from .demo import DEMO_FRAMES, demo_model
from .frame_source import SyntheticSource
from .loop import AgentLoop
from .model import RecordedModel
from .video_source import VideoSource

__all__ = ["build_parser", "main"]


def _parse_pair(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError as exc:  # noqa: TRY003 - argparse wants a clean message
        raise argparse.ArgumentTypeError(f"expected WxH, got {text!r}") from exc


def _parse_quad(text: str) -> tuple[float, float, float, float]:
    parts = text.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"expected X,Y,W,H, got {text!r}")
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected X,Y,W,H, got {text!r}") from exc


def _common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--backend", choices=["dryrun", "uinput"], default="dryrun")
    sp.add_argument("--fps", type=float, default=60.0, help="tick rate (default 60)")
    sp.add_argument("--screen", type=_parse_pair, default=(1920, 1080), metavar="WxH")
    sp.add_argument("--window", type=_parse_quad, default=None, metavar="X,Y,W,H",
                    help="game area in screen pixels (Wayland cannot detect it)")
    sp.add_argument("--game-size", type=_parse_pair, default=None, metavar="WxH",
                    help="letterbox a game of this size into the output")
    sp.add_argument("--capacity", type=int, default=10, help="physical MT slots")
    sp.add_argument("--policy", choices=["steal_lru", "drop"], default="steal_lru")
    sp.add_argument("--duration", type=float, default=0.0, help="stop after N seconds")
    sp.add_argument("--lead-in", type=float, default=0.0,
                    help="seconds to wait before the first frame (uinput: focus the game)")
    sp.add_argument("--no-realtime", dest="realtime", action="store_false",
                    help="run as fast as possible (no deadline pacing)")
    sp.add_argument("--no-invert-y", dest="invert_y", action="store_false",
                    help="treat y as screen-style (+y down); default is +y up")
    sp.add_argument("--dump", help="write the dry-run ops to this JSONL file")
    sp.add_argument("--json", action="store_true", help="print one JSON object")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="beatflow-main",
        description="BeatFlow main program: capture -> model -> virtual touch injection.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("run", help="replay a prediction dump, or run a model over a video")
    _common(sp)
    src = sp.add_mutually_exclusive_group(required=True)
    src.add_argument("--recorded", metavar="PATH",
                     help="JSONL: one {\"frame\", \"contacts\"} row per non-empty frame")
    src.add_argument("--frames", metavar="VIDEO",
                     help="decode VIDEO with ffmpeg and run --model over every frame")
    sp.add_argument("--model-dir", metavar="DIR",
                    help="path to a cortex-sama checkout (with --frames)")
    sp.add_argument("--ckpt", metavar="PATH", help="model checkpoint .pt (with --frames)")
    sp.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    sp.add_argument("--input-size", type=_parse_pair, default=(320, 180), metavar="WxH",
                    help="model input size (default 320x180)")
    sp.add_argument("--min-score", type=float, default=0.0,
                    help="drop slots whose argmax probability is below this")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("demo", help="run a built-in prediction sequence (no data files)")
    _common(sp)
    sp.add_argument("--export", metavar="PATH",
                    help="write the demo predictions as a recorded-prediction JSONL")
    sp.set_defaults(func=cmd_demo)
    return p


# -- commands ---------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    if args.recorded:
        try:
            model = RecordedModel.from_jsonl(args.recorded)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if model.frame_count == 0:
            print(f"error: {args.recorded} contains no frames", file=sys.stderr)
            return 2
        return _run(model, args)

    # -- live model over a video (offline end-to-end, no screen capture yet)
    if not args.model_dir or not args.ckpt:
        print("error: --frames needs --model-dir and --ckpt", file=sys.stderr)
        return 2
    if not Path(args.frames).is_file():
        print(f"error: video not found: {args.frames}", file=sys.stderr)
        return 2

    from .cortex import CortexModel  # local: torch must not load for the JSONL path

    model = CortexModel(
        model_dir=args.model_dir,
        ckpt=args.ckpt,
        device=args.device,
        input_size=args.input_size,
        min_score=args.min_score,
    )
    try:
        model.load()
    except (OSError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return _run(model, args, source=VideoSource(args.frames, size=args.input_size))


def cmd_demo(args: argparse.Namespace) -> int:
    model = demo_model()
    if args.export:
        rows = model.to_jsonl(args.export)
        print(f"predictions -> {args.export} ({rows} frames)")
    # The built-in sequence is longer than its last event; keep the full length.
    model_frames = DEMO_FRAMES
    return _run(model, args, frames_override=model_frames)


def _run(
    model,
    args: argparse.Namespace,
    frames_override: int | None = None,
    source=None,
) -> int:
    cfg = RunConfig(
        fps=args.fps,
        capacity=args.capacity,
        policy=args.policy,
        screen=args.screen,
        window=args.window,
        game_size=args.game_size,
        realtime=args.realtime,
        duration=args.duration,
        lead_in=args.lead_in,
        invert_y=args.invert_y,
    )
    if source is None:
        total = (
            frames_override
            if frames_override is not None
            else model.frame_count + TAIL_FRAMES
        )
        cap = cfg.max_frames()
        if cap is not None:
            total = min(total, cap)
        source = SyntheticSource(total)
    else:
        # Unknown length: stop at EOF, or at --duration when given.
        total = cfg.max_frames()

    geometry = cfg.geometry()
    backend = make_backend(args.backend, capacity=cfg.capacity)
    try:
        backend.open()
    except BackendUnavailable as exc:
        print(f"cannot start {args.backend} backend: {exc}", file=sys.stderr)
        return 1

    injector = Injector(backend, geometry, capacity=cfg.capacity, policy=cfg.policy)
    stream = ModelStream(cfg.interpreter())
    loop = AgentLoop(
        source, model, injector, stream,
        fps=cfg.fps,
        realtime=cfg.realtime,
        max_frames=total,
        flush=cfg.flush,
    )

    describe = getattr(model, "describe", None)
    if callable(describe):
        print(f"model    {describe()}")
    else:
        rejected = getattr(model, "rejected", 0)
        print(f"model    {model.name}: {model.contact_count} contacts over "
              f"{model.frame_count} frames"
              + (f"  ({rejected} rejected)" if rejected else ""))
    if total is None:
        print(f"plan     until the source ends, {cfg.fps:g} fps, realtime={cfg.realtime}")
    else:
        print(f"plan     {total} frames at {cfg.fps:g} fps "
              f"({total / cfg.fps:.2f}s), realtime={cfg.realtime}")
    print(f"geometry {geometry.describe()}")
    print(f"backend  {backend.name} capacity={cfg.capacity}")
    if cfg.fps != 60.0:
        print(f"note     tap/hold/flick durations are frame counts; at {cfg.fps:g} fps "
              f"they are {60.0 / cfg.fps:.2f}x the 60 fps value")
    if args.backend == "uinput" and cfg.lead_in:
        print(f"starting in {cfg.lead_in:.1f}s - focus the game window")

    try:
        if cfg.lead_in and args.backend == "uinput":
            time.sleep(cfg.lead_in)
        result = loop.run()
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    finally:
        try:
            injector.release_all()
        finally:
            backend.close()

    if args.dump and isinstance(backend, DryRunBackend):
        backend.dump(args.dump)

    if args.json:
        print(json.dumps({
            "loop": result.stats.summary(),
            "stream": stream.stats.as_dict(),
            "injector": injector.stats.as_dict(),
            "allocator": injector.allocator.stats.as_dict(),
        }, ensure_ascii=False))
    else:
        print()
        print(result.stats.format())
        print(json.dumps(stream.stats.as_dict(), ensure_ascii=False))
        print(json.dumps(injector.stats.as_dict(), ensure_ascii=False))
        if args.dump:
            print(f"recorded -> {args.dump}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
