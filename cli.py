"""Command line entry point: ``python3 -m beatflow_input <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .actions import InterpreterConfig, build_timeline
from .backends import BackendUnavailable, DryRunBackend, UinputBackend, make_backend
from .geometry import Geometry
from .injector import Injector
from .labels import open_labelset
from .scheduler import TimelineRunner
from .visualizer import RENDERERS, LiveModel, RenderProcess, Streamer, phase_map

DEFAULT_DEVICE = "BeatFlow Virtual Touch"


def _find_labelset(explicit: str | None) -> str:
    if explicit:
        return explicit
    here = Path.cwd()
    for base in (here, *here.parents, Path(__file__).resolve().parent.parent):
        for name in ("map-set-labels.zip", "map-set-labels"):
            candidate = base / name
            if candidate.exists():
                return str(candidate)
    raise SystemExit(
        "label set not found; pass --labels /path/to/map-set-labels.zip"
    )


def _geometry(args) -> Geometry:
    if args.window:
        x, y, w, h = (float(v) for v in args.window)
        return Geometry.windowed(args.screen[0], args.screen[1], x, y, w, h)
    if args.game_size:
        w, h = (float(v) for v in args.game_size)
        return Geometry.letterboxed(args.screen[0], args.screen[1], w, h)
    return Geometry.fullscreen(args.screen[0], args.screen[1])


def _config(args) -> InterpreterConfig:
    return InterpreterConfig(
        fps=args.fps,
        tap_frames=args.tap_frames,
        flick_frames=args.flick_frames,
        flick_delta=(0.0, args.flick_dy),
        hold_grace_frames=args.hold_grace,
        hold_tracking=args.hold_tracking,
    )


# -- commands ---------------------------------------------------------------
def cmd_probe(args) -> int:
    print(f"python      {sys.version.split()[0]}  ({sys.executable})")
    print(f"session     {os.environ.get('XDG_SESSION_TYPE', '?')} "
          f"(wayland={os.environ.get('WAYLAND_DISPLAY', '-')})")
    print(f"/dev/uinput {'present' if os.path.exists('/dev/uinput') else 'MISSING'}")
    try:
        backend = UinputBackend(device_name=args.device)
        with backend:
            print(f"device      {backend.node or '(node not found, but created)'}")
            status = backend.status()
            print(f"capacity    {status['capacity']} slots, abs {status['abs_range']}")
        print("result      OK - uinput is usable")
        return 0
    except BackendUnavailable as exc:
        print(f"result      FAIL - {exc}")
        return 1


def cmd_list(args) -> int:
    with open_labelset(_find_labelset(args.labels)) as ls:
        hits = ls.find(args.query) if args.query else ls.entries
        for e in hits[: args.limit]:
            print(f"{e.num_frames / 60:8.1f}s  max_slot={e.max_slots:<3d} {e.folder}")
        print(f"-- {len(hits)} charts" + (f" (showing {args.limit})" if len(hits) > args.limit else ""))
    return 0


def cmd_plan(args) -> int:
    with open_labelset(_find_labelset(args.labels)) as ls:
        chart = ls.chart(args.chart)
    timeline = build_timeline(chart, _config(args))
    print(f"chart   {chart.name}")
    print(f"frames  num_frames={chart.num_frames}  labelled={chart.first_frame}..{chart.last_frame}")
    print(f"events  {chart.events}  {chart.phase_counts()}")
    print(f"plan    {json.dumps(timeline.summary(), ensure_ascii=False)}")
    print()
    for commit in timeline.head(args.show):
        ops = " ".join(
            f"{o.kind}({o.key}{'' if o.kind == 'release' else f' {o.x:+.2f},{o.y:+.2f}'})"
            for o in commit.ops
        )
        print(f"  f{commit.frame:<6d} t={commit.t:7.3f}s  {ops}")
    return 0


def cmd_replay(args) -> int:
    with open_labelset(_find_labelset(args.labels)) as ls:
        chart = ls.chart(args.chart)
    timeline = build_timeline(chart, _config(args))
    if args.duration:
        timeline.commits = [c for c in timeline.commits if c.t <= args.duration]
    print(f"chart {chart.name}: {len(timeline.commits)} commits, {timeline.n_ops} ops, "
          f"{len(timeline.commits) and timeline.commits[-1].t or 0:.2f}s")

    geometry = _geometry(args)
    print(f"geometry {geometry.describe()}")

    kwargs = {"capacity": args.capacity}
    backend = make_backend(args.backend, **kwargs)
    if isinstance(backend, DryRunBackend):
        backend.open()
        injector = Injector(backend, geometry, policy=args.policy)
        runner = TimelineRunner(injector, speed=args.speed, lead_in=0.0)
        stats = runner.run(timeline)
        print()
        print(json.dumps(backend.summary(), ensure_ascii=False))
        print(json.dumps(stats.summary(), ensure_ascii=False))
        if args.dump:
            backend.dump(args.dump)
            print(f"recorded -> {args.dump}")
        return 0

    try:
        backend.open()
    except BackendUnavailable as exc:
        print(f"cannot start uinput backend: {exc}")
        return 1
    try:
        injector = Injector(backend, geometry, policy=args.policy)
        if args.lead_in:
            print(f"starting in {args.lead_in:.1f}s - focus the game window")
            time.sleep(args.lead_in)
        runner = TimelineRunner(injector, speed=args.speed, lead_in=0.5)
        stats = runner.run(timeline)
        injector.release_all()
    finally:
        backend.close()
    print()
    print(stats.format())
    print(json.dumps(injector.stats.as_dict(), ensure_ascii=False))
    print(json.dumps(injector.allocator.stats.as_dict(), ensure_ascii=False))
    return 0


def cmd_watch(args) -> int:
    """Play a chart in real time with a live visualizer attached."""
    with open_labelset(_find_labelset(args.labels)) as ls:
        chart = ls.chart(args.chart)
    timeline = build_timeline(chart, _config(args))
    if args.duration:
        timeline.commits = [c for c in timeline.commits if c.t <= args.duration]
    print(f"chart {chart.name}: {len(timeline.commits)} commits, {timeline.n_ops} ops, "
          f"{timeline.duration:.1f}s   visualizer={args.viz}")

    geometry = _geometry(args)
    backend = make_backend(args.backend, capacity=args.capacity)
    try:
        backend.open()
    except BackendUnavailable as exc:
        print(f"cannot start {args.backend} backend: {exc}")
        return 1

    injector = Injector(backend, geometry, policy=args.policy)
    model = LiveModel(phase_map(chart))
    viz: RenderProcess | None = None
    streamer: Streamer | None = None
    # Chart time for the HUD; the model keeps monotonic timestamps for ages.
    t0 = time.monotonic() + max(0.0, args.lead_in)
    try:
        if args.viz != "none":
            # Only the terminal renderer has an ASCII mode; Tk ignores the flag.
            extra = ("--ascii",) if args.viz_ascii and args.viz == "terminal" else ()
            try:
                viz = RenderProcess(
                    args.viz, title=chart.name, capacity=args.capacity,
                    fps=args.viz_fps, extra_args=extra,
                )
            except RuntimeError as exc:
                print(f"visualizer unavailable: {exc}")
                return 1
            streamer = Streamer(model, viz, now=lambda: time.monotonic() - t0,
                                fps=args.viz_fps).start()

        def on_commit(sample, commit) -> None:
            now = time.monotonic()
            for op in commit.ops:
                if op.kind == "release":
                    model.release(op.key, now)
                elif op.kind == "press":
                    # Mirror the injector, not the timeline: a press the slot
                    # allocator dropped must not show up as a finger.
                    if injector.allocator.is_active(op.key):
                        slot = injector.allocator.slot_of(op.key)
                        model.press(op.key, -1 if slot is None else slot, op.x, op.y, now)
                else:
                    model.move(op.key, op.x, op.y, now)

        runner = TimelineRunner(injector, speed=args.speed, lead_in=max(0.0, args.lead_in),
                                on_commit=on_commit)
        stats = runner.run(timeline)
        injector.release_all()
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    finally:
        if streamer is not None:
            streamer.stop()
        if viz is not None:
            viz.close({"frames": streamer.frames if streamer else 0,
                       "commits": len(timeline.commits)})
        backend.close()

    print()
    print(stats.format())
    print(json.dumps(injector.stats.as_dict(), ensure_ascii=False))
    return 0


def cmd_selftest(args) -> int:
    """Press/move/release on the real device to prove the plumbing works."""
    geometry = _geometry(args)
    backend = UinputBackend(capacity=args.capacity, device_name=args.device)
    try:
        backend.open()
    except BackendUnavailable as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"device   {backend.node}  capacity={backend.capacity}")
    try:
        points = [(-0.6, 0.375), (0.0, 0.375), (0.6, 0.375)]
        for step in range(args.repeat):
            for i, (gx, gy) in enumerate(points):
                x, y = geometry.to_device(gx, gy)
                t0 = time.monotonic()
                backend.press(i, 100 + step * 10 + i, x, y)
                backend.commit()
                time.sleep(0.03)
                backend.press(i, 100 + step * 10 + i, x, y)  # noop keepalive
                backend.move(i, x, y)
                backend.commit()
                time.sleep(0.03)
                backend.release(i)
                backend.commit()
                print(f"  contact slot={i} -> ({x:5d},{y:5d}) "
                      f"{(time.monotonic() - t0) * 1000:5.1f} ms")
        print("result   OK - 3 contacts pressed/moved/released without error")
        return 0
    finally:
        backend.close()


# -- parser -----------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="beatflow_input", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--labels", help="path to map-set-labels.zip or extracted dir")
        sp.add_argument("--screen", nargs=2, type=int, default=[1920, 1080],
                        metavar=("W", "H"), help="output resolution (default 1920x1080)")
        sp.add_argument("--window", nargs=4, type=float, metavar=("X", "Y", "W", "H"),
                        help="game area in screen pixels")
        sp.add_argument("--game-size", nargs=2, type=float, metavar=("W", "H"),
                        help="letterbox a game of this size into the output")
        sp.add_argument("--capacity", type=int, default=10, help="physical MT slots (default 10)")
        sp.add_argument("--device", default=DEFAULT_DEVICE, help="uinput device name")

    def plan_opts(sp):
        sp.add_argument("--fps", type=float, default=60.0)
        sp.add_argument("--tap-frames", type=int, default=2)
        sp.add_argument("--flick-frames", type=int, default=3)
        sp.add_argument("--flick-dy", type=float, default=0.20)
        sp.add_argument("--hold-grace", type=int, default=2)
        sp.add_argument("--hold-tracking", choices=["pinned", "follow", "judgment_line"],
                        default="pinned")

    sp = sub.add_parser("probe", help="check /dev/uinput and create/remove a test device")
    common(sp)
    sp.set_defaults(func=cmd_probe)

    sp = sub.add_parser("list", help="list charts in the label set")
    common(sp)
    sp.add_argument("--query", default="")
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("plan", help="show the compiled action timeline")
    common(sp)
    plan_opts(sp)
    sp.add_argument("--chart", default=None)
    sp.add_argument("--show", type=int, default=15)
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("replay", help="play a chart through a backend")
    common(sp)
    plan_opts(sp)
    sp.add_argument("--chart", default=None)
    sp.add_argument("--backend", choices=["dryrun", "uinput"], default="dryrun")
    sp.add_argument("--speed", type=float, default=1.0,
                    help="1=realtime, 0=as fast as possible (dryrun default 0)")
    sp.add_argument("--duration", type=float, default=0.0, help="stop after N seconds of chart")
    sp.add_argument("--policy", choices=["drop", "steal_lru"], default="steal_lru")
    sp.add_argument("--lead-in", type=float, default=3.0)
    sp.add_argument("--dump", help="write recorded ops to this JSONL file")
    sp.set_defaults(func=cmd_replay)

    sp = sub.add_parser("watch", help="play a chart in real time with a live 4-finger visualizer")
    common(sp)
    plan_opts(sp)
    sp.add_argument("--chart", default=None)
    sp.add_argument("--backend", choices=["dryrun", "uinput"], default="dryrun")
    sp.add_argument("--viz", choices=["none", *sorted(RENDERERS)], default="terminal",
                    help="renderer to attach (default terminal)")
    sp.add_argument("--viz-fps", type=float, default=60.0, help="renderer frame rate")
    sp.add_argument("--viz-ascii", action="store_true", help="ASCII glyphs in the terminal renderer")
    sp.add_argument("--speed", type=float, default=1.0,
                    help="1=realtime; the visualizer is only meaningful in realtime")
    sp.add_argument("--duration", type=float, default=0.0, help="stop after N seconds of chart")
    sp.add_argument("--policy", choices=["drop", "steal_lru"], default="steal_lru")
    sp.add_argument("--lead-in", type=float, default=1.0,
                    help="seconds to wait before the first frame (lets the window appear)")
    sp.set_defaults(func=cmd_watch)

    sp = sub.add_parser("selftest", help="press/move/release on the real uinput device")
    common(sp)
    sp.add_argument("--repeat", type=int, default=1)
    sp.set_defaults(func=cmd_selftest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # `watch` is about watching, so a dry run must stay in realtime there.
    if getattr(args, "command", "") != "watch":
        if getattr(args, "backend", "") == "dryrun" and getattr(args, "speed", None) == 1.0:
            args.speed = 0.0
    try:
        return args.func(args)
    except LookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("hint: use `list --query ...` to see matching chart names", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
