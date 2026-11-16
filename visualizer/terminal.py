"""ANSI terminal visualizer - works over SSH, in tmux, and with no GUI at all.

Run it directly (it reads NDJSON frames on stdin, written by
``python3 -m beatflow_input watch``)::

    python3 -m beatflow_input watch --chart "Glitch" --viz terminal

or watch the built-in demo, which is the quickest way to see the three phases
without a chart or a device::

    python3 -m beatflow_input.visualizer.terminal --demo

The same :class:`~beatflow_input.visualizer.scene.Scene` drives the Tk window,
so what you learn to read here transfers to ``--viz tk``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from typing import Iterator, Mapping, MutableMapping

from .protocol import ContactState, FrameState, parse
from .scene import JUDGMENT_LINE_Y, PALETTE, Scene, Viewport

__all__ = ["TerminalCanvas", "render_frame", "build_screen", "demo_frames", "main"]

# Glyph sets. Unicode looks better; ASCII is the fallback for a console that
# mangles box drawing (Windows conhost, some serial terminals).
GLYPHS_UNICODE = {
    "contact_down": "●",
    "contact_hold": "◉",
    "contact_flick": "✦",
    "line": "─",
    "lane": "┊",
    "cross": "┼",
}
GLYPHS_ASCII = {
    "contact_down": "o",
    "contact_hold": "@",
    "contact_flick": "*",
    "line": "-",
    "lane": "|",
    "cross": "+",
}

RESET = "\x1b[0m"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
CURSOR_HOME = "\x1b[H"
CLEAR_TO_END = "\x1b[J"
CLEAR_SCREEN = "\x1b[2J"

DIM = (92, 104, 124)


def _enable_ansi() -> bool:
    """Turn on VT processing on Windows consoles; no-op elsewhere."""
    if os.name != "nt":
        return True
    try:  # pragma: no cover - Windows only
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:  # pragma: no cover
        return False


def _fg(color: tuple[int, int, int]) -> str:
    return f"\x1b[38;2;{color[0]};{color[1]};{color[2]}m"


def _prepare_stdout() -> None:
    """Prefer UTF-8 so the box-drawing glyphs survive a redirected/piped stdout.

    A Windows console defaults to cp936/cp1252, which turns ``◉`` into a
    UnicodeEncodeError halfway through a frame; that is a crash, not a
    cosmetic issue, so it is fixed before the first frame.
    """
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower().replace("-", "")
    if encoding in ("utf8", ""):
        return
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover - exotic stdout
        pass


def _encodable(glyphs: dict[str, str]) -> bool:
    """True if every glyph survives the current stdout encoding."""
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        for ch in glyphs.values():
            ch.encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def _scale(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    f = 0.0 if factor < 0.0 else 1.0 if factor > 1.0 else factor
    return (int(color[0] * f), int(color[1] * f), int(color[2] * f))


class TerminalCanvas:
    """A character grid with a per-cell colour, projected from game space."""

    def __init__(self, width: int, height: int, viewport: Viewport | None = None,
                 glyphs: Mapping[str, str] | None = None) -> None:
        self.w = max(20, width)
        self.h = max(6, height)
        self.glyphs = glyphs or GLYPHS_UNICODE
        self.viewport = viewport or Viewport.for_grid(self.w, self.h)
        self.chars = [[" "] * self.w for _ in range(self.h)]
        self.colors: list[list[tuple[int, int, int] | None]] = [[None] * self.w for _ in range(self.h)]

    # -- projection --------------------------------------------------------
    def project(self, x: float, y: float) -> tuple[int, int]:
        u, v = self.viewport.to_unit(x, y)
        col = int(round(u * (self.w - 1)))
        row = int(round(v * (self.h - 1)))
        return (min(self.w - 1, max(0, col)), min(self.h - 1, max(0, row)))

    def row_of(self, y: float) -> int:
        return self.project(0.0, y)[1]

    # -- primitives --------------------------------------------------------
    def put(self, col: int, row: int, ch: str, color: tuple[int, int, int] | None) -> None:
        if 0 <= col < self.w and 0 <= row < self.h:
            self.chars[row][col] = ch
            if color is not None:
                self.colors[row][col] = color

    def put_world(self, x: float, y: float, ch: str, color) -> None:
        self.put(*self.project(x, y), ch, color)

    # -- output ------------------------------------------------------------
    def to_text(self, use_color: bool = True) -> str:
        out: list[str] = []
        for row in range(self.h):
            if not use_color:
                out.append("".join(self.chars[row]))
                continue
            line: list[str] = [RESET]
            current = None
            for col in range(self.w):
                color = self.colors[row][col]
                if color != current:
                    line.append(_fg(color) if color is not None else _fg(DIM))
                    current = color
                line.append(self.chars[row][col])
            line.append(RESET)
            out.append("".join(line))
        return "\n".join(out)


def render_frame(scene: Scene, canvas: TerminalCanvas) -> None:
    """Draw ``scene`` into ``canvas`` (no stdout side effects)."""
    g = canvas.glyphs
    line_y = scene.judgement_line_y

    # 1. guides ----------------------------------------------------------
    line_row = canvas.row_of(line_y)
    for col in range(canvas.w):
        canvas.put(col, line_row, g["line"], _scale(DIM, 0.85))
    canvas.put(*canvas.project(0.0, line_y), g["cross"], DIM)
    for lane in (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6):
        canvas.put(canvas.project(lane, line_y)[0], max(0, line_row - 1), g["lane"], _scale(DIM, 0.55))

    # 2. the fingers themselves -----------------------------------------
    for c in scene.contacts:
        canvas.put_world(c.x, c.y, g[f"contact_{c.phase}"], c.color)


def build_screen(
    scene: Scene,
    *,
    now: float,
    name: str,
    width: int,
    height: int,
    glyphs: Mapping[str, str],
    use_color: bool,
    duration: float = 0.0,
) -> str:
    """Compose the full screen: HUD row, play field, legend row."""
    viewport = Viewport.for_grid(width, height)
    canvas = TerminalCanvas(width, height, viewport, glyphs)
    render_frame(scene, canvas)

    counts = scene.phase_counts()
    clock = f"{now:6.2f}s" + (f" / {duration:.1f}s" if duration else "")
    hud = f" {name}"[: max(0, width - 34)] + f"  {clock}  fingers={len(scene.contacts)}".rjust(28)
    hud = hud[:width].ljust(width)

    legend = (
        f"down {glyphs['contact_down']} {counts['down']}   "
        f"hold {glyphs['contact_hold']} {counts['hold']}   "
        f"flick {glyphs['contact_flick']} {counts['flick']}   (colour = finger)"
    )[:width]

    body = canvas.to_text(use_color=use_color).split("\n")
    if not use_color:
        return "\n".join([hud, *body, legend])

    swatches = "".join(_fg(color) + "●" + RESET for color in PALETTE)
    header = f"{_fg(DIM)}{hud}{RESET}"
    footer = f"{_fg(DIM)}{legend[: max(0, width - 6)]}{RESET} {swatches}"
    return "\n".join([header, *body, footer])


def demo_frames(duration: float = 8.0, fps: float = 60.0) -> Iterator[FrameState]:
    """A synthetic 4-finger routine that exercises all three phases.

    Shape-identical to real traffic (``FrameState`` in, renderer out), so the
    demo cannot drift away from what ``watch`` sends. Finger 1 holds, finger 2
    flicks, fingers 0 and 3 tap on and off the beat, and every finger is on
    screen at once around t=1.0.
    """
    beat = 0.5  # 120 BPM
    holds = ((0.5, 2.2, -0.2), (3.0, 4.6, 0.2))
    total = int(duration * fps)
    for i in range(total + 1):
        t = i / fps
        contacts: list[ContactState] = []

        phase_t = t % beat
        if phase_t < 0.14:
            contacts.append(ContactState(key=1, x=-0.6, y=0.40, slot=0, age=phase_t))

        for start, end, x in holds:
            if start <= t < end:
                contacts.append(ContactState(key=2, x=x, y=JUDGMENT_LINE_Y, slot=1, age=t - start))
                break

        # finger 2 flicks: press low, slide up-right over ~0.14 s
        flick_t = (t - 1.2) % 1.5
        if t >= 1.2 and flick_t < 0.14:
            d = flick_t
            contacts.append(
                ContactState(key=3, x=0.2 + 1.4 * d, y=0.42 + 0.4 * d, slot=2,
                             vx=1.4, vy=0.4, age=d)
            )

        offbeat = (t + beat / 2.0) % beat
        if 2.0 <= t and offbeat < 0.12:
            contacts.append(ContactState(key=4, x=0.6, y=0.30, slot=3, age=offbeat))

        yield FrameState(t=t, contacts=tuple(contacts), done=(i == total))


def _terminal_size(args) -> tuple[int, int]:
    if args.width and args.height:
        return args.width, args.height
    size = shutil.get_terminal_size(fallback=(100, 32))
    return (args.width or size.columns), (args.height or size.lines)


def _run_stream(
    frames: Iterator[FrameState],
    meta: MutableMapping[str, str],
    glyphs: Mapping[str, str],
    args,
    use_color: bool,
    duration: float,
) -> int:
    """Draw ``frames`` until they end; ``meta['name']`` may change mid-stream."""
    scene = Scene(judgement_line_y=JUDGMENT_LINE_Y)
    if use_color:
        sys.stdout.write(HIDE_CURSOR + CLEAR_SCREEN)
    draw_interval = 1.0 / max(1.0, args.fps)
    last_draw = 0.0
    drawn = 0
    cols, rows = _terminal_size(args)
    chart_end = duration
    try:
        for st in frames:
            if st is None:
                continue
            # Always ingest: dropping a state update could lose a press/release.
            scene.update(st.t, st.contacts)
            chart_end = max(chart_end, st.t)
            now = time.monotonic()
            if drawn and (now - last_draw) < draw_interval and not st.done:
                continue
            last_draw = now

            screen = build_screen(
                scene, now=st.t, name=meta["name"], width=cols, height=max(6, rows - 2),
                glyphs=glyphs, use_color=use_color, duration=chart_end,
            )
            if use_color:
                sys.stdout.write(CURSOR_HOME + screen + CLEAR_TO_END)
            else:
                if drawn:
                    sys.stdout.write("\n")
                sys.stdout.write(screen + "\n")
            sys.stdout.flush()
            drawn += 1
            if args.frames and drawn >= args.frames:
                break
            if st.done:
                break
    except KeyboardInterrupt:
        return 130
    finally:
        if use_color:
            sys.stdout.write(SHOW_CURSOR + "\n")
            sys.stdout.flush()
    return 0


def _stdin_frames(meta: MutableMapping[str, str]) -> Iterator[FrameState]:
    """Blocking NDJSON reader; ends when the producer closes stdin."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = parse(line)
        except ValueError as exc:
            print(f"visualizer: bad frame ({exc})", file=sys.stderr)
            continue
        kind = msg.get("kind")
        if kind == "hello":
            meta["name"] = str(msg.get("name", ""))
            continue
        if kind == "bye":
            return
        if kind != "frame":
            continue
        yield FrameState.from_dict(msg)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="beatflow_input.visualizer.terminal",
                                description="ASCII/ANSI multi-touch visualizer")
    p.add_argument("--demo", action="store_true", help="play a synthetic 4-finger routine")
    p.add_argument("--demo-seconds", type=float, default=8.0)
    p.add_argument("--fps", type=float, default=60.0, help="max redraw rate")
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--height", type=int, default=0)
    p.add_argument("--title", default="", help="name shown in the HUD")
    p.add_argument("--ascii", action="store_true", help="ASCII glyphs only")
    p.add_argument("--no-color", action="store_true", help="plain text (logs and tests)")
    p.add_argument("--frames", type=int, default=0, help="stop after N drawn frames")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _prepare_stdout()
    # Colour needs no negotiation on Linux; a real Windows console has to be
    # switched into VT mode first, but a redirected stream keeps the escapes
    # (that is what makes `... | less -R` work).
    use_color = not args.no_color
    if use_color and os.name == "nt" and sys.stdout.isatty():
        use_color = _enable_ansi()
    glyphs = GLYPHS_ASCII if args.ascii else GLYPHS_UNICODE
    if glyphs is GLYPHS_UNICODE and not _encodable(glyphs):
        glyphs = GLYPHS_ASCII

    meta = {"name": args.title or "BeatFlow"}
    if args.demo:
        meta["name"] = args.title or "BeatFlow demo"
        return _run_stream(demo_frames(args.demo_seconds), meta, glyphs, args, use_color,
                           args.demo_seconds)

    if sys.stdin.isatty():
        print(
            "terminal visualizer: no frames on stdin.\n"
            "  try:  python3 -m beatflow_input.visualizer.terminal --demo\n"
            "  or :  python3 -m beatflow_input watch --chart '...' --viz terminal",
            file=sys.stderr,
        )
        return 2

    return _run_stream(_stdin_frames(meta), meta, glyphs, args, use_color, 0.0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
