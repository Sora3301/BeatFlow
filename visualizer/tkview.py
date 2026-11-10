"""Tk window visualizer - the pretty one.

    python3 -m beatflow_input watch --chart "Glitch" --viz tk
    python3 -m beatflow_input.visualizer.tkview --demo

Runs as its own process for two reasons: Tk owns its main thread (the scheduler
spins on absolute deadlines), and a 60 fps redraw must never be able to delay an
injection. Frames arrive as NDJSON on stdin; the window only ever reads them.

Design notes
------------
* Tk canvas has no alpha, so every fading effect blends towards the background
  colour explicitly. The background is a vertical gradient, so blending uses the
  colour local to the item's row.
* Items are drawn into one of two alternating tags and the *other* tag is
  deleted afterwards, which avoids the full-canvas flicker of ``delete("all")``
  while keeping the draw code stateless.
* The clock is interpolated from the wall clock between frames, so a stalled
  producer freezes the fingers but not the particles.
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time

from .protocol import FrameState, parse
from .scene import JUDGMENT_LINE_Y, PALETTE, Contact, Scene, Viewport

__all__ = ["TkRenderer", "main"]

BG_TOP = (11, 15, 20)
BG_BOTTOM = (5, 7, 11)
LINE_COLOR = (120, 220, 255)
GRID_COLOR = (28, 38, 50)
TEXT_COLOR = (150, 168, 190)


def _blend(fg: tuple[int, int, int], bg: tuple[int, int, int], alpha: float) -> str:
    a = 0.0 if alpha < 0.0 else 1.0 if alpha > 1.0 else alpha
    return "#%02x%02x%02x" % (
        int(bg[0] + (fg[0] - bg[0]) * a),
        int(bg[1] + (fg[1] - bg[1]) * a),
        int(bg[2] + (fg[2] - bg[2]) * a),
    )


def _hex(color: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % color


class TkRenderer:
    """Owns the window, the item pool and the frame clock."""

    def __init__(self, width: int = 1100, height: int = 660, fps: float = 60.0,
                 title: str = "BeatFlow - virtual input", demo_seconds: float = 0.0,
                 max_frames: int = 0) -> None:
        import tkinter as tk  # imported here so --help works without a display

        self.tk = tk
        self.fps = max(1.0, fps)
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg=_hex(BG_TOP))
        self.canvas = tk.Canvas(self.root, width=width, height=height,
                                bg=_hex(BG_TOP), highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.root.bind("<Escape>", lambda _e: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.width = width
        self.height = height
        self.scene = Scene(judgement_line_y=JUDGMENT_LINE_Y)

        self._queue: queue.Queue = queue.Queue(maxsize=8)
        self._closed = False
        self._eof = False
        self._eof_at = 0.0
        self._tag = "fx_a"
        self._last_t = 0.0
        self._last_wall = time.monotonic()
        self._duration = 0.0
        self._name = title
        self._frames = 0
        self._max_frames = max(0, max_frames)
        self._demo = demo_seconds > 0.0

        self._build_background()

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._closed = True
        try:
            self.root.destroy()
        except Exception:  # pragma: no cover - already gone
            pass

    def _build_background(self) -> None:
        """Static layer: gradient, lane guides, judgment line, HUD text."""
        c = self.canvas
        bands = 48
        for i in range(bands):
            t = i / (bands - 1)
            color = _blend(BG_BOTTOM, BG_TOP, 1.0 - t)
            c.create_rectangle(0, self.height * i / bands, self.width,
                               self.height * (i + 1) / bands + 1,
                               fill=color, outline="")

        self._viewport = Viewport.for_grid(self.width, self.height, char_aspect=1.0)
        c.create_line(0, self._row_of(JUDGMENT_LINE_Y), self.width, self._row_of(JUDGMENT_LINE_Y),
                      fill=_hex(LINE_COLOR), width=2)
        for lane in (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6):
            x = self._col_of(lane)
            c.create_line(x, self._row_of(JUDGMENT_LINE_Y) - 10, x,
                          self._row_of(JUDGMENT_LINE_Y),
                          fill=_hex(GRID_COLOR), width=2)
        self._hud = c.create_text(16, 10, anchor="nw", text="", fill=_hex(TEXT_COLOR),
                                  font=("Consolas", 11), tag="hud")
        self._legend = c.create_text(self.width - 16, 10, anchor="ne", text="", fill=_hex(TEXT_COLOR),
                                     font=("Consolas", 11), tag="legend")

        # static colour swatches for the four fingers, under the legend
        for i, color in enumerate(PALETTE):
            x = self.width - 16 - i * 18
            c.create_oval(x - 5, 34, x + 5, 44, fill=_hex(color), outline="", tag="swatch")

    # -- projection --------------------------------------------------------
    def _col_of(self, x: float) -> float:
        return self._viewport.to_unit(x, 0.0)[0] * (self.width - 1)

    def _row_of(self, y: float) -> float:
        return self._viewport.to_unit(0.0, y)[1] * (self.height - 1)

    def _point(self, x: float, y: float) -> tuple[float, float]:
        u, v = self._viewport.to_unit(x, y)
        return u * (self.width - 1), v * (self.height - 1)

    def _px(self, radius: float) -> float:
        """Game-space length -> pixels (the viewport is 2.0 units wide)."""
        return radius * (self.width - 1) / 2.0

    def _bg_at(self, row: float) -> tuple[int, int, int]:
        t = 0.0 if self.height <= 1 else max(0.0, min(1.0, row / (self.height - 1)))
        return (
            int(BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * t),
            int(BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * t),
            int(BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * t),
        )

    # -- frame intake ------------------------------------------------------
    def feed(self, st: FrameState) -> None:
        try:
            self._queue.put_nowait(st)
        except queue.Full:
            # Keep the newest state: a stale frame would show a ghost finger.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(st)
            except queue.Full:  # pragma: no cover
                pass

    def _drain(self) -> FrameState | None:
        latest = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self._last_t = latest.t
            self._last_wall = time.monotonic()
            self._duration = max(self._duration, latest.t)
            self.scene.update(latest.t, latest.contacts)
        return latest

    # -- drawing -----------------------------------------------------------
    def _draw_rings(self, tag: str, now: float) -> None:
        for ring in self.scene.rings:
            cx, cy = self._point(ring.x, ring.y)
            r = self._px(ring.radius(now))
            if r < 0.5:
                continue
            bg = self._bg_at(cy)
            if ring.filled:
                # soft impact flash: fades quickly, no outline
                alpha = ring.alpha(now) ** 1.6
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                        fill=_blend(ring.color, bg, 0.15 + 0.85 * alpha),
                                        outline="", tag=tag)
                continue
            color = _blend(ring.color, bg, 0.30 + 0.70 * ring.alpha(now) ** 0.6)
            width = max(1.5, ring.width * self.width / 700.0)
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    outline=color, width=width, tag=tag)

    def _draw_particles(self, tag: str, now: float) -> None:
        for p in self.scene.particles:
            x, y = p.position(now)
            cx, cy = self._point(x, y)
            r = max(0.9, 1.2 * p.size * (1.0 - 0.5 * p.progress(now)))
            bg = self._bg_at(cy)
            color = _blend(p.color, bg, 0.18 + 0.82 * (1.0 - p.progress(now)))
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    fill=color, outline="", tag=tag)

    def _draw_contact(self, tag: str, c: Contact, now: float) -> None:
        cx, cy = self._point(c.x, c.y)
        bg = self._bg_at(cy)
        base = c.effect_color

        if c.phase == "hold":
            # breathing halo + a rotating arc: unmistakably "still down"
            pulse = 2.0 * math.sin(now * math.tau * 2.2)
            r = self._px(0.052) + pulse
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    outline=_blend(base, bg, 0.85), width=2, tag=tag)
            arc_r = self._px(0.085)
            self.canvas.create_arc(
                cx - arc_r, cy - arc_r, cx + arc_r, cy + arc_r,
                start=(now * 220.0) % 360.0, extent=70.0, style="arc",
                outline=_blend(base, bg, 0.95), width=3, tag=tag,
            )
            self.canvas.create_line(cx, cy, cx, cy + self._px(0.05),
                                    fill=_blend(base, bg, 0.55), width=1, tag=tag)

        if c.phase == "flick":
            dx, dy = c.direction()
            # screen y grows downwards, so the direction flips
            sx, sy = dx, -dy
            tail = self._px(0.16)
            for i in range(1, 7):
                t = i / 6.0
                px, py = cx - sx * tail * t, cy - sy * tail * t
                rr = max(1.0, 4.2 * (1.0 - t))
                self.canvas.create_oval(px - rr, py - rr, px + rr, py + rr,
                                        fill=_blend(base, bg, 0.75 * (1.0 - t) + 0.1),
                                        outline="", tag=tag)
            head = self._px(0.055)
            self.canvas.create_polygon(
                cx + sx * head, cy + sy * head,
                cx - sx * head * 0.6 - sy * head * 0.8, cy - sy * head * 0.6 + sx * head * 0.8,
                cx - sx * head * 0.6 + sy * head * 0.8, cy - sy * head * 0.6 - sx * head * 0.8,
                fill=_blend(base, bg, 0.95), outline="", tag=tag,
            )
            return

        # down: a compact dot with a white-hot core
        r = self._px(0.020)
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                fill=_blend(base, bg, 1.0), outline="", tag=tag)
        core = max(1.0, r * 0.45)
        self.canvas.create_oval(cx - core, cy - core, cx + core, cy + core,
                                fill=_blend((255, 255, 255), bg, 0.9), outline="", tag=tag)

    def _update_hud(self, now: float) -> None:
        counts = self.scene.phase_counts()
        hud = (f"{self._name[:46]}   t={now:6.2f}s"
               + (f" / {self._duration:.1f}s" if self._duration else "")
               + f"   fingers={len(self.scene.contacts)}")
        self.canvas.itemconfigure(self._hud, text=hud)

        legend = (f"down {counts['down']}    hold {counts['hold']}    "
                  f"flick {counts['flick']}    colour = finger")
        self.canvas.itemconfigure(self._legend, text=legend)

    def _redraw(self) -> None:
        now = self._last_t + (time.monotonic() - self._last_wall)
        self.scene.tick(now)

        tag = "fx_b" if self._tag == "fx_a" else "fx_a"
        self._draw_rings(tag, now)
        self._draw_particles(tag, now)
        for c in self.scene.contacts:
            self._draw_contact(tag, c, now)
        self._update_hud(now)

        self.canvas.delete(self._tag)
        self._tag = tag

    # -- main loop ---------------------------------------------------------
    def _tick(self) -> None:
        if self._closed:
            return
        self._drain()
        self._redraw()
        self._frames += 1
        if self._max_frames and self._frames >= self._max_frames:
            self.close()
            return
        # EOF is handled here, on the Tk thread: calling Tk from the reader
        # thread is undefined behaviour, so the reader only raises a flag.
        if self._eof and not self._eof_at:
            self._eof_at = time.monotonic()
        if self._eof and self._queue.empty() and (time.monotonic() - self._eof_at) > 0.35:
            self.close()
            return
        self.root.after(int(1000 / self.fps), self._tick)

    def run(self, frames=None) -> int:
        if frames is not None:
            threading.Thread(target=self._pump, args=(frames,), daemon=True).start()
        else:
            threading.Thread(target=self._pump_stdin, daemon=True).start()
        self.root.after(int(1000 / self.fps), self._tick)
        try:
            self.root.mainloop()
        except KeyboardInterrupt:  # pragma: no cover
            self.close()
        return 0

    def _pump(self, frames) -> None:
        for st in frames:
            if self._closed:
                return
            self.feed(st)
            if self._demo:
                # reproduce 60 fps so the demo looks like the real thing
                time.sleep(1.0 / self.fps)
        self._eof = True

    def _pump_stdin(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = parse(line)
            except ValueError:
                continue
            kind = msg.get("kind")
            if kind == "hello":
                self._name = str(msg.get("name", self._name))
                continue
            if kind == "bye":
                break
            if kind == "frame":
                self.feed(FrameState.from_dict(msg))
        self._eof = True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="beatflow_input.visualizer.tkview",
                                description="Tk multi-touch visualizer")
    p.add_argument("--demo", action="store_true", help="play a synthetic 4-finger routine")
    p.add_argument("--demo-seconds", type=float, default=0.0)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--width", type=int, default=1100)
    p.add_argument("--height", type=int, default=660)
    p.add_argument("--title", default="BeatFlow - virtual input")
    p.add_argument("--frames", type=int, default=0, help="quit after N rendered frames (tests)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    demo_seconds = args.demo_seconds or (8.0 if args.demo else 0.0)
    try:
        renderer = TkRenderer(args.width, args.height, args.fps, args.title, demo_seconds,
                              max_frames=args.frames)
    except Exception as exc:  # pragma: no cover - depends on the host
        print(f"canvas visualizer unavailable: {exc}", file=sys.stderr)
        return 2
    if args.demo:
        from .terminal import demo_frames

        return renderer.run(demo_frames(demo_seconds))
    return renderer.run(None)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
