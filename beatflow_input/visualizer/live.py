"""Producer side: mirror what the injector is actually doing and stream it out.

Two pieces:

:class:`LiveModel`
    a thread-safe shadow of the contacts currently down. It is fed from the
    backend-facing ops *after* :meth:`Injector.apply`, and a press that the
    allocator dropped never enters it - so the picture can never claim a finger
    that the kernel never saw.

:class:`RenderProcess`
    a child process running one of the renderers, plus :class:`Streamer`, the
    thread that pushes frames at a fixed rate.

The parent pays ``json.dumps`` + one pipe write per frame (tens of
microseconds); all drawing happens in the child. That is deliberate: an
injector that misses its 16.7 ms deadline because a renderer got scheduled in
between would be a real defect, and this is the cheapest way to rule it out.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from .protocol import ContactState, FrameState, bye, encode, hello
from .scene import JUDGMENT_LINE_Y

__all__ = ["LiveModel", "RenderProcess", "Streamer", "phase_map", "RENDERERS"]

#: Renderer name -> module path. Adding a renderer is one line here.
RENDERERS: dict[str, str] = {
    "terminal": "beatflow_input.visualizer.terminal",
    "tk": "beatflow_input.visualizer.tkview",
}


def phase_map(chart) -> dict[int, str]:
    """``note id -> phase`` for a chart, preferring the strongest evidence.

    A hold note is labelled ``down`` on its first frame and ``hold`` after, so a
    plain dict would remember the weak label. Holds and flicks win.
    """
    strength = {"down": 0, "flick": 1, "hold": 2}
    out: dict[int, str] = {}
    for ev in chart.iter_events():
        if ev.bad:
            continue
        current = out.get(ev.id)
        if current is None or strength[ev.phase] > strength[current]:
            out[ev.id] = ev.phase
    return out


@dataclass
class _Held:
    """One live contact: the injector's view, timestamps in monotonic seconds."""

    slot: int
    x: float
    y: float
    born: float
    vx: float = 0.0
    vy: float = 0.0
    updated: float = 0.0


class LiveModel:
    """Thread-safe shadow of the contacts the injector currently holds."""

    def __init__(self, phases: Mapping[int, str] | None = None) -> None:
        self._lock = threading.Lock()
        self._phases = dict(phases or {})
        self._contacts: dict[int, _Held] = {}

    def press(self, key: int, slot: int, x: float, y: float, now: float) -> None:
        with self._lock:
            self._contacts[key] = _Held(slot=slot, x=x, y=y, born=now, updated=now)

    def move(self, key: int, x: float, y: float, now: float) -> None:
        with self._lock:
            contact = self._contacts.get(key)
            if contact is None:
                return
            dt = max(1e-3, now - contact.updated)
            contact.vx = (x - contact.x) / dt
            contact.vy = (y - contact.y) / dt
            contact.x, contact.y = x, y
            contact.updated = now

    def release(self, key: int, now: float) -> None:
        with self._lock:
            self._contacts.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._contacts.clear()

    def snapshot(self, now: float) -> tuple[ContactState, ...]:
        with self._lock:
            return tuple(
                ContactState(
                    key=key,
                    x=c.x,
                    y=c.y,
                    phase=self._phases.get(key, "auto"),
                    age=max(0.0, now - c.born),
                    vx=c.vx,
                    vy=c.vy,
                    slot=c.slot,
                )
                for key, c in self._contacts.items()
            )


class RenderProcess:
    """A renderer child process fed with NDJSON on stdin."""

    def __init__(self, kind: str, title: str = "BeatFlow", capacity: int = 10,
                 fps: float = 60.0, extra_args: Iterable[str] = ()) -> None:
        module = RENDERERS.get(kind)
        if module is None:
            raise ValueError(f"unknown visualizer {kind!r} (have {', '.join(sorted(RENDERERS))})")
        self.kind = kind
        self.title = title
        self._failed = False
        cmd = [sys.executable, "-m", module, "--title", title, "--fps", str(fps), *extra_args]
        try:
            self.proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                cmd,
                stdin=subprocess.PIPE,
                stdout=None,
                stderr=None,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            self._failed = True
            raise RuntimeError(f"cannot start {kind} visualizer: {exc}") from exc
        self._write(hello(title, fps=fps, judgement_line_y=JUDGMENT_LINE_Y, capacity=capacity))

    # -- io ----------------------------------------------------------------
    def _write(self, message: dict) -> bool:
        if self._failed or self.proc.stdin is None:
            return False
        if self.proc.poll() is not None:
            # The child died (no display, user closed the window). Stop trying;
            # the injection loop must not notice.
            self._failed = True
            return False
        try:
            self.proc.stdin.write(encode(message))
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            self._failed = True
            return False
        return True

    def send(self, state: FrameState) -> bool:
        """Push one frame; ``False`` once the child is gone."""
        message = state.as_dict()
        message["kind"] = "frame"
        return self._write(message)

    @property
    def failed(self) -> bool:
        return self._failed

    def fail(self) -> None:
        """Mark the child as gone; later writes become no-ops."""
        self._failed = True

    def close(self, stats: Mapping[str, object] | None = None, timeout: float = 2.0) -> None:
        if not self._failed:
            self._write(bye(**dict(stats or {})))
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self) -> "RenderProcess":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


class Streamer:
    """Pushes snapshots at a fixed rate on a daemon thread."""

    def __init__(self, model: LiveModel, process: RenderProcess,
                 now: Callable[[], float], fps: float = 60.0) -> None:
        self.model = model
        self.process = process
        self.now = now
        self.fps = max(1.0, fps)
        self.frames = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="beatflow-viz", daemon=True)

    def start(self) -> "Streamer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        period = 1.0 / self.fps
        next_at = time.monotonic()
        while not self._stop.is_set():
            try:
                wall = time.monotonic()
                # monotonic() is the same clock the model stores, so ages line up
                state = FrameState(t=self.now(), contacts=self.model.snapshot(wall))
                if not self.process.send(state):
                    return
                self.frames += 1
            except Exception as exc:  # pragma: no cover - never break the injector
                print(f"visualizer stream stopped: {exc}", file=sys.stderr)
                self.process.fail()
                return
            next_at += period
            delay = next_at - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_at = time.monotonic()

    def __enter__(self) -> "Streamer":
        return self.start()

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False
