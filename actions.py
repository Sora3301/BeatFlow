"""Label events -> an injectable action timeline.

The labels are a *sparse description of what the chart does*, not a finger
recording: there is no ``up`` phase, tap notes appear for a single frame, hold
notes repeat every frame, and a flick is one frame with no direction. Turning
that into touch events requires a small state machine, which is what this
module is - **the only place where release synthesis happens**.

Model
-----
For every frame the interpreter derives the *set of contacts that should be
down* (``key -> (x, y, epoch)``). Consecutive sets are diffed to produce
press / move / release ops. Diffing instead of emitting on sight makes the
interpreter robust to gaps: a dropped frame simply produces no ops, and an id
that vanishes produces exactly one release.

Policies you can tune (see :class:`InterpreterConfig`) because they encode
guesses, not facts:

* ``tap_frames`` - how long a tap stays pressed (default 2 frames ≈ 33 ms at
  60 Hz; the game reads state per frame, 1 frame is enough in theory but 2 is
  forgiving of a one-frame aliasing error);
* ``flick_frames`` / ``flick_delta`` - a flick is press -> move -> release; the
  direction is **not in the data**, so a default is used and must be validated
  visually;
* ``hold_grace_frames`` - a hold ends when its id stops appearing; the true
  release frame is unknown (the game likely releases 1-3 frames later);
* ``hold_tracking`` - the recorded ``y`` of a hold id climbs monotonically
  (0.059 -> 0.375 in the sample) while ``x`` stays fixed. That is the *note*
  moving, not the finger. Default ``"pinned"`` freezes the contact where it
  landed, which is what a real finger does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from .labels import Chart, LabelEvent

__all__ = [
    "Op",
    "Commit",
    "Timeline",
    "InterpreterConfig",
    "build_timeline",
    "diff_ops",
    "flick_offset",
    "JUDGMENT_LINE_Y",
]

#: Phigros judgment line in game-normalised space (empirical mode of ``y``).
JUDGMENT_LINE_Y = 0.375

HoldTracking = Literal["pinned", "follow", "judgment_line"]


@dataclass(frozen=True)
class Op:
    """One low-level intent. ``key`` is the logical note id from the labels."""

    kind: str          # "press" | "move" | "release"
    key: int
    x: float = 0.0     # game-normalised, only meaningful for press/move
    y: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("press", "move", "release"):
            raise ValueError(f"bad op kind {self.kind!r}")


@dataclass(frozen=True)
class Commit:
    """All ops scheduled for one frame boundary."""

    frame: int
    t: float                    # seconds from chart start
    ops: tuple[Op, ...]

    @property
    def is_empty(self) -> bool:
        return not self.ops


@dataclass
class Timeline:
    commits: list[Commit]
    fps: float
    end_frame: int

    @property
    def duration(self) -> float:
        return self.end_frame / self.fps

    @property
    def n_ops(self) -> int:
        return sum(len(c.ops) for c in self.commits)

    @property
    def n_presses(self) -> int:
        return sum(1 for c in self.commits for o in c.ops if o.kind == "press")

    def head(self, n: int = 20) -> list[Commit]:
        return self.commits[:n]

    def summary(self) -> dict:
        kinds: dict[str, int] = {}
        for c in self.commits:
            for o in c.ops:
                kinds[o.kind] = kinds.get(o.kind, 0) + 1
        return {
            "commits": len(self.commits),
            "ops": self.n_ops,
            "by_kind": kinds,
            "duration_s": round(self.duration, 3),
            "fps": self.fps,
        }


@dataclass
class InterpreterConfig:
    fps: float = 60.0
    tap_frames: int = 2
    flick_frames: int = 3
    flick_delta: tuple[float, float] = (0.0, 0.20)
    hold_grace_frames: int = 2
    hold_tracking: HoldTracking = "pinned"
    drop_bad: bool = True
    time_offset: float = 0.0

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.tap_frames < 1 or self.flick_frames < 2:
            raise ValueError("tap_frames >= 1 and flick_frames >= 2 required")
        if self.hold_tracking not in ("pinned", "follow", "judgment_line"):
            raise ValueError(f"bad hold_tracking {self.hold_tracking!r}")


@dataclass
class _Contact:
    x: float
    y: float
    epoch: int
    kind: str
    end: int
    move_at: int | None = None
    move_to: tuple[float, float] | None = None


def diff_ops(
    prev: Mapping[int, tuple[float, float, int]],
    cur: Mapping[int, tuple[float, float, int]],
    ops: list[Op],
) -> None:
    """Turn two contact snapshots into press / move / release ops.

    Public because the real-time model path
    (:class:`beatflow_input.model_api.ModelStream`) must produce *identical*
    ops to the offline path; sharing this function is what keeps them honest.
    """
    for key in prev:
        if key not in cur:
            ops.append(Op("release", key))
    for key, (x, y, epoch) in cur.items():
        old = prev.get(key)
        if old is None:
            ops.append(Op("press", key, x, y))
        elif old[2] != epoch:
            ops.append(Op("release", key))
            ops.append(Op("press", key, x, y))
        elif (old[0], old[1]) != (x, y):
            ops.append(Op("move", key, x, y))



def flick_offset(
    direction: tuple[float, float] | None,
    cfg: InterpreterConfig,
) -> tuple[float, float]:
    """Displacement for one flick.

    A predicted or labelled direction is normalised so that every flick swipes
    the same distance (``|cfg.flick_delta|``); only the heading changes. With
    no direction at all we fall back to ``cfg.flick_delta`` - the same guess the
    dataset path makes, because the label set has no flick heading either.
    """
    base_x, base_y = cfg.flick_delta
    length = math.hypot(base_x, base_y)
    if direction is None:
        return (base_x, base_y)
    dx, dy = float(direction[0]), float(direction[1])
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return (base_x, base_y)
    return (dx / norm * length, dy / norm * length)


def build_timeline(chart: Chart, cfg: InterpreterConfig | None = None) -> Timeline:
    """Compile a :class:`~beatflow_input.labels.Chart` into press/move/release ops."""
    cfg = cfg or InterpreterConfig()
    if not chart.frames:
        return Timeline([], cfg.fps, 0)

    contacts: dict[int, _Contact] = {}
    state: dict[int, tuple[float, float, int]] = {}
    commits: list[Commit] = []

    first = chart.first_frame
    last_event = chart.last_frame
    # Run past the last event so pending releases still get emitted.
    horizon = last_event + max(cfg.tap_frames, cfg.flick_frames, cfg.hold_grace_frames) + 2

    for frame in range(first, horizon + 1):
        events: Sequence[LabelEvent] = chart.frames.get(frame, ())

        for e in sorted(events, key=lambda ev: ev.id):
            if cfg.drop_bad and e.bad:
                continue
            if e.phase == "down":
                prev = contacts.get(e.id)
                epoch = 1 if prev is None else prev.epoch + 1
                contacts[e.id] = _Contact(e.x, e.y, epoch, "tap", frame + cfg.tap_frames)
            elif e.phase == "hold":
                c = contacts.get(e.id)
                if c is None:
                    contacts[e.id] = _Contact(e.x, e.y, 1, "hold", frame + cfg.hold_grace_frames)
                else:
                    # A hold note is emitted as ``down`` on its first frame and
                    # ``hold`` afterwards, so the common case here is a tap
                    # contact that must be *upgraded*, not re-pressed.
                    if c.kind != "hold":
                        c.kind = "hold"
                        c.move_at = None
                        c.move_to = None
                    c.end = frame + cfg.hold_grace_frames
                    if cfg.hold_tracking == "follow":
                        c.x, c.y = e.x, e.y
                    elif cfg.hold_tracking == "judgment_line":
                        c.y = JUDGMENT_LINE_Y
            elif e.phase == "flick":
                prev = contacts.get(e.id)
                epoch = 1 if prev is None else prev.epoch + 1
                dx, dy = flick_offset(e.direction, cfg)
                contacts[e.id] = _Contact(
                    e.x,
                    e.y,
                    epoch,
                    "flick",
                    frame + cfg.flick_frames,
                    move_at=frame + 1,
                    move_to=(e.x + dx, e.y + dy),
                )

        for c in contacts.values():
            if c.move_at == frame and c.move_to is not None:
                c.x, c.y = c.move_to
                c.move_to = None
                c.move_at = None

        for key in [k for k, c in contacts.items() if c.end <= frame]:
            del contacts[key]

        cur = {k: (c.x, c.y, c.epoch) for k, c in contacts.items()}
        if cur != state:
            ops: list[Op] = []
            diff_ops(state, cur, ops)
            if ops:
                commits.append(
                    Commit(frame=frame, t=frame / cfg.fps + cfg.time_offset, ops=tuple(ops))
                )
            state = cur

    return Timeline(commits, cfg.fps, horizon)
