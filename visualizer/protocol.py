"""Wire format between the injection process and a visualizer process.

Why a separate process
----------------------
The injector runs against absolute deadlines with a busy-spin tail; a renderer
that borrows the 16.7 ms frame budget shows up as late commits. Tk also insists
on owning its main thread while :class:`~beatflow_input.scheduler.TimelineRunner`
spins. Both problems disappear if the renderer is its own process fed through a
pipe:

    beatflow_input watch  --(NDJSON on stdin)-->  visualizer.terminal
                                                  visualizer.tkview

The parent therefore only ever pays for ``json.dumps`` plus one pipe write per
frame, which is tens of microseconds.

Frames are *state*, not events: the renderer diffs consecutive ``contacts``
lists to detect presses and releases, so a dropped or coalesced frame can never
leave a ghost finger on screen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

__all__ = [
    "PROTOCOL_VERSION",
    "PHASES",
    "ContactState",
    "FrameState",
    "encode",
    "parse",
    "hello",
    "bye",
]

#: Bumped whenever a field changes meaning. Renderers reject mismatches loudly
#: instead of drawing nonsense.
PROTOCOL_VERSION = 1

#: ``auto`` means "no label phase available" - the renderer infers the phase
#: from lifetime and velocity (that is the only option for model predictions).
PHASES = ("down", "hold", "flick", "auto")


@dataclass(frozen=True)
class ContactState:
    """One finger that should be down right now."""

    key: int
    x: float
    y: float
    phase: str = "auto"
    age: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    slot: int = -1

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": int(self.key),
            "x": round(float(self.x), 5),
            "y": round(float(self.y), 5),
            "phase": self.phase if self.phase in PHASES else "auto",
            "age": round(float(self.age), 4),
            "vx": round(float(self.vx), 4),
            "vy": round(float(self.vy), 4),
            "slot": int(self.slot),
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ContactState":
        return cls(
            key=int(row.get("key", 0)),
            x=float(row.get("x", 0.0)),
            y=float(row.get("y", 0.0)),
            phase=str(row.get("phase", "auto")),
            age=float(row.get("age", 0.0)),
            vx=float(row.get("vx", 0.0)),
            vy=float(row.get("vy", 0.0)),
            slot=int(row.get("slot", -1)),
        )


@dataclass(frozen=True)
class FrameState:
    """Everything on screen at one instant."""

    t: float
    contacts: tuple[ContactState, ...] = ()
    done: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "t": round(float(self.t), 4),
            "contacts": [c.as_dict() for c in self.contacts],
            "done": bool(self.done),
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "FrameState":
        return cls(
            t=float(row.get("t", 0.0)),
            contacts=tuple(ContactState.from_dict(c) for c in row.get("contacts", ())),
            done=bool(row.get("done", False)),
        )


def encode(obj: dict[str, Any]) -> str:
    """One NDJSON line."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n"


def parse(line: str) -> dict[str, Any]:
    obj = json.loads(line)
    if not isinstance(obj, dict):
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
    if "kind" not in obj:
        raise ValueError("message without a 'kind' field")
    return obj


def hello(
    name: str,
    fps: float = 60.0,
    judgement_line_y: float = 0.375,
    capacity: int = 10,
    style: str = "default",
) -> dict[str, Any]:
    return {
        "kind": "hello",
        "v": PROTOCOL_VERSION,
        "name": name,
        "fps": float(fps),
        "judgement_line_y": float(judgement_line_y),
        "capacity": int(capacity),
        "style": style,
    }


def bye(**stats: Any) -> dict[str, Any]:
    return {"kind": "bye", "v": PROTOCOL_VERSION, "stats": stats}
