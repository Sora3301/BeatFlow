"""Recording backend: no kernel involvement, used for replay, CI and timing.

``DryRunBackend`` is the reference implementation of the backend contract. It
records every op with a monotonic timestamp so a run can be

* diffed against expectations (regression test for the action interpreter),
* replayed later through a real backend,
* used to measure scheduler accuracy without touching the desktop.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .base import TouchBackend

__all__ = ["DryRunBackend", "RecordedOp"]


@dataclass(frozen=True)
class RecordedOp:
    t: float              # time.monotonic() at the moment the op was queued
    kind: str             # press | move | release | commit
    slot: int = -1
    tracking_id: int = -1
    x: int = 0
    y: int = 0

    def as_dict(self) -> dict:
        return {
            "t": round(self.t, 9),
            "kind": self.kind,
            "slot": self.slot,
            "tracking_id": self.tracking_id,
            "x": self.x,
            "y": self.y,
        }


class DryRunBackend(TouchBackend):
    """Collects ops in memory; ``commit()`` closes a frame."""

    name = "dryrun"

    def __init__(self, capacity: int = 10) -> None:
        self._capacity = capacity
        self.ops: list[RecordedOp] = []
        self.frames: list[list[RecordedOp]] = []
        self._frame: list[RecordedOp] = []
        self._active: set[int] = set()
        self._pos: dict[int, tuple[int, int]] = {}
        self._open = False

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        if self._active:
            for slot in sorted(self._active):
                self.release(slot)
            self.commit()
        self._open = False

    def press(self, slot: int, tracking_id: int, x: int, y: int) -> None:
        if slot in self._active:
            raise AssertionError(f"press on already-active slot {slot}")
        op = RecordedOp(time.monotonic(), "press", slot, tracking_id, x, y)
        self._frame.append(op)
        self._active.add(slot)
        self._pos[slot] = (x, y)

    def move(self, slot: int, x: int, y: int) -> None:
        if slot not in self._active:
            raise AssertionError(f"move on inactive slot {slot}")
        if self._pos.get(slot) == (x, y):
            return
        self._frame.append(RecordedOp(time.monotonic(), "move", slot, -1, x, y))
        self._pos[slot] = (x, y)

    def release(self, slot: int) -> None:
        if slot not in self._active:
            raise AssertionError(f"release on inactive slot {slot}")
        self._frame.append(RecordedOp(time.monotonic(), "release", slot, -1))
        self._active.discard(slot)
        self._pos.pop(slot, None)

    def commit(self) -> None:
        stamp = RecordedOp(time.monotonic(), "commit")
        self._frame.append(stamp)
        self.frames.append(self._frame)
        self.ops.extend(self._frame)
        self._frame = []

    # -- inspection --------------------------------------------------------
    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for op in self.ops:
                fh.write(json.dumps(op.as_dict(), ensure_ascii=False) + "\n")

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for op in self.ops:
            counts[op.kind] = counts.get(op.kind, 0) + 1
        return {
            "backend": self.name,
            "frames": len(self.frames),
            "ops": len(self.ops),
            "by_kind": counts,
            "max_simultaneous": self._max_simultaneous(),
        }

    def _max_simultaneous(self) -> int:
        live = 0
        peak = 0
        for op in self.ops:
            if op.kind == "press":
                live += 1
                peak = max(peak, live)
            elif op.kind == "release":
                live -= 1
        return peak
