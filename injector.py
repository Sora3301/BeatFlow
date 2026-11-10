"""Turns a :class:`~beatflow_input.actions.Commit` into backend calls.

Responsibilities, in order:

1. bind logical note ids to physical MT slots (:mod:`beatflow_input.slots`);
2. convert game-normalised coordinates to device units
   (:mod:`beatflow_input.geometry`);
3. apply the ops **in list order** and close the frame with ``commit()``.

Batch-per-frame instead of per-op matters: a commit writes one
``SYN_REPORT``, so a frame with three notes costs one syscall and the kernel
sees them as simultaneous rather than 0.2 ms apart.
"""

from __future__ import annotations

from dataclasses import dataclass

from .actions import Commit, Op
from .backends.base import TouchBackend
from .geometry import Geometry
from .slots import SlotAllocator

__all__ = ["Injector", "InjectionStats"]


@dataclass
class InjectionStats:
    commits: int = 0
    presses: int = 0
    moves: int = 0
    releases: int = 0
    dropped: int = 0
    stolen: int = 0
    max_simultaneous: int = 0

    def as_dict(self) -> dict:
        return {
            "commits": self.commits,
            "presses": self.presses,
            "moves": self.moves,
            "releases": self.releases,
            "dropped_on_overflow": self.dropped,
            "stolen_on_overflow": self.stolen,
            "max_simultaneous": self.max_simultaneous,
        }


class Injector:
    """Stateful op applier; one instance per run."""

    def __init__(
        self,
        backend: TouchBackend,
        geometry: Geometry,
        capacity: int | None = None,
        policy: str = "steal_lru",
        tracking_id_start: int = 1,
    ) -> None:
        self.backend = backend
        self.geometry = geometry
        cap = capacity if capacity is not None else backend.capacity
        self.allocator = SlotAllocator(cap, policy=policy, tracking_id_start=tracking_id_start)
        self.stats = InjectionStats()

    def apply(self, commit: Commit) -> None:
        for op in commit.ops:
            if op.kind == "release":
                slot = self.allocator.release(op.key)
                if slot is not None:
                    self.backend.release(slot)
                    self.stats.releases += 1
            elif op.kind == "press":
                self._press(op)
            elif op.kind == "move":
                slot = self.allocator.slot_of(op.key)
                if slot is not None:
                    x, y = self.geometry.to_device(op.x, op.y)
                    self.backend.move(slot, x, y)
                    self.stats.moves += 1
        self.backend.commit()
        self.stats.commits += 1
        self.stats.max_simultaneous = max(self.stats.max_simultaneous, len(self.allocator))

    def _press(self, op: Op) -> None:
        # Defensive: a press for an already-live key is a reposition, not a
        # new contact. The interpreter should never emit this.
        if self.allocator.is_active(op.key):
            slot = self.allocator.slot_of(op.key)
            x, y = self.geometry.to_device(op.x, op.y)
            self.backend.move(slot, x, y)  # type: ignore[arg-type]
            self.stats.moves += 1
            return

        assignment = self.allocator.acquire(op.key)
        if assignment is None:
            self.stats.dropped += 1
            return
        if assignment.evicted_slot is not None:
            self.backend.release(assignment.evicted_slot)
            self.stats.stolen += 1
        x, y = self.geometry.to_device(op.x, op.y)
        self.backend.press(assignment.slot, assignment.tracking_id, x, y)
        self.stats.presses += 1

    def release_all(self) -> None:
        """Lift every remaining contact (used on shutdown / abort)."""
        for key in list(self.allocator.active_keys()):
            slot = self.allocator.release(key)
            if slot is not None:
                self.backend.release(slot)
        self.backend.commit()
