"""Logical note id -> physical MT slot allocation.

Why this exists: the label set has notes whose ``slot`` field reaches **94**
(38 % of charts exceed 4 slots), while a real touch panel reports 5-10
simultaneous contacts. ``slot`` in the labels is a *persistence key*, not a
finger number, so it cannot be forwarded to the kernel.

:class:`SlotAllocator` keeps the mapping concrete:

* a note that appears gets a free physical slot and a fresh ``tracking_id``;
* a note that disappears frees its slot for reuse;
* when the panel is full the configured policy decides what is sacrificed.

Tracking ids matter on their own: the kernel and libinput identify a contact by
``(slot, tracking_id)``. Re-using a tracking id too quickly can look like a
"jump" instead of a release+press, hence the monotonic counter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["SlotAllocator", "Assignment", "SlotStats"]

POLICIES = ("drop", "steal_lru")


@dataclass(frozen=True)
class Assignment:
    """Result of :meth:`SlotAllocator.acquire`."""

    key: int                  # logical id from the labels
    slot: int                 # physical MT slot
    tracking_id: int
    evicted: int | None = None      # logical id that had to die to make room
    evicted_slot: int | None = None  # its physical slot; release it *first*


@dataclass
class SlotStats:
    presses: int = 0
    releases: int = 0
    dropped: int = 0
    stolen: int = 0
    peak_active: int = 0

    def as_dict(self) -> dict:
        return {
            "presses": self.presses,
            "releases": self.releases,
            "dropped_on_overflow": self.dropped,
            "stolen_on_overflow": self.stolen,
            "peak_active": self.peak_active,
        }


@dataclass
class _Entry:
    slot: int
    tracking_id: int
    seq: int


class SlotAllocator:
    """Maps logical note ids onto a fixed pool of physical slots.

    Parameters
    ----------
    capacity:
        Number of physical slots (must match the backend).
    policy:
        ``"drop"``      - refuse new contacts when full (counts a drop);
        ``"steal_lru"`` - evict the oldest live contact and reuse its slot
                          (counts a steal). Better default: Phigros is far more
                          sensitive to a *missing* note than to a stolen one,
                          and in practice >4 simultaneous notes are rare
                          (about 2 % of frames in the dataset).
    tracking_id_start:
        First tracking id; wraps at 65535.
    """

    def __init__(self, capacity: int, policy: str = "steal_lru", tracking_id_start: int = 1) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self.capacity = capacity
        self.policy = policy
        self._free: list[int] = list(range(capacity))
        self._active: dict[int, _Entry] = {}
        self._seq = 0
        self._next_tid = tracking_id_start
        self.stats = SlotStats()

    # -- introspection -----------------------------------------------------
    def __len__(self) -> int:
        return len(self._active)

    @property
    def free_slots(self) -> int:
        return len(self._free)

    def slot_of(self, key: int) -> int | None:
        entry = self._active.get(key)
        return entry.slot if entry else None

    def is_active(self, key: int) -> bool:
        return key in self._active

    def active_keys(self) -> list[int]:
        return [k for k, _ in sorted(self._active.items(), key=lambda kv: kv[1].seq)]

    # -- mutation ----------------------------------------------------------
    def acquire(self, key: int) -> Assignment | None:
        """Get a slot for ``key``.

        Returns ``None`` when the pool is exhausted and the policy is
        ``"drop"``. If ``key`` is already active the existing assignment is
        returned unchanged (idempotent).
        """
        existing = self._active.get(key)
        if existing is not None:
            return Assignment(key, existing.slot, existing.tracking_id)

        evicted: int | None = None
        evicted_slot: int | None = None
        if not self._free:
            if self.policy == "drop":
                self.stats.dropped += 1
                return None
            victim_key, victim = min(self._active.items(), key=lambda kv: kv[1].seq)
            evicted, evicted_slot = victim_key, victim.slot
            self.release(victim_key, count=False)
            self.stats.stolen += 1

        slot = self._free.pop(0)
        tid = self._next_tid
        self._next_tid = 1 if self._next_tid >= 65535 else self._next_tid + 1
        self._seq += 1
        self._active[key] = _Entry(slot, tid, self._seq)
        self.stats.presses += 1
        self.stats.peak_active = max(self.stats.peak_active, len(self._active))
        return Assignment(key, slot, tid, evicted, evicted_slot)

    def release(self, key: int, count: bool = True) -> int | None:
        """Free ``key``'s slot, returning it, or ``None`` if not active."""
        entry = self._active.pop(key, None)
        if entry is None:
            return None
        self._free.append(entry.slot)
        self._free.sort()
        if count:
            self.stats.releases += 1
        return entry.slot

    def reset(self) -> None:
        self._free = list(range(self.capacity))
        self._active.clear()
        self._seq = 0
