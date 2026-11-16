"""Turn an id-less stream of detections into stable contact identities.

The model predicts, per frame, *what* it sees - a list of positions and note
kinds. It does not know, and must not have to know, that the input layer needs a
stable key per contact in order to emit ``press``/``move``/``release`` instead of
a stream of independent taps.

That is this module's whole job:

    frame 40:  [(x=-0.40, y=0.375, tap), (x=0.42, y=0.375, flick)]
                   │                    │
                   ▼                    ▼
               id 0 (new)           id 1 (new)
    frame 41:  [(x=-0.39, y=0.375, tap), (x=0.42, y=0.375, flick)]
                   │                    │
               id 0 (matched, moved)  id 1 (matched)

Matching is greedy nearest-neighbour against a velocity-extrapolated position,
with a distance gate and a small penalty for kind changes. Greedy is enough:
a Phigros frame has at most ~10 contacts, and the alternative (Hungarian) buys
nothing at that size while costing clarity.

Design choices worth knowing:

* **ids are never reused.** A new contact always gets a fresh id even if it
  appears exactly where an old one died. Reusing would make the injector emit
  "move" where the game needs "release then press" (a new note), and would
  silently merge two fingers that happen to touch the same spot.
* **a track survives ``max_gap`` missing frames.** One dropped detection must
  not tear a hold into two taps.
* **kind is sticky-ish.** A contact that starts as ``tap`` and keeps being seen
  is upgraded to ``hold`` by the consumer, not here; here the latest observed
  kind simply wins, because the model may refine its own guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "KINDS",
    "KIND_ALIASES",
    "Detection",
    "Tracked",
    "TrackerConfig",
    "ContactTracker",
]

#: Canonical note kinds. ``tap``/``drag`` need a momentary touch, ``hold`` keeps
#: it down, ``flick`` is a swipe.
KINDS = ("tap", "drag", "hold", "flick")

#: What we accept from callers. ``down`` is the label-set name for a tap, so a
#: model trained on the labels will naturally emit it.
KIND_ALIASES = {"down": "tap", "press": "tap", "swipe": "flick", "hold_note": "hold"}


def normalise_kind(kind: str) -> str:
    k = str(kind).strip().lower()
    k = KIND_ALIASES.get(k, k)
    if k not in KINDS:
        raise ValueError(f"unknown contact kind {kind!r} (expected one of {KINDS})")
    return k


@dataclass(frozen=True)
class Detection:
    """One contact the model reports on a single frame. **No id.**"""

    x: float
    y: float
    kind: str = "tap"
    #: Optional swipe heading for ``flick``, ``(dx, dy)``. When absent the
    #: interpreter falls back to the configured default direction.
    direction: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not (-1.0 <= self.x <= 1.0 and -1.0 <= self.y <= 1.0):
            raise ValueError(
                f"detection out of game space: ({self.x}, {self.y}). "
                "Apply tanh or clamp on the model side - a silently clamped "
                "contact would look like a typo in the model, not in this layer."
            )
        object.__setattr__(self, "kind", normalise_kind(self.kind))


@dataclass(frozen=True)
class Tracked:
    """A detection with an identity attached."""

    id: int
    x: float
    y: float
    kind: str
    direction: tuple[float, float] | None = None
    age: int = 0          # frames this id has been alive (0 on the birth frame)
    vx: float = 0.0       # game units per frame, measured
    vy: float = 0.0


@dataclass
class TrackerConfig:
    """Matching knobs. All distances are in game units (screen is 2.0 wide)."""

    #: Max distance between a prediction and a detection to be the same finger.
    #: A 4-finger chord spacing is ~0.4, so 0.25 keeps lanes from swapping.
    max_distance: float = 0.25
    #: How many consecutive frames a track may go unmatched before it dies.
    max_gap: int = 2
    #: Distance penalty for changing kind, to prefer a same-kind explanation.
    kind_penalty: float = 0.06
    #: Weight of the velocity extrapolation (0 = pure last position).
    velocity_weight: float = 0.5
    #: First id handed out; monotonic from there.
    start_id: int = 0

    def __post_init__(self) -> None:
        if self.max_distance <= 0:
            raise ValueError("max_distance must be positive")
        if self.max_gap < 0:
            raise ValueError("max_gap must be >= 0")


@dataclass
class _Track:
    id: int
    x: float
    y: float
    kind: str
    direction: tuple[float, float] | None
    last_frame: int
    age: int = 0
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class TrackerStats:
    matched: int = 0
    created: int = 0
    expired: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"matched": self.matched, "created": self.created, "expired": self.expired}


class ContactTracker:
    """Assigns stable ids to id-less detections, frame by frame."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.stats = TrackerStats()
        self._tracks: dict[int, _Track] = {}
        self._next_id = self.config.start_id
        self._frame = -1

    # -- inspection --------------------------------------------------------
    @property
    def active_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._tracks))

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = self.config.start_id
        self._frame = -1
        self.stats = TrackerStats()

    # -- main entry point --------------------------------------------------
    def update(self, frame: int, detections: Iterable[Detection]) -> tuple[Tracked, ...]:
        """Match ``detections`` for ``frame`` onto existing tracks.

        Returns one :class:`Tracked` per detection, sorted by id, so the caller
        can iterate deterministically.
        """
        if frame <= self._frame:
            raise ValueError(f"frame must advance: got {frame} after {self._frame}")
        self._frame = frame
        dets: Sequence[Detection] = tuple(detections)

        self._expire(frame)

        assigned = self._match(dets, frame)   # detection index -> track id

        # unmatched detections become new contacts, left to right so that a
        # chord always gets the same ids for the same input.
        # NB: index the *keys* of ``assigned`` - its values are track ids, and
        # mixing the two id spaces silently steals a detection from its match.
        fresh = sorted(
            (i for i in range(len(dets)) if i not in assigned),
            key=lambda i: (dets[i].x, dets[i].y),
        )
        for i in fresh:
            det = dets[i]
            track = _Track(
                id=self._next_id,
                x=det.x,
                y=det.y,
                kind=det.kind,
                direction=det.direction,
                last_frame=frame,
            )
            self._next_id += 1
            self._tracks[track.id] = track
            self.stats.created += 1
            assigned[i] = track.id

        out = []
        for i, det in enumerate(dets):
            track = self._tracks[assigned[i]]
            out.append(
                Tracked(
                    id=track.id,
                    x=det.x,
                    y=det.y,
                    kind=det.kind,
                    direction=det.direction if det.direction is not None else track.direction,
                    age=track.age,
                    vx=track.vx,
                    vy=track.vy,
                )
            )
        return tuple(sorted(out, key=lambda t: t.id))

    # -- internals ---------------------------------------------------------
    def _expire(self, frame: int) -> None:
        for tid in [t.id for t in self._tracks.values() if frame - t.last_frame > self.config.max_gap]:
            del self._tracks[tid]
            self.stats.expired += 1

    def _predict(self, track: _Track) -> tuple[float, float]:
        w = self.config.velocity_weight
        return (track.x + track.vx * w, track.y + track.vy * w)

    def _match(self, dets: Sequence[Detection], frame: int) -> dict[int, int]:
        """``detection index -> track id`` for the pairs that match."""
        candidates: list[tuple[float, float, int, int]] = []
        for track in self._tracks.values():
            px, py = self._predict(track)
            for i, det in enumerate(dets):
                distance = math.hypot(det.x - px, det.y - py)
                if distance > self.config.max_distance:
                    continue
                score = distance + (0.0 if det.kind == track.kind else self.config.kind_penalty)
                candidates.append((score, distance, track.id, i))

        candidates.sort()
        taken_tracks: set[int] = set()
        assigned: dict[int, int] = {}
        for _score, _distance, tid, i in candidates:
            if tid in taken_tracks or i in assigned:
                continue
            taken_tracks.add(tid)
            assigned[i] = tid
            track = self._tracks[tid]
            det = dets[i]
            track.vx = det.x - track.x
            track.vy = det.y - track.y
            track.x, track.y = det.x, det.y
            track.kind = det.kind
            if det.direction is not None:
                track.direction = det.direction
            track.last_frame = frame
            track.age += 1
            self.stats.matched += 1
        return assigned
