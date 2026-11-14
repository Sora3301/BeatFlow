"""The contract between the model and this module.

The CNN+LSTM predicts, **for every frame**, which contacts should be on the
screen. It says *what it sees* - a position and a note kind - and nothing about
identity. This module owns identity, lifetime and release synthesis.

Two ways in, same semantics
---------------------------
Real time (the one a live agent uses)::

    stream = ModelStream()
    while frame := capture():
        commit = stream.push(model(frame))     # list of RawContact, no ids
        injector.apply(commit)

Batch (tests, offline evaluation of a recorded prediction dump)::

    frames = [(0, [RawContact(-0.4, 0.375, "tap")]), (1, [...])]
    timeline = timeline_from_predictions(predictions_from_raw(frames))

``ModelStream`` and ``timeline_from_predictions`` are checked against each other
by ``tests/test_model_api.py``: for the same detections they must emit the same
ops on the same frames. That test is the only thing stopping the two paths from
drifting apart.

What the model emits
--------------------
:class:`RawContact` - ``x``, ``y``, ``kind``, and optionally a flick ``direction``.
No ``id``: :class:`~beatflow_input.tracking.ContactTracker` assigns one, stably,
from frame to frame (nearest-neighbour against a velocity-extrapolated position).

* ``x``/``y`` are game-normalised, ``[-1, 1]``, same convention as the labels.
  Out-of-range values raise instead of being clamped - a silently clamped
  contact looks like a typo in the model, not in this layer.
* ``kind`` is ``tap`` | ``drag`` | ``hold`` | ``flick``. ``down`` is accepted as
  an alias for ``tap`` because that is the label-set name.

How each kind is turned into touch events
-----------------------------------------
========================  ==================================================
``tap`` / ``drag``        press on the frame it is seen; released
                          ``tap_frames`` later. Present for several frames, it
                          becomes a hold (the model changed its mind, or the
                          note is a hold).
``hold``                  press, then keep alive for ``hold_grace_frames`` past
                          the last frame it was seen.
``flick``                 press, then a **scheduled** move one frame later
                          (that movement is what the game reads as a swipe),
                          then release at ``flick_frames``. The swipe is
                          synthesised, because a flick note is a single point
                          in both the labels and the model output.
========================  ==================================================

A flick reported on several frames triggers **once**: repeating the press would
look like a double tap and cancel the swipe.

The one thing that is a guess: the swipe direction. If the model provides
``direction`` we use its heading, normalised to the configured flick length. If
it does not, we fall back to ``InterpreterConfig.flick_delta`` (straight up) -
the same guess the dataset path makes, since the label set has no heading
either. See README §7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .actions import (
    JUDGMENT_LINE_Y,
    Commit,
    InterpreterConfig,
    Op,
    Timeline,
    build_timeline,
    diff_ops,
    flick_offset,
)
from .labels import Chart, LabelEvent
from .tracking import ContactTracker, Detection, TrackerConfig, Tracked, normalise_kind

__all__ = [
    "RawContact",
    "PredictedContact",
    "FramePrediction",
    "ModelStream",
    "StreamStats",
    "predictions_from_raw",
    "timeline_from_predictions",
    "validate",
]

#: What the model hands over: a position and a kind, no identity.
RawContact = Detection


@dataclass(frozen=True)
class PredictedContact:
    """A contact **after** identity assignment.

    Order of fields is frozen for backwards compatibility: ``PredictedContact(0,
    x, y)`` still works and means a tap.
    """

    id: int
    x: float
    y: float
    kind: str = "tap"
    direction: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not (-1.0 <= self.x <= 1.0 and -1.0 <= self.y <= 1.0):
            raise ValueError(
                f"contact {self.id} out of game space: ({self.x}, {self.y}). "
                "If the model regresses raw values, apply tanh or clamp before "
                "handing them over."
            )
        object.__setattr__(self, "kind", normalise_kind(self.kind))

    @classmethod
    def from_tracked(cls, tracked: Tracked) -> "PredictedContact":
        return cls(
            id=tracked.id,
            x=tracked.x,
            y=tracked.y,
            kind=tracked.kind,
            direction=tracked.direction,
        )


@dataclass(frozen=True)
class FramePrediction:
    frame: int
    contacts: tuple[PredictedContact, ...] = ()


def validate(frames: Iterable[FramePrediction]) -> list[str]:
    """Return a list of contract violations (empty == fine)."""
    problems: list[str] = []
    last = -1
    for fp in frames:
        if fp.frame <= last:
            problems.append(f"frames not strictly increasing at {fp.frame}")
        last = fp.frame
        ids = [c.id for c in fp.contacts]
        if len(ids) != len(set(ids)):
            problems.append(f"duplicate contact id on frame {fp.frame}")
    return problems


# -- batch path -------------------------------------------------------------
def predictions_from_raw(
    frames: Iterable[tuple[int, Sequence[RawContact]]],
    tracker: ContactTracker | None = None,
    tracker_config: TrackerConfig | None = None,
) -> list[FramePrediction]:
    """Run the tracker over an id-less dump and return ordinary predictions."""
    tr = tracker if tracker is not None else ContactTracker(tracker_config)
    out: list[FramePrediction] = []
    for frame, detections in frames:
        tracked = tr.update(frame, detections)
        out.append(FramePrediction(frame, tuple(PredictedContact.from_tracked(t) for t in tracked)))
    return out


def _chart_from_predictions(frames: Iterable[FramePrediction], cfg: InterpreterConfig) -> Chart:
    """Predictions -> label-shaped events, preserving each contact's kind.

    A flick is emitted once (on the frame the contact is born as a flick);
    repeating it would make the interpreter release and re-press, which the game
    reads as a double tap rather than a swipe.

    "Is this a continuation?" is decided by *when the id was last seen*, not by
    "was it in the previous frame": one dropped detection in the middle of a
    hold must not split it into two taps, which is exactly what the tracker's
    ``max_gap`` tolerance is for.
    """
    rows: dict[int, list[LabelEvent]] = {}
    last_seen: dict[int, int] = {}
    flick_sent: set[int] = set()
    last_frame = 0
    grace = cfg.hold_grace_frames + 1

    for fp in sorted(frames, key=lambda f: f.frame):
        last_frame = max(last_frame, fp.frame)
        events: list[LabelEvent] = []
        for contact in fp.contacts:
            if contact.kind == "flick":
                if contact.id in flick_sent:
                    continue
                flick_sent.add(contact.id)
                events.append(
                    LabelEvent(
                        frame=fp.frame,
                        id=contact.id,
                        slot=0,
                        phase="flick",
                        x=contact.x,
                        y=contact.y,
                        direction=contact.direction,
                    )
                )
                continue
            continued = fp.frame - last_seen.get(contact.id, -(10**9)) <= grace
            events.append(
                LabelEvent(
                    frame=fp.frame,
                    id=contact.id,
                    slot=0,
                    phase="hold" if continued else "down",
                    x=contact.x,
                    y=contact.y,
                )
            )
            last_seen[contact.id] = fp.frame
        if events:
            rows[fp.frame] = events

    return Chart(
        name="model-output",
        frames={f: tuple(v) for f, v in rows.items()},
        num_frames=last_frame + 1,
    )


def timeline_from_predictions(
    frames: Iterable[FramePrediction],
    cfg: InterpreterConfig | None = None,
) -> Timeline:
    """Compile model output into press/move/release ops.

    ``cfg.hold_tracking`` defaults to ``"follow"`` here: the predicted position
    *is* the finger position, unlike dataset replay where the recorded ``y``
    drifts because it tracks the note.
    """
    cfg = cfg or InterpreterConfig(hold_tracking="follow")
    return build_timeline(_chart_from_predictions(frames, cfg), cfg)


# -- real-time path ---------------------------------------------------------
@dataclass
class StreamStats:
    frames: int = 0
    presses: int = 0
    moves: int = 0
    releases: int = 0
    contacts: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "frames": self.frames,
            "presses": self.presses,
            "moves": self.moves,
            "releases": self.releases,
            "contacts": self.contacts,
        }


@dataclass
class _Live:
    """One contact the injector currently holds (mirrors actions._Contact)."""

    key: int
    x: float
    y: float
    epoch: int
    kind: str
    end: int
    move_at: int | None = None
    move_to: tuple[float, float] | None = None


class ModelStream:
    """Consumes one frame of model output at a time and emits :class:`Commit`.

    This is the real-time API: no buffering, no lookahead, one commit per frame
    (which the injector can apply immediately). Feed it **every** frame; if you
    skip frames it internally replays the missed ones so a scheduled flick move
    or a grace release is never lost, and the ops are merged into the commit you
    get back.

    Semantics match :func:`timeline_from_predictions` exactly - see the module
    docstring.
    """

    def __init__(
        self,
        cfg: InterpreterConfig | None = None,
        tracker: ContactTracker | None = None,
        tracker_config: TrackerConfig | None = None,
    ) -> None:
        self.cfg = cfg or InterpreterConfig(hold_tracking="follow")
        self.tracker = tracker if tracker is not None else ContactTracker(tracker_config)
        self.stats = StreamStats()
        self.frame = -1
        self._live: dict[int, _Live] = {}
        self._state: dict[int, tuple[float, float, int]] = {}

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self.tracker.reset()
        self.stats = StreamStats()
        self.frame = -1
        self._live.clear()
        self._state.clear()

    @property
    def active_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._live))

    # -- main entry point --------------------------------------------------
    def push(
        self,
        detections: Iterable[RawContact],
        frame: int | None = None,
    ) -> Commit:
        """Track ``detections`` and return the ops due on ``frame``."""
        dets = tuple(detections)
        target = self.frame + 1 if frame is None else int(frame)
        if target <= self.frame:
            raise ValueError(f"frame must advance: got {target} after {self.frame}")

        ops: list[Op] = []
        for f in range(self.frame + 1, target):
            ops.extend(self._tick(f))
        self.frame = target

        tracked = self.tracker.update(target, dets)
        ops.extend(self._tick(target, tracked))

        self.stats.frames += 1
        for op in ops:
            if op.kind == "press":
                self.stats.presses += 1
            elif op.kind == "move":
                self.stats.moves += 1
            else:
                self.stats.releases += 1

        return Commit(
            frame=target,
            t=target / self.cfg.fps + self.cfg.time_offset,
            ops=tuple(ops),
        )

    def flush(self) -> Commit:
        """Drain pending flicks and grace releases at the end of a stream."""
        horizon = self.frame + max(
            self.cfg.tap_frames, self.cfg.flick_frames, self.cfg.hold_grace_frames
        ) + 2
        return self.push((), frame=horizon)

    # -- internals ---------------------------------------------------------
    def _tick(self, frame: int, tracked: Sequence[Tracked] | None = None) -> list[Op]:
        ops: list[Op] = []
        if tracked is not None:
            for t in tracked:
                self._ingest(frame, t)

        # a scheduled flick move happens whether or not the model still sees it
        for c in self._live.values():
            if c.move_at == frame and c.move_to is not None:
                c.x, c.y = c.move_to
                c.move_to = None
                c.move_at = None

        for key in [k for k, c in self._live.items() if c.end <= frame]:
            del self._live[key]

        cur = {k: (c.x, c.y, c.epoch) for k, c in self._live.items()}
        if cur != self._state:
            diff_ops(self._state, cur, ops)
            self._state = cur
        return ops

    def _ingest(self, frame: int, t: Tracked) -> None:
        c = self._live.get(t.id)

        if c is None:
            self._live[t.id] = self._birth(frame, t)
            self.stats.contacts += 1
            return

        if t.kind == "flick":
            if c.kind == "flick":
                # already triggered: pressing again would read as a double tap
                return
            # the model refined a tap into a flick mid-life -> restart as a swipe
            c.epoch += 1
            c.kind = "flick"
            c.x, c.y = t.x, t.y
            c.end = frame + self.cfg.flick_frames
            dx, dy = flick_offset(t.direction, self.cfg)
            c.move_at = frame + 1
            c.move_to = (t.x + dx, t.y + dy)
            return

        if c.kind == "flick":
            # keep the scheduled swipe intact; ignore the trailing detections
            return

        if c.kind != "hold":
            c.kind = "hold"
            c.move_at = None
            c.move_to = None
        c.end = frame + self.cfg.hold_grace_frames
        if self.cfg.hold_tracking == "follow":
            c.x, c.y = t.x, t.y
        elif self.cfg.hold_tracking == "judgment_line":
            c.y = JUDGMENT_LINE_Y

    def _birth(self, frame: int, t: Tracked) -> _Live:
        if t.kind == "flick":
            dx, dy = flick_offset(t.direction, self.cfg)
            return _Live(
                t.id, t.x, t.y, 1, "flick", frame + self.cfg.flick_frames,
                move_at=frame + 1, move_to=(t.x + dx, t.y + dy),
            )
        if t.kind == "hold":
            return _Live(t.id, t.x, t.y, 1, "hold", frame + self.cfg.hold_grace_frames)
        return _Live(t.id, t.x, t.y, 1, "tap", frame + self.cfg.tap_frames)
