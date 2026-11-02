"""The contract between the model and this module.

The CNN+LSTM predicts, **for every frame**, which contacts should be on the
screen. This module turns that into kernel events. The interface is deliberately
dumb and lossless so the model side never needs to know about uinput, slots or
coordinate spaces:

.. code-block:: python

    # per frame (60 Hz), the model emits a list of contacts
    [
        PredictedContact(id=0, x=-0.40, y=0.375),
        PredictedContact(id=1, x=0.42, y=0.375),
    ]

Rules:

* ``x``/``y`` are game-normalised, ``[-1, 1]``, same convention as the labels;
* ``id`` must stay stable while a note is held and must not be reused within
  one frame. Monotonic integers are the simplest choice;
* a contact present for one frame becomes a tap; present for N frames becomes a
  hold; whether the game sees a flick is decided by whether the position moves
  while the contact is down;
* empty frame == no contact, which is the common case and costs nothing.

The adapter below rebuilds label-shaped events and reuses
:func:`beatflow_input.actions.build_timeline`, so model-driven injection and
dataset replay share one code path - a bug fixed for playback is fixed for the
live agent too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .actions import InterpreterConfig, Timeline, build_timeline
from .labels import Chart, LabelEvent

__all__ = ["PredictedContact", "FramePrediction", "timeline_from_predictions", "validate"]


@dataclass(frozen=True)
class PredictedContact:
    """One finger the model wants down on a given frame."""

    id: int
    x: float
    y: float

    def __post_init__(self) -> None:
        if not (-1.0 <= self.x <= 1.0 and -1.0 <= self.y <= 1.0):
            raise ValueError(
                f"contact {self.id} out of game space: ({self.x}, {self.y}). "
                "If the model regresses raw values, apply tanh or clamp before "
                "handing them over."
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
    rows: dict[int, list[LabelEvent]] = {}
    previous: set[int] = set()
    last_frame = 0

    for fp in sorted(frames, key=lambda f: f.frame):
        last_frame = max(last_frame, fp.frame)
        current = {c.id for c in fp.contacts}
        events = []
        for contact in fp.contacts:
            phase = "hold" if contact.id in previous else "down"
            events.append(
                LabelEvent(
                    frame=fp.frame,
                    id=contact.id,
                    slot=0,
                    phase=phase,
                    x=contact.x,
                    y=contact.y,
                )
            )
        if events:
            rows[fp.frame] = events
        previous = current

    chart = Chart(name="model-output", frames={f: tuple(v) for f, v in rows.items()},
                  num_frames=last_frame + 1)
    return build_timeline(chart, cfg)
