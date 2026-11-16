"""Contact lifecycle for the visualizer, independent of how it is drawn.

Every renderer consumes the same :class:`Scene`, so a ``down`` *means* the same
thing in the terminal and in the Tk window; only the primitive changes (a
character cell vs. a bitmap).

The scene keeps three things and nothing else:

* **which fingers are down** - keyed by the logical contact id, carrying the
  physical slot, the position and the phase;
* **how long each has been down** - so a renderer attached mid-flight can tell
  a fresh tap from a long hold;
* **a phase guess for unlabelled input** - model predictions carry no ``phase``,
  so a contact that survives :data:`HOLD_AFTER` is treated as a hold and one
  that moves faster than :data:`FLICK_SPEED` as a flick.

It deliberately does **not** own any effects: no rings, sparks, trails or
glows. The finger sprites are the whole picture; anything animated on top of
them was removed on purpose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .protocol import ContactState

__all__ = [
    "PALETTE",
    "JUDGMENT_LINE_Y",
    "HOLD_AFTER",
    "FLICK_SPEED",
    "Contact",
    "Scene",
    "Viewport",
    "finger_color",
]

#: One colour per finger, in physical-slot order. Derived from the four sprite
#: drawings (``visualizer/assets/finger*.png``): each value is that sprite's most
#: chromatic colour lightened to a readable brightness, so the terminal glyphs
#: and the HUD swatches match the fish.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (130, 181, 170),   # slot 0 - teal     (silver/teal fish)
    (152, 167, 186),   # slot 1 - slate    (navy fish)
    (51, 236, 203),    # slot 2 - cyan     (neon fish)
    (170, 157, 189),   # slot 3 - lavender (violet fish)
)

#: Empirical Phigros judgment line in game-normalised space.
JUDGMENT_LINE_Y = 0.375

#: A contact that survives this long with no label phase is treated as a hold.
HOLD_AFTER = 0.12

#: Game-units/second above which an ``auto`` contact is treated as a flick.
FLICK_SPEED = 0.55


def finger_color(slot: int, key: int) -> tuple[int, int, int]:
    """Colour for a contact: physical slot first, note key as a fallback."""
    index = slot if slot >= 0 else key
    return PALETTE[index % len(PALETTE)]


@dataclass(frozen=True)
class Viewport:
    """Which slice of game space the renderer shows.

    Contacts live almost entirely within a thin band around the judgment line,
    so mapping the full ``[-1, 1]`` onto 30 terminal rows would compress every
    finger into two rows. The viewport instead keeps a horizontal span of 2.0
    game units and derives the vertical span from the cell aspect ratio, so a
    sprite keeps its proportions instead of becoming a smear.

    ``+y`` is up in game space; ``to_unit`` flips it because screen rows grow
    downwards.
    """

    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -0.15
    y_max: float = 1.10

    @classmethod
    def for_grid(cls, cols: int, rows: int, line_y: float = JUDGMENT_LINE_Y,
                 char_aspect: float = 2.0) -> "Viewport":
        """Fit the vertical span to a character grid without distorting shapes.

        ``char_aspect`` is height/width of one cell (≈2 for monospace text,
        1 for pixels).
        """
        cols = max(8, cols)
        rows = max(4, rows)
        span = rows * (2.0 / cols) * char_aspect
        span = min(1.7, max(0.85, span))
        # keep the judgment line low in frame, like the game does
        y_min = line_y - 0.42 * span
        return cls(-1.0, 1.0, y_min, y_min + span)

    def to_unit(self, x: float, y: float) -> tuple[float, float]:
        u = (x - self.x_min) / (self.x_max - self.x_min)
        v = (self.y_max - y) / (self.y_max - self.y_min)
        return u, v


@dataclass
class Contact:
    """A finger currently held down."""

    key: int
    slot: int
    x: float
    y: float
    vx: float
    vy: float
    phase: str
    born: float
    locked: bool = False          # True once a label phase (not a guess) set it

    @property
    def color(self) -> tuple[int, int, int]:
        return finger_color(self.slot, self.key)

    def age(self, now: float) -> float:
        return max(0.0, now - self.born)


class Scene:
    """Reconciles per-frame contact snapshots with what is currently on screen."""

    def __init__(self, judgement_line_y: float = JUDGMENT_LINE_Y) -> None:
        self.judgement_line_y = judgement_line_y
        self.now = 0.0
        self._contacts: dict[int, Contact] = {}

    # -- inspection --------------------------------------------------------
    @property
    def contacts(self) -> Sequence[Contact]:
        return tuple(self._contacts.values())

    def phase_counts(self) -> dict[str, int]:
        counts = {"down": 0, "hold": 0, "flick": 0}
        for c in self._contacts.values():
            counts[c.phase] = counts.get(c.phase, 0) + 1
        return counts

    # -- main entry points -------------------------------------------------
    def update(self, t: float, states: Iterable[ContactState]) -> None:
        """Advance to ``t`` and reconcile with the authoritative contact set."""
        self.tick(t)

        seen: set[int] = set()
        for st in states:
            if st.key in seen:
                continue
            seen.add(st.key)
            contact = self._contacts.get(st.key)

            if contact is None:
                # Join mid-flight: the producer reports how long the finger has
                # been down, so a renderer attached late still knows its age.
                born = t - max(0.0, st.age)
                phase, locked = self._initial_phase(st.phase)
                self._contacts[st.key] = Contact(
                    key=st.key,
                    slot=st.slot,
                    x=st.x,
                    y=st.y,
                    vx=st.vx,
                    vy=st.vy,
                    phase=phase,
                    born=born,
                    locked=locked,
                )
                continue

            dx = st.x - contact.x
            dy = st.y - contact.y
            if st.vx or st.vy:
                contact.vx, contact.vy = st.vx, st.vy
            elif dx or dy:
                contact.vx, contact.vy = dx, dy
            contact.x, contact.y = st.x, st.y
            if st.slot >= 0:
                contact.slot = st.slot

            if st.phase in ("down", "hold", "flick") and st.phase != contact.phase:
                contact.phase = st.phase
                contact.locked = True

        for key in [k for k in self._contacts if k not in seen]:
            del self._contacts[key]

        self._infer_phases(t)

    def tick(self, t: float) -> None:
        """Advance the clock (kept so renderers can drive the scene while idle)."""
        if t > self.now:
            self.now = t

    def reset(self) -> None:
        self._contacts.clear()
        self.now = 0.0

    # -- inference ---------------------------------------------------------
    @staticmethod
    def _initial_phase(label: str) -> tuple[str, bool]:
        if label in ("down", "hold", "flick"):
            return label, True
        return "down", False

    def _infer_phases(self, t: float) -> None:
        """Guess a phase for contacts the producer could not label."""
        for c in self._contacts.values():
            if c.locked:
                continue
            if math.hypot(c.vx, c.vy) > FLICK_SPEED:
                c.phase = "flick"
            elif t - c.born >= HOLD_AFTER and c.phase == "down":
                c.phase = "hold"
