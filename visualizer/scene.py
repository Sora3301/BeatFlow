"""Contact lifecycle and per-phase particle effects (renderer agnostic).

Every renderer consumes the same :class:`Scene`, so a ``down`` *means* the same
thing in the terminal and in the Tk window; only the primitive changes (a
character cell vs. a canvas oval).

Three effects, one per label phase, chosen so they stay distinguishable even in
a 100x30 character grid and at a glance in a 60 fps window:

``down``
    a decisive hit: one expanding ring plus a radial spray of short sparks.
    Reads as an impact, over in ~0.25 s.

``hold``
    a sustained state: the finger keeps a slow pulsing halo with a rotating arc
    and continuously sheds embers upward. It never "finishes", which is the
    point - the finger is still down.

``flick``
    a swipe: a directed comet of fast sparks along the flick vector plus a
    trailing smear. Reads as motion, not impact.

Colour identifies the *finger* (physical MT slot, falling back to the note key
when the slot is unknown), so four simultaneous fingers are four colours while
the effect shape still tells you the phase.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from .protocol import ContactState

__all__ = [
    "PALETTE",
    "JUDGMENT_LINE_Y",
    "HOLD_AFTER",
    "FLICK_SPEED",
    "Contact",
    "Particle",
    "Ring",
    "Scene",
    "Viewport",
    "finger_color",
    "phase_color",
]

#: One colour per finger, in physical-slot order.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 84, 112),    # slot 0 - rose
    (255, 209, 102),   # slot 1 - amber
    (74, 222, 128),    # slot 2 - green
    (56, 189, 248),    # slot 3 - sky
)

#: Empirical Phigros judgment line in game-normalised space.
JUDGMENT_LINE_Y = 0.375

#: A contact that survives this long with no label phase is treated as a hold.
HOLD_AFTER = 0.12

#: Game-units/second above which an ``auto`` contact is treated as a flick.
FLICK_SPEED = 0.55

RING_LIFE_DOWN = 0.26
RING_LIFE_RELEASE = 0.34
RING_LIFE_FLICK = 0.18

#: Hard ceiling so a pathological chart cannot melt a renderer.
MAX_PARTICLES = 700


def finger_color(slot: int, key: int) -> tuple[int, int, int]:
    """Colour for a contact: physical slot first, note key as a fallback."""
    index = slot if slot >= 0 else key
    return PALETTE[index % len(PALETTE)]


def phase_color(phase: str, base: tuple[int, int, int]) -> tuple[int, int, int]:
    """Nudge the finger colour so the phase is readable even in one colour."""
    r, g, b = base
    if phase == "hold":
        # calmer, slightly desaturated: a state, not an impact
        return (int(r * 0.78 + 30), int(g * 0.78 + 30), int(b * 0.78 + 30))
    if phase == "flick":
        # hotter, pushed toward white: motion and energy
        return (min(255, int(r * 0.45 + 165)), min(255, int(g * 0.45 + 165)), min(255, int(b * 0.45 + 165)))
    return (r, g, b)


def _ease_out_cubic(p: float) -> float:
    q = 1.0 - p
    return 1.0 - q * q * q


def _progress(born: float, life: float, now: float) -> float:
    """Clamped 0..1 lifetime fraction, shared by rings and particles."""
    if life <= 0.0:
        return 1.0
    return min(1.0, max(0.0, (now - born) / life))


def _alive(born: float, life: float, now: float) -> bool:
    return now - born < life


@dataclass(frozen=True)
class Viewport:
    """Which slice of game space the renderer shows.

    Contacts live almost entirely within a thin band around the judgment line,
    so mapping the full ``[-1, 1]`` onto 30 terminal rows would compress every
    finger into two rows. The viewport instead keeps a horizontal span of 2.0
    game units and derives the vertical span from the cell aspect ratio, so a
    ring stays visually circular instead of becoming a smear.

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
        """Fit the vertical span to a character grid without distorting rings.

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

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    born: float
    life: float
    size: float
    color: tuple[int, int, int]
    drag: float = 0.0

    def progress(self, now: float) -> float:
        return _progress(self.born, self.life, now)

    def position(self, now: float) -> tuple[float, float]:
        dt = max(0.0, now - self.born)
        if self.drag > 0.0:
            # exponential slowdown, integrated analytically
            k = self.drag
            f = (1.0 - math.exp(-k * dt)) / k
        else:
            f = dt
        return (self.x + self.vx * f, self.y + self.vy * f)

    def alive(self, now: float) -> bool:
        return _alive(self.born, self.life, now)


@dataclass
class Ring:
    x: float
    y: float
    r0: float
    r1: float
    born: float
    life: float
    color: tuple[int, int, int]
    width: float = 1.0
    filled: bool = False

    def progress(self, now: float) -> float:
        return _progress(self.born, self.life, now)

    def radius(self, now: float) -> float:
        p = self.progress(now)
        return self.r0 + (self.r1 - self.r0) * _ease_out_cubic(p)

    def alpha(self, now: float) -> float:
        return 1.0 - self.progress(now)

    def alive(self, now: float) -> bool:
        return _alive(self.born, self.life, now)


@dataclass
class Contact:
    """A finger currently held down, plus the animation state it earned."""

    key: int
    slot: int
    x: float
    y: float
    vx: float
    vy: float
    phase: str
    born: float
    updated: float
    ember_acc: float = 0.0
    locked: bool = False          # True once a label phase (not a guess) set it

    @property
    def color(self) -> tuple[int, int, int]:
        return finger_color(self.slot, self.key)

    @property
    def effect_color(self) -> tuple[int, int, int]:
        return phase_color(self.phase, self.color)

    def age(self, now: float) -> float:
        return max(0.0, now - self.born)

    def direction(self) -> tuple[float, float]:
        n = math.hypot(self.vx, self.vy)
        if n < 1e-6:
            return (0.0, 1.0)          # the configured default flick: upwards
        return (self.vx / n, self.vy / n)


class Scene:
    """Turns a stream of contact snapshots into animated effects."""

    def __init__(self, seed: int = 20261101, judgement_line_y: float = JUDGMENT_LINE_Y) -> None:
        self.judgement_line_y = judgement_line_y
        self.now = 0.0
        self._rng = random.Random(seed)
        self._contacts: dict[int, Contact] = {}
        self._rings: list[Ring] = []
        self._particles: list[Particle] = []
        self._dt = 1.0 / 60.0

    # -- inspection --------------------------------------------------------
    @property
    def contacts(self) -> Sequence[Contact]:
        return tuple(self._contacts.values())

    @property
    def rings(self) -> Sequence[Ring]:
        return tuple(self._rings)

    @property
    def particles(self) -> Sequence[Particle]:
        return tuple(self._particles)

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
                # been down, so a renderer attached late still animates sanely.
                born = t - max(0.0, st.age)
                phase, locked = self._initial_phase(st.phase)
                contact = Contact(
                    key=st.key,
                    slot=st.slot,
                    x=st.x,
                    y=st.y,
                    vx=st.vx,
                    vy=st.vy,
                    phase=phase,
                    born=born,
                    updated=t,
                    locked=locked,
                )
                self._contacts[st.key] = contact
                self._on_press(contact, t)
                continue

            dx = st.x - contact.x
            dy = st.y - contact.y
            if st.vx or st.vy:
                contact.vx, contact.vy = st.vx, st.vy
            elif dx or dy:
                contact.vx, contact.vy = dx, dy
            contact.x, contact.y = st.x, st.y
            contact.updated = t
            if st.slot >= 0:
                contact.slot = st.slot

            if st.phase in ("down", "hold", "flick") and st.phase != contact.phase:
                contact.phase = st.phase
                contact.locked = True
                self._on_phase(contact, t)

        for key in [k for k in self._contacts if k not in seen]:
            gone = self._contacts.pop(key)
            self._on_release(gone, t)

        self._infer_phases(t)
        self._shed_embers(t)

    def tick(self, t: float) -> None:
        """Advance the clock and age out finished effects."""
        if t > self.now:
            self._dt = min(0.25, t - self.now)   # clamp so a stall cannot jump particles
            self.now = t
        self._rings = [r for r in self._rings if r.alive(self.now)]
        self._particles = [p for p in self._particles if p.alive(self.now)]

    def reset(self) -> None:
        self._contacts.clear()
        self._rings.clear()
        self._particles.clear()

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
            speed = math.hypot(c.vx, c.vy)
            if speed > FLICK_SPEED:
                if c.phase != "flick":
                    # Only on the transition: a fast contact must not spawn a
                    # fresh comet every frame it keeps moving.
                    c.phase = "flick"
                    self._on_phase(c, t)
            elif t - c.born >= HOLD_AFTER and c.phase == "down":
                c.phase = "hold"
                self._on_phase(c, t)

    # -- effects -----------------------------------------------------------
    def _on_press(self, c: Contact, t: float) -> None:
        color = c.effect_color
        if c.phase == "flick":
            self._flash(c.x, c.y, 0.05, t, 0.10, color)
            self._ring(c.x, c.y, 0.02, 0.13, t, RING_LIFE_FLICK, color, 2.0)
            self._flick_spray(c, t)
            return
        if c.phase == "hold":
            self._flash(c.x, c.y, 0.06, t, 0.14, color)
            self._ring(c.x, c.y, 0.03, 0.12, t, 0.22, color, 2.0)
            return
        # down: a hot core that decays fast, plus a double ripple - reads as impact
        self._flash(c.x, c.y, 0.050, t, 0.10, color)
        self._ring(c.x, c.y, 0.02, 0.115, t, RING_LIFE_DOWN, color, 2.4)
        self._ring(c.x, c.y, 0.01, 0.062, t, 0.15, color, 1.8)
        self._sparks(c.x, c.y, color, 9, 0.7, 1.7, 0.24)

    def _on_phase(self, c: Contact, t: float) -> None:
        color = c.effect_color
        if c.phase == "hold":
            # a "lock-on": the finger stops being an impact and becomes a state
            self._ring(c.x, c.y, 0.06, 0.15, t, 0.20, color, 2.0)
            self._sparks(c.x, c.y, color, 6, 0.25, 0.7, 0.30, angle=math.pi / 2.0, spread=2.0)
        elif c.phase == "flick":
            self._flash(c.x, c.y, 0.05, t, 0.10, color)
            self._ring(c.x, c.y, 0.02, 0.13, t, RING_LIFE_FLICK, color, 2.0)
            self._flick_spray(c, t)

    def _on_release(self, c: Contact, t: float) -> None:
        color = c.color
        self._ring(c.x, c.y, 0.04, 0.18, t, RING_LIFE_RELEASE, color, 1.4)
        self._sparks(c.x, c.y, color, 5, 0.35, 1.0, 0.20)

    def _flick_spray(self, c: Contact, t: float) -> None:
        dx, dy = c.direction()
        angle = math.atan2(dy, dx)
        self._sparks(
            c.x, c.y, c.effect_color, 14, 1.4, 3.0, 0.20,
            angle=angle, spread=0.9, drag=3.2,
        )

    def _shed_embers(self, t: float) -> None:
        """Holds emit a slow trickle so a long hold never looks frozen."""
        for c in self._contacts.values():
            if c.phase != "hold":
                continue
            c.ember_acc += self._dt
            if c.ember_acc < 0.07:
                continue
            c.ember_acc = 0.0
            self._sparks(
                c.x, c.y, c.effect_color, 1, 0.18, 0.42, 0.55,
                angle=math.pi / 2.0, spread=1.1,
            )

    # -- primitive spawners ------------------------------------------------
    def _ring(self, x: float, y: float, r0: float, r1: float, t: float, life: float,
              color: tuple[int, int, int], width: float) -> None:
        self._rings.append(Ring(x, y, r0, r1, t, life, color, width))

    def _flash(self, x: float, y: float, radius: float, t: float, life: float,
               color: tuple[int, int, int]) -> None:
        """A soft filled disc that decays fast - the "something just happened" cue."""
        self._rings.append(Ring(x, y, radius * 0.35, radius, t, life, color, 0.0, filled=True))

    def _sparks(self, x: float, y: float, color: tuple[int, int, int], n: int,
                speed_lo: float, speed_hi: float, life: float,
                angle: float | None = None, spread: float = math.tau,
                drag: float = 1.6) -> None:
        # Spawn only into the free slots of the budget instead of trimming after
        # the fact; keeps the cap in one place.
        for i in range(min(n, max(0, MAX_PARTICLES - len(self._particles)))):
            if angle is None:
                a = spread * (i / n) + self._rng.uniform(-0.18, 0.18)
            else:
                a = angle + self._rng.uniform(-spread, spread) * 0.5
            speed = self._rng.uniform(speed_lo, speed_hi)
            self._particles.append(
                Particle(
                    x=x,
                    y=y,
                    vx=math.cos(a) * speed,
                    vy=math.sin(a) * speed,
                    born=self.now,
                    life=life * self._rng.uniform(0.75, 1.15),
                    size=self._rng.uniform(0.6, 1.4),
                    color=color,
                    drag=drag,
                )
            )
