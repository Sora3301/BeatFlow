"""Live multi-touch visualizer plugins.

The visualizer answers one question: *what does the agent think its fingers are
doing right now?* It renders the contacts the injector actually holds, colour
coded per finger, with a different effect for each label phase.

Architecture (deliberate)
-------------------------
The renderer runs in its **own process**, fed by NDJSON on stdin
(:mod:`~beatflow_input.visualizer.protocol`). Two reasons:

1. the injector spins on absolute deadlines and must not lose its 16.7 ms frame
   to a renderer that got scheduled in between;
2. Tk owns its main thread, which the scheduler is already using.

Adding a renderer is one entry in :data:`RENDERERS` plus a module with a
``main()`` that reads frames from stdin.

Available renderers
-------------------
``terminal``
    ANSI/truecolour, no GUI needed, works over SSH
    (:mod:`~beatflow_input.visualizer.terminal`).
``tk``
    a real window with rings, particles and arcs
    (:mod:`~beatflow_input.visualizer.tkview`).

Both can also be run standalone against a synthetic 4-finger routine::

    python3 -m beatflow_input.visualizer.terminal --demo
    python3 -m beatflow_input.visualizer.tkview --demo
"""

from __future__ import annotations

from .live import RENDERERS, LiveModel, RenderProcess, Streamer, phase_map
from .protocol import PROTOCOL_VERSION, ContactState, FrameState
from .scene import (
    FLICK_SPEED,
    HOLD_AFTER,
    JUDGMENT_LINE_Y,
    PALETTE,
    Contact,
    Particle,
    Ring,
    Scene,
    Viewport,
    finger_color,
    phase_color,
)

__all__ = [
    "RENDERERS",
    "Contact",
    "ContactState",
    "FLICK_SPEED",
    "FrameState",
    "HOLD_AFTER",
    "JUDGMENT_LINE_Y",
    "LiveModel",
    "PALETTE",
    "PROTOCOL_VERSION",
    "Particle",
    "RenderProcess",
    "Ring",
    "Scene",
    "Streamer",
    "Viewport",
    "finger_color",
    "phase_color",
    "phase_map",
    "available",
]


def available() -> list[str]:
    """Renderer names accepted by ``--viz``."""
    return sorted(RENDERERS)
