"""Backend registry + the two implementations that matter."""

from __future__ import annotations

from .base import TRACKING_NONE, BackendUnavailable, TouchBackend
from .dryrun import DryRunBackend, RecordedOp
from .uinput import UinputBackend

__all__ = [
    "TouchBackend",
    "BackendUnavailable",
    "TRACKING_NONE",
    "DryRunBackend",
    "RecordedOp",
    "UinputBackend",
    "make_backend",
]


def make_backend(kind: str, **kwargs) -> TouchBackend:
    """Factory used by the CLI and by tests.

    ``kind`` is ``"uinput"`` or ``"dryrun"``. Unknown kinds are rejected
    loudly instead of silently falling back - a silent dry run during a live
    demo would look like "the bot does nothing".
    """
    kind = kind.lower()
    if kind == "uinput":
        return UinputBackend(**kwargs)
    if kind == "dryrun":
        return DryRunBackend(**kwargs)
    raise ValueError(f"unknown backend {kind!r} (expected 'uinput' or 'dryrun')")
