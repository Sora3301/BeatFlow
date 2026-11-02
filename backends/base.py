"""Backend abstraction for absolute multitouch injection (Linux MT protocol B).

Every backend speaks the same low-level language::

    press(slot, tracking_id, x, y)   -> a new contact appears in ``slot``
    move(slot, x, y)                 -> that contact changes position
    release(slot)                    -> the contact in ``slot`` disappears
    commit()                         -> "this is one input frame"

Coordinates handed to a backend are always **absolute device units**
(``0 .. dev_max``), never game-normalised values. Mapping happens above this
layer (see :mod:`beatflow_input.geometry`).

Slots are *physical* MT slots and must stay in ``0 .. capacity-1``. Mapping a
game note id onto a physical slot is the job of
:class:`beatflow_input.slots.SlotAllocator`, not of the backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["TouchBackend", "TRACKING_NONE", "BackendUnavailable"]

#: ``ABS_MT_TRACKING_ID`` value that lifts a contact (kernel API uses -1).
TRACKING_NONE = -1


class BackendUnavailable(RuntimeError):
    """Raised when the host cannot provide this backend (missing device, ...)."""


class TouchBackend(ABC):
    """Sink for one frame of absolute multitouch events."""

    #: Short identifier used in logs / the CLI.
    name: str = "base"

    @property
    @abstractmethod
    def capacity(self) -> int:
        """How many contacts can be reported simultaneously."""

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> None:  # noqa: B027 - optional hook
        """Acquire the underlying resource. Must be idempotent."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release everything, lifting any contact still held."""

    # -- event vocabulary --------------------------------------------------
    @abstractmethod
    def press(self, slot: int, tracking_id: int, x: int, y: int) -> None:
        """Start a contact in ``slot`` with a fresh ``tracking_id``."""

    @abstractmethod
    def move(self, slot: int, x: int, y: int) -> None:
        """Move the contact currently living in ``slot``."""

    @abstractmethod
    def release(self, slot: int) -> None:
        """Lift the contact in ``slot``."""

    @abstractmethod
    def commit(self) -> None:
        """Publish everything queued since the previous commit as one frame."""

    # -- sugar -------------------------------------------------------------
    def __enter__(self) -> "TouchBackend":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<{type(self).__name__} capacity={self.capacity}>"
