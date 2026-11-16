"""帧从哪来。

正式的采集后端（Wayland screencopy / xdg-desktop-portal + PipeWire）后续接在
这个协议后面；M1 阶段用 :class:`SyntheticSource` 产出空帧，让整条回路在没有
显示器、没有游戏的情况下也能跑起来并压测时序。回放式模型不看像素，所以空帧
完全够用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

__all__ = ["Frame", "FrameSource", "SyntheticSource"]


@dataclass(frozen=True)
class Frame:
    """One captured frame.

    ``image`` is deliberately untyped: the capture backend decides whether it is
    an ``ndarray``, an RGB buffer, a mapped SHM file, ... The model adapter is
    the only code that needs to know.
    """

    index: int
    captured_at: float
    image: object | None = None


class FrameSource(Protocol):
    """Anything that can hand over frames, one per tick."""

    def open(self) -> None: ...

    def read(self) -> Frame: ...

    def close(self) -> None: ...


class SyntheticSource:
    """Yields empty frames.

    ``frames=None`` means "forever" (the loop then stops on ``max_frames`` or an
    abort); a finite count makes ``read`` raise ``StopIteration``, which the loop
    treats as a clean end.
    """

    name = "synthetic"

    def __init__(
        self,
        frames: int | None = None,
        image: object | None = None,
        clock=time.monotonic,
    ) -> None:
        self.total = frames
        self.image = image
        self.clock = clock
        self._index = 0
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def read(self) -> Frame:
        if not self._open:
            raise RuntimeError("source is not open")
        if self.total is not None and self._index >= self.total:
            raise StopIteration
        frame = Frame(self._index, self.clock(), self.image)
        self._index += 1
        return frame
