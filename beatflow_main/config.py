"""主程序的一次运行配置。

只放「怎么跑」的参数；「怎么解释手指」的参数在
:class:`beatflow_input.actions.InterpreterConfig`，由 :meth:`RunConfig.interpreter`
构造。两者分开，是为了让调时序的人不用碰合成规则。
"""

from __future__ import annotations

from dataclasses import dataclass

from beatflow_input.actions import InterpreterConfig
from beatflow_input.geometry import Geometry

__all__ = ["RunConfig", "TAIL_FRAMES"]

#: 录制的预测播完之后再补几帧。补帧期间模型输出为空，让该发的 release
#: 落在它自然的帧上，而不是全部挤进最后一次 flush。
#: 8 帧足够覆盖 ``tap_frames`` / ``flick_frames`` / ``hold_grace_frames`` 的默认值。
TAIL_FRAMES = 8


@dataclass
class RunConfig:
    """Everything the CLI needs to build a run."""

    fps: float = 60.0
    capacity: int = 10
    policy: str = "steal_lru"
    screen: tuple[int, int] = (1920, 1080)
    window: tuple[float, float, float, float] | None = None
    game_size: tuple[float, float] | None = None
    realtime: bool = True
    duration: float = 0.0
    lead_in: float = 0.0
    flush: bool = True
    #: ``False`` treats the label ``y`` as screen-style (``+y`` down). The
    #: dataset convention is still an open question (README §7), so it is a
    #: runtime knob rather than a constant.
    invert_y: bool = True

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.capacity < 1:
            raise ValueError("capacity must be >= 1")
        if self.window is not None and self.game_size is not None:
            raise ValueError("window and game_size are mutually exclusive")

    # -- derived -----------------------------------------------------------
    def geometry(self) -> Geometry:
        """Collect the three coordinate spaces this run expects."""
        if self.window is not None:
            x, y, w, h = self.window
            return Geometry.windowed(
                self.screen[0], self.screen[1], x, y, w, h, invert_y=self.invert_y
            )
        if self.game_size is not None:
            return Geometry.letterboxed(
                self.screen[0], self.screen[1], *self.game_size, invert_y=self.invert_y
            )
        return Geometry.fullscreen(
            self.screen[0], self.screen[1], invert_y=self.invert_y
        )

    def interpreter(self) -> InterpreterConfig:
        return InterpreterConfig(fps=self.fps)

    def max_frames(self) -> int | None:
        """``duration`` -> frames, or ``None`` for "until the source ends"."""
        if self.duration <= 0:
            return None
        return max(1, int(round(self.duration * self.fps)))
