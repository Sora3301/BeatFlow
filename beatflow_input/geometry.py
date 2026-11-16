"""Coordinate spaces: game-normalised ``[-1, 1]`` -> screen px -> uinput units.

Three spaces are in play and mixing them up is the classic source of "the bot
taps 200 px too low":

1. **game space** - what the labels store: ``x, y`` in ``[-1, 1]``, origin at
   the centre of the play field, ``+x`` right, ``+y`` up (Phigros convention).
   Values outside ``[-1, 1]`` appear in the dataset (~0.7 % of events) and are
   flagged ``bad`` - see :meth:`clamp01`.
2. **screen space** - pixels on the compositor output.
3. **device space** - what ``ABS_MT_POSITION_X/Y`` carry: ``0 .. dev_max``,
   ``+y`` **down**, covering the whole output.

:class:`Geometry` composes 1 -> 2 -> 3 and also knows about letterboxing: a
windowed or 4:3 game inside a 16:9 output does not fill the output, and a
naive full-output mapping silently shifts every note.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Rect", "Geometry"]


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in screen pixels (float for sub-pixel maths)."""

    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def aspect(self) -> float:
        return self.w / self.h if self.h else float("inf")

    def inset(self, px: float) -> "Rect":
        return Rect(self.x + px, self.y + px, self.w - 2 * px, self.h - 2 * px)

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass(frozen=True)
class Geometry:
    """Maps game-normalised coordinates onto absolute device units.

    Parameters
    ----------
    screen_w, screen_h:
        Size of the output in pixels (the space ``area`` is expressed in).
    area:
        Rectangle of the *play field* inside that output. ``None`` = the whole
        output, which is correct for a fullscreen borderless game.
    invert_y:
        Game space has ``+y`` up, device space has ``+y`` down -> ``True``.
        Flip this only if you discover the labels use screen-style ``+y``.
    clamp:
        Clamp out-of-range coordinates into ``[0, dev_max]`` instead of letting
        them wrap. Keep ``True`` in production; set ``False`` in tests that
        want to see raw overflow.
    """

    screen_w: int = 1920
    screen_h: int = 1080
    area: Rect | None = None
    dev_max_x: int = 32767
    dev_max_y: int = 32767
    invert_y: bool = True
    clamp: bool = True

    def __post_init__(self) -> None:
        if self.screen_w <= 0 or self.screen_h <= 0:
            raise ValueError("screen size must be positive")
        if self.area is None:
            object.__setattr__(self, "area", Rect(0.0, 0.0, float(self.screen_w), float(self.screen_h)))
        else:
            a = self.area
            if a.w <= 0 or a.h <= 0:
                raise ValueError("area must have positive size")
            if a.x < 0 or a.y < 0 or a.right > self.screen_w + 0.5 or a.bottom > self.screen_h + 0.5:
                raise ValueError(f"area {a} does not fit inside {self.screen_w}x{self.screen_h}")

    # -- constructors ------------------------------------------------------
    @classmethod
    def fullscreen(cls, screen_w: int, screen_h: int, **kwargs) -> "Geometry":
        return cls(screen_w=screen_w, screen_h=screen_h, area=None, **kwargs)

    @classmethod
    def letterboxed(
        cls,
        screen_w: int,
        screen_h: int,
        game_w: float,
        game_h: float,
        mode: str = "fit",
        **kwargs,
    ) -> "Geometry":
        """Window of ``game_w x game_h`` centred on the output.

        ``mode="fit"``  keeps aspect ratio and adds bars (what games do);
        ``mode="stretch"`` fills the output (what a resized window might do).
        """
        if mode not in {"fit", "stretch"}:
            raise ValueError("mode must be 'fit' or 'stretch'")
        if mode == "stretch":
            area = Rect(0.0, 0.0, float(screen_w), float(screen_h))
        else:
            scale = min(screen_w / game_w, screen_h / game_h)
            w, h = game_w * scale, game_h * scale
            area = Rect((screen_w - w) / 2.0, (screen_h - h) / 2.0, w, h)
        return cls(screen_w=screen_w, screen_h=screen_h, area=area, **kwargs)

    @classmethod
    def windowed(
        cls,
        screen_w: int,
        screen_h: int,
        win_x: float,
        win_y: float,
        win_w: float,
        win_h: float,
        **kwargs,
    ) -> "Geometry":
        """Game drawn in an explicit window rectangle."""
        return cls(screen_w=screen_w, screen_h=screen_h, area=Rect(win_x, win_y, win_w, win_h), **kwargs)

    # -- helpers -----------------------------------------------------------
    @property
    def play_area(self) -> Rect:
        assert self.area is not None
        return self.area

    @staticmethod
    def clamp01(v: float) -> float:
        return -1.0 if v < -1.0 else (1.0 if v > 1.0 else v)

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        """Game space -> screen pixels (no clamping)."""
        a = self.play_area
        sx = a.x + (x + 1.0) * 0.5 * a.w
        sy = a.y + ((1.0 - y) * 0.5 if self.invert_y else (y + 1.0) * 0.5) * a.h
        return sx, sy

    def to_device(self, x: float, y: float) -> tuple[int, int]:
        """Game space -> absolute device units (what a backend takes)."""
        if self.clamp:
            x = self.clamp01(x)
            y = self.clamp01(y)
        sx, sy = self.to_screen(x, y)
        dx = sx / self.screen_w * self.dev_max_x
        dy = sy / self.screen_h * self.dev_max_y
        ix, iy = int(round(dx)), int(round(dy))
        if self.clamp:
            ix = min(max(ix, 0), self.dev_max_x)
            iy = min(max(iy, 0), self.dev_max_y)
        return ix, iy

    def to_game(self, dx: int, dy: int) -> tuple[float, float]:
        """Inverse map, mainly for debugging / round-trip tests."""
        sx = dx / self.dev_max_x * self.screen_w
        sy = dy / self.dev_max_y * self.screen_h
        a = self.play_area
        x = (sx - a.x) / a.w * 2.0 - 1.0
        ty = (sy - a.y) / a.h
        y = 1.0 - ty * 2.0 if self.invert_y else ty * 2.0 - 1.0
        return x, y

    def describe(self) -> str:
        a = self.play_area
        return (
            f"{self.screen_w}x{self.screen_h} screen, play area "
            f"({a.x:.0f},{a.y:.0f}) {a.w:.0f}x{a.h:.0f}, "
            f"device 0..{self.dev_max_x}x0..{self.dev_max_y}"
        )
