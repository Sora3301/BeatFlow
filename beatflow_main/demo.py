"""A built-in prediction sequence, so the loop can be demoed with no data files.

It is hand-written rather than generated: the point is to exercise every path the
interpreter has (single tap, chord, hold, flick with a heading) in a few seconds,
so a wrong release or a missing swipe is obvious in the logs / visualizer.
"""

from __future__ import annotations

from beatflow_input import RawContact

from .model import RecordedModel

__all__ = ["DEMO_FRAMES", "demo_model"]

#: How long the demo runs, in frames (≈ 4 s at 60 Hz). The last event is at 205.
DEMO_FRAMES = 240


def demo_model() -> RecordedModel:
    frames: dict[int, list[RawContact]] = {}

    def add(frame: int, contact: RawContact) -> None:
        frames.setdefault(frame, []).append(contact)

    # 单点 tap：按下 2 帧后抬起
    add(10, RawContact(-0.40, 0.375, "tap"))
    add(25, RawContact(0.00, 0.375, "tap"))

    # 双押：同一帧两个触点
    add(40, RawContact(-0.20, 0.375, "tap"))
    add(40, RawContact(0.25, 0.375, "tap"))

    # hold：连续 31 帧都在，靠 hold_grace 收尾
    for frame in range(70, 101):
        add(frame, RawContact(-0.60, 0.375, "hold"))

    # flick：给方向，长度会被归一化到 cfg.flick_delta
    add(130, RawContact(0.20, 0.375, "flick", (0.8, 0.6)))

    # 第二段 hold
    for frame in range(160, 191):
        add(frame, RawContact(0.60, 0.375, "hold"))

    add(205, RawContact(-0.10, 0.375, "tap"))
    return RecordedModel.from_sequence((f, tuple(v)) for f, v in sorted(frames.items()))
