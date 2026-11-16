"""主循环：采集 → 推理 → 追踪/合成 → 注入。

一帧的固定顺序::

    wait_until(target)        # 绝对 deadline，误差不随曲长累积
    t0 开始计时
    capture()                 # 帧源
    model.predict(frame)      # 模型
    stream.push(dets, frame)  # 追踪 id + 生命周期 + flick/move/release 合成
    injector.apply(commit)    # 槽位 + 坐标 + uinput

为什么先等 deadline 再采集：游戏读的是「此刻屏幕上有什么」，采集越接近
deadline，模型看到的世界越新。代价是注入落在 ``target + 采集 + 推理``，所以
端到端延迟要按这四段分别打点，而不是只看注入那 100 µs。

``realtime=False`` 会跳过等待（测试、离线压测用），此时 ``tick`` 的耗时仍然有
参考价值，但 ``lateness`` 没有意义。
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Sequence

from beatflow_input import Commit, Injector, ModelStream
from beatflow_input.model_api import RawContact

from .frame_source import FrameSource
from .model import Model

__all__ = ["TickSample", "LoopStats", "LoopResult", "AgentLoop", "wait_until"]


def wait_until(
    target: float,
    clock=time.monotonic,
    spin: float = 1.5e-3,
) -> None:
    """Sleep most of the way, then spin (same trick as ``scheduler.TimelineRunner``)."""
    while (remaining := target - clock()) > spin:
        time.sleep(remaining - spin)
    while clock() < target:
        pass


@dataclass(frozen=True)
class TickSample:
    """One loop iteration, with all four segments timed."""

    frame: int
    target: float
    started: float
    captured: float
    inferred: float
    compiled: float
    applied: float

    @property
    def late_us(self) -> float:
        return (self.started - self.target) * 1e6

    @property
    def capture_us(self) -> float:
        return (self.captured - self.started) * 1e6

    @property
    def infer_us(self) -> float:
        return (self.inferred - self.captured) * 1e6

    @property
    def compile_us(self) -> float:
        return (self.compiled - self.inferred) * 1e6

    @property
    def inject_us(self) -> float:
        return (self.applied - self.compiled) * 1e6

    @property
    def tick_us(self) -> float:
        """Whole iteration, from the deadline to the last uinput write."""
        return (self.applied - self.started) * 1e6


def _block(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def pct(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] * (1 - (pos - lo)) + ordered[hi] * (pos - lo)

    return {
        "n": len(values),
        "mean_us": round(statistics.fmean(values), 2),
        "p50_us": round(pct(0.50), 2),
        "p95_us": round(pct(0.95), 2),
        "p99_us": round(pct(0.99), 2),
        "max_us": round(max(values), 2),
        "min_us": round(min(values), 2),
    }


@dataclass
class LoopStats:
    samples: list[TickSample] = field(default_factory=list)
    wall_s: float = 0.0
    fps: float = 60.0
    aborted: bool = False
    realtime: bool = True

    def summary(self) -> dict:
        frames = len(self.samples)
        return {
            "frames": frames,
            "wall_s": round(self.wall_s, 3),
            "fps_actual": round(frames / self.wall_s, 2) if self.wall_s else 0.0,
            "aborted": self.aborted,
            "realtime": self.realtime,
            # Without deadline pacing "late" is meaningless (negative by design).
            "lateness": _block([s.late_us for s in self.samples]) if self.realtime else {"n": 0},
            "capture": _block([s.capture_us for s in self.samples]),
            "infer": _block([s.infer_us for s in self.samples]),
            "compile": _block([s.compile_us for s in self.samples]),
            "inject": _block([s.inject_us for s in self.samples]),
            "tick": _block([s.tick_us for s in self.samples]),
            "ticks_over_frame": sum(
                1 for s in self.samples if s.tick_us > 1e6 / self.fps
            ),
        }

    def format(self) -> str:
        s = self.summary()
        rows = [
            f"frames {s['frames']}   wall {s['wall_s']:.3f}s   "
            f"({s['fps_actual']:.1f} fps)" + ("   ABORTED" if s["aborted"] else "")
        ]
        for label, key in (
            ("lateness", "lateness"),
            ("capture", "capture"),
            ("infer", "infer"),
            ("compile", "compile"),
            ("inject", "inject"),
            ("tick", "tick"),
        ):
            block = s[key]
            if not block.get("n"):
                rows.append(f"{label:<9} (n/a" + (": realtime=False)" if key == "lateness" else ")"))
                continue
            rows.append(
                f"{label:<9} p50 {block['p50_us']:>8.1f} us   "
                f"p95 {block['p95_us']:>8.1f} us   "
                f"p99 {block['p99_us']:>8.1f} us   "
                f"max {block['max_us']:>9.1f} us"
            )
        rows.append(f"ticks over one frame: {s['ticks_over_frame']}")
        return "\n".join(rows)


@dataclass
class LoopResult:
    commits: list[Commit]
    stats: LoopStats

    def ops(self) -> list[tuple[int, str, int, float, float]]:
        """Flattened ops, in the same shape the batch path produces."""
        return [
            (c.frame, o.kind, o.key, round(o.x, 4), round(o.y, 4))
            for c in self.commits
            for o in c.ops
        ]

    def n_ops(self) -> int:
        return sum(len(c.ops) for c in self.commits)


class AgentLoop:
    """Runs source + model + injector at a fixed frame rate."""

    def __init__(
        self,
        source: FrameSource,
        model: Model,
        injector: Injector,
        stream: ModelStream,
        *,
        fps: float = 60.0,
        realtime: bool = True,
        max_frames: int | None = None,
        flush: bool = True,
        abort: threading.Event | None = None,
        clock=time.monotonic,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.source = source
        self.model = model
        self.injector = injector
        self.stream = stream
        self.fps = fps
        self.realtime = realtime
        self.max_frames = max_frames
        self.flush = flush
        self.abort = abort
        self.clock = clock

    # ------------------------------------------------------------------
    def run(self) -> LoopResult:
        self.source.open()
        commits: list[Commit] = []
        samples: list[TickSample] = []
        aborted = False
        t0 = self.clock()
        frame = 0

        try:
            while self.max_frames is None or frame < self.max_frames:
                if self.abort is not None and self.abort.is_set():
                    aborted = True
                    break

                target = t0 + frame / self.fps
                if self.realtime:
                    wait_until(target, self.clock)

                started = self.clock()
                try:
                    image = self.source.read()
                except StopIteration:
                    break
                captured = self.clock()

                detections: Sequence[RawContact] = self.model.predict(image)
                inferred = self.clock()

                commit = self.stream.push(detections, frame)
                compiled = self.clock()

                self.injector.apply(commit)
                applied = self.clock()

                commits.append(commit)
                samples.append(
                    TickSample(
                        frame=frame,
                        target=target,
                        started=started,
                        captured=captured,
                        inferred=inferred,
                        compiled=compiled,
                        applied=applied,
                    )
                )
                frame += 1
        finally:
            self.source.close()

        if self.flush and not aborted:
            commit = self.stream.flush()
            if commit.ops:
                self.injector.apply(commit)
                commits.append(commit)

        stats = LoopStats(
            samples=samples,
            wall_s=self.clock() - t0,
            fps=self.fps,
            aborted=aborted,
            realtime=self.realtime,
        )
        return LoopResult(commits=commits, stats=stats)
