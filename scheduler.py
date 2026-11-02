"""Real-time playback of a :class:`~beatflow_input.actions.Timeline`.

A naive ``sleep(t)`` loop drifts by several milliseconds per frame: ``sleep``
only guarantees "at least this long" and the scheduler adds jitter on top.
Phigros judges notes well inside one 16.7 ms frame, so the scheduler here uses
the usual hybrid:

* coarse sleep while more than ``spin_margin`` remains, then
* a busy spin for the last ~1.5 ms, which lands within tens of microseconds.

Everything is measured against an absolute ``t0`` rather than accumulated
deltas, so the error does not compound over a 7-minute chart.
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field

from .actions import Commit, Timeline
from .injector import Injector

__all__ = ["TimingSample", "TimingStats", "TimelineRunner"]


@dataclass(frozen=True)
class TimingSample:
    index: int
    frame: int
    target: float       # absolute monotonic deadline
    started: float      # right before the backend ops
    finished: float     # right after commit()

    @property
    def late_us(self) -> float:
        return (self.started - self.target) * 1e6

    @property
    def apply_us(self) -> float:
        return (self.finished - self.started) * 1e6


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


@dataclass
class TimingStats:
    samples: list[TimingSample] = field(default_factory=list)
    wall_time: float = 0.0
    aborted: bool = False
    frame_period_us: float = 16_666.7

    def add(self, sample: TimingSample) -> None:
        self.samples.append(sample)

    def _block(self, values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        return {
            "n": len(values),
            "mean_us": round(statistics.fmean(values), 2),
            "p50_us": round(_pct(values, 0.50), 2),
            "p95_us": round(_pct(values, 0.95), 2),
            "p99_us": round(_pct(values, 0.99), 2),
            "max_us": round(max(values), 2),
            "min_us": round(min(values), 2),
            "stdev_us": round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0,
        }

    def summary(self) -> dict:
        late = [s.late_us for s in self.samples]
        apply = [s.apply_us for s in self.samples]
        over_frame = sum(1 for v in late if v > self.frame_period_us)
        early = sum(1 for v in late if v < -self.frame_period_us)
        return {
            "commits": len(self.samples),
            "wall_time_s": round(self.wall_time, 3),
            "aborted": self.aborted,
            "scheduler_lateness": self._block(late),
            "injection_cost": self._block(apply),
            "commits_later_than_one_frame": over_frame,
            "commits_earlier_than_one_frame": early,
        }

    def format(self) -> str:
        s = self.summary()
        late, cost = s["scheduler_lateness"], s["injection_cost"]
        lines = [
            f"commits            {s['commits']}   wall {s['wall_time_s']:.3f}s"
            + ("   (ABORTED)" if s["aborted"] else ""),
            "lateness (target -> ops start)",
            f"  p50 {late.get('p50_us', 0):>8.1f} us   p95 {late.get('p95_us', 0):>8.1f} us"
            f"   p99 {late.get('p99_us', 0):>8.1f} us   max {late.get('max_us', 0):>9.1f} us",
            "injection cost (ops -> commit returned)",
            f"  p50 {cost.get('p50_us', 0):>8.1f} us   p95 {cost.get('p95_us', 0):>8.1f} us"
            f"   p99 {cost.get('p99_us', 0):>8.1f} us   max {cost.get('max_us', 0):>9.1f} us",
            f"commits >1 frame late: {s['commits_later_than_one_frame']}"
            f"   >1 frame early: {s['commits_earlier_than_one_frame']}",
        ]
        return "\n".join(lines)


class TimelineRunner:
    """Plays commits at their absolute deadlines.

    ``speed``: ``1.0`` realtime, ``0`` as fast as possible (record/replay
    tests), ``2.0`` twice as fast (useful to stress the scheduler).
    """

    def __init__(
        self,
        injector: Injector,
        speed: float = 1.0,
        lead_in: float = 0.5,
        spin_margin_us: float = 1500.0,
        abort: threading.Event | None = None,
        on_commit=None,
    ) -> None:
        if speed < 0:
            raise ValueError("speed must be >= 0")
        self.injector = injector
        self.speed = speed
        self.lead_in = lead_in
        self.spin_margin = spin_margin_us / 1e6
        self.abort = abort or threading.Event()
        self.on_commit = on_commit

    def _wait_until(self, target: float) -> None:
        while True:
            remaining = target - time.monotonic()
            if remaining <= 0:
                return
            if remaining > self.spin_margin:
                time.sleep(remaining - self.spin_margin)
            else:
                break
        while time.monotonic() < target:
            pass

    def run(self, timeline: Timeline) -> TimingStats:
        fps = timeline.fps
        stats = TimingStats(frame_period_us=1e6 / fps)
        started_wall = time.monotonic()
        t0 = started_wall + self.lead_in

        for index, commit in enumerate(timeline.commits):
            if self.abort.is_set():
                stats.aborted = True
                break
            target = t0 + commit.t / self.speed if self.speed else time.monotonic()
            if self.speed:
                self._wait_until(target)
            before = time.monotonic()
            self.injector.apply(commit)
            after = time.monotonic()
            sample = TimingSample(index, commit.frame, target, before, after)
            stats.add(sample)
            if self.on_commit is not None:
                self.on_commit(sample, commit)

        stats.wall_time = time.monotonic() - started_wall
        return stats
