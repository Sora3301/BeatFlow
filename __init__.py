"""BeatFlow virtual input module.

Vision-only Phigros agent: the model predicts touch actions, this package turns
them into real multitouch input on Linux/Wayland through ``/dev/uinput``.

Public surface
--------------
``Chart`` / ``LabelSet``      read the official label set (sparse JSONL)
``build_timeline``            label events -> press/move/release ops
``RawContact`` / ``ModelStream``  the model's per-frame output (no ids!) ->
                              commits, with ids tracked and flicks synthesised
``ContactTracker``            id-less detections -> stable contact ids
``Injector``                  ops -> physical MT slots -> device coordinates
``TimelineRunner``            real-time playback + lateness measurement
``UinputBackend``             the real device (``/dev/uinput``, MT protocol B)
``DryRunBackend``             recording backend for tests and replay
``Geometry``                  game-normalised -> screen -> device units

Quick start::

    python3 -m beatflow_input probe
    python3 -m beatflow_input plan --chart "Glitch"
    python3 -m beatflow_input replay --chart "Glitch" --backend dryrun
    python3 -m beatflow_input replay --chart "Glitch" --backend uinput
    python3 -m beatflow_input watch --chart "Glitch" --viz tk
"""

from __future__ import annotations

from .actions import Commit, InterpreterConfig, Op, Timeline, build_timeline
from .backends import DryRunBackend, TouchBackend, UinputBackend, make_backend
from .geometry import Geometry, Rect
from .injector import InjectionStats, Injector
from .labels import Chart, ChartEntry, LabelEvent, LabelSet, open_labelset
from .model_api import (
    FramePrediction,
    ModelStream,
    PredictedContact,
    RawContact,
    StreamStats,
    predictions_from_raw,
    timeline_from_predictions,
)
from .scheduler import TimelineRunner, TimingStats
from .slots import SlotAllocator
from .tracking import ContactTracker, Detection, TrackerConfig, Tracked

__version__ = "0.2.0"

__all__ = [
    "Chart",
    "ChartEntry",
    "Commit",
    "ContactTracker",
    "Detection",
    "DryRunBackend",
    "FramePrediction",
    "Geometry",
    "InjectionStats",
    "Injector",
    "InterpreterConfig",
    "LabelEvent",
    "LabelSet",
    "ModelStream",
    "Op",
    "PredictedContact",
    "RawContact",
    "Rect",
    "SlotAllocator",
    "StreamStats",
    "Timeline",
    "TimelineRunner",
    "TimingStats",
    "TouchBackend",
    "Tracked",
    "TrackerConfig",
    "UinputBackend",
    "build_timeline",
    "make_backend",
    "open_labelset",
    "predictions_from_raw",
    "timeline_from_predictions",
]
