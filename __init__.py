"""BeatFlow virtual input module.

Vision-only Phigros agent: the model predicts touch actions, this package turns
them into real multitouch input on Linux/Wayland through ``/dev/uinput``.

Public surface
--------------
``Chart`` / ``LabelSet``      read the official label set (sparse JSONL)
``build_timeline``            label events -> press/move/release ops
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
"""

from __future__ import annotations

from .actions import Commit, InterpreterConfig, Op, Timeline, build_timeline
from .backends import DryRunBackend, TouchBackend, UinputBackend, make_backend
from .geometry import Geometry, Rect
from .injector import InjectionStats, Injector
from .labels import Chart, ChartEntry, LabelEvent, LabelSet, open_labelset
from .model_api import FramePrediction, PredictedContact, timeline_from_predictions
from .scheduler import TimelineRunner, TimingStats
from .slots import SlotAllocator

__version__ = "0.1.0"

__all__ = [
    "Chart",
    "ChartEntry",
    "Commit",
    "DryRunBackend",
    "FramePrediction",
    "Geometry",
    "InjectionStats",
    "Injector",
    "InterpreterConfig",
    "LabelEvent",
    "LabelSet",
    "Op",
    "PredictedContact",
    "Rect",
    "SlotAllocator",
    "Timeline",
    "TimelineRunner",
    "TimingStats",
    "TouchBackend",
    "UinputBackend",
    "build_timeline",
    "make_backend",
    "open_labelset",
    "timeline_from_predictions",
]
