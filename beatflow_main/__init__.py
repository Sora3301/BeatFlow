"""BeatFlow 主程序：把「采集 → 模型 → 注入」串成一条 60 Hz 的实时回路。

注入层（``beatflow_input``）只负责「给定每帧的手指位置，把它变成内核里的
多点触控事件」。这个包负责它前面的半段：

* 从帧源拿到一帧画面（M1 用合成帧，真实采集后续接在这里）；
* 让模型对每帧吐出 ``list[RawContact]``（没有 id）；
* 按 60 Hz 的绝对 deadline 把结果喂给 ``ModelStream`` 并立即注入；
* 记录采集 / 推理 / 注入 / tick 四段耗时，作为调参依据。

入口::

    python3 main.py demo                       # 内置序列，不需要任何数据文件
    python3 main.py run --recorded preds.jsonl # 回放一份录好的模型输出
    python3 main.py run --recorded preds.jsonl --backend uinput
"""

from __future__ import annotations

from .config import TAIL_FRAMES, RunConfig
from .cortex import CortexModel, contacts_from_types, torch_available
from .demo import demo_model
from .frame_source import Frame, FrameSource, SyntheticSource
from .loop import AgentLoop, LoopResult, LoopStats, TickSample, wait_until
from .model import Model, RecordedModel, contact_or_none, parse_contacts
from .video_source import VideoSource, ffmpeg_available

__all__ = [
    "TAIL_FRAMES",
    "AgentLoop",
    "CortexModel",
    "Frame",
    "FrameSource",
    "LoopResult",
    "LoopStats",
    "Model",
    "RecordedModel",
    "RunConfig",
    "SyntheticSource",
    "TickSample",
    "VideoSource",
    "contact_or_none",
    "contacts_from_types",
    "demo_model",
    "ffmpeg_available",
    "parse_contacts",
    "torch_available",
    "wait_until",
]
