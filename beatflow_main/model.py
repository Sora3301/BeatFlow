"""模型侧的契约，以及 M1 用的离线替身。

模型只干一件事：一帧进，一串 :class:`~beatflow_input.model_api.RawContact` 出，
**没有 id**。这个模块把「模型输出」和「主程序」之间的脏活收在一处：

* 构造 ``RawContact`` 时兜住越界 / 类型错误（注入层故意让越界抛错，主程序
  不能因为模型跑飞一帧就整场崩掉，所以这里丢弃并计数）；
* :class:`RecordedModel` 回放一份录好的预测，既让主循环在真模型到位前就能跑，
  也是「在线路径和批量路径必须一致」的回归基准。

录制的 JSONL 格式（一行一个非空帧；缺的帧 = 空帧）::

    {"frame": 40, "contacts": [{"x": -0.4, "y": 0.375, "kind": "tap"}]}
    {"frame": 41, "contacts": [{"x": -0.4, "y": 0.375, "kind": "flick", "direction": [0.9, 0.4]}]}

``contacts`` 也接受标签集里的别名 ``events``，方便直接把一份预测转成标签形状。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

from beatflow_input import RawContact

from .frame_source import Frame

__all__ = [
    "Model",
    "RecordedModel",
    "contact_or_none",
    "parse_contacts",
]


class Model(Protocol):
    """Per-frame predictor. The only method the loop calls."""

    def predict(self, frame: Frame) -> Sequence[RawContact]: ...


def _direction(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    dx, dy = value  # type: ignore[misc]
    return (float(dx), float(dy))


def contact_or_none(
    x: object,
    y: object,
    kind: object = "tap",
    direction: object = None,
) -> RawContact | None:
    """Build one contact, or return ``None`` when the contract rejects it.

    ``RawContact`` raises on out-of-range game coordinates on purpose (see
    ``beatflow_input.model_api``): a silently clamped value would hide a model
    that regressed raw pixels. In a live run the right response is to drop that
    contact and count it, not to kill the loop.
    """
    try:
        return RawContact(float(x), float(y), str(kind), _direction(direction))
    except (TypeError, ValueError):
        return None


def parse_contacts(rows: Iterable[Mapping[str, object]]) -> tuple[tuple[RawContact, ...], int]:
    """``[{"x": ..}, ..]`` -> ``(contacts, rejected_count)``.

    ``kind`` and ``phase`` are both accepted: the model team's
    ``infer_testing.py --out pred.jsonl`` writes ``phase`` (label vocabulary),
    while a hand-written dump usually writes ``kind``. ``down`` is the label
    name for a tap and is normalised by ``RawContact`` itself.

    Extra keys (``slot``, ``px``, ``py``, ``score``) are ignored on purpose:
    identity belongs to the tracker, and the screen-pixel projection is only
    there for the model team's visualisations.
    """
    contacts: list[RawContact] = []
    rejected = 0
    for row in rows or ():
        if not isinstance(row, Mapping):
            rejected += 1
            continue
        contact = contact_or_none(
            row.get("x"),
            row.get("y"),
            row.get("kind") or row.get("phase") or "tap",
            row.get("direction"),
        )
        if contact is None:
            rejected += 1
        else:
            contacts.append(contact)
    return tuple(contacts), rejected


class RecordedModel:
    """Replays a recorded per-frame prediction dump.

    ``predict`` is a pure lookup: the frame index the loop counts is the frame
    index the recording was made at. Missing frames are empty frames.
    """

    name = "recorded"

    def __init__(
        self,
        by_frame: Mapping[int, Sequence[RawContact]],
        rejected: int = 0,
        frame_count: int = 0,
    ) -> None:
        self._by_frame = {int(k): tuple(v) for k, v in by_frame.items()}
        self.rejected = rejected
        self._frames = sorted(self._by_frame)
        last = self._frames[-1] + 1 if self._frames else 0
        # A trailing row whose contacts were all rejected still marks the end of
        # the recording; otherwise the run would stop short of it.
        self._frame_count = max(last, int(frame_count))

    # -- construction ------------------------------------------------------
    @classmethod
    def from_sequence(
        cls,
        rows: Iterable[tuple[int, Sequence[RawContact]]],
    ) -> "RecordedModel":
        by_frame: dict[int, list[RawContact]] = {}
        end = 0
        for frame, contacts in rows:
            frame = int(frame)
            end = max(end, frame + 1)
            by_frame.setdefault(frame, []).extend(contacts)
        return cls(by_frame, frame_count=end)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "RecordedModel":
        by_frame: dict[int, list[RawContact]] = {}
        rejected = 0
        end = 0
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: bad JSON: {exc}") from exc
                try:
                    frame = int(row["frame"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{lineno}: missing integer 'frame'") from exc
                end = max(end, frame + 1)
                contacts, bad = parse_contacts(row.get("contacts") or row.get("events") or ())
                if contacts:
                    by_frame.setdefault(frame, []).extend(contacts)
                rejected += bad
        return cls(by_frame, rejected, frame_count=end)

    # -- inspection --------------------------------------------------------
    @property
    def frame_count(self) -> int:
        """One past the last frame mentioned by the recording."""
        return self._frame_count

    @property
    def contact_count(self) -> int:
        return sum(len(v) for v in self._by_frame.values())

    def frames(self) -> list[int]:
        return list(self._frames)

    # -- model protocol ----------------------------------------------------
    def predict(self, frame: Frame) -> Sequence[RawContact]:
        return self._by_frame.get(frame.index, ())

    # -- export ------------------------------------------------------------
    def to_jsonl(self, path: str | Path) -> int:
        """Write the accepted frames back out in the recorded-prediction schema."""
        rows = 0
        with open(path, "w", encoding="utf-8") as fh:
            for frame in self._frames:
                contacts = self._by_frame[frame]
                if not contacts:
                    continue
                fh.write(json.dumps(
                    {"frame": frame, "contacts": [_contact_row(c) for c in contacts]},
                    ensure_ascii=False,
                ) + "\n")
                rows += 1
        return rows


def _contact_row(contact: RawContact) -> dict:
    row: dict[str, object] = {
        "x": round(contact.x, 6),
        "y": round(contact.y, 6),
        "kind": contact.kind,
    }
    if contact.direction is not None:
        row["direction"] = [round(v, 6) for v in contact.direction]
    return row
