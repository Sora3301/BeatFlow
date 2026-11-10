"""Reader for the official ``map-set-labels`` label set.

Two supported sources:

* ``map-set-labels.zip`` (what we actually have) - read straight from the
  archive with :mod:`zipfile`, nothing is extracted;
* an extracted directory - same files on disk.

Schema (authoritative, from ``prepare_labels.py`` in the same archive)::

    {"frame": 102, "events": [
        {"id": 115, "slot": 0, "phase": "down", "x": -0.4, "y": 0.375},
        {"id": 115, "slot": 0, "phase": "hold", "x": -0.4, "y": 0.40},
        {"id": 116, "slot": 1, "phase": "flick", "x": 0.2, "y": 0.375, "bad": true}
    ]}

Facts that matter downstream (all verified against the archive):

* ``phase`` is only ``down`` / ``hold`` / ``flick`` - **there is no ``up``**;
* frames are **sparse**: a frame with no events is entirely idle and is simply
  absent from the file;
* frame numbers may skip - never assume ``frame == index``;
* ``bad`` marks out-of-range coordinates (~0.68 % of events);
* ``manifest.json`` carries the true chart length in ``num_frames``, which is
  larger than the last labelled frame (the tail is idle).
"""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

__all__ = ["LabelEvent", "Chart", "ChartEntry", "LabelSet", "open_labelset", "parse_jsonl"]

PHASES = ("down", "hold", "flick")


@dataclass(frozen=True)
class LabelEvent:
    frame: int
    id: int
    slot: int
    phase: str
    x: float
    y: float
    bad: bool = False

    @property
    def out_of_range(self) -> bool:
        return abs(self.x) > 1.0 or abs(self.y) > 1.0


@dataclass(frozen=True)
class ChartEntry:
    """One row of ``manifest.json``."""

    folder: str
    label_name: str      # basename inside the archive
    num_frames: int
    num_events: int
    n_slots: int
    max_slots: int
    invalid_frames: int = 0

    @property
    def duration(self) -> float:
        return self.num_frames / 60.0


@dataclass
class Chart:
    name: str
    frames: dict[int, tuple[LabelEvent, ...]] = field(default_factory=dict)
    num_frames: int = 0
    meta: ChartEntry | None = None

    @property
    def events(self) -> int:
        return sum(len(v) for v in self.frames.values())

    @property
    def first_frame(self) -> int:
        return min(self.frames) if self.frames else 0

    @property
    def last_frame(self) -> int:
        return max(self.frames) if self.frames else 0

    def iter_events(self) -> Iterator[LabelEvent]:
        for frame in sorted(self.frames):
            yield from self.frames[frame]

    def phase_counts(self) -> dict[str, int]:
        counts = {p: 0 for p in PHASES}
        for ev in self.iter_events():
            counts[ev.phase] = counts.get(ev.phase, 0) + 1
        return counts


def parse_jsonl(text: str) -> dict[int, tuple[LabelEvent, ...]]:
    """Parse the JSONL payload into ``frame -> events``."""
    frames: dict[int, list[LabelEvent]] = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"bad JSON on line {lineno}: {exc}") from exc
        frame = int(row["frame"])
        bucket = frames.setdefault(frame, [])
        for e in row.get("events", ()):
            phase = e.get("phase")
            if phase not in PHASES:
                raise ValueError(f"line {lineno}: unknown phase {phase!r}")
            x = float(e.get("x", 0.0))
            y = float(e.get("y", 0.0))
            bucket.append(
                LabelEvent(
                    frame=frame,
                    id=int(e.get("id", 0)),
                    slot=int(e.get("slot", 0)),
                    phase=phase,
                    x=x,
                    y=y,
                    bad=bool(e.get("bad", False)) or abs(x) > 1.0 or abs(y) > 1.0,
                )
            )
    return {f: tuple(v) for f, v in frames.items()}


class LabelSet:
    """Read-only view over the label archive or an extracted folder."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._zip = zipfile.ZipFile(self.path) if self.path.is_file() else None
        self._dir = None if self._zip else self.path
        self._entries: list[ChartEntry] = []
        self._by_name: dict[str, ChartEntry] = {}
        self._load_manifest()

    # -- construction ------------------------------------------------------
    def __enter__(self) -> "LabelSet":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def _names(self) -> list[str]:
        if self._zip is not None:
            return self._zip.namelist()
        return [str(p.relative_to(self._dir)) for p in self._dir.rglob("*") if p.is_file()]

    def _read_bytes(self, name: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(name)
        return (self._dir / name).read_bytes()

    def _load_manifest(self) -> None:
        names = self._names()
        manifest = next((n for n in names if n.endswith("manifest.json")), None)
        if manifest is None:
            # Plain directory of .jsonl files: synthesise entries.
            for n in names:
                if n.endswith(".jsonl"):
                    self._register(ChartEntry(os.path.dirname(n), os.path.basename(n), 0, 0, 4, 4))
            return
        rows = json.loads(self._read_bytes(manifest).decode("utf-8"))
        for row in rows:
            label = os.path.basename(str(row.get("labels", "")))
            if not label:
                continue
            self._register(
                ChartEntry(
                    folder=str(row.get("folder", "")),
                    label_name=label,
                    num_frames=int(row.get("num_frames", 0)),
                    num_events=int(row.get("num_events", 0)),
                    n_slots=int(row.get("n_slots", 4)),
                    max_slots=int(row.get("max_slots", 4)),
                    invalid_frames=len(row.get("invalid_frames", []) or []),
                )
            )

    def _register(self, entry: ChartEntry) -> None:
        self._entries.append(entry)
        self._by_name[entry.label_name] = entry

    # -- lookup ------------------------------------------------------------
    @property
    def entries(self) -> list[ChartEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[ChartEntry]:
        return iter(self._entries)

    def find(self, query: str) -> list[ChartEntry]:
        """Substring match on folder / label name (case-insensitive)."""
        q = query.lower()
        return [e for e in self._entries if q in e.folder.lower() or q in e.label_name.lower()]

    def resolve(self, query: str | None = None) -> ChartEntry:
        """Pick exactly one chart by name, path basename, or substring."""
        if not self._entries:
            raise LookupError("label set is empty")
        if query is None:
            return self._entries[0]
        base = os.path.basename(query)
        if base in self._by_name:
            return self._by_name[base]
        hits = self.find(query)
        if not hits:
            raise LookupError(f"no chart matches {query!r}")
        if len(hits) > 1:
            names = ", ".join(h.folder for h in hits[:5])
            raise LookupError(f"{query!r} is ambiguous ({len(hits)} matches: {names}, ...)")
        return hits[0]

    def read_text(self, entry: ChartEntry) -> str:
        names = self._names()
        target = next(
            (n for n in names if n.replace("\\", "/").endswith("/" + entry.label_name) or n == entry.label_name),
            None,
        )
        if target is None:
            raise FileNotFoundError(entry.label_name)
        return self._read_bytes(target).decode("utf-8")

    def chart(self, query: str | None = None) -> Chart:
        entry = self.resolve(query)
        frames = parse_jsonl(self.read_text(entry))
        num_frames = entry.num_frames or (max(frames) + 1 if frames else 0)
        return Chart(name=entry.folder or entry.label_name, frames=frames, num_frames=num_frames, meta=entry)


def open_labelset(path: str | os.PathLike[str]) -> LabelSet:
    """Convenience wrapper around :class:`LabelSet`."""
    return LabelSet(path)
