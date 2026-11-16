"""Decode a video file into frames with ffmpeg (no Python codec dependencies).

This is the offline frame source: it lets the real model + the real injection
path run over a recorded game video, without screen capture. The output is
RGB24 at the model's input size, which is exactly what
:class:`beatflow_main.cortex.CortexModel` expects.

Only ffmpeg is required (it is already needed to build the training videos).
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from .frame_source import Frame

__all__ = ["VideoSource", "ffmpeg_available"]


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


class VideoSource:
    """Streams ``file -> RGB24 frames`` through a ffmpeg pipe.

    ``size`` is ``(width, height)`` and must match the model's input size.
    ``max_frames`` limits decoding (0 = whole file). The pipe applies natural
    backpressure: if the loop consumes slower than ffmpeg decodes, ffmpeg
    blocks on write instead of buffering the whole video in RAM.

    The loop paces at its own ``fps``; pass ``--fps`` matching the video (or the
    game, for live capture) or the replay runs too fast/slow.
    """

    name = "video"

    def __init__(
        self,
        path: str | Path,
        size: tuple[int, int] = (320, 180),
        max_frames: int = 0,
        clock=time.monotonic,
    ) -> None:
        self.path = Path(path)
        self.width, self.height = int(size[0]), int(size[1])
        self.frame_bytes = self.width * self.height * 3
        self.max_frames = int(max_frames)
        self.clock = clock
        self._proc: subprocess.Popen[bytes] | None = None
        self._index = 0

    def open(self) -> None:
        if self._proc is not None:
            return
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if not ffmpeg_available():
            raise RuntimeError("ffmpeg is not on PATH; it is required by --frames")
        cmd = ["ffmpeg", "-v", "error", "-i", str(self.path), "-an"]
        if self.max_frames > 0:
            cmd += ["-frames:v", str(self.max_frames)]
        cmd += [
            "-vf", f"scale={self.width}:{self.height}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]
        self._proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            cmd, stdout=subprocess.PIPE,
        )

    def read(self) -> Frame:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("source is not open")
        data = self._proc.stdout.read(self.frame_bytes)
        if len(data) < self.frame_bytes:
            raise StopIteration
        frame = Frame(self._index, self.clock(), data)
        self._index += 1
        return frame

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:  # pragma: no cover - ffmpeg stuck
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
