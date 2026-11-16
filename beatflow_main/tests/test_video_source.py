"""VideoSource: ffmpeg pipe -> RGB frames. Skipped when ffmpeg is absent."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from beatflow_main.video_source import VideoSource, ffmpeg_available


def _make_video(path: Path, frames: int = 10, width: int = 64, height: int = 36) -> None:
    subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate=10",
            "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


@unittest.skipUnless(ffmpeg_available(), "ffmpeg not installed")
class VideoSourceTests(unittest.TestCase):
    def test_reads_every_frame_at_the_requested_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.mp4"
            _make_video(path)
            source = VideoSource(path, size=(64, 36))
            source.open()
            frames = []
            try:
                while True:
                    frames.append(source.read())
            except StopIteration:
                pass
            finally:
                source.close()

        self.assertEqual(len(frames), 10)
        self.assertEqual([f.index for f in frames], list(range(10)))
        self.assertTrue(all(len(f.image) == 64 * 36 * 3 for f in frames))

    def test_max_frames_stops_early(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.mp4"
            _make_video(path)
            source = VideoSource(path, size=(64, 36), max_frames=3)
            source.open()
            frames = []
            try:
                for _ in range(3):
                    frames.append(source.read())
                with self.assertRaises(StopIteration):
                    source.read()
            finally:
                source.close()
        self.assertEqual(len(frames), 3)

    def test_missing_file_is_an_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            VideoSource("/definitely/not/here.mp4").open()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
