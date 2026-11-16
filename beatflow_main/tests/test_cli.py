"""Smoke tests for the main-program CLI (no device, no data files)."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from beatflow_main.cli import main
from beatflow_main.config import TAIL_FRAMES
from beatflow_main.cortex import torch_available
from beatflow_main.demo import DEMO_FRAMES


def _last_json(text: str) -> dict:
    return json.loads(text.strip().splitlines()[-1])


class CliTests(unittest.TestCase):
    def test_demo_runs_without_data_files(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = main(["demo", "--no-realtime", "--json"])
        self.assertEqual(rc, 0)
        payload = _last_json(out.getvalue())
        self.assertEqual(payload["loop"]["frames"], DEMO_FRAMES)
        self.assertGreater(payload["injector"]["presses"], 0)
        self.assertEqual(
            payload["injector"]["presses"], payload["injector"]["releases"]
        )

    def test_run_recorded_file(self) -> None:
        rows = [
            {"frame": 10, "contacts": [{"x": -0.4, "y": 0.375, "kind": "tap"}]},
            {"frame": 40, "contacts": [{"x": 0.0, "y": 0.375, "kind": "hold"}]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preds.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = main(["run", "--recorded", str(path), "--no-realtime", "--json"])
        self.assertEqual(rc, 0)
        payload = _last_json(out.getvalue())
        self.assertEqual(payload["loop"]["frames"], 41 + TAIL_FRAMES)

    def test_missing_file_is_a_clean_error(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            rc = main(["run", "--recorded", "/definitely/not/here.jsonl", "--no-realtime"])
        self.assertEqual(rc, 2)

    def test_frames_requires_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "v.mp4"
            video.write_bytes(b"")
            with contextlib.redirect_stderr(io.StringIO()):
                rc = main(["run", "--frames", str(video), "--no-realtime"])
        self.assertEqual(rc, 2)

    @unittest.skipIf(torch_available(), "torch is installed")
    def test_frames_without_torch_explains_the_venv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "v.mp4"
            video.write_bytes(b"")
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                rc = main([
                    "run", "--frames", str(video),
                    "--model-dir", ".", "--ckpt", "none.pt",
                    "--no-realtime",
                ])
        self.assertEqual(rc, 1)
        self.assertIn("PyTorch", err.getvalue())

    def test_dump_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dump = Path(tmp) / "ops.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = main(["demo", "--no-realtime", "--json", "--dump", str(dump)])
            self.assertEqual(rc, 0)
            lines = [json.loads(line) for line in dump.read_text().splitlines() if line]
        self.assertTrue(any(row["kind"] == "press" for row in lines))
        self.assertTrue(any(row["kind"] == "release" for row in lines))

    def test_demo_export_feeds_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preds.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                rc_export = main(
                    ["demo", "--no-realtime", "--json", "--export", str(path)]
                )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc_run = main(
                    ["run", "--recorded", str(path), "--no-realtime", "--json"]
                )
        self.assertEqual((rc_export, rc_run), (0, 0))
        payload = _last_json(out.getvalue())
        self.assertEqual(payload["injector"]["presses"], 8)
        self.assertEqual(payload["injector"]["presses"], payload["injector"]["releases"])

    def test_consumes_infer_testing_predictions(self) -> None:
        """The model team's pred.jsonl -> presses/holds/flicks end to end."""
        rows = [
            {"frame": 10, "events": [
                {"slot": 0, "phase": "down", "x": -0.4, "y": 0.375, "score": 0.99}
            ]},
            *[
                {"frame": frame, "events": [
                    {"slot": 1, "phase": "hold", "x": 0.1, "y": 0.2, "score": 0.9}
                ]}
                for frame in range(20, 26)
            ],
            {"frame": 30, "events": [
                {"slot": 2, "phase": "flick", "x": 0.3, "y": 0.1, "score": 0.8}
            ]},
            {"frame": 31, "events": []},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pred.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = main(["run", "--recorded", str(path), "--no-realtime", "--json"])
        self.assertEqual(rc, 0)
        payload = _last_json(out.getvalue())
        self.assertEqual(payload["injector"]["presses"], 3)
        self.assertEqual(payload["injector"]["releases"], 3)
        self.assertEqual(payload["injector"]["moves"], 1)   # the flick


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
