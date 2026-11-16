"""The loop, end to end, against the batch path and against a recording backend.

The parity test is the important one: it is the M1 acceptance criterion. If the
main loop ever disagrees with ``timeline_from_predictions`` about *which* ops
happen on *which* frame, this suite goes red before a game is involved.
"""

from __future__ import annotations

import unittest

from beatflow_input import Geometry, Injector, ModelStream, RawContact
from beatflow_input.backends import DryRunBackend
from beatflow_input.model_api import predictions_from_raw, timeline_from_predictions

from beatflow_main.config import TAIL_FRAMES
from beatflow_main.frame_source import SyntheticSource
from beatflow_main.loop import AgentLoop
from beatflow_main.model import RecordedModel


def batch_ops(sequence):
    timeline = timeline_from_predictions(predictions_from_raw(sequence))
    return [
        (c.frame, o.kind, o.key, round(o.x, 4), round(o.y, 4))
        for c in timeline.commits
        for o in c.ops
    ]


def run_loop(sequence, *, fps: float = 60.0, tail: int = TAIL_FRAMES):
    model = RecordedModel.from_sequence(sequence)
    total = model.frame_count + tail
    backend = DryRunBackend(capacity=10)
    backend.open()
    injector = Injector(backend, Geometry.fullscreen(1920, 1080), capacity=10)
    stream = ModelStream()
    loop = AgentLoop(
        SyntheticSource(total), model, injector, stream,
        fps=fps,
        realtime=False,
        max_frames=total,
    )
    result = loop.run()
    injector.release_all()
    backend.close()
    return result, backend, stream


class ParityTests(unittest.TestCase):
    """The live loop must emit exactly the batch path's ops, frame for frame."""

    def test_mixed_sequence(self) -> None:
        sequence = [
            (0, [RawContact(-0.40, 0.375, "tap")]),
            (1, []),
            (3, [RawContact(0.00, 0.375, "tap")]),
            (10, [RawContact(0.10, 0.375, "hold")]),
            (11, [RawContact(0.10, 0.375, "hold")]),
            (12, []),
            (20, [RawContact(0.20, 0.375, "flick", (0.8, 0.6))]),
            (21, []),
            (30, [RawContact(-0.2, 0.375, "tap"), RawContact(0.2, 0.375, "tap")]),
        ]
        result, _, _ = run_loop(sequence)
        self.assertEqual(result.ops(), batch_ops(sequence))

    def test_dropped_frame_does_not_split_a_hold(self) -> None:
        sequence = [
            (0, [RawContact(0.0, 0.0, "hold")]),
            (2, [RawContact(0.0, 0.0, "hold")]),
            (4, [RawContact(0.0, 0.0, "hold")]),
        ]
        result, _, _ = run_loop(sequence)
        self.assertEqual(result.ops(), batch_ops(sequence))

    def test_tap_that_keeps_being_seen_becomes_a_hold(self) -> None:
        sequence = [
            (0, [RawContact(0.0, 0.0, "tap")]),
            (1, [RawContact(0.0, 0.0, "tap")]),
            (2, [RawContact(0.0, 0.0, "tap")]),
        ]
        result, _, _ = run_loop(sequence)
        self.assertEqual(result.ops(), batch_ops(sequence))


class BehaviourTests(unittest.TestCase):
    def test_flick_presses_moves_then_releases(self) -> None:
        result, _, _ = run_loop([(0, [RawContact(0.0, 0.0, "flick", (1.0, 0.0))])])
        ops = result.ops()
        self.assertEqual([(f, k) for f, k, *_ in ops][:2], [(0, "press"), (1, "move")])
        self.assertEqual(ops[-1][1], "release")
        move = ops[1]
        self.assertAlmostEqual(move[3], 0.20, places=4)   # |flick_delta|
        self.assertAlmostEqual(move[4], 0.0, places=4)

    def test_held_contact_is_released_at_the_end(self) -> None:
        result, _, _ = run_loop([(0, [RawContact(0.0, 0.0, "hold")])])
        self.assertEqual(result.ops()[-1][1], "release")

    def test_every_op_reaches_the_backend(self) -> None:
        _, backend, _ = run_loop([(0, [RawContact(0.0, 0.0, "tap")])])
        kinds = [op.kind for op in backend.ops]
        self.assertIn("press", kinds)
        self.assertIn("release", kinds)
        self.assertEqual(kinds.count("press"), kinds.count("release"))

    def test_stats_cover_every_tick(self) -> None:
        result, _, _ = run_loop([(0, [RawContact(0.0, 0.0, "tap")])], tail=8)
        summary = result.stats.summary()
        self.assertEqual(summary["frames"], 9)
        self.assertFalse(summary["realtime"])
        self.assertEqual(summary["lateness"]["n"], 0)   # meaningless without pacing
        for key in ("capture", "infer", "compile", "inject", "tick"):
            self.assertEqual(summary[key]["n"], 9, key)

    def test_loop_runs_every_frame_of_the_recording(self) -> None:
        result, _, _ = run_loop(
            [(0, []), (1, []), (2, [RawContact(0.0, 0.0, "tap")])], tail=2
        )
        self.assertEqual(len(result.commits), 5)
        self.assertEqual(result.stats.summary()["frames"], 5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
