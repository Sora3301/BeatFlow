from __future__ import annotations

import json
import unittest

from beatflow_input.actions import InterpreterConfig, build_timeline
from beatflow_input.backends import DryRunBackend
from beatflow_input.geometry import Geometry
from beatflow_input.injector import Injector
from beatflow_input.labels import Chart, parse_jsonl
from beatflow_input.scheduler import TimelineRunner


def chart_from(rows: list[dict]) -> Chart:
    text = "\n".join(json.dumps(r) for r in rows)
    frames = parse_jsonl(text)
    return Chart(name="synthetic", frames=frames, num_frames=max(frames) + 30)


def ev(frame, eid, phase, x=0.0, y=0.375, slot=0, **kw):
    d = {"id": eid, "slot": slot, "phase": phase, "x": x, "y": y}
    d.update(kw)
    return {"frame": frame, "events": [d]}


class TimelineTests(unittest.TestCase):
    def test_tap_press_then_release(self):
        tl = build_timeline(chart_from([ev(10, 1, "down")]), InterpreterConfig(tap_frames=2))
        kinds = [(c.frame, o.kind) for c in tl.commits for o in c.ops]
        self.assertEqual(kinds, [(10, "press"), (12, "release")])

    def test_hold_expires_after_grace(self):
        rows = [ev(f, 5, "down" if f == 10 else "hold") for f in range(10, 14)]
        tl = build_timeline(chart_from(rows), InterpreterConfig(hold_grace_frames=2))
        self.assertEqual([o.kind for c in tl.commits for o in c.ops], ["press", "release"])
        self.assertEqual(tl.commits[-1].frame, 13 + 2)

    def test_hold_does_not_move_when_pinned(self):
        rows = [
            {"frame": 10, "events": [{"id": 5, "slot": 0, "phase": "down", "x": -0.4, "y": 0.059}]},
        ] + [
            {"frame": f, "events": [{"id": 5, "slot": 0, "phase": "hold", "x": -0.4, "y": 0.059 + 0.05 * (f - 10)}]}
            for f in range(11, 16)
        ]
        tl = build_timeline(chart_from(rows))
        self.assertEqual([o.kind for c in tl.commits for o in c.ops], ["press", "release"])

    def test_hold_follow_moves(self):
        rows = [
            {"frame": 10, "events": [{"id": 5, "slot": 0, "phase": "down", "x": -0.4, "y": 0.1}]},
            {"frame": 11, "events": [{"id": 5, "slot": 0, "phase": "hold", "x": -0.4, "y": 0.2}]},
        ]
        tl = build_timeline(chart_from(rows), InterpreterConfig(hold_tracking="follow"))
        self.assertIn("move", [o.kind for c in tl.commits for o in c.ops])

    def test_flick_press_move_release(self):
        tl = build_timeline(chart_from([ev(4, 9, "flick", 0.2, 0.375)]),
                            InterpreterConfig(flick_frames=3, flick_delta=(0.0, 0.2)))
        ops = [(c.frame, o.kind) for c in tl.commits for o in c.ops]
        self.assertEqual(ops, [(4, "press"), (5, "move"), (7, "release")])
        move = next(o for c in tl.commits for o in c.ops if o.kind == "move")
        self.assertAlmostEqual(move.y, 0.575)

    def test_bad_events_dropped(self):
        tl = build_timeline(chart_from([ev(0, 1, "down", x=3.0, y=0.0)]))
        self.assertEqual(tl.n_ops, 0)
        tl2 = build_timeline(chart_from([ev(0, 1, "down", x=3.0, y=0.0)]),
                             InterpreterConfig(drop_bad=False))
        self.assertEqual(tl2.n_ops, 2)

    def test_simultaneous_notes_share_one_commit(self):
        row = {"frame": 7, "events": [
            {"id": 1, "slot": 0, "phase": "down", "x": -0.2, "y": 0.375},
            {"id": 2, "slot": 1, "phase": "down", "x": 0.2, "y": 0.375},
        ]}
        tl = build_timeline(chart_from([row]))
        press_commit = next(c for c in tl.commits if any(o.kind == "press" for o in c.ops))
        self.assertEqual(sum(1 for o in press_commit.ops if o.kind == "press"), 2)

    def test_repress_after_gap_uses_new_epoch(self):
        rows = [ev(0, 3, "down"), ev(20, 3, "down")]
        tl = build_timeline(chart_from(rows), InterpreterConfig(tap_frames=1))
        kinds = [o.kind for c in tl.commits for o in c.ops]
        self.assertEqual(kinds, ["press", "release", "press", "release"])

    def test_empty_chart(self):
        tl = build_timeline(Chart(name="x", frames={}))
        self.assertEqual(tl.commits, [])
        self.assertEqual(tl.duration, 0.0)


class InjectorTests(unittest.TestCase):
    def test_dry_run_roundtrip(self):
        rows = [ev(f, 1, "down" if f == 0 else "hold") for f in range(0, 5)]
        rows.append(ev(1, 2, "down", x=0.5))
        tl = build_timeline(chart_from(rows))
        backend = DryRunBackend(capacity=4)
        injector = Injector(backend, Geometry.fullscreen(1920, 1080))
        backend.open()
        TimelineRunner(injector, speed=0).run(tl)
        backend.close()
        self.assertEqual(injector.stats.presses, 2)
        self.assertEqual(injector.stats.releases, 2)
        self.assertEqual(injector.stats.dropped, 0)
        self.assertLessEqual(backend.summary()["max_simultaneous"], 4)

    def test_overflow_never_breaks_protocol(self):
        row = {"frame": 0, "events": [
            {"id": i, "slot": i, "phase": "down", "x": -0.8 + 0.2 * i, "y": 0.375} for i in range(6)
        ]}
        tl = build_timeline(chart_from([row]))
        backend = DryRunBackend(capacity=4)
        injector = Injector(backend, Geometry.fullscreen(1920, 1080), policy="steal_lru")
        backend.open()
        TimelineRunner(injector, speed=0).run(tl)
        backend.close()
        # 4 fit, the last 2 are steals -> 6 press ops on the wire.
        self.assertEqual(injector.stats.presses, 6)
        self.assertEqual(injector.stats.stolen, 2)
        self.assertEqual(injector.allocator.stats.peak_active, 4)

    def test_scheduler_records_every_commit(self):
        rows = [ev(f, 1, "down" if f == 0 else "hold") for f in range(0, 8)]
        tl = build_timeline(chart_from(rows))
        backend = DryRunBackend(capacity=4)
        injector = Injector(backend, Geometry.fullscreen(1920, 1080))
        backend.open()
        stats = TimelineRunner(injector, speed=0).run(tl)
        backend.close()
        self.assertEqual(len(stats.samples), len(tl.commits))
        self.assertFalse(stats.aborted)
        summary = stats.summary()
        self.assertIn("p50_us", summary["scheduler_lateness"])


if __name__ == "__main__":
    unittest.main()


class ModelApiTests(unittest.TestCase):
    def test_single_frame_contact_becomes_tap(self):
        from beatflow_input.model_api import FramePrediction, PredictedContact, timeline_from_predictions

        tl = timeline_from_predictions([
            FramePrediction(0, (PredictedContact(0, -0.4, 0.375),)),
            FramePrediction(1, ()),
        ])
        self.assertEqual([o.kind for c in tl.commits for o in c.ops], ["press", "release"])

    def test_multi_frame_contact_becomes_hold_and_moves(self):
        from beatflow_input.model_api import FramePrediction, PredictedContact, timeline_from_predictions

        tl = timeline_from_predictions([
            FramePrediction(0, (PredictedContact(0, -0.4, 0.20),)),
            FramePrediction(1, (PredictedContact(0, -0.4, 0.40),)),
            FramePrediction(2, (PredictedContact(0, -0.4, 0.60),)),
        ])
        kinds = [o.kind for c in tl.commits for o in c.ops]
        self.assertEqual(kinds.count("press"), 1)
        self.assertGreaterEqual(kinds.count("move"), 2)
        self.assertEqual(kinds[-1], "release")

    def test_rejects_out_of_range(self):
        from beatflow_input.model_api import PredictedContact

        with self.assertRaises(ValueError):
            PredictedContact(0, 1.5, 0.0)

    def test_validate_catches_duplicates(self):
        from beatflow_input.model_api import FramePrediction, PredictedContact, validate

        problems = validate([FramePrediction(0, (PredictedContact(1, 0, 0), PredictedContact(1, 0.1, 0)))])
        self.assertTrue(problems)
