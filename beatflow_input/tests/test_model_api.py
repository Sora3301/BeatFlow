"""Tests for the id-less model contract, the tracker and the real-time stream.

The centrepiece is :class:`ParityTests`: the batch path
(``predictions_from_raw`` + ``timeline_from_predictions``) and the real-time
path (``ModelStream``) must emit *the same ops on the same frames* for the same
detections. Everything else exists to pin down one rule that parity alone would
not catch.
"""

from __future__ import annotations

import unittest

from beatflow_input.actions import InterpreterConfig
from beatflow_input.model_api import (
    ModelStream,
    PredictedContact,
    RawContact,
    predictions_from_raw,
    timeline_from_predictions,
    validate,
)
from beatflow_input.tracking import (
    ContactTracker,
    Detection,
    TrackerConfig,
    normalise_kind,
)

TAP, HOLD, FLICK, DRAG = "tap", "hold", "flick", "drag"


def batch_ops(seq, cfg=None):
    frames = predictions_from_raw(seq)
    timeline = timeline_from_predictions(frames, cfg)
    return [
        (c.frame, o.kind, o.key, round(o.x, 4), round(o.y, 4))
        for c in timeline.commits
        for o in c.ops
    ]


def stream_ops(seq, cfg=None, tail=12):
    stream = ModelStream(cfg)
    out = []
    last = max(frame for frame, _ in seq)
    for frame, detections in seq:
        commit = stream.push(detections, frame)
        out += [(commit.frame, o.kind, o.key, round(o.x, 4), round(o.y, 4)) for o in commit.ops]
    for frame in range(last + 1, last + tail + 1):
        commit = stream.push((), frame)
        out += [(commit.frame, o.kind, o.key, round(o.x, 4), round(o.y, 4)) for o in commit.ops]
    return out


class ParityTests(unittest.TestCase):
    """Batch and real-time must not drift apart."""

    def test_tap(self):
        seq = [(0, [RawContact(-0.4, 0.375, TAP)]), (1, []), (2, []), (3, [])]
        self.assertEqual(batch_ops(seq), stream_ops(seq))

    def test_hold(self):
        seq = [(i, [RawContact(-0.4, 0.375, HOLD)]) for i in range(6)] + [(6, []), (7, [])]
        self.assertEqual(batch_ops(seq), stream_ops(seq))

    def test_single_frame_flick_is_a_swipe(self):
        seq = [(0, [RawContact(0.2, 0.375, FLICK)])] + [(i, []) for i in range(1, 6)]
        self.assertEqual(batch_ops(seq), stream_ops(seq))
        # press, then a move (that is what the game reads as a swipe), then release
        self.assertEqual([op[1] for op in batch_ops(seq)], ["press", "move", "release"])

    def test_repeated_flick_fires_once(self):
        seq = [(i, [RawContact(0.2, 0.375, FLICK)]) for i in range(4)] + [(i, []) for i in range(4, 8)]
        self.assertEqual(batch_ops(seq), stream_ops(seq))
        kinds = [op[1] for op in batch_ops(seq)]
        self.assertEqual(kinds.count("press"), 1)
        self.assertEqual(kinds.count("move"), 1)

    def test_moving_contact_emits_moves(self):
        seq = [(i, [RawContact(-0.4 + i * 0.1, 0.375, TAP)]) for i in range(5)] + [(i, []) for i in range(5, 8)]
        self.assertEqual(batch_ops(seq), stream_ops(seq))

    def test_two_fingers(self):
        seq = [(i, [RawContact(-0.6, 0.375, TAP), RawContact(0.4, 0.375, FLICK)]) for i in range(2)]
        seq += [(i, []) for i in range(2, 7)]
        self.assertEqual(batch_ops(seq), stream_ops(seq))

    def test_dropped_frame_does_not_split_a_hold(self):
        seq = [
            (0, [RawContact(-0.4, 0.375, HOLD)]),
            (1, []),                                   # one detection lost
            (2, [RawContact(-0.4, 0.375, HOLD)]),
            (3, [RawContact(-0.4, 0.375, HOLD)]),
            (4, []), (5, []),
        ]
        self.assertEqual(batch_ops(seq), stream_ops(seq))
        ops = batch_ops(seq)
        self.assertEqual(sum(1 for op in ops if op[1] == "press"), 1)

    def test_tap_refined_into_flick_restarts_as_swipe(self):
        seq = [
            (0, [RawContact(0.1, 0.375, TAP)]),
            (1, [RawContact(0.1, 0.375, FLICK)]),
            (2, [RawContact(0.1, 0.375, FLICK)]),
            (3, []), (4, []), (5, []),
        ]
        self.assertEqual(batch_ops(seq), stream_ops(seq))


class TrackerTests(unittest.TestCase):
    def test_first_frame_gets_left_to_right_ids(self):
        tracker = ContactTracker()
        out = tracker.update(0, [
            Detection(0.6, 0.375), Detection(-0.6, 0.375), Detection(0.0, 0.375),
        ])
        self.assertEqual([t.id for t in out], [0, 1, 2])
        by_x = {round(t.x, 1): t.id for t in out}
        self.assertLess(by_x[-0.6], by_x[0.0])
        self.assertLess(by_x[0.0], by_x[0.6])

    def test_ids_are_stable_across_frames(self):
        tracker = ContactTracker()
        tracker.update(0, [Detection(-0.6, 0.375), Detection(0.6, 0.375)])
        out = tracker.update(1, [Detection(-0.58, 0.375), Detection(0.62, 0.375)])
        self.assertEqual([t.id for t in out], [0, 1])
        self.assertEqual([t.age for t in out], [1, 1])

    def test_ids_are_never_reused(self):
        tracker = ContactTracker()
        tracker.update(0, [Detection(0.0, 0.375)])
        for frame in range(1, 8):
            tracker.update(frame, [])
        out = tracker.update(8, [Detection(0.0, 0.375)])
        self.assertEqual(out[0].id, 1)

    def test_track_survives_a_short_gap(self):
        tracker = ContactTracker(TrackerConfig(max_gap=2))
        tracker.update(0, [Detection(0.0, 0.375)])
        tracker.update(1, [])
        out = tracker.update(2, [Detection(0.02, 0.375)])
        self.assertEqual(out[0].id, 0)

    def test_track_dies_after_the_gap(self):
        tracker = ContactTracker(TrackerConfig(max_gap=1))
        tracker.update(0, [Detection(0.0, 0.375)])
        for frame in range(1, 5):
            tracker.update(frame, [])
        out = tracker.update(5, [Detection(0.0, 0.375)])
        self.assertEqual(out[0].id, 1)

    def test_lane_spacing_is_not_confused(self):
        tracker = ContactTracker()
        tracker.update(0, [Detection(x, 0.375) for x in (-0.6, -0.2, 0.2, 0.6)])
        out = tracker.update(1, [Detection(x + 0.03, 0.375) for x in (-0.6, -0.2, 0.2, 0.6)])
        self.assertEqual([t.id for t in out], [0, 1, 2, 3])

    def test_a_lifted_finger_does_not_steal_a_neighbour(self):
        tracker = ContactTracker()
        tracker.update(0, [Detection(-0.6, 0.375), Detection(0.0, 0.375)])
        out = tracker.update(1, [Detection(0.0, 0.375)])
        self.assertEqual(out[0].id, 1)

    def test_frames_must_advance(self):
        tracker = ContactTracker()
        tracker.update(5, [])
        with self.assertRaises(ValueError):
            tracker.update(5, [])


class ContractTests(unittest.TestCase):
    def test_kind_aliases(self):
        self.assertEqual(normalise_kind("DOWN"), TAP)
        self.assertEqual(normalise_kind("Hold_Note"), HOLD)
        self.assertEqual(Detection(0.0, 0.0, "down").kind, TAP)

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            Detection(0.0, 0.0, "banana")

    def test_out_of_range_is_rejected(self):
        with self.assertRaises(ValueError):
            Detection(1.5, 0.0)
        with self.assertRaises(ValueError):
            PredictedContact(0, 0.0, -1.2)

    def test_predicted_contact_keeps_the_old_positional_form(self):
        c = PredictedContact(3, -0.4, 0.375)
        self.assertEqual((c.id, c.x, c.y, c.kind), (3, -0.4, 0.375, TAP))

    def test_direction_is_carried_through(self):
        det = Detection(0.0, 0.375, FLICK, direction=(1.0, 0.0))
        tracked = ContactTracker().update(0, [det])
        self.assertEqual(tracked[0].direction, (1.0, 0.0))
        pred = predictions_from_raw([(0, [det])])[0]
        self.assertEqual(pred.contacts[0].direction, (1.0, 0.0))

    def test_validate_still_works(self):
        frames = predictions_from_raw([(0, [Detection(0.0, 0.0), Detection(0.1, 0.0)])])
        self.assertEqual(validate(frames), [])


class FlickTests(unittest.TestCase):
    def test_direction_controls_the_swipe(self):
        cfg = InterpreterConfig(hold_tracking="follow", flick_delta=(0.0, 0.2))
        seq = [(0, [RawContact(0.0, 0.375, FLICK, direction=(1.0, 0.0))]),
               (1, []), (2, []), (3, [])]
        ops = batch_ops(seq, cfg)
        move = next(op for op in ops if op[1] == "move")
        # normalised to the configured flick length, heading preserved
        self.assertGreater(move[3], 0.15)
        self.assertAlmostEqual(move[4], 0.375, places=3)

    def test_default_direction_when_model_gives_none(self):
        seq = [(0, [RawContact(0.0, 0.375, FLICK)]), (1, []), (2, []), (3, [])]
        move = next(op for op in batch_ops(seq) if op[1] == "move")
        self.assertGreater(move[4], 0.375)     # swiped upwards by default
        self.assertAlmostEqual(move[3], 0.0, places=3)

    def test_flick_releases_after_flick_frames(self):
        cfg = InterpreterConfig(hold_tracking="follow", flick_frames=4)
        seq = [(0, [RawContact(0.0, 0.375, FLICK)])] + [(i, []) for i in range(1, 6)]
        ops = batch_ops(seq, cfg)
        self.assertEqual([op[0] for op in ops], [0, 1, 4])


class StreamTests(unittest.TestCase):
    def test_push_returns_one_commit_per_frame(self):
        stream = ModelStream()
        for frame in range(4):
            commit = stream.push([RawContact(0.0, 0.375, TAP)] if frame == 0 else [])
            self.assertEqual(commit.frame, frame)
        self.assertEqual(stream.stats.frames, 4)

    def test_frames_must_advance(self):
        stream = ModelStream()
        stream.push([], 3)
        with self.assertRaises(ValueError):
            stream.push([], 3)

    def test_skipped_frames_are_replayed(self):
        stream = ModelStream()
        stream.push([RawContact(0.0, 0.375, FLICK)], 0)
        commit = stream.push([], 5)          # jump: the flick move must still fire
        kinds = [o.kind for o in commit.ops]
        self.assertIn("move", kinds)

    def test_flush_drains_everything(self):
        stream = ModelStream()
        stream.push([RawContact(0.0, 0.375, HOLD)], 0)
        stream.flush()
        self.assertEqual(stream.active_ids, ())

    def test_empty_stream_is_harmless(self):
        stream = ModelStream()
        commits = [stream.push([]) for _ in range(5)]
        self.assertTrue(all(c.is_empty for c in commits))

    def test_reset_clears_identity(self):
        stream = ModelStream()
        stream.push([RawContact(0.0, 0.375, TAP)], 0)
        stream.reset()
        commit = stream.push([RawContact(0.0, 0.375, TAP)], 0)
        self.assertEqual([o.key for o in commit.ops], [0])


if __name__ == "__main__":
    unittest.main()
