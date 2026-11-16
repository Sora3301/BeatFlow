"""Tests for the main program's model adapter and recorded replay."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from beatflow_input import RawContact

from beatflow_main.frame_source import Frame
from beatflow_main.model import RecordedModel, parse_contacts


class ParseContactsTests(unittest.TestCase):
    def test_builds_a_contact(self) -> None:
        contacts, rejected = parse_contacts([{"x": -0.4, "y": 0.375, "kind": "tap"}])
        self.assertEqual(rejected, 0)
        self.assertEqual(contacts, (RawContact(-0.4, 0.375, "tap"),))

    def test_out_of_range_is_dropped_not_clamped(self) -> None:
        contacts, rejected = parse_contacts([{"x": 1.5, "y": 0.0}])
        self.assertEqual(contacts, ())
        self.assertEqual(rejected, 1)

    def test_bad_value_is_dropped(self) -> None:
        contacts, rejected = parse_contacts(
            [{"x": None, "y": 0.0}, {"x": float("nan"), "y": 0.0}]
        )
        self.assertEqual(contacts, ())
        self.assertEqual(rejected, 2)

    def test_unknown_kind_is_dropped(self) -> None:
        contacts, rejected = parse_contacts([{"x": 0.0, "y": 0.0, "kind": "explode"}])
        self.assertEqual(contacts, ())
        self.assertEqual(rejected, 1)

    def test_direction_survives(self) -> None:
        contacts, rejected = parse_contacts(
            [{"x": 0.0, "y": 0.0, "kind": "flick", "direction": [0.6, 0.8]}]
        )
        self.assertEqual(rejected, 0)
        self.assertEqual(contacts[0].direction, (0.6, 0.8))

    def test_accepts_the_model_teams_events_schema(self) -> None:
        """infer_testing.py writes phase (not kind) plus slot/px/py/score."""
        contacts, rejected = parse_contacts([
            {"slot": 0, "phase": "down", "x": -0.4, "y": 0.375,
             "px": 96.0, "py": 123.8, "score": 0.99},
            {"slot": 2, "phase": "hold", "x": 0.1, "y": 0.2, "score": 0.8},
            {"slot": 3, "phase": "flick", "x": 0.3, "y": 0.1, "score": 0.7},
        ])
        self.assertEqual(rejected, 0)
        self.assertEqual([c.kind for c in contacts], ["tap", "hold", "flick"])
        self.assertEqual((contacts[0].x, contacts[0].y), (-0.4, 0.375))

    def test_non_mapping_row_is_counted_not_fatal(self) -> None:
        contacts, rejected = parse_contacts([["not", "a", "contact"]])
        self.assertEqual(contacts, ())
        self.assertEqual(rejected, 1)


class RecordedModelTests(unittest.TestCase):
    def test_missing_frames_are_empty(self) -> None:
        model = RecordedModel.from_sequence([(3, [RawContact(0.0, 0.0, "tap")])])
        self.assertEqual(model.frame_count, 4)
        self.assertEqual(model.predict(Frame(0, 0.0)), ())
        self.assertEqual(len(model.predict(Frame(3, 0.0))), 1)

    def test_jsonl_round_trip(self) -> None:
        rows = [
            {"frame": 0, "contacts": [{"x": -0.4, "y": 0.375, "kind": "tap"}]},
            {"frame": 2, "contacts": [
                {"x": 0.2, "y": 0.3, "kind": "flick", "direction": [0.8, 0.6]}
            ]},
            {"frame": 3, "contacts": [{"x": 9.0, "y": 0.0}]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preds.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            model = RecordedModel.from_jsonl(path)

        self.assertEqual(model.frame_count, 4)
        self.assertEqual(model.contact_count, 2)
        self.assertEqual(model.rejected, 1)
        flick = model.predict(Frame(2, 0.0))[0]
        self.assertEqual(flick.kind, "flick")
        self.assertEqual(flick.direction, (0.8, 0.6))

    def test_blank_lines_and_events_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preds.jsonl"
            path.write_text('\n\n{"frame": 1, "events": [{"x": 0, "y": 0}]}\n',
                            encoding="utf-8")
            model = RecordedModel.from_jsonl(path)
        self.assertEqual(model.frame_count, 2)

    def test_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preds.jsonl"
            path.write_text("", encoding="utf-8")
            model = RecordedModel.from_jsonl(path)
        self.assertEqual(model.frame_count, 0)
        self.assertEqual(model.contact_count, 0)

    def test_missing_frame_key_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preds.jsonl"
            path.write_text('{"contacts": []}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                RecordedModel.from_jsonl(path)

    def test_to_jsonl_round_trip(self) -> None:
        model = RecordedModel.from_sequence([
            (0, [RawContact(-0.4, 0.375, "tap")]),
            (2, [RawContact(0.2, 0.3, "flick", (0.8, 0.6))]),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preds.jsonl"
            self.assertEqual(model.to_jsonl(path), 2)
            again = RecordedModel.from_jsonl(path)
        self.assertEqual(again.frame_count, model.frame_count)
        self.assertEqual(again.contact_count, model.contact_count)
        self.assertEqual(again.predict(Frame(2, 0.0))[0].direction, (0.8, 0.6))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
