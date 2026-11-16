"""Tests for the cortex-sama adapter that do not need torch.

The tensor path can only be exercised on a machine with a 3.12 venv and the
model repo; everything that can be pure is kept pure so it is still covered
here.
"""

from __future__ import annotations

import unittest

from beatflow_input import RawContact

from beatflow_main.cortex import (
    TYPE_DOWN,
    TYPE_FLICK,
    TYPE_HOLD,
    TYPE_IDLE,
    CortexModel,
    contacts_from_types,
    torch_available,
)


class ContactsFromTypesTests(unittest.TestCase):
    def test_maps_the_three_action_types(self) -> None:
        contacts = contacts_from_types(
            types=[TYPE_IDLE, TYPE_DOWN, TYPE_HOLD, TYPE_FLICK],
            coords=[(0.0, 0.0), (-0.4, 0.375), (0.1, 0.2), (0.3, 0.1)],
        )
        self.assertEqual([c.kind for c in contacts], ["tap", "hold", "flick"])
        self.assertEqual(contacts[0], RawContact(-0.4, 0.375, "tap"))

    def test_idle_slots_are_skipped(self) -> None:
        contacts = contacts_from_types([TYPE_IDLE, TYPE_IDLE], [(0.9, -0.9), (0.0, 0.0)])
        self.assertEqual(contacts, ())

    def test_min_score_drops_low_confidence_slots(self) -> None:
        probs = [
            (0.1, 0.8, 0.05, 0.05),   # down, confident
            (0.1, 0.2, 0.65, 0.05),   # hold, below a 0.8 threshold
        ]
        contacts = contacts_from_types(
            [TYPE_DOWN, TYPE_HOLD],
            [(0.0, 0.0), (0.1, 0.1)],
            probs=probs,
            min_score=0.8,
        )
        self.assertEqual(len(contacts), 1)
        self.assertEqual(contacts[0].kind, "tap")

    def test_out_of_range_coordinate_is_dropped(self) -> None:
        contacts = contacts_from_types([TYPE_DOWN], [(1.4, 0.0)])
        self.assertEqual(contacts, ())

    def test_unknown_type_is_ignored(self) -> None:
        contacts = contacts_from_types([7], [(0.0, 0.0)])
        self.assertEqual(contacts, ())


@unittest.skipIf(torch_available(), "torch is installed; this checks the fallback")
class WithoutTorchTests(unittest.TestCase):
    def test_load_explains_the_missing_venv(self) -> None:
        model = CortexModel(model_dir=".", ckpt="checkpoints/none.pt")
        with self.assertRaises(RuntimeError) as ctx:
            model.load()
        message = str(ctx.exception)
        self.assertIn("PyTorch", message)
        self.assertIn("3.12", message)

    def test_a_registered_model_has_no_frames(self) -> None:
        model = CortexModel(model_dir=".", ckpt="checkpoints/none.pt")
        self.assertEqual(model.name, "cortex")
        self.assertEqual(model.rejected, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
