"""RunConfig -> Geometry: the y-direction knob and the three window modes."""

from __future__ import annotations

import unittest

from beatflow_main.config import RunConfig


class GeometryTests(unittest.TestCase):
    def test_plus_y_up_puts_a_high_y_above_the_centre(self) -> None:
        geometry = RunConfig(invert_y=True).geometry()
        _, sy = geometry.to_screen(0.0, 0.375)
        self.assertLess(sy, 540.0)

    def test_no_invert_y_treats_high_y_as_lower_on_screen(self) -> None:
        geometry = RunConfig(invert_y=False).geometry()
        _, sy = geometry.to_screen(0.0, 0.375)
        self.assertGreater(sy, 540.0)

    def test_window_mode(self) -> None:
        geometry = RunConfig(window=(300, 300, 600, 600)).geometry()
        self.assertEqual(geometry.play_area.x, 300)
        self.assertEqual(geometry.play_area.w, 600)
        # x=-1 is the left edge of the play area
        self.assertAlmostEqual(geometry.to_screen(-1.0, 0.0)[0], 300.0)

    def test_letterbox_mode_is_centred(self) -> None:
        geometry = RunConfig(screen=(1920, 1080), game_size=(1024, 768)).geometry()
        area = geometry.play_area
        self.assertAlmostEqual(area.w / area.h, 1024 / 768)
        self.assertAlmostEqual(area.x, (1920 - area.w) / 2)

    def test_window_and_game_size_are_exclusive(self) -> None:
        with self.assertRaises(ValueError):
            RunConfig(window=(0, 0, 10, 10), game_size=(640, 480))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
