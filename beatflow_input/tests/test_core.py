from __future__ import annotations

import unittest

from beatflow_input.geometry import Geometry, Rect
from beatflow_input.slots import SlotAllocator


class GeometryTests(unittest.TestCase):
    def test_fullscreen_corners(self):
        g = Geometry.fullscreen(1000, 500)
        self.assertEqual(g.to_device(-1, 1), (0, 0))        # top-left
        self.assertEqual(g.to_device(1, -1), (32767, 32767))  # bottom-right
        x, y = g.to_device(0, 0)
        self.assertAlmostEqual(x / 32767, 0.5, places=3)
        self.assertAlmostEqual(y / 32767, 0.5, places=3)

    def test_round_trip(self):
        g = Geometry.letterboxed(1920, 1080, 1024, 768)
        for gx, gy in [(-0.9, 0.2), (0.0, 0.0), (0.33, -0.75), (1.0, 1.0)]:
            dx, dy = g.to_device(gx, gy)
            bx, by = g.to_game(dx, dy)
            self.assertAlmostEqual(bx, gx, places=3)
            self.assertAlmostEqual(by, gy, places=3)

    def test_letterbox_centres_and_bars(self):
        g = Geometry.letterboxed(1920, 1080, 1024, 768)
        a = g.play_area
        self.assertAlmostEqual(a.aspect, 4 / 3, places=5)
        self.assertAlmostEqual(a.x, (1920 - 1440) / 2, places=5)  # 768*4/3 = 1024
        self.assertAlmostEqual(a.y, 0.0, places=5)
        # Centre of the play area is the centre of the output.
        self.assertAlmostEqual(g.to_device(0.0, 0.0)[0], 32767 / 2, delta=2)

    def test_clamping(self):
        g = Geometry.fullscreen(100, 100)
        self.assertEqual(g.to_device(-4.0, 9.0), (0, 0))
        raw = Geometry.fullscreen(100, 100, clamp=False)
        self.assertLess(raw.to_device(-4.0, 0.0)[0], 0)

    def test_invert_y(self):
        a = Geometry.fullscreen(100, 100, invert_y=True)
        b = Geometry.fullscreen(100, 100, invert_y=False)
        self.assertEqual(a.to_device(0, 1)[1], 0)
        self.assertEqual(b.to_device(0, 1)[1], 32767)

    def test_rejects_bad_area(self):
        with self.assertRaises(ValueError):
            Geometry(screen_w=100, screen_h=100, area=Rect(50, 50, 100, 100))


class SlotTests(unittest.TestCase):
    def test_allocates_distinct_slots_and_reuses(self):
        sa = SlotAllocator(4)
        a = sa.acquire(10)
        b = sa.acquire(11)
        self.assertNotEqual(a.slot, b.slot)
        self.assertNotEqual(a.tracking_id, b.tracking_id)
        self.assertEqual(sa.release(10), a.slot)
        self.assertEqual(sa.acquire(12).slot, a.slot)

    def test_drop_policy(self):
        sa = SlotAllocator(2, policy="drop")
        self.assertIsNotNone(sa.acquire(1))
        self.assertIsNotNone(sa.acquire(2))
        self.assertIsNone(sa.acquire(3))
        self.assertEqual(sa.stats.dropped, 1)
        self.assertEqual(len(sa), 2)

    def test_steal_lru(self):
        sa = SlotAllocator(2, policy="steal_lru")
        first = sa.acquire(1)
        sa.acquire(2)
        stolen = sa.acquire(3)
        self.assertEqual(stolen.evicted, 1)
        self.assertEqual(stolen.evicted_slot, first.slot)
        self.assertFalse(sa.is_active(1))
        self.assertTrue(sa.is_active(3))
        self.assertEqual(sa.stats.stolen, 1)

    def test_idempotent_acquire(self):
        sa = SlotAllocator(2)
        a = sa.acquire(7)
        b = sa.acquire(7)
        self.assertEqual((a.slot, a.tracking_id), (b.slot, b.tracking_id))
        self.assertEqual(sa.stats.presses, 1)

    def test_tracking_ids_wrap(self):
        sa = SlotAllocator(1, tracking_id_start=65534)
        a = sa.acquire(1)
        sa.release(1)
        b = sa.acquire(2)
        sa.release(2)
        c = sa.acquire(3)
        self.assertEqual(a.tracking_id, 65534)
        self.assertEqual(b.tracking_id, 65535)
        self.assertEqual(c.tracking_id, 1)


if __name__ == "__main__":
    unittest.main()
