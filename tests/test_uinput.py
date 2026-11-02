from __future__ import annotations

import os
import unittest

from beatflow_input.backends import BackendUnavailable, UinputBackend
from beatflow_input.geometry import Geometry


def uinput_available() -> bool:
    path = "/dev/uinput"
    if not os.path.exists(path):
        return False
    try:
        os.close(os.open(path, os.O_WRONLY | os.O_NONBLOCK))
        return True
    except OSError:
        return False


@unittest.skipUnless(uinput_available(), "no writable /dev/uinput")
class UinputTests(unittest.TestCase):
    def test_create_destroy(self):
        backend = UinputBackend(capacity=4, device_name="BeatFlow Test Device")
        backend.open()
        try:
            self.assertTrue(backend.is_open)
            self.assertEqual(backend.capacity, 4)
            status = backend.status()
            self.assertEqual(status["capacity"], 4)
        finally:
            backend.close()
        self.assertFalse(backend.is_open)

    def test_press_move_release(self):
        g = Geometry.fullscreen(1920, 1080)
        backend = UinputBackend(capacity=4, device_name="BeatFlow Test Device")
        backend.open()
        try:
            for slot in range(4):
                x, y = g.to_device(-0.9 + 0.6 * slot, 0.375)
                backend.press(slot, 1000 + slot, x, y)
            backend.commit()
            backend.move(0, *g.to_device(0.5, 0.5))
            backend.commit()
            for slot in range(4):
                backend.release(slot)
            backend.commit()
        finally:
            backend.close()

    def test_rejects_slot_overflow(self):
        backend = UinputBackend(capacity=2, device_name="BeatFlow Test Device")
        backend.open()
        try:
            with self.assertRaises(ValueError):
                backend.press(2, 1, 0, 0)
        finally:
            backend.close()

    def test_unavailable_path(self):
        backend = UinputBackend(device_path="/dev/does-not-exist")
        with self.assertRaises(BackendUnavailable):
            backend.open()


if __name__ == "__main__":
    unittest.main()
