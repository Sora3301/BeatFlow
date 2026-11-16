"""Tests for the visualizer plugin (no display, no /dev/uinput, no timing)."""

from __future__ import annotations

import json
import struct
import unittest
from types import SimpleNamespace

from beatflow_input.labels import Chart, LabelEvent
from beatflow_input.visualizer import (
    RENDERERS,
    HOLD_AFTER,
    JUDGMENT_LINE_Y,
    LiveModel,
    Scene,
    Viewport,
    available,
    finger_color,
    phase_map,
)
from beatflow_input.visualizer.protocol import (
    ContactState,
    FrameState,
    encode,
    hello,
    parse,
)
from beatflow_input.visualizer.terminal import (
    GLYPHS_ASCII,
    GLYPHS_UNICODE,
    TerminalCanvas,
    build_screen,
    demo_frames,
)


def make_chart() -> Chart:
    frames = {
        10: (
            LabelEvent(frame=10, id=1, slot=0, phase="down", x=-0.4, y=0.375),
            LabelEvent(frame=10, id=2, slot=1, phase="down", x=-0.2, y=0.375),
        ),
        11: (LabelEvent(frame=11, id=2, slot=1, phase="hold", x=-0.2, y=0.375),),
        12: (LabelEvent(frame=12, id=3, slot=2, phase="flick", x=0.2, y=0.375),),
        13: (LabelEvent(frame=13, id=9, slot=3, phase="down", x=9.0, y=0.0, bad=True),),
    }
    return Chart(name="test", frames=frames, num_frames=400)


class ProtocolTests(unittest.TestCase):
    def test_frame_round_trip(self):
        state = FrameState(
            t=1.25,
            contacts=(
                ContactState(key=7, x=-0.4, y=0.375, phase="hold", age=0.5, slot=2),
                ContactState(key=8, x=0.1, y=0.4, phase="auto", vx=0.2, vy=-0.1),
            ),
        )
        line = encode({"kind": "frame"} | state.as_dict())
        self.assertTrue(line.endswith("\n"))
        parsed = parse(line)
        back = FrameState.from_dict(parsed)
        self.assertAlmostEqual(back.t, 1.25, places=4)
        self.assertEqual(len(back.contacts), 2)
        self.assertEqual(back.contacts[0].phase, "hold")
        self.assertEqual(back.contacts[0].slot, 2)
        self.assertAlmostEqual(back.contacts[1].vx, 0.2, places=4)

    def test_parse_rejects_junk(self):
        with self.assertRaises(ValueError):
            parse("[1, 2, 3]")
        with self.assertRaises(ValueError):
            parse('{"no_kind": true}')

    def test_hello_carries_the_line(self):
        msg = hello("Glitch", fps=60.0, judgement_line_y=0.375)
        self.assertEqual(msg["kind"], "hello")
        self.assertEqual(msg["fps"], 60.0)
        self.assertAlmostEqual(msg["judgement_line_y"], 0.375)


class SceneTests(unittest.TestCase):
    def test_press_and_release(self):
        scene = Scene()
        scene.update(0.0, [ContactState(key=1, x=-0.3, y=0.375, phase="down", slot=0)])
        self.assertEqual(len(scene.contacts), 1)
        self.assertEqual(scene.contacts[0].slot, 0)
        scene.update(0.05, [])
        self.assertEqual(len(scene.contacts), 0)

    def test_scene_has_no_effect_state(self):
        """Effects were removed on purpose; guard against them creeping back."""
        scene = Scene()
        scene.update(0.0, [ContactState(key=1, x=0.0, y=0.375, phase="down")])
        self.assertFalse(hasattr(scene, "rings"))
        self.assertFalse(hasattr(scene, "particles"))

    def test_label_phase_is_trusted(self):
        scene = Scene()
        scene.update(0.0, [ContactState(key=1, x=0.0, y=0.375, phase="flick")])
        self.assertEqual(scene.contacts[0].phase, "flick")
        self.assertEqual(scene.phase_counts()["flick"], 1)

    def test_auto_phase_upgrades_to_hold(self):
        scene = Scene()
        state = [ContactState(key=1, x=0.0, y=0.375, phase="auto", slot=0)]
        scene.update(0.0, state)
        self.assertEqual(scene.contacts[0].phase, "down")
        scene.update(HOLD_AFTER + 0.02, state)
        self.assertEqual(scene.contacts[0].phase, "hold")

    def test_auto_phase_detects_flick_by_speed(self):
        scene = Scene()
        scene.update(0.0, [ContactState(key=1, x=0.0, y=0.375, phase="auto", vx=1.5)])
        self.assertEqual(scene.contacts[0].phase, "flick")

    def test_join_mid_flight_keeps_age(self):
        scene = Scene()
        scene.update(5.0, [ContactState(key=1, x=0.0, y=0.375, phase="auto", age=0.5)])
        # a renderer attached late must not treat a long hold as a fresh tap
        self.assertEqual(scene.contacts[0].phase, "hold")

    def test_colours_follow_the_finger(self):
        self.assertEqual(finger_color(0, 99), finger_color(0, 1))
        self.assertNotEqual(finger_color(0, 0), finger_color(1, 0))

    def test_clock_never_goes_backwards(self):
        scene = Scene()
        scene.tick(1.0)
        scene.tick(0.5)
        self.assertEqual(scene.now, 1.0)


class ViewportTests(unittest.TestCase):
    def test_judgment_line_sits_below_the_middle(self):
        vp = Viewport.for_grid(100, 30)
        _, v = vp.to_unit(0.0, JUDGMENT_LINE_Y)
        self.assertGreater(v, 0.5)
        self.assertLess(v, 0.8)

    def test_horizontal_edges(self):
        vp = Viewport.for_grid(100, 30)
        self.assertAlmostEqual(vp.to_unit(-1.0, 0.0)[0], 0.0)
        self.assertAlmostEqual(vp.to_unit(1.0, 0.0)[0], 1.0)

    def test_span_survives_a_tall_terminal(self):
        narrow = Viewport.for_grid(40, 60)
        self.assertGreater(narrow.y_max - narrow.y_min, 0.8)


class TerminalRenderTests(unittest.TestCase):
    def _scene_with_four(self) -> Scene:
        scene = Scene()
        scene.update(0.0, [
            ContactState(key=1, x=-0.6, y=0.40, phase="down", slot=0),
            ContactState(key=2, x=-0.2, y=0.375, phase="hold", slot=1),
            ContactState(key=3, x=0.2, y=0.375, phase="flick", slot=2, vx=1.0),
            ContactState(key=4, x=0.6, y=0.375, phase="down", slot=3),
        ])
        return scene

    def test_draws_one_glyph_per_finger(self):
        scene = self._scene_with_four()
        text = build_screen(
            scene, now=0.01, name="t", width=100, height=24,
            glyphs=GLYPHS_ASCII, use_color=False,
        )
        body = "\n".join(text.split("\n")[1:-1])   # skip HUD and legend text
        self.assertEqual(body.count(GLYPHS_ASCII["contact_down"]), 2)
        self.assertEqual(body.count(GLYPHS_ASCII["contact_hold"]), 1)
        self.assertEqual(body.count(GLYPHS_ASCII["contact_flick"]), 1)

    def test_has_a_judgment_line_and_hud(self):
        scene = self._scene_with_four()
        text = build_screen(
            scene, now=0.0, name="SUPERNOVA", width=90, height=20,
            glyphs=GLYPHS_UNICODE, use_color=False,
        )
        self.assertIn(GLYPHS_UNICODE["line"], text)
        self.assertIn("SUPERNOVA", text)
        self.assertIn("fingers=4", text)
        self.assertIn(f"hold {GLYPHS_UNICODE['contact_hold']} 1", text)

    def test_plain_mode_has_no_escapes(self):
        scene = self._scene_with_four()
        text = build_screen(scene, now=0.0, name="x", width=60, height=14,
                            glyphs=GLYPHS_ASCII, use_color=False)
        self.assertNotIn("\x1b", text)

    def test_every_line_is_the_requested_width(self):
        scene = self._scene_with_four()
        text = build_screen(scene, now=0.0, name="x", width=64, height=12,
                            glyphs=GLYPHS_ASCII, use_color=False)
        for line in text.split("\n"):
            self.assertLessEqual(len(line), 64)

    def test_projection_keeps_points_inside_the_grid(self):
        canvas = TerminalCanvas(40, 12, glyphs=GLYPHS_ASCII)
        for x in (-3.0, -1.0, 0.0, 1.0, 5.0):
            for y in (-5.0, 0.375, 4.0):
                col, row = canvas.project(x, y)
                self.assertTrue(0 <= col < canvas.w)
                self.assertTrue(0 <= row < canvas.h)

    def test_demo_exercises_all_three_phases(self):
        scene = Scene()
        seen: set[str] = set()
        max_fingers = 0
        for st in demo_frames(5.0):
            scene.update(st.t, st.contacts)
            max_fingers = max(max_fingers, len(st.contacts))
            for c in scene.contacts:
                seen.add(c.phase)
        self.assertEqual(seen, {"down", "hold", "flick"})
        self.assertGreaterEqual(max_fingers, 2)


class LiveModelTests(unittest.TestCase):
    def test_press_move_release(self):
        model = LiveModel({1: "hold"})
        model.press(1, 0, -0.4, 0.375, 10.0)
        snap = model.snapshot(10.5)
        self.assertEqual(len(snap), 1)
        self.assertEqual(snap[0].phase, "hold")
        self.assertAlmostEqual(snap[0].age, 0.5, places=3)

        model.move(1, -0.2, 0.375, 10.2)
        snap = model.snapshot(10.3)
        self.assertAlmostEqual(snap[0].x, -0.2, places=5)
        self.assertGreater(snap[0].vx, 0.0)

        model.release(1, 10.4)
        self.assertEqual(len(model.snapshot(10.4)), 0)

    def test_move_of_unknown_key_is_ignored(self):
        model = LiveModel()
        model.move(42, 0.0, 0.0, 1.0)
        self.assertEqual(model.snapshot(1.0), ())

    def test_phase_map_prefers_the_stronger_label(self):
        phases = phase_map(make_chart())
        self.assertEqual(phases[1], "down")
        self.assertEqual(phases[2], "hold")   # upgraded from its first "down"
        self.assertEqual(phases[3], "flick")
        self.assertNotIn(9, phases)           # out-of-range notes are dropped


class RegistryTests(unittest.TestCase):
    def test_renderers_are_discoverable(self):
        self.assertEqual(available(), sorted(RENDERERS))
        self.assertIn("terminal", available())
        self.assertIn("tk", available())

    def test_unknown_renderer_is_rejected(self):
        from beatflow_input.visualizer import RenderProcess

        with self.assertRaises(ValueError):
            RenderProcess("hologram")


class CliTests(unittest.TestCase):
    def test_watch_is_wired_up(self):
        from beatflow_input.cli import build_parser

        args = build_parser().parse_args(["watch", "--chart", "x", "--viz", "terminal"])
        self.assertEqual(args.command, "watch")
        self.assertEqual(args.viz, "terminal")
        self.assertAlmostEqual(args.speed, 1.0)

    def test_watch_rejects_unknown_viz(self):
        from beatflow_input.cli import build_parser

        with self.assertRaises(SystemExit):
            build_parser().parse_args(["watch", "--viz", "hologram"])


class SpriteAssetTests(unittest.TestCase):
    """The four pixel-art fingers and where their beat point sits."""

    def setUp(self):
        from beatflow_input.visualizer.tkview import ASSETS

        self.assets = ASSETS
        self.manifest = json.loads((ASSETS / "sprites.json").read_text(encoding="utf-8"))

    def test_four_fingers(self):
        self.assertEqual(len(self.manifest), 4)
        self.assertEqual([e["slot"] for e in self.manifest], [0, 1, 2, 3])

    def test_manifest_matches_the_pngs(self):
        for entry in self.manifest:
            data = (self.assets / entry["file"]).read_bytes()
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n", entry["file"])
            width, height = struct.unpack(">II", data[16:24])   # PNG IHDR
            self.assertEqual((width, height), (entry["width"], entry["height"]))
            hx, hy = entry["head"]
            self.assertTrue(0 <= hx < width and 0 <= hy < height, entry["file"])

    def test_origin_puts_the_beat_point_on_the_contact(self):
        from beatflow_input.visualizer.tkview import Sprite

        sprite = Sprite(image=None, width=15, height=15, head_x=14, head_y=4, zoom=5)
        cx, cy = 192.0, 359.0
        x, y = sprite.origin(cx, cy)
        # the head cell's right edge and top edge both land on the contact
        self.assertAlmostEqual(x + (14 + 1) * 5, cx)
        self.assertAlmostEqual(y + 4 * 5, cy)

    def test_origin_scales_with_zoom(self):
        from beatflow_input.visualizer.tkview import Sprite

        sprite = Sprite(image=None, width=16, height=16, head_x=15, head_y=0, zoom=1)
        x, y = sprite.origin(100.0, 200.0)
        self.assertAlmostEqual((x, y), (100.0 - 16, 200.0))

    def test_sprite_follows_the_finger_not_the_phase(self):
        from beatflow_input.visualizer.tkview import SpriteSet

        sprites = SpriteSet.__new__(SpriteSet)
        sprites.sprites = [None, None, None, None]
        # slot wins when known, key is the fallback (same rule as the colour)
        self.assertIsNone(sprites.for_contact(SimpleNamespace(slot=2, key=99)))
        self.assertEqual(len(sprites.sprites), 4)

    def test_empty_set_falls_back_to_dots(self):
        from beatflow_input.visualizer.tkview import SpriteSet

        sprites = SpriteSet.__new__(SpriteSet)
        sprites.sprites = []
        self.assertIsNone(sprites.for_contact(SimpleNamespace(slot=0, key=0)))
        self.assertFalse(sprites)


if __name__ == "__main__":
    unittest.main()
