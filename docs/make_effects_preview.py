"""Render a still of the three effects - documentation asset only.

This is **not** part of the package: ``beatflow_input`` stays dependency-free,
while this script needs Pillow. It mirrors ``visualizer/tkview.py``'s drawing
math so the picture is representative of what the Tk window paints.

    pip install pillow
    python docs/make_effects_preview.py docs/visualizer-effects.png
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# the repo root *is* the package, so import from its parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image, ImageDraw                                       # noqa: E402

from beatflow_input.visualizer.protocol import ContactState            # noqa: E402
from beatflow_input.visualizer.scene import (                          # noqa: E402
    JUDGMENT_LINE_Y, PALETTE, Scene, Viewport,
)

W, H = 1100, 620
BG_TOP = (11, 15, 20)
BG_BOTTOM = (5, 7, 11)
LINE = (120, 220, 255)
GRID = (28, 38, 50)
TEXT = (150, 168, 190)
NOW = 0.075


def blend(fg, bg, a):
    a = max(0.0, min(1.0, a))
    return tuple(int(bg[i] + (fg[i] - bg[i]) * a) for i in range(3))


def bg_at(row):
    t = max(0.0, min(1.0, row / (H - 1)))
    return tuple(int(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t) for i in range(3))


def draw(img, scene, vp):
    d = ImageDraw.Draw(img)
    for y in range(H):
        d.line([(0, y), (W, y)], fill=bg_at(y))

    def pt(x, y):
        u, v = vp.to_unit(x, y)
        return u * (W - 1), v * (H - 1)

    def px(r):
        return r * (W - 1) / 2.0

    ly = pt(0, JUDGMENT_LINE_Y)[1]
    d.line([(0, ly), (W, ly)], fill=LINE, width=2)
    for lane in (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6):
        x = pt(lane, 0)[0]
        d.line([(x, ly - 10), (x, ly)], fill=GRID, width=2)

    for ring in scene.rings:
        cx, cy = pt(ring.x, ring.y)
        r = px(ring.radius(NOW))
        if r < 0.5:
            continue
        if ring.filled:
            col = blend(ring.color, bg_at(cy), 0.15 + 0.85 * ring.alpha(NOW) ** 1.6)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
            continue
        col = blend(ring.color, bg_at(cy), 0.30 + 0.70 * ring.alpha(NOW) ** 0.6)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col,
                  width=max(1, int(ring.width * 1.6)))

    for p in scene.particles:
        x, y = p.position(NOW)
        cx, cy = pt(x, y)
        r = max(0.9, 1.2 * p.size * (1.0 - 0.5 * p.progress(NOW)))
        col = blend(p.color, bg_at(cy), 0.18 + 0.82 * (1.0 - p.progress(NOW)))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)

    for c in scene.contacts:
        cx, cy = pt(c.x, c.y)
        bg, base = bg_at(cy), c.effect_color
        if c.phase == "hold":
            r = px(0.052) + 2.0 * math.sin(NOW * math.tau * 2.2)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=blend(base, bg, 0.85), width=2)
            ar = px(0.085)
            start = (NOW * 220.0) % 360.0
            d.arc([cx - ar, cy - ar, cx + ar, cy + ar], start, start + 70,
                  fill=blend(base, bg, 0.95), width=3)
            d.line([(cx, cy), (cx, cy + px(0.05))], fill=blend(base, bg, 0.55), width=1)
        if c.phase == "flick":
            dx, dy = c.direction()
            sx, sy = dx, -dy
            tail = px(0.16)
            for i in range(1, 7):
                t = i / 6.0
                qx, qy = cx - sx * tail * t, cy - sy * tail * t
                rr = max(1.0, 4.2 * (1.0 - t))
                d.ellipse([qx - rr, qy - rr, qx + rr, qy + rr],
                          fill=blend(base, bg, 0.75 * (1.0 - t) + 0.1))
            head = px(0.055)
            d.polygon([
                (cx + sx * head, cy + sy * head),
                (cx - sx * head * 0.6 - sy * head * 0.8, cy - sy * head * 0.6 + sx * head * 0.8),
                (cx - sx * head * 0.6 + sy * head * 0.8, cy - sy * head * 0.6 - sx * head * 0.8),
            ], fill=blend(base, bg, 0.95))
            continue
        r = px(0.020)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=blend(base, bg, 1.0))
        core = max(1.0, r * 0.45)
        d.ellipse([cx - core, cy - core, cx + core, cy + core],
                  fill=blend((255, 255, 255), bg, 0.9))

    d.text((16, 10), "BeatFlow  virtual input  -  four fingers, three label phases",
           fill=TEXT)
    labels = [("down", 0, "tap: impact ring + sparks"),
              ("hold", 1, "hold: pulsing halo, arc, embers"),
              ("flick", 2, "flick: directed comet"),
              ("down", 3, "colour = finger")]
    for i, (phase, slot, note) in enumerate(labels):
        x = 16 + i * 265
        d.ellipse([x, H - 24, x + 9, H - 15], fill=PALETTE[slot])
        d.text((x + 14, H - 26), f"{phase}", fill=PALETTE[slot])
        d.text((x + 14, H - 38), note, fill=(90, 102, 122))
    for i, color in enumerate(PALETTE):
        x = W - 16 - i * 18
        d.ellipse([x - 5, 14, x + 5, 24], fill=color)


def main(out: str = "visualizer-effects.png") -> int:
    scene = Scene(judgement_line_y=JUDGMENT_LINE_Y)
    scene.update(0.0, [
        ContactState(key=1, x=-0.65, y=0.375, phase="down", slot=0),
        ContactState(key=2, x=-0.25, y=JUDGMENT_LINE_Y, phase="hold", slot=1, age=0.6),
        ContactState(key=3, x=0.25, y=JUDGMENT_LINE_Y, phase="flick", slot=2,
                     vx=1.1, vy=0.45),
        ContactState(key=4, x=0.65, y=0.375, phase="down", slot=3),
    ])
    scene.tick(NOW)
    vp = Viewport.for_grid(W, H, char_aspect=1.0)
    img = Image.new("RGB", (W, H), BG_TOP)
    draw(img, scene, vp)
    img.save(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "visualizer-effects.png"))
