"""Render a still of the visualizer - documentation asset only.

This is **not** part of the package: ``beatflow_input`` stays dependency-free,
while this script needs Pillow. It composites the real 16x16 finger sprites
with the same anchor maths as ``visualizer/tkview.py``, so the picture shows
exactly where the beat point lands.

    pip install pillow
    python docs/make_preview.py docs/visualizer-preview.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# the repo root *is* the package, so import from its parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image, ImageDraw                                       # noqa: E402

from beatflow_input.visualizer.protocol import ContactState            # noqa: E402
from beatflow_input.visualizer.scene import (                          # noqa: E402
    JUDGMENT_LINE_Y, PALETTE, Scene, Viewport,
)

ASSETS = Path(__file__).resolve().parent.parent / "visualizer" / "assets"
SPRITE_ZOOM = 5

W, H = 1100, 620
BG_TOP = (11, 15, 20)
BG_BOTTOM = (5, 7, 11)
LINE = (120, 220, 255)
GRID = (28, 38, 50)
TEXT = (150, 168, 190)


def bg_at(row: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, row / (H - 1)))
    return tuple(int(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t) for i in range(3))


def load_sprites() -> list[tuple[dict, Image.Image]]:
    manifest = ASSETS / "sprites.json"
    if not manifest.exists():
        return []
    out = []
    for entry in sorted(json.loads(manifest.read_text(encoding="utf-8")), key=lambda e: e["slot"]):
        art = Image.open(ASSETS / entry["file"]).convert("RGBA")
        out.append((entry, art.resize(
            (art.width * SPRITE_ZOOM, art.height * SPRITE_ZOOM), Image.NEAREST)))
    return out


def draw(img: Image.Image, scene: Scene, sprites: list, vp: Viewport) -> None:
    d = ImageDraw.Draw(img)
    for y in range(H):
        d.line([(0, y), (W, y)], fill=bg_at(y))

    def pt(x: float, y: float) -> tuple[float, float]:
        u, v = vp.to_unit(x, y)
        return u * (W - 1), v * (H - 1)

    line_y = pt(0.0, JUDGMENT_LINE_Y)[1]
    d.line([(0, line_y), (W, line_y)], fill=LINE, width=2)
    for lane in (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6):
        x = pt(lane, 0.0)[0]
        d.line([(x, line_y - 10), (x, line_y)], fill=GRID, width=2)

    for c in scene.contacts:
        cx, cy = pt(c.x, c.y)
        if not sprites:
            continue
        entry, art = sprites[(c.slot if c.slot >= 0 else c.key) % len(sprites)]
        hx, hy = entry["head"]
        # beat point = top-right corner of the head cell (see tkview.Sprite.origin)
        img.paste(art, (int(cx - (hx + 1) * SPRITE_ZOOM), int(cy - hy * SPRITE_ZOOM)), art)

    d.text((16, 10), "BeatFlow  virtual input  -  four fingers", fill=TEXT)
    labels = [("down", 0), ("hold", 1), ("flick", 2), ("down", 3)]
    for i, (phase, slot) in enumerate(labels):
        x = 16 + i * 265
        d.ellipse([x, H - 24, x + 9, H - 15], fill=PALETTE[slot])
        d.text((x + 14, H - 24), phase, fill=PALETTE[slot])
    for i, color in enumerate(PALETTE):
        x = W - 16 - i * 18
        d.ellipse([x - 5, 14, x + 5, 24], fill=color)


def main(out: str = "visualizer-preview.png") -> int:
    sprites = load_sprites()
    scene = Scene(judgement_line_y=JUDGMENT_LINE_Y)
    scene.update(0.0, [
        ContactState(key=1, x=-0.65, y=0.375, phase="down", slot=0),
        ContactState(key=2, x=-0.25, y=JUDGMENT_LINE_Y, phase="hold", slot=1, age=0.6),
        ContactState(key=3, x=0.25, y=JUDGMENT_LINE_Y, phase="flick", slot=2, vx=1.1, vy=0.45),
        ContactState(key=4, x=0.65, y=0.375, phase="down", slot=3),
    ])
    img = Image.new("RGB", (W, H), BG_TOP)
    draw(img, scene, sprites, Viewport.for_grid(W, H, char_aspect=1.0))
    img.save(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "visualizer-preview.png"))
