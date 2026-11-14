"""Cut the four hand-drawn finger sprites down to their native 16x16 art.

The source images are 1024x1024 upscales of a 16x16 pixel drawing (64 px per
art pixel). Three ship with a real alpha channel; ``finger3.png`` does not and
carries a flat black background instead, so it is flood-filled from the border
before sampling - interior dark pixels of the sprite must survive.

Output: ``visualizer/assets/finger0..3.png``, trimmed to the opaque bounding
box, RGBA, 16x16 or smaller. They are then drawn with the *top-right corner of
the head* on the contact point (see ``tkview.Sprite.origin``), which is what
"beat point at the top-right of the pattern" means in practice.

Needs Pillow. The package itself stays stdlib-only; this is an asset build step.

    python docs/make_sprites.py <src-dir-or-file...>
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "visualizer" / "assets"
ART = 16          # native resolution of the drawings
BG_TOLERANCE = 12  # max channel value still considered "background black"


def cut_background(rgba: Image.Image) -> Image.Image:
    """Flood-fill near-black from the border, leaving interior darks alone."""
    w, h = rgba.size
    px = rgba.load()
    seen = [[False] * w for _ in range(h)]
    stack = []
    for x in range(w):
        stack += [(x, 0), (x, h - 1)]
    for y in range(h):
        stack += [(0, y), (w - 1, y)]

    while stack:
        x, y = stack.pop()
        if x < 0 or y < 0 or x >= w or y >= h or seen[y][x]:
            continue
        seen[y][x] = True
        r, g, b, a = px[x, y]
        if a == 0:
            continue
        if max(r, g, b) > BG_TOLERANCE:
            continue
        px[x, y] = (r, g, b, 0)
        stack += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    return rgba


def has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA") or "transparency" in img.info or img.mode == "PA"


def sample_art(rgba: Image.Image) -> Image.Image:
    """Pick the colour at each art-pixel centre (no averaging: keeps it crisp)."""
    cell = rgba.width // ART
    out = Image.new("RGBA", (ART, ART), (0, 0, 0, 0))
    px_in = rgba.load()
    for y in range(ART):
        for x in range(ART):
            sx = min(rgba.width - 1, x * cell + cell // 2)
            sy = min(rgba.height - 1, y * cell + cell // 2)
            out.putpixel((x, y), px_in[sx, sy])
    return out


def trim(rgba: Image.Image) -> Image.Image:
    mask = rgba.getchannel("A").point(lambda v: 255 if v > 8 else 0)
    box = mask.getbbox()
    return rgba.crop(box) if box else rgba


def dominant(rgba: Image.Image) -> tuple[int, int, int]:
    counts: Counter = Counter()
    for y in range(rgba.height):
        for x in range(rgba.width):
            r, g, b, a = rgba.getpixel((x, y))
            if a > 128 and max(r, g, b) > 40:
                counts[(r, g, b)] += 1
    return counts.most_common(1)[0][0] if counts else (255, 255, 255)


def accent(rgba: Image.Image, target_luma: float = 165.0) -> tuple[int, int, int]:
    """The sprite's most *chromatic* colour, lightened to a readable brightness.

    Picking the brightest pixel would return a white highlight; picking the most
    common one returns the dark body. The most saturated pixel is what actually
    identifies the finger. Brightening is done by mixing toward white rather
    than scaling the channels: scaling turns a dark purple into electric violet,
    mixing keeps the hue and lands on a soft lavender.
    """
    best, best_chroma = None, -1
    for y in range(rgba.height):
        for x in range(rgba.width):
            r, g, b, a = rgba.getpixel((x, y))
            if a <= 128:
                continue
            chroma = max(r, g, b) - min(r, g, b)
            if chroma > best_chroma:
                best, best_chroma = (r, g, b), chroma
    if best is None:
        return (200, 200, 200)
    luma = 0.299 * best[0] + 0.587 * best[1] + 0.114 * best[2]
    if luma >= target_luma:
        return best
    t = (target_luma - luma) / max(1.0, 255.0 - luma)
    return tuple(int(c + (255 - c) * t) for c in best)


def head_cell(rgba: Image.Image) -> tuple[int, int]:
    """Top-right-most opaque cell: (x, y) with the largest x, then smallest y."""
    best = None
    for y in range(rgba.height):
        for x in range(rgba.width):
            if rgba.getpixel((x, y))[3] > 8:
                if best is None or x > best[0] or (x == best[0] and y < best[1]):
                    best = (x, y)
    return best or (rgba.width - 1, 0)


def main(srcs: list[str]) -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, src in enumerate(srcs):
        img = Image.open(src)
        rgba = img.convert("RGBA")
        if not has_alpha(img):
            print(f"  finger{i}: no alpha channel (mode {img.mode}) -> flood fill")
            rgba = cut_background(rgba)
        art = trim(sample_art(rgba))
        out = ASSETS / f"finger{i}.png"
        art.save(out)
        colour = dominant(art)
        acc = accent(art)
        hx, hy = head_cell(art)
        manifest.append({
            "slot": i,
            "file": out.name,
            "width": art.width,
            "height": art.height,
            "head": [hx, hy],
            "accent": list(acc),
        })
        print(f"  {out.name}: {art.width}x{art.height}  head=({hx},{hy})  "
              f"dominant=#{colour[0]:02x}{colour[1]:02x}{colour[2]:02x}  "
              f"accent=#{acc[0]:02x}{acc[1]:02x}{acc[2]:02x}")

    path = ASSETS / "sprites.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: make_sprites.py <source image> [more...]")
    raise SystemExit(main(sys.argv[1:]))
