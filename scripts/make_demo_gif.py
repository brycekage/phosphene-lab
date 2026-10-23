"""Phase 5: assemble the README GIF from browser screenshots of the running demo app.

The frames are screenshots of ``app/app.py`` taken in a browser (numbered 01.png, 02.png, ...
in the order shown). Each segment of the config crops its frames, holds each for a number of
seconds, and everything is padded to one size and scaled to a fixed width.

Usage:
    python scripts/make_demo_gif.py [--config configs/demo.yaml] [--quick]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from phosphene_lab.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--quick", action="store_true", help="first segment only")
    args = parser.parse_args()
    cfg = load_config(args.config, args.quick)["gif"]
    segments = cfg["segments"][:1] if args.quick else cfg["segments"]
    output = Path("results/quick/phase5_demo.gif" if args.quick else cfg["output"])

    frames, durations = [], []
    for first, last, bottom, seconds in segments:
        for k in range(first, last + 1):
            img = Image.open(Path(cfg["frames_dir"]) / f"{k:02d}.png").convert("RGB")
            frames.append(img.crop((0, 0, img.width - 30, bottom)))  # drop the scrollbar
            durations.append(round(seconds * 1000))

    width, height = max(f.width for f in frames), max(f.height for f in frames)
    background = frames[0].getpixel((5, 5))
    scale = cfg["width"] / width
    size = (cfg["width"], round(height * scale))
    out = []
    for f in frames:
        canvas = Image.new("RGB", (width, height), background)
        canvas.paste(f, (0, 0))
        out.append(canvas.resize(size, Image.LANCZOS).quantize(colors=128, dither=0))

    output.parent.mkdir(parents=True, exist_ok=True)
    out[0].save(
        output, save_all=True, append_images=out[1:], duration=durations, loop=0, optimize=True
    )
    total = sum(durations) / 1000
    print(f"wrote {output}: {len(out)} frames, {total:.0f} s, {output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
