#!/usr/bin/env python3
"""Generate binary sliding-window masks, a manifest, and a contact sheet."""

import argparse
from fractions import Fraction
import json
from math import ceil, isqrt
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]


def decimal_string(value: Fraction) -> str:
    """Render an exact terminating decimal, without rounded coordinate aliases."""
    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        raise ValueError("Normalized coordinates must have exact terminating decimals")
    places = max(twos, fives)
    scaled = value.numerator * (10**places // value.denominator)
    if not places:
        return str(scaled)
    digits = str(scaled).zfill(places + 1)
    return (digits[:-places] + "." + digits[-places:]).rstrip("0").rstrip(".")


def mask_layout(height: int, width: int, scale: int, overlap: Fraction):
    if min(height, width, scale) <= 0:
        raise ValueError("Height, width, and scale must be positive integers")
    if not 0 <= overlap < 1:
        raise ValueError("Overlap must satisfy 0 <= overlap < 1")
    if height % scale or width % scale:
        raise ValueError("Height and width must be divisible by scale")
    window_h, window_w = height // scale, width // scale
    steps = [Fraction(size) * (1 - overlap) for size in (window_h, window_w)]
    if any(step.denominator != 1 or step < 1 for step in steps):
        raise ValueError("Overlap must produce positive integer pixel strides")
    stride_y, stride_x = map(int, steps)
    if (height - window_h) % stride_y or (width - window_w) % stride_x:
        raise ValueError("Stride must land exactly on the last window for complete edge coverage")
    records = []
    for y in range(0, height - window_h + 1, stride_y):
        for x in range(0, width - window_w + 1, stride_x):
            ny, nx = decimal_string(Fraction(y, height)), decimal_string(Fraction(x, width))
            records.append({
                "filename": f"{ny}_{nx}.png",
                "top_left_normalized": {"y": ny, "x": nx},
                "bounds_pixels": {"y0": y, "x0": x, "y1": y + window_h, "x1": x + window_w},
            })
    return (window_h, window_w), (stride_y, stride_x), records


def generate(height, width, scale, overlap, output_root):
    window, stride, records = mask_layout(height, width, scale, overlap)
    tag = f"s{scale}-v{decimal_string(overlap)}"
    directory = output_root / tag
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError(f"Output already exists and is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)

    # A separate preview subdirectory keeps *.png at the root limited to masks.
    columns = isqrt(len(records) - 1) + 1
    tile_w, tile_h = 100, 100
    sheet = Image.new("RGB", (columns * tile_w, ceil(len(records) / columns) * tile_h), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    for i, record in enumerate(records):
        bounds = record["bounds_pixels"]
        mask = Image.new("L", (width, height), 0)
        mask.paste(255, (bounds["x0"], bounds["y0"], bounds["x1"], bounds["y1"]))
        mask.save(directory / record["filename"])
        preview = mask.copy()
        preview.thumbnail((80, 80), Image.Resampling.NEAREST)
        left, top = (i % columns) * tile_w, (i // columns) * tile_h
        sheet.paste(preview, (left + 10, top + 2))
        draw.text((left + 3, top + 84), f"{bounds['y0']},{bounds['x0']}", fill="black")
    (directory / "preview").mkdir()
    sheet.save(directory / "preview" / "contact_sheet.png")
    manifest = {
        "schema_version": 1,
        "height": height, "width": width, "scale": scale,
        "overlap": float(overlap), "overlap_exact": decimal_string(overlap),
        "window_pixels": {"height": window[0], "width": window[1]},
        "stride_pixels": {"y": stride[0], "x": stride[1]},
        "mask_count": len(records), "valid_value": 255, "masked_value": 0,
        "bounds_convention": "half-open: [y0, y1), [x0, x1)",
        "coordinate_convention": "top-left y/height and x/width; exact decimal strings",
        "contact_sheet": "preview/contact_sheet.png",
        "contact_sheet_labels": "top-left y,x in pixels; row-major order",
        "masks": records,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(records)} masks in {directory}")
    print(f"Window: {window}; pixel stride: {stride}")
    print(f"Contact sheet: {directory / 'preview/contact_sheet.png'}")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--overlap", default="0.5", help="Fraction of window overlapped, in [0,1)")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "mask")
    args = parser.parse_args()
    try:
        generate(args.height, args.width, args.scale, Fraction(args.overlap), args.output_root)
    except (ValueError, ZeroDivisionError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
