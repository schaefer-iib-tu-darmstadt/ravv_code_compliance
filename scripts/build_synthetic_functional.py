"""
Build synthetic functional-performance questions by redrawing the one printed value that decides compliance.
Five DesignQA images, values at +1/+5/+15 % beyond the limit (violations) and 5 % inside (control).
Writes dataset_synthetic/rule_functional_synthetic/; images are regenerated here, not distributed.

Usage:
    uv run python scripts/build_synthetic_functional.py            # build all
    uv run python scripts/build_synthetic_functional.py --preview  # label crops only
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "dataset" / "rule_compliance" / "rule_functional_performance_qa"
SRC_CSV = SRC_DIR / "rule_functional_performance_qa.csv"
OUT_DIR = REPO / "dataset_synthetic" / "rule_functional_synthetic"
FONTS = {"regular": Path("C:/Windows/Fonts/arial.ttf"), "bold": Path("C:/Windows/Fonts/arialbd.ttf")}

VIOLATION_MARGINS = (0.01, 0.05, 0.15)
COMPLIANT_MARGIN = 0.05


@dataclass(frozen=True)
class Item:
    image: str
    window: tuple[int, int, int, int]
    limit: float
    kind: str
    fmt: str
    align: str
    weight: str = "regular"
    evidence: str = "direct"
    note: str = ""


ITEMS: list[Item] = [
    Item("2.png", (630, 715, 195, 380), 240.0, "max", "{:.2f}", "left",
         note="battery structure FEA max, 6061-T6 yield (F.3.5.3b)"),
    Item("9.png", (620, 705, 260, 440), 180.0, "max", "{:.2f}", "left",
         note="welded steel FEA max (F.3.4.2b)"),
    Item("16.png", (25, 95, 108, 250), 240.0, "max", "{:.2f}", "left",
         note="motor mount FEA max, 6061-T6 yield (F.3.5.3b)"),
    Item("12.png", (218, 258, 3120, 3235), 2000.0, "min", "{:.0f}", "right",
         note="brake pedal input force (T.3.2.2)"),
    Item("4.png", (2016, 2050, 1575, 1660), 102.0, "min", "{:.1f}", "right", "bold",
         evidence="derived", note="largest accommodated driver vs. 95th-percentile weight (V.2.1.1)"),
]


def quantity(item: Item, margin: float, violating: bool) -> float:
    outward = 1 if item.kind == "max" else -1
    sign = outward if violating else -outward
    return item.limit * (1 + sign * margin)


def locate_value(img: Image.Image, window: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    wy0, wy1, wx0, wx1 = window
    g = np.asarray(img.convert("L"), dtype=np.int16)[wy0:wy1, wx0:wx1]
    ys, xs = np.nonzero(g < 128)
    if ys.size == 0:
        raise ValueError("no text pixels in window")
    return wy0 + ys.min(), wy0 + ys.max(), wx0 + xs.min(), wx0 + xs.max()


def redraw(img: Image.Image, box: tuple[int, int, int, int], text: str, item: Item) -> Image.Image:
    y0, y1, x0, x1 = box
    font = ImageFont.truetype(str(FONTS[item.weight]), int(round((y1 - y0) / 0.716)))
    width = font.getlength(text)
    out = img.copy()
    d = ImageDraw.Draw(out)
    pad = 4
    wy0, wy1, wx0, wx1 = item.window
    d.rectangle([max(x0 - pad, wx0), max(y0 - pad, wy0), min(x1 + pad, wx1), min(y1 + pad, wy1)], fill="white")
    tx = x0 if item.align == "left" else x1 - width
    d.text((tx, y1), text, font=font, fill="black", anchor="ls")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--preview", action="store_true", help="write label crops only")
    args = ap.parse_args()

    src = pd.read_csv(SRC_CSV)
    img_dir = OUT_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in ITEMS:
        base = Image.open(SRC_DIR / "images" / item.image).convert("RGB")
        box = locate_value(base, item.window)
        if args.preview:
            y0, y1, x0, x1 = box
            base.crop((x0 - 200, y0 - 80, x1 + 200, y1 + 80)).save(img_dir / f"preview_{item.image}")
            continue
        q_row = src[src.image == item.image].iloc[0]
        variants = [(m, True) for m in VIOLATION_MARGINS] + [(COMPLIANT_MARGIN, False)]
        for margin, violating in variants:
            text = item.fmt.format(quantity(item, margin, violating))
            tag = f"{'v' if violating else 'c'}{int(round(margin * 100)):02d}"
            name = f"{Path(item.image).stem}_{tag}.png"
            redraw(base, box, text, item).save(img_dir / name)
            rows.append({
                "question": q_row.question,
                "ground_truth": "no" if violating else "yes",
                "image": name,
                "explanation": f"synthetic: {item.note}; drawn value {text}, limit {item.limit} ({item.kind})",
                "source_image": item.image,
                "evidence": item.evidence,
                "rel_margin": margin,
                "drawn_label": text,
            })
            print(f"{name:20s} {text:>10s}  gt={'no' if violating else 'yes'}")
    if rows:
        pd.DataFrame(rows).to_csv(OUT_DIR / "rule_functional_synthetic.csv", index=False)
        print(f"\n{len(rows)} questions -> {OUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
