"""
Build synthetic dimension-compliance questions by redrawing the deciding label in DesignQA drawings.
The a/b drawing difference locates the label; values at +1/+5/+15 % beyond the limit and 5 % inside.
Writes dataset_synthetic/rule_dimension_synthetic/; images are regenerated here, not distributed.

Usage:
    uv run python scripts/build_synthetic_violations.py            # build all
    uv run python scripts/build_synthetic_violations.py --preview  # label crops only
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None

REPO = Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "dataset" / "rule_compliance" / "rule_dimension_qa" / "detailed_context"
SRC_CSV = SRC_DIR / "rule_dimension_qa_detailed_context.csv"
OUT_DIR = REPO / "dataset_synthetic" / "rule_dimension_synthetic"
FONT = Path("C:/Windows/Fonts/arial.ttf")

VIOLATION_MARGINS = (0.01, 0.05, 0.15)
COMPLIANT_MARGIN = 0.05


@dataclass(frozen=True)
class Item:
    image: str
    evidence: str
    limit: float
    kind: str
    fmt: str
    to_label: object = None
    note: str = ""


ITEMS: list[Item] = [
    Item("V.1.2b.jpg", "direct", 1525.0, "min", "{:.1f}", note="wheelbase"),
    Item("V.4.1b.jpg", "derived", 203.2, "min", "R{:.1f}", lambda d: d / 2, "diameter = 2R"),
    Item("V.1.4.2b.jpg", "derived", 75.0, "max", "{:.1f}", lambda c: 203.2 - c, "clearance = R - label"),
    Item("F.5.7.5b.jpg", "direct", 250.0, "max", "{:.2f}", note="hoop to wheel"),
    Item("F.5.7.6b.jpg", "direct", 20.0, "max", "{:.1f}°", note="hoop inclination"),
    Item("F.5.9.4b.jpg", "derived", 160.0, "max", "{:.2f}", note="brace offset (+12.7 under derived reading)"),
    Item("F.6.4.4bb.jpg", "direct", 240.0, "min", "{:.2f}", note="side impact height"),
    Item("F.3.2.1g (which references F.3.4.1)b.jpg", "crossref", 25.0, "min", "{:.2f}", note="size A OD"),
    Item("F.3.2.1h (which references F.3.4.1)b.jpg", "crossref", 25.0, "min", "{:.2f}", note="size B OD"),
    Item("T.5.2.7ab.jpg", "direct", 2.66, "min", "{:.2f}", note="shield thickness"),
    Item("T.5.2.7bb.jpg", "derived", 39.0, "min", "{:.2f}", note="width >= 3 x 13"),
    Item("T.7.5.ab.jpg", "direct", 700.0, "max", "{:.2f}", note="front aero"),
    Item("T.7.5.bb.jpg", "direct", 250.0, "max", "{:.2f}", note="rear aero"),
    Item("T.7.7.1ab.jpg", "direct", 1200.0, "max", "{:.2f}", note="rear aero height"),
]


def quantity(item: Item, margin: float, violating: bool) -> float:
    outward = 1 if item.kind == "max" else -1
    sign = outward if violating else -outward
    return item.limit * (1 + sign * margin)


def label_value(item: Item, q: float, violating: bool) -> float:
    if item.to_label is not None:
        return item.to_label(q)
    if item.image == "F.5.9.4b.jpg" and not violating:
        return q - 12.7
    return q


def locate_label(b_path: Path) -> tuple[int, int, int, int]:
    a_path = b_path.with_name(b_path.name[:-5] + "a.jpg")
    A = np.asarray(Image.open(a_path).convert("L"), dtype=np.int16)
    B = np.asarray(Image.open(b_path).convert("L"), dtype=np.int16)
    diff = ndimage.binary_dilation(np.abs(A - B) > 60, iterations=15)
    ys, xs = np.nonzero(diff)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    if y1 - y0 > 400:
        sys.exit(f"{b_path.name}: a/b differ beyond a single label")

    dark = B[y0:y1 + 1] < 128
    text_col = dark.sum(axis=0) >= 16
    gap = 70
    lo, hi = x0, x1
    while True:
        left = text_col[max(lo - gap, 0):lo]
        if left.any():
            lo = lo - gap + int(np.argmax(left))
            continue
        break
    while True:
        right = text_col[hi + 1:hi + 1 + gap]
        if right.any():
            hi = hi + 1 + int(len(right) - 1 - np.argmax(right[::-1]))
            continue
        break
    cols = np.nonzero(text_col[lo:hi + 1])[0]
    lo, hi = lo + cols.min(), lo + cols.max()
    rows = np.nonzero((B[y0:y1 + 1, lo:hi + 1] < 128).any(axis=1))[0]
    return y0 + rows.min(), y0 + rows.max(), lo, hi


def redraw(img: Image.Image, box: tuple[int, int, int, int], text: str) -> Image.Image:
    y0, y1, x0, x1 = box
    digit_h = y1 - y0
    font = ImageFont.truetype(str(FONT), int(round(digit_h / 0.716)))
    width = font.getlength(text)
    cx = (x0 + x1) / 2
    pad = 12
    out = img.copy()
    d = ImageDraw.Draw(out)
    half = max(width / 2, (x1 - x0) / 2) + pad
    d.rectangle([cx - half, y0 - pad, cx + half, y1 + pad], fill="white")
    d.text((cx - width / 2, y1), text, font=font, fill="black", anchor="ls")
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
        b_path = SRC_DIR / item.image
        box = locate_label(b_path)
        base = Image.open(b_path).convert("RGB")
        if args.preview:
            y0, y1, x0, x1 = box
            base.crop((x0 - 300, y0 - 150, x1 + 300, y1 + 150)).save(img_dir / f"preview_{item.image}.png")
            continue
        q_row = src[src.image == item.image].iloc[0]
        variants = [(m, True) for m in VIOLATION_MARGINS] + [(COMPLIANT_MARGIN, False)]
        for margin, violating in variants:
            q = quantity(item, margin, violating)
            text = item.fmt.format(label_value(item, q, violating))
            tag = f"{'v' if violating else 'c'}{int(round(margin * 100)):02d}"
            name = f"{Path(item.image).stem}_{tag}.jpg"
            redraw(base, box, text).save(img_dir / name, quality=92)
            rows.append({
                "question": q_row.question,
                "ground_truth": "no" if violating else "yes",
                "image": name,
                "dimension_type": "direct",
                "explanation": f"synthetic: {item.note}; drawn label {text}, limit {item.limit} ({item.kind})",
                "source_image": item.image,
                "evidence": item.evidence,
                "rel_margin": margin,
                "drawn_label": text,
            })
            print(f"{name:55s} {text:>10s}  gt={'no' if violating else 'yes'}")
    if rows:
        pd.DataFrame(rows).to_csv(OUT_DIR / "rule_dimension_synthetic.csv", index=False)
        print(f"\n{len(rows)} questions -> {OUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
