"""
Missed violations on the 20 non-compliant DesignQA dimension questions by relative margin and evidence type.
Manual annotation: values read from the detailed_context drawings, limits from the rules text. Evidence:
direct (one labelled dimension), derived (computed from labels), crossref (limit in a referenced clause),
geometric (spatial relation or count, no margin). Relative margin = |value - limit| / limit.

Usage:
    uv run python scripts/analyze_violation_strata.py [--no-write]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_three_way_decision import CONFIGS, load_votes

OUT_DIR = REPO / "results_for_publication" / "rule_compliance" / "violation_strata"

ANNOTATION: dict[str, tuple[str, float | None, float | None, str]] = {
    "V.1.2b.jpg": ("direct", 1524.9, 1525.0, "wheelbase >= 1525"),
    "V.4.1b.jpg": ("derived", 201.2, 203.2, "diameter = 2 x R100.6 >= 203.2"),
    "V.1.4.2b.jpg": ("derived", 102.9, 75.0, "clearance = R203.2 - 100.3 <= 75"),
    "F.5.7.5b.jpg": ("direct", 261.57, 250.0, "hoop to steering wheel <= 250"),
    "F.5.7.6b.jpg": ("direct", 21.5, 20.0, "front hoop inclination < 20 deg"),
    "F.5.9.4b.jpg": ("derived", 173.76, 160.0, "161.06 + 12.7 <= 160"),
    "F.5.9.5b.jpg": ("direct", 28.76, 30.0, "included angle >= 30 deg"),
    "F.6.3.4b.jpg": ("derived", 55.04, 50.0, "26.59 + R28.45 <= 50"),
    "F.3.2.1g (which references F.3.4.1)b.jpg": ("crossref", 24.3, 25.0, "size A OD >= 25"),
    "F.6.4.4bb.jpg": ("direct", 231.56, 240.0, "upper side impact >= 240"),
    "F.3.2.1h (which references F.3.4.1)b.jpg": ("crossref", 23.43, 25.0, "size B OD >= 25"),
    "T.5.2.7ab.jpg": ("direct", 2.56, 2.66, "shield thickness >= 2.66"),
    "T.5.2.7cb.jpg": ("derived", 22.15, 22.65, "centred: 45.3 / 2 vs 22.15"),
    "T.5.2.7bb.jpg": ("derived", 38.0, 39.0, "width >= 3 x 13"),
    "T.8.2.4b.jpg": ("geometric", None, None, "count of projecting threads"),
    "T.7.5.ab.jpg": ("direct", 705.28, 700.0, "front aero <= 700 forward"),
    "T.7.5.bb.jpg": ("direct", 261.11, 250.0, "rear aero <= 250 rearward"),
    "T.7.6.1b.jpg": ("geometric", None, None, "front wing inside tyres"),
    "T.7.6.3 (which refers to T.7.3.3)b.jpg": ("geometric", None, None, "rear wing in aero zone"),
    "T.7.7.1ab.jpg": ("direct", 1201.31, 1200.0, "rear aero height <= 1200 (benchmark explanation says 1404.51; drawing shows 1201.31)"),
}
MARGIN_BINS = [0.0, 0.02, 0.05, 1.0]
MARGIN_LABELS = ["<2%", "2-5%", ">5%"]


def item_table() -> pd.DataFrame:
    rows = []
    for label, run_dir in CONFIGS["dimension"].items():
        votes, _ = load_votes(run_dir)
        for key, rec in votes.items():
            if rec["gt"] != "no":
                continue
            image = key.split(" || ")[0]
            if image not in ANNOTATION:
                sys.exit(f"missing annotation for {image}")
            ev, value, limit, note = ANNOTATION[image]
            margin = abs(value - limit) / limit if value is not None else None
            rows.append({
                "image": image, "config": label, "evidence": ev, "note": note,
                "rel_margin": margin,
                "misses_of_5": sum(p == "yes" for p in rec["preds"]),
            })
    df = pd.DataFrame(rows)
    df["margin_bin"] = pd.cut(df["rel_margin"], MARGIN_BINS, labels=MARGIN_LABELS)
    df["missed_majority"] = df["misses_of_5"] >= 3
    return df


def summarise(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby([by, "config"], observed=True)
    out = g.agg(items=("image", "count"),
                missed_majority=("missed_majority", "sum"),
                mean_miss_rate=("misses_of_5", lambda s: s.mean() / 5))
    return out.unstack("config")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--no-write", action="store_true", help="stdout only")
    args = ap.parse_args()

    df = item_table()
    if len(df) != 20 * len(CONFIGS["dimension"]):
        sys.exit(f"expected 20 violations per config, got {len(df)}")

    wide = df.pivot(index=["image", "evidence", "rel_margin"], columns="config",
                    values="misses_of_5").sort_index(level="evidence")
    by_ev, by_mag = summarise(df, "evidence"), summarise(df, "margin_bin")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("misses out of 5 runs per violation\n", wide.to_string(float_format=lambda x: f"{x:.3f}"))
        print("\nby evidence type\n", by_ev.to_string(float_format=lambda x: f"{x:.2f}"))
        print("\nby relative margin (numeric items)\n", by_mag.to_string(float_format=lambda x: f"{x:.2f}"))

    if not args.no_write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_DIR / "violation_items.csv", index=False)
        by_ev.to_csv(OUT_DIR / "by_evidence_type.csv")
        by_mag.to_csv(OUT_DIR / "by_margin.csv")
        (OUT_DIR / "annotation.json").write_text(
            json.dumps({k: dict(zip(("evidence", "value", "limit", "note"), v))
                        for k, v in ANNOTATION.items()}, indent=2),
            encoding="utf-8",
        )
        print(f"\nwritten to {OUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
