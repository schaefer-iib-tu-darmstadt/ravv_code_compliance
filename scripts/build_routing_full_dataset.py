"""Merge the six DesignQA benchmark CSVs (n=1391) into data/processed/routing_questions_full.csv.

Columns match the 60-question routing CSV (task_type, question, image), so
src/orchestration/orchestration_tests.py reads it unchanged.

Usage:
    uv run python scripts/build_routing_full_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation.benchmark_evaluation import BENCHMARKS

TASKS = ["retrieval", "compilation", "definition", "presence", "dimension", "functional"]

OUT_CSV = REPO / "data" / "processed" / "routing_questions_full.csv"


def main() -> None:
    frames = []
    for task in TASKS:
        spec = BENCHMARKS[task]
        df = pd.read_csv(REPO / spec.dataset_csv)
        frames.append(
            pd.DataFrame(
                {
                    "task_type": task,
                    "question": df["question"],
                    "image": df["image"] if "image" in df.columns else "",
                }
            )
        )
        print(f"[ok] {task}: n={len(df)} ({spec.dataset_csv})")

    full = pd.concat(frames, ignore_index=True)
    if full["question"].isna().any():
        sys.exit("ERROR: empty question cells in the merged routing dataset")

    full.to_csv(OUT_CSV, index=False)
    print(f"\n[saved] {OUT_CSV} (n={len(full)})")
    print(full["task_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
