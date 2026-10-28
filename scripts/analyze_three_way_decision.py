"""
Three-way decision (compliant / non-compliant / unknown) for the Gemma compliance configs from their 5 run_*.csv.
A question is decided when >= k runs give the same binary verdict (k = 3 majority, k = 5 unanimity), else
deferred; abstentions ("Rule not found") never vote. Reports coverage, selective risk (wrong / decided),
missed violations, false rejections and deferrals split by ground truth.

Usage:
    uv run python scripts/analyze_three_way_decision.py [--no-write]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_ravv_confusion import parse_pred

BASE = REPO / "results_for_publication" / "rule_compliance"
OUT_DIR = BASE / "three_way"

CONFIGS: dict[str, dict[str, Path]] = {
    task: {
        "No RAG (ICR)": BASE / task / "gemma_no_rag_vision",
        "RAVV (Gemma)": BASE / task / "gemma2gemma_ravv",
        "RAVV (GPT-OSS)": BASE / task / "gpt2gemma_ravv",
    }
    for task in ("dimension", "functional")
}
N_RUNS = 5
K_VALUES = (3, 4, 5)


def load_votes(run_dir: Path) -> tuple[dict[str, dict], int]:
    run_csvs = sorted(run_dir.glob("run_*.csv"))
    if len(run_csvs) != N_RUNS:
        sys.exit(f"expected {N_RUNS} run CSVs in {run_dir}, found {len(run_csvs)}")
    votes: dict[str, dict] = {}
    n_abstain = 0
    for csv_path in run_csvs:
        for row in pd.read_csv(csv_path).itertuples(index=False):
            key = f"{row.image} || {row.question}"
            gt = str(row.ground_truth).strip().lower()
            pred = parse_pred(str(row.model_prediction), gt, str(row.eval))
            n_abstain += int(pred == "abstain")
            rec = votes.setdefault(key, {"gt": gt, "preds": []})
            if rec["gt"] != gt:
                sys.exit(f"inconsistent ground truth in {run_dir}: {key[:80]}")
            rec["preds"].append(pred)
    for key, rec in votes.items():
        if len(rec["preds"]) != N_RUNS:
            sys.exit(f"question seen {len(rec['preds'])}x in {run_dir}: {key[:80]}")
    return votes, n_abstain


def decide(preds: list[str], k: int) -> str:
    counts = Counter(p for p in preds if p in ("yes", "no"))
    for label in ("yes", "no"):
        if counts[label] >= k:
            return label
    return "unknown"


def evaluate(votes: dict[str, dict], k: int) -> dict:
    c: Counter = Counter()
    for rec in votes.values():
        gt, d = rec["gt"], decide(rec["preds"], k)
        c["n"] += 1
        c[f"n_{gt}"] += 1
        if d == "unknown":
            c["deferred_nc" if gt == "no" else "deferred_c"] += 1
            continue
        c["decided"] += 1
        if d != gt:
            c["wrong"] += 1
            c["missed_violations" if gt == "no" else "false_rejections"] += 1
    decided = c["decided"]
    return {
        "n": c["n"],
        "n_noncompliant": c["n_no"],
        "n_compliant": c["n_yes"],
        "decided": decided,
        "coverage": decided / c["n"],
        "selective_risk": c["wrong"] / decided if decided else None,
        "missed_violations": c["missed_violations"],
        "false_rejections": c["false_rejections"],
        "deferred_nc": c["deferred_nc"],
        "deferred_c": c["deferred_c"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--no-write", action="store_true", help="stdout only")
    args = ap.parse_args()

    rows = []
    for task, configs in CONFIGS.items():
        for label, run_dir in configs.items():
            votes, n_abstain = load_votes(run_dir)
            for k in K_VALUES:
                rows.append({"task": task, "config": label, "k": k,
                             "explicit_abstentions": n_abstain,
                             **evaluate(votes, k)})
    df = pd.DataFrame(rows)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if not args.no_write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_DIR / "three_way.csv", index=False)
        (OUT_DIR / "three_way.json").write_text(
            json.dumps({"decision_rule": "decided if >= k of 5 runs agree; abstentions never vote",
                        "rows": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"\nwritten to {OUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
