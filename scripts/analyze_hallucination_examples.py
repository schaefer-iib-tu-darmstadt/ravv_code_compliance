"""Multi-run yes/no analysis of the compliance benchmarks: confusion matrices, yes-bias, known hallucination cases.

Reads results/results_index.csv (only -rN multi-run entries unless --all-runs) and each run's results.csv.

Usage:
    uv run python scripts/analyze_hallucination_examples.py --pipeline gemma_no_rag_vision
    uv run python scripts/analyze_hallucination_examples.py --output results/hallucination_analysis.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.analysis_helpers import (
    extract_rule_from_question,
    extract_yes_no,
    resolve_path,
)

TARGET_PIPELINES = [
    "gemma_no_rag_vision",
    "gemma2gemma_retrieval2vision",
    "gpt2gemma_retrieval2vision",
]

KNOWN_HALLUCINATION_CASES = [
    {"rule": "T.5.2.7", "benchmark": "rule_dimension", "label": "T.5.2.7a"},
    {"rule": "EV.5.8.4", "benchmark": "rule_functional", "label": "EV.5.8.4 (Q1)"},
    {"rule": "EV.5.8.4", "benchmark": "rule_functional", "label": "EV.5.8.4 (Q2)", "occurrence": 2},
]

COMPLIANCE_BENCHMARKS = ["rule_dimension", "rule_functional"]


def compute_confusion_matrix(gt_labels: list, pred_labels: list) -> dict:
    tp = fp = tn = fn = n_valid = 0
    n_unparseable = 0

    for gt, pred in zip(gt_labels, pred_labels):
        gt_str = str(gt).strip().lower() if pd.notna(gt) else None
        if gt_str not in ("yes", "no"):
            continue

        if pred is None:
            n_unparseable += 1
            continue

        n_valid += 1
        if gt_str == "yes" and pred == "yes":
            tp += 1
        elif gt_str == "no" and pred == "yes":
            fp += 1
        elif gt_str == "no" and pred == "no":
            tn += 1
        elif gt_str == "yes" and pred == "no":
            fn += 1

    n_scored = n_valid + n_unparseable
    accuracy = (tp + tn) / n_scored if n_scored > 0 else 0.0
    n_gt_yes = tp + fn
    n_gt_no = tn + fp
    n_pred_yes = tp + fp
    n_pred_no = tn + fn

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "accuracy": round(accuracy, 4),
        "n_valid": n_valid,
        "n_unparseable": n_unparseable,
        "n_gt_yes": n_gt_yes,
        "n_gt_no": n_gt_no,
        "n_pred_yes": n_pred_yes,
        "n_pred_no": n_pred_no,
        "yes_rate_gt": round(n_gt_yes / n_valid, 4) if n_valid else 0,
        "yes_rate_pred": round(n_pred_yes / n_valid, 4) if n_valid else 0,
    }


def track_known_cases(df: pd.DataFrame, benchmark: str) -> list:
    results = []
    rule_occurrence_counter = defaultdict(int)

    for _, row in df.iterrows():
        question = row.get("question") or row.get("asked_question", "")
        rule = extract_rule_from_question(question)
        if rule is None:
            continue

        rule_occurrence_counter[rule] += 1
        occ = rule_occurrence_counter[rule]

        gt = str(row.get("ground_truth", "")).strip().lower()
        pred = extract_yes_no(str(row.get("model_prediction", "")))

        for case in KNOWN_HALLUCINATION_CASES:
            if case["benchmark"] != benchmark:
                continue
            case_rule = case["rule"].upper()
            if not rule.startswith(case_rule):
                continue
            expected_occ = case.get("occurrence", 1)
            if occ != expected_occ:
                continue

            results.append({
                "label": case["label"],
                "rule": rule,
                "benchmark": benchmark,
                "ground_truth": gt,
                "predicted": pred,
                "correct": (gt == pred) if pred else False,
                "prediction_text_snippet": str(row.get("model_prediction", ""))[:200],
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-run hallucination analysis.")
    parser.add_argument("--pipeline", type=str, default=None, help="Filter by pipeline")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--multi-run-only", action="store_true", default=True,
                        help="Only include runs with -rN suffix (default: True)")
    parser.add_argument("--all-runs", action="store_true", default=False,
                        help="Include all runs (overrides --multi-run-only)")
    args = parser.parse_args()
    if args.all_runs:
        args.multi_run_only = False

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    index_path = os.path.join(project_root, "results", "results_index.csv")

    if not os.path.exists(index_path):
        print(f"ERROR: results_index.csv not found at {index_path}")
        sys.exit(1)

    index = pd.read_csv(index_path)

    target = args.pipeline.split(",") if args.pipeline else TARGET_PIPELINES
    index = index[
        (index["benchmark"].isin(COMPLIANCE_BENCHMARKS))
        & (index["pipeline"].isin(target))
    ]

    if args.multi_run_only:
        mask = index["run_id"].str.contains(r"-r\d+$", regex=True, na=False)
        n_before = len(index)
        index = index[mask]
        print(f"Multi-run filter: {n_before} -> {len(index)} entries (kept -rN suffixed runs only)\n")

    if index.empty:
        print("No matching compliance runs found.")
        sys.exit(0)

    index = index.sort_values("timestamp").drop_duplicates(
        subset=["pipeline", "benchmark", "run_id"], keep="last"
    )

    print(f"Found {len(index)} run entries across {len(target)} pipelines.\n")

    all_results = defaultdict(lambda: defaultdict(list))

    for _, run_entry in index.iterrows():
        pipeline = run_entry["pipeline"]
        benchmark = run_entry["benchmark"]
        run_id = run_entry["run_id"]
        result_dir = resolve_path(str(run_entry["result_dir"]), project_root)
        csv_path = os.path.join(result_dir, "results.csv")

        if not os.path.exists(csv_path):
            print(f"[WARN] CSV not found: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Could not read {csv_path}: {e}")
            continue

        pred_labels = [extract_yes_no(str(p)) for p in df["model_prediction"]]

        cm = compute_confusion_matrix(df["ground_truth"].tolist(), pred_labels)

        known = track_known_cases(df, benchmark)

        all_results[pipeline][benchmark].append({
            "run_id": run_id,
            "n_questions": len(df),
            "confusion_matrix": cm,
            "known_cases": known,
        })

        print(f"  {pipeline}/{benchmark} run={run_id}: "
              f"ACC={cm['accuracy']:.3f} TP={cm['TP']} FP={cm['FP']} TN={cm['TN']} FN={cm['FN']} "
              f"yes_rate_pred={cm['yes_rate_pred']:.2f}")

    print(f"\n{'=' * 80}")
    print("AGGREGATED RESULTS")
    print(f"{'=' * 80}\n")

    output = {}

    for pipeline in target:
        output[pipeline] = {}

        for benchmark in COMPLIANCE_BENCHMARKS:
            runs = all_results[pipeline][benchmark]
            if not runs:
                continue

            n_runs = len(runs)

            cm_keys = ["TP", "FP", "TN", "FN", "accuracy", "yes_rate_pred", "yes_rate_gt"]
            cm_agg = {}
            for key in cm_keys:
                values = [r["confusion_matrix"][key] for r in runs]
                cm_agg[key] = {
                    "mean": round(float(np.mean(values)), 4),
                    "std": round(float(np.std(values)), 4) if len(values) > 1 else 0.0,
                    "values": values,
                }

            yes_bias = cm_agg["yes_rate_pred"]["mean"] - cm_agg["yes_rate_gt"]["mean"]

            print(f"--- {pipeline} / {benchmark} ({n_runs} runs) ---")
            print(f"  Accuracy:      {cm_agg['accuracy']['mean']:.4f} +/- {cm_agg['accuracy']['std']:.4f}")
            print(f"  TP:            {cm_agg['TP']['mean']:.1f} +/- {cm_agg['TP']['std']:.1f}")
            print(f"  FP:            {cm_agg['FP']['mean']:.1f} +/- {cm_agg['FP']['std']:.1f}")
            print(f"  TN:            {cm_agg['TN']['mean']:.1f} +/- {cm_agg['TN']['std']:.1f}")
            print(f"  FN:            {cm_agg['FN']['mean']:.1f} +/- {cm_agg['FN']['std']:.1f}")
            print(f"  Yes-Rate (GT): {cm_agg['yes_rate_gt']['mean']:.4f}")
            print(f"  Yes-Rate (Pred): {cm_agg['yes_rate_pred']['mean']:.4f}")
            print(f"  Yes-Bias:      {yes_bias:+.4f}")

            known_stability = defaultdict(list)
            for run in runs:
                for case in run["known_cases"]:
                    known_stability[case["label"]].append({
                        "run_id": run["run_id"],
                        "predicted": case["predicted"],
                        "correct": case["correct"],
                        "ground_truth": case["ground_truth"],
                    })

            if known_stability:
                print("\n  Known Hallucination Cases:")
                for label, entries in known_stability.items():
                    n_correct = sum(1 for e in entries if e["correct"])
                    preds = [e["predicted"] for e in entries]
                    gt = entries[0]["ground_truth"] if entries else "?"
                    print(f"    {label} (GT={gt}): "
                          f"{n_correct}/{len(entries)} correct, "
                          f"predictions={preds}")

            print()

            output[pipeline][benchmark] = {
                "n_runs": n_runs,
                "confusion_matrix_aggregate": cm_agg,
                "yes_bias": round(yes_bias, 4),
                "known_cases_stability": {
                    label: {
                        "n_runs_tracked": len(entries),
                        "n_correct": sum(1 for e in entries if e["correct"]),
                        "predictions": [e["predicted"] for e in entries],
                        "ground_truth": entries[0]["ground_truth"] if entries else None,
                        "stable": len(set(e["predicted"] for e in entries)) == 1,
                    }
                    for label, entries in known_stability.items()
                },
                "per_run": [
                    {
                        "run_id": r["run_id"],
                        "confusion_matrix": r["confusion_matrix"],
                    }
                    for r in runs
                ],
            }

    print(f"{'=' * 80}")
    print("CROSS-PIPELINE COMPARISON")
    print(f"{'=' * 80}\n")

    for benchmark in COMPLIANCE_BENCHMARKS:
        print(f"--- {benchmark} ---")
        print(f"  {'Pipeline':<40} {'ACC':>7} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5} {'YesBias':>8} {'Runs':>5}")
        for pipeline in target:
            data = output.get(pipeline, {}).get(benchmark)
            if not data:
                continue
            cm = data["confusion_matrix_aggregate"]
            print(f"  {pipeline:<40} "
                  f"{cm['accuracy']['mean']:>7.4f} "
                  f"{cm['TP']['mean']:>5.1f} "
                  f"{cm['FP']['mean']:>5.1f} "
                  f"{cm['TN']['mean']:>5.1f} "
                  f"{cm['FN']['mean']:>5.1f} "
                  f"{data['yes_bias']:>+8.4f} "
                  f"{data['n_runs']:>5}")
        print()

    out_path = args.output or os.path.join(project_root, "results", "hallucination_analysis.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
