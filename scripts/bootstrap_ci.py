"""Per-run bootstrap 95% CIs (10,000 question resamples, seed 42) plus cross-run mean +/- std.

Reads results/results_index.csv and each run's results.csv (offline, no LLM calls).
Writes results/bootstrap_ci_results.csv and results/bootstrap_ci_results_cross_run.csv.

Usage:
    uv run python scripts/bootstrap_ci.py
    uv run python scripts/bootstrap_ci.py --benchmark rule_retrieval --all-runs
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.analysis_helpers import resolve_path


def bootstrap_ci(
    scores: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    scores = scores[~np.isnan(scores)]
    n = len(scores)
    if n == 0:
        return {"mean": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n": 0}
    if n == 1:
        val = float(scores[0])
        return {"mean": val, "std": 0.0, "ci_lower": val, "ci_upper": val, "n": 1}

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1 - ci
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=0)),
        "ci_lower": lower,
        "ci_upper": upper,
        "n": n,
    }


BENCHMARK_METRIC_NAMES = {
    "rule_retrieval": "F1 (BoW)",
    "rule_compilation": "F1 (BoC)",
    "rule_definition": "F1 (BoC)",
    "rule_presence": "Accuracy",
    "rule_dimension": "Accuracy",
    "rule_functional": "Accuracy",
}


def main():
    parser = argparse.ArgumentParser(description="Bootstrap 95% CIs on evaluation metrics.")
    parser.add_argument("--benchmark", type=str, default=None, help="Filter by benchmark name")
    parser.add_argument("--pipeline", type=str, default=None, help="Filter by pipeline name")
    parser.add_argument("--n-bootstrap", type=int, default=10_000, help="Number of bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None, help="Custom output CSV path")
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

    if args.benchmark:
        index = index[index["benchmark"] == args.benchmark]
    if args.pipeline:
        index = index[index["pipeline"].str.contains(args.pipeline)]

    if index.empty:
        print("No matching runs found in results_index.csv.")
        sys.exit(0)

    if args.multi_run_only:
        mask = index["run_id"].str.contains(r"-r\d+$", regex=True, na=False)
        n_before = len(index)
        index = index[mask]
        print(f"Multi-run filter: {n_before} -> {len(index)} entries (kept -rN suffixed runs only)\n")

    index = index.sort_values("timestamp").drop_duplicates(
        subset=["pipeline", "benchmark", "run_id"], keep="last"
    )

    rows = []

    for _, run in index.iterrows():
        result_dir = resolve_path(str(run["result_dir"]), project_root)
        csv_path = os.path.join(result_dir, "results.csv")

        if not os.path.exists(csv_path):
            print(f"[WARN] CSV not found: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Could not read {csv_path}: {e}")
            continue

        benchmark = run["benchmark"]
        pipeline = run["pipeline"]
        run_id = run["run_id"]
        config_id = re.sub(r"-r\d+$", "", str(run_id))
        metric_name = BENCHMARK_METRIC_NAMES.get(benchmark, "eval")

        if "eval" in df.columns:
            scores = pd.to_numeric(df["eval"], errors="coerce").dropna().values
            result = bootstrap_ci(scores, n_bootstrap=args.n_bootstrap, seed=args.seed)
            rows.append({
                "pipeline": pipeline,
                "benchmark": benchmark,
                "run_id": run_id,
                "config_id": config_id,
                "metric": metric_name,
                **result,
            })

        if "bleu_2" in df.columns:
            bleu_scores = pd.to_numeric(df["bleu_2"], errors="coerce").dropna().values
            if len(bleu_scores) > 0:
                result = bootstrap_ci(bleu_scores, n_bootstrap=args.n_bootstrap, seed=args.seed)
                rows.append({
                    "pipeline": pipeline,
                    "benchmark": benchmark,
                    "run_id": run_id,
                    "config_id": config_id,
                    "metric": "BLEU-2",
                    **result,
                })

        if "rouge_l" in df.columns:
            rouge_scores = pd.to_numeric(df["rouge_l"], errors="coerce").dropna().values
            if len(rouge_scores) > 0:
                result = bootstrap_ci(rouge_scores, n_bootstrap=args.n_bootstrap, seed=args.seed)
                rows.append({
                    "pipeline": pipeline,
                    "benchmark": benchmark,
                    "run_id": run_id,
                    "config_id": config_id,
                    "metric": "ROUGE-L",
                    **result,
                })

    if not rows:
        print("No valid results found to compute CIs.")
        sys.exit(0)

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(["benchmark", "pipeline", "metric"])

    out_path = args.output or os.path.join(project_root, "results", "bootstrap_ci_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Results saved to: {out_path}\n")

    print(f"{'Pipeline':<45} {'Benchmark':<20} {'Metric':<12} {'N':>5} "
          f"{'Mean':>7} {'Std':>7} {'CI_lo':>7} {'CI_hi':>7}")
    print("-" * 120)

    prev_group = None
    for _, r in results_df.iterrows():
        group = (r["benchmark"], r["pipeline"])
        if prev_group and group != prev_group:
            print()
        prev_group = group

        print(f"{r['pipeline']:<45} {r['benchmark']:<20} {r['metric']:<12} {r['n']:>5} "
              f"{r['mean']:>7.4f} {r['std']:>7.4f} {r['ci_lower']:>7.4f} {r['ci_upper']:>7.4f}")

    print(f"\n{'='*120}")
    print("CROSS-RUN AGGREGATION (mean +/- std of per-run macro scores)")
    print(f"{'='*120}\n")

    def std_ddof0(x):
        return x.std(ddof=0)

    cross_run = results_df.groupby(["pipeline", "benchmark", "metric"]).agg(
        n_runs=("mean", "count"),
        n_configs=("config_id", "nunique"),
        cross_mean=("mean", "mean"),
        cross_std=("mean", std_ddof0),
    ).reset_index()

    mixed = cross_run[cross_run["n_configs"] > 1]
    if not mixed.empty:
        print("[WARN] The following groups pool runs from MORE THAN ONE config "
              "(distinct run_id prefixes) — cross-run mean/std mixes configurations:")
        for _, r in mixed.iterrows():
            ids = sorted(results_df.loc[
                (results_df["pipeline"] == r["pipeline"])
                & (results_df["benchmark"] == r["benchmark"])
                & (results_df["metric"] == r["metric"]),
                "config_id",
            ].unique())
            print(f"  {r['pipeline']} / {r['benchmark']} / {r['metric']}: {ids}")
        print()

    print(f"{'Pipeline':<45} {'Benchmark':<20} {'Metric':<12} {'Runs':>5} "
          f"{'Mean':>7} {'Std':>7}")
    print("-" * 105)

    prev_group = None
    for _, r in cross_run.iterrows():
        group = (r["benchmark"], r["pipeline"])
        if prev_group and group != prev_group:
            print()
        prev_group = group

        std_str = f"{r['cross_std']:>7.4f}" if pd.notna(r["cross_std"]) else "    N/A"
        print(f"{r['pipeline']:<45} {r['benchmark']:<20} {r['metric']:<12} {r['n_runs']:>5} "
              f"{r['cross_mean']:>7.4f} {std_str}")

    cross_out = out_path.replace(".csv", "_cross_run.csv")
    cross_run.to_csv(cross_out, index=False)
    print(f"\nCross-run summary saved to: {cross_out}")

    print(f"\n{'='*120}")
    print("NOTE: Per-run CIs use per-question resampling. Cross-run aggregation uses run-level means.")


if __name__ == "__main__":
    main()
