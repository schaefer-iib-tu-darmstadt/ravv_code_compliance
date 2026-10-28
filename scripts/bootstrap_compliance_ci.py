"""Paired bootstrap 95% CIs for compliance safety: violation recall, false alarms, missed violations.

Offline over the five run_*.csv per config, gated against the published means; questions are
resampled by their across-run rate (10,000 resamples, seed 42), paired RAVV vs. No RAG.

Usage:
    uv run python scripts/bootstrap_compliance_ci.py
    uv run python scripts/bootstrap_compliance_ci.py --n-bootstrap 50000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_ravv_confusion import parse_pred

BASE = REPO / "results_for_publication" / "rule_compliance"
OUT_DIR = BASE / "bootstrap_ci"

CONFIGS: dict[str, dict[str, Path]] = {
    "dimension": {
        "No RAG (ICR)": BASE / "dimension" / "gemma_no_rag_vision",
        "RAVV (Gemma)": BASE / "dimension" / "gemma2gemma_ravv",
        "RAVV (GPT-OSS)": BASE / "dimension" / "gpt2gemma_ravv",
    },
    "functional": {
        "No RAG (ICR)": BASE / "functional" / "gemma_no_rag_vision",
        "RAVV (Gemma)": BASE / "functional" / "gemma2gemma_ravv",
        "RAVV (GPT-OSS)": BASE / "functional" / "gpt2gemma_ravv",
    },
}

BASELINE = "No RAG (ICR)"
N_RUNS = 5

PUBLISHED: dict[tuple[str, str], tuple[float, float, float]] = {
    ("dimension", "No RAG (ICR)"): (0.500, 0.165, 10.0),
    ("dimension", "RAVV (Gemma)"): (0.570, 0.330, 8.6),
    ("dimension", "RAVV (GPT-OSS)"): (0.780, 0.160, 4.4),
    ("functional", "No RAG (ICR)"): (0.625, 0.150, 3.0),
    ("functional", "RAVV (Gemma)"): (0.925, 0.375, 0.6),
    ("functional", "RAVV (GPT-OSS)"): (0.800, 0.225, 1.6),
}

RATE_TOL = 0.005
COUNT_TOL = 0.051


def load_question_rates(run_dir: Path) -> pd.DataFrame:
    run_csvs = sorted(run_dir.glob("run_*.csv"))
    if len(run_csvs) != N_RUNS:
        sys.exit(f"expected {N_RUNS} run CSVs in {run_dir}, found {len(run_csvs)}")

    counts: dict[str, dict[str, float]] = {}
    for csv_path in run_csvs:
        df = pd.read_csv(csv_path)
        for row in df.itertuples(index=False):
            q = f"{row.image} || {row.question}"
            gt = str(row.ground_truth).strip().lower()
            pred = parse_pred(str(row.model_prediction), gt, str(row.eval))
            rec = counts.setdefault(q, {"gt": gt, "n": 0, "miss": 0, "caught": 0, "fa": 0})
            if rec["gt"] != gt:
                sys.exit(f"inconsistent ground truth for question in {run_dir}: {q[:80]}")
            rec["n"] += 1
            if gt == "no":
                rec["miss"] += int(pred == "yes")
                rec["caught"] += int(pred == "no")
            elif gt == "yes":
                rec["fa"] += int(pred == "no")

    rows = []
    for q, rec in counts.items():
        if rec["n"] != N_RUNS:
            sys.exit(f"question seen {rec['n']}x (expected {N_RUNS}) in {run_dir}: {q[:80]}")
        rows.append({
            "question": q, "gt": rec["gt"],
            "miss_rate": rec["miss"] / N_RUNS,
            "caught_rate": rec["caught"] / N_RUNS,
            "fa_rate": rec["fa"] / N_RUNS,
        })
    return pd.DataFrame(rows).set_index("question")


def percentile_ci(samples: np.ndarray, ci: float = 0.95) -> tuple[float, float]:
    alpha = 1 - ci
    return (
        float(np.percentile(samples, 100 * alpha / 2)),
        float(np.percentile(samples, 100 * (1 - alpha / 2))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    per_config_rows = []
    delta_rows = []

    for task, config_dirs in CONFIGS.items():
        rates = {label: load_question_rates(d) for label, d in config_dirs.items()}
        ref = rates[BASELINE]
        noncomp_qs = sorted(ref.index[ref["gt"] == "no"])
        comp_qs = sorted(ref.index[ref["gt"] == "yes"])
        for label, df in rates.items():
            if sorted(df.index) != sorted(ref.index):
                sys.exit(f"question sets differ between configs on task '{task}' ({label})")

        n_nc, n_c = len(noncomp_qs), len(comp_qs)
        print(f"\n[{task}] non-compliant n={n_nc}, compliant n={n_c}")

        caught = {l: df.loc[noncomp_qs, "caught_rate"].to_numpy() for l, df in rates.items()}
        miss = {l: df.loc[noncomp_qs, "miss_rate"].to_numpy() for l, df in rates.items()}
        fa = {l: df.loc[comp_qs, "fa_rate"].to_numpy() for l, df in rates.items()}

        for label in config_dirs:
            recall_m = caught[label].mean()
            fa_m = fa[label].mean()
            missed_m = miss[label].sum()
            pub_recall, pub_fa, pub_missed = PUBLISHED[(task, label)]
            for name, got, want, tol in (
                ("violation recall", recall_m, pub_recall, RATE_TOL),
                ("false-alarm rate", fa_m, pub_fa, RATE_TOL),
                ("missed count", missed_m, pub_missed, COUNT_TOL),
            ):
                if abs(got - want) > tol:
                    sys.exit(
                        f"SANITY GATE FAILED [{task} / {label}] {name}: "
                        f"recomputed {got:.4f} vs published {want:.4f}"
                    )
            print(f"[ok] {task} / {label}: recall {recall_m:.3f}, "
                  f"false-alarm {fa_m:.3f}, missed {missed_m:.1f} == published")

        idx_nc, idx_c = np.arange(n_nc), np.arange(n_c)
        for label in config_dirs:
            boot_rec = np.empty(args.n_bootstrap)
            boot_missed = np.empty(args.n_bootstrap)
            boot_fa = np.empty(args.n_bootstrap)
            for i in range(args.n_bootstrap):
                s_nc = rng.choice(idx_nc, size=n_nc, replace=True)
                s_c = rng.choice(idx_c, size=n_c, replace=True)
                boot_rec[i] = caught[label][s_nc].mean()
                boot_missed[i] = miss[label][s_nc].mean() * n_nc
                boot_fa[i] = fa[label][s_c].mean()
            rec_lo, rec_hi = percentile_ci(boot_rec)
            mis_lo, mis_hi = percentile_ci(boot_missed)
            fa_lo, fa_hi = percentile_ci(boot_fa)
            per_config_rows.append({
                "task": task, "config": label,
                "n_noncompliant": n_nc, "n_compliant": n_c,
                "viol_recall": caught[label].mean(),
                "viol_recall_ci_lower": rec_lo, "viol_recall_ci_upper": rec_hi,
                "false_alarm": fa[label].mean(),
                "false_alarm_ci_lower": fa_lo, "false_alarm_ci_upper": fa_hi,
                "missed_count": miss[label].sum(),
                "missed_ci_lower": mis_lo, "missed_ci_upper": mis_hi,
            })

        for label in config_dirs:
            if label == BASELINE:
                continue
            d_rec = caught[label] - caught[BASELINE]
            d_miss = miss[label] - miss[BASELINE]
            d_fa = fa[label] - fa[BASELINE]
            boot_drec = np.empty(args.n_bootstrap)
            boot_dmiss = np.empty(args.n_bootstrap)
            boot_dfa = np.empty(args.n_bootstrap)
            boot_rel = np.full(args.n_bootstrap, np.nan)
            for i in range(args.n_bootstrap):
                s_nc = rng.choice(idx_nc, size=n_nc, replace=True)
                s_c = rng.choice(idx_c, size=n_c, replace=True)
                boot_drec[i] = d_rec[s_nc].mean()
                boot_dmiss[i] = d_miss[s_nc].mean() * n_nc
                boot_dfa[i] = d_fa[s_c].mean()
                base_missed = miss[BASELINE][s_nc].sum()
                if base_missed > 0:
                    boot_rel[i] = 1.0 - miss[label][s_nc].sum() / base_missed
            rel = boot_rel[~np.isnan(boot_rel)]
            rel_dropped = 1.0 - len(rel) / args.n_bootstrap
            drec_lo, drec_hi = percentile_ci(boot_drec)
            dmiss_lo, dmiss_hi = percentile_ci(boot_dmiss)
            dfa_lo, dfa_hi = percentile_ci(boot_dfa)
            rel_lo, rel_hi = percentile_ci(rel)
            base_missed_total = miss[BASELINE].sum()
            delta_rows.append({
                "task": task, "comparison": f"{label} - {BASELINE}",
                "n_noncompliant": n_nc,
                "delta_recall": d_rec.mean(),
                "delta_recall_ci_lower": drec_lo, "delta_recall_ci_upper": drec_hi,
                "p_boot_delta_recall_le_0": float((boot_drec <= 0).mean()),
                "delta_missed": d_miss.mean() * n_nc,
                "delta_missed_ci_lower": dmiss_lo, "delta_missed_ci_upper": dmiss_hi,
                "delta_false_alarm": d_fa.mean(),
                "delta_fa_ci_lower": dfa_lo, "delta_fa_ci_upper": dfa_hi,
                "rel_miss_reduction": 1.0 - miss[label].sum() / base_missed_total,
                "rel_reduction_ci_lower": rel_lo, "rel_reduction_ci_upper": rel_hi,
                "rel_resamples_dropped": rel_dropped,
            })

    per_config = pd.DataFrame(per_config_rows)
    paired = pd.DataFrame(delta_rows)

    pd.set_option("display.width", 220)
    print("\n=== Per-config rates, 95% percentile bootstrap CI over questions ===")
    print(per_config.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== Paired deltas (RAVV minus No RAG), incl. relative missed-violation reduction ===")
    print(paired.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_config.to_csv(OUT_DIR / "per_config_ci.csv", index=False)
    paired.to_csv(OUT_DIR / "paired_delta_ci.csv", index=False)
    with open(OUT_DIR / "README.txt", "w", encoding="utf-8") as fh:
        fh.write(
            "Paired bootstrap 95% CIs for the compliance safety comparison and the "
            f"relative missed-violation reduction ({args.n_bootstrap} resamples, "
            f"seed {args.seed}).\n"
            "Produced by scripts/bootstrap_compliance_ci.py from the persisted "
            "run_*.csv (5 runs per config); predictions parsed with the "
            "canonical parser from scripts/analyze_ravv_confusion.py and "
            "aggregates sanity-gated against the published means.\n"
            "CIs resample QUESTIONS (20 non-compliant / 40 compliant dimension, "
            "8/8 functional) using across-run rates per question; they express "
            "question-sampling uncertainty on top of the run-to-run std "
            "of the published results.\n"
            "For the relative reduction, resamples with a zero No-RAG missed "
            "count are dropped (share in rel_resamples_dropped).\n"
        )
    print(f"\n[saved] {OUT_DIR}")


if __name__ == "__main__":
    main()
