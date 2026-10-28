"""
ICR vs. RAVV (GPT-OSS retriever), both with gemma-3-27b-it via OpenRouter, on the synthetic violations.
Variants of one source image are not independent, so CIs are cluster bootstraps over source images (10,000
resamples, seed 7) and the paired test is an exact sign-flip permutation on per-source miss differences;
item-level McNemar/Wilcoxon for reference only. Also: false-alarm rate and three-way policy (k = 4, 5).

Usage:
    uv run python scripts/analyze_synthetic_gemma3_openrouter.py [--no-write]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_ravv_confusion import parse_pred

RESULTS = REPO / "results" / "rule_compliance"
OUT_DIR = REPO / "results_for_publication" / "rule_compliance" / "synthetic_gemma3_openrouter"
CONFIGS = {
    "ICR": "or_gemma3_no_rag_vision",
    "RAVV": "or_gpt2gemma3_retrieval2vision",
}
TASKS = {
    "dimension": ("rule_dimension_synthetic",
                  REPO / "dataset_synthetic" / "rule_dimension_synthetic" / "rule_dimension_synthetic.csv"),
    "functional": ("rule_functional_synthetic",
                   REPO / "dataset_synthetic" / "rule_functional_synthetic" / "rule_functional_synthetic.csv"),
}
N_BOOT = 10_000
SEED = 7


def load(bench: str, pipeline: str) -> pd.DataFrame:
    run_csvs = sorted(p for p in (RESULTS / bench / pipeline).glob("*-r*/results.csv")
                      if "-test" not in p.parent.name)
    frames = []
    for p in run_csvs:
        df = pd.read_csv(p)
        df["run"] = p.parent.name.rsplit("-r", 1)[1]
        df["label"] = df["ground_truth"].str.strip().str.lower()
        df["pred"] = [parse_pred(str(m), g, str(e))
                      for m, g, e in zip(df.model_prediction, df.label, df["eval"])]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def per_item(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("image")
    return pd.DataFrame({
        "truth": g["label"].first(),
        "runs": g.size(),
        "misses": g.apply(lambda x: ((x.label == "no") & (x.pred == "yes")).sum()),
        "false_alarms": g.apply(lambda x: ((x.label == "yes") & (x.pred == "no")).sum()),
        "votes_yes": g.apply(lambda x: (x.pred == "yes").sum()),
        "votes_no": g.apply(lambda x: (x.pred == "no").sum()),
    })


def cluster_ci(viol: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    agg = viol.groupby("source_image")[["misses", "runs"]].sum()
    m, n = agg.misses.to_numpy(), agg.runs.to_numpy()
    idx = rng.integers(0, len(agg), size=(N_BOOT, len(agg)))
    rates = m[idx].sum(axis=1) / n[idx].sum(axis=1)
    return float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))


def three_way(it: pd.DataFrame, k: int) -> dict:
    decided_yes = it.votes_yes >= k
    decided_no = it.votes_no >= k
    decided = decided_yes | decided_no
    wrong = (decided_yes & (it.truth == "no")) | (decided_no & (it.truth == "yes"))
    viol = it.truth == "no"
    return {
        "coverage": round(float(decided.mean()), 3),
        "selective_risk": round(float(wrong[decided].mean()), 3) if decided.any() else None,
        "decided_missed_violations": int((decided_yes & viol).sum()),
        "deferred_violations": int((~decided & viol).sum()),
        "decided_false_rejections": int((decided_no & ~viol).sum()),
        "deferred_compliant": int((~decided & ~viol).sum()),
    }


def summarise(it: pd.DataFrame, rng: np.random.Generator) -> dict:
    viol, comp = it[it.truth == "no"], it[it.truth == "yes"]
    runs = int(it.runs.min())
    lo, hi = cluster_ci(viol, rng)
    return {
        "runs": runs,
        "n_items": len(it),
        "n_violations": len(viol),
        "n_sources": int(it.source_image.nunique()),
        "missed_mean_per_run": round(viol.misses.sum() / runs, 2),
        "miss_rate": round(viol.misses.sum() / viol.runs.sum(), 3),
        "miss_rate_ci95_cluster": [round(lo, 3), round(hi, 3)],
        "missed_majority": int((viol.misses / viol.runs > 0.5).sum()),
        "false_alarm_rate": round(comp.false_alarms.sum() / comp.runs.sum(), 3) if len(comp) else None,
        "miss_rate_by_margin": (viol.groupby("rel_margin").misses.sum()
                                / viol.groupby("rel_margin").runs.sum()).round(3).to_dict(),
        "miss_rate_by_evidence": (viol.groupby("evidence").misses.sum()
                                  / viol.groupby("evidence").runs.sum()).round(3).to_dict(),
        "three_way_k4": three_way(it, 4),
        "three_way_k5": three_way(it, 5),
    }


def paired(icr: pd.DataFrame, ravv: pd.DataFrame, rng: np.random.Generator) -> dict:
    v = icr[icr.truth == "no"].join(ravv[ravv.truth == "no"][["misses", "runs"]],
                                    lsuffix="_icr", rsuffix="_ravv", how="inner")
    agg = v.groupby("source_image")[["misses_icr", "misses_ravv"]].sum()
    d = (agg.misses_icr - agg.misses_ravv).to_numpy()
    obs = d.sum()
    flips = np.array(list(itertools.product([1, -1], repeat=len(d))))
    p_perm = float((np.abs(flips @ d) >= abs(obs) - 1e-9).mean())
    mi, mr = agg.misses_icr.to_numpy(), agg.misses_ravv.to_numpy()
    idx = rng.integers(0, len(agg), size=(N_BOOT, len(agg)))
    si, sr = mi[idx].sum(axis=1), mr[idx].sum(axis=1)
    red = 1 - sr[si > 0] / si[si > 0]
    maj_i = v.misses_icr / v.runs_icr > 0.5
    maj_r = v.misses_ravv / v.runs_ravv > 0.5
    b_, c_ = int((maj_i & ~maj_r).sum()), int((~maj_i & maj_r).sum())
    di = (v.misses_icr - v.misses_ravv)
    di = di[di != 0]
    return {
        "n_violations": len(v),
        "n_sources": len(agg),
        "missed_icr_total": int(mi.sum()),
        "missed_ravv_total": int(mr.sum()),
        "relative_reduction": round(1 - mr.sum() / mi.sum(), 3) if mi.sum() else None,
        "relative_reduction_ci95_cluster": ([round(float(np.quantile(red, q)), 3) for q in (0.025, 0.975)]
                                            if len(red) else None),
        "cluster_signflip_p": round(p_perm, 4),
        "item_mcnemar_icr_only_missed": b_,
        "item_mcnemar_ravv_only_missed": c_,
        "item_mcnemar_exact_p": round(binomtest(b_, b_ + c_, 0.5).pvalue, 4) if b_ + c_ else None,
        "item_wilcoxon_p": round(float(wilcoxon(di).pvalue), 4) if len(di) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--no-write", action="store_true", help="stdout only")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    items: dict[tuple[str, str], pd.DataFrame] = {}
    for task, (bench, csv) in TASKS.items():
        meta = pd.read_csv(csv).set_index("image")[["evidence", "rel_margin", "source_image"]]
        meta["source_image"] = task + ":" + meta.source_image
        for cfg, pipe in CONFIGS.items():
            df = load(bench, pipe)
            if df.empty:
                print(f"no runs for {cfg} / {task}")
                continue
            items[(cfg, task)] = per_item(df).join(meta)
    for cfg in CONFIGS:
        parts = [items[(cfg, t)] for t in TASKS if (cfg, t) in items]
        if len(parts) == len(TASKS):
            items[(cfg, "pooled")] = pd.concat(parts)

    summary = {f"{cfg} | {task}": summarise(it, rng) for (cfg, task), it in items.items()}
    tests = {task: paired(items[("ICR", task)], items[("RAVV", task)], rng)
             for task in [*TASKS, "pooled"] if ("ICR", task) in items and ("RAVV", task) in items}

    result = {"summary": summary, "paired_tests": tests}
    print(json.dumps(result, indent=2, default=str))
    if not args.no_write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        for (cfg, task), it in items.items():
            if task != "pooled":
                it.to_csv(OUT_DIR / f"items_{cfg.lower()}_{task}.csv")
        print(f"\nwritten to {OUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
