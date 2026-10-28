"""Paired bootstrap 95% CIs for rule compilation (30 topics, 10,000 resamples, seed 42; offline).

Per-topic F1 is recomputed with eval_compilation_qa and must match the published macro-F1;
configs are paired by topic. Single runs at temperature 0, so CIs reflect topic sampling only.

Usage:
    uv run python scripts/bootstrap_compilation_ci.py
    uv run python scripts/bootstrap_compilation_ci.py --n-bootstrap 50000
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.metrics.metrics import eval_compilation_qa

PUB = REPO / "results_for_publication" / "rule_extraction" / "compilation"
RUNS = REPO / "results" / "rule_extraction" / "rule_compilation"
OUT_DIR = PUB / "bootstrap_ci"

TOPIC_RE = re.compile(r"relevant to `([^`]+)`")

CONFIGS: dict[str, tuple[Path, float]] = {
    "Structure-aware Graph RAG (full)": (
        RUNS / "structured_graph_rag" / "d_norr-20e6f730f7" / "results.csv",
        0.9156,
    ),
    "GPT-5 ICR": (PUB / "gpt-5_no_rag.csv", 0.7432),
    "Qwen3 ICR": (PUB / "qwen3_no_rag.csv", 0.4633),
    "Vector RAG best (Gemma Basic RAG)": (PUB / "gemma_basic_rag.csv", 0.3441),
    "Ontology-guided Graph RAG (GPT-5-mini)": (PUB / "gpt-5-mini_schema_graph_rag.csv", 0.4437),
    "Zero-shot Graph RAG (GPT-5-mini)": (PUB / "gpt-5-mini_simple_graph_rag.csv", 0.4075),
}

PRIMARY = "Structure-aware Graph RAG (full)"
PAIRED_REFS = ["GPT-5 ICR", "Qwen3 ICR", "Vector RAG best (Gemma Basic RAG)"]

SANITY_TOL = 0.005


def load_topic_scores(csv_path: Path) -> dict[str, float]:
    macro, scores = eval_compilation_qa(csv_path)
    df = pd.read_csv(csv_path)
    topics = []
    for q in df["question"]:
        m = TOPIC_RE.search(str(q))
        if not m:
            raise ValueError(f"could not parse topic from question in {csv_path}")
        topics.append(m.group(1))
    if len(topics) != len(scores):
        raise ValueError(f"row/score count mismatch in {csv_path}")
    if len(set(topics)) != len(topics):
        raise ValueError(f"duplicate topics in {csv_path}")
    return dict(zip(topics, scores))


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

    topic_scores: dict[str, dict[str, float]] = {}
    for label, (csv_path, published) in CONFIGS.items():
        scores = load_topic_scores(csv_path)
        macro = float(np.mean(list(scores.values())))
        if abs(macro - published) > SANITY_TOL:
            sys.exit(
                f"SANITY GATE FAILED for '{label}': recomputed macro-F1 "
                f"{macro:.4f} vs published {published:.4f} ({csv_path})"
            )
        topic_scores[label] = scores
        print(f"[ok] {label}: macro-F1 {macro:.4f} == published {published:.4f} "
              f"(n={len(scores)})")

    rows = []
    for label, scores in topic_scores.items():
        vals = np.array(list(scores.values()))
        n = len(vals)
        boot = np.empty(args.n_bootstrap)
        for i in range(args.n_bootstrap):
            boot[i] = rng.choice(vals, size=n, replace=True).mean()
        lo, hi = percentile_ci(boot)
        rows.append({
            "config": label, "n_topics": n, "macro_f1": vals.mean(),
            "ci_lower": lo, "ci_upper": hi,
        })
    per_config = pd.DataFrame(rows)

    primary = topic_scores[PRIMARY]
    delta_rows = []
    for ref in PAIRED_REFS:
        ref_scores = topic_scores[ref]
        common = sorted(set(primary) & set(ref_scores))
        if len(common) != len(primary):
            print(f"[warn] topic overlap {len(common)}/{len(primary)} for '{ref}'")
        a = np.array([primary[t] for t in common])
        b = np.array([ref_scores[t] for t in common])
        deltas = a - b
        n = len(deltas)
        boot = np.empty(args.n_bootstrap)
        idx_range = np.arange(n)
        for i in range(args.n_bootstrap):
            idx = rng.choice(idx_range, size=n, replace=True)
            boot[i] = deltas[idx].mean()
        lo, hi = percentile_ci(boot)
        delta_rows.append({
            "comparison": f"{PRIMARY} - {ref}", "n_topics": n,
            "delta_mean": deltas.mean(), "ci_lower": lo, "ci_upper": hi,
            "p_boot_delta_le_0": float((boot <= 0).mean()),
        })
    paired = pd.DataFrame(delta_rows)

    pd.set_option("display.width", 160)
    print("\n=== Per-config macro-F1, 95% percentile bootstrap CI over topics ===")
    print(per_config.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== Paired deltas (structure-aware graph minus reference) ===")
    print(paired.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_config.to_csv(OUT_DIR / "per_config_ci.csv", index=False)
    paired.to_csv(OUT_DIR / "paired_delta_ci.csv", index=False)
    with open(OUT_DIR / "README.txt", "w", encoding="utf-8") as fh:
        fh.write(
            "Paired bootstrap 95% CIs for the rule-compilation comparisons "
            f"(n=30 topics, {args.n_bootstrap} resamples, seed {args.seed}).\n"
            "Produced by scripts/bootstrap_compilation_ci.py; per-topic F1 "
            "recomputed with eval.metrics.metrics.eval_compilation_qa and "
            "sanity-gated against the published macro values.\n"
            "Runs are deterministic (temp 0, single run); CIs express "
            "topic-sampling uncertainty, not run variance.\n"
        )
    print(f"\n[saved] {OUT_DIR}")


if __name__ == "__main__":
    main()
