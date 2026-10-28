"""Paired bootstrap 95% CIs for rule retrieval and ablations (1192 queries, 10,000 resamples, seed 42).

Offline: per-query BoW-F1 is recomputed with eval_retrieval_qa and must match the published
macro-F1; all configs share one question order. Single runs, so CIs reflect query sampling only.

Usage:
    uv run python scripts/bootstrap_retrieval_ci.py
    uv run python scripts/bootstrap_retrieval_ci.py --n-bootstrap 50000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.metrics.metrics import eval_retrieval_qa

PUB = REPO / "results_for_publication" / "rule_extraction" / "retrieval"
OUT_DIR = PUB / "bootstrap_ci"

CONFIGS: dict[str, tuple[Path, float]] = {
    "Gemma No RAG (ICR)": (PUB / "gemma_no_rag.csv", 0.383),
    "Mistral No RAG (ICR)": (PUB / "mistral_no_rag.csv", 0.756),
    "GPT-OSS No RAG (ICR)": (PUB / "gpt-oss_no_rag.csv", 0.794),
    "Qwen3 No RAG (ICR)": (PUB / "qwen3_no_rag.csv", 0.947),
    "Gemma Basic RAG (TXT)": (PUB / "gemma_basic_rag_txt.csv", 0.167),
    "Mistral Basic RAG (TXT)": (PUB / "mistral_basic_rag_txt.csv", 0.157),
    "GPT-OSS Basic RAG (TXT)": (PUB / "gpt-oss_basic_rag_txt.csv", 0.158),
    "Gemma Basic RAG (MD)": (PUB / "gemma_basic_rag_md.csv", 0.187),
    "Mistral Basic RAG (MD)": (PUB / "mistral_basic_rag_md.csv", 0.171),
    "GPT-OSS Basic RAG (MD)": (PUB / "gpt-oss_basic_rag_md.csv", 0.174),
    "Gemma Advanced RAG": (PUB / "gemma_advanced_rag.csv", 0.706),
    "Mistral Advanced RAG": (PUB / "mistral_advanced_rag.csv", 0.882),
    "GPT-OSS Advanced RAG": (PUB / "gpt-oss_advanced_rag.csv", 0.900),
    "GPT-OSS A1 (no parent expansion)": (PUB / "ablation_no_parent_expansion.csv", 0.281),
    "GPT-OSS A2 (no reranking)": (PUB / "ablation_no_reranking.csv", 0.888),
    "GPT-OSS A3 (flat chunking)": (PUB / "ablation_flat_chunking.csv", 0.315),
}

PAIRED: list[tuple[str, str]] = [
    ("Gemma Advanced RAG", "Gemma No RAG (ICR)"),
    ("Mistral Advanced RAG", "Mistral No RAG (ICR)"),
    ("GPT-OSS Advanced RAG", "GPT-OSS No RAG (ICR)"),
    ("Gemma Advanced RAG", "Gemma Basic RAG (MD)"),
    ("Mistral Advanced RAG", "Mistral Basic RAG (MD)"),
    ("GPT-OSS Advanced RAG", "GPT-OSS Basic RAG (MD)"),
    ("Gemma Basic RAG (MD)", "Gemma Basic RAG (TXT)"),
    ("Mistral Basic RAG (MD)", "Mistral Basic RAG (TXT)"),
    ("GPT-OSS Basic RAG (MD)", "GPT-OSS Basic RAG (TXT)"),
    ("GPT-OSS Advanced RAG", "GPT-OSS A1 (no parent expansion)"),
    ("GPT-OSS Advanced RAG", "GPT-OSS A2 (no reranking)"),
    ("GPT-OSS Advanced RAG", "GPT-OSS A3 (flat chunking)"),
]

SANITY_TOL = 0.005


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

    scores: dict[str, np.ndarray] = {}
    ref_questions: list[str] | None = None
    for label, (csv_path, published) in CONFIGS.items():
        macro, per_query = eval_retrieval_qa(csv_path)
        if abs(macro - published) > SANITY_TOL:
            sys.exit(
                f"SANITY GATE FAILED for '{label}': recomputed macro-F1 "
                f"{macro:.4f} vs published {published:.4f} ({csv_path})"
            )
        questions = pd.read_csv(csv_path)["question"].astype(str).tolist()
        if len(questions) != len(per_query):
            sys.exit(f"row/score count mismatch in {csv_path}")
        if ref_questions is None:
            ref_questions = questions
        elif questions != ref_questions:
            sys.exit(
                f"ALIGNMENT FAILED for '{label}': question sequence differs "
                f"from the first config, positional pairing would be invalid"
            )
        scores[label] = np.asarray(per_query, dtype=float)
        print(f"[ok] {label}: macro-F1 {macro:.4f} == published {published:.4f} "
              f"(n={len(per_query)})")

    n = len(ref_questions or [])
    idx_range = np.arange(n)

    rows = []
    for label, vals in scores.items():
        boot = np.empty(args.n_bootstrap)
        for i in range(args.n_bootstrap):
            boot[i] = vals[rng.choice(idx_range, size=n, replace=True)].mean()
        lo, hi = percentile_ci(boot)
        rows.append({
            "config": label, "n_queries": n, "macro_f1": vals.mean(),
            "ci_lower": lo, "ci_upper": hi,
        })
    per_config = pd.DataFrame(rows)

    delta_rows = []
    for minuend, subtrahend in PAIRED:
        deltas = scores[minuend] - scores[subtrahend]
        boot = np.empty(args.n_bootstrap)
        for i in range(args.n_bootstrap):
            boot[i] = deltas[rng.choice(idx_range, size=n, replace=True)].mean()
        lo, hi = percentile_ci(boot)
        delta_rows.append({
            "comparison": f"{minuend} - {subtrahend}", "n_queries": n,
            "delta_mean": deltas.mean(), "ci_lower": lo, "ci_upper": hi,
            "p_boot_delta_le_0": float((boot <= 0).mean()),
        })
    paired = pd.DataFrame(delta_rows)

    pd.set_option("display.width", 160)
    print("\n=== Per-config macro-F1, 95% percentile bootstrap CI over queries ===")
    print(per_config.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== Paired deltas (minuend minus subtrahend) ===")
    print(paired.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_config.to_csv(OUT_DIR / "per_config_ci.csv", index=False)
    paired.to_csv(OUT_DIR / "paired_delta_ci.csv", index=False)
    with open(OUT_DIR / "README.txt", "w", encoding="utf-8") as fh:
        fh.write(
            "Paired bootstrap 95% CIs for the retrieval and ablation comparisons "
            f"(n=1192 queries, {args.n_bootstrap} resamples, seed {args.seed}).\n"
            "Produced by scripts/bootstrap_retrieval_ci.py; per-query BoW-F1 "
            "recomputed with eval.metrics.metrics.eval_retrieval_qa and "
            "sanity-gated against the published macro values.\n"
            "Single runs at temperature 0; CIs express query-sampling "
            "uncertainty, not run variance.\n"
        )
    print(f"\n[saved] {OUT_DIR}")


if __name__ == "__main__":
    main()
