"""Basic RAG failure analysis on rule_retrieval (3 LLMs x TXT/MD, 250-token chunks, dense top-15).

F1 distribution, refusal rate, cross-LLM failure consistency (same questions fail for every LLM,
i.e. the retriever is the bottleneck), ground-truth length vs. F1, and examples.

Usage:
    uv run python scripts/analyze_basic_rag_failure.py --output scripts/basic_rag_failure_analysis.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.analysis_helpers import extract_rule_id, load_result_csv

RESULTS_BASE = Path("results/rule_extraction/rule_retrieval")


PIPELINES = {
    "gemma_txt": {
        "path": RESULTS_BASE / "gemma_basic_rag_small_chunks/sent250_d_k15_t00_norr-dc6c7b3cb2/results.csv",
        "llm": "Gemma-3-27B",
        "parsing": "TXT",
    },
    "mistral_txt": {
        "path": RESULTS_BASE / "mistral_basic_rag_small_chunks/sent250_d_k15_t00_norr-222679d69d/results.csv",
        "llm": "Mistral-Large",
        "parsing": "TXT",
    },
    "gpt_txt": {
        "path": RESULTS_BASE / "gpt-oss_basic_rag_small_chunks/sent250_d_k15_t00_norr-29b720c3c7/results.csv",
        "llm": "GPT-OSS-120B",
        "parsing": "TXT",
    },
    "gemma_md": {
        "path": RESULTS_BASE / "gemma_basic_rag_small_chunks_md/sent250_d_k15_t00_norr-105c9c24f2/results.csv",
        "llm": "Gemma-3-27B",
        "parsing": "MD",
    },
    "mistral_md": {
        "path": RESULTS_BASE / "mistral_basic_rag_small_chunks_md/sent250_d_k15_t00_norr-510d65b2b3/results.csv",
        "llm": "Mistral-Large",
        "parsing": "MD",
    },
    "gpt_md": {
        "path": RESULTS_BASE / "gpt-oss_basic_rag_small_chunks_md/sent250_d_k15_t00_norr-976b2689e0/results.csv",
        "llm": "GPT-OSS-120B",
        "parsing": "MD",
    },
}

TXT_PIPELINE_KEYS = ["gemma_txt", "mistral_txt", "gpt_txt"]

REFUSAL_PATTERNS = [
    r"context does not contain",
    r"provided context does not",
    r"not found in the (provided |given )?context",
    r"no information (is )?(provided|available|found)",
    r"cannot (find|determine|answer)",
    r"does not (provide|mention|include|contain)",
    r"not mentioned in",
    r"not (provided|included|available) in",
    r"unable to (find|answer|determine)",
    r"i (do not|don't) have (enough )?(information|context)",
]

REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def load_pipeline(key):
    return load_result_csv(PIPELINES[key]["path"])


def is_refusal(prediction):
    if not isinstance(prediction, str):
        return False
    return bool(REFUSAL_RE.search(prediction))


def approx_token_count(text):
    if not isinstance(text, str):
        return 0
    return len(text.split())


def analyze_f1_distribution(data):
    results = {}
    bins = [i / 10 for i in range(11)]

    for key, df in data.items():
        scores = df["eval"].dropna().values
        counts, _ = np.histogram(scores, bins=bins)
        cfg = PIPELINES[key]

        results[key] = {
            "llm": cfg["llm"],
            "parsing": cfg["parsing"],
            "n_questions": len(scores),
            "mean_f1": float(np.mean(scores)),
            "median_f1": float(np.median(scores)),
            "std_f1": float(np.std(scores)),
            "failure_rate_below_0.1": float(np.mean(scores < 0.1)),
            "failure_rate_zero": float(np.mean(scores == 0.0)),
            "success_rate_above_0.5": float(np.mean(scores > 0.5)),
            "histogram_bins": [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)],
            "histogram_counts": [int(c) for c in counts],
        }

    return results


def analyze_refusal_rate(data):
    results = {}

    for key, df in data.items():
        refusals = df["model_prediction"].apply(is_refusal)
        cfg = PIPELINES[key]

        results[key] = {
            "llm": cfg["llm"],
            "parsing": cfg["parsing"],
            "n_total": len(df),
            "n_refusals": int(refusals.sum()),
            "refusal_rate": float(refusals.mean()),
            "mean_f1_refusals": float(df.loc[refusals, "eval"].mean()) if refusals.any() else None,
            "mean_f1_non_refusals": float(df.loc[~refusals, "eval"].mean()) if (~refusals).any() else None,
        }

    return results


def analyze_cross_llm_consistency(data):
    dfs = {}
    for key in TXT_PIPELINE_KEYS:
        if key not in data:
            continue
        df = data[key].copy()
        df = df.rename(columns={"eval": f"f1_{key}"})
        dfs[key] = df[["question", f"f1_{key}"]]

    if len(dfs) < 3:
        return {"error": "Need all 3 TXT pipelines for cross-LLM analysis"}

    merged = dfs[TXT_PIPELINE_KEYS[0]]
    for key in TXT_PIPELINE_KEYS[1:]:
        merged = merged.merge(dfs[key], on="question", how="inner")

    f1_cols = [f"f1_{k}" for k in TXT_PIPELINE_KEYS]
    n_matched = len(merged)

    threshold = 0.1

    fail_matrix = merged[f1_cols].values < threshold
    n_failing = fail_matrix.sum(axis=1)

    any_fail_mask = n_failing >= 1
    n_any_fail = int(any_fail_mask.sum())

    fail_at_1 = int((n_failing == 1).sum())
    fail_at_2 = int((n_failing == 2).sum())
    fail_at_3 = int((n_failing == 3).sum())

    consistency_rate = fail_at_3 / n_any_fail if n_any_fail > 0 else 0.0

    correlations = {}
    for i, k1 in enumerate(TXT_PIPELINE_KEYS):
        for k2 in TXT_PIPELINE_KEYS[i + 1:]:
            col1 = f"f1_{k1}"
            col2 = f"f1_{k2}"
            valid = merged[[col1, col2]].dropna()
            pearson_r, pearson_p = stats.pearsonr(valid[col1], valid[col2])
            spearman_r, spearman_p = stats.spearmanr(valid[col1], valid[col2])
            pair = f"{PIPELINES[k1]['llm']} vs {PIPELINES[k2]['llm']}"
            correlations[pair] = {
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
            }

    return {
        "n_matched_questions": n_matched,
        "failure_threshold": threshold,
        "n_any_fail": n_any_fail,
        "fail_at_1_llm_only": fail_at_1,
        "fail_at_2_llms": fail_at_2,
        "fail_at_3_llms": fail_at_3,
        "consistency_rate": float(consistency_rate),
        "interpretation": (
            f"{consistency_rate:.1%} of failing questions fail at ALL 3 LLMs "
            f"(retriever bottleneck {'confirmed' if consistency_rate > 0.7 else 'not confirmed'})"
        ),
        "correlations": correlations,
    }


def analyze_gt_length(data):
    key = "gemma_txt"
    if key not in data:
        return {"error": f"{key} not available"}

    df = data[key].copy()
    df["gt_chars"] = df["ground_truth"].apply(lambda x: len(x) if isinstance(x, str) else 0)
    df["gt_tokens"] = df["ground_truth"].apply(approx_token_count)
    df = df.dropna(subset=["eval"])

    pearson_r, pearson_p = stats.pearsonr(df["gt_tokens"], df["eval"])
    spearman_r, spearman_p = stats.spearmanr(df["gt_tokens"], df["eval"])

    chunk_size = 250
    over_chunk = df["gt_tokens"] > chunk_size
    under_chunk = df["gt_tokens"] <= chunk_size

    bins = [0, 50, 100, 150, 250, 500, float("inf")]
    labels = ["0-50", "51-100", "101-150", "151-250", "251-500", "500+"]
    df["gt_bin"] = pd.cut(df["gt_tokens"], bins=bins, labels=labels, right=True)
    binned = df.groupby("gt_bin", observed=True)["eval"].agg(["mean", "count"]).reset_index()

    return {
        "n_questions": len(df),
        "gt_token_stats": {
            "mean": float(df["gt_tokens"].mean()),
            "median": float(df["gt_tokens"].median()),
            "std": float(df["gt_tokens"].std()),
            "max": int(df["gt_tokens"].max()),
        },
        "correlation": {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
        },
        "chunk_boundary_analysis": {
            "chunk_size_tokens": chunk_size,
            "n_over_chunk": int(over_chunk.sum()),
            "pct_over_chunk": float(over_chunk.mean()),
            "mean_f1_over_chunk": float(df.loc[over_chunk, "eval"].mean()) if over_chunk.any() else None,
            "mean_f1_under_chunk": float(df.loc[under_chunk, "eval"].mean()) if under_chunk.any() else None,
        },
        "binned_f1": [
            {"bin": row["gt_bin"], "mean_f1": float(row["mean"]), "n": int(row["count"])}
            for _, row in binned.iterrows()
        ],
    }


def select_examples(data):
    dfs = {}
    for key in TXT_PIPELINE_KEYS:
        if key not in data:
            continue
        dfs[key] = data[key].set_index("question")

    if len(dfs) < 3:
        return {"error": "Need all 3 TXT pipelines"}

    common_qs = set(dfs[TXT_PIPELINE_KEYS[0]].index)
    for key in TXT_PIPELINE_KEYS[1:]:
        common_qs &= set(dfs[key].index)

    failures = []
    successes = []

    for q in common_qs:
        f1_scores = {k: float(dfs[k].loc[q, "eval"]) for k in TXT_PIPELINE_KEYS}
        gt = str(dfs[TXT_PIPELINE_KEYS[0]].loc[q, "ground_truth"])
        gt_tokens = approx_token_count(gt)
        rule_id = extract_rule_id(q) or "unknown"

        all_fail = all(f1 < 0.1 for f1 in f1_scores.values())
        all_success = all(f1 > 0.5 for f1 in f1_scores.values())

        entry = {
            "rule_id": rule_id,
            "gt_tokens": gt_tokens,
            "gt_preview": gt[:200] + ("..." if len(gt) > 200 else ""),
            "f1_scores": {PIPELINES[k]["llm"]: round(f1_scores[k], 3) for k in TXT_PIPELINE_KEYS},
        }

        if all_fail:
            pred = str(dfs[TXT_PIPELINE_KEYS[0]].loc[q, "model_prediction"])
            entry["prediction_preview"] = pred[:200] + ("..." if len(pred) > 200 else "")
            entry["is_refusal"] = is_refusal(pred)
            failures.append(entry)

        if all_success:
            successes.append(entry)

    failures.sort(key=lambda x: x["gt_tokens"])
    successes.sort(key=lambda x: -min(x["f1_scores"].values()))

    return {
        "n_common_questions": len(common_qs),
        "n_all_fail": len(failures),
        "n_all_succeed": len(successes),
        "failure_examples": failures[:5],
        "success_examples": successes[:3],
    }


def print_summary(results):
    sep = "=" * 80

    print(f"\n{sep}")
    print("  1. F1 SCORE DISTRIBUTION")
    print(sep)
    print(f"{'Pipeline':<25} {'N':>5} {'Mean':>6} {'Med':>6} {'F1<0.1':>7} {'F1=0':>6} {'F1>0.5':>7}")
    print("-" * 70)
    for key, stats_d in results["f1_distribution"].items():
        cfg = PIPELINES[key]
        label = f"{cfg['llm']} ({cfg['parsing']})"
        print(
            f"{label:<25} {stats_d['n_questions']:>5} "
            f"{stats_d['mean_f1']:>6.3f} {stats_d['median_f1']:>6.3f} "
            f"{stats_d['failure_rate_below_0.1']:>6.1%} {stats_d['failure_rate_zero']:>5.1%} "
            f"{stats_d['success_rate_above_0.5']:>6.1%}"
        )

    print(f"\n{sep}")
    print("  2. RETRIEVAL-FAILURE REFUSAL RATE")
    print(sep)
    print(f"{'Pipeline':<25} {'Refusals':>9} {'Rate':>7} {'F1 ref.':>8} {'F1 non-ref.':>11}")
    print("-" * 65)
    for key, ref in results["refusal_rate"].items():
        cfg = PIPELINES[key]
        label = f"{cfg['llm']} ({cfg['parsing']})"
        f1_ref = f"{ref['mean_f1_refusals']:.3f}" if ref["mean_f1_refusals"] is not None else "n/a"
        f1_nonref = f"{ref['mean_f1_non_refusals']:.3f}" if ref["mean_f1_non_refusals"] is not None else "n/a"
        print(
            f"{label:<25} {ref['n_refusals']:>4}/{ref['n_total']:<4} "
            f"{ref['refusal_rate']:>6.1%} {f1_ref:>8} {f1_nonref:>11}"
        )

    print(f"\n{sep}")
    print("  3. CROSS-LLM FAILURE CONSISTENCY (TXT pipelines)")
    print(sep)
    cc = results["cross_llm_consistency"]
    if "error" not in cc:
        print(f"  Matched questions: {cc['n_matched_questions']}")
        print(f"  Questions failing at any LLM:  {cc['n_any_fail']}")
        print(f"    - at exactly 1 LLM:  {cc['fail_at_1_llm_only']}")
        print(f"    - at exactly 2 LLMs: {cc['fail_at_2_llms']}")
        print(f"    - at ALL 3 LLMs:     {cc['fail_at_3_llms']}")
        print(f"  Consistency rate: {cc['consistency_rate']:.1%}")
        print(f"  >> {cc['interpretation']}")
        print()
        print("  F1 Correlations between LLMs:")
        for pair, corr in cc["correlations"].items():
            print(f"    {pair}: Pearson r={corr['pearson_r']:.3f}, Spearman r={corr['spearman_r']:.3f}")
    else:
        print(f"  {cc['error']}")

    print(f"\n{sep}")
    print("  4. GROUND-TRUTH LENGTH vs. F1 SCORE")
    print(sep)
    gl = results["gt_length_analysis"]
    if "error" not in gl:
        gts = gl["gt_token_stats"]
        print(f"  GT token stats: mean={gts['mean']:.0f}, median={gts['median']:.0f}, max={gts['max']}")
        corr = gl["correlation"]
        print(f"  Correlation (tokens vs F1): Pearson r={corr['pearson_r']:.3f} (p={corr['pearson_p']:.2e})")
        cb = gl["chunk_boundary_analysis"]
        print(f"  GT > {cb['chunk_size_tokens']} tokens: {cb['n_over_chunk']} ({cb['pct_over_chunk']:.1%})")
        if cb["mean_f1_over_chunk"] is not None:
            print(f"    Mean F1 (over chunk):  {cb['mean_f1_over_chunk']:.3f}")
            print(f"    Mean F1 (under chunk): {cb['mean_f1_under_chunk']:.3f}")
        print()
        print(f"  {'Token bin':<12} {'Mean F1':>8} {'N':>5}")
        print("  " + "-" * 28)
        for b in gl["binned_f1"]:
            print(f"  {b['bin']:<12} {b['mean_f1']:>8.3f} {b['n']:>5}")
    else:
        print(f"  {gl['error']}")

    print(f"\n{sep}")
    print("  5. CONCRETE EXAMPLES")
    print(sep)
    ex = results["examples"]
    if "error" not in ex:
        print(f"  Common questions: {ex['n_common_questions']}")
        print(f"  All-3-LLM failures: {ex['n_all_fail']}, All-3-LLM successes: {ex['n_all_succeed']}")

        print("\n  --- FAILURE EXAMPLES (all 3 LLMs F1 < 0.1) ---")
        for i, f in enumerate(ex["failure_examples"], 1):
            f1s = ", ".join(f"{llm}: {v}" for llm, v in f["f1_scores"].items())
            print(f"\n  [{i}] Rule {f['rule_id']} (GT: {f['gt_tokens']} tokens, refusal: {f['is_refusal']})")
            print(f"      F1: {f1s}")
            print(f"      GT:   {f['gt_preview'][:120]}")
            print(f"      Pred: {f['prediction_preview'][:120]}")

        print("\n  --- SUCCESS EXAMPLES (all 3 LLMs F1 > 0.5) ---")
        for i, s in enumerate(ex["success_examples"], 1):
            f1s = ", ".join(f"{llm}: {v}" for llm, v in s["f1_scores"].items())
            print(f"\n  [{i}] Rule {s['rule_id']} (GT: {s['gt_tokens']} tokens)")
            print(f"      F1: {f1s}")
            print(f"      GT:   {s['gt_preview'][:120]}")

    print(f"\n{sep}\n")


def main():
    parser = argparse.ArgumentParser(description="Basic RAG Failure Analysis")
    parser.add_argument(
        "--output", "-o",
        default="scripts/basic_rag_failure_analysis.json",
        help="Path for JSON output (default: scripts/basic_rag_failure_analysis.json)",
    )
    args = parser.parse_args()

    data = {}
    for key in PIPELINES:
        df = load_pipeline(key)
        if df is not None:
            data[key] = df

    if not data:
        print("ERROR: No pipeline data found. Check file paths.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(data)} pipelines: {', '.join(data.keys())}")

    results = {
        "f1_distribution": analyze_f1_distribution(data),
        "refusal_rate": analyze_refusal_rate(data),
        "cross_llm_consistency": analyze_cross_llm_consistency(data),
        "gt_length_analysis": analyze_gt_length(data),
        "examples": select_examples(data),
    }

    print_summary(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
