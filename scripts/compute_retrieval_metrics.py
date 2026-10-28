"""Hit Rate@k (k = 1, 3, 5, 10, 15) and MRR of the retriever alone (retrieve(), no answer synthesis).

A node is relevant if it contains the question's rule ID or over 50 % of the ground-truth tokens.

Usage:
    uv run python scripts/compute_retrieval_metrics.py                             # basic_md + advanced, 1192 questions
    uv run python scripts/compute_retrieval_metrics.py --pipelines basic_md --n 5  # quick check
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llama_index.core.schema import QueryBundle

from src.utils.rag.query_engine_helpers import get_query_engine
from src.utils.rag.eval_utils import normalize_node
from src.utils.analysis_helpers import extract_rule_id

DATASET_PATH = Path("dataset/rule_extraction/rule_retrieval_qa.csv")

K_VALUES = [1, 3, 5, 10, 15]

PIPELINE_ALIASES = {
    "basic_md": "gemma_basic_rag_small_chunks_md",
    "basic_txt": "gemma_basic_rag_small_chunks",
    "advanced": "gpt-oss_advanced_hierarchical",
    "advanced_no_rr": "gpt-oss_ablation_no_reranking",
    "ablation_a1": "gpt-oss_ablation_no_parent_exp",
}

DEFAULT_PIPELINES = ["basic_md", "advanced"]


def is_relevant(node_text: str, rule_id: str, ground_truth: str) -> bool:
    pattern = r"(?<![A-Za-z0-9])" + re.escape(rule_id) + r"(?!\d)(?!\.\d)"
    if re.search(pattern, node_text):
        return True

    gt_tokens = set(ground_truth.lower().split())
    if not gt_tokens:
        return False
    node_tokens = set(node_text.lower().split())
    overlap = len(gt_tokens & node_tokens) / len(gt_tokens)
    return overlap > 0.5


def compute_hit_rate_at_k(relevance_flags: list[bool], k: int) -> int:
    return int(any(relevance_flags[:k]))


def compute_reciprocal_rank(relevance_flags: list[bool]) -> float:
    for i, flag in enumerate(relevance_flags):
        if flag:
            return 1.0 / (i + 1)
    return 0.0


def run_retrieval_evaluation(
    pipeline_name: str,
    df: pd.DataFrame,
    k_values: list[int],
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Pipeline: {pipeline_name}")
    print(f"  Questions: {len(df)}")
    print(f"{'='*60}")

    query_engine = get_query_engine(pipeline_name)
    if query_engine is None:
        return {"error": f"Failed to initialize pipeline '{pipeline_name}'"}

    per_query_results = []
    skipped = 0
    t_start = time.time()

    for idx, row in df.iterrows():
        question = row["question"]
        ground_truth = str(row["ground_truth"])
        rule_id = extract_rule_id(question)

        if rule_id is None:
            skipped += 1
            continue

        try:
            raw_nodes = query_engine.retrieve(QueryBundle(question))
        except Exception as e:
            print(f"  ERROR on question {idx}: {e}")
            skipped += 1
            continue

        nodes = [normalize_node(n) for n in raw_nodes]
        relevance_flags = [
            is_relevant(n["text"], rule_id, ground_truth) for n in nodes
        ]

        hit_rates = {k: compute_hit_rate_at_k(relevance_flags, k) for k in k_values}
        rr = compute_reciprocal_rank(relevance_flags)

        per_query_results.append({
            "question_idx": int(idx),
            "rule_id": rule_id,
            "n_nodes": len(nodes),
            "n_relevant": sum(relevance_flags),
            "first_relevant_rank": next(
                (i + 1 for i, f in enumerate(relevance_flags) if f), None
            ),
            "hit_rates": hit_rates,
            "reciprocal_rank": rr,
        })

        done = len(per_query_results) + skipped
        if done % 100 == 0 or done == len(df):
            elapsed = time.time() - t_start
            print(f"  [{done}/{len(df)}] {elapsed:.0f}s elapsed")

    elapsed_total = time.time() - t_start
    n_evaluated = len(per_query_results)

    if n_evaluated == 0:
        return {"error": "No questions evaluated (all skipped)"}

    avg_hit_rates = {}
    for k in k_values:
        hits = [q["hit_rates"][k] for q in per_query_results]
        avg_hit_rates[f"HR@{k}"] = sum(hits) / n_evaluated

    mrr = sum(q["reciprocal_rank"] for q in per_query_results) / n_evaluated
    avg_nodes = sum(q["n_nodes"] for q in per_query_results) / n_evaluated

    summary = {
        "pipeline": pipeline_name,
        "n_evaluated": n_evaluated,
        "n_skipped": skipped,
        "elapsed_seconds": round(elapsed_total, 1),
        "avg_nodes_per_query": round(avg_nodes, 1),
        **avg_hit_rates,
        "MRR": mrr,
    }

    print(f"\n  Summary for {pipeline_name}:")
    for k in k_values:
        print(f"    HR@{k:>2}: {avg_hit_rates[f'HR@{k}']:.3f}")
    print(f"    MRR:   {mrr:.3f}")
    print(f"    Avg nodes: {avg_nodes:.1f}")

    return {
        "summary": summary,
        "per_query": per_query_results,
    }


def print_comparison_table(all_results: dict, k_values: list[int]):
    sep = "=" * 80
    print(f"\n{sep}")
    print("  RETRIEVAL METRICS COMPARISON")
    print(sep)

    hr_cols = "".join(f"{'HR@'+str(k):>8}" for k in k_values)
    print(f"{'Pipeline':<40}{hr_cols}{'MRR':>8}{'Nodes':>8}")
    print("-" * (40 + 8 * len(k_values) + 16))

    for name, result in all_results.items():
        if "error" in result:
            print(f"{name:<40}  ERROR: {result['error']}")
            continue
        s = result["summary"]
        hr_vals = "".join(f"{s[f'HR@{k}']:>8.3f}" for k in k_values)
        print(f"{name:<40}{hr_vals}{s['MRR']:>8.3f}{s['avg_nodes_per_query']:>8.1f}")

    print(f"{sep}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute Hit Rate@k and MRR retrieval metrics for RAG pipelines"
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=DEFAULT_PIPELINES,
        choices=list(PIPELINE_ALIASES.keys()),
        help=f"Pipeline aliases to evaluate (default: {DEFAULT_PIPELINES})",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of questions to evaluate (default: all 1192)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling (only used with --n)",
    )
    parser.add_argument(
        "--output", "-o",
        default="scripts/retrieval_metrics_results.json",
        help="Path for JSON output",
    )
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df)} questions from {DATASET_PATH}")

    if args.n is not None and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
        print(f"Subsampled to {len(df)} questions (seed={args.seed})")

    pipeline_names = [PIPELINE_ALIASES[alias] for alias in args.pipelines]

    all_results = {}
    for alias, full_name in zip(args.pipelines, pipeline_names):
        result = run_retrieval_evaluation(full_name, df, K_VALUES)
        all_results[full_name] = result

    print_comparison_table(all_results, K_VALUES)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "k_values": K_VALUES,
            "n_questions": len(df),
            "pipelines": pipeline_names,
        },
        "summaries": {
            name: result["summary"]
            for name, result in all_results.items()
            if "summary" in result
        },
        "per_query": {
            name: result["per_query"]
            for name, result in all_results.items()
            if "per_query" in result
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
