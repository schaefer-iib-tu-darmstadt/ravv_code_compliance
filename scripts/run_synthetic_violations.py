"""
Run ICR vs. RAVV (GPT-OSS retriever), both with gemma-3-27b-it via OpenRouter, on the synthetic violations.
Inputs come from build_synthetic_violations.py and build_synthetic_functional.py; results are stored
via evaluate_benchmark under results/rule_compliance/ and results_index.csv, like main.py runs.

Usage:
    uv run python scripts/run_synthetic_violations.py --test          # small subset
    uv run python scripts/run_synthetic_violations.py --runs 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation.benchmark_evaluation import evaluate_benchmark
from src.strategies import PIPELINE_CONFIGS
from src.utils.logging_helpers import set_logging_level
from src.utils.rag.query_engine_helpers import get_query_engine

PIPELINES = ["or_gemma3_no_rag_vision", "or_gpt2gemma3_retrieval2vision"]
BENCHMARK_KEYS = ["dimension_synthetic", "functional_synthetic"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--start-run", type=int, default=1)
    ap.add_argument("--test", action="store_true", help="small subset per benchmark")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pipelines", nargs="+", default=PIPELINES)
    ap.add_argument("--benchmarks", nargs="+", default=BENCHMARK_KEYS)
    args = ap.parse_args()
    set_logging_level("INFO")

    for name in args.pipelines:
        engine = get_query_engine(name)
        if not engine:
            sys.exit(f"failed to initialise {name}")
        for key in args.benchmarks:
            for run in range(args.start_run, args.start_run + args.runs):
                print(f"\n=== {name} | {key} | run {run} ===", flush=True)
                evaluate_benchmark(
                    engine, key,
                    selected_pipeline=name,
                    cfg=PIPELINE_CONFIGS[name],
                    test_run=args.test,
                    run_number=run,
                    max_workers=args.workers,
                )


if __name__ == "__main__":
    main()
