"""Run one text benchmark for named pipelines with the same evaluate_benchmark harness as main.py.
Defaults to the full compilation benchmark (30 topics).

Usage:
    uv run python scripts/run_compilation_eval.py structured_graph_rag
    uv run python scripts/run_compilation_eval.py --benchmark compilation pipe_a pipe_b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from src.strategies import PIPELINE_CONFIGS
from src.evaluation.benchmark_evaluation import evaluate_benchmark
from src.utils.rag.query_engine_helpers import get_query_engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipelines", nargs="+", help="pipeline config name(s) from strategies.py")
    parser.add_argument("--benchmark", default="compilation", help="benchmark key (default: compilation)")
    parser.add_argument("--test-run", action="store_true", help="evaluate the small subset instead of the full set")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel query workers (default 1 = serial)")
    parser.add_argument("--rpm", type=float, default=-1.0,
                        help="per-minute request cap (default -1 = automatic schedule, 0 = off)")
    parser.add_argument("--rps", type=float, default=2.0,
                        help="HARD per-second burst ceiling (default 2 = GWDG limit). 0 to disable.")
    parser.add_argument("--max-retries", type=int, default=4,
                        help="bounded exponential-backoff retries per query on transient errors (default 4).")
    parser.add_argument("--on-failure", choices=["exclude", "mark"], default="exclude",
                        help="unrecoverable query -> 'exclude' (NA, dropped from metric) or 'mark' (API-Error).")
    args = parser.parse_args()

    for name in args.pipelines:
        if name not in PIPELINE_CONFIGS:
            print(f"[skip] unknown pipeline '{name}'")
            continue
        print(f"\n{'=' * 60}\nPIPELINE: {name}  |  BENCHMARK: {args.benchmark}\n{'=' * 60}")
        engine = get_query_engine(name)
        if not engine:
            print(f"[skip] could not initialise '{name}'")
            continue
        macro_f1, _ = evaluate_benchmark(
            engine,
            args.benchmark,
            selected_pipeline=name,
            cfg=PIPELINE_CONFIGS[name],
            test_run=args.test_run,
            max_workers=args.workers,
            requests_per_minute=(None if args.rpm < 0 else args.rpm),
            requests_per_second=args.rps,
            max_retries=args.max_retries,
            on_failure=args.on_failure,
        )
        print(f"\n>>> {name}: macro-F1 = {macro_f1:.4f}\n")


if __name__ == "__main__":
    main()
