"""Run the selected DesignQA benchmarks for the selected pipelines (``uv run python -m main``)."""

from src.strategies import PIPELINE_CONFIGS
from src.evaluation.benchmark_evaluation import BENCHMARKS, evaluate_benchmark

from src.utils.rag.query_engine_helpers import get_query_engine
from src.utils.logging_helpers import set_logging_level

SELECTED_PIPELINES = [
    "gpt-oss_advanced_hierarchical",
]

BENCHMARKS_TO_RUN = ["retrieval"]

N_RUNS = 5
START_RUN = 1
TEST_RUN = True
LOG_LEVEL = "INFO"

MAX_WORKERS = 8


def main():
    set_logging_level(LOG_LEVEL)

    print(f"\n{'=' * 60}")
    print("STARTING EVALUATION")
    print(f"Pipelines: {len(SELECTED_PIPELINES)} | Benchmarks: {', '.join(BENCHMARKS_TO_RUN)}")
    print(f"Mode: {'TEST RUN (subset)' if TEST_RUN else 'FULL RUN (all questions)'}")
    print(f"{'=' * 60}\n")

    for pipeline_name in SELECTED_PIPELINES:
        print(f"\n{'=' * 60}")
        print(f"PIPELINE: {pipeline_name}")
        print(f"{'=' * 60}")

        config = PIPELINE_CONFIGS[pipeline_name]

        query_engine = get_query_engine(pipeline_name)
        if not query_engine:
            print(f"Failed to initialize pipeline '{pipeline_name}'")
            continue

        for benchmark_key in BENCHMARKS_TO_RUN:
            spec = BENCHMARKS[benchmark_key]
            n_runs = N_RUNS if spec.supports_run_number else 1
            start_run = START_RUN if spec.supports_run_number else 1

            for run_num in range(start_run, start_run + n_runs):
                if spec.supports_run_number:
                    print(f"\n{'~' * 40}")
                    print(f"  RUN {run_num}/{start_run + n_runs - 1} — {benchmark_key} — {pipeline_name}")
                    print(f"{'~' * 40}")

                evaluate_benchmark(
                    query_engine,
                    benchmark_key,
                    selected_pipeline=pipeline_name,
                    cfg=config,
                    test_run=TEST_RUN,
                    run_number=run_num if spec.supports_run_number else None,
                    max_workers=MAX_WORKERS,
                )

        print(f"\nCompleted benchmarks {BENCHMARKS_TO_RUN} for '{pipeline_name}'")


if __name__ == "__main__":
    main()
