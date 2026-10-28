"""Generic DesignQA subtask evaluation driven by the ``BENCHMARKS`` registry.

Flow: load CSV -> query loop -> drop failed/empty predictions -> score -> persist.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from eval.metrics.metrics import (
    eval_retrieval_qa,
    eval_compilation_qa,
    eval_definition_qa,
    eval_presence_qa,
    eval_dimensions_qa,
    eval_functional_performance_qa,
)
from src.utils.rag.version_config_helpers import make_run_id, build_version_label
from src.utils.rag.eval_common import (
    build_results_paths,
    run_query_loop,
    run_query_loop_with_image,
    evaluate_and_persist,
    write_pretty_json,
)
from src.utils.llm_api.rate_limit import build_gwdg_limiter, set_request_limiter


@dataclass(frozen=True)
class BenchmarkSpec:
    label: str
    dataset_csv: str
    base_dir: str
    eval_fn: Callable
    benchmark: str
    test_run_n: int
    image_base_path: Optional[str] = None
    supports_run_number: bool = False
    score_label: str = "F1"


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "retrieval": BenchmarkSpec(
        label="Retrieval Benchmark",
        dataset_csv="dataset/rule_extraction/rule_retrieval_qa.csv",
        base_dir="results/rule_extraction/rule_retrieval",
        eval_fn=eval_retrieval_qa,
        benchmark="rule_retrieval",
        test_run_n=5,
    ),
    "retrieval_sample": BenchmarkSpec(
        label="Retrieval Benchmark (random subset n=200, seed=42)",
        dataset_csv="data/processed/rule_retrieval_qa_sample200_seed42.csv",
        base_dir="results/rule_extraction/rule_retrieval_sample",
        eval_fn=eval_retrieval_qa,
        benchmark="rule_retrieval_sample",
        test_run_n=5,
    ),
    "compilation": BenchmarkSpec(
        label="Compilation Benchmark",
        dataset_csv="dataset/rule_extraction/rule_compilation_qa.csv",
        base_dir="results/rule_extraction/rule_compilation",
        eval_fn=eval_compilation_qa,
        benchmark="rule_compilation",
        test_run_n=1,
    ),
    "definition": BenchmarkSpec(
        label="definition Benchmark",
        dataset_csv="dataset/rule_comprehension/rule_definition_qa.csv",
        base_dir="results/rule_comprehension/rule_definition",
        eval_fn=eval_definition_qa,
        benchmark="rule_definition",
        test_run_n=1,
        image_base_path="dataset/rule_comprehension/rule_definition_qa",
    ),
    "presence": BenchmarkSpec(
        label="presence Benchmark",
        dataset_csv="dataset/rule_comprehension/rule_presence_qa.csv",
        base_dir="results/rule_comprehension/rule_presence",
        eval_fn=eval_presence_qa,
        benchmark="rule_presence",
        test_run_n=1,
        image_base_path="dataset/rule_comprehension/rule_presence_qa",
    ),
    "dimension": BenchmarkSpec(
        label="dimension Benchmark",
        dataset_csv="dataset/rule_compliance/rule_dimension_qa/detailed_context/rule_dimension_qa_detailed_context.csv",
        base_dir="results/rule_compliance/rule_dimension",
        eval_fn=eval_dimensions_qa,
        benchmark="rule_dimension",
        test_run_n=5,
        image_base_path="dataset/rule_compliance/rule_dimension_qa/detailed_context",
        supports_run_number=True,
        score_label="Accuracy",
    ),
    "dimension_synthetic": BenchmarkSpec(
        label="dimension Benchmark (synthetic violations)",
        dataset_csv="dataset_synthetic/rule_dimension_synthetic/rule_dimension_synthetic.csv",
        base_dir="results/rule_compliance/rule_dimension_synthetic",
        eval_fn=eval_dimensions_qa,
        benchmark="rule_dimension_synthetic",
        test_run_n=4,
        image_base_path="dataset_synthetic/rule_dimension_synthetic/images",
        supports_run_number=True,
        score_label="Accuracy",
    ),
    "functional": BenchmarkSpec(
        label="functional Benchmark",
        dataset_csv="dataset/rule_compliance/rule_functional_performance_qa/rule_functional_performance_qa.csv",
        base_dir="results/rule_compliance/rule_functional",
        eval_fn=eval_functional_performance_qa,
        benchmark="rule_functional",
        test_run_n=5,
        image_base_path="dataset/rule_compliance/rule_functional_performance_qa/images",
        supports_run_number=True,
        score_label="Accuracy",
    ),
    "functional_synthetic": BenchmarkSpec(
        label="functional Benchmark (synthetic violations)",
        dataset_csv="dataset_synthetic/rule_functional_synthetic/rule_functional_synthetic.csv",
        base_dir="results/rule_compliance/rule_functional_synthetic",
        eval_fn=eval_functional_performance_qa,
        benchmark="rule_functional_synthetic",
        test_run_n=4,
        image_base_path="dataset_synthetic/rule_functional_synthetic/images",
        supports_run_number=True,
        score_label="Accuracy",
    ),
}


def evaluate_benchmark(
    query_engine,
    benchmark_key: str,
    *,
    selected_pipeline: str,
    cfg: dict,
    test_run: bool = True,
    run_number: Optional[int] = None,
    max_workers: int = 1,
    requests_per_minute: Optional[float] = None,
    requests_per_second: float = 2.0,
    max_retries: int = 4,
    on_failure: str = "exclude",
):
    if benchmark_key not in BENCHMARKS:
        raise KeyError(
            f"Unknown benchmark '{benchmark_key}'. Available: {', '.join(BENCHMARKS)}"
        )
    spec = BENCHMARKS[benchmark_key]

    benchmark_is_vision = spec.image_base_path is not None
    try:
        engine_params = inspect.signature(query_engine.query).parameters
        expects_image = "image_path" in engine_params
    except (ValueError, TypeError):
        expects_image = None
    if expects_image is not None and expects_image != benchmark_is_vision:
        raise ValueError(
            f"Modality mismatch: benchmark '{benchmark_key}' is "
            f"{'vision' if benchmark_is_vision else 'text-only'}, but pipeline "
            f"'{selected_pipeline}' query() "
            f"{'expects' if expects_image else 'does not accept'} an image_path. "
            f"Refusing to run (it would fail every row)."
        )

    print(f"\n--- Querying the RAG System ({spec.label}) ---")

    run_id = make_run_id(cfg, run_number=run_number, test_run=test_run)
    paths = build_results_paths(
        base_dir=spec.base_dir,
        pipeline=selected_pipeline,
        run_id=run_id,
    )

    df = pd.read_csv(spec.dataset_csv)
    df["model_prediction"] = pd.NA
    n = spec.test_run_n if test_run else len(df)

    def question_builder(row):
        return row["question"]

    disable_rps = requests_per_second is not None and requests_per_second <= 0
    disable_minute = requests_per_minute is not None and requests_per_minute <= 0
    fixed_minute = requests_per_minute if (requests_per_minute is not None and requests_per_minute > 0) else None
    if disable_rps and disable_minute:
        set_request_limiter(None)
        print("[Throughput] request limiter disabled")
    else:
        set_request_limiter(build_gwdg_limiter(
            rps=(0.0 if disable_rps else float(requests_per_second)),
            minute_rpm=(0.0 if disable_minute else fixed_minute),
        ))
        rps_desc = "off" if disable_rps else f"{requests_per_second:.0f}/s"
        if disable_minute:
            minute_desc = "off"
        elif fixed_minute is not None:
            minute_desc = f"{fixed_minute:.0f}/min (fixed)"
        else:
            minute_desc = "60 per min"
        print(f"[Throughput] request limiter: {rps_desc} + {minute_desc} "
              f"(every backend call counted, one shared per-key budget)")
    if max_workers and max_workers > 1:
        print(f"[Throughput] max_workers={max_workers}")

    loop_kwargs = dict(
        df=df,
        n=n,
        query_engine=query_engine,
        question_builder=question_builder,
        collect_nodes=False,
        periodic_save_every=100,
        results_path=paths["csv"],
        max_workers=max_workers,
        max_retries=max_retries,
        on_failure=on_failure,
    )
    if spec.image_base_path is None:
        df, json_records = run_query_loop(**loop_kwargs)
    else:
        df, json_records = run_query_loop_with_image(
            **loop_kwargs, image_base_path=Path(spec.image_base_path)
        )

    print("\n--- Saving Results ---")
    original = len(df)
    df["model_prediction"] = df["model_prediction"].map(
        lambda v: pd.NA if (isinstance(v, str) and v.strip() == "") else v
    )
    df.dropna(subset=["model_prediction"], inplace=True)
    print(f"Filtered DataFrame: removed {original - len(df)} rows.")
    df.to_csv(paths["csv"], index=False, encoding="utf-8")

    macro_avg, all_scores = evaluate_and_persist(
        df=df,
        eval_fn=spec.eval_fn,
        results_path=paths["csv"],
        results_txt_path=paths["txt"],
        score_label=spec.score_label,
        pipeline=selected_pipeline,
        benchmark=spec.benchmark,
        result_dir=paths["dir"],
        version_label=build_version_label(cfg),
    )
    score_lookup = df["eval"].to_dict()
    write_pretty_json(
        json_records,
        paths["pretty"],
        score_lookup=score_lookup,
        overall_score=macro_avg,
        overall_label="macro_f1",
    )
    return macro_avg, all_scores
