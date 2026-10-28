"""Per-call latency and token measurement (RAG_TIMING=1), one query at a time so limiter waits stay near 0.

Bypasses main.py (no metrics, published result dirs untouched); retrieval cells use a seed-42 sample,
others their full sets. Writes results/latency_cost/<label>/; aggregate with analyze_latency_cost.py.

Usage:
    uv run python scripts/measure_latency_cost.py           # text (GWDG) + vision (OpenRouter) cells
    uv run python scripts/measure_latency_cost.py --smoke   # cheap end-to-end check
"""

import os

os.environ["RAG_TIMING"] = "1"
os.environ.setdefault("GWDG_LLM_TIMEOUT", "180")

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.evaluation.benchmark_evaluation import BENCHMARKS
from src.utils.rag.query_engine_helpers import get_query_engine
from src.utils.rag.eval_common import _is_retryable
from src.utils.llm_api import timing_log

RETRIEVAL_SAMPLE_KEY = "retrieval_sample"

TEXT_MATRIX = [
    ("gpt-oss_no_rag", ["retrieval"]),
    ("gpt-oss_basic_rag_small_chunks", ["retrieval"]),
    ("gpt-oss_advanced_hierarchical", ["retrieval"]),
    ("structured_graph_rag", ["compilation"]),
]
TEXT_SMOKE_MATRIX = [
    ("structured_graph_rag", [("compilation", 2)]),
    ("gpt-oss_basic_rag_small_chunks", [("retrieval", 2)]),
]

VISION_MATRIX = [
    ("or_gpt2gemma3_retrieval2vision", ["dimension", "functional"]),
    ("or_gemma3_no_rag_vision", ["dimension", "functional"]),
]
VISION_SMOKE_MATRIX = [
    ("or_gpt2gemma3_retrieval2vision", [("functional", 1)]),
    ("or_gemma3_no_rag_vision", [("functional", 1)]),
]

MATRICES = {
    "text": (TEXT_MATRIX, TEXT_SMOKE_MATRIX),
    "vision": (VISION_MATRIX, VISION_SMOKE_MATRIX),
    "all": (TEXT_MATRIX + VISION_MATRIX, TEXT_SMOKE_MATRIX + VISION_SMOKE_MATRIX),
}


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _sample_rows(benchmark_key: str, n_retrieval: int, seed: int, limit: int | None):
    if benchmark_key == "retrieval":
        spec = BENCHMARKS[RETRIEVAL_SAMPLE_KEY]
        df = pd.read_csv(spec.dataset_csv)
        df = df.sample(n=min(n_retrieval, len(df)), random_state=seed).sort_index()
    else:
        spec = BENCHMARKS[benchmark_key]
        df = pd.read_csv(spec.dataset_csv)
    if limit is not None:
        df = df.head(limit)
    return spec, list(df.iterrows())


def _run_one(engine, qid, question, image_path, *, warmup: bool, max_attempts: int):
    for attempt in range(1, max_attempts + 1):
        with timing_log.query_context(qid, attempt=attempt, warmup=warmup):
            t0 = time.perf_counter()
            try:
                if image_path is not None:
                    response = engine.query(question, image_path)
                else:
                    response = engine.query(question)
                e2e = time.perf_counter() - t0
                timing_log.log_query(qid, e2e, ok=True, attempt=attempt, warmup=warmup)
                return {"qid": qid, "warmup": warmup, "ok": True, "attempts": attempt,
                        "e2e_s": round(e2e, 4), "answer_len": len(str(response))}
            except Exception as exc:
                e2e = time.perf_counter() - t0
                timing_log.log_query(qid, e2e, ok=False, attempt=attempt, warmup=warmup)
                print(f"[measure] qid={qid} attempt {attempt}/{max_attempts} failed "
                      f"after {e2e:.1f}s: {type(exc).__name__}: {exc}")
                if attempt < max_attempts and _is_retryable(exc):
                    time.sleep(3.0)
                    continue
                return {"qid": qid, "warmup": warmup, "ok": False, "attempts": attempt,
                        "e2e_s": round(e2e, 4), "answer_len": 0}


def run_cell(engine, pipeline: str, benchmark_key: str, spec, rows, out_dir: Path,
             max_attempts: int) -> dict:
    cell_dir = out_dir / pipeline / benchmark_key
    cell_dir.mkdir(parents=True, exist_ok=True)
    timing_log.set_run_meta(pipeline=pipeline, benchmark=benchmark_key, run_label=out_dir.name)
    timing_log.set_output(cell_dir / "timing.jsonl")

    is_vision = spec.image_base_path is not None

    def _image(row):
        return str(Path(spec.image_base_path) / row["image"]) if is_vision else None

    t_cell = time.perf_counter()
    summaries = []

    first_i, first_row = rows[0]
    print(f"\n=== {pipeline} x {benchmark_key}: warm-up (qid={first_i}) ===")
    summaries.append(_run_one(engine, first_i, first_row["question"], _image(first_row),
                              warmup=True, max_attempts=max_attempts))

    for pos, (i, row) in enumerate(rows, start=1):
        print(f"\n=== {pipeline} x {benchmark_key}: {pos}/{len(rows)} (qid={i}) ===")
        summaries.append(_run_one(engine, i, row["question"], _image(row),
                                  warmup=False, max_attempts=max_attempts))

    pd.DataFrame(summaries).to_csv(cell_dir / "queries.csv", index=False, encoding="utf-8")
    measured = [s for s in summaries if not s["warmup"]]
    n_ok = sum(1 for s in measured if s["ok"])
    elapsed = time.perf_counter() - t_cell
    print(f"\n[measure] cell {pipeline} x {benchmark_key}: {n_ok}/{len(measured)} ok, "
          f"{elapsed / 60:.1f} min")
    return {"pipeline": pipeline, "benchmark": benchmark_key,
            "n_queries": len(measured), "n_ok": n_ok,
            "question_indices": [int(i) for i, _ in rows],
            "cell_minutes": round(elapsed / 60, 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency/token measurement runner")
    parser.add_argument("--pipelines", nargs="*", default=None,
                        help="Subset of matrix pipelines to run (default: all)")
    parser.add_argument("--n-retrieval", type=int, default=50,
                        help="Sample size for retrieval cells (default 50)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None,
                        help="Run dir (default results/latency_cost/run_<date>)")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny instrumentation smoke test (n=2/2/1)")
    parser.add_argument("--matrix", choices=sorted(MATRICES), default="all",
                        help="text = GWDG text cells, vision = OpenRouter vision cells, all = both")
    args = parser.parse_args()

    label = Path(args.out).name if args.out else (
        f"smoke_{datetime.now():%Y-%m-%d_%H%M}" if args.smoke
        else f"run_{datetime.now():%Y-%m-%d}")
    out_dir = Path(args.out) if args.out else Path("results/latency_cost") / label
    out_dir.mkdir(parents=True, exist_ok=True)

    full, smoke = MATRICES[args.matrix]
    if args.smoke:
        matrix = [(p, benches) for p, benches in smoke]
    else:
        matrix = [(p, [(b, None) for b in benches]) for p, benches in full]
    if args.pipelines:
        matrix = [(p, benches) for p, benches in matrix if p in set(args.pipelines)]
        if not matrix:
            sys.exit(f"No matrix pipelines match {args.pipelines}")

    manifest = {
        "label": label,
        "git_commit": _git_commit(),
        "args": vars(args),
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "env": {k: os.environ.get(k) for k in (
            "GWDG_LLM_TIMEOUT", "GWDG_RATE_TZ",
            "QDRANT_LOCAL_PATH", "OPENROUTER_LLM_TIMEOUT") if os.environ.get(k)},
        "note": ("Sequential (1 in-flight query); limiter waits recorded separately "
                 "and subtracted in analysis. Model availability per GWDG catalog "
                 "at run date."),
        "cells": [],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for pipeline, benches in matrix:
        print(f"\n{'#' * 70}\n# Pipeline: {pipeline}\n{'#' * 70}")
        timing_log.set_run_meta(pipeline=pipeline, benchmark="setup", run_label=label)
        timing_log.set_output(out_dir / pipeline / "setup.jsonl")
        engine = get_query_engine(pipeline)
        if engine is None:
            sys.exit(f"Unknown pipeline config: {pipeline}")
        for bench_key, limit in benches:
            spec, rows = _sample_rows(bench_key, args.n_retrieval, args.seed, limit)
            manifest["cells"].append(
                run_cell(engine, pipeline, bench_key, spec, rows, out_dir, args.max_attempts))
            manifest["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
            (out_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n[measure] done -> {out_dir}")
    print(f"Next: uv run python scripts/analyze_latency_cost.py --run {out_dir}")


if __name__ == "__main__":
    main()
