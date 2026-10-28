"""Shared benchmark query loop, scoring and result persistence.

Transient API errors are retried with backoff; rows that still fail are written as NA
(excluded from the metric, listed in ``failures.json``) instead of being scored as 0.
"""
from __future__ import annotations
from pathlib import Path
import os, json, csv, random, time, openai, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Dict, Any, Optional, Tuple
from src.utils.rag.eval_utils import normalize_node
from src.utils.llm_api.rate_limit import RateLimiter

API_ERROR_MARKER = "API-Error"

_RETRYABLE_EXC = tuple(
    t for t in (
        getattr(openai, "RateLimitError", None),
        getattr(openai, "APITimeoutError", None),
        getattr(openai, "APIConnectionError", None),
        getattr(openai, "InternalServerError", None),
        getattr(openai, "APIError", None),
    )
    if isinstance(t, type)
)

_RETRYABLE_HINTS = (
    "rate limit", "ratelimit", "429", "timeout", "timed out", "502", "503",
    "504", "overloaded", "temporarily unavailable", "connection", "reset by peer",
    "remoteprotocolerror", "too many requests", "reasoning block",
)


def _is_retryable(exc: BaseException) -> bool:
    if _RETRYABLE_EXC and isinstance(exc, _RETRYABLE_EXC):
        return True
    msg = str(exc).lower()
    return any(hint in msg for hint in _RETRYABLE_HINTS)


def _query_with_retry(
    do_query: Callable[[], Any],
    *,
    collect_nodes: bool,
    max_retries: int,
    base_delay: float,
) -> Tuple[Optional[str], List[Dict[str, Any]], bool]:
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            response = do_query()
            prediction_text = str(response)
            nodes: List[Dict[str, Any]] = []
            if collect_nodes:
                try:
                    nodes = [normalize_node(n) for n in getattr(response, "source_nodes", [])]
                except Exception:
                    nodes = []
            return prediction_text, nodes, True
        except Exception as exc:
            last_exc = exc
            retryable = _is_retryable(exc)
            if attempt < max_retries and retryable:
                delay = base_delay * (2 ** attempt) + random.uniform(0.0, base_delay)
                print(f"[Retry {attempt + 1}/{max_retries}] transient error, backing off {delay:.1f}s: {exc}")
                time.sleep(delay)
                continue
            if attempt < min(1, max_retries) and not retryable:
                time.sleep(base_delay)
                continue
            break
    print(f"[API-Error] giving up after {max_retries} retries: {type(last_exc).__name__}: {last_exc}")
    return None, [], False

def build_results_paths(base_dir: str, pipeline: str, run_id: str, version: str = "", prefix: str = "") -> Dict[str, str]:
    run_dir = os.path.join(base_dir, pipeline, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return {
        "dir": run_dir,
        "csv": os.path.join(run_dir, "results.csv"),
        "txt": os.path.join(run_dir, "summary.txt"),
        "pretty": os.path.join(run_dir, "results_pretty.json"),
    }


def append_to_results_index(
    benchmark: str,
    pipeline: str,
    version_label: str,
    run_id: str,
    n_questions: int,
    macro_f1: float,
    bleu2: float = None,
    rouge_l: float = None,
    result_dir: str = "",
    index_path: str = "results/results_index.csv",
):
    import time as _time
    cols = ["timestamp", "benchmark", "pipeline", "version_label", "run_id",
            "n_questions", "macro_f1", "bleu2", "rouge_l", "result_dir"]
    row = {
        "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark": benchmark,
        "pipeline": pipeline,
        "version_label": version_label,
        "run_id": run_id,
        "n_questions": n_questions,
        "macro_f1": round(macro_f1, 6) if macro_f1 is not None else "",
        "bleu2": round(bleu2, 6) if bleu2 is not None else "",
        "rouge_l": round(rouge_l, 6) if rouge_l is not None else "",
        "result_dir": result_dir,
    }
    os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else ".", exist_ok=True)
    file_exists = os.path.exists(index_path)
    if file_exists:
        with open(index_path, "r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header is None:
            file_exists = False
        elif header != cols:
            raise ValueError(
                f"results index {index_path} has header {header}, expected {cols}; "
                "refusing to append a misaligned row."
            )
    with open(index_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not file_exists:
            w.writeheader()
        w.writerow(row)
    print(f"[Index] Row appended to {index_path}.")

def safe_query(
    query_engine,
    question: str,
    collect_nodes: bool = True,
    *,
    max_retries: int = 4,
    base_delay: float = 2.0,
) -> Tuple[Optional[str], List[Dict[str, Any]], bool]:
    return _query_with_retry(
        lambda: query_engine.query(question),
        collect_nodes=collect_nodes,
        max_retries=max_retries,
        base_delay=base_delay,
    )

def safe_query_with_image(
    query_engine,
    question: str,
    image_path: str,
    collect_nodes: bool = True,
    *,
    max_retries: int = 4,
    base_delay: float = 2.0,
) -> Tuple[Optional[str], List[Dict[str, Any]], bool]:
    return _query_with_retry(
        lambda: query_engine.query(question, image_path),
        collect_nodes=collect_nodes,
        max_retries=max_retries,
        base_delay=base_delay,
    )

def _atomic_to_csv(df: pd.DataFrame, path: str) -> None:
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


def _write_failures_sidecar(results_path: str, failures: List[Dict[str, Any]]) -> str:
    side = os.path.join(os.path.dirname(results_path) or ".", "failures.json")
    with open(side, "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)
    return side


def _finalize_failures(
    df: pd.DataFrame,
    items: List[Tuple[Any, pd.Series]],
    *,
    ok_by_idx: Dict[Any, bool],
    asked_by_idx: Dict[Any, str],
    gt_by_idx: Dict[Any, Any],
    nodes_by_idx: Dict[Any, List[Dict[str, Any]]],
    collect_nodes: bool,
    on_failure: str,
    results_path: str,
    total: int,
) -> List[Dict[str, Any]]:
    still_failed = [i for i, _ in items if not ok_by_idx.get(i, False)]
    failures: List[Dict[str, Any]] = []
    for i in still_failed:
        df.loc[i, 'model_prediction'] = API_ERROR_MARKER if on_failure == "mark" else pd.NA
        failures.append({
            "index": int(i),
            "asked_question": asked_by_idx.get(i),
            "ground_truth": str(gt_by_idx.get(i)),
        })

    json_records: List[Dict[str, Any]] = []
    for i, _ in items:
        ok = ok_by_idx.get(i, False)
        json_records.append({
            "index": int(i),
            "asked_question": asked_by_idx.get(i),
            "ground_truth": gt_by_idx.get(i),
            "model_prediction": (df.loc[i, 'model_prediction'] if ok else None),
            "retrieved_nodes": nodes_by_idx.get(i, []) if collect_nodes else [],
        })

    if still_failed:
        side = _write_failures_sidecar(results_path, failures)
        rate = len(still_failed) / max(1, total)
        policy = "excluded from metric (NA)" if on_failure != "mark" else f"kept as '{API_ERROR_MARKER}'"
        print(f"\n[WARNING] {len(still_failed)}/{total} queries failed unrecoverably "
              f"({rate:.1%}); logged to {side}; {policy}.")
        if rate > 0.05:
            print(f"[CRITICAL] failure rate {rate:.1%} > 5% -- treat this run's metric as unreliable.")

    return json_records


def run_query_loop(
    df: pd.DataFrame,
    n: int,
    query_engine,
    question_builder: Callable[[pd.Series], str],
    collect_nodes: bool,
    periodic_save_every: int,
    results_path: str,
    *,
    max_workers: int = 1,
    limiter: Optional[RateLimiter] = None,
    max_retries: int = 4,
    cleanup_failures: bool = True,
    on_failure: str = "exclude",
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    subset = df.head(n)
    items: List[Tuple[Any, pd.Series]] = list(subset.iterrows())
    total = len(items)

    asked_by_idx: Dict[Any, str] = {}
    nodes_by_idx: Dict[Any, List[Dict[str, Any]]] = {}
    ok_by_idx: Dict[Any, bool] = {}
    gt_by_idx: Dict[Any, Any] = {i: row.get('ground_truth') for i, row in items}

    def _work(i, row):
        asked = question_builder(row)
        if limiter is not None:
            limiter.acquire()
        pred, nodes, ok = safe_query(query_engine, asked, collect_nodes=collect_nodes, max_retries=max_retries)
        return i, asked, pred, nodes, ok

    def _consume(i, asked, pred, nodes, ok, done):
        asked_by_idx[i] = asked
        nodes_by_idx[i] = nodes if collect_nodes else []
        ok_by_idx[i] = ok
        df.loc[i, 'asked_question'] = asked
        df.loc[i, 'model_prediction'] = pred if ok else pd.NA
        if collect_nodes:
            df.loc[i, 'retrieved_nodes'] = json.dumps(nodes_by_idx[i], ensure_ascii=False)
        print(f"\n--- Completed {done}/{total} (Index: {i}{'' if ok else ', FAILED'}) ---")
        print(f"Asked: {asked}")
        print(f"Ground Truth: {gt_by_idx.get(i)}")
        print("Response:")
        print(pred if ok else "[unrecoverable error]")
        if periodic_save_every and total > periodic_save_every and done % periodic_save_every == 0:
            print(f"[Safety] Intermediate save after {done} entries.")
            _atomic_to_csv(df, results_path)

    if max_workers and max_workers > 1:
        ceiling = f", rate ceiling {limiter.max_per_window:.0f}/{limiter.window:.0f}s" if limiter else ""
        print(f"[Parallel] {total} queries, {max_workers} workers{ceiling}")
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_work, i, row) for i, row in items]
            for fut in as_completed(futures):
                i, asked, pred, nodes, ok = fut.result()
                done += 1
                _consume(i, asked, pred, nodes, ok, done)
    else:
        for done, (i, row) in enumerate(items, start=1):
            _, asked, pred, nodes, ok = _work(i, row)
            _consume(i, asked, pred, nodes, ok, done)

    failed_idx = [i for i, _ in items if not ok_by_idx.get(i, False)]
    if cleanup_failures and failed_idx:
        print(f"\n[Cleanup] Retrying {len(failed_idx)} failed queries serially (extra backoff)...")
        for i in failed_idx:
            asked = asked_by_idx[i]
            if limiter is not None:
                limiter.acquire()
            pred, nodes, ok = safe_query(query_engine, asked, collect_nodes=collect_nodes, max_retries=max_retries + 2)
            if ok:
                ok_by_idx[i] = True
                nodes_by_idx[i] = nodes if collect_nodes else []
                df.loc[i, 'model_prediction'] = pred
                if collect_nodes:
                    df.loc[i, 'retrieved_nodes'] = json.dumps(nodes_by_idx[i], ensure_ascii=False)
                print(f"[Cleanup] index {i} recovered.")

    json_records = _finalize_failures(
        df, items,
        ok_by_idx=ok_by_idx, asked_by_idx=asked_by_idx, gt_by_idx=gt_by_idx,
        nodes_by_idx=nodes_by_idx, collect_nodes=collect_nodes,
        on_failure=on_failure, results_path=results_path, total=total,
    )
    _atomic_to_csv(df, results_path)
    return df, json_records

def run_query_loop_with_image(
    df: pd.DataFrame,
    n: int,
    query_engine,
    question_builder: Callable[[pd.Series], str],
    collect_nodes: bool,
    periodic_save_every: int,
    results_path: str,
    image_base_path: Path,
    *,
    max_workers: int = 1,
    limiter: Optional[RateLimiter] = None,
    max_retries: int = 4,
    cleanup_failures: bool = True,
    on_failure: str = "exclude",
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    subset = df.head(n)
    items: List[Tuple[Any, pd.Series]] = list(subset.iterrows())
    total = len(items)

    asked_by_idx: Dict[Any, str] = {}
    nodes_by_idx: Dict[Any, List[Dict[str, Any]]] = {}
    ok_by_idx: Dict[Any, bool] = {}
    gt_by_idx: Dict[Any, Any] = {i: row.get('ground_truth') for i, row in items}

    def _work(i, row):
        asked = question_builder(row)
        image_path = str(Path(image_base_path) / row.get('image'))
        if limiter is not None:
            limiter.acquire()
        pred, nodes, ok = safe_query_with_image(
            query_engine, asked, image_path=image_path,
            collect_nodes=collect_nodes, max_retries=max_retries,
        )
        return i, asked, pred, nodes, ok

    def _consume(i, asked, pred, nodes, ok, done):
        asked_by_idx[i] = asked
        nodes_by_idx[i] = nodes if collect_nodes else []
        ok_by_idx[i] = ok
        df.loc[i, 'asked_question'] = asked
        df.loc[i, 'model_prediction'] = pred if ok else pd.NA
        if collect_nodes:
            df.loc[i, 'retrieved_nodes'] = json.dumps(nodes_by_idx[i], ensure_ascii=False)
        print(f"\n--- Completed {done}/{total} (Index: {i}{'' if ok else ', FAILED'}) ---")
        print(f"Asked: {asked}")
        print(f"Ground Truth: {gt_by_idx.get(i)}")
        print("Response:")
        print(pred if ok else "[unrecoverable error]")
        if periodic_save_every and total > periodic_save_every and done % periodic_save_every == 0:
            print(f"[Safety] Intermediate save after {done} entries.")
            _atomic_to_csv(df, results_path)

    if max_workers and max_workers > 1:
        ceiling = f", rate ceiling {limiter.max_per_window:.0f}/{limiter.window:.0f}s" if limiter else ""
        print(f"[Parallel] {total} image queries, {max_workers} workers{ceiling}")
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_work, i, row) for i, row in items]
            for fut in as_completed(futures):
                i, asked, pred, nodes, ok = fut.result()
                done += 1
                _consume(i, asked, pred, nodes, ok, done)
    else:
        for done, (i, row) in enumerate(items, start=1):
            _, asked, pred, nodes, ok = _work(i, row)
            _consume(i, asked, pred, nodes, ok, done)

    failed_idx = [i for i, _ in items if not ok_by_idx.get(i, False)]
    if cleanup_failures and failed_idx:
        print(f"\n[Cleanup] Retrying {len(failed_idx)} failed image queries serially (extra backoff)...")
        for i in failed_idx:
            row = subset.loc[i]
            image_path = str(Path(image_base_path) / row.get('image'))
            if limiter is not None:
                limiter.acquire()
            pred, nodes, ok = safe_query_with_image(
                query_engine, asked_by_idx[i], image_path=image_path,
                collect_nodes=collect_nodes, max_retries=max_retries + 2,
            )
            if ok:
                ok_by_idx[i] = True
                nodes_by_idx[i] = nodes if collect_nodes else []
                df.loc[i, 'model_prediction'] = pred
                if collect_nodes:
                    df.loc[i, 'retrieved_nodes'] = json.dumps(nodes_by_idx[i], ensure_ascii=False)
                print(f"[Cleanup] index {i} recovered.")

    json_records = _finalize_failures(
        df, items,
        ok_by_idx=ok_by_idx, asked_by_idx=asked_by_idx, gt_by_idx=gt_by_idx,
        nodes_by_idx=nodes_by_idx, collect_nodes=collect_nodes,
        on_failure=on_failure, results_path=results_path, total=total,
    )
    _atomic_to_csv(df, results_path)
    return df, json_records

def write_pretty_json(
    json_records: List[Dict[str, Any]],
    path: str,
    preview_len: int = 220,
    score_lookup: Dict[int, float] | None = None,
    overall_score: float | None = None,
    overall_label: str = "macro_f1"
):
    def _p(t: str, n: int = preview_len) -> str:
        return t if isinstance(t, str) and len(t) <= n else (t[:n] + "…") if isinstance(t, str) else ""

    structured: List[Dict[str, Any]] = []
    if overall_score is not None:
        structured.append({
            "overall_eval": {
                "label": overall_label,
                "score": overall_score,
                "items": len(json_records)
            }
        })

    for rec in json_records:
        nodes = rec.get("retrieved_nodes", []) or []
        idx = rec.get("index")
        eval_score = score_lookup.get(idx) if (score_lookup and idx in score_lookup) else None
        structured.append({
            "question": {
                "index": idx,
                "asked_question": rec.get("asked_question"),
            },
            "answer": {
                "ground_truth": rec.get("ground_truth"),
                "model_prediction": rec.get("model_prediction"),
                "eval_score": eval_score,
            },
            "retrieval": {
                "top_k": len(nodes),
                "nodes": [
                    {
                        "rank": j + 1,
                        "score": nd.get("score"),
                        "id": nd.get("id"),
                        "header_path": nd.get("header_path"),
                        "text_preview": _p(nd.get("text")),
                    } for j, nd in enumerate(nodes)
                ]
            }
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)
    print(f"Pretty JSON saved to: {path}")

def evaluate_and_persist(
    df: pd.DataFrame,
    eval_fn,
    results_path: str,
    results_txt_path: str,
    score_label: str = "F1",
    pipeline: str = "",
    benchmark: str = "",
    result_dir: str = "",
    version_label: str | None = None,
):
    print("\n--- Evaluating Results ---")

    result = eval_fn(results_path)

    if not isinstance(result, tuple):
        macro_avg = result
        all_scores = []
    else:
        num_values = len(result)

        macro_avg = result[0]

        if num_values == 2:
            all_scores = result[1]

        elif num_values == 5:
            _, def_avg, mentioned_avg, no_mention_avg, all_scores = result
            print("Subcategory Scores:")
            print(f"  Definition Avg: {def_avg}")
            print(f"  Mentioned Avg: {mentioned_avg}")
            print(f"  No Mention Avg: {no_mention_avg}")

        elif num_values == 6:
            _, all_scores, bleu_avg, all_bleus, rouge_avg, all_rogues = result
            print("Additional Metrics:")
            print(f"  BLEU-2 Avg: {bleu_avg}")
            print(f"  ROUGE-L Avg: {rouge_avg}")

        elif num_values == 8:
            _, direct_dim_avg, scale_bar_avg, all_scores, bleu_avg, all_bleus, rouge_avg, all_rogues = result
            print("Subcategory Scores:")
            print(f"  Direct Dimension Avg: {direct_dim_avg}")
            print(f"  Scale Bar Avg: {scale_bar_avg}")
            print("Additional Metrics:")
            print(f"  BLEU-2 Avg: {bleu_avg}")
            print(f"  ROUGE-L Avg: {rouge_avg}")

        else:
            all_scores = result[1] if num_values > 1 else []
            print(f"Unexpected return format: {num_values} values")

    df.loc[:, 'eval'] = all_scores

    if isinstance(result, tuple):
        if len(result) == 6:
            df.loc[:, 'bleu_2'] = all_bleus
            df.loc[:, 'rouge_l'] = all_rogues
        elif len(result) == 8:
            bleu_col = [None] * len(df)
            rouge_col = [None] * len(df)
            bleu_idx = 0
            for pos, (_row_i, row) in enumerate(df.iterrows()):
                if row.get('dimension_type') == 'direct':
                    if bleu_idx < len(all_bleus):
                        bleu_col[pos] = all_bleus[bleu_idx]
                        rouge_col[pos] = all_rogues[bleu_idx]
                        bleu_idx += 1
            df.loc[:, 'bleu_2'] = bleu_col
            df.loc[:, 'rouge_l'] = rouge_col

    df.to_csv(results_path, index=False, encoding="utf-8")

    bleu2_val = None
    rouge_l_val = None
    if isinstance(result, tuple):
        if len(result) == 6:
            bleu2_val = result[2]
            rouge_l_val = result[4]
        elif len(result) == 8:
            bleu2_val = result[4]
            rouge_l_val = result[6]

    import time as _time
    run_id_from_dir = os.path.basename(result_dir) if result_dir else ""
    if version_label is None:
        version_label = run_id_from_dir.rsplit("-", 1)[0] if "-" in run_id_from_dir else run_id_from_dir

    with open(results_txt_path, "w", encoding="utf-8") as f:
        if pipeline:
            f.write(f"Pipeline:    {pipeline}\n")
        if run_id_from_dir:
            f.write(f"Version:     {run_id_from_dir}\n")
        if benchmark:
            f.write(f"Benchmark:   {benchmark}\n")
        f.write(f"Timestamp:   {_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"N Questions: {len(df)}\n")
        f.write(f"Macro {score_label}: {macro_avg}\n")
        if bleu2_val is not None:
            f.write(f"Macro BLEU-2: {bleu2_val}\n")
        if rouge_l_val is not None:
            f.write(f"Macro ROUGE-L: {rouge_l_val}\n")
        f.write(f"\nAll Individual {score_label} Scores: {all_scores}\n")

    print(f"\nMacro Average {score_label} Score: {macro_avg}")
    print(f"All Individual {score_label} Scores: {all_scores}")

    if benchmark and pipeline:
        append_to_results_index(
            benchmark=benchmark,
            pipeline=pipeline,
            version_label=version_label,
            run_id=run_id_from_dir,
            n_questions=len(df),
            macro_f1=macro_avg,
            bleu2=bleu2_val,
            rouge_l=rouge_l_val,
            result_dir=result_dir,
        )

    return macro_avg, all_scores