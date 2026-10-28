"""
Aggregates the timing.jsonl records of a measure_latency_cost.py run into the latency/token/cost table.
Per (pipeline, benchmark, stage): calls/query, median [p25-p75] seconds with client-side limiter wait
subtracted, median tokens and cost per query; warm-up and failed attempts excluded. Writes CSV/MD/TeX.

Usage:
    uv run python scripts/analyze_latency_cost.py --run results/latency_cost/<label>
    uv run python scripts/analyze_latency_cost.py --run results/latency_cost/<label> --validate
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

PRICE_TABLE: dict[str, tuple | None] = {
    "text-embedding-3-large": (0.13, 0.0, "https://platform.openai.com/docs/pricing", "verify"),
    "gpt-oss": None,
    "e5-mistral": None,
    "mistral-large": None,
    "qwen": None,
}

EXPECTED_SIGNATURE = {
    "gpt-oss_no_rag": (0, 1),
    "gpt-oss_basic_rag_small_chunks": (1, 1),
    "gpt-oss_advanced_hierarchical": (1, 2),
    "structured_graph_rag": (0, 0),
    "or_gpt2gemma3_retrieval2vision": (1, 3),
    "or_gemma3_no_rag_vision": (0, 1),
}

RANKGPT_FINGERPRINTS = ("i will provide you with", "passages, each indicated by")


def load_records(run_dir: Path) -> list[dict]:
    records = []
    files = sorted(run_dir.glob("*/*/timing.jsonl"))
    if not files:
        sys.exit(f"No timing.jsonl found under {run_dir}")
    for path in files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def select_attempts(records: list[dict]) -> tuple[dict, list[dict]]:
    queries = [r for r in records if r["type"] == "query" and not r.get("warmup")]
    selected: dict = {}
    for r in queries:
        key = (r["pipeline"], r["benchmark"], r["qid"])
        if r.get("ok") and (key not in selected or r["attempt"] > selected[key]["attempt"]):
            selected[key] = r
    return selected, queries


def classify_stage(rec: dict, chat_order: dict) -> str:
    if rec.get("stage"):
        return rec["stage"]
    if rec.get("endpoint") == "embedding":
        return "embed"
    head = (rec.get("prompt_head") or "").lower()
    if head:
        return "rerank" if any(fp in head for fp in RANKGPT_FINGERPRINTS) else "synthesis"
    total, pos = chat_order.get(id(rec), (1, 1))
    return "synthesis" if pos == total else "rerank"


def build_frames(records: list[dict]):
    selected, all_queries = select_attempts(records)

    http = [r for r in records if r["type"] == "http" and not r.get("warmup")]
    stage_recs = [r for r in records if r["type"] == "stage" and not r.get("warmup")]

    per_query_chats: dict = defaultdict(list)
    kept_http = []
    for r in http:
        key = (r["pipeline"], r["benchmark"], r["qid"])
        sel = selected.get(key)
        if sel is None or r["attempt"] != sel["attempt"]:
            continue
        kept_http.append(r)
        if r["endpoint"] == "chat" and r["status"] == 200 and not r.get("stage"):
            per_query_chats[key].append(r)
    chat_order: dict = {}
    for key, chats in per_query_chats.items():
        chats.sort(key=lambda r: r["ts"])
        for pos, r in enumerate(chats, start=1):
            chat_order[id(r)] = (len(chats), pos)

    http_rows = []
    for r in kept_http:
        usage = r.get("usage") or {}
        http_rows.append({
            "pipeline": r["pipeline"], "benchmark": r["benchmark"], "qid": r["qid"],
            "stage": classify_stage(r, chat_order),
            "endpoint": r["endpoint"], "status": r["status"], "model": r.get("model"),
            "limiter_wait_s": r.get("limiter_wait_s") or 0.0,
            "http_s": r.get("http_s"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_head": r.get("prompt_head") or "",
        })
    http_df = pd.DataFrame(http_rows)

    query_rows = []
    for (pipeline, benchmark, qid), r in selected.items():
        if http_df.empty:
            waits = 0.0
        else:
            mask = ((http_df["pipeline"] == pipeline) & (http_df["benchmark"] == benchmark)
                    & (http_df["qid"] == qid))
            waits = float(http_df.loc[mask, "limiter_wait_s"].sum())
        query_rows.append({
            "pipeline": pipeline, "benchmark": benchmark, "qid": qid,
            "attempts": r["attempt"], "e2e_s": r["e2e_s"],
            "limiter_wait_s": round(waits, 4),
            "e2e_net_s": round(max(0.0, r["e2e_s"] - waits), 4),
        })
    query_df = pd.DataFrame(query_rows)

    stage_rows = [{
        "pipeline": r["pipeline"], "benchmark": r["benchmark"], "qid": r["qid"],
        "stage": r["stage"], "wall_s": r["wall_s"],
    } for r in stage_recs
        if (r["pipeline"], r["benchmark"], r["qid"]) in selected
        and r["attempt"] == selected[(r["pipeline"], r["benchmark"], r["qid"])]["attempt"]]
    stage_df = pd.DataFrame(stage_rows)

    n_failed = len({(q["pipeline"], q["benchmark"], q["qid"]) for q in all_queries}
                   - set(selected.keys()))
    return query_df, http_df, stage_df, n_failed


def price_for(model: str | None) -> tuple | None:
    if not model:
        return None
    for key, price in PRICE_TABLE.items():
        if key in model:
            return price
    return None


def _q(series, q):
    return round(float(series.quantile(q)), 2) if len(series) else None


def summarize(query_df, http_df, stage_df) -> pd.DataFrame:
    rows = []
    for (pipeline, benchmark), qgrp in query_df.groupby(["pipeline", "benchmark"]):
        n_q = len(qgrp)
        cell_http = http_df[(http_df["pipeline"] == pipeline)
                            & (http_df["benchmark"] == benchmark)] if not http_df.empty else http_df

        if cell_http is not None and not cell_http.empty:
            ok_http = cell_http[cell_http["status"] == 200]
            for stage, sgrp in ok_http.groupby("stage"):
                per_q_http = sgrp.groupby("qid")["http_s"].sum()
                per_q_prompt = sgrp.groupby("qid")["prompt_tokens"].sum(min_count=1)
                per_q_compl = sgrp.groupby("qid")["completion_tokens"].sum(min_count=1)
                rows.append({
                    "pipeline": pipeline, "benchmark": benchmark, "row": f"stage:{stage}",
                    "n_queries": n_q,
                    "calls_per_query": round(len(sgrp) / n_q, 2),
                    "median_s": _q(per_q_http, 0.5),
                    "p25_s": _q(per_q_http, 0.25),
                    "p75_s": _q(per_q_http, 0.75),
                    "prompt_tok_med": _q(per_q_prompt.dropna(), 0.5),
                    "completion_tok_med": _q(per_q_compl.dropna(), 0.5),
                    "cost_usd_med": None,
                    "models": ";".join(sorted(sgrp["model"].dropna().unique())),
                })

        cost_per_q, unpriced = [], set()
        if cell_http is not None and not cell_http.empty:
            ok_http = cell_http[cell_http["status"] == 200]
            for qid, qh in ok_http.groupby("qid"):
                cost = 0.0
                for _, r in qh.iterrows():
                    price = price_for(r["model"])
                    if price is None:
                        if r["model"]:
                            unpriced.add(r["model"])
                        continue
                    p_in, p_out = price[0], price[1]
                    def _tok(v):
                        return 0.0 if v is None or pd.isna(v) else float(v)
                    cost += _tok(r["prompt_tokens"]) / 1e6 * p_in
                    cost += _tok(r["completion_tokens"]) / 1e6 * p_out
                cost_per_q.append(cost)
            per_q_prompt = ok_http.groupby("qid")["prompt_tokens"].sum(min_count=1).dropna()
            per_q_compl = ok_http.groupby("qid")["completion_tokens"].sum(min_count=1).dropna()
            calls_pq = round(len(ok_http) / n_q, 2)
            retried = int((cell_http["status"] != 200).sum())
        else:
            per_q_prompt = per_q_compl = pd.Series(dtype=float)
            calls_pq, retried = 0.0, 0

        cost_series = pd.Series(cost_per_q, dtype=float)
        rows.append({
            "pipeline": pipeline, "benchmark": benchmark, "row": "total",
            "n_queries": n_q,
            "calls_per_query": calls_pq,
            "median_s": _q(qgrp["e2e_net_s"], 0.5),
            "p25_s": _q(qgrp["e2e_net_s"], 0.25),
            "p75_s": _q(qgrp["e2e_net_s"], 0.75),
            "prompt_tok_med": _q(per_q_prompt, 0.5),
            "completion_tok_med": _q(per_q_compl, 0.5),
            "cost_usd_med": round(float(cost_series.median()), 5) if len(cost_series) else None,
            "models": "",
            "limiter_wait_total_s": round(float(qgrp["limiter_wait_s"].sum()), 1),
            "non200_calls": retried,
            "unpriced_models": ";".join(sorted(unpriced)),
        })
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame) -> str:
    def fmt(v):
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in rec) + " |"
              for rec in df.itertuples(index=False)]
    return "\n".join(lines)


def to_markdown(summary: pd.DataFrame, n_failed: int) -> str:
    lines = ["# Latency/Cost Summary", ""]
    lines.append(f"Excluded: warm-up queries and {n_failed} queries with no successful attempt.")
    lines.append("Latency = seconds with limiter wait subtracted; tokens = median per query.")
    lines.append("")
    cols = ["row", "calls_per_query", "median_s", "p25_s", "p75_s",
            "prompt_tok_med", "completion_tok_med", "cost_usd_med"]
    for (pipeline, benchmark), grp in summary.groupby(["pipeline", "benchmark"]):
        lines.append(f"## {pipeline} x {benchmark} (n={int(grp['n_queries'].iloc[0])})")
        lines.append("")
        lines.append(_md_table(grp[cols]))
        total = grp[grp["row"] == "total"].iloc[0]
        extras = []
        if total.get("limiter_wait_total_s"):
            extras.append(f"limiter wait total {total['limiter_wait_total_s']}s (subtracted)")
        if total.get("non200_calls"):
            extras.append(f"{int(total['non200_calls'])} non-200 calls")
        if isinstance(total.get("unpriced_models"), str) and total["unpriced_models"]:
            extras.append(f"unpriced models: {total['unpriced_models']}")
        if extras:
            lines.append("")
            lines.append("_" + "; ".join(extras) + "_")
        lines.append("")
    return "\n".join(lines)


def to_latex(summary: pd.DataFrame) -> str:
    def fmt_s(r):
        if r["median_s"] is None:
            return "--"
        return f"{r['median_s']:.1f} [{r['p25_s']:.1f}--{r['p75_s']:.1f}]"

    def fmt_tok(v):
        return "--" if v is None or pd.isna(v) else f"{v:,.0f}"

    def fmt_cost(v):
        if v is None or pd.isna(v):
            return "--"
        if v == 0:
            return "0"
        return f"{v:.4f}" if v >= 0.0005 else f"{v:.1e}"

    lines = [
        "% Auto-generated by scripts/analyze_latency_cost.py -- do not edit by hand",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Pipeline & Latency (s, median [IQR]) & Calls/q & Prompt tok & Compl.\ tok & Cost/q (USD) \\",
        r"\midrule",
    ]
    for (pipeline, benchmark), grp in summary.groupby(["pipeline", "benchmark"]):
        total = grp[grp["row"] == "total"].iloc[0]
        name = f"{pipeline.replace('_', r'\_')} ({benchmark})"
        lines.append(f"{name} & {fmt_s(total)} & {total['calls_per_query']} & "
                     f"{fmt_tok(total['prompt_tok_med'])} & {fmt_tok(total['completion_tok_med'])} & "
                     f"{fmt_cost(total['cost_usd_med'])} \\\\")
        stages = grp[grp["row"].str.startswith("stage:")]
        for _, s in stages.iterrows():
            sname = s["row"].replace("stage:", "").replace("_", r"\_")
            lines.append(rf"\quad {sname} & {fmt_s(s)} & {s['calls_per_query']} & "
                         f"{fmt_tok(s['prompt_tok_med'])} & {fmt_tok(s['completion_tok_med'])} & -- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def validate(query_df, http_df, stage_df) -> int:
    failures, warnings = [], []

    for (pipeline, benchmark), qgrp in query_df.groupby(["pipeline", "benchmark"]):
        expected = EXPECTED_SIGNATURE.get(pipeline)
        cell_http = http_df[(http_df["pipeline"] == pipeline)
                            & (http_df["benchmark"] == benchmark)] if not http_df.empty else http_df

        if (qgrp["e2e_s"] <= 0).any():
            failures.append(f"{pipeline}/{benchmark}: query with e2e_s <= 0")

        if expected == (0, 0):
            if cell_http is not None and not cell_http.empty:
                failures.append(f"{pipeline}/{benchmark}: expected zero HTTP records, "
                                f"found {len(cell_http)}")
            continue

        if cell_http is None or cell_http.empty:
            failures.append(f"{pipeline}/{benchmark}: no HTTP records at all")
            continue

        ok_http = cell_http[cell_http["status"] == 200]
        for qid in qgrp["qid"]:
            qh = ok_http[ok_http["qid"] == qid]
            if qh.empty:
                failures.append(f"{pipeline}/{benchmark} qid={qid}: ok query without "
                                f"any 200 HTTP record")
                continue
            if expected is not None:
                n_embed = int((qh["endpoint"] == "embedding").sum())
                n_chat = int((qh["endpoint"] == "chat").sum())
                if n_embed != expected[0] or n_chat < expected[1]:
                    warnings.append(f"{pipeline}/{benchmark} qid={qid}: call signature "
                                    f"(embed={n_embed}, chat={n_chat}) outside expected "
                                    f"(embed=={expected[0]}, chat>={expected[1]})")

        chat200 = ok_http[ok_http["endpoint"] == "chat"]
        if len(chat200):
            usage_rate = chat200["prompt_tokens"].notna().mean()
            if usage_rate < 0.95:
                failures.append(f"{pipeline}/{benchmark}: usage present on only "
                                f"{usage_rate:.0%} of 200-status chat records")

        if not stage_df.empty:
            cell_stage = stage_df[(stage_df["pipeline"] == pipeline)
                                  & (stage_df["benchmark"] == benchmark)]
            if not cell_stage.empty:
                unlabeled = cell_http[cell_http["stage"].astype(str).str.startswith(("embed", "rerank", "synthesis"))]
                if len(unlabeled) == len(cell_http):
                    warnings.append(f"{pipeline}/{benchmark}: stage records exist but no "
                                    f"http record carries a stage label")
                merged = cell_stage.groupby("qid")["wall_s"].sum().rename("stage_sum")
                joined = qgrp.set_index("qid").join(merged, how="inner")
                bad = joined[joined["stage_sum"] > joined["e2e_s"] + 0.5]
                for qid, r in bad.iterrows():
                    failures.append(f"{pipeline}/{benchmark} qid={qid}: stage sum "
                                    f"{r['stage_sum']:.1f}s > e2e {r['e2e_s']:.1f}s")

        rerank = ok_http[ok_http["stage"] == "rerank"]
        for _, r in rerank.iterrows():
            head = str(r["prompt_head"]).lower()
            if head and not any(fp in head for fp in RANKGPT_FINGERPRINTS):
                warnings.append(f"{pipeline}/{benchmark} qid={r['qid']}: 'rerank' record "
                                f"without RankGPT fingerprint: {head[:60]!r}")

    print("\n=== VALIDATION ===")
    for w in warnings:
        print(f"[warn] {w}")
    for f in failures:
        print(f"[FAIL] {f}")
    print(f"{len(failures)} failures, {len(warnings)} warnings")
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate latency/cost records")
    parser.add_argument("--run", required=True, help="Run dir with */*/timing.jsonl")
    parser.add_argument("--validate", action="store_true", help="Check smoke invariants")
    parser.add_argument("--price", action="append", default=[],
                        help='Override price: "model_substring=usd_in,usd_out" (per 1M tokens)')
    args = parser.parse_args()

    for spec in args.price:
        key, _, val = spec.partition("=")
        p_in, _, p_out = val.partition(",")
        PRICE_TABLE[key.strip()] = (float(p_in), float(p_out or 0.0), "cli-override", "")

    run_dir = Path(args.run)
    records = load_records(run_dir)
    query_df, http_df, stage_df, n_failed = build_frames(records)
    if query_df.empty:
        sys.exit("No successful (non-warmup) queries found.")

    if args.validate:
        sys.exit(validate(query_df, http_df, stage_df))

    summary = summarize(query_df, http_df, stage_df)
    summary.to_csv(run_dir / "latency_cost_summary.csv", index=False, encoding="utf-8")
    md = to_markdown(summary, n_failed)
    (run_dir / "latency_cost_summary.md").write_text(md, encoding="utf-8")
    tex = to_latex(summary)
    (run_dir / "latency_cost_table.tex").write_text(tex, encoding="utf-8")

    print(md)
    print("\n=== LaTeX snippet ===\n")
    print(tex)
    unpriced = summary[summary["row"] == "total"]["unpriced_models"].dropna()
    unpriced = sorted({m for cell in unpriced if cell for m in cell.split(";")})
    if unpriced:
        print(f"\n[note] unpriced models (cost column incomplete): {', '.join(unpriced)}")
        print("       GWDG models are intentionally unpriced (academic hosting); "
              "for other providers verify the price page and re-run with --price.")
    print(f"\nWritten: {run_dir}\\latency_cost_summary.csv / .md / latency_cost_table.tex")


if __name__ == "__main__":
    main()
