"""Abstention rate and answered-usable / abstained / answered-wrong split (F1-BoW vs. tau) on rule_retrieval.

Reads the curated top-level CSVs in results_for_publication/rule_extraction/{retrieval,compilation}/;
whether abstentions are correct rejections is judged only via HR@k (retrieval_metrics.json), since
per-question retrieved context was not stored.

Usage:
    uv run python scripts/analyze_abstention.py --output results/abstention_analysis.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.analysis_helpers import load_result_csv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_basic_rag_failure import REFUSAL_RE

STRICT_TEMPLATE = "does not contain the answer to this question"

TAU_MAIN = 0.3
TAU_SENSITIVITY = (0.2, 0.3, 0.5)

FAMILY_ORDER = ["no_rag", "basic_rag_txt", "basic_rag_md", "advanced_rag", "ablation"]
LLM_ORDER = ["gemma", "mistral", "gpt-oss", "qwen3", "gpt-5"]


def is_strict_abstention(prediction) -> bool:
    return isinstance(prediction, str) and STRICT_TEMPLATE in prediction.lower()


def is_broad_refusal(prediction) -> bool:
    return isinstance(prediction, str) and bool(REFUSAL_RE.search(prediction))


def sort_key(stem: str) -> tuple[int, int, str]:
    family = next((i for i, f in enumerate(FAMILY_ORDER) if f in stem), len(FAMILY_ORDER))
    llm = next((i for i, m in enumerate(LLM_ORDER) if stem.startswith(m)), len(LLM_ORDER))
    return (family, llm, stem)


def _scores_from_eval_txt(path: Path, n: int) -> pd.Series | None:
    eval_txt = path.with_name(path.stem + "_eval.txt")
    if not eval_txt.exists():
        return None
    m = re.search(r"All Individual F1 Scores: \[(.*?)\]",
                  eval_txt.read_text(encoding="utf-8"), re.S)
    if not m:
        return None
    values = [float(x) for x in m.group(1).split(",")]
    if len(values) != n:
        return None
    return pd.Series(values)


def analyze_file(path: Path) -> dict:
    df = load_result_csv(path)
    if df is None:
        return {}

    strict = df["model_prediction"].apply(is_strict_abstention)
    broad = df["model_prediction"].apply(is_broad_refusal)
    n = len(df)

    result = {
        "n": n,
        "n_abstain_strict": int(strict.sum()),
        "abstention_rate_strict": float(strict.mean()),
        "n_abstain_broad": int(broad.sum()),
        "abstention_rate_broad": float(broad.mean()),
        "has_eval": "eval" in df.columns,
    }

    if result["has_eval"]:
        scores = df["eval"]
    else:
        scores = _scores_from_eval_txt(path, n)
        result["eval_source"] = "eval_txt" if scores is not None else None
        if scores is None:
            result["three_way_split"] = None
            return result
        scores.index = df.index
    result["mean_f1_abstain"] = float(scores[strict].mean()) if strict.any() else None
    result["mean_f1_answered"] = float(scores[~strict].mean()) if (~strict).any() else None
    unscored = (~strict) & scores.isna()
    result["n_answered_unscored"] = int(unscored.sum())

    result["three_way_split"] = {}
    for tau in TAU_SENSITIVITY:
        answered_ok = (~strict) & (scores >= tau)
        answered_wrong = (~strict) & (scores < tau)
        result["three_way_split"][f"tau_{tau}"] = {
            "answered_usable": int(answered_ok.sum()),
            "abstained": int(strict.sum()),
            "answered_wrong": int(answered_wrong.sum()),
            "answered_wrong_rate": float(answered_wrong.mean()),
        }
    return result


def compilation_note(comp_dir: Path) -> dict:
    note = {}
    for path in sorted(comp_dir.glob("*.csv")):
        df = load_result_csv(path)
        if df is None or "model_prediction" not in df.columns:
            continue
        preds = df["model_prediction"]
        flags = preds.apply(
            lambda p: (not isinstance(p, str))
            or p.strip() == ""
            or p.strip().lower().startswith("no rules found")
        )
        if flags.any():
            note[path.stem] = {"n": len(df), "n_no_rules_found": int(flags.sum())}
    return note


def fmt_pct(x: float | None) -> str:
    return "   n/a" if x is None else f"{100 * x:5.1f}%"


def print_report(retrieval: dict, hr_context: dict, comp: dict) -> None:
    print("=" * 100)
    print("ABSTENTION / CORRECT-REJECTION ANALYSIS  (rule_retrieval, n per config below)")
    print(f"strict = canonical QA_PROMPT template ('{STRICT_TEMPLATE}')")
    print("broad  = refusal regex from analyze_basic_rag_failure.py (sensitivity check)")
    print("=" * 100)
    header = (
        f"{'config':32s} {'n':>5s} {'abst.strict':>11s} {'abst.broad':>10s} "
        f"{'F1 abst':>8s} {'F1 answ':>8s} {'usable':>7s} {'wrong':>7s}"
    )
    print(header)
    print("-" * len(header))
    for stem in sorted(retrieval, key=sort_key):
        r = retrieval[stem]
        split = (r.get("three_way_split") or {}).get(f"tau_{TAU_MAIN}")
        f1_a = r.get("mean_f1_abstain")
        f1_n = r.get("mean_f1_answered")
        print(
            f"{stem:32s} {r['n']:>5d} "
            f"{r['n_abstain_strict']:>4d} ({100 * r['abstention_rate_strict']:4.1f}%) "
            f"{fmt_pct(r['abstention_rate_broad']):>10s} "
            f"{('  n/a' if f1_a is None else f'{f1_a:5.3f}'):>8s} "
            f"{('  n/a' if f1_n is None else f'{f1_n:5.3f}'):>8s} "
            f"{fmt_pct(None if split is None else split['answered_usable'] / r['n']):>7s} "
            f"{fmt_pct(None if split is None else split['answered_wrong_rate']):>7s}"
        )
    print(f"\nThree-way split at tau={TAU_MAIN} (F1-BoW): usable = answered & F1>=tau, "
          f"wrong = answered & F1<tau (hallucination proxy).")
    print(f"Sensitivity thresholds in the JSON output: tau in {TAU_SENSITIVITY}.")

    print("\n" + "-" * 100)
    print("AGGREGATE correct- vs. false-rejection context (per-question retrieval hits were")
    print("never persisted -- nodes in results_pretty.json are empty -- so this is the only")
    print("level at which rejection correctness can be argued without a retrieval re-run):")
    for name, s in hr_context.items():
        print(f"  {s.get('paper_column', name):18s} HR@1={s['HR@1']:.3f}  HR@5={s['HR@5']:.3f}  "
              f"HR@15={s['HR@15']:.3f}  MRR={s['MRR']:.3f}")
    print("  -> Basic RAG: rule absent from top-15 in ~66% of queries; its abstentions are")
    print("     predominantly correct rejections of a failed retrieval.")
    print("  -> Advanced RAG: HR@15 = 1.0; its rare abstentions are false rejections.")

    if comp:
        print("\n" + "-" * 100)
        print("SIDE NOTE rule_compilation ('No rules found' / empty predictions, n=30 topics):")
        for stem, c in comp.items():
            print(f"  {stem:32s} {c['n_no_rules_found']}/{c['n']}")
        print("  Too sparse for a rate analysis; abstention story rests on rule_retrieval.")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Abstention/correct-rejection analysis")
    ap.add_argument("--base", default=str(root / "results_for_publication"),
                    help="results_for_publication root")
    ap.add_argument("--output", "-o", default=str(root / "results" / "abstention_analysis.json"),
                    help="JSON output path")
    args = ap.parse_args()

    base = Path(args.base)
    retrieval_dir = base / "rule_extraction" / "retrieval"
    comp_dir = base / "rule_extraction" / "compilation"

    retrieval = {}
    for path in sorted(retrieval_dir.glob("*.csv")):
        r = analyze_file(path)
        if r:
            retrieval[path.stem] = r

    hr_context = {}
    metrics_path = retrieval_dir / "retrieval_metrics.json"
    if metrics_path.exists():
        hr_context = json.loads(metrics_path.read_text(encoding="utf-8"))["summaries"]

    comp = compilation_note(comp_dir)

    print_report(retrieval, hr_context, comp)

    out = {
        "description": "Abstention/correct-rejection analysis on the curated "
                       "rule_retrieval prediction CSVs (no re-run).",
        "definitions": {
            "strict_abstention": f"model_prediction contains '{STRICT_TEMPLATE}' "
                                 "(canonical template enforced by QA_PROMPT)",
            "broad_refusal": "REFUSAL_RE from scripts/analyze_basic_rag_failure.py",
            "answered_wrong": "no abstention and per-question F1-BoW < tau "
                              "(hallucination proxy)",
            "tau_main": TAU_MAIN,
            "caveat": "Correct- vs. false-rejection is an aggregate argument via HR@k "
                      "(retrieval_metrics.json); per-question retrieval hits were not "
                      "persisted in the published runs.",
        },
        "retrieval": {k: retrieval[k] for k in sorted(retrieval, key=sort_key)},
        "retrieval_hr_context": hr_context,
        "compilation_no_rules_found": comp,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nJSON written to {out_path}")


if __name__ == "__main__":
    main()
