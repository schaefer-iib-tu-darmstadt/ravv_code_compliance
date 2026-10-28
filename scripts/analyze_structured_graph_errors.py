"""
Residual-error decomposition of the structured-graph compilation run (30 topics) under the DesignQA
multiset F1 (eval/metrics/metrics.py::token_f1_score; asserted against the stored eval column).
Per topic: extra tokens, missed unique clauses vs. duplicated ids in the ground-truth lists (multiset
recall misses), and ground-truth spread over chapters/sections (cf. the advanced pipeline's 5-chapter cap).

Usage:
    uv run python scripts/analyze_structured_graph_errors.py --output <out.json>
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = (REPO / "results" / "rule_extraction" / "rule_compilation"
                   / "structured_graph_rag" / "d_norr-20e6f730f7" / "results.csv")
DEFAULT_OUT = (REPO / "results_for_publication" / "rule_extraction" / "compilation"
               / "structured_graph_error_analysis.json")


def analyze(results_csv: Path) -> dict:
    df = pd.read_csv(results_csv)
    topics = []
    for _, row in df.iterrows():
        term = re.search(r"relevant to `(.*?)`", row["question"]).group(1)
        pred_tokens = str(row["model_prediction"]).split(", ")
        gt_tokens = ast.literal_eval(str(row["ground_truth"]))

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        prec = num_same / len(pred_tokens) if pred_tokens else 0.0
        rec = num_same / len(gt_tokens) if gt_tokens else 0.0
        f1 = 2 * prec * rec / (prec + rec) if num_same else 0.0
        assert abs(f1 - float(row["eval"])) < 1e-6, f"metric mismatch on '{term}'"

        gt_unique = set(gt_tokens)
        extra = list((Counter(pred_tokens) - Counter(gt_tokens)).elements())
        missed_unique = sorted(gt_unique - set(pred_tokens))
        n_gt_duplicates = len(gt_tokens) - len(gt_unique)

        chapters = {m.group(1) for t in gt_unique
                    if (m := re.match(r"([A-Z]+)\.", t))}
        sections = {".".join(t.split(".")[:2]) for t in gt_unique if "." in t}

        topics.append({
            "term": term,
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall_multiset": round(rec, 4),
            "recall_unique": round(len(gt_unique & set(pred_tokens)) / len(gt_unique), 4),
            "n_gt_tokens": len(gt_tokens),
            "n_gt_unique": len(gt_unique),
            "n_gt_duplicates": n_gt_duplicates,
            "n_pred": len(pred_tokens),
            "n_extra": len(extra),
            "missed_unique_clauses": missed_unique,
            "gt_chapter_spread": len(chapters),
            "gt_section_spread": len(sections),
        })

    n = len(topics)
    agg = {
        "n_topics": n,
        "macro_f1": round(sum(t["f1"] for t in topics) / n, 4),
        "macro_precision": round(sum(t["precision"] for t in topics) / n, 4),
        "macro_recall_multiset": round(sum(t["recall_multiset"] for t in topics) / n, 4),
        "macro_recall_unique": round(sum(t["recall_unique"] for t in topics) / n, 4),
        "total_extra_tokens": sum(t["n_extra"] for t in topics),
        "total_missed_tokens_multiset": sum(t["n_gt_tokens"] - round(t["recall_multiset"] * t["n_gt_tokens"]) for t in topics),
        "total_gt_duplicate_tokens": sum(t["n_gt_duplicates"] for t in topics),
        "topics_with_extra": sum(1 for t in topics if t["n_extra"] > 0),
        "topics_with_missed_unique": sum(1 for t in topics if t["missed_unique_clauses"]),
        "topics_with_gt_duplicates": sum(1 for t in topics if t["n_gt_duplicates"] > 0),
        "gt_chapter_spread_mean": round(sum(t["gt_chapter_spread"] for t in topics) / n, 1),
        "gt_chapter_spread_max": max(t["gt_chapter_spread"] for t in topics),
        "gt_section_spread_mean": round(sum(t["gt_section_spread"] for t in topics) / n, 1),
        "gt_section_spread_max": max(t["gt_section_spread"] for t in topics),
    }
    return {
        "description": "Residual-error decomposition of the structured-graph "
                       "compilation run under the official DesignQA multiset F1.",
        "source_run": str(results_csv.relative_to(REPO)),
        "aggregate": agg,
        "topics": sorted(topics, key=lambda t: t["f1"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(DEFAULT_RESULTS))
    ap.add_argument("--output", "-o", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    report = analyze(Path(args.results))
    agg = report["aggregate"]
    print(json.dumps(agg, indent=2))
    print("\nfive lowest-scoring topics:")
    for t in report["topics"][:5]:
        print(f"  {t['term']:42s} f1={t['f1']:.3f}  prec={t['precision']:.3f}  "
              f"rec_unique={t['recall_unique']:.3f}  gt_dup={t['n_gt_duplicates']}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nJSON written to {out}")


if __name__ == "__main__":
    main()
