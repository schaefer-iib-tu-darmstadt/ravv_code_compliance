"""
Offline clause coverage of the LLM-extracted graphs (data/graphs/) and the deterministic clause graph.
Upper bounds for graph retrieval: share of compilation ground-truth clauses present as nodes (strict id,
or lenient id with a trailing title stripped), per-topic recall ceiling, deterministic edges reproduced.

Usage:
    uv run python scripts/analyze_graph_construction_coverage.py [--no-write]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.rag.graph.structured_graph import (
    REL_CROSSREF,
    REL_SUBCLAUSE,
    build_graph_from_markdown,
)
from src.strategies import PATHS

LLM_STORES = {
    "ontology_guided_gpt-5-mini": "schema_graph_gpt-5-mini",
    "zero_shot_gpt-5-mini": "simple_graph_gpt-5-mini",
}
GT_CSV = REPO / "dataset" / "rule_extraction" / "rule_compilation_qa.csv"
OUT_DIR = REPO / "results_for_publication" / "rule_extraction" / "graph_construction_coverage"
LEAD_ID = re.compile(r"^([A-Z]{1,3}(?:\.\d+)+(?:\.[a-z])?)(?=$|\s)")


def load_gt() -> dict[str, set[str]]:
    df = pd.read_csv(GT_CSV)
    topics = df.question.str.extract(r"relevant to `([^`]+)`")[0]
    return {t: {r.strip() for r in ast.literal_eval(g)} for t, g in zip(topics, df.ground_truth)}


def llm_graph(store: Path) -> dict:
    d = json.loads(store.read_text(encoding="utf-8"))
    rule_names = [k for k, v in d["nodes"].items()
                  if v["label"] != "text_chunk" and LEAD_ID.match(k)]
    lenient = defaultdict(set)
    for name in rule_names:
        m = LEAD_ID.match(name)
        if m:
            lenient[m.group(1)].add(name)

    def edges(label: str | None) -> set[tuple[str, str]]:
        out = set()
        for r in d["relations"].values():
            if label is not None and r["label"] != label:
                continue
            s, t = LEAD_ID.match(r["source_id"]), LEAD_ID.match(r["target_id"])
            if s and t:
                out.add((s.group(1), t.group(1)))
        return out

    labels = {r["label"] for r in d["relations"].values()}
    typed = "HAS_SUBRULE" in labels
    return {
        "rule_nodes": len(rule_names),
        "strict_ids": {n for n in rule_names if LEAD_ID.fullmatch(n)},
        "lenient_ids": set(lenient),
        "ids_with_title": sum(1 for n in rule_names if LEAD_ID.match(n) and not LEAD_ID.fullmatch(n)),
        "ids_multiple_surfaces": sum(1 for v in lenient.values() if len(v) > 1),
        "sub": edges("HAS_SUBRULE" if typed else None),
        "xref": edges("CITES" if typed else None),
    }


def det_graph() -> dict:
    g = build_graph_from_markdown(REPO / PATHS["md_rules"])
    ids = set(g.nodes)

    def edges(key: str) -> set[tuple[str, str]]:
        return {(u, v) for u, v, k in g.edges(keys=True) if k == key}

    return {"rule_nodes": len(ids), "strict_ids": ids, "lenient_ids": ids,
            "ids_with_title": 0, "ids_multiple_surfaces": 0,
            "sub": edges(REL_SUBCLAUSE), "xref": edges(REL_CROSSREF)}


def coverage(ids: set[str], gt: dict[str, set[str]]) -> dict:
    allgt = set().union(*gt.values())
    per_topic = [len(v & ids) / len(v) for v in gt.values() if v]
    return {"gt_coverage": round(len(allgt & ids) / len(allgt), 3),
            "topic_recall_ceiling_macro": round(sum(per_topic) / len(per_topic), 3),
            "topics_full_coverage": sum(1 for x in per_topic if x == 1.0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--no-write", action="store_true", help="stdout only")
    args = ap.parse_args()

    gt = load_gt()
    graphs = {name: llm_graph(REPO / "data" / "graphs" / d / "property_graph_store.json")
              for name, d in LLM_STORES.items()
              if (REPO / "data" / "graphs" / d / "property_graph_store.json").exists()}
    graphs["deterministic"] = det_graph()
    det = graphs["deterministic"]
    report = {"n_topics": len(gt), "n_gt_clauses": len(set().union(*gt.values()))}
    for name, gr in graphs.items():
        report[name] = {
            "rule_nodes": gr["rule_nodes"],
            "distinct_clause_ids": len(gr["lenient_ids"]),
            "ids_with_title": gr["ids_with_title"],
            "ids_multiple_surfaces": gr["ids_multiple_surfaces"],
            "strict": coverage(gr["strict_ids"], gt),
            "lenient": coverage(gr["lenient_ids"], gt),
            "hierarchy_edges": len(gr["sub"]),
            "crossref_edges": len(gr["xref"]),
            "det_hierarchy_reproduced": round(len(gr["sub"] & det["sub"]) / len(det["sub"]), 3),
            "det_crossref_reproduced": round(len(gr["xref"] & det["xref"]) / len(det["xref"]), 3),
        }

    print(json.dumps(report, indent=2))
    if not args.no_write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "coverage_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwritten to {OUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
