"""
Structure perturbation of the clause graph on the 30 compilation topics (offline, no LLM).
Re-parses perturbed markdown and scores token F1: dropped cross-references and OCR-style id noise
(5 seeds each), hyphenated ids and stripped markup (each also with a grammar adapted by one rule).

Usage:
    uv run python scripts/run_structure_perturbation.py
    uv run python scripts/run_structure_perturbation.py --no-write
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import statistics
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from eval.metrics.metrics import token_f1_score
from src.rag.graph.structured_graph import (
    DEFAULT_GRAMMAR,
    ID_RE,
    MARKER_CATALOG,
    REL_CROSSREF,
    ClauseGrammar,
    build_graph,
    format_answer,
    parse_clauses,
    retrieve_ids,
)
from src.strategies import PATHS

GT_CSV = REPO / "dataset" / "rule_extraction" / "rule_compilation_qa.csv"
OUT_DIR = REPO / "results_for_publication" / "rule_extraction" / "structure_perturbation"
SEEDS = range(5)
CROSSREF_DROP = (0.25, 0.5, 0.75, 1.0)
OCR_RATES = (0.01, 0.02, 0.05, 0.10)
HYPHEN_GRAMMAR = ClauseGrammar(normalizations=((r"\b([A-Z]{1,3}(?:\.\d+)*)-(\d+)\b", r"\1.\2"),) * 6)
MARKER_CATALOG["bare_id"] = (r"^({ID})()\s*$", 1, 2)
BARE_ID_GRAMMAR = ClauseGrammar(marker_kinds=DEFAULT_GRAMMAR.marker_kinds + ("bare_id",))


def score(g, qa: pd.DataFrame) -> float:
    f1 = [token_f1_score(format_answer(retrieve_ids(g, q)).split(", "), ast.literal_eval(gt))
          for q, gt in zip(qa.question, qa.ground_truth)]
    return sum(f1) / len(f1)


def graph(md: str, grammar: ClauseGrammar = DEFAULT_GRAMMAR):
    return build_graph(parse_clauses(md, grammar), id_re=grammar.id_re())


def drop_crossrefs(g, p: float, rng: random.Random):
    g = g.copy()
    edges = sorted((u, v, k) for u, v, k in g.edges(keys=True) if k == REL_CROSSREF)
    g.remove_edges_from(rng.sample(edges, round(p * len(edges))))
    return g


def _corrupt(cid: str, rng: random.Random) -> str:
    dots = [i for i, c in enumerate(cid) if c == "."]
    ops = ["comma", "space", "drop"] + (["l"] if "1" in cid else []) + (["O"] if "0" in cid else [])
    op, i = rng.choice(ops), rng.choice(dots)
    if op == "comma":
        return cid[:i] + "," + cid[i + 1:]
    if op == "space":
        return cid[:i + 1] + " " + cid[i + 1:]
    if op == "drop":
        return cid[:i] + cid[i + 1:]
    if op == "l":
        return cid.replace("1", "l", 1)
    return cid.replace("0", "O", 1)


def ocr_noise(md: str, p: float, rng: random.Random) -> str:
    return ID_RE.sub(lambda m: _corrupt(m.group(1), rng) if rng.random() < p else m.group(1), md)


def hyphenate(md: str) -> str:
    return ID_RE.sub(lambda m: m.group(1).replace(".", "-"), md)


def strip_markup(md: str) -> str:
    md = re.sub(r"^#{1,6}\s+", "", md, flags=re.M)
    return md.replace("**", "")


def summarise(vals: list[float]) -> dict:
    return {"mean": round(statistics.mean(vals), 3), "sd": round(statistics.stdev(vals), 3),
            "min": round(min(vals), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--no-write", action="store_true", help="stdout only")
    args = ap.parse_args()

    qa = pd.read_csv(GT_CSV)
    md = (REPO / PATHS["md_rules"]).read_text(encoding="utf-8")
    base = graph(md)
    report = {"baseline": round(score(base, qa), 3),
              "n_crossref_edges": sum(1 for *_e, k in base.edges(keys=True) if k == REL_CROSSREF)}

    report["crossref_drop"] = {
        str(p): summarise([score(drop_crossrefs(base, p, random.Random(s)), qa) for s in SEEDS])
        for p in CROSSREF_DROP}
    report["ocr_id_noise"] = {
        str(p): summarise([score(graph(ocr_noise(md, p, random.Random(s))), qa) for s in SEEDS])
        for p in OCR_RATES}
    hy = hyphenate(md)
    report["numbering_hyphen"] = {"default_grammar": round(score(graph(hy), qa), 3),
                                  "adapted_grammar_1_rule": round(score(graph(hy, HYPHEN_GRAMMAR), qa), 3)}
    flat = strip_markup(md)
    report["markup_stripped"] = {"default_grammar": round(score(graph(flat), qa), 3),
                                 "adapted_grammar_1_marker": round(score(graph(flat, BARE_ID_GRAMMAR), qa), 3)}

    print(json.dumps(report, indent=2))
    if not args.no_write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "perturbation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwritten to {OUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
