"""
Offline propagation of the GPT-OSS router's full-set decisions to end-to-end retrieval/compilation scores.
Each question takes the score of the pipeline its predicted route dispatches to; routes without a result
for that subtask (e.g. retrieval-to-vision, no image) score 0. Also reports the misrouting matrix, the
score under correct routing and the worst case (every misroute scored 0).

Usage:
    uv run python scripts/analyze_routing_error_propagation.py [--no-write]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PUB = REPO / "results_for_publication"
ROUTING_CSV = REPO / "results" / "agentic_orchestration" / "full_set" / "openai-gpt-oss-120b.csv"
OUT_DIR = PUB / "orchestration" / "error_propagation"
CORRECT = {"retrieval": "vector-rag", "compilation": "graph-rag"}
ROUTE_SCORES = {
    "retrieval": {
        "vector-rag": PUB / "rule_extraction" / "retrieval" / "gpt-oss_advanced_rag.csv",
        "other": PUB / "rule_extraction" / "retrieval" / "gpt-oss_no_rag.csv",
    },
    "compilation": {
        "graph-rag": REPO / "results" / "rule_extraction" / "rule_compilation"
        / "structured_graph_rag" / "d_norr-20e6f730f7" / "results.csv",
        "other": PUB / "rule_extraction" / "compilation" / "gpt-oss_no_rag.csv",
    },
}


def scores(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    return df.drop_duplicates("question").set_index("question")["eval"].astype(float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--no-write", action="store_true", help="stdout only")
    args = ap.parse_args()

    r = pd.read_csv(ROUTING_CSV)
    matrix = pd.crosstab(r.ground_truth, r.predicted_category_normalized)
    report = {"misrouting_matrix": matrix.to_dict(orient="index")}
    for task, routes in ROUTE_SCORES.items():
        q = r[r.ground_truth == task][["question", "predicted_category_normalized"]]
        s = {route: scores(p) for route, p in routes.items()}
        correct = s[CORRECT[task]].reindex(q.question)
        routed = [s[rt].get(qq, 0.0) if rt in s else 0.0
                  for qq, rt in zip(q.question, q.predicted_category_normalized)]
        worst = [c if rt == CORRECT[task] else 0.0
                 for c, rt in zip(correct, q.predicted_category_normalized)]
        report[task] = {
            "n": len(q),
            "missing_scores": int(correct.isna().sum()),
            "correct_routing": round(float(correct.mean()), 3),
            "routed_end_to_end": round(sum(routed) / len(q), 3),
            "worst_case": round(sum(worst) / len(q), 3),
        }

    print(json.dumps(report, indent=2))
    if not args.no_write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "error_propagation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwritten to {OUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
