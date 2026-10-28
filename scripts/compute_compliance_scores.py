"""Accuracy, BLEU-2 and ROUGE-L for the curated compliance CSVs (offline, no LLM calls).

Reads results_for_publication/rule_compliance/{dimension,functional}/ ({pipeline}/run_*.csv as
mean +/- std across runs, else {pipeline}.csv) and writes compliance_scores.json there.

Usage:
    uv run python scripts/compute_compliance_scores.py
"""

import sys, os, json, glob
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.metrics.metrics import eval_dimensions_qa, eval_functional_performance_qa

BASE = os.path.join(os.path.dirname(__file__), "..", "results_for_publication", "rule_compliance")

PIPELINES = [
    ("gemma_no_rag_vision", "Gemma / No RAG (ICR)"),
    ("gemma2gemma_ravv", "Gemma / RAVV (Gemma)"),
    ("gpt2gemma_ravv", "Gemma / RAVV (GPT-OSS)"),
    ("mistral2qwen_ravv", "Qwen2.5-VL / RAVV (Mistral)"),
    ("gpt2qwen_ravv", "Qwen2.5-VL / RAVV (GPT-OSS)"),
]


def _resolve_csvs(benchmark: str, pipeline: str) -> list[str]:
    subdir = os.path.join(BASE, benchmark, pipeline)
    runs = sorted(glob.glob(os.path.join(subdir, "run_*.csv")))
    if runs:
        return runs
    flat = os.path.join(BASE, benchmark, f"{pipeline}.csv")
    if os.path.exists(flat):
        return [flat]
    return []


def _score_one(benchmark: str, csv_path: str) -> dict:
    if benchmark == "dimension":
        (macro_acc, _direct, _scale, all_acc,
         bleu_avg, all_bleus, rouge_avg, _all_rogues) = eval_dimensions_qa(csv_path)
        return {"acc": macro_acc, "bleu2": bleu_avg, "rouge_l": rouge_avg,
                "n_total": len(all_acc), "n_bleu": len(all_bleus)}
    else:
        (macro_acc, all_acc, bleu_avg, _all_bleus, rouge_avg, _all_rogues) = \
            eval_functional_performance_qa(csv_path)
        return {"acc": macro_acc, "bleu2": bleu_avg, "rouge_l": rouge_avg,
                "n_total": len(all_acc), "n_bleu": None}


def _aggregate(scores: list[dict]) -> dict:
    def agg(key):
        vals = [s[key] for s in scores if s.get(key) is not None]
        if not vals:
            return None, None
        mean = sum(vals) / len(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else None
        return mean, std

    acc_mean, acc_std = agg("acc")
    bleu_mean, _ = agg("bleu2")
    rouge_mean, _ = agg("rouge_l")
    return {
        "acc": round(acc_mean, 4) if acc_mean is not None else None,
        "acc_std": round(acc_std, 4) if acc_std is not None else None,
        "bleu2": round(bleu_mean, 4) if bleu_mean is not None else None,
        "rouge_l": round(rouge_mean, 4) if rouge_mean is not None else None,
        "n_runs": len(scores),
        "n_total": scores[0].get("n_total") if scores else None,
    }


def main():
    all_results = {}
    resolved_any = False

    for pipeline, label in PIPELINES:
        entry = {"label": label}

        for benchmark in ("dimension", "functional"):
            csvs = _resolve_csvs(benchmark, pipeline)
            if not csvs:
                print(f"[WARN] Missing: {benchmark}/{pipeline}")
                entry[benchmark] = None
                continue
            resolved_any = True
            per_run = [_score_one(benchmark, c) for c in csvs]
            entry[benchmark] = _aggregate(per_run)

        all_results[pipeline] = entry

    if not resolved_any:
        print("\n[ERROR] No compliance CSVs resolved for any pipeline -- refusing to "
              "overwrite compliance_scores.json. Check the PIPELINES list and directory layout.")
        sys.exit(1)

    out_path = os.path.join(BASE, "compliance_scores.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nScores saved to: {out_path}\n")

    header = (f"{'Pipeline':<30} | {'DIM ACC':>8} {'DIM BLEU':>9} {'DIM ROU':>8} | "
              f"{'FUN ACC':>8} {'FUN BLEU':>9} {'FUN ROU':>8}")
    print(header)
    print("-" * len(header))

    def fmt(v):
        return f"{v:.3f}" if v is not None else "   -  "

    for pipeline, label in PIPELINES:
        e = all_results[pipeline]
        d = e.get("dimension") or {}
        fu = e.get("functional") or {}
        print(
            f"{label:<30} | {fmt(d.get('acc')):>8} {fmt(d.get('bleu2')):>9} {fmt(d.get('rouge_l')):>8}"
            f" | {fmt(fu.get('acc')):>8} {fmt(fu.get('bleu2')):>9} {fmt(fu.get('rouge_l')):>8}"
        )


if __name__ == "__main__":
    main()
