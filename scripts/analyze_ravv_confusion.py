"""Confusion counts of the RAVV compliance runs; positive class = non-compliant (ground truth "no").

Accuracy, violation recall, false-alarm rate, missed violations (mean ± SD over run_*.csv); the scale subset
is all compliant. Verdict = last 'Answer:' line, else inferred from eval; "Rule Not Found" = abstention.

Usage:
    uv run python scripts/analyze_ravv_confusion.py --base results_for_publication/rule_compliance
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics as st
from collections import defaultdict

ANSWER_RE = re.compile(
    r"answer\s*[:\-]?\s*\**\s*(rule\s+not\s+found|not\s+found|yes|no|cannot|unable)",
    re.IGNORECASE,
)
SCALE_LABELS = {"scale", "scale_bar"}


def parse_pred(text: str, gt: str, ev: str) -> str:
    t = (text or "").strip().lower()
    m = None
    for m in ANSWER_RE.finditer(t):
        pass
    if m:
        tok = m.group(1).lower()
        if tok.startswith("yes"):
            return "yes"
        if tok.startswith("no") and "not found" not in tok:
            return "no"
        return "abstain"
    try:
        ev_val = float(ev)
    except (TypeError, ValueError):
        return "abstain"
    return gt if ev_val >= 0.5 else ("no" if gt == "yes" else "yes")


def analyse_run(path: str) -> dict:
    c: dict = defaultdict(int)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt = r["ground_truth"].strip().lower()
            dt = (r.get("dimension_type") or "").strip().lower()
            pred = parse_pred(r["model_prediction"], gt, r["eval"].strip())
            subset = "scale" if dt in SCALE_LABELS else ("direct" if dt == "direct" else "func")
            correct = 1 if pred == gt else 0
            c["n"] += 1
            c[f"n_{subset}"] += 1
            c["correct"] += correct
            c[f"correct_{subset}"] += correct
            if gt == "no":
                c["gt_noncomp"] += 1
                if pred == "no":
                    c["viol_caught"] += 1
                elif pred == "yes":
                    c["missed_viol"] += 1
                else:
                    c["abstain"] += 1
            elif gt == "yes":
                c["gt_comp"] += 1
                c[f"gt_comp_{subset}"] += 1
                if pred == "no":
                    c["false_alarm"] += 1
                    c[f"false_alarm_{subset}"] += 1
                elif pred != "yes":
                    c["abstain"] += 1
    return c


def _msd(vals: list[float]) -> tuple[float, float]:
    return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)


def aggregate(config_dir: str) -> dict | None:
    runs = sorted(glob.glob(os.path.join(config_dir, "run_*.csv")))
    if not runs:
        runs = [config_dir] if os.path.isfile(config_dir) else []
    if not runs:
        return None
    per = [analyse_run(p) for p in runs]

    def series(fn):
        return [fn(c) for c in per]

    out = {"n_runs": len(per)}
    out["acc"] = _msd(series(lambda c: c["correct"] / max(c["n"], 1)))
    if per[0]["n_direct"]:
        out["acc_direct"] = _msd(series(lambda c: c["correct_direct"] / max(c["n_direct"], 1)))
    if per[0]["n_scale"]:
        out["acc_scale"] = _msd(series(lambda c: c["correct_scale"] / max(c["n_scale"], 1)))
        out["fa_scale"] = _msd(series(lambda c: c["false_alarm_scale"] / max(c["gt_comp_scale"], 1)))
    if per[0]["gt_noncomp"]:
        out["viol_recall"] = _msd(series(lambda c: c["viol_caught"] / max(c["gt_noncomp"], 1)))
        out["missed_viol"] = _msd(series(lambda c: c["missed_viol"]))
    if per[0]["gt_comp"]:
        out["false_alarm_rate"] = _msd(series(lambda c: c["false_alarm"] / max(c["gt_comp"], 1)))
    out["abstain"] = _msd(series(lambda c: c["abstain"]))
    out["split"] = {k: per[0].get(k, 0) for k in ("n", "n_direct", "n_scale", "gt_comp", "gt_noncomp")}
    return out


def _f(pair) -> str:
    return "    --   " if pair is None else f"{pair[0]:5.3f}±{pair[1]:4.3f}"


def report(task: str, configs: list[tuple[str, str]]) -> None:
    print(f"\n{'=' * 92}\n{task.upper()}\n{'=' * 92}")
    print(f"{'config':30s} {'ACC':>11s} {'scaleACC':>11s} {'ViolRecall':>11s} "
          f"{'FalseAlarm':>11s} {'MissViol':>10s} {'scaleFA':>11s}")
    for label, d in configs:
        a = aggregate(d)
        if a is None:
            print(f"{label:30s}  (no data at {d})")
            continue
        print(f"{label:30s} {_f(a.get('acc')):>11s} {_f(a.get('acc_scale')):>11s} "
              f"{_f(a.get('viol_recall')):>11s} {_f(a.get('false_alarm_rate')):>11s} "
              f"{_f(a.get('missed_viol')):>10s} {_f(a.get('fa_scale')):>11s}")


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(here, "results_for_publication", "rule_compliance"))
    args = ap.parse_args()
    dim, fun = os.path.join(args.base, "dimension"), os.path.join(args.base, "functional")

    def triplet(root: str) -> list[tuple[str, str]]:
        return [
            ("No-RAG ICR (Gemma)",         os.path.join(root, "gemma_no_rag_vision")),
            ("RAVV weak (Gemma->Gemma)",   os.path.join(root, "gemma2gemma_ravv")),
            ("RAVV strong (GPT->Gemma)",   os.path.join(root, "gpt2gemma_ravv")),
            ("RAVV (GPT->Qwen2.5VL)",      os.path.join(root, "gpt2qwen_ravv.csv")),
            ("RAVV (Mistral->Qwen2.5VL)",  os.path.join(root, "mistral2qwen_ravv.csv")),
        ]

    report("dimension (n=60)", triplet(dim))
    report("functional (n=16)", triplet(fun))


if __name__ == "__main__":
    main()
