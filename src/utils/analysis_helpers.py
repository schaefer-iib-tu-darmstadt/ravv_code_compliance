"""Answer/rule-ID parsing and path helpers for the post-evaluation analysis scripts.

Analysis only; the DesignQA evaluator in ``eval/`` parses answers itself.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd


def extract_yes_no(prediction: str) -> str | None:
    if not isinstance(prediction, str):
        return None

    text = prediction.strip().lower()

    m = re.search(r"answer\s*:\s*(yes|no)\b", text)
    if m:
        return m.group(1)

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        last = lines[-1].strip("., ")
        if last in ("yes", "no"):
            return last

    m = re.match(r"(yes|no)\b", text)
    if m:
        return m.group(1)

    return None


def extract_rule_from_question(question: str) -> str | None:
    if not isinstance(question, str):
        return None
    m = re.search(r"rule\s+([A-Z]+\.[\d.]+[a-z]?)\b", question, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"([A-Z]+\.[\d.]+[a-z]?)\s+(?:in the rule|specified in)", question)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-Z]{1,2}\.[\d.]+[a-z]?)\b", question)
    if m:
        return m.group(1)
    return None


def extract_rule_id(question: str) -> str | None:
    if not isinstance(question, str):
        return None
    m = re.search(r"rule\s+([A-Z]+(?:\.\d+)+[a-z]?)", question, re.IGNORECASE)
    return m.group(1) if m else None


def resolve_path(result_dir: str, project_root: str) -> str:
    result_dir = result_dir.replace("\\", "/")
    if os.path.isabs(result_dir):
        return result_dir
    return os.path.join(project_root, result_dir)


def load_result_csv(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        print(f"WARNING: {path} not found, skipping")
        return None
    df = pd.read_csv(path)
    if "eval" in df.columns:
        df["eval"] = pd.to_numeric(df["eval"], errors="coerce")
    return df
