"""Qdrant collection naming (pipeline, embedding model, sparse mode) and small config helpers."""
import re
from typing import Any, List

def sanitize_collection_name(name: str, max_len: int = 100) -> str:
    name = name.lower()
    name = re.sub(r"[^0-9a-z_\-]+", "-", name)
    name = re.sub(r"-{2,}", "-", name).strip("-_")
    return name[:max_len]

def build_collection_name(
    pipeline_name: str | None,
    embed_model: str | None = None,
    fastembed_sparse_model: str | None = None,
    hybrid_search: bool = False,
    query_mode: str | None = None,
) -> str:
    mode_label = "dense"
    if hybrid_search:
        mode_label = "sparse" if query_mode == "sparse" else "hybrid"

    base_name = pipeline_name
    for suf in ("_dense", "_hybrid", "_sparse"):
        if base_name.endswith(suf):
            base_name = base_name[: -len(suf)]
            break

    def tail(model: str | None) -> str:
        if not model:
            return ""
        short = model.split("/")[-1]
        return sanitize_collection_name(short)

    parts = [base_name, mode_label, tail(embed_model)]
    if mode_label != "dense" and fastembed_sparse_model:
        parts.append(tail(fastembed_sparse_model))

    name = "_".join([p for p in parts if p]).lower()
    return sanitize_collection_name(name)

def ensure_list(obj: Any) -> List[Any]:
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj]

def make_instance(maybe_callable: Any) -> Any:
    return maybe_callable() if callable(maybe_callable) else maybe_callable