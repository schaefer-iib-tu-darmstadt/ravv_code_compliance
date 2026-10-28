"""Version labels, config hashes and run ids for the result directories."""
import re
import json
import hashlib
from typing import Any, Dict


def _short_parser_name(name: str) -> str:
    ln = str(name).lower()
    if "markdown" in ln:
        return "md"
    if "sentence" in ln:
        return "sent"
    if "hier" in ln:
        return "hier"
    return re.sub(r"[^a-z0-9]+", "", ln)[:6] or "par"


def build_version_label(cfg: dict) -> str:
    ret = (cfg.get("retrieval") or cfg)
    llm = (cfg.get("llm") or cfg)

    rk = ret.get("retriever_kwargs", {}) or {}
    mode = rk.get("vector_store_query_mode") or ("hybrid" if ret.get("hybrid_search") else "default")
    mode_short = {"default": "d", "hybrid": "h", "sparse": "s"}.get(str(mode).lower(), str(mode))

    k = rk.get("similarity_top_k")
    sk = rk.get("sparse_top_k")
    alpha = rk.get("alpha")
    temp = llm.get("temperature")

    parser = ret.get("parser")
    parser_name = None
    if callable(parser):
        n = getattr(parser, "__name__", "") or getattr(getattr(parser, "__class__", object), "__name__", "")
        parser_name = _short_parser_name(n)
    elif isinstance(parser, dict):
        parser_name = _short_parser_name(parser.get("class", ""))
        chunk = (parser.get("kwargs") or {}).get("chunk_size")
        if isinstance(chunk, (int, float)):
            parser_name = f"{parser_name}{int(chunk)}"
    elif isinstance(parser, str):
        parser_name = re.sub(r"[^a-z0-9]+", "", parser.lower())[:8]

    node_pp = ret.get("node_postprocessor") or cfg.get("node_postprocessors")
    rerank = "rr" if node_pp else "norr"

    parts = [
        parser_name,
        mode_short,
        f"k{int(k)}" if isinstance(k, (int, float)) else None,
        f"sk{int(sk)}" if isinstance(sk, (int, float)) else None,
        f"a{int(round(alpha * 10))}" if isinstance(alpha, (int, float)) else None,
        f"t{str(temp).replace('.', '')}" if temp is not None else None,
        rerank,
    ]
    label = "_".join(p for p in parts if p)

    label = label.lower().replace(" ", "")
    label = re.sub(r"[^a-z0-9_\-]", "", label)
    return label


def _to_serializable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(v) for v in list(obj)]
    if callable(obj):
        name = getattr(obj, "__name__", None) or obj.__class__.__name__
        mod = getattr(obj, "__module__", None)
        return f"{mod}.{name}" if mod else name
    cls = obj.__class__
    name = f"{getattr(cls, '__module__', '')}.{getattr(cls, '__name__', cls)}".strip(".")
    params = getattr(obj, "__dict__", None)
    if isinstance(params, dict) and params:
        try:
            return {"__obj__": name, "params": _to_serializable(params)}
        except Exception:
            return name
    return name


def canonical_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return _to_serializable(cfg)


def config_hash(cfg: Dict[str, Any], length: int = 10) -> str:
    norm = canonical_config(cfg)
    s = json.dumps(norm, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return h[:length]


def make_run_id(cfg: Dict[str, Any], hash_len: int = 10, run_number: int = None,
                test_run: bool = False) -> str:
    base = f"{build_version_label(cfg)}-{config_hash(cfg, length=hash_len)}"
    if test_run:
        base = f"{base}-test"
    if run_number is not None:
        return f"{base}-r{run_number}"
    return base
