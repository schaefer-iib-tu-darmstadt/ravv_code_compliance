"""Conversion of retrieved nodes into JSON-serialisable records for the result files."""
def _to_primitive(x):
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): _to_primitive(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_to_primitive(v) for v in x]
    return str(x)

def normalize_node(node_with_score):
    meta = getattr(node_with_score.node, "metadata", {}) or {}
    header_path = meta.get("header_path")
    if isinstance(header_path, (list, tuple, set)):
        header_path = [str(v) for v in header_path]
    elif header_path is not None:
        header_path = str(header_path)
    return {
        "score": float(node_with_score.score) if getattr(node_with_score, "score", None) is not None else None,
        "text": str(getattr(node_with_score.node, "text", "")),
        "id": str(getattr(node_with_score.node, "id_", "")),
        "header_path": header_path,
        "metadata": _to_primitive(meta),
    }