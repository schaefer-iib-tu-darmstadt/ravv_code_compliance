"""Opt-in (``RAG_TIMING=1``, set before clients are built) per-call latency/token log at the httpx layer.

Fields: ``limiter_wait_s`` (rate-limiter blocking), ``http_s`` (send + server + body read),
``wall_s`` (stage wall-clock, incl. CPU-only stages). Records go to JSONL; disabled runs are unchanged.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LOCK = threading.Lock()
_PROMPT_HEAD_CHARS = 100

_OUTPUT_PATH: Optional[Path] = None
_RUN_META: dict[str, Any] = {}

_QUERY_CTX: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "timing_query_ctx", default=None
)


def enabled() -> bool:
    return os.environ.get("RAG_TIMING") == "1"


def set_output(path: Optional[str | Path]) -> None:
    global _OUTPUT_PATH
    with _LOCK:
        _OUTPUT_PATH = Path(path) if path is not None else None


def set_run_meta(pipeline: str = "", benchmark: str = "", run_label: str = "") -> None:
    global _RUN_META
    with _LOCK:
        _RUN_META = {"pipeline": pipeline, "benchmark": benchmark, "run_label": run_label}


def _append(record: dict) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
              **_RUN_META, **record}
    line = json.dumps(record, ensure_ascii=False)
    with _LOCK:
        path = _OUTPUT_PATH or (_PROJECT_ROOT / "results" / "latency_cost"
                                / f"orphan_{datetime.now().strftime('%Y-%m-%d')}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _ctx_fields() -> dict:
    ctx = _QUERY_CTX.get()
    if not ctx:
        return {"qid": None, "attempt": None, "stage": None, "warmup": False}
    return {"qid": ctx.get("qid"), "attempt": ctx.get("attempt"),
            "stage": ctx.get("stage"), "warmup": bool(ctx.get("warmup"))}


@contextlib.contextmanager
def query_context(qid, attempt: int = 1, warmup: bool = False):
    token = _QUERY_CTX.set({"qid": qid, "attempt": attempt, "stage": None, "warmup": warmup})
    try:
        yield
    finally:
        _QUERY_CTX.reset(token)


@contextlib.contextmanager
def stage(label: str):
    ctx = _QUERY_CTX.get()
    prev = ctx.get("stage") if ctx else None
    if ctx is not None:
        ctx["stage"] = label
    t0 = time.perf_counter()
    try:
        yield
    finally:
        wall = time.perf_counter() - t0
        if ctx is not None:
            ctx["stage"] = prev
        if enabled():
            try:
                _append({"type": "stage", **_ctx_fields(), "stage": label,
                         "wall_s": round(wall, 4)})
            except Exception as exc:
                print(f"[timing-log] stage record skipped: {exc}")


def log_query(qid, e2e_s: float, ok: bool, attempt: int = 1, warmup: bool = False) -> None:
    if not enabled():
        return
    try:
        _append({"type": "query", "qid": qid, "attempt": attempt,
                 "e2e_s": round(e2e_s, 4), "ok": bool(ok), "warmup": bool(warmup)})
    except Exception as exc:
        print(f"[timing-log] query record skipped: {exc}")


def _mark_enter(request) -> None:
    try:
        request.extensions["timing"] = {"t_enter": time.perf_counter()}
    except Exception:
        pass


def _mark_sent(request) -> None:
    try:
        request.extensions.setdefault("timing", {})["t_sent"] = time.perf_counter()
    except Exception:
        pass


def _classify_endpoint(url_path: str) -> str:
    if "/embeddings" in url_path:
        return "embedding"
    if "/chat/completions" in url_path:
        return "chat"
    return "other"


def _prompt_head(request) -> Optional[str]:
    try:
        body = json.loads(request.content.decode("utf-8"))
        for message in body.get("messages", []):
            if message.get("role") == "system":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content[:_PROMPT_HEAD_CHARS]
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return str(part.get("text", ""))[:_PROMPT_HEAD_CHARS]
        inp = body.get("input")
        if isinstance(inp, str):
            return inp[:_PROMPT_HEAD_CHARS]
        if isinstance(inp, list) and inp and isinstance(inp[0], str):
            return inp[0][:_PROMPT_HEAD_CHARS]
    except Exception:
        pass
    return None


def _request_model(request) -> Optional[str]:
    try:
        return json.loads(request.content.decode("utf-8")).get("model")
    except Exception:
        return None


def _on_response(response) -> None:
    try:
        timing = response.request.extensions.get("timing", {})
        t_enter = timing.get("t_enter")
        t_sent = timing.get("t_sent")
        body = None
        if "text/event-stream" not in response.headers.get("content-type", ""):
            response.read()
            try:
                body = json.loads(response.content)
            except Exception:
                body = None
        t_done = time.perf_counter()

        usage = (body or {}).get("usage") if isinstance(body, dict) else None
        record = {
            "type": "http",
            **_ctx_fields(),
            "host": response.request.url.host,
            "path": str(response.request.url.path),
            "endpoint": _classify_endpoint(str(response.request.url.path)),
            "status": response.status_code,
            "limiter_wait_s": round(t_sent - t_enter, 4)
            if (t_enter is not None and t_sent is not None) else None,
            "http_s": round(t_done - t_sent, 4) if t_sent is not None else None,
            "model": (body or {}).get("model") if isinstance(body, dict) else None,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            } if isinstance(usage, dict) else None,
            "prompt_head": _prompt_head(response.request),
        }
        if record["model"] is None:
            record["model"] = _request_model(response.request)
        _append(record)
    except Exception as exc:
        print(f"[timing-log] http record skipped: {exc}")


def wrap_event_hooks(request_hooks: list[Callable]) -> dict:
    hooks = list(request_hooks)
    if not enabled():
        return {"request": hooks}
    return {"request": [_mark_enter, *hooks, _mark_sent], "response": [_on_response]}


def timing_only_http_client():
    if not enabled():
        return None
    import httpx
    return httpx.Client(event_hooks=wrap_event_hooks([]))
