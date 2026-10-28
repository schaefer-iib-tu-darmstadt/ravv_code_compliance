"""Process-wide rate limiting for the GWDG backend (limits apply per API key).

Default: 2 requests/s plus 60 requests/min, enforced jointly; every HTTP attempt
(including SDK retries) draws from one shared budget.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable, Deque, Optional, Sequence, Tuple, Union

CapSpec = Union[float, Callable[[], float]]


class RateLimiter:
    def __init__(self, max_per_window: float, window: float = 60.0):
        self.max_per_window = float(max_per_window)
        self.window = float(window)
        self._times: Deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.max_per_window <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] > self.window:
                    self._times.popleft()
                if len(self._times) < self.max_per_window:
                    self._times.append(now)
                    return
                sleep_for = self.window - (now - self._times[0])
            time.sleep(max(0.01, min(sleep_for, self.window)))


def _as_cap(cap: CapSpec) -> float:
    return float(cap() if callable(cap) else cap)


class MultiWindowRateLimiter:
    def __init__(self, limits: Sequence[Tuple[CapSpec, float]]):
        self._limits: list[Tuple[CapSpec, float]] = [(cap, float(win)) for cap, win in limits]
        self._times: list[Deque[float]] = [deque() for _ in self._limits]
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if not self._limits:
            return
        max_win = max(win for _, win in self._limits)
        while True:
            with self._lock:
                now = time.monotonic()
                wait = 0.0
                ok = True
                for (cap, win), dq in zip(self._limits, self._times):
                    while dq and now - dq[0] > win:
                        dq.popleft()
                    c = _as_cap(cap)
                    if c <= 0:
                        continue
                    if len(dq) >= c:
                        ok = False
                        wait = max(wait, win - (now - dq[0]))
                if ok:
                    for dq in self._times:
                        dq.append(now)
                    return
            time.sleep(max(0.01, min(wait, max_win)))


def _default_now() -> datetime:
    tz_name = os.environ.get("GWDG_RATE_TZ", "Europe/Berlin")
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(tz_name))
    except Exception:
        return datetime.now()


def gwdg_minute_cap(
    now_fn: Optional[Callable[[], datetime]] = None,
    day_rpm: float = 60.0,
    night_rpm: float = 60.0,
) -> float:
    now = (now_fn or _default_now)()
    weekday_business = now.weekday() < 5 and 7 <= now.hour < 19
    return day_rpm if weekday_business else night_rpm


def build_gwdg_limiter(
    rps: float = 2.0,
    minute_rpm: Optional[float] = None,
    day_rpm: float = 60.0,
    night_rpm: float = 60.0,
    now_fn: Optional[Callable[[], datetime]] = None,
) -> MultiWindowRateLimiter:
    if minute_rpm is not None:
        minute_cap: CapSpec = float(minute_rpm)
    else:
        minute_cap = lambda: gwdg_minute_cap(now_fn, day_rpm, night_rpm)
    return MultiWindowRateLimiter([(float(rps), 1.0), (minute_cap, 60.0)])


_PROCESS_LIMITER: Optional[object] = None
_PROCESS_LOCK = threading.Lock()


def set_request_limiter(limiter: Optional[object]) -> None:
    global _PROCESS_LIMITER
    with _PROCESS_LOCK:
        _PROCESS_LIMITER = limiter


def get_request_limiter() -> Optional[object]:
    return _PROCESS_LIMITER


def gate() -> None:
    lim = _PROCESS_LIMITER
    if lim is not None:
        lim.acquire()


def ensure_default_limiter() -> None:
    global _PROCESS_LIMITER
    with _PROCESS_LOCK:
        if _PROCESS_LIMITER is None:
            _PROCESS_LIMITER = build_gwdg_limiter()
            print("[RateLimit] default GWDG limiter installed "
                  "(2/s + 60 per min, every HTTP attempt counted)")


def gated_http_client(limiter: Optional[object] = None, **kwargs) -> "httpx.Client":
    import httpx

    def _gate_hook(_request) -> None:
        if limiter is None:
            gate()
        else:
            limiter.acquire()

    from src.utils.llm_api import timing_log
    return httpx.Client(event_hooks=timing_log.wrap_event_hooks([_gate_hook]), **kwargs)


def gated_async_http_client(limiter: Optional[object] = None, **kwargs) -> "httpx.AsyncClient":
    import httpx

    async def _gate_hook(_request) -> None:
        if limiter is None:
            gate()
        else:
            limiter.acquire()

    return httpx.AsyncClient(event_hooks={"request": [_gate_hook]}, **kwargs)
