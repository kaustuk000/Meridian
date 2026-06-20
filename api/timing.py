"""
meridian/api/timing.py
Lightweight performance logger — writes JSON-lines to logs/perf.log,
completely separate from the application logger.
"""

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# File logger — one JSON record per line for easy parsing / grep
# ---------------------------------------------------------------------------
Path("logs").mkdir(exist_ok=True)

_perf_log = logging.getLogger("meridian.perf")
_perf_log.setLevel(logging.INFO)
_perf_log.propagate = False          # never mix with the app/console logger

_fh = logging.FileHandler("logs/perf.log", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(message)s"))
_perf_log.addHandler(_fh)


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------
@contextmanager
def timer():
    """
    Usage:
        with timer() as t:
            do_work()
        print(t["ms"])   # elapsed milliseconds
    """
    state: dict = {"ms": 0.0}
    t0 = time.perf_counter()
    try:
        yield state
    finally:
        state["ms"] = (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------
def _record(op: str, fields: dict) -> None:
    row = {
        "ts":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "op":  op,
        **{k: round(v, 2) if isinstance(v, float) else v for k, v in fields.items()},
    }
    _perf_log.info(json.dumps(row))


def log_encode(encode_ms: float) -> None:
    """Model forward pass (encode_image + encode_text)."""
    _record("encode", {"encode_ms": encode_ms})


def log_search(
    score_ms: float,
    topk_ms: float,
    n_candidates: int,
    index_size: int,
) -> None:
    """Score computation + topk selection."""
    _record("search", {
        "score_ms":    score_ms,
        "topk_ms":     topk_ms,
        "total_ms":    score_ms + topk_ms,
        "n_candidates": n_candidates,
        "index_size":  index_size,
    })


def log_request(
    route: str,
    encode_ms: float,
    search_ms: float,
    cache_ms: float,
    total_ms: float,
) -> None:
    """/search route end-to-end breakdown."""
    _record("request", {
        "route":     route,
        "encode_ms": encode_ms,
        "search_ms": search_ms,
        "cache_ms":  cache_ms,
        "total_ms":  total_ms,
    })


def log_hierarchy(
    n_items: int,
    dist_ms: float,
    linkage_ms: float,
    render_ms: float,
) -> None:
    """/hierarchy/* tree building breakdown."""
    _record("hierarchy", {
        "n_items":    n_items,
        "dist_ms":    dist_ms,
        "linkage_ms": linkage_ms,
        "render_ms":  render_ms,
        "total_ms":   dist_ms + linkage_ms + render_ms,
    })