"""Optional on-disk cache for paid tool calls (text_search, image_search).

Enabled when ``TOOL_CACHE_DB`` points at a sqlite file. Several samples of the
same question (probe rollouts, RL groups) issue largely identical searches; with
the cache the environment is also deterministic for a given query, which is what
an RL reward should see. Only successful results are stored, never errors, so a
transient failure is retried on the next call. Safe across threads and
processes (sqlite WAL, one connection per thread).
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from typing import Callable, Optional

_local = threading.local()
_stats = {"hit": 0, "miss": 0}


def _db_path() -> Optional[str]:
    return os.environ.get("TOOL_CACHE_DB") or None


def _conn() -> Optional[sqlite3.Connection]:
    path = _db_path()
    if not path:
        return None
    conn = getattr(_local, "conn", None)
    if conn is None or getattr(_local, "path", None) != path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        conn = sqlite3.connect(path, timeout=60, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, tool TEXT, v TEXT, t REAL)"
        )
        conn.commit()
        _local.conn, _local.path = conn, path
    return conn


def key(tool: str, *parts: object) -> str:
    h = hashlib.sha256()
    for p in (tool, *parts):
        h.update(p if isinstance(p, bytes) else str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def cached(tool: str, k: str, compute: Callable[[], str]) -> str:
    """Return the cached result for ``k`` or compute, store if successful."""
    conn = _conn()
    if conn is None:
        return compute()
    row = conn.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
    if row is not None:
        _stats["hit"] += 1
        return row[0]
    _stats["miss"] += 1
    value = compute()
    if isinstance(value, str) and value.startswith("Tool execution result"):
        for attempt in range(5):
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO cache VALUES (?,?,?,?)", (k, tool, value, time.time())
                )
                conn.commit()
                break
            except sqlite3.OperationalError:  # database is locked
                time.sleep(0.2 * (attempt + 1))
    return value


def stats() -> dict:
    return dict(_stats)
