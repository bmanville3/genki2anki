"""Durable pipeline state.

* ``atomic_*``   - results are written to ``<name>.part-*`` and renamed, so a kill at any
                   instant can never leave a half-written result behind.
* ``Ledger``     - sqlite table of (stage, key) -> done/failed (+attempts, input fingerprint).
                   A task counts as done only if the ledger says so *and* its artifact exists.
* ``STOP``       - set by Ctrl-C/SIGTERM; stages finish what is in flight and return.
* ``run_tasks``  - bounded worker pool that stops submitting new work when STOP is set.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import sqlite3
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Optional

STOP = threading.Event()


def stop_requested() -> bool:
    return STOP.is_set()


def install_signal_handlers() -> None:
    def handler(signum, frame):  # noqa: ARG001
        if STOP.is_set():
            print("\n[state] second interrupt: exiting now. Everything finished so far is saved.",
                  flush=True)
            os._exit(130)
        STOP.set()
        print("\n[state] stop requested: finishing in-flight items, saving, then exiting. "
              "Ctrl-C again to quit immediately. Re-run the same command to resume.", flush=True)

    signal.signal(signal.SIGINT, handler)
    with contextlib.suppress(Exception):
        signal.signal(signal.SIGTERM, handler)


# ------------------------------------------------------------------ atomic file helpers
def _tmp_for(path: Path) -> Path:
    return path.with_name(f"{path.stem}.part-{os.getpid()}-{threading.get_ident()}{path.suffix}")


@contextlib.contextmanager
def atomic_path(path):
    """Yield a temp path (same extension as ``path``); rename it over ``path`` on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_for(path)
    try:
        yield tmp
        if not tmp.exists():
            raise FileNotFoundError(f"{tmp} was never written")
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def atomic_write_bytes(path, data: bytes) -> None:
    with atomic_path(path) as tmp:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())


def atomic_write_text(path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path, obj: Any, indent: Optional[int] = 1) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent))


def clean_partials(root) -> int:
    """Remove leftovers of interrupted atomic writes."""
    n = 0
    for p in Path(root).rglob("*.part-*"):
        with contextlib.suppress(OSError):
            p.unlink()
            n += 1
    return n


def fingerprint(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------ ledger
class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._con = sqlite3.connect(str(self.path), check_same_thread=False, timeout=60,
                                    isolation_level=None)
        with self._lock:
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.execute(
                "CREATE TABLE IF NOT EXISTS tasks ("
                " stage TEXT NOT NULL, key TEXT NOT NULL, status TEXT NOT NULL, fp TEXT,"
                " attempts INTEGER NOT NULL DEFAULT 0, error TEXT, updated REAL,"
                " PRIMARY KEY (stage, key))")

    def close(self) -> None:
        with self._lock:
            self._con.close()

    def get(self, stage: str, key: str) -> Optional[dict]:
        with self._lock:
            row = self._con.execute(
                "SELECT status, fp, attempts, error FROM tasks WHERE stage=? AND key=?",
                (stage, key)).fetchone()
        return None if row is None else {"status": row[0], "fp": row[1], "attempts": row[2],
                                         "error": row[3]}

    def is_done(self, stage: str, key: str, fp: Optional[str] = None,
                artifact: Optional[Path] = None) -> bool:
        row = self.get(stage, key)
        if not row or row["status"] != "done":
            return False
        if fp is not None and row["fp"] != fp:
            return False
        return artifact is None or Path(artifact).exists()

    def should_try(self, stage: str, key: str, fp: Optional[str] = None,
                   max_attempts: int = 3, artifact: Optional[Path] = None) -> bool:
        if self.is_done(stage, key, fp, artifact):
            return False
        row = self.get(stage, key)
        if row and row["status"] == "failed" and row["fp"] == fp:
            return row["attempts"] < max_attempts
        return True

    def mark_done(self, stage: str, key: str, fp: Optional[str] = None) -> None:
        with self._lock:
            self._con.execute(
                "INSERT INTO tasks(stage,key,status,fp,attempts,error,updated) "
                "VALUES(?,?,'done',?,0,NULL,?) "
                "ON CONFLICT(stage,key) DO UPDATE SET status='done', fp=excluded.fp, "
                "error=NULL, updated=excluded.updated",
                (stage, key, fp, time.time()))

    def mark_failed(self, stage: str, key: str, error: str, fp: Optional[str] = None) -> None:
        with self._lock:
            row = self.get(stage, key)
            attempts = (row["attempts"] if row and row["fp"] == fp else 0) + 1
            self._con.execute(
                "INSERT INTO tasks(stage,key,status,fp,attempts,error,updated) "
                "VALUES(?,?,'failed',?,?,?,?) "
                "ON CONFLICT(stage,key) DO UPDATE SET status='failed', fp=excluded.fp, "
                "attempts=excluded.attempts, error=excluded.error, updated=excluded.updated",
                (stage, key, fp, attempts, str(error)[:500], time.time()))

    def counts(self, stage: str) -> dict[str, int]:
        with self._lock:
            rows = self._con.execute(
                "SELECT status, COUNT(*) FROM tasks WHERE stage=? GROUP BY status",
                (stage,)).fetchall()
        return {s: n for s, n in rows}

    def failed(self, stage: str) -> list[tuple[str, int, str]]:
        with self._lock:
            return self._con.execute(
                "SELECT key, attempts, error FROM tasks WHERE stage=? AND status='failed' "
                "ORDER BY key", (stage,)).fetchall()

    def reset(self, stage: str, keys: Optional[list[str]] = None, failed_only: bool = False) -> None:
        with self._lock:
            if keys is None:
                q = "DELETE FROM tasks WHERE stage=?" + (" AND status='failed'" if failed_only else "")
                self._con.execute(q, (stage,))
            else:
                for k in keys:
                    self._con.execute("DELETE FROM tasks WHERE stage=? AND key=?", (stage, k))


# ------------------------------------------------------------------ worker pool
def run_tasks(items: list, fn: Callable[[Any], Any], workers: int, label: str,
              progress_every: int = 10) -> dict[str, int]:
    """Run ``fn(item)`` for every item. ``fn`` returns False (or raises) on failure.

    New items stop being submitted as soon as STOP is set; in-flight ones finish.
    """
    stats = {"done": 0, "failed": 0, "total": len(items)}
    workers = max(1, workers)
    it = iter(items)
    exhausted = False
    inflight: dict = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        while True:
            while not exhausted and not STOP.is_set() and len(inflight) < workers * 2:
                try:
                    item = next(it)
                except StopIteration:
                    exhausted = True
                    break
                inflight[ex.submit(fn, item)] = item
            if not inflight:
                break
            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED, timeout=0.5)
            for f in done:
                item = inflight.pop(f)
                try:
                    ok = f.result() is not False
                except Exception as e:  # noqa: BLE001
                    print(f"[{label}] {item!r}: unexpected error: {e}", flush=True)
                    ok = False
                stats["done" if ok else "failed"] += 1
                n = stats["done"] + stats["failed"]
                if n % progress_every == 0 or n == stats["total"]:
                    print(f"[{label}] {n}/{stats['total']} (failed: {stats['failed']})", flush=True)
    stats["left"] = stats["total"] - stats["done"] - stats["failed"]
    if stats["left"]:
        print(f"[{label}] stopped early; {stats['left']} items left for the next run", flush=True)
    return stats
