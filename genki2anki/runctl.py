"""Run control: graceful stop, atomic file writes, bounded parallel execution.

Every stage of the pipeline stores its progress as small files written atomically
(temp file + rename), so killing the process at *any* moment - Ctrl-C, SIGTERM, power
loss - can never leave a half-written file behind. Re-running the same command skips
whatever is already finished.
"""
from __future__ import annotations

import json
import os
import signal
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

STOP = threading.Event()


def stopping() -> bool:
    return STOP.is_set()


def install_signal_handlers() -> None:
    """First Ctrl-C / SIGTERM: finish in-flight work, save, exit. Second: abort immediately."""

    def handler(signum, frame):
        if STOP.is_set():
            signal.signal(signum, signal.default_int_handler)
            raise KeyboardInterrupt
        STOP.set()
        print("\n[stop] finishing in-flight work and saving state "
              "(press Ctrl-C again to abort immediately; re-run the same command to resume)",
              flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except ValueError:  # not the main thread
            pass


# ---------------------------------------------------------------------------- atomic IO
def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path | str, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path | str, obj: Any, indent: Optional[int] = None) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent))


def read_json(path: Path | str, default: Any = None) -> Any:
    """JSON file contents, or ``default`` if it is missing or unreadable (e.g. corrupt)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, NotADirectoryError):
        return default


def tmp_sibling(path: Path, suffix: Optional[str] = None) -> Path:
    """A hidden temp name next to ``path`` that keeps a usable extension (for ffmpeg etc.)."""
    suffix = suffix or path.suffix
    return path.with_name(f".{path.stem}.{os.getpid()}.tmp{suffix}")


# ---------------------------------------------------------------------------- parallel
def run_parallel(items: Iterable[Any], fn: Callable[[Any], Any], workers: int,
                 on_result: Optional[Callable[[Any, Any, Optional[BaseException]], None]] = None,
                 label: str = "", every: int = 10,
                 abort: Optional[threading.Event] = None) -> tuple[int, int, int]:
    """Run ``fn(item)`` over ``items`` with at most ``2*workers`` in flight.

    Submission stops as soon as a stop is requested (or ``abort`` is set); work already
    running is allowed to finish. ``on_result(item, result, error)`` runs in the calling
    thread. Returns ``(done, failed, not_started)``.
    """
    items = list(items)
    total = len(items)
    workers = max(1, workers)
    it = iter(items)
    inflight: dict = {}
    done = failed = 0
    ex = ThreadPoolExecutor(max_workers=workers)

    def halted() -> bool:
        return STOP.is_set() or (abort is not None and abort.is_set())

    def submit_more() -> None:
        while len(inflight) < workers * 2 and not halted():
            try:
                item = next(it)
            except StopIteration:
                return
            inflight[ex.submit(fn, item)] = item

    try:
        submit_more()
        while inflight:
            finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in finished:
                item = inflight.pop(fut)
                try:
                    res, err = fut.result(), None
                except Exception as e:  # noqa: BLE001 - per-item failures are reported, not fatal
                    res, err = None, e
                done += 1
                failed += err is not None
                if on_result is not None:
                    on_result(item, res, err)
                if label and (done % every == 0 or done == total):
                    print(f"[{label}] {done}/{total}", flush=True)
            submit_more()
    except KeyboardInterrupt:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)
    return done, failed, total - done
