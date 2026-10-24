"""`genki2anki status`: what is done and what a re-run would still do (reads files only)."""
from __future__ import annotations

from pathlib import Path

from . import runctl
from .config import Config


def _count(d: Path, pattern: str = "*.json", key=None) -> int:
    n = 0
    for p in d.glob(pattern):
        rec = runctl.read_json(p)
        if rec is not None and (key is None or key(rec)):
            n += 1
    return n


def run_status(cfg: Config) -> None:
    w = cfg.work
    print(f"work dir: {w.resolve()}")
    book = runctl.read_json(w / "book.json")

    # extract
    ok = _count(w / "extract", "p*.json", lambda r: r.get("ok"))
    bad = _count(w / "extract", "p*.json", lambda r: not r.get("ok"))
    total = ""
    if cfg.pdf and Path(cfg.pdf).exists():
        from .pdf import page_count, parse_pages
        total = f"/{len(parse_pages(cfg.pages, page_count(cfg.pdf)))}"
    print(f"extract    : {ok}{total} pages done" + (f", {bad} failed (will retry)" if bad else ""))
    if book:
        print(f"book       : {len(book['vocab'])} vocab, {len(book['grammar'])} grammar points, "
              f"{sum(len(g['examples']) for g in book['grammar'])} sentences")

    # asr
    plans = list((w / "asr" / "parts").glob("*.plan.json"))
    finals = [p for p in (w / "asr").glob("*.json")]
    if plans or finals:
        chunks_done = chunks_total = 0
        for p in plans:
            plan = runctl.read_json(p) or {}
            stem = p.name[:-len(".plan.json")]
            chunks_total += len(plan.get("chunks", []))
            chunks_done += len(list((w / "asr" / "parts" / stem).glob("*.json")))
        print(f"transcribe : {len(finals)} audio files complete; chunks {chunks_done}/{chunks_total} "
              f"in planned files")
    else:
        print("transcribe : not started")

    # align
    lessons = list((w / "align").glob("lesson_*.json"))
    matched = total_v = 0
    for p in lessons:
        d = runctl.read_json(p) or {}
        for r in d.get("rows", []):
            if r[1] == "vocab":
                total_v += 1
                matched += bool(r[4])
    print(f"align      : {len(lessons)} lessons done; vocab with real audio {matched}/{total_v}"
          if lessons else "align      : not started")

    nvocab = len(book["vocab"]) if book else 0
    nsent = sum(len(g["examples"]) for g in book["grammar"]) if book else 0
    tts = runctl.read_json(w / "tts_manifest.json", {}) or {}
    print(f"tts        : {len(tts)} clips")
    e_ok = _count(w / "enrich", "*.json", lambda r: r.get("status") in ("ok", "not_found"))
    e_shot = _count(w / "enrich", "*.json", lambda r: r.get("screenshot"))
    print(f"jisho      : {e_ok}/{nvocab} words looked up, {e_shot} screenshots")
    n_ok = _count(w / "normalize")
    print(f"normalize  : {n_ok}/{nsent} sentences")
    out = Path(cfg.out_dir)
    print(f"export     : {'vocab.csv + grammar.csv present in ' + str(out) if (out / 'vocab.csv').exists() else 'not exported yet'}")
