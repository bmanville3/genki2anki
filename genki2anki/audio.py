"""Stage 3: align the book's words to the real audio and cut clips. Checkpointed per lesson."""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from . import jp, runctl
from .align import LessonAligner, Target
from .asr import load_asr
from .clips import cut_all
from .config import Config
from .energy import Envelope
from .stream import Stream, assign_windows, make_windows


def load_book(cfg: Config) -> dict:
    book = runctl.read_json(cfg.work / "book.json")
    if book is None:
        raise SystemExit("book.json not found - run `extract` first")
    return book


def load_manifest(cfg: Config) -> dict:
    """All audio known for the book: TTS entries overlaid by real-audio (Genki) entries."""
    m: dict = dict(runctl.read_json(cfg.work / "tts_manifest.json", {}) or {})
    for p in sorted((cfg.work / "align").glob("lesson_*.json")):
        d = runctl.read_json(p)
        if d:
            m.update(d.get("matches", {}))
    return m


def lesson_lexicon(book: dict) -> dict[int, set[str]]:
    lex: dict[int, set[str]] = {}
    for v in book["vocab"]:
        if v["lesson"] is None:
            continue
        lex.setdefault(v["lesson"], set()).update(jp.kana_alternatives(v["kana"] or jp.to_kana(v["kanji"] or "")))
    return lex


def build_targets(book: dict, include_examples: bool) -> dict[int, list[Target]]:
    by_lesson: dict[int, list[Target]] = {}
    for v in book["vocab"]:
        if v["lesson"] is None:
            continue
        # a kanji-only entry has no printed reading: use the generated one for matching
        alts = jp.kana_alternatives(v["kana"]) or jp.kana_alternatives(jp.to_kana(jp.primary_form(v["kanji"] or "")))
        if alts:
            by_lesson.setdefault(v["lesson"], []).append(
                Target(v["id"], alts, v["lesson"], "vocab", v["kanji"] or v["kana"]))
    if include_examples:
        for g in book["grammar"]:
            if g["lesson"] is None:
                continue
            for e in g["examples"]:
                kana = jp.to_kana(e["ja"])
                if len(kana) >= 4:
                    by_lesson.setdefault(g["lesson"], []).append(
                        Target(e["id"], [kana], g["lesson"], "example", e["ja"]))
    return by_lesson


def _asr_fingerprint(cfg: Config) -> list:
    return [[p.name, p.stat().st_size] for p in sorted((cfg.work / "asr").glob("*.json"))]


def write_report(cfg: Config) -> tuple[int, int]:
    rows = []
    for p in sorted((cfg.work / "align").glob("lesson_*.json")):
        d = runctl.read_json(p)
        if d:
            rows.extend(d.get("rows", []))
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", lineterminator="\n")
    w.writerow(["lesson", "kind", "id", "text", "matched", "similarity", "heard_kana",
                "t0", "t1", "file_idx", "isolated", "tight", "recovered"])
    w.writerows(rows)
    runctl.atomic_write_text(cfg.work / "report" / "alignment.tsv", buf.getvalue())
    vocab = [r for r in rows if r[1] == "vocab"]
    return sum(1 for r in vocab if r[4]), len(vocab)


def run_align(cfg: Config, force: bool = False) -> dict:
    book = load_book(cfg)
    asrs = load_asr(cfg)
    stream = Stream.build(asrs)
    print(f"[align] stream: {len(stream)} kana chars from {len(asrs)} audio files")
    if len(stream) == 0:
        raise SystemExit("empty transcript stream")

    envs: dict[int, Envelope | None] = {}
    for fi, f in enumerate(stream.files):
        try:
            envs[fi] = Envelope.for_wav(Path(f["wav"]))
        except Exception as e:  # noqa: BLE001
            print(f"[align] no energy envelope for {f['wav']}: {e}")
            envs[fi] = None

    windows = make_windows(len(stream), cfg.align.window_chars)
    texts = [stream.chars[a:b] for a, b in windows]
    win_lesson = assign_windows(texts, lesson_lexicon(book), cfg.align.monotone_lessons)

    targets_by_lesson = build_targets(book, cfg.align.align_examples)
    asr_fp = _asr_fingerprint(cfg)
    media = cfg.work / "media"
    align_dir = cfg.work / "align"
    align_dir.mkdir(parents=True, exist_ok=True)
    skipped_stop = 0
    for lesson in sorted(targets_by_lesson):
        if runctl.stopping():
            skipped_stop += 1
            continue
        idxs = [i for i, l in enumerate(win_lesson) if l == lesson]
        if not idxs:
            print(f"[align] lesson {lesson}: no audio windows assigned")
            continue
        keep = set(idxs)
        for i in idxs:                       # one window of margin either side
            keep.update(j for j in (i - 1, i + 1) if 0 <= j < len(windows))
        targets = targets_by_lesson[lesson]
        fp = hashlib.sha1(json.dumps(
            [asr_fp, [[t.id, t.alts, t.kind] for t in targets], sorted(keep),
             dataclasses.asdict(cfg.align), windows[0] if windows else None],
            ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        lf = align_dir / f"lesson_{lesson:02d}.json"
        cached = runctl.read_json(lf)
        if (cached and not force and cached.get("fp") == fp
                and all((media / e["file"]).exists() for e in cached["matches"].values())):
            print(f"[align] lesson {lesson}: done already")
            continue

        gidx = np.concatenate([np.arange(*windows[i]) for i in sorted(keep)])
        matches = LessonAligner(stream, gidx, envs, cfg.align).align(targets)
        n_vocab = sum(1 for t in targets if t.kind == "vocab")
        n_vm = sum(1 for t in targets if t.kind == "vocab" and t.id in matches)
        extra = ""
        if cfg.align.align_examples:
            ne = sum(1 for t in targets if t.kind == "example")
            extra = f", {sum(1 for t in targets if t.kind == 'example' and t.id in matches)}/{ne} examples"
        print(f"[align] lesson {lesson}: matched {n_vm}/{n_vocab} vocab{extra}")

        rows, jobs = [], []
        for t in targets:
            m = matches.get(t.id)
            rows.append([lesson, t.kind, t.id, t.label, bool(m),
                         f"{m.sim:.2f}" if m else "", m.heard if m else "",
                         f"{m.t0:.2f}" if m else "", f"{m.t1:.2f}" if m else "",
                         m.file if m else "", m.isolated if m else "", m.tight if m else "",
                         m.recovered if m else ""])
            if m:
                jobs.append((t.id, m))
        entries = cut_all(jobs, stream.files, envs, media, cfg.align)
        runctl.atomic_write_json(lf, {"fp": fp, "lesson": lesson, "matches": entries, "rows": rows})

    got, total = write_report(cfg)
    note = f" (stopped: {skipped_stop} lessons not processed yet)" if skipped_stop else ""
    print(f"[align] vocab with real audio so far: {got}/{total}{note}. "
          f"Review: {cfg.work / 'report' / 'alignment.tsv'}")
    return load_manifest(cfg)
