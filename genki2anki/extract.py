"""Stage 1: scanned PDF pages -> structured vocab + grammar (book.json). Resumable per page."""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Optional

from . import jp, runctl
from .config import Config
from .pdf import page_count, parse_pages, render_pages
from .schema import PAGE_KINDS, example_id, vocab_id
from .vlm import VLMClient

_ALLOWED_TAGS = {"p", "b", "i", "ul", "li", "br", "table", "tr", "th", "td"}
_TAG_RE = re.compile(r"<(/?)([a-zA-Z0-9]+)[^>]*?(/?)>")
_KANA_ONLY_RE = re.compile(r"^[぀-ヿー・〜～（）()/／\s,、;；.\-]+$")
_LEAD_NUM_RE = re.compile(r"^\s*(?:[•・▪●○\-]|\(?\d{1,2}[.)）])\s*")


def sanitize_html(html: str) -> str:
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html or "", flags=re.S | re.I)

    def repl(m: re.Match) -> str:
        closing, name = m.group(1), m.group(2).lower()
        if name not in _ALLOWED_TAGS:
            return ""
        if name == "br":
            return "<br>"
        return f"<{closing}{name}>"

    out = _TAG_RE.sub(repl, html)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def _s(v) -> str:
    return re.sub(r"[ 　]+", " ", str(v or "")).strip()


def clean_vocab_item(it: dict) -> Optional[dict]:
    """Normalise one vocabulary entry. ``kana`` may come back empty when the book prints only
    a kanji spelling - the reading is then generated (and flagged) at export time."""
    kana = _s(it.get("kana"))
    kanji = _s(it.get("kanji")) or None
    english = _s(it.get("english"))
    if not english or not (kana or kanji):
        return None
    if kana and jp.has_kanji(kana):      # model put the kanji spelling into the kana field
        kanji = kanji or kana
        kana = ""
    if kana and not _KANA_ONLY_RE.match(kana):
        # latin letters / stray symbols -> keep only the kana part
        kana = re.sub(r"[^぀-ヿー・〜～（）()/／,、;；]", "", kana)
    if not kana and not kanji:
        return None
    if kanji and kanji == kana:
        kanji = None
    return {"kana": kana, "kanji": kanji, "english": english,
            "category": _s(it.get("category")) or None}


def clean_example(ex: dict) -> Optional[dict]:
    ja = _LEAD_NUM_RE.sub("", _s(ex.get("japanese")))
    en = _s(ex.get("english"))
    if not ja or not jp.has_japanese(ja) or not en:
        return None
    return {"ja": ja, "en": en}


def normalize_page(raw: dict) -> dict:
    kind = raw.get("page_kind")
    kind = kind if kind in PAGE_KINDS else "other"
    lesson = raw.get("lesson")
    lesson = lesson if isinstance(lesson, int) and 0 < lesson < 60 else None
    vocab = [v for v in (clean_vocab_item(i) for i in raw.get("vocabulary") or []) if v]
    grammar = []
    for g in raw.get("grammar_points") or []:
        examples = [e for e in (clean_example(x) for x in g.get("examples") or []) if e]
        title = _s(g.get("title"))
        expl = sanitize_html(g.get("explanation_html") or "")
        if not (title or expl or examples):
            continue
        num = g.get("number")
        grammar.append({
            "number": num if isinstance(num, int) else None,
            "title": title,
            "continued": bool(g.get("continued_from_previous_page")) or not title,
            "explanation_html": expl,
            "examples": examples,
        })
    return {"kind": kind, "lesson": lesson, "vocab": vocab, "grammar": grammar}


def merge_pages(pages: list[tuple[int, dict]], skip_kinds: list[str]) -> dict:
    """Combine per-page results (already normalised) into one book."""
    vocab: list[dict] = []
    grammar: list[dict] = []
    seen_vocab: set[tuple] = set()
    seen_ex: set[tuple] = set()
    lesson: Optional[int] = None
    for page_no, pg in sorted(pages, key=lambda x: x[0]):
        if pg["lesson"] is not None:
            lesson = pg["lesson"]
        if pg["kind"] in skip_kinds:
            continue
        for v in pg["vocab"]:
            key = (lesson, v["kana"], v["kanji"])
            if key in seen_vocab:
                continue
            seen_vocab.add(key)
            vocab.append({"id": vocab_id(lesson, v["kana"], v["kanji"]), "lesson": lesson,
                          "page": page_no, **v})
        for g in pg["grammar"]:
            prev = grammar[-1] if grammar else None
            if g["continued"] and prev is not None and prev["lesson"] == lesson:
                if g["explanation_html"]:
                    prev["explanation_html"] = (prev["explanation_html"] + "\n"
                                                + g["explanation_html"]).strip()
                tgt = prev
                if page_no not in prev["pages"]:
                    prev["pages"].append(page_no)
            else:
                tgt = {"lesson": lesson, "number": g["number"], "title": g["title"] or "(untitled)",
                       "explanation_html": g["explanation_html"], "examples": [], "pages": [page_no]}
                grammar.append(tgt)
            for e in g["examples"]:
                key = (lesson, e["ja"])
                if key in seen_ex:
                    continue
                seen_ex.add(key)
                tgt["examples"].append({"id": example_id(lesson, e["ja"]), "page": page_no, **e})
    return {"vocab": vocab, "grammar": grammar}


def _tsv(header: list[str], rows: list[list]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def write_review_tsv(book: dict, work: Path) -> None:
    runctl.atomic_write_text(work / "book_vocab.tsv", _tsv(
        ["lesson", "page", "category", "kana", "kanji", "english"],
        [[v["lesson"], v["page"], v["category"], v["kana"], v["kanji"] or "", v["english"]]
         for v in book["vocab"]]))
    runctl.atomic_write_text(work / "book_grammar.tsv", _tsv(
        ["lesson", "page", "number", "title", "ja", "en"],
        [[g["lesson"], e.get("page"), g["number"], g["title"], e["ja"], e["en"]]
         for g in book["grammar"] for e in g["examples"]]))


def rebuild_book(cfg: Config) -> dict:
    """Merge every cached page result into work/book.json."""
    work = cfg.work
    pages: list[tuple[int, dict]] = []
    for p in sorted((work / "extract").glob("p*.json")):
        rec = runctl.read_json(p)
        if rec and rec.get("ok"):
            pages.append((rec["page"], normalize_page(rec["data"])))
    book = merge_pages(pages, cfg.skip_kinds)
    book["meta"] = {"pages_processed": len(pages), "pdf": cfg.pdf}
    runctl.atomic_write_json(work / "book.json", book, indent=1)
    write_review_tsv(book, work)
    return book


def run_extract(cfg: Config, force: bool = False) -> dict:
    if not cfg.pdf:
        raise SystemExit("extract: --pdf is required")
    work = cfg.work
    ex_dir = work / "extract"
    ex_dir.mkdir(parents=True, exist_ok=True)
    n = page_count(cfg.pdf)
    pages = parse_pages(cfg.pages, n)

    todo = []
    for p in pages:
        rec = runctl.read_json(ex_dir / f"p{p:04d}.json")
        if force or not (rec and rec.get("ok")):
            todo.append(p)
    print(f"[extract] {len(pages)} pages requested, {len(todo)} to process "
          f"({len(pages) - len(todo)} already done)")

    if todo:
        images = render_pages(cfg.pdf, todo, work / "pages", cfg.vlm.dpi, cfg.vlm.max_side)
        client = VLMClient(cfg.vlm)
        failed: list[int] = []

        def job(p: int) -> bool:
            try:
                data = client.extract_page(images[p])
                rec = {"page": p, "ok": True, "data": data}
            except Exception as e:  # noqa: BLE001 - recorded per page, retried on the next run
                rec = {"page": p, "ok": False, "error": str(e)}
            runctl.atomic_write_json(ex_dir / f"p{p:04d}.json", rec, indent=1)
            if not rec["ok"]:
                failed.append(p)
                print(f"[extract] page {p} FAILED: {rec['error'][:200]}")
            return rec["ok"]

        done, _, left = runctl.run_parallel(todo, job, cfg.vlm.workers, label="extract")
        if left:
            print(f"[extract] stopped early: {left} pages not started (they will be picked up next run)")
        if failed:
            print(f"[extract] {len(failed)} pages failed; re-run to retry them: {sorted(failed)}")

    book = rebuild_book(cfg)
    lessons = sorted({v["lesson"] for v in book["vocab"] if v["lesson"] is not None})
    n_ex = sum(len(g["examples"]) for g in book["grammar"])
    span = f"{lessons[0]}..{lessons[-1]}" if lessons else "none detected"
    print(f"[extract] book.json: {len(book['vocab'])} vocab, {len(book['grammar'])} grammar points, "
          f"{n_ex} example sentences, lessons {span}")
    return book
