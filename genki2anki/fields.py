"""Compose CSV rows. Pure functions of: book entry + Jisho record + sentence record + audio.

Every field that was *generated* rather than read from the textbook adds a reason to
``verify_reasons``; any reason makes ``needs_verification`` TRUE.
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Optional

from . import jp
from .config import Config
from .jisho import eng_tokens
from .normalize import final_sentence

VOCAB_COLUMNS = [
    "id", "lesson", "category",
    "front", "front_script", "kanji", "hiragana", "katakana", "furigana", "romaji", "english",
    "needs_verification", "still_needs_review", "verify_reasons", "verified", "notes",
    "jisho_status", "jisho_word", "jisho_reading", "jisho_english", "jisho_common", "jlpt",
    "jisho_url", "jisho_screenshot",
    "page", "page_image", "tb_kanji", "tb_kana", "kanji_source", "reading_source",
    "audio_file", "audio_source", "audio_confidence", "needs_audio_check",
    "audio_sim", "audio_t0", "audio_t1", "audio_src_file",
]

GRAMMAR_COLUMNS = [
    "id", "lesson", "grammar_number", "grammar_title",
    "sentence", "sentence_textbook", "hiragana", "furigana", "romaji", "english",
    "explanation_text", "explanation_html",
    "needs_verification", "still_needs_review", "verify_reasons", "verified", "notes",
    "front_source", "page", "page_image", "other_page_images",
    "audio_file", "audio_source", "audio_confidence", "needs_audio_check",
    "audio_sim", "audio_t0", "audio_t1", "audio_src_file",
]


def b(x: bool) -> str:
    return "TRUE" if x else "FALSE"


def page_image(page: Optional[int]) -> str:
    return f"pages/p{page:04d}.png" if page else ""


def html_to_text(h: str) -> str:
    t = re.sub(r"</(p|li|tr|ul|table)>|<br\s*/?>", "\n", h or "", flags=re.I)
    t = re.sub(r"</t[dh]>", " | ", t, flags=re.I)
    t = html.unescape(re.sub(r"<[^>]+>", "", t))
    t = re.sub(r"[ \t]*\|[ \t]*(?=\n|$)", "", t)
    return re.sub(r"\n\s*\n+", "\n", t).strip()


def audio_columns(entry: Optional[dict], media_dir: Path, cfg: Config) -> dict:
    cols = {"audio_file": "", "audio_source": "none", "audio_confidence": "", "needs_audio_check": "FALSE",
            "audio_sim": "", "audio_t0": "", "audio_t1": "", "audio_src_file": ""}
    if not entry or not (media_dir / entry["file"]).exists():
        return cols
    cols["audio_file"] = f"audio/{entry['file']}"
    cols["audio_source"] = entry.get("source", "")
    if entry.get("source") == "genki":
        good = entry.get("sim", 0) >= cfg.align.confident_similarity and entry.get("tight")
        cols["audio_confidence"] = "high" if good else "low"
        cols["needs_audio_check"] = b(not good)
        cols.update(audio_sim=entry.get("sim", ""), audio_t0=entry.get("t0", ""),
                    audio_t1=entry.get("t1", ""), audio_src_file=Path(entry.get("src_file", "")).name)
    else:
        cols["audio_confidence"] = "tts"
    return cols


def compose_vocab(v: dict, enr: Optional[dict], audio: Optional[dict], media_dir: Path,
                  cfg: Config) -> dict:
    reasons: list[str] = []
    tbk = jp.primary_form(v.get("kanji") or "")
    tbr = jp.primary_form(v.get("kana") or "")

    # ---- reading (hiragana is the canonical reading; katakana words keep their kana spelling)
    if tbr:
        reading, reading_source = tbr, "textbook"
    else:
        reading, reading_source = jp.reading_hira(tbk), "generated"
        reasons.append("reading_generated")
    hira = jp.kata_to_hira(reading)
    kata = jp.hira_to_kata(reading)

    # ---- what Jisho says
    status = "not_checked"
    exact = False
    if enr:
        if enr.get("status") == "not_found":
            status = "not_found"
            reasons.append("jisho_not_found")
        else:
            exact = enr.get("match") == "exact"
            status = {"exact": "exact", "partial": "partial", "none": "no_form_match"}[enr.get("match", "none")]
            if not exact:
                reasons.append("jisho_form_mismatch")
            if eng_tokens(v["english"]) and enr.get("overlap", 1) == 0:
                reasons.append("meaning_mismatch")
    else:
        reasons.append("not_checked")

    # ---- kanji spelling
    kanji, kanji_source = tbk, ("textbook" if tbk else "")
    usually_kana = bool(enr and exact and enr.get("usually_kana"))
    jw = (enr or {}).get("word") or ""
    if not tbk and exact and jp.has_kanji(jw) and not usually_kana:
        kanji, kanji_source = jw, "jisho"
        reasons.append("kanji_from_jisho")

    # ---- front: the most common way to write the word
    if usually_kana or not kanji:
        front = reading
    else:
        front = kanji
    if tbk and usually_kana:
        pass  # textbook printed kanji, but Jisho says it is normally written in kana: front = kana

    furigana = jp.furigana_word(kanji, hira).strip() if kanji else ""
    row = {
        "id": v["id"], "lesson": v.get("lesson") or "", "category": v.get("category") or "",
        "front": front, "front_script": jp.script_of(front), "kanji": kanji,
        "hiragana": hira, "katakana": kata, "furigana": furigana,
        "romaji": jp.romaji(hira), "english": v["english"],
        "needs_verification": b(bool(reasons)), "verify_reasons": ";".join(reasons),
        "jisho_status": status, "jisho_word": jw, "jisho_reading": (enr or {}).get("reading") or "",
        "jisho_english": (enr or {}).get("english") or "",
        "jisho_common": b(bool((enr or {}).get("is_common"))) if enr and enr.get("status") == "ok" else "",
        "jlpt": ",".join((enr or {}).get("jlpt") or []),
        "jisho_url": (enr or {}).get("url") or "",
        "jisho_screenshot": (enr or {}).get("screenshot") or "",
        "page": v.get("page") or "", "page_image": page_image(v.get("page")),
        "tb_kanji": v.get("kanji") or "", "tb_kana": v.get("kana") or "",
        "kanji_source": kanji_source, "reading_source": reading_source,
    }
    row.update(audio_columns(audio, media_dir, cfg))
    return row


def compose_grammar(g: dict, e: dict, norm: Optional[dict], audio: Optional[dict],
                    media_dir: Path, cfg: Config) -> dict:
    reasons: list[str] = []
    front = final_sentence(norm, e["ja"])
    if norm is None:
        front_source = "textbook"
    elif norm.get("status") == "changed":
        front_source = "llm_script_normalized"
        reasons.append("script_changed")
    elif norm.get("status") == "reading_mismatch":
        front_source = "textbook"
        reasons.append("script_unverified")
    else:
        front_source = "textbook"
    hira = jp.reading_hira(front)
    page = e.get("page") or (g.get("pages") or [None])[0]
    others = [page_image(p) for p in g.get("pages", []) if p != page]
    title = g["title"]
    row = {
        "id": e["id"], "lesson": g.get("lesson") or "", "grammar_number": g.get("number") or "",
        "grammar_title": title, "sentence": front, "sentence_textbook": e["ja"],
        "hiragana": hira, "furigana": jp.sentence_furigana(front), "romaji": jp.romaji(hira),
        "english": e["en"], "explanation_text": html_to_text(g["explanation_html"]),
        "explanation_html": g["explanation_html"],
        "needs_verification": b(bool(reasons)), "verify_reasons": ";".join(reasons),
        "front_source": front_source, "page": page or "", "page_image": page_image(page),
        "other_page_images": ";".join(others),
    }
    row.update(audio_columns(audio, media_dir, cfg))
    return row
