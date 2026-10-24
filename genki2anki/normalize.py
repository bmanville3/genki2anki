"""Stage: rewrite example sentences in their most common orthography (local LLM), safely.

The book often prints words in kana whose everyday spelling uses kanji (or the reverse).
The model is only allowed to change the *script*; the result is accepted only if its
reading (kana) is identical to the original sentence's reading, which is checked in code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import jp, runctl
from .audio import load_book
from .config import Config
from .vlm import VLMClient

SCHEMA = {"type": "object", "properties": {"sentence": {"type": "string"}}, "required": ["sentence"]}

SYSTEM = (
    "You normalise the orthography of Japanese sentences for a dictionary-quality study deck. "
    "You output only JSON."
)

PROMPT = """Rewrite this Japanese sentence using the most common, standard everyday orthography:
- kanji where Japanese normally uses kanji (even if the textbook printed kana),
- hiragana for particles, inflections and words normally written in kana,
- katakana for loanwords and words normally written in katakana.
Keep wording, word order, politeness level, grammar and punctuation EXACTLY the same. Do not
translate, correct, shorten or add anything. If it is already in standard orthography, return it
unchanged. The English meaning is given only to disambiguate homophones.

Japanese: {ja}
English: {en}

Return JSON: {{"sentence": "<the rewritten Japanese sentence>"}}"""


def norm_path(cfg: Config, eid: str) -> Path:
    return cfg.work / "normalize" / f"{eid}.json"


def load_norm(cfg: Config) -> dict[str, dict]:
    out = {}
    for p in (cfg.work / "normalize").glob("*.json"):
        d = runctl.read_json(p)
        if d:
            out[d["id"]] = d
    return out


def classify(original: str, rewritten: Optional[str]) -> tuple[str, str]:
    """-> (status, sentence_to_use): unchanged | changed | reading_mismatch."""
    new = (rewritten or "").strip()
    if not new or not jp.has_japanese(new) or new == original:
        return "unchanged", original
    if jp.to_kana(new) == jp.to_kana(original):
        return "changed", new
    return "reading_mismatch", original


def final_sentence(rec: Optional[dict], original: str) -> str:
    if rec and rec.get("status") == "changed" and rec.get("rewritten"):
        return rec["rewritten"]
    return original


def run_normalize(cfg: Config, force: bool = False) -> None:
    if not cfg.normalize.enabled:
        print("[normalize] disabled")
        return
    book = load_book(cfg)
    examples = {}
    for g in book["grammar"]:
        for e in g["examples"]:
            examples[e["id"]] = e
    todo = [e for eid, e in examples.items()
            if force or runctl.read_json(norm_path(cfg, eid)) is None]
    print(f"[normalize] {len(examples)} sentences, {len(todo)} to process")
    if not todo:
        return
    client = VLMClient(cfg.vlm)

    def job(e: dict) -> str:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT.format(ja=e["ja"], en=e["en"])}]
        res = client.complete_json(msgs, SCHEMA, "sentence", label=e["id"], max_tokens=400)
        status, used = classify(e["ja"], res.get("sentence"))
        runctl.atomic_write_json(norm_path(cfg, e["id"]), {
            "id": e["id"], "original": e["ja"], "rewritten": res.get("sentence"),
            "status": status, "used": used})
        return status

    stats: dict[str, int] = {}
    errors = []

    def on_result(item, res, err):
        if err is not None:
            errors.append(f"{item['id']}: {err}")
        else:
            stats[res] = stats.get(res, 0) + 1

    _, failed, left = runctl.run_parallel(todo, job, cfg.normalize.workers, on_result,
                                          label="normalize", every=50)
    print(f"[normalize] results: {stats}" + (f"; {failed} requests failed (retry on next run)" if failed else ""))
    for line in errors[:3]:
        print(f"[normalize]   {line[:200]}")
    if left:
        print(f"[normalize] stopped early: {left} sentences not started")
