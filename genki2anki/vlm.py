"""Client for a local OpenAI-compatible vision-language model server (vLLM)."""
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import requests

from .config import VLMConfig
from .schema import PAGE_SCHEMA

SYSTEM_PROMPT = (
    "You are a meticulous transcription engine for scanned pages of the Japanese textbook "
    "'Genki: An Integrated Course in Elementary Japanese'. You output only JSON that matches "
    "the requested schema. You transcribe exactly what is printed on the page and never invent "
    "or 'improve' content. If something is not on the page, leave it out."
)

PAGE_PROMPT = """This is one scanned page of the Genki textbook. Extract its content as JSON.

page_kind - classify the page:
  "vocabulary": a vocabulary list (Japanese words with English meanings, grouped under headings such as Nouns, U-verbs, Ru-verbs, Irregular verbs, I-adjectives, Na-adjectives, Adverbs and other expressions, Question words, Counters, Numbers...).
  "grammar": grammar explanation (numbered grammar points with explanations and example sentences).
  "dialogue": the lesson's conversation dialogues.
  "exercise": practice, drills, activities, listening practice, workbook-like tasks.
  "reading": reading and writing sections, kanji lists.
  "culture": culture / expression notes only.
  "index": an appendix, glossary or alphabetical index of words.
  "front_matter": cover, table of contents, preface, title pages, blank pages.
  "other".
If a page mixes vocabulary and grammar, pick the dominant kind but still fill both arrays.

lesson - the lesson number if it is printed on this page (e.g. "Lesson 3", 第3課, running header), else null.

vocabulary - one entry per printed vocabulary item, in reading order (finish one column before the next):
  kana: the reading in hiragana/katakana exactly as printed (no romaji; keep parentheses such as (お)かね and separators such as ・ if printed). If the book prints only a kanji spelling and no kana, use an empty string.
  kanji: the kanji spelling if the book prints one for the item, otherwise null.
  english: the English meaning exactly as printed (keep semicolons, parentheses, 'to ...' for verbs).
  category: the heading the item sits under (e.g. "Nouns", "U-Verbs"), otherwise null.
  Ignore pitch-accent marks/lines. Do not put example sentences here.

grammar_points - one entry per grammar point (or per continuation of one) on the page:
  number: the point's number if printed, else null.
  title: the heading text of the grammar point, e.g. "Noun に Verb (destination)". Empty string if this page only continues a point started on an earlier page.
  continued_from_previous_page: true if the page begins with the continuation of a grammar point whose heading is on an earlier page.
  explanation_html: faithful transcription of the explanatory prose and notes (NOT the example sentences listed in 'examples') as simple HTML using only <p>, <b>, <i>, <ul>, <li>, <br>, <table>, <tr>, <th>, <td>. Keep Japanese in Japanese script and keep the book's English wording. Include conjugation tables and tip/note boxes that belong to the point.
  examples: every example sentence of the point with its English translation. "japanese" holds only the Japanese text (no furigana, romaji, bullets or numbering). For short two-person exchanges join the lines with a space and keep the speaker labels out. "english" is the printed translation.

Rules: transcribe kanji/kana exactly; do not translate or add anything; if a field does not apply use an empty array, empty string or null. Return only the JSON object."""


def encode_image(path: Path) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    suffix = Path(path).suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return f"data:image/{mime};base64,{data}"


def parse_json_loose(text: str) -> dict:
    """Parse JSON that may be wrapped in code fences or surrounded by chatter."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            return json.loads(text[a:b + 1])
        raise


class VLMError(RuntimeError):
    pass


class VLMClient:
    def __init__(self, cfg: VLMConfig):
        self.cfg = cfg
        self.mode = cfg.structured_output

    def _payload(self, messages: list, schema: dict, name: str, mode: str,
                 max_tokens: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema", "json_schema": {"name": name, "schema": schema}}
        elif mode == "guided_json":
            payload["guided_json"] = schema
        return payload

    def _post(self, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        r = requests.post(self.cfg.base_url.rstrip("/") + "/chat/completions",
                          json=payload, headers=headers, timeout=self.cfg.timeout)
        if r.status_code >= 400:
            raise VLMError(f"HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    def complete_json(self, messages: list, schema: dict, name: str, label: str = "request",
                      max_tokens: Optional[int] = None) -> dict:
        """Chat completion that must return a JSON object (retries, structured-output fallback)."""
        last_err: Optional[Exception] = None
        max_tokens = max_tokens or self.cfg.max_tokens
        for attempt in range(self.cfg.retries):
            try:
                resp = self._post(self._payload(messages, schema, name, self.mode, max_tokens))
                choice = resp["choices"][0]
                content = choice["message"]["content"]
                if choice.get("finish_reason") == "length":
                    max_tokens = int(max_tokens * 1.5)
                    raise VLMError("output truncated (finish_reason=length)")
                return parse_json_loose(content)
            except VLMError as e:
                last_err = e
                # server rejected the structured-output option -> fall back to prompt-only JSON
                if "HTTP 400" in str(e) and self.mode != "none":
                    self.mode = "none"
            except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
                last_err = e
            time.sleep(min(2 ** attempt, 8))
        raise VLMError(f"giving up on {label}: {last_err}")

    def extract_page(self, image_path: Path) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
                {"type": "text", "text": PAGE_PROMPT},
            ]},
        ]
        return self.complete_json(messages, PAGE_SCHEMA, "genki_page", label=Path(image_path).name)
