"""Shared constants: page kinds, JSON schema for the VLM, stable ids."""
from __future__ import annotations

import hashlib
from typing import Optional

PAGE_KINDS = [
    "vocabulary", "grammar", "dialogue", "exercise", "reading",
    "culture", "index", "front_matter", "other",
]

_NULLABLE_INT = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
_NULLABLE_STR = {"anyOf": [{"type": "string"}, {"type": "null"}]}

PAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "page_kind": {"type": "string", "enum": PAGE_KINDS},
        "lesson": _NULLABLE_INT,
        "vocabulary": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": _NULLABLE_STR,
                    "kana": {"type": "string"},
                    "kanji": _NULLABLE_STR,
                    "english": {"type": "string"},
                },
                "required": ["kana", "kanji", "english", "category"],
            },
        },
        "grammar_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": _NULLABLE_INT,
                    "title": {"type": "string"},
                    "continued_from_previous_page": {"type": "boolean"},
                    "explanation_html": {"type": "string"},
                    "examples": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "japanese": {"type": "string"},
                                "english": {"type": "string"},
                            },
                            "required": ["japanese", "english"],
                        },
                    },
                },
                "required": ["number", "title", "continued_from_previous_page",
                             "explanation_html", "examples"],
            },
        },
    },
    "required": ["page_kind", "lesson", "vocabulary", "grammar_points"],
}


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def vocab_id(lesson: Optional[int], kana: str, kanji: Optional[str]) -> str:
    return "v" + _h(f"{lesson}|{kana}|{kanji or ''}")


def example_id(lesson: Optional[int], ja: str) -> str:
    return "g" + _h(f"{lesson}|{ja}")
