import json
from pathlib import Path

import pytest

from genki2anki.config import load_config
from genki2anki.extract import merge_pages, normalize_page, run_extract, sanitize_html
from genki2anki.pdf import parse_pages

from helpers import start_server

PAGES = [
    # p1: vocabulary page, lesson printed
    {"page_kind": "vocabulary", "lesson": 1, "grammar_points": [], "vocabulary": [
        {"category": "Nouns", "kana": "がくせい", "kanji": "学生", "english": "student"},
        {"category": "Nouns", "kana": "せんせい", "kanji": None, "english": "teacher"},
        {"category": "Nouns", "kana": "学校", "kanji": None, "english": "school"},  # kanji in kana field
        {"category": "Nouns", "kana": "せんせい", "kanji": None, "english": "teacher"},  # duplicate
    ]},
    # p2: grammar, lesson number not printed on this page
    {"page_kind": "grammar", "lesson": None, "vocabulary": [], "grammar_points": [{
        "number": 1, "title": "X は Y です", "continued_from_previous_page": False,
        "explanation_html": "<p onclick='x()'>Use <b>です</b> to say <script>bad()</script>A is B.</p>",
        "examples": [{"japanese": "1. 私は学生です。", "english": "I am a student."}]}]},
    # p3: continuation of that grammar point
    {"page_kind": "grammar", "lesson": None, "vocabulary": [], "grammar_points": [{
        "number": None, "title": "", "continued_from_previous_page": True,
        "explanation_html": "<p>More notes.</p>",
        "examples": [{"japanese": "田中さんは先生です。", "english": "Mr. Tanaka is a teacher."},
                     {"japanese": "no japanese here", "english": "x"}]}]},
    # p4: exercise page whose content must be ignored
    {"page_kind": "exercise", "lesson": None, "grammar_points": [], "vocabulary": [
        {"category": None, "kana": "ほん", "kanji": None, "english": "book"}]},
]


def make_pdf(path: Path, n: int):
    import pymupdf
    doc = pymupdf.open()
    for i in range(n):
        page = doc.new_page(width=300, height=400)
        page.insert_text((40, 80), f"page {i + 1}")
    doc.save(str(path))


def vlm_respond(counter):
    def respond(path, payload):
        if not path.endswith("/chat/completions"):
            return 404, {}
        content = payload["messages"][1]["content"]
        assert content[0]["type"] == "image_url" and content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert payload["response_format"]["type"] == "json_schema"
        i = counter["n"]
        counter["n"] += 1
        return 200, {"choices": [{"index": 0, "finish_reason": "stop",
                                  "message": {"role": "assistant",
                                              "content": "```json\n" + json.dumps(PAGES[i], ensure_ascii=False) + "\n```"}}]}
    return respond


def test_parse_pages():
    assert parse_pages("2-4,9", 20) == [2, 3, 4, 9]
    assert parse_pages(None, 3) == [1, 2, 3]
    assert parse_pages("18-", 20) == [18, 19, 20]
    assert parse_pages("1-99", 5) == [1, 2, 3, 4, 5]


def test_sanitize_html():
    out = sanitize_html("<p class='a' onclick='x()'>hi <b>there</b><script>evil()</script><img src=x><br/></p>")
    assert out == "<p>hi <b>there</b><br></p>"


def test_extract_end_to_end(tmp_path):
    pdf = tmp_path / "genki.pdf"
    make_pdf(pdf, 4)
    counter = {"n": 0}
    srv, url = start_server(vlm_respond(counter))
    cfg = load_config(None, {"work_dir": str(tmp_path / "work"), "pdf": str(pdf),
                             "vlm.base_url": url + "/v1", "vlm.workers": 1, "vlm.dpi": 60})
    book = run_extract(cfg)
    srv.shutdown()
    assert counter["n"] == 4
    kanas = [(v["lesson"], v["kana"], v["kanji"]) for v in book["vocab"]]
    # a kanji printed in the kana field becomes the kanji spelling; the reading is generated later
    assert kanas == [(1, "がくせい", "学生"), (1, "せんせい", None), (1, "", "学校")]
    assert not any(v["english"] == "book" for v in book["vocab"])  # exercise page skipped
    assert len(book["grammar"]) == 1
    g = book["grammar"][0]
    assert g["lesson"] == 1 and g["pages"] == [2, 3] and g["title"] == "X は Y です"
    assert "<script" not in g["explanation_html"] and "onclick" not in g["explanation_html"]
    assert "More notes." in g["explanation_html"]
    assert [e["ja"] for e in g["examples"]] == ["私は学生です。", "田中さんは先生です。"]
    # cached re-run makes no new requests
    srv2, url2 = start_server(vlm_respond({"n": 0}))
    cfg.vlm.base_url = url2 + "/v1"
    run_extract(cfg)
    assert srv2.requests == []
    srv2.shutdown()
    assert (cfg.work / "book_vocab.tsv").exists() and (cfg.work / "book.json").exists()


def test_vlm_falls_back_when_schema_rejected(tmp_path):
    pdf = tmp_path / "g.pdf"
    make_pdf(pdf, 1)
    state = {"n": 0}

    def respond(path, payload):
        state["n"] += 1
        if "response_format" in payload:
            return 400, {"error": "json_schema unsupported"}
        return 200, {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(PAGES[0])}}]}

    srv, url = start_server(respond)
    cfg = load_config(None, {"work_dir": str(tmp_path / "w"), "pdf": str(pdf),
                             "vlm.base_url": url + "/v1", "vlm.workers": 1, "vlm.dpi": 60})
    book = run_extract(cfg)
    srv.shutdown()
    assert len(book["vocab"]) == 3 and state["n"] == 2


def test_merge_keeps_grammar_separate_across_lessons():
    pg = lambda les, title, cont: (1, {"kind": "grammar", "lesson": les, "vocab": [], "grammar": [
        {"number": 1, "title": title, "continued": cont, "explanation_html": "<p>x</p>",
         "examples": [{"ja": f"例{les}。", "en": "ex"}]}]})
    a, b = pg(1, "A", False), (2, pg(2, "", True)[1])
    book = merge_pages([a, b], [])
    assert len(book["grammar"]) == 2        # continuation must not leak into another lesson
