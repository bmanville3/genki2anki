import csv
import json
import shutil
from pathlib import Path

import pytest

from genki2anki import runctl
from genki2anki.config import load_config
from genki2anki.enrich import load_enrich, run_enrich
from genki2anki.export import run_export
from genki2anki.jisho import eng_tokens, pick_entry, summarize
from genki2anki.normalize import classify, run_normalize
from helpers import start_server

CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"


def entry(slug, forms, defs, common=True, tags=None, jlpt=None):
    return {"slug": slug, "is_common": common, "jlpt": jlpt or [],
            "japanese": [{"word": w, "reading": r} if w else {"reading": r} for w, r in forms],
            "senses": [{"english_definitions": defs, "parts_of_speech": ["Noun"], "tags": tags or []}]}


DB = {
    "学生": [entry("学生", [("学生", "がくせい")], ["student (esp. a university student)"], jlpt=["jlpt-n5"])],
    "がくせい": [entry("学生", [("学生", "がくせい")], ["student"], jlpt=["jlpt-n5"])],
    "かみ": [entry("神", [("神", "かみ")], ["god"]), entry("紙", [("紙", "かみ")], ["paper"]),
            entry("髪", [("髪", "かみ")], ["hair"])],
    "ありがとう": [entry("ありがとう", [("有難う", "ありがとう"), (None, "ありがとう")], ["thank you"],
                     tags=["Usually written using kana alone"])],
    "せんせい": [entry("先生", [("先生", "せんせい")], ["teacher; master"])],
}


def jisho_server(log=None, on_call=None):
    def respond(path, payload):
        if log is not None:
            log.append(path)
        if on_call:
            on_call(path)
        if path.startswith("/api?keyword="):
            kw = path.split("=", 1)[1]
            return 200, {"meta": {"status": 200}, "data": DB.get(kw, [])}
        if path.startswith("/search/"):
            kw = path[len("/search/"):]
            blocks = "".join(
                f'<div class="concept_light clearfix"><span class="furigana">{e["japanese"][0]["reading"]}</span>'
                f'<span class="text">{e["japanese"][0].get("word") or ""}</span>'
                f'<div class="meanings">{"; ".join(e["senses"][0]["english_definitions"])}</div></div>'
                for e in DB.get(kw, []))
            return 200, f"<html><body><div id='primary'>{blocks or '<div id=no-matches>none</div>'}</div></body></html>"
        return 404, {}
    return start_server(respond)


def test_pick_entry_uses_english_to_choose_between_homophones():
    p = pick_entry(DB["かみ"], "", "かみ", "paper")
    assert p.match == "exact" and p.entry["slug"] == "紙" and p.index == 1 and p.overlap == 1.0
    assert summarize(p)["word"] == "紙"
    s = summarize(pick_entry(DB["ありがとう"], "", "ありがとう", "thank you"))
    assert s["usually_kana"] is True
    n = pick_entry(DB["かみ"], "", "かみ", "banana")
    assert n.overlap == 0
    assert pick_entry([], "犬", "いぬ", "dog").entry is None
    assert pick_entry(DB["学生"], "学校", "", "school").match == "none"
    assert eng_tokens("to eat; (a) meals") == {"eat", "meal"}


def book_with(vocab):
    return {"vocab": vocab, "grammar": []}


def setup_book(tmp_path, vocab, **overrides):
    cfg = load_config(None, {"work_dir": str(tmp_path / "w"), "out_dir": str(tmp_path / "out"),
                             "jisho.screenshots": False, "jisho.delay_s": 0, **overrides})
    cfg.work.mkdir(parents=True, exist_ok=True)
    runctl.atomic_write_json(cfg.work / "book.json", book_with(vocab))
    return cfg


def V(i, kana, kanji, en, page=1):
    return {"id": i, "lesson": 1, "page": page, "kana": kana, "kanji": kanji, "english": en, "category": "Nouns"}


def test_enrich_resumes_without_repeating_lookups(tmp_path):
    vocab = [V("a", "がくせい", None, "student"), V("b", "かみ", None, "paper"),
             V("c", "ありがとう", None, "thank you"), V("d", "せんせい", "先生", "teacher")]
    log = []

    def on_call(path):
        if path.endswith("=かみ"):
            runctl.STOP.set()                           # interrupted while looking up word 2

    srv, url = jisho_server(log, on_call)
    cfg = setup_book(tmp_path, vocab, **{"jisho.api_url": url + "/api", "jisho.web_url": url + "/search"})
    run_enrich(cfg)
    done_first = set(load_enrich(cfg))
    assert "a" in done_first and "d" not in done_first
    runctl.STOP.clear()
    api_calls_first = len([p for p in log if p.startswith("/api")])
    run_enrich(cfg)
    srv.shutdown()
    recs = load_enrich(cfg)
    assert set(recs) == {"a", "b", "c", "d"}
    api = [p for p in log if p.startswith("/api")]
    assert len(api) == len(set(api))                    # no keyword was fetched twice
    assert recs["b"]["word"] == "紙" and recs["c"]["usually_kana"] is True
    assert recs["a"]["word"] == "学生" and recs["a"]["jlpt"] == ["N5"]
    assert api_calls_first < len(api)
    # third run: everything is cached, nothing is requested
    n = len(log)
    srv2, url2 = jisho_server(log)
    cfg.jisho.api_url = url2 + "/api"
    run_enrich(cfg)
    srv2.shutdown()
    assert len(log) == n


def test_enrich_gives_up_politely_when_jisho_is_unreachable(tmp_path, capsys):
    vocab = [V(str(i), "いぬ", None, "dog") for i in range(8)]
    cfg = setup_book(tmp_path, vocab, **{"jisho.api_url": "http://127.0.0.1:9/api", "jisho.timeout": 1,
                                         "jisho.max_consecutive_errors": 3})
    run_enrich(cfg)
    out = capsys.readouterr().out
    assert "is jisho.org reachable" in out and load_enrich(cfg) == {}


@pytest.mark.skipif(not Path(CHROME).exists(), reason="chromium not available")
def test_enrich_takes_screenshots_and_retakes_missing_ones(tmp_path):
    pytest.importorskip("playwright")
    vocab = [V("a", "かみ", None, "paper")]
    srv, url = jisho_server()
    cfg = setup_book(tmp_path, vocab, **{"jisho.api_url": url + "/api", "jisho.web_url": url + "/search",
                                         "jisho.screenshots": True, "jisho.chromium_path": CHROME})
    run_enrich(cfg)
    rec = load_enrich(cfg)["a"]
    shot = cfg.work / rec["screenshot"]
    assert shot.exists() and shot.stat().st_size > 500 and shot.read_bytes()[:4] == b"\x89PNG"
    assert rec["url"].endswith("/search/%E3%81%8B%E3%81%BF") or rec["url"].endswith("/search/かみ")
    shot.unlink()
    run_enrich(cfg)                                      # screenshot is missing -> only it is redone
    srv.shutdown()
    assert shot.exists()


def test_classify_sentence_rewrites():
    assert classify("わたしはがくせいです。", "私は学生です。") == ("changed", "私は学生です。")
    assert classify("わたしはがくせいです。", "わたしはがくせいです。")[0] == "unchanged"
    assert classify("わたしはがくせいです。", "私は先生です。") == ("reading_mismatch", "わたしはがくせいです。")
    assert classify("わたしはがくせいです。", "")[0] == "unchanged"


def test_normalize_stage_with_local_llm(tmp_path):
    book = {"vocab": [], "grammar": [{"lesson": 1, "number": 1, "title": "t", "explanation_html": "", "pages": [3],
                                      "examples": [{"id": "g1", "ja": "わたしはがくせいです。", "en": "I am a student.", "page": 3},
                                                   {"id": "g2", "ja": "あのひとはせんせいです。", "en": "That person is a teacher.", "page": 3},
                                                   {"id": "g3", "ja": "これはほんです。", "en": "This is a book.", "page": 3}]}]}
    answers = {"わたしはがくせいです。": "私は学生です。", "あのひとはせんせいです。": "あの人は医者です。",
               "これはほんです。": "これはほんです。"}

    def respond(path, payload):
        user = payload["messages"][1]["content"]
        ja = [l for l in user.splitlines() if l.startswith("Japanese: ")][0][len("Japanese: "):]
        return 200, {"choices": [{"finish_reason": "stop",
                                  "message": {"content": json.dumps({"sentence": answers[ja]}, ensure_ascii=False)}}]}

    srv, url = start_server(respond)
    cfg = load_config(None, {"work_dir": str(tmp_path / "w"), "vlm.base_url": url + "/v1", "vlm.workers": 2})
    runctl.atomic_write_json(cfg.work / "book.json", book)
    run_normalize(cfg)
    n_requests = len(srv.requests)
    run_normalize(cfg)                                   # resume: nothing left to do
    srv.shutdown()
    assert len(srv.requests) == n_requests == 3
    from genki2anki.normalize import load_norm
    n = load_norm(cfg)
    assert n["g1"]["status"] == "changed" and n["g1"]["used"] == "私は学生です。"
    assert n["g2"]["status"] == "reading_mismatch" and n["g2"]["used"] == "あのひとはせんせいです。"
    assert n["g3"]["status"] == "unchanged"


def test_export_csv_files_flags_and_preserved_review(tmp_path):
    vocab = [V("a", "がくせい", None, "student", page=1), V("b", "ありがとう", None, "thank you", page=2),
             V("c", "", "学校", "school", page=2)]
    cfg = setup_book(tmp_path, vocab)
    book = runctl.read_json(cfg.work / "book.json")
    book["grammar"] = [{"lesson": 1, "number": 1, "title": "X は Y です", "pages": [3, 4],
                        "explanation_html": "<p>A is B.</p>",
                        "examples": [{"id": "g1", "ja": "わたしはがくせいです。", "en": "I am a student.", "page": 3}]}]
    runctl.atomic_write_json(cfg.work / "book.json", book)
    for p in (1, 2, 3, 4):
        (cfg.work / "pages").mkdir(exist_ok=True)
        (cfg.work / "pages" / f"p{p:04d}.png").write_bytes(b"\x89PNG" + bytes([p]) * 20)
    (cfg.work / "jisho").mkdir()
    (cfg.work / "jisho" / "a.png").write_bytes(b"\x89PNGshot")
    runctl.atomic_write_json(cfg.work / "enrich" / "a.json", {
        "id": "a", "status": "ok", "match": "exact", "word": "学生", "reading": "がくせい", "english": "student",
        "is_common": True, "jlpt": ["N5"], "usually_kana": False, "overlap": 1.0, "query": "がくせい",
        "url": "https://jisho.org/search/がくせい", "screenshot": "jisho/a.png"})
    runctl.atomic_write_json(cfg.work / "enrich" / "b.json", {
        "id": "b", "status": "ok", "match": "exact", "word": "有難う", "reading": "ありがとう", "english": "thanks",
        "is_common": True, "jlpt": [], "usually_kana": True, "overlap": 1.0, "query": "ありがとう"})
    runctl.atomic_write_json(cfg.work / "normalize" / "g1.json", {
        "id": "g1", "original": "わたしはがくせいです。", "rewritten": "私は学生です。", "status": "changed",
        "used": "私は学生です。"})
    (cfg.work / "media").mkdir()
    (cfg.work / "media" / "genki_a.mp3").write_bytes(b"ID3" + b"0" * 50)
    runctl.atomic_write_json(cfg.work / "tts_manifest.json", {"a": {"file": "genki_a.mp3", "source": "tts"}})

    out = run_export(cfg)
    rows = list(csv.DictReader(open(out / "vocab.csv", encoding="utf-8-sig")))
    by = {r["id"]: r for r in rows}
    assert by["a"]["front"] == "学生" and by["a"]["kanji_source"] == "jisho"
    assert by["a"]["needs_verification"] == "TRUE" and by["a"]["still_needs_review"] == "TRUE"
    assert by["a"]["page_image"] == "pages/p0001.png" and (out / "pages" / "p0001.png").exists()
    assert by["a"]["jisho_screenshot"] == "jisho/a.png" and (out / "jisho" / "a.png").exists()
    assert by["a"]["audio_file"] == "audio/genki_a.mp3" and (out / "audio" / "genki_a.mp3").exists()
    assert by["b"]["front"] == "ありがとう" and by["b"]["needs_verification"] == "FALSE"
    assert by["c"]["jisho_status"] == "not_checked" and "reading_generated" in by["c"]["verify_reasons"]
    assert by["c"]["hiragana"] == "がっこう" and by["c"]["audio_source"] == "none"

    g = list(csv.DictReader(open(out / "grammar.csv", encoding="utf-8-sig")))
    assert len(g) == 1 and g[0]["sentence"] == "私は学生です。" and g[0]["english"] == "I am a student."
    assert g[0]["page_image"] == "pages/p0003.png" and g[0]["other_page_images"] == "pages/p0004.png"
    assert (out / "pages" / "p0004.png").exists()

    # the user reviews row "a" in the CSV; a re-export must not lose that
    text = (out / "vocab.csv").read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    head = lines[0].split(",")
    r = list(csv.reader(lines))
    ia, iv, inotes = head.index("id"), head.index("verified"), head.index("notes")
    for row in r[1:]:
        if row[ia] == "a":
            row[iv], row[inotes] = "TRUE", "checked on jisho"
    with open(out / "vocab.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(r)
    run_export(cfg)
    by2 = {x["id"]: x for x in csv.DictReader(open(out / "vocab.csv", encoding="utf-8-sig"))}
    assert by2["a"]["verified"] == "TRUE" and by2["a"]["notes"] == "checked on jisho"
    assert by2["a"]["needs_verification"] == "TRUE" and by2["a"]["still_needs_review"] == "FALSE"
    assert by2["c"]["still_needs_review"] == "TRUE"


@pytest.mark.skipif(not Path(CHROME).exists(), reason="chromium not available")
def test_viewer_renders_and_filters(tmp_path):
    pytest.importorskip("playwright")
    from playwright.sync_api import sync_playwright
    test_export_csv_files_flags_and_preserved_review(tmp_path)
    html = tmp_path / "out" / "viewer.html"
    assert html.exists()
    errs = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=CHROME)
        pg = b.new_page()
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(html.as_uri())
        assert pg.locator(".row").count() == 3
        pg.select_option("#f", "review")
        assert pg.locator(".row").count() == 1
        pg.click("#tg")
        assert pg.locator(".row").count() == 1 and "私は学生です。" in pg.inner_text("#m")
        b.close()
    assert not errs
