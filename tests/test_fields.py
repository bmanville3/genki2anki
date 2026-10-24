from pathlib import Path

from genki2anki import jp
from genki2anki.config import Config
from genki2anki.fields import compose_grammar, compose_vocab, html_to_text

CFG = Config()
MEDIA = Path("/nonexistent")


def vocab(kana="", kanji=None, english="x", **kw):
    return {"id": "v1", "lesson": 1, "page": 5, "kana": kana, "kanji": kanji, "english": english,
            "category": "Nouns", **kw}


def jisho(word, reading, match="exact", usually_kana=False, overlap=1.0, **kw):
    return {"id": "v1", "status": "ok", "query": word or reading, "match": match, "word": word,
            "reading": reading, "english": "x", "is_common": True, "jlpt": ["N5"],
            "usually_kana": usually_kana, "overlap": overlap, "screenshot": "jisho/v1.png",
            "url": "https://jisho.org/search/x", **kw}


def test_furigana_alignment():
    assert jp.furigana_word("食べる", "たべる") == " 食[た]べる"
    assert jp.furigana_word("学生", "がくせい") == " 学生[がくせい]"
    assert jp.furigana_word("お金", "おかね") == "お 金[かね]"
    assert jp.furigana_word("飲み物", "のみもの") == " 飲[の]み 物[もの]"
    assert jp.furigana_word("ありがとう", "ありがとう") == "ありがとう"
    assert jp.furigana_word("1人", "ひとり") == " 1人[ひとり]"      # unalignable -> whole-word ruby


def test_sentence_furigana_and_reading():
    s = jp.sentence_furigana("私は学生です。")
    assert "私[わたし]" in s and "学生[がくせい]" in s and s.endswith("です。")
    assert jp.reading_hira("コーヒーを飲みます") .startswith("こーひーを")
    assert jp.script_of("コーヒー") == "katakana" and jp.script_of("学生") == "kanji"
    assert jp.script_of("ありがとう") == "hiragana" and jp.script_of("ビルの") == "mixed"


def test_kana_only_textbook_gets_kanji_from_jisho_and_is_flagged():
    r = compose_vocab(vocab(kana="がくせい"), jisho("学生", "がくせい"), None, MEDIA, CFG)
    assert (r["front"], r["front_script"], r["kanji"], r["kanji_source"]) == ("学生", "kanji", "学生", "jisho")
    assert r["hiragana"] == "がくせい" and r["katakana"] == "ガクセイ" and r["romaji"] == "gakusei"
    assert r["furigana"] == "学生[がくせい]"
    assert r["needs_verification"] == "TRUE" and r["verify_reasons"] == "kanji_from_jisho"
    assert r["jisho_screenshot"] == "jisho/v1.png" and r["jlpt"] == "N5"


def test_usually_kana_word_keeps_kana_front_and_is_not_flagged():
    r = compose_vocab(vocab(kana="ありがとう"), jisho("有難う", "ありがとう", usually_kana=True), None, MEDIA, CFG)
    assert r["front"] == "ありがとう" and r["front_script"] == "hiragana" and r["kanji"] == ""
    assert r["needs_verification"] == "FALSE"


def test_textbook_kanji_but_jisho_says_usually_kana_uses_kana_front():
    r = compose_vocab(vocab(kana="いい", kanji="良い"), jisho("良い", "いい", usually_kana=True), None, MEDIA, CFG)
    assert r["front"] == "いい" and r["kanji"] == "良い" and r["kanji_source"] == "textbook"
    assert r["needs_verification"] == "FALSE"


def test_kanji_only_textbook_generates_reading_and_is_flagged():
    r = compose_vocab(vocab(kana="", kanji="学校"), jisho("学校", "がっこう"), None, MEDIA, CFG)
    assert r["front"] == "学校" and r["reading_source"] == "generated"
    assert r["hiragana"] == "がっこう" and r["verify_reasons"] == "reading_generated"
    assert r["needs_verification"] == "TRUE"


def test_both_printed_and_confirmed_needs_nothing():
    r = compose_vocab(vocab(kana="せんせい", kanji="先生"), jisho("先生", "せんせい"), None, MEDIA, CFG)
    assert r["needs_verification"] == "FALSE" and r["jisho_status"] == "exact"


def test_katakana_word():
    r = compose_vocab(vocab(kana="コーヒー"), jisho("コーヒー", "コーヒー"), None, MEDIA, CFG)
    assert r["front"] == "コーヒー" and r["front_script"] == "katakana"
    assert r["hiragana"] == "こーひー" and r["kanji"] == ""


def test_optional_parts_and_alternatives():
    r = compose_vocab(vocab(kana="(お)かね", kanji="(お)金"), jisho("お金", "おかね"), None, MEDIA, CFG)
    assert r["front"] == "お金" and r["hiragana"] == "おかね"


def test_flags_for_missing_or_disagreeing_jisho():
    assert "not_checked" in compose_vocab(vocab(kana="いぬ"), None, None, MEDIA, CFG)["verify_reasons"]
    nf = compose_vocab(vocab(kana="いぬ"), {"id": "v1", "status": "not_found", "match": "none"}, None, MEDIA, CFG)
    assert nf["jisho_status"] == "not_found" and "jisho_not_found" in nf["verify_reasons"]
    r = compose_vocab(vocab(kana="かみ", english="paper"), jisho("髪", "かみ", overlap=0.0), None, MEDIA, CFG)
    assert "meaning_mismatch" in r["verify_reasons"]
    r = compose_vocab(vocab(kana="せんせい", kanji="先性"), jisho("先生", "せんせい", match="partial"), None, MEDIA, CFG)
    assert r["jisho_status"] == "partial" and "jisho_form_mismatch" in r["verify_reasons"]
    assert "kanji_from_jisho" not in r["verify_reasons"]


def test_audio_columns(tmp_path):
    (tmp_path / "a.mp3").write_bytes(b"x" * 10)
    ent = {"file": "a.mp3", "source": "genki", "sim": 0.9, "tight": True, "t0": 1.0, "t1": 2.0, "src_file": "/x/y.mp3"}
    r = compose_vocab(vocab(kana="いぬ"), None, ent, tmp_path, CFG)
    assert r["audio_file"] == "audio/a.mp3" and r["audio_confidence"] == "high" and r["needs_audio_check"] == "FALSE"
    ent["sim"] = 0.75
    assert compose_vocab(vocab(kana="いぬ"), None, ent, tmp_path, CFG)["needs_audio_check"] == "TRUE"
    assert compose_vocab(vocab(kana="いぬ"), None, None, tmp_path, CFG)["audio_source"] == "none"


def test_grammar_row():
    g = {"lesson": 2, "number": 1, "title": "X は Y です", "pages": [7, 8],
         "explanation_html": "<p>A is <b>B</b>.</p><table><tr><td>a</td><td>b</td></tr></table>"}
    e = {"id": "g1", "ja": "わたしはがくせいです。", "en": "I am a student.", "page": 7}
    norm = {"id": "g1", "status": "changed", "rewritten": "私は学生です。"}
    r = compose_grammar(g, e, norm, None, MEDIA, CFG)
    assert r["sentence"] == "私は学生です。" and r["sentence_textbook"] == e["ja"]
    assert r["front_source"] == "llm_script_normalized" and r["verify_reasons"] == "script_changed"
    assert r["page_image"] == "pages/p0007.png" and r["other_page_images"] == "pages/p0008.png"
    assert r["hiragana"].startswith("わたしは") and "私[わたし]" in r["furigana"]
    assert "A is B." in r["explanation_text"] and "a | b" in r["explanation_text"]
    r2 = compose_grammar(g, e, {"id": "g1", "status": "reading_mismatch", "rewritten": "俺は"}, None, MEDIA, CFG)
    assert r2["sentence"] == e["ja"] and r2["verify_reasons"] == "script_unverified"
    r3 = compose_grammar(g, e, None, None, MEDIA, CFG)
    assert r3["needs_verification"] == "FALSE"


def test_html_to_text():
    assert html_to_text("<p>one</p><ul><li>two</li><li>three</li></ul>") == "one\ntwo\nthree"
