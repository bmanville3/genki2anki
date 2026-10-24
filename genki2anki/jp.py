"""Japanese text helpers.

* ``normalize_kana``   - any kana text -> comparable hiragana string (long-vowel marks expanded)
* ``to_kana_map``      - mixed kanji/kana text -> (kana, index-of-source-char for each kana char)
* ``kana_alternatives``- split a textbook vocab "kana" field into matchable alternatives
* ``reading_hira``     - display reading of any text, in hiragana (long-vowel marks kept)
* ``furigana_word`` / ``sentence_furigana`` - Anki-style ``漢字[かんじ]`` markup
"""
from __future__ import annotations

import re
import unicodedata

_ROWS = {
    "あ": "あかがさざただなはばぱまやらわぁゃゎ",
    "い": "いきぎしじちぢにひびぴみりぃ",
    "う": "うくぐすずつづぬふぶぷむゆるぅゅゔ",
    "え": "えけげせぜてでねへべぺめれぇ",
    "お": "おこごそぞとどのほぼぽもよろをぉょ",
}
_VOWEL = {ch: v for v, chars in _ROWS.items() for ch in chars}
_DIGITS = str.maketrans("0123456789０１２３４５６７８９", "〇一二三四五六七八九" * 2)
_JP_RE = re.compile(r"[぀-ヿ一-鿿々〆]")
_KANJI_RE = re.compile(r"[一-鿿々〆]")
_SPLIT_RE = re.compile(r"[/／・,、;；]")
_PAREN_RE = re.compile(r"[（(]([^）)]*)[）)]")
_RUN_RE = re.compile(r"([一-鿿々〆]+)|([^一-鿿々〆]+)")


def has_japanese(s: str) -> bool:
    return bool(_JP_RE.search(s or ""))


def has_kanji(s: str) -> bool:
    return bool(_KANJI_RE.search(s or ""))


def kata_to_hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def hira_to_kata(s: str) -> str:
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in s)


def normalize_kana(s: str) -> str:
    """Hiragana-only comparable form. Drops everything that is not kana.

    The long-vowel mark ー is replaced by the vowel of the preceding kana so that
    コーヒー and こうひい-style spellings compare sensibly.
    """
    s = kata_to_hira(unicodedata.normalize("NFKC", s or ""))
    out: list[str] = []
    for ch in s:
        if ch == "ー":
            if out and out[-1] in _VOWEL:
                out.append(_VOWEL[out[-1]])
            continue
        if "ぁ" <= ch <= "ゖ" or ch == "ゔ":
            out.append(ch)
    return "".join(out)


_BACKEND = None


def _backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        from fugashi import Tagger  # type: ignore

        _BACKEND = ("fugashi", Tagger())
        return _BACKEND
    except Exception:
        pass
    try:
        import pykakasi  # type: ignore

        _BACKEND = ("pykakasi", pykakasi.kakasi())
        return _BACKEND
    except Exception:
        pass
    _BACKEND = ("none", None)
    return _BACKEND


def backend_name() -> str:
    return _backend()[0]


def _pieces(t: str) -> list[tuple[int, int, str]]:
    """Tokenise ``t`` into (start, length, reading) pieces; reading is raw kana (any script)."""
    kind, obj = _backend()
    pieces: list[tuple[int, int, str]] = []
    if kind == "fugashi":
        pos = 0
        for w in obj(t):
            s = w.surface
            i = t.find(s, pos)
            if i < 0:
                i = pos
            pos = i + len(s)
            feat = w.feature
            reading = getattr(feat, "kana", None) or getattr(feat, "pron", None)
            if not reading or reading == "*":
                reading = s
            pieces.append((i, len(s), reading))
    elif kind == "pykakasi":
        pos = 0
        for item in obj.convert(t):
            s = item["orig"]
            pieces.append((pos, len(s), item.get("hira") or s))
            pos += len(s)
    else:
        pieces = [(i, 1, c) for i, c in enumerate(t)]
    return pieces


def to_kana_map(text: str) -> tuple[str, list[int]]:
    """Convert text to normalised hiragana.

    Returns ``(kana, src_idx)`` where ``src_idx[k]`` is the index into ``text`` of the
    character that produced ``kana[k]`` (readings longer than their surface are spread
    proportionally). Used to carry ASR word timestamps over to kana characters.
    """
    t = text.translate(_DIGITS)
    kana: list[str] = []
    idx: list[int] = []
    for start, ln, reading in _pieces(t):
        r = normalize_kana(reading)
        if not r or ln <= 0:
            continue
        for k, ch in enumerate(r):
            kana.append(ch)
            idx.append(start + min(ln - 1, (k * ln) // len(r)))
    return "".join(kana), idx


def to_kana(text: str) -> str:
    return to_kana_map(text)[0]


def primary_form(s: str) -> str:
    """First alternative of a textbook form, optional parts kept: "(お)かね/ほか" -> "おかね"."""
    first = _SPLIT_RE.split(s or "")[0]
    return _PAREN_RE.sub(r"\1", first).replace("〜", "").replace("～", "").strip()


speakable = primary_form


def kana_alternatives(kana: str) -> list[str]:
    """Matchable normalised spellings of a textbook vocab reading.

    "(お)かね" -> ["おかね", "かね"];  "ゼロ・れい" -> ["ぜろ", "れい"].
    """
    alts: list[str] = []
    for part in _SPLIT_RE.split(kana or ""):
        base = normalize_kana(_PAREN_RE.sub("", part))
        full = normalize_kana(_PAREN_RE.sub(r"\1", part))
        for a in (full, base):
            if a and a not in alts:
                alts.append(a)
    return alts


# ------------------------------------------------------------------ display readings
def reading_hira(text: str) -> str:
    """Reading of arbitrary text as hiragana (punctuation kept, long-vowel marks kept)."""
    if not text:
        return ""
    out: list[str] = []
    cursor = 0
    for start, ln, reading in _pieces(text):
        if start > cursor:
            out.append(text[cursor:start])
        out.append(kata_to_hira(unicodedata.normalize("NFKC", reading)))
        cursor = start + ln
    out.append(text[cursor:])
    return "".join(out)


def furigana_word(surface: str, reading_hira_: str) -> str:
    """Anki-style furigana for one word, okurigana aware: 食べる + たべる -> ' 食[た]べる'."""
    if not has_kanji(surface):
        return surface
    parts = [(m.group(1) is not None, m.group(0)) for m in _RUN_RE.finditer(surface)]
    pattern = "".join("(.+?)" if is_k else re.escape(kata_to_hira(txt)) for is_k, txt in parts)
    m = re.fullmatch(pattern, reading_hira_)
    out: list[str] = []
    if not m:
        return f" {surface}[{reading_hira_}]"
    gi = 1
    for is_k, txt in parts:
        if is_k:
            out.append(f" {txt}[{m.group(gi)}]")
            gi += 1
        else:
            out.append(txt)
    return "".join(out)


def sentence_furigana(text: str) -> str:
    out: list[str] = []
    cursor = 0
    for start, ln, reading in _pieces(text):
        if start > cursor:
            out.append(text[cursor:start])
        surface = text[start:start + ln]
        if has_kanji(surface):
            out.append(furigana_word(surface, kata_to_hira(unicodedata.normalize("NFKC", reading))))
        else:
            out.append(surface)
        cursor = start + ln
    out.append(text[cursor:])
    return "".join(out).strip()


_KKS = None


def romaji(text: str) -> str:
    """Hepburn romaji of any text (via pykakasi); empty string if unavailable."""
    global _KKS
    if not text:
        return ""
    try:
        if _KKS is None:
            import pykakasi  # type: ignore

            _KKS = pykakasi.kakasi()
        return " ".join(i["hepburn"] for i in _KKS.convert(text) if i.get("hepburn")).strip()
    except Exception:
        return ""


def script_of(s: str) -> str:
    """kanji | hiragana | katakana | mixed | other - the script a string is written in."""
    if has_kanji(s):
        return "kanji"
    hira = bool(re.search(r"[ぁ-ゖ]", s or ""))
    kata = bool(re.search(r"[ァ-ヺヽ-ヿ]", s or ""))
    if hira and kata:
        return "mixed"
    return "hiragana" if hira else "katakana" if kata else "other"
