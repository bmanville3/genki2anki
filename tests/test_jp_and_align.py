import numpy as np

from genki2anki import jp
from genki2anki.align import LessonAligner, Target, sellers
from genki2anki.config import AlignConfig
from genki2anki.energy import Envelope
from genki2anki.stream import Stream, assign_windows, make_windows

from helpers import fake_asr, write_speechlike_wav


def test_normalize_and_alternatives():
    assert jp.normalize_kana("コーヒー") == "こおひい"
    assert jp.normalize_kana("ｶﾞｸｾｲ") == "がくせい"
    assert jp.kana_alternatives("(お)かね") == ["おかね", "かね"]
    assert jp.kana_alternatives("ゼロ・れい") == ["ぜろ", "れい"]
    assert jp.speakable("(お)かね/ほか") == "おかね"


def test_to_kana_map_offsets():
    kana, idx = jp.to_kana_map("私は学生です")
    assert kana == "わたしはがくせいです"
    assert len(kana) == len(idx) and idx == sorted(idx)
    assert idx[0] == 0 and idx[-1] == 5


def test_sellers_exact_and_fuzzy():
    text = "xxがくせいyyせんせえzz"
    codes = np.array([ord(c) for c in text], dtype=np.int32)
    cost, start = sellers("がくせい", codes)
    e = 2 + 4
    assert cost[e] == 0 and start[e] == 2
    cost, start = sellers("せんせい", codes)
    j = text.index("せんせえ") + 4           # end of the substituted spelling
    assert cost[j] == 1 and text[start[j]:j] == "せんせえ"


def test_assign_windows_monotone():
    lex = {1: {"がくせい", "せんせい"}, 2: {"くるまで", "ともだち"}}
    wins = ["ああがくせいいい", "せんせいxx", "ともだちあ", "くるまでxx"]
    assert assign_windows(wins, lex, True) == [1, 1, 2, 2]
    # a stray lesson-1 hit late in the audio must not drag the path backwards
    wins2 = ["ともだち", "くるまで", "がくせい"]
    path = assign_windows(wins2, lex, True)
    assert path == sorted(path)


def _build(tmp_path, words, noise=()):
    """words: list of (asr_text, kana_expected, start, dur). Returns stream, env, burst times."""
    bursts = [(s, s + d) for _, s, d in [(w[0], w[2], w[3]) for w in words]]
    total = max(b for _, b in bursts) + 2
    wav = tmp_path / "a.wav"
    write_speechlike_wav(wav, bursts, total)
    asr = fake_asr([(w[0], w[2], w[2] + w[3]) for w in words], wav)
    stream = Stream.build([asr])
    env = Envelope.for_wav(wav)
    return stream, env, bursts


def test_aligner_end_to_end(tmp_path):
    # (what ASR heard, start, duration)  - note kanji forms and one substitution error
    heard = [
        ("学生", 1.0, 0.6),
        ("ハロー", 3.0, 0.5),            # English chatter between vocab words
        ("先生", 5.0, 0.6),
        ("ともたち", 7.0, 0.7),          # ASR error: だ -> た
        ("お母さん", 9.0, 0.8),
        ("犬", 11.0, 0.4),
        ("え", 13.0, 0.3),               # 1-kana word, isolated
        ("本", 15.0, 0.4),
        ("車", 17.0, 0.4),
    ]
    words = [(t, None, s, d) for t, s, d in heard]
    stream, env, bursts = _build(tmp_path, [(w[0], None, w[2], w[3]) for w in words])
    targets = [
        Target("a", ["がくせい"], 1), Target("b", ["せんせい"], 1), Target("c", ["ともだち"], 1),
        Target("d", ["おかあさん"], 1), Target("e", ["いぬ"], 1), Target("f", ["え"], 1),
        Target("g", ["ほん"], 1), Target("h", ["くるま"], 1),
        Target("zz", ["ぜったいない"], 1),  # not in the audio
    ]
    al = LessonAligner(stream, np.arange(len(stream)), {0: env}, AlignConfig())
    m = al.align(targets)
    assert "zz" not in m
    want = {"a": 0, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8}
    for tid, bi in want.items():
        assert tid in m, f"{tid} not matched"
        s, e = bursts[bi]
        assert abs(m[tid].t0 - s) < 0.15 and abs(m[tid].t1 - e) < 0.15, (tid, m[tid])
    assert m["c"].sim < 1.0 and m["a"].sim == 1.0
    assert m["f"].isolated and m["f"].tight


def test_short_word_rejected_when_not_isolated(tmp_path):
    # "えんぴつ" spoken as one run; the single kana え inside it must not match target "え"
    stream, env, _ = _build(tmp_path, [("えんぴつ", None, 1.0, 1.0)])
    al = LessonAligner(stream, np.arange(len(stream)), {0: env}, AlignConfig())
    assert al.align([Target("x", ["え"], 1)]) == {}


def test_recovery_handles_swapped_order(tmp_path):
    heard = [("犬", 1.0, 0.5), ("猫", 3.0, 0.5), ("鳥", 5.0, 0.5), ("魚", 7.0, 0.5)]
    stream, env, bursts = _build(tmp_path, [(t, None, s, d) for t, s, d in heard])
    # book order has the middle two swapped relative to the audio
    targets = [Target("1", ["いぬ"], 1), Target("3", ["とり"], 1),
               Target("2", ["ねこ"], 1), Target("4", ["さかな"], 1)]
    al = LessonAligner(stream, np.arange(len(stream)), {0: env}, AlignConfig())
    m = al.align(targets)
    assert set(m) == {"1", "2", "3", "4"}
    assert abs(m["3"].t0 - bursts[2][0]) < 0.15 and abs(m["2"].t0 - bursts[1][0]) < 0.15
    assert any(x.recovered for x in m.values())
