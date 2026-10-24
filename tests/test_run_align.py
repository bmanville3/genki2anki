import json
import shutil

import pytest

from genki2anki.audio import load_manifest, run_align
from genki2anki.config import load_config
from helpers import fake_asr, write_speechlike_wav


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg needed")
def test_run_align_two_lessons_one_file(tmp_path):
    # one continuous file: lesson 1 vocab, long pause, lesson 2 vocab
    L1 = [("学生", "がくせい"), ("先生", "せんせい"), ("友達", "ともだち"), ("お母さん", "おかあさん")]
    L2 = [("車", "くるま"), ("電話", "でんわ"), ("新聞", "しんぶん"), ("手紙", "てがみ")]
    words, bursts, t = [], [], 1.0
    for text, _ in L1 + L2:
        words.append((text, t, t + 0.6))
        bursts.append((t, t + 0.6))
        t += 2.0
    wav = tmp_path / "audio" / "all.wav"
    write_speechlike_wav(wav, bursts, t + 1)
    work = tmp_path / "work"
    (work / "asr" / "wav").mkdir(parents=True)
    (work / "asr" / "all.json").write_text(json.dumps(fake_asr(words, wav)), encoding="utf-8")
    (work / "asr" / "wav" / "all.wav").write_bytes(wav.read_bytes())
    asr = json.loads((work / "asr" / "all.json").read_text())
    asr["wav"] = str(work / "asr" / "wav" / "all.wav")
    (work / "asr" / "all.json").write_text(json.dumps(asr), encoding="utf-8")

    vocab = []
    for les, group in ((1, L1), (2, L2)):
        for i, (_, kana) in enumerate(group):
            vocab.append({"id": f"v{les}{i}", "lesson": les, "page": 1, "kana": kana,
                          "kanji": None, "english": "x", "category": None})
    vocab.append({"id": "vmissing", "lesson": 2, "page": 1, "kana": "ぜったいない", "kanji": None,
                  "english": "x", "category": None})
    (work / "book.json").write_text(json.dumps({"vocab": vocab, "grammar": []}), encoding="utf-8")

    cfg = load_config(None, {"work_dir": str(work), "align.window_chars": 12, "align.clip_workers": 2})
    try:
        man = run_align(cfg)
    except Exception as e:  # ffmpeg without mp3 encoder
        pytest.skip(str(e))
    assert set(man) == {f"v{l}{i}" for l in (1, 2) for i in range(4)}
    assert "vmissing" not in man
    assert (work / "media" / man["v10"]["file"]).exists()
    assert abs(man["v23"]["t0"] - (bursts[7][0] - 0.06)) < 0.2
    rep = (work / "report" / "alignment.tsv").read_text(encoding="utf-8")
    assert "vmissing" in rep and rep.count("\n") == 10


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg needed")
def test_align_is_checkpointed_per_lesson_and_resumes(tmp_path, monkeypatch):
    from genki2anki import audio, runctl

    L1 = [("学生", "がくせい"), ("先生", "せんせい"), ("友達", "ともだち")]
    L2 = [("車", "くるま"), ("電話", "でんわ"), ("新聞", "しんぶん")]
    words, bursts, t = [], [], 1.0
    for text, _ in L1 + L2:
        words.append((text, t, t + 0.6)); bursts.append((t, t + 0.6)); t += 2.0
    work = tmp_path / "work"
    wav = work / "asr" / "wav" / "all.wav"
    write_speechlike_wav(wav, bursts, t + 1)
    (work / "asr").mkdir(exist_ok=True)
    (work / "asr" / "all.json").write_text(json.dumps(fake_asr(words, wav)), encoding="utf-8")
    vocab = [{"id": f"v{l}{i}", "lesson": l, "page": 1, "kana": k, "kanji": None, "english": "x", "category": None}
             for l, g in ((1, L1), (2, L2)) for i, (_, k) in enumerate(g)]
    (work / "book.json").write_text(json.dumps({"vocab": vocab, "grammar": []}), encoding="utf-8")
    cfg = load_config(None, {"work_dir": str(work), "align.window_chars": 9, "align.clip_workers": 2})

    real_cut_all = audio.cut_all
    calls = []

    def stopping_cut_all(*a, **k):
        calls.append(1)
        out = real_cut_all(*a, **k)
        runctl.STOP.set()                      # interrupted right after lesson 1 is saved
        return out

    monkeypatch.setattr(audio, "cut_all", stopping_cut_all)
    try:
        audio.run_align(cfg)
    except Exception as e:  # no mp3 encoder in this ffmpeg
        pytest.skip(str(e))
    assert (work / "align" / "lesson_01.json").exists() and not (work / "align" / "lesson_02.json").exists()
    l1_mtime = (work / "align" / "lesson_01.json").stat().st_mtime_ns
    clips1 = {p.name: p.stat().st_mtime_ns for p in (work / "media").glob("*.mp3")}

    runctl.STOP.clear()
    monkeypatch.setattr(audio, "cut_all", lambda *a, **k: (calls.append(1), real_cut_all(*a, **k))[1])
    man = audio.run_align(cfg)
    assert len(calls) == 2                      # lesson 1 was NOT redone on resume
    assert (work / "align" / "lesson_01.json").stat().st_mtime_ns == l1_mtime
    assert all((work / "media" / n).stat().st_mtime_ns == m for n, m in clips1.items())
    assert set(man) == {f"v{l}{i}" for l in (1, 2) for i in range(3)}
    assert (work / "report" / "alignment.tsv").read_text(encoding="utf-8").count("\n") == 7
