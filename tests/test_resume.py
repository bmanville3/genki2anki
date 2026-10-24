"""Stop/restart behaviour of every long-running stage."""
import json
import sys
import types
from pathlib import Path

import pytest

from genki2anki import runctl
from genki2anki.config import load_config
from genki2anki.extract import run_extract
from helpers import start_server, write_speechlike_wav
from test_pipeline import PAGES, make_pdf


def test_atomic_write_and_corrupt_read(tmp_path):
    p = tmp_path / "a" / "x.json"
    runctl.atomic_write_json(p, {"k": "日本語"})
    assert json.loads(p.read_text(encoding="utf-8")) == {"k": "日本語"}
    assert [f.name for f in p.parent.iterdir()] == ["x.json"]          # no temp file left
    p.write_text('{"k": "trunc', encoding="utf-8")                      # simulated torn write
    assert runctl.read_json(p, "fallback") == "fallback"
    assert runctl.read_json(tmp_path / "missing.json") is None


def test_run_parallel_stops_submitting_but_finishes_inflight():
    seen = []

    def fn(i):
        seen.append(i)
        if i == 3:
            runctl.STOP.set()
        return i

    done, failed, left = runctl.run_parallel(range(50), fn, workers=1)
    assert done == len(seen) and left == 50 - done and left > 40 and failed == 0


def test_run_parallel_reports_errors_without_aborting():
    def fn(i):
        if i % 2:
            raise ValueError(i)
        return i

    errs = []
    done, failed, left = runctl.run_parallel(range(6), fn, 2, lambda it, r, e: errs.append(e))
    assert (done, failed, left) == (6, 3, 0) and sum(e is not None for e in errs) == 3


def test_extract_stop_then_resume_processes_each_page_once(tmp_path):
    pdf = tmp_path / "g.pdf"
    make_pdf(pdf, 4)
    state = {"n": 0}

    def respond(path, payload):
        i = state["n"]
        state["n"] += 1
        if state["n"] == 2:
            runctl.STOP.set()                      # "Ctrl-C" arrives while page 2 is in flight
        return 200, {"choices": [{"finish_reason": "stop",
                                  "message": {"content": json.dumps(PAGES[i])}}]}

    srv, url = start_server(respond)
    cfg = load_config(None, {"work_dir": str(tmp_path / "w"), "pdf": str(pdf), "vlm.base_url": url + "/v1",
                             "vlm.workers": 1, "vlm.dpi": 60})
    run_extract(cfg)
    first = state["n"]
    assert 2 <= first < 4
    assert len(list((cfg.work / "extract").glob("p*.json"))) == first
    runctl.STOP.clear()
    book = run_extract(cfg)
    srv.shutdown()
    assert state["n"] == 4                              # every page requested exactly once overall
    assert len(book["vocab"]) == 3 and len(book["grammar"]) == 1


def test_extract_retries_failed_pages_only(tmp_path):
    pdf = tmp_path / "g.pdf"
    make_pdf(pdf, 2)
    calls = []

    def respond(path, payload):
        calls.append(1)
        if len(calls) == 2:
            return 500, {"error": "boom"}
        return 200, {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(PAGES[0])}}]}

    srv, url = start_server(respond)
    cfg = load_config(None, {"work_dir": str(tmp_path / "w"), "pdf": str(pdf), "vlm.base_url": url + "/v1",
                             "vlm.workers": 1, "vlm.dpi": 60, "vlm.retries": 1})
    run_extract(cfg)
    oks = sorted(json.loads(p.read_text())["ok"] for p in (cfg.work / "extract").glob("p*.json"))
    assert oks == [False, True]
    n_before = len(calls)
    run_extract(cfg)
    srv.shutdown()
    assert len(calls) == n_before + 1                    # only the failed page was retried


def _install_fake_whisper(fail_after=None):
    calls = {"n": 0, "durations": []}

    class W:
        def __init__(self, word, start, end):
            self.word, self.start, self.end, self.probability = word, start, end, 0.9

    class Seg:
        def __init__(self):
            self.start, self.end, self.text = 1.0, 2.0, "あ"
            self.words = [W("あ", 1.0, 2.0)]

    class Info:
        duration = 0.0

    class WhisperModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, **kw):
            calls["n"] += 1
            calls["durations"].append(len(audio) / 16000)
            if fail_after is not None and calls["n"] > fail_after:
                raise KeyboardInterrupt            # hard kill in the middle of a chunk
            return [Seg()], Info()

    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = WhisperModel
    sys.modules["faster_whisper"] = mod
    return calls


@pytest.mark.skipif(not __import__("shutil").which("ffmpeg"), reason="ffmpeg needed")
def test_transcribe_resumes_at_first_unfinished_chunk(tmp_path):
    from genki2anki.asr import load_asr, run_transcribe

    audio = tmp_path / "audio"
    write_speechlike_wav(audio / "lesson1.wav", [(t, t + 0.5) for t in range(5, 125, 10)], 130)
    cfg = load_config(None, {"work_dir": str(tmp_path / "w"), "audio_dir": str(audio),
                             "asr.chunk_s": 40, "asr.chunk_search_s": 10})
    try:
        calls = _install_fake_whisper(fail_after=2)
        with pytest.raises(KeyboardInterrupt):
            run_transcribe(cfg)
        assert calls["n"] == 3
        parts = list((cfg.work / "asr" / "parts" / "lesson1").glob("*.json"))
        assert len(parts) == 2                          # two chunks safely on disk
        assert not list((cfg.work / "asr").glob("*.json"))     # no final transcript yet
        plan = json.loads((cfg.work / "asr" / "parts" / "lesson1.plan.json").read_text())
        n_chunks = len(plan["chunks"])
        assert n_chunks >= 3
        # chunk edges sit in the silent gaps (bursts are 0.5 s long, every 10 s starting at 5 s)
        for _, edge in plan["chunks"][:-1]:
            assert (edge - 5) % 10 > 0.6, edge

        calls2 = _install_fake_whisper()
        assert run_transcribe(cfg) is True
        assert calls2["n"] == n_chunks - 2              # only the unfinished chunks were redone
        doc = load_asr(cfg)[0]
        assert len(doc["segments"]) == n_chunks
        starts = [s["start"] for s in doc["segments"]]
        assert starts == sorted(starts) and starts[0] == pytest.approx(1.0)
        assert starts[1] == pytest.approx(1.0 + plan["chunks"][1][0])   # offsets applied

        calls3 = _install_fake_whisper()
        assert run_transcribe(cfg) is True and calls3["n"] == 0        # fully cached
    finally:
        sys.modules.pop("faster_whisper", None)
