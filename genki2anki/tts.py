"""Optional local TTS used to fill audio gaps (and grammar example sentences). Resumable."""
from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import requests

from . import jp, runctl
from .audio import load_book, load_manifest
from .config import Config, TTSConfig
from .normalize import final_sentence, load_norm


def _to_mp3(src: Path, dst: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1",
                    "-ar", "44100", "-c:a", "libmp3lame", "-q:a", "4", str(dst)], check=True)


def synthesize(text: str, out: Path, cfg: TTSConfig) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = runctl.tmp_sibling(out, ".mp3")
    with tempfile.TemporaryDirectory() as td:
        if cfg.backend == "voicevox":
            base = cfg.voicevox_url.rstrip("/")
            q = requests.post(f"{base}/audio_query",
                              params={"text": text, "speaker": cfg.voicevox_speaker}, timeout=60)
            q.raise_for_status()
            r = requests.post(f"{base}/synthesis", params={"speaker": cfg.voicevox_speaker},
                              json=q.json(), timeout=120)
            r.raise_for_status()
            wav = Path(td) / "tts.wav"
            wav.write_bytes(r.content)
            _to_mp3(wav, tmp_out)
        elif cfg.backend == "command":
            if not cfg.command:
                raise SystemExit("tts.backend=command needs tts.command")
            tmp = Path(td) / "tts_out.wav"
            argv = [a.replace("{text}", text).replace("{out}", str(tmp))
                    for a in shlex.split(cfg.command)]
            subprocess.run(argv, check=True)
            if not tmp.exists():
                alt = tmp.with_suffix(".mp3")
                if not alt.exists():
                    return False
                tmp = alt
            _to_mp3(tmp, tmp_out)
        else:
            return False
    os.replace(tmp_out, out)
    return True


def run_tts(cfg: Config) -> None:
    t = cfg.tts
    if t.backend == "none":
        print("[tts] backend is 'none' - nothing to do (set tts.backend to voicevox or command)")
        return
    book = load_book(cfg)
    media = cfg.work / "media"
    mpath = cfg.work / "tts_manifest.json"
    tts_m: dict = {k: v for k, v in (runctl.read_json(mpath, {}) or {}).items()
                   if (media / v["file"]).exists()}
    have = load_manifest(cfg)
    have.update(tts_m)
    norm = load_norm(cfg)
    jobs: list[tuple[str, str]] = []   # (id, text)
    if t.vocab:
        for v in book["vocab"]:
            if v["id"] not in have:
                text = jp.speakable(v["kana"]) or jp.primary_form(v["kanji"] or "")
                if text:
                    jobs.append((v["id"], text))
    if t.grammar:
        for g in book["grammar"]:
            for e in g["examples"]:
                if e["id"] not in have:
                    jobs.append((e["id"], final_sentence(norm.get(e["id"]), e["ja"])))
    print(f"[tts] {len(jobs)} items to synthesise with {t.backend}")

    def work(job):
        tid, text = job
        name = f"genki_tts_{hashlib.sha1(text.encode()).hexdigest()[:12]}.mp3"
        if not (media / name).exists():
            if not synthesize(text, media / name, t):
                raise RuntimeError(f"no audio produced for {text!r}")
        return name

    count = [0]

    def on_result(item, name, err):
        if err is not None:
            print(f"[tts] failed for {item[1]!r}: {err}")
            return
        tts_m[item[0]] = {"file": name, "source": "tts"}
        count[0] += 1
        if count[0] % 25 == 0:
            runctl.atomic_write_json(mpath, tts_m)

    try:
        _, failed, left = runctl.run_parallel(jobs, work, t.workers, on_result, label="tts", every=25)
    finally:
        runctl.atomic_write_json(mpath, tts_m)
    print(f"[tts] manifest has {len(tts_m)} TTS entries" + (f"; {failed} failed" if failed else "")
          + (f"; stopped early ({left} not started)" if left else ""))
