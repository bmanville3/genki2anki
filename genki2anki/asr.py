"""Stage 2: audio files -> word-timestamped Japanese transcripts (faster-whisper).

Long recordings are transcribed in ~10 minute chunks (edges snapped to pauses). Every chunk is
checkpointed to disk, so an interrupted run resumes at the first unfinished chunk, not at the
start of a multi-hour file.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from . import runctl
from .config import Config
from .energy import Envelope


def natural_key(p: Path | str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]


def list_audio(audio_dir: str, exts: list[str]) -> list[Path]:
    root = Path(audio_dir)
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in set(exts)]
    return sorted(files, key=lambda p: natural_key(p.relative_to(root)))


def stem_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return re.sub(r"[^\w\-.]+", "_", "__".join(rel.parts))


def to_wav16k(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = runctl.tmp_sibling(dst, ".wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(tmp)], check=True)
    os.replace(tmp, dst)


def wav_duration(wav: Path) -> float:
    with wave.open(str(wav), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def read_slice(wav: Path, t0: float, t1: float) -> np.ndarray:
    with wave.open(str(wav), "rb") as w:
        sr = w.getframerate()
        w.setpos(min(int(t0 * sr), w.getnframes()))
        raw = w.readframes(max(int((t1 - t0) * sr), 0))
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def plan_chunks(wav: Path, chunk_s: float, search_s: float) -> list[list[float]]:
    """Split [0, duration] into ~chunk_s pieces whose edges sit in the longest nearby pause."""
    dur = wav_duration(wav)
    if chunk_s <= 0 or dur <= chunk_s * 1.25:
        return [[0.0, dur]]
    search_s = min(search_s, chunk_s / 4)
    env = Envelope.for_wav(wav)
    gs, ge = (env.ends[:-1], env.starts[1:]) if len(env.starts) > 1 else (np.zeros(0), np.zeros(0))
    bounds = [0.0]
    t = chunk_s
    while dur - t > chunk_s * 0.25:
        lo, hi = t - search_s, t + search_s
        b = t
        if len(gs):
            m = (ge > lo) & (gs < hi)
            if m.any():
                lens = np.where(m, np.minimum(ge, hi) - np.maximum(gs, lo), -1.0)
                k = int(lens.argmax())
                b = float((max(gs[k], lo) + min(ge[k], hi)) / 2)
        bounds.append(b)
        t = b + chunk_s
    bounds.append(dur)
    return [[bounds[i], bounds[i + 1]] for i in range(len(bounds) - 1)]


def _is_valid_final(d) -> bool:
    return isinstance(d, dict) and "segments" in d and "wav" in d


def run_transcribe(cfg: Config, force: bool = False) -> bool:
    """Returns True when every audio file has a complete transcript."""
    if not cfg.audio_dir:
        raise SystemExit("transcribe: --audio-dir is required")
    root = Path(cfg.audio_dir)
    files = list_audio(cfg.audio_dir, cfg.asr.extensions)
    if not files:
        raise SystemExit(f"no audio files found in {root}")
    asr_dir = cfg.work / "asr"
    parts = asr_dir / "parts"
    wav_dir = asr_dir / "wav"
    asr_dir.mkdir(parents=True, exist_ok=True)

    model = None
    complete = True
    for i, src in enumerate(files, 1):
        if runctl.stopping():
            complete = False
            break
        stem = stem_for(src, root)
        final = asr_dir / f"{stem}.json"
        plan_path = parts / f"{stem}.plan.json"
        chunk_dir = parts / stem
        if force:
            final.unlink(missing_ok=True)
            plan_path.unlink(missing_ok=True)
            shutil.rmtree(chunk_dir, ignore_errors=True)
        if _is_valid_final(runctl.read_json(final)):
            print(f"[transcribe] {i}/{len(files)} done already: {src.name}")
            continue

        wav = wav_dir / f"{stem}.wav"
        if not wav.exists():
            print(f"[transcribe] {i}/{len(files)} converting {src.name}")
            to_wav16k(src, wav)
        sig = {"src_size": src.stat().st_size, "chunk_s": cfg.asr.chunk_s,
               "search_s": cfg.asr.chunk_search_s}
        plan = runctl.read_json(plan_path)
        if not plan or {k: plan.get(k) for k in sig} != sig:
            shutil.rmtree(chunk_dir, ignore_errors=True)       # stale chunks belong to another plan
            plan = {**sig, "chunks": plan_chunks(wav, cfg.asr.chunk_s, cfg.asr.chunk_search_s)}
            runctl.atomic_write_json(plan_path, plan)
        chunks = plan["chunks"]

        file_complete = True
        for k, (t0, t1) in enumerate(chunks):
            cp = chunk_dir / f"{k:04d}.json"
            if runctl.read_json(cp) is not None:
                continue
            if runctl.stopping():
                file_complete = False
                break
            if model is None:
                from faster_whisper import WhisperModel  # lazy: GPU/heavy import

                model = WhisperModel(cfg.asr.model, device=cfg.asr.device,
                                     compute_type=cfg.asr.compute_type,
                                     download_root=cfg.asr.download_root)
            print(f"[transcribe] {i}/{len(files)} {src.name} chunk {k + 1}/{len(chunks)} "
                  f"({t0:.0f}s-{t1:.0f}s)")
            runctl.atomic_write_json(cp, {"start": t0, "end": t1,
                                          "segments": _transcribe_chunk(model, cfg, wav, t0, t1)})
        if not file_complete:
            complete = False
            break
        segments: list = []
        for k in range(len(chunks)):
            segments.extend(runctl.read_json(chunk_dir / f"{k:04d}.json")["segments"])
        runctl.atomic_write_json(final, {
            "source": str(src.resolve()), "wav": str(wav.resolve()),
            "duration": wav_duration(wav), "chunks": len(chunks), "segments": segments})
    return complete


def _transcribe_chunk(model, cfg: Config, wav: Path, t0: float, t1: float) -> list[dict]:
    audio = read_slice(wav, t0, t1)
    kwargs = dict(
        language=cfg.asr.language, beam_size=cfg.asr.beam_size, word_timestamps=True,
        vad_filter=cfg.asr.vad,
        vad_parameters={"min_silence_duration_ms": 250, "speech_pad_ms": 150},
        condition_on_previous_text=False, initial_prompt=cfg.asr.initial_prompt,
    )
    try:
        segs, _info = model.transcribe(audio, hallucination_silence_threshold=2.0, **kwargs)
    except TypeError:  # older faster-whisper without that option
        segs, _info = model.transcribe(audio, **kwargs)
    out = []
    for s in segs:
        words = [{"word": w.word, "start": float(w.start) + t0, "end": float(w.end) + t0,
                  "probability": float(getattr(w, "probability", 1.0))}
                 for w in (s.words or [])]
        out.append({"start": float(s.start) + t0, "end": float(s.end) + t0,
                    "text": s.text, "words": words})
    return out


def load_asr(cfg: Config) -> list[dict]:
    docs = []
    for p in (cfg.work / "asr").glob("*.json"):
        d = runctl.read_json(p)
        if _is_valid_final(d):
            docs.append(d)
    if not docs:
        raise SystemExit("no complete transcripts found - run `transcribe` first")
    return sorted(docs, key=lambda d: natural_key(d["source"]))
