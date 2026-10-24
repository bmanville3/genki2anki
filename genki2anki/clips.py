"""Cut matched words out of the source audio with ffmpeg (atomic, skip-if-exists)."""
from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from .align import Match
from .config import AlignConfig
from .energy import INF, Envelope
from .runctl import tmp_sibling


def plan_clip(m: Match, env: Optional[Envelope], cfg: AlignConfig) -> tuple[float, float]:
    """Clip bounds in seconds. Snap to real silence when the energy envelope agrees with ASR."""
    snap = env.snap(m.t0, m.t1) if env is not None else None
    if snap is not None and (snap.end - snap.start) <= (m.t1 - m.t0) + cfg.tight_slack_s:
        pb = snap.pause_before if snap.pause_before != INF else 1.0
        pa = snap.pause_after if snap.pause_after != INF else 1.0
        cs = snap.start - min(cfg.pad_before_s, pb / 2)
        ce = snap.end + min(cfg.pad_after_s, pa / 2)
    else:
        cs, ce = m.t0 - 0.12, m.t1 + 0.18
    cs = max(cs, 0.0)
    ce = min(ce, cs + cfg.max_clip_s)
    return cs, ce


def cut(src: str, cs: float, ce: float, out: Path) -> None:
    dur = max(ce - cs, 0.1)
    fade = f"afade=t=in:st=0:d=0.015,afade=t=out:st={max(dur - 0.04, 0):.3f}:d=0.04"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_sibling(out, ".mp3")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{cs:.3f}", "-t", f"{dur:.3f}", "-i", src,
         "-vn", "-ac", "1", "-ar", "44100", "-af", fade,
         "-c:a", "libmp3lame", "-q:a", "4", str(tmp)],
        check=True)
    os.replace(tmp, out)


def cut_all(jobs: list[tuple[str, Match]], files: list[dict], envs: dict, media_dir: Path,
            cfg: AlignConfig) -> dict[str, dict]:
    """jobs: [(target_id, Match)] -> manifest entries {id: {file, source, ...}}.

    The file name encodes the clip bounds, so a clip that already exists is by construction
    the right one and is not cut again.
    """
    manifest: dict[str, dict] = {}

    def work(job):
        tid, m = job
        cs, ce = plan_clip(m, envs.get(m.file), cfg)
        name = f"genki_{tid}_{int(round(cs * 100))}_{int(round(ce * 100))}.mp3"
        path = media_dir / name
        if not (path.exists() and path.stat().st_size > 0):
            cut(files[m.file]["source"], cs, ce, path)
        return tid, {"file": name, "source": "genki", "sim": round(m.sim, 3),
                     "isolated": m.isolated, "tight": m.tight, "recovered": m.recovered,
                     "t0": round(cs, 3), "t1": round(ce, 3), "src_file": files[m.file]["source"]}

    with ThreadPoolExecutor(max_workers=cfg.clip_workers) as ex:
        for tid, entry in ex.map(work, jobs):
            manifest[tid] = entry
    return manifest
