"""Energy-based speech/silence envelope, used to snap clip edges to real silence and to
decide whether a matched word is *isolated* (surrounded by pauses)."""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

FRAME_S = 0.01
INF = float("inf")


def compute_rms(wav_path: Path) -> np.ndarray:
    """10 ms RMS frames of a 16-bit PCM wav (any rate, mono-mixed)."""
    out: list[np.ndarray] = []
    with wave.open(str(wav_path), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if sw != 2:
            raise ValueError(f"{wav_path}: expected 16-bit PCM, got {sw * 8}-bit")
        hop = int(sr * FRAME_S)
        carry = np.zeros(0, dtype=np.float32)
        while True:
            raw = w.readframes(sr * 60)
            if not raw:
                break
            x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            if ch > 1:
                x = x[: len(x) // ch * ch].reshape(-1, ch).mean(axis=1)
            x = np.concatenate([carry, x])
            n = len(x) // hop
            if n:
                out.append(np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1)))
            carry = x[n * hop:]
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)  # end exclusive


@dataclass
class Snap:
    start: float
    end: float
    pause_before: float
    pause_after: float


class Envelope:
    def __init__(self, rms: np.ndarray, hole_s: float = 0.12, min_run_s: float = 0.05):
        self.n_frames = len(rms)
        if len(rms) == 0:
            self.starts = self.ends = np.zeros(0)
            return
        noise = float(np.percentile(rms, 5))
        speech = float(np.percentile(rms, 95))
        thr = max(noise * 4.0, speech * 0.06, 1e-4)
        s, e = _runs(rms > thr)
        # fill short holes (stop closures, small dips) so words stay in one run
        hole = int(round(hole_s / FRAME_S))
        ms, me = [], []
        for a, b in zip(s, e):
            if ms and a - me[-1] <= hole:
                me[-1] = b
            else:
                ms.append(int(a))
                me.append(int(b))
        keep = [(a, b) for a, b in zip(ms, me) if (b - a) * FRAME_S >= min_run_s]
        self.starts = np.array([a for a, _ in keep], dtype=np.float64) * FRAME_S
        self.ends = np.array([b for _, b in keep], dtype=np.float64) * FRAME_S

    @classmethod
    def for_wav(cls, wav_path: Path) -> "Envelope":
        cache = Path(str(wav_path) + ".rms.npy")
        if cache.exists() and cache.stat().st_mtime >= Path(wav_path).stat().st_mtime:
            rms = np.load(cache)
        else:
            rms = compute_rms(wav_path)
            np.save(cache, rms)
        return cls(rms)

    def snap(self, t0: float, t1: float, tol: float = 0.05) -> Optional[Snap]:
        """Speech runs overlapping [t0, t1]; returns their union and the pauses around it."""
        if len(self.starts) == 0:
            return None
        i = int(np.searchsorted(self.ends, t0 - tol, side="right"))
        j = int(np.searchsorted(self.starts, t1 + tol, side="left"))
        if i >= j:
            return None
        rs, re_ = float(self.starts[i]), float(self.ends[j - 1])
        pb = rs - float(self.ends[i - 1]) if i > 0 else INF
        pa = float(self.starts[j]) - re_ if j < len(self.starts) else INF
        return Snap(rs, re_, pb, pa)
