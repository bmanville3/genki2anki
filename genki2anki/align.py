"""Find where each target word / sentence is spoken in the transcript stream.

For every target we run an approximate-substring search (Sellers edit distance, vectorised
with numpy) against the lesson's kana stream to get candidate spans, then pick the best
*ordered* chain of candidates with dynamic programming (the book order is the audio order).
Targets the chain skipped get a second, order-free chance with a stricter threshold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import AlignConfig
from .energy import INF, Envelope
from .stream import Stream


@dataclass
class Target:
    id: str
    alts: list[str]
    lesson: int
    kind: str = "vocab"      # vocab | example
    label: str = ""


@dataclass
class Match:
    id: str
    gstart: int              # global char index (inclusive)
    gend: int                # global char index (exclusive)
    sim: float
    alt: str
    isolated: bool
    tight: bool
    recovered: bool = False
    t0: float = 0.0          # clip start (seconds, in source file)
    t1: float = 0.0
    file: int = 0
    heard: str = ""


@dataclass
class _Cand:
    ti: int
    s: int
    e: int
    sim: float
    alt: str
    score: float = 0.0
    isolated: bool = False
    tight: bool = False
    snap_t: tuple = field(default=(0.0, 0.0))


def sellers(target: str, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximate substring matching. Returns (cost[j], start[j]) for substrings text[start:j]."""
    m = len(codes)
    idx = np.arange(m + 1, dtype=np.int32)
    prev = np.zeros(m + 1, dtype=np.int32)
    prev_st = idx.copy()
    for i, ch in enumerate(target, 1):
        mism = (codes != ord(ch)).astype(np.int32)
        diag = prev[:-1] + mism
        up = prev[1:] + 1
        tmp = np.empty(m + 1, dtype=np.int32)
        tmp_st = np.empty(m + 1, dtype=np.int32)
        tmp[0], tmp_st[0] = i, 0
        use_diag = diag <= up
        tmp[1:] = np.where(use_diag, diag, up)
        tmp_st[1:] = np.where(use_diag, prev_st[:-1], prev_st[1:])
        v = tmp - idx
        cm = np.minimum.accumulate(v)
        src = np.maximum.accumulate(np.where(v == cm, idx, 0))
        prev = cm + idx
        prev_st = tmp_st[src]
    return prev, prev_st


class LessonAligner:
    def __init__(self, stream: Stream, gidx: np.ndarray, envs: dict[int, Optional[Envelope]],
                 cfg: AlignConfig):
        self.stream, self.gidx, self.envs, self.cfg = stream, gidx, envs, cfg
        self.text = "".join(stream.chars[i] for i in gidx)
        self.codes = np.fromiter((ord(c) for c in self.text), dtype=np.int32, count=len(self.text))
        self.used = np.zeros(len(self.text), dtype=bool)

    # -- span geometry -------------------------------------------------------------------
    def _span_ok(self, s: int, e: int) -> bool:
        g = self.gidx
        if e <= s or g[e - 1] - g[s] != e - s - 1:           # must be contiguous in the stream
            return False
        st = self.stream
        if st.file[g[s]] != st.file[g[e - 1]]:
            return False
        if e - s > 1:
            gaps = st.t0[g[s + 1:e]] - st.t1[g[s:e - 1]]
            if gaps.max() > self.cfg.max_gap_s:
                return False
        dur = st.t1[g[e - 1]] - st.t0[g[s]]
        return dur <= self.cfg.max_clip_s - 1.0

    def _geometry(self, c: _Cand) -> None:
        st, g = self.stream, self.gidx
        fi = int(st.file[g[c.s]])
        t0, t1 = float(st.t0[g[c.s]]), float(st.t1[g[c.e - 1]])
        env = self.envs.get(fi)
        snap = env.snap(t0, t1) if env is not None else None
        if snap is not None and (snap.end - snap.start) <= (t1 - t0) + self.cfg.tight_slack_s:
            c.tight = True
            c.isolated = (snap.pause_before >= self.cfg.isolation_pause_s
                          and snap.pause_after >= self.cfg.isolation_pause_s)
        else:
            c.tight = False
            c.isolated = False

    # -- candidates ----------------------------------------------------------------------
    def candidates(self, ti: int, t: Target, min_sim: float, max_per_alt: int = 12) -> list[_Cand]:
        out: list[_Cand] = []
        for alt in t.alts:
            L = len(alt)
            if L == 0 or len(self.codes) == 0:
                continue
            cost, start = sellers(alt, self.codes)
            tol = int(math.floor((1.0 - min_sim) * L + 1e-9)) if L > 2 else 0
            left = np.r_[INF, cost[:-1]].astype(np.float64)
            right = np.r_[cost[1:], INF].astype(np.float64)
            ok = (cost <= tol) & (cost <= left) & (cost < right)
            ok[0] = False
            ends = np.flatnonzero(ok)
            if len(ends) > max_per_alt:
                ends = ends[np.argsort(cost[ends], kind="stable")[:max_per_alt]]
            for e in ends:
                s = int(start[e])
                if not self._span_ok(s, int(e)):
                    continue
                sim = 1.0 - float(cost[e]) / max(L, int(e) - s)
                c = _Cand(ti, s, int(e), sim, alt)
                self._geometry(c)
                if L <= 2 and not (c.isolated and sim >= 0.999):
                    continue                                   # short words must stand alone
                if self.cfg.require_tight and not c.tight:
                    continue
                c.score = (sim - 0.5) * math.sqrt(L) + (0.3 if c.isolated else 0.0)
                out.append(c)
        return out

    # -- selection -----------------------------------------------------------------------
    @staticmethod
    def _chain(cands: list[_Cand]) -> list[_Cand]:
        if not cands:
            return []
        cands = sorted(cands, key=lambda c: (c.ti, c.s))
        n = len(cands)
        ti = np.array([c.ti for c in cands])
        s_arr = np.array([c.s for c in cands])
        e_arr = np.array([c.e for c in cands])
        best = np.zeros(n)
        prev = np.full(n, -1)
        for a in range(n):
            mask = (ti[:a] < ti[a]) & (e_arr[:a] <= s_arr[a])
            gain, p = 0.0, -1
            if mask.any():
                cand_scores = np.where(mask, best[:a], -1.0)
                p = int(cand_scores.argmax())
                gain = float(cand_scores[p])
                if gain <= 0:
                    gain, p = 0.0, -1
            best[a] = cands[a].score + gain
            prev[a] = p
        k = int(best.argmax())
        chosen = []
        while k >= 0:
            chosen.append(cands[k])
            k = int(prev[k])
        chosen.reverse()
        return chosen

    def align(self, targets: list[Target]) -> dict[str, Match]:
        cfg = self.cfg
        all_cands: list[_Cand] = []
        per_target: dict[int, list[_Cand]] = {}
        for ti, t in enumerate(targets):
            cs = self.candidates(ti, t, cfg.min_similarity)
            per_target[ti] = cs
            all_cands.extend(cs)
        chosen = self._chain(all_cands)
        result: dict[int, _Cand] = {}
        recovered: set[int] = set()
        for c in chosen:
            result[c.ti] = c
            self.used[c.s:c.e] = True
        # order-free recovery pass for everything the chain skipped
        for ti, t in enumerate(targets):
            if ti in result:
                continue
            best: Optional[_Cand] = None
            for c in per_target[ti]:
                if c.sim < cfg.recover_similarity or self.used[c.s:c.e].any():
                    continue
                if len(c.alt) <= 3 and not c.isolated:
                    continue
                if best is None or c.score > best.score:
                    best = c
            if best is not None:
                result[ti] = best
                recovered.add(ti)
                self.used[best.s:best.e] = True
        matches: dict[str, Match] = {}
        st = self.stream
        for ti, c in result.items():
            g0, g1 = int(self.gidx[c.s]), int(self.gidx[c.e - 1]) + 1
            matches[targets[ti].id] = Match(
                id=targets[ti].id, gstart=g0, gend=g1, sim=c.sim, alt=c.alt,
                isolated=c.isolated, tight=c.tight, recovered=ti in recovered,
                t0=float(st.t0[g0]), t1=float(st.t1[g1 - 1]), file=int(st.file[g0]),
                heard=st.chars[g0:g1],
            )
        return matches
