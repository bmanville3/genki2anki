"""Turn ASR transcripts into one timed kana character stream, and assign stretches of that
stream to lessons by how well they match each lesson's vocabulary."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import jp


@dataclass
class Stream:
    chars: str
    t0: np.ndarray
    t1: np.ndarray
    file: np.ndarray           # index into ``files``
    files: list[dict]          # [{"source":..., "wav":...}]

    def __len__(self) -> int:
        return len(self.chars)

    @classmethod
    def build(cls, asrs: list[dict]) -> "Stream":
        chars: list[str] = []
        t0: list[float] = []
        t1: list[float] = []
        fil: list[int] = []
        for fi, asr in enumerate(asrs):
            for seg in asr.get("segments", []):
                words = seg.get("words") or []
                seg_s, seg_e = float(seg["start"]), float(seg["end"])
                if words:
                    texts = [w["word"] for w in words]
                    seg_text = "".join(texts)
                    kana, src = jp.to_kana_map(seg_text)
                    word_of, word_off = [], []
                    off = 0
                    for wi, t in enumerate(texts):
                        word_off.append(off)
                        word_of.extend([wi] * len(t))
                        off += len(t)
                    k = 0
                    while k < len(kana):
                        j = k
                        while j < len(kana) and src[j] == src[k]:
                            j += 1
                        n = j - k
                        wi = word_of[src[k]]
                        w = words[wi]
                        wl = max(len(texts[wi]), 1)
                        pos = src[k] - word_off[wi]
                        ws = float(w["start"]) if w.get("start") is not None else seg_s
                        we = float(w["end"]) if w.get("end") is not None else seg_e
                        dur = max(we - ws, 0.0)
                        a, b = ws + dur * pos / wl, ws + dur * (pos + 1) / wl
                        for q in range(n):
                            chars.append(kana[k + q])
                            t0.append(a + (b - a) * q / n)
                            t1.append(a + (b - a) * (q + 1) / n)
                            fil.append(fi)
                        k = j
                else:  # no word timings: spread the segment evenly
                    kana = jp.to_kana(seg.get("text", ""))
                    n = len(kana)
                    for q, ch in enumerate(kana):
                        chars.append(ch)
                        t0.append(seg_s + (seg_e - seg_s) * q / n)
                        t1.append(seg_s + (seg_e - seg_s) * (q + 1) / n)
                        fil.append(fi)
        return cls("".join(chars), np.array(t0), np.array(t1), np.array(fil, dtype=np.int32),
                   [{"source": a.get("source"), "wav": a.get("wav")} for a in asrs])


def make_windows(n_chars: int, size: int) -> list[tuple[int, int]]:
    return [(a, min(a + size, n_chars)) for a in range(0, n_chars, size)]


def assign_windows(window_texts: list[str], lexicon: dict[int, set[str]],
                   monotone: bool = True) -> list[int]:
    """Pick the best lesson for every window.

    Score = sum of idf-weighted vocabulary words (>=3 kana) found verbatim in the window.
    With ``monotone`` the lesson sequence is constrained to be non-decreasing (Viterbi).
    """
    lessons = sorted(lexicon)
    K, W = len(lessons), len(window_texts)
    if K == 0 or W == 0:
        return [lessons[0] if lessons else 0] * W
    df: dict[str, int] = {}
    for words in lexicon.values():
        for w in words:
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log(1 + K / c) for w, c in df.items()}
    score = np.zeros((W, K))
    for wi, text in enumerate(window_texts):
        for ki, les in enumerate(lessons):
            s = 0.0
            for word in lexicon[les]:
                if len(word) >= 3 and word in text:
                    s += idf[word]
            score[wi, ki] = s
    if not monotone:
        return [lessons[int(k)] for k in score.argmax(axis=1)]
    stay = 0.01
    dp = np.full((W, K), -np.inf)
    back = np.zeros((W, K), dtype=np.int32)
    dp[0] = score[0]
    for wi in range(1, W):
        run_max = np.maximum.accumulate(dp[wi - 1])
        run_arg = np.zeros(K, dtype=np.int32)
        best = 0
        for k in range(K):
            if dp[wi - 1][k] >= dp[wi - 1][best]:
                best = k
            run_arg[k] = best
        for k in range(K):
            stay_val = dp[wi - 1][k] + stay
            if stay_val >= run_max[k]:
                dp[wi, k] = score[wi, k] + stay_val
                back[wi, k] = k
            else:
                dp[wi, k] = score[wi, k] + run_max[k]
                back[wi, k] = run_arg[k]
    k = int(dp[-1].argmax())
    path = [k]
    for wi in range(W - 1, 0, -1):
        k = int(back[wi, k])
        path.append(k)
    path.reverse()
    return [lessons[k] for k in path]
