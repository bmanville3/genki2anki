"""Jisho.org lookups (public JSON API) and result-page screenshots (playwright).

Used as an independent third-party check on the textbook transcription and to fill in the
spelling/reading the book does not print. Requests are rate limited and cached on disk.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

from . import jp, runctl
from .config import JishoConfig

KANA_ONLY_TAG = "usually written using kana alone"
_STOP = {"to", "a", "an", "the", "be", "of", "or", "and", "in", "on", "at", "for", "with",
         "one", "ones", "something", "someone", "esp", "etc", "is", "it", "as", "by"}


class JishoError(RuntimeError):
    pass


class Throttle:
    def __init__(self, delay_s: float):
        self.delay_s = delay_s
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            gap = self._last + self.delay_s - time.time()
            if gap > 0:
                time.sleep(gap)
            self._last = time.time()


class JishoClient:
    def __init__(self, cfg: JishoConfig, cache_dir: Path, throttle: Optional[Throttle] = None):
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.throttle = throttle or Throttle(cfg.delay_s)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = cfg.user_agent

    def search(self, keyword: str) -> list[dict]:
        cache = self.cache_dir / f"{hashlib.sha1(keyword.encode('utf-8')).hexdigest()}.json"
        cached = runctl.read_json(cache)
        if cached is not None:
            return cached["data"]
        self.throttle.wait()
        try:
            r = self.session.get(self.cfg.api_url, params={"keyword": keyword},
                                 timeout=self.cfg.timeout)
        except requests.RequestException as e:
            raise JishoError(f"{type(e).__name__}: {e}") from e
        if r.status_code == 429:
            time.sleep(30)
            raise JishoError("rate limited (HTTP 429); waiting then retrying later")
        if r.status_code >= 400:
            raise JishoError(f"HTTP {r.status_code}")
        try:
            data = r.json().get("data", [])
        except ValueError as e:
            raise JishoError(f"bad JSON: {e}") from e
        runctl.atomic_write_json(cache, {"keyword": keyword, "data": data})
        return data


# --------------------------------------------------------------------------- matching
def _stem(w: str) -> str:
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def eng_tokens(s: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z]+", (s or "").lower()) if w not in _STOP and len(w) > 1}


def english_overlap(textbook_en: str, entry: dict) -> float:
    tb = eng_tokens(textbook_en)
    if not tb:
        return 1.0
    defs = " ".join(d for s in entry.get("senses", []) for d in s.get("english_definitions", []))
    return len(tb & eng_tokens(defs)) / len(tb)


@dataclass
class Pick:
    entry: Optional[dict]
    form: Optional[dict]
    match: str            # exact | partial | none
    overlap: float
    index: int
    score: float

    @property
    def rank(self) -> tuple:
        order = {"exact": 2, "partial": 1, "none": 0}
        return (order[self.match] if self.entry else -1, self.score)


def pick_entry(results: list[dict], tb_kanji: str, tb_kana: str, tb_english: str) -> Pick:
    """Choose the Jisho entry that best corresponds to a textbook vocabulary item.

    ``exact``  - every spelling the textbook printed (kanji and/or kana) is a form of the entry.
    ``partial``- only one of two printed spellings matches.
    ``none``   - results exist but no form matches (nearest by meaning is returned for reference).
    """
    nk = jp.normalize_kana(tb_kana) if tb_kana else ""
    best: Optional[Pick] = None
    for rank, e in enumerate(results):
        form_score, form = 0, None
        for f in e.get("japanese", []):
            ok_k = bool(tb_kanji) and (f.get("word") or "") == tb_kanji
            ok_r = bool(nk) and jp.normalize_kana(f.get("reading") or "") == nk
            if tb_kanji and nk:
                s = 2 * ok_k + 2 * ok_r
            else:
                s = 4 if (ok_k or ok_r) else 0
            if s > form_score:
                form_score, form = s, f
        ov = english_overlap(tb_english, e)
        score = form_score + 2 * ov + (0.4 if e.get("is_common") else 0.0) - rank * 0.01
        match = "exact" if form_score >= 4 else "partial" if form_score > 0 else "none"
        cand = Pick(e, form, match, ov, rank, score)
        if best is None or cand.rank > best.rank:
            best = cand
    return best or Pick(None, None, "none", 0.0, -1, 0.0)


def summarize(p: Pick) -> dict:
    e = p.entry or {}
    senses = e.get("senses", [])
    first_tags = [t.lower() for t in (senses[0].get("tags", []) if senses else [])]
    defs = "; ".join("/".join(s.get("english_definitions", [])[:3]) for s in senses[:3])
    return {
        "word": (p.form or {}).get("word"), "reading": (p.form or {}).get("reading"),
        "english": defs[:300], "is_common": bool(e.get("is_common")),
        "jlpt": [j.replace("jlpt-", "").upper() for j in e.get("jlpt", [])],
        "usually_kana": KANA_ONLY_TAG in first_tags, "slug": e.get("slug"), "index": p.index,
        "overlap": round(p.overlap, 2), "match": p.match,
    }


# --------------------------------------------------------------------------- screenshots
class Shooter:
    """Screenshots Jisho result pages with headless Chromium (playwright)."""

    def __init__(self, cfg: JishoConfig, throttle: Throttle):
        self.cfg, self.throttle = cfg, throttle
        self.state = "new"            # new | ready | unavailable
        self.error = ""
        self._pw = self._browser = self._page = None

    def ensure(self) -> bool:
        if self.state == "ready":
            return True
        if self.state == "unavailable":
            return False
        try:
            from playwright.sync_api import sync_playwright

            self._pw = sync_playwright().start()
            kw = {"executable_path": self.cfg.chromium_path} if self.cfg.chromium_path else {}
            self._browser = self._pw.chromium.launch(headless=True, **kw)
            ctx = self._browser.new_context(
                viewport={"width": self.cfg.viewport_width, "height": self.cfg.viewport_height},
                user_agent=self.cfg.user_agent, locale="en-US")
            self._page = ctx.new_page()
            self.state = "ready"
        except Exception as e:  # noqa: BLE001
            self.state, self.error = "unavailable", f"{type(e).__name__}: {e}"
            self.close()
        return self.state == "ready"

    def shoot(self, keyword: str, want_word: Optional[str], want_reading: Optional[str],
              prefer_index: Optional[int], out: Path) -> str:
        """Screenshot the result block for the entry we matched. Returns the page URL."""
        page = self._page
        url = f"{self.cfg.web_url.rstrip('/')}/{quote(keyword, safe='')}"
        self.throttle.wait()
        page.goto(url, wait_until="domcontentloaded", timeout=self.cfg.timeout * 1000)
        try:
            page.wait_for_selector(".concept_light, #no-matches", timeout=8000)
        except Exception:  # noqa: BLE001 - fall through to a plain page screenshot
            pass
        loc = page.locator(".concept_light")
        n = loc.count()
        target = None
        if n:
            order = list(range(min(n, 10)))
            if prefer_index is not None and 0 <= prefer_index < n:
                order = [prefer_index] + [i for i in order if i != prefer_index]
            needles = [x for x in (want_word, want_reading) if x]
            for i in order:
                try:
                    text = loc.nth(i).inner_text()
                except Exception:  # noqa: BLE001
                    continue
                if any(nd in text for nd in needles):
                    target = loc.nth(i)
                    break
            target = target or loc.nth(0)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = runctl.tmp_sibling(out, ".png")
        if target is not None:
            target.screenshot(path=str(tmp))
        else:
            page.screenshot(path=str(tmp))
        os.replace(tmp, out)
        return url

    def close(self) -> None:
        for obj, meth in ((self._browser, "close"), (self._pw, "stop")):
            try:
                if obj is not None:
                    getattr(obj, meth)()
            except Exception:  # noqa: BLE001
                pass
        self._browser = self._pw = self._page = None
