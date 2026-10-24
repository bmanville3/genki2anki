"""Stage: look every vocabulary item up on Jisho (data + screenshot). Resumable per word."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import jp, runctl
from .audio import load_book
from .config import Config
from .jisho import JishoClient, JishoError, Shooter, Throttle, pick_entry, summarize


def enrich_path(cfg: Config, vid: str) -> Path:
    return cfg.work / "enrich" / f"{vid}.json"


def load_enrich(cfg: Config) -> dict[str, dict]:
    out = {}
    for p in (cfg.work / "enrich").glob("*.json"):
        d = runctl.read_json(p)
        if d:
            out[d["id"]] = d
    return out


def lookup(client: JishoClient, v: dict) -> dict:
    tbk = jp.primary_form(v.get("kanji") or "")
    tbr = jp.primary_form(v.get("kana") or "")
    best, best_kw = None, None
    for kw in dict.fromkeys(k for k in (tbk, tbr) if k):
        pick = pick_entry(client.search(kw), tbk, tbr, v["english"])
        if best is None or pick.rank > best.rank:
            best, best_kw = pick, kw
        if pick.match == "exact":
            break
    if best is None or best.entry is None:
        return {"id": v["id"], "status": "not_found", "query": best_kw or tbk or tbr, "match": "none"}
    return {"id": v["id"], "status": "ok", "query": best_kw, **summarize(best)}


def run_enrich(cfg: Config, force: bool = False) -> None:
    j = cfg.jisho
    if not j.enabled:
        print("[enrich] jisho disabled - rows will be marked not_checked")
        return
    book = load_book(cfg)
    vocab = book["vocab"]
    throttle = Throttle(j.delay_s)
    client = JishoClient(j, cfg.work / "jisho_cache", throttle)
    shooter: Optional[Shooter] = Shooter(j, throttle) if j.screenshots else None
    shots_dir = cfg.work / "jisho"
    errors_in_row = done = skipped = shot_failures = 0
    warned = False
    print(f"[enrich] {len(vocab)} vocabulary items")
    try:
        for n, v in enumerate(vocab, 1):
            if runctl.stopping():
                print(f"[enrich] stopped at {n - 1}/{len(vocab)}; re-run to continue")
                break
            path = enrich_path(cfg, v["id"])
            old = None if force else runctl.read_json(path)
            need_api = old is None or old.get("status") not in ("ok", "not_found")
            need_shot = (shooter is not None and shooter.state != "unavailable"
                         and (old is None or not old.get("screenshot")
                              or not (cfg.work / old["screenshot"]).exists()))
            if not need_api and not need_shot:
                skipped += 1
                continue
            try:
                rec = lookup(client, v) if need_api else dict(old)
                if need_api and old and old.get("screenshot"):
                    rec["screenshot"], rec["url"] = old["screenshot"], old.get("url")
                errors_in_row = 0
            except Exception as e:  # noqa: BLE001 - network errors and odd API payloads alike
                errors_in_row += 1
                print(f"[enrich] {v['id']} lookup failed: {type(e).__name__}: {e}")
                if errors_in_row >= j.max_consecutive_errors:
                    print(f"[enrich] {errors_in_row} failures in a row - is jisho.org reachable? "
                          "stopping this stage (re-run to retry; export still works, rows are "
                          "marked not_checked)")
                    break
                continue
            if need_shot and shooter is not None and shooter.ensure():
                try:
                    out = shots_dir / f"{v['id']}.png"
                    rec["url"] = shooter.shoot(rec.get("query") or "", rec.get("word"),
                                               rec.get("reading"), rec.get("index"), out)
                    rec["screenshot"] = f"jisho/{v['id']}.png"
                    rec.pop("screenshot_error", None)
                except Exception as e:  # noqa: BLE001 - keep the data, retry the shot next run
                    shot_failures += 1
                    rec["screenshot_error"] = f"{type(e).__name__}: {str(e)[:200]}"
                    print(f"[enrich] screenshot failed for {v['id']}: {rec['screenshot_error']}")
            elif need_shot and shooter is not None and not warned:
                warned = True
                print("[enrich] screenshots unavailable (" + shooter.error[:160] + ") - continuing "
                      "without them; install with `pip install playwright && playwright install "
                      "chromium` and re-run to add them")
            runctl.atomic_write_json(path, rec)
            done += 1
            if done % 25 == 0:
                print(f"[enrich] {n}/{len(vocab)} ({done} looked up this run, {skipped} already done)")
    finally:
        if shooter is not None:
            shooter.close()
    print(f"[enrich] finished: {done} looked up, {skipped} already done"
          + (f", {shot_failures} screenshot failures" if shot_failures else ""))
