"""Final stage: book + Jisho + audio -> vocab.csv / grammar.csv plus a self-contained folder
of the images and audio the CSVs point at. Always safe to re-run (it only reads work/)."""
from __future__ import annotations

import csv
import io
import os
import shutil
from pathlib import Path

from . import runctl
from .audio import load_book, load_manifest
from .config import Config
from .enrich import load_enrich
from .fields import GRAMMAR_COLUMNS, VOCAB_COLUMNS, b, compose_grammar, compose_vocab
from .normalize import load_norm
from .pdf import render_pages


def _prev_review(path: Path) -> dict[str, dict]:
    """Hand-entered review columns from the previous export, so re-exporting never loses them."""
    out: dict[str, dict] = {}
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("id") and (r.get("verified") or r.get("notes")):
                    out[r["id"]] = {"verified": r.get("verified", ""), "notes": r.get("notes", "")}
    except FileNotFoundError:
        pass
    return out


def _copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = runctl.tmp_sibling(dst)
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    return True


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    # utf-8 with BOM so Excel opens the Japanese text correctly
    runctl.atomic_write_bytes(path, b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8"))


def _finish(row: dict, prev: dict) -> None:
    p = prev.get(row["id"], {})
    row["verified"], row["notes"] = p.get("verified", ""), p.get("notes", "")
    row["still_needs_review"] = b(row["needs_verification"] == "TRUE"
                                  and row["verified"].strip().upper() != "TRUE")


def run_export(cfg: Config) -> Path:
    book = load_book(cfg)
    manifest = load_manifest(cfg)
    enr = load_enrich(cfg)
    norm = load_norm(cfg)
    out = Path(cfg.out_dir)
    media = cfg.work / "media"
    prev_v = _prev_review(out / "vocab.csv")
    prev_g = _prev_review(out / "grammar.csv")

    vrows = []
    for v in sorted(book["vocab"], key=lambda x: (x["lesson"] or 0, x["page"] or 0)):
        r = compose_vocab(v, enr.get(v["id"]), manifest.get(v["id"]), media, cfg)
        _finish(r, prev_v)
        vrows.append(r)
    grows = []
    for g in sorted(book["grammar"], key=lambda x: (x["lesson"] or 0, (x["pages"] or [0])[0])):
        for e in g["examples"]:
            r = compose_grammar(g, e, norm.get(e["id"]), manifest.get(e["id"]), media, cfg)
            _finish(r, prev_g)
            grows.append(r)

    # ---- make sure every referenced textbook page exists as an image, then copy everything
    need_pages = {int(p[7:11]) for r in vrows + grows
                  for p in [r["page_image"], *filter(None, r.get("other_page_images", "").split(";"))] if p}
    missing = [p for p in need_pages if not (cfg.work / "pages" / f"p{p:04d}.png").exists()]
    if missing and cfg.pdf:
        render_pages(cfg.pdf, missing, cfg.work / "pages", cfg.vlm.dpi, cfg.vlm.max_side)
    sources = {"pages": cfg.work / "pages", "audio": media, "jisho": cfg.work / "jisho"}
    for r in vrows + grows:
        for col in ("page_image", "other_page_images", "audio_file", "jisho_screenshot"):
            kept = []
            for rel in filter(None, str(r.get(col, "")).split(";")):
                top, _, name = rel.partition("/")
                if _copy(sources[top] / name, out / rel):
                    kept.append(rel)
                elif col in ("page_image", "other_page_images"):
                    kept.append(rel)
                    print(f"[export] warning: textbook page image missing: {rel}")
            if col in r:
                r[col] = ";".join(kept)

    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "vocab.csv", VOCAB_COLUMNS, vrows)
    _write_csv(out / "grammar.csv", GRAMMAR_COLUMNS, grows)
    nv = sum(r["needs_verification"] == "TRUE" for r in vrows)
    ng = sum(r["needs_verification"] == "TRUE" for r in grows)
    na = sum(r["needs_audio_check"] == "TRUE" for r in vrows + grows)
    has_a = sum(bool(r["audio_file"]) for r in vrows)
    print(f"[export] {out}/vocab.csv: {len(vrows)} rows ({nv} need verification, "
          f"{has_a} with audio); grammar.csv: {len(grows)} rows ({ng} need verification); "
          f"{na} clips worth a listen")
    from .viewer import run_viewer
    run_viewer(out)
    return out
