"""PDF -> page images."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _fitz():
    try:
        import pymupdf as fitz  # newer name
    except ImportError:  # pragma: no cover
        import fitz  # type: ignore
    return fitz


def page_count(pdf: str) -> int:
    fitz = _fitz()
    with fitz.open(pdf) as doc:
        return doc.page_count


def parse_pages(spec: Optional[str], n_pages: int) -> list[int]:
    """'12-30,35' -> [12..30, 35] (1-based, inclusive). None -> all pages."""
    if not spec:
        return list(range(1, n_pages + 1))
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo = int(a) if a else 1
            hi = int(b) if b else n_pages
            pages.extend(range(lo, hi + 1))
        else:
            pages.append(int(part))
    return sorted({p for p in pages if 1 <= p <= n_pages})


def render_pages(pdf: str, pages: list[int], out_dir: Path, dpi: int = 170,
                 max_side: int = 2200) -> dict[int, Path]:
    """Render the given 1-based pages to PNG. Existing images are reused (writes are atomic)."""
    fitz = _fitz()
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[int, Path] = {}
    with fitz.open(pdf) as doc:
        for p in pages:
            path = out_dir / f"p{p:04d}.png"
            if not path.exists():
                page = doc[p - 1]
                zoom = dpi / 72.0
                rect = page.rect
                longest = max(rect.width, rect.height) * zoom
                if longest > max_side:
                    zoom *= max_side / longest
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                tmp = out_dir / f".p{p:04d}.{os.getpid()}.tmp.png"
                pix.save(str(tmp))
                os.replace(tmp, path)
            result[p] = path
    return result
