"""Command line interface: genki2anki <command> [options].

Every command is safe to interrupt (Ctrl-C once = finish current work and save) and safe to
re-run: finished work is skipped, unfinished work continues where it stopped.
"""
from __future__ import annotations

import argparse
import shutil
import sys

from . import runctl
from .config import Config, load_config


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="YAML config file (see config.example.yaml)")
    p.add_argument("--work", dest="work_dir", help="working directory / saved state (default: ./work)")
    p.add_argument("--pdf", help="path to the scanned Genki PDF")
    p.add_argument("--audio-dir", dest="audio_dir", help="directory with the Genki audio files")
    p.add_argument("--out-dir", dest="out_dir", help="where vocab.csv/grammar.csv go (default: ./genki_export)")
    p.add_argument("--pages", help="page range to process, e.g. 12-380 or 20-24,30")
    p.add_argument("--vlm-url", dest="vlm.base_url", help="vLLM OpenAI-compatible base URL")
    p.add_argument("--vlm-model", dest="vlm.model", help="model name served by vLLM")
    p.add_argument("--vlm-workers", dest="vlm.workers", type=int, help="concurrent page requests")
    p.add_argument("--whisper-model", dest="asr.model", help="faster-whisper model (default large-v3)")
    p.add_argument("--tts", dest="tts.backend", choices=["none", "voicevox", "command"])
    p.add_argument("--tts-command", dest="tts.command")
    p.add_argument("--align-examples", dest="align.align_examples", action="store_true", default=None,
                   help="also try to find grammar example sentences in the audio")
    p.add_argument("--no-jisho", dest="jisho.enabled", action="store_false", default=None,
                   help="skip Jisho lookups (rows are marked not_checked)")
    p.add_argument("--no-screenshots", dest="jisho.screenshots", action="store_false", default=None,
                   help="look words up on Jisho but do not take screenshots")
    p.add_argument("--no-normalize", dest="normalize.enabled", action="store_false", default=None,
                   help="keep example sentences exactly as the textbook prints them")
    p.add_argument("--force", action="store_true", help="redo already-finished work for this stage")


def _cfg(args: argparse.Namespace) -> Config:
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("cmd", "config", "force") and v is not None}
    return load_config(args.config, overrides)


def cmd_doctor(cfg: Config) -> int:
    import importlib

    required = ("pymupdf", "numpy", "requests", "yaml")
    ok = True
    for mod, why in [("pymupdf", "PDF rendering"), ("numpy", ""), ("requests", ""), ("yaml", ""),
                     ("fugashi", "kanji→kana (preferred)"), ("pykakasi", "kanji→kana (fallback), romaji"),
                     ("faster_whisper", "transcription"),
                     ("playwright", "Jisho screenshots")]:
        try:
            importlib.import_module(mod)
            print(f"  ok       {mod}")
        except Exception as e:  # noqa: BLE001
            level = "MISSING " if mod in required else "missing "
            print(f"  {level} {mod}  {why}  ({type(e).__name__})")
            ok = ok and mod not in required
    print(f"  {'ok      ' if shutil.which('ffmpeg') else 'MISSING '} ffmpeg")
    ok = ok and bool(shutil.which("ffmpeg"))
    from . import jp
    print(f"  kana backend: {jp.backend_name()}")
    import requests
    try:
        r = requests.get(cfg.vlm.base_url.rstrip("/") + "/models", timeout=5,
                         headers={"Authorization": f"Bearer {cfg.vlm.api_key}"})
        names = [m["id"] for m in r.json().get("data", [])]
        print(f"  ok       VLM server at {cfg.vlm.base_url}: {names}")
        if cfg.vlm.model not in names:
            print(f"  warning  configured model {cfg.vlm.model!r} is not served")
    except Exception as e:  # noqa: BLE001
        print(f"  not up   VLM server at {cfg.vlm.base_url} ({type(e).__name__}) - start scripts/serve_vlm.sh")
    try:
        r = requests.get(cfg.jisho.api_url, params={"keyword": "学生"}, timeout=10,
                         headers={"User-Agent": cfg.jisho.user_agent})
        print(f"  {'ok      ' if r.ok else 'warning '} jisho API: HTTP {r.status_code}")
    except Exception as e:  # noqa: BLE001
        print(f"  not up   jisho API ({type(e).__name__}) - Jisho stage will be skipped/retried")
    return 0 if ok else 1


COMMANDS = {
    "doctor": "check dependencies and servers",
    "status": "show what is finished and what a re-run would still do",
    "extract": "scanned PDF -> work/book.json via local VLM (per-page checkpoints)",
    "rebuild-book": "re-merge cached page results into book.json (after editing work/extract/*.json)",
    "transcribe": "audio -> word-timestamped transcripts via faster-whisper (per-chunk checkpoints)",
    "align": "match vocab to the audio and cut clips (per-lesson checkpoints)",
    "tts": "fill missing audio with local TTS (voicevox/command)",
    "enrich": "look vocab up on Jisho: verification data + screenshots (per-word checkpoints)",
    "normalize": "rewrite example sentences in standard orthography with the local LLM (checked)",
    "export": "write vocab.csv + grammar.csv + images/audio folders (always safe to re-run)",
    "viewer": "rebuild genki_export/viewer.html from the CSVs",
    "run": "extract -> normalize -> transcribe -> align -> tts -> enrich -> export",
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="genki2anki", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, h in COMMANDS.items():
        _add_common(sub.add_parser(name, help=h, description=h))
    args = ap.parse_args(argv)
    cfg = _cfg(args)
    cfg.work.mkdir(parents=True, exist_ok=True)
    runctl.install_signal_handlers()
    force = args.force

    def do_extract():
        from .extract import run_extract
        run_extract(cfg, force)

    def do_normalize():
        from .normalize import run_normalize
        run_normalize(cfg, force)

    def do_transcribe():
        from .asr import run_transcribe
        run_transcribe(cfg, force)

    def do_align():
        from .audio import run_align
        run_align(cfg, force)

    def do_tts():
        from .tts import run_tts
        run_tts(cfg)

    def do_enrich():
        from .enrich import run_enrich
        run_enrich(cfg, force)

    def do_export():
        from .export import run_export
        run_export(cfg)

    if args.cmd == "doctor":
        sys.exit(cmd_doctor(cfg))
    elif args.cmd == "status":
        from .status import run_status
        run_status(cfg)
    elif args.cmd == "viewer":
        from .viewer import run_viewer
        run_viewer(cfg.out_dir)
    elif args.cmd == "rebuild-book":
        from .extract import rebuild_book
        rebuild_book(cfg)
    elif args.cmd == "run":
        steps = []
        if cfg.pdf:
            steps.append(("extract", do_extract))
            steps.append(("normalize", do_normalize))
        else:
            print("[run] no --pdf given: using the existing work/book.json")
        if cfg.audio_dir:
            steps += [("transcribe", do_transcribe), ("align", do_align)]
        else:
            print("[run] no --audio-dir given: skipping transcription/alignment")
        if cfg.tts.backend != "none":
            steps.append(("tts", do_tts))
        steps += [("enrich", do_enrich), ("export", do_export)]
        for name, fn in steps:
            if runctl.stopping():
                break
            print(f"\n=== {name} ===")
            fn()
        if runctl.stopping():
            print("\n[run] stopped. State is saved - re-run the same command to resume.")
            sys.exit(130)
    else:
        {"extract": do_extract, "normalize": do_normalize, "transcribe": do_transcribe,
         "align": do_align, "tts": do_tts, "enrich": do_enrich, "export": do_export}[args.cmd]()
        if runctl.stopping():
            print("[stop] state saved - re-run the same command to resume.")
            sys.exit(130)


if __name__ == "__main__":
    main()
