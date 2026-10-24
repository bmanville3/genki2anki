"""Configuration dataclasses, YAML loading and dotted-key overrides."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class VLMConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen3-VL-32B-Instruct"
    api_key: str = "EMPTY"
    workers: int = 16
    max_tokens: int = 7000
    temperature: float = 0.0
    timeout: int = 600
    retries: int = 3
    # json_schema | guided_json | none  (how to ask vLLM for structured output)
    structured_output: str = "json_schema"
    dpi: int = 170
    max_side: int = 2200


@dataclass
class ASRConfig:
    model: str = "large-v3"          # any faster-whisper model name or local CTranslate2 dir
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "ja"
    beam_size: int = 5
    vad: bool = True
    initial_prompt: Optional[str] = None
    download_root: Optional[str] = None
    chunk_s: float = 600.0           # transcribe in ~10 min pieces; each piece is checkpointed
    chunk_search_s: float = 30.0     # chunk edges snap to the longest pause within +-this
    extensions: list = field(default_factory=lambda: [
        ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".wma", ".mp4", ".opus"])


@dataclass
class AlignConfig:
    window_chars: int = 120          # window size (kana chars) used to assign audio to lessons
    monotone_lessons: bool = True    # audio is in book order (lesson N before lesson N+1)
    min_similarity: float = 0.72     # fuzzy-match threshold for a vocab word vs. transcript
    recover_similarity: float = 0.85 # stricter threshold for the order-free recovery pass
    max_gap_s: float = 2.5           # max silence *inside* a matched span (rejects bridging)
    isolation_pause_s: float = 0.20  # pause needed around a word to call it "isolated"
    tight_slack_s: float = 0.60      # energy-run may exceed ASR span by this much to be "tight"
    pad_before_s: float = 0.06
    pad_after_s: float = 0.10
    max_clip_s: float = 8.0
    align_examples: bool = False     # also try to find grammar example sentences in the audio
    require_tight: bool = False      # if True, only keep clips whose edges sit in real silence
    clip_workers: int = 8
    confident_similarity: float = 0.85   # below this (or not tight) a clip is flagged for a listen


@dataclass
class TTSConfig:
    backend: str = "none"            # none | voicevox | command
    voicevox_url: str = "http://127.0.0.1:50021"
    voicevox_speaker: int = 3
    # command backend: shell-style template, e.g. "my_tts --text {text} --out {out}"
    # {out} must be written as a .wav or .mp3 file.
    command: str = ""
    vocab: bool = True               # fill missing vocab audio
    grammar: bool = True             # make audio for grammar example sentences
    workers: int = 4


@dataclass
class JishoConfig:
    enabled: bool = True             # look every vocab word up on Jisho to verify/complete it
    api_url: str = "https://jisho.org/api/v1/search/words"
    web_url: str = "https://jisho.org/search"
    delay_s: float = 1.2             # minimum seconds between requests (be polite)
    timeout: int = 30
    screenshots: bool = True         # save a screenshot of the Jisho result (needs playwright)
    chromium_path: Optional[str] = None   # optional explicit chromium executable
    viewport_width: int = 1100
    viewport_height: int = 900
    user_agent: str = "genki2anki/0.2 (personal study tool)"
    max_consecutive_errors: int = 5  # stop the stage after this many failed lookups in a row


@dataclass
class NormalizeConfig:
    enabled: bool = True             # rewrite example sentences in standard orthography (LLM)
    workers: int = 16


@dataclass
class Config:
    work_dir: str = "work"
    pdf: Optional[str] = None
    audio_dir: Optional[str] = None
    out_dir: str = "genki_export"
    pages: Optional[str] = None      # e.g. "12-380" (1-based, inclusive)
    skip_kinds: list = field(default_factory=lambda: ["exercise", "index", "front_matter"])
    vlm: VLMConfig = field(default_factory=VLMConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    jisho: JishoConfig = field(default_factory=JishoConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)

    @property
    def work(self) -> Path:
        return Path(self.work_dir)


def _apply(dc: Any, data: dict) -> None:
    for key, value in (data or {}).items():
        if not hasattr(dc, key):
            raise ValueError(f"Unknown config key: {key!r} (in {type(dc).__name__})")
        cur = getattr(dc, key)
        if dataclasses.is_dataclass(cur):
            _apply(cur, value)
        else:
            setattr(dc, key, value)


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> Config:
    cfg = Config()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            _apply(cfg, yaml.safe_load(f) or {})
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        parts = dotted.split(".")
        target: Any = cfg
        for p in parts[:-1]:
            target = getattr(target, p)
        if not hasattr(target, parts[-1]):
            raise ValueError(f"Unknown config key: {dotted!r}")
        setattr(target, parts[-1], value)
    return cfg
