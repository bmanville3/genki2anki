# genki2anki

Local pipeline: **scanned Genki PDF + audio → two CSV files** (vocab, grammar) that you inspect
and filter yourself, with the source textbook page, a third-party (Jisho) screenshot and the
audio clip saved next to every row. Nothing is sent anywhere except lookups to jisho.org.

A vision-language model served by vLLM reads the page images, faster-whisper transcribes the
audio, and the rest is plain Python + ffmpeg.

```
PDF pages ──render──> PNG ──vLLM VLM──> per-page JSON ──merge──> work/book.json
                                                  │
audio ──ffmpeg + faster-whisper (10-min chunks)──> transcripts ──align──> clips
                                                  │
book.json ──Jisho API + screenshot──> work/enrich/*.json
example sentences ──local LLM (reading-checked)──> work/normalize/*.json
                                                  ▼
                              export ──> genki_export/vocab.csv, grammar.csv, pages/, audio/, jisho/
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[asr,screenshots]"     # core + faster-whisper + playwright
playwright install chromium             # for the Jisho screenshots
pip install vllm                        # for the VLM server (or use your existing install)
genki2anki doctor                       # checks everything
```
ffmpeg must be on PATH. Optional: `pip install -e ".[fugashi]"` for more accurate kanji readings than the
default pykakasi. Pre-download models and set `HF_HUB_OFFLINE=1` to stay offline for the AI parts.

## Run it

```bash
scripts/serve_vlm.sh                    # terminal 1: Qwen3-VL-32B (FP8) on the A100; or MODEL=<any ≤35B VLM>

# try three pages first - the cheapest place to find a wrong model name or flag
genki2anki extract --pdf genki1.pdf --pages 30-32 --work work
less work/book_vocab.tsv work/book_grammar.tsv

# then everything (it prints progress; Ctrl-C any time)
genki2anki run --pdf genki1.pdf --audio-dir ./genki_audio --work work --out-dir genki_export
genki2anki status --work work           # what is done / what a re-run would still do
```
`run` = `extract → normalize → transcribe → align → (tts) → enrich → export`; each stage is also its own
command. The Jisho stage is the slow one (≈1 request/sec, ~2 per word with screenshots) - it just keeps
going across restarts.

## Stopping and restarting

* **Ctrl-C once**: finish the work in flight, save, exit (code 130). **Twice**: abort immediately.
  Killing the process (`kill -9`, power loss) is also safe: every file is written atomically
  (temp file + rename), so nothing is ever half-written.
* **Re-run the same command** and it continues. The unit of saved progress per stage:

  | Stage | Saved after every… | Where |
  |---|---|---|
  | extract | page | `work/extract/pNNNN.json` (failed pages are retried, finished ones skipped) |
  | normalize | sentence | `work/normalize/*.json` |
  | transcribe | ~10-minute audio chunk | `work/asr/parts/…` (a 3-hour file resumes at the last finished chunk) |
  | align | lesson | `work/align/lesson_NN.json` (re-done automatically if transcripts/book change) |
  | tts | clip (manifest every 25) | `work/tts_manifest.json`, `work/media/` |
  | enrich | word | `work/enrich/*.json`, `work/jisho/*.png`, `work/jisho_cache/` |
  | export | – | re-runs in seconds; only reads `work/` |

* `--force` redoes a finished stage. To fix a bad page by hand, edit `work/extract/pNNNN.json`
  and run `genki2anki rebuild-book`, then `export`.

## The CSVs (`genki_export/`, UTF-8 with BOM so Excel shows Japanese correctly)

Open them in a spreadsheet or load with pandas. All file paths are relative to `genki_export/`:
`pages/pNNNN.png` (the **textbook page the row came from**, always present), `jisho/…png`
(third-party screenshot), `audio/…mp3`.

### `vocab.csv`

| Column(s) | Meaning |
|---|---|
| `front`, `front_script` | The word in its **most common written form** (kanji, hiragana or katakana), regardless of how the book prints it |
| `kanji`, `hiragana`, `katakana`, `furigana`, `romaji`, `english` | All fields. `furigana` is Anki style: ` 食[た]べる` |
| `needs_verification`, `verify_reasons` | **Filter on `needs_verification = TRUE`.** Reasons below |
| `verified`, `notes`, `still_needs_review` | You fill `verified` (TRUE) and `notes`; `still_needs_review` = needs verification and not yet verified. Your entries survive re-exports |
| `jisho_status`, `jisho_word`, `jisho_reading`, `jisho_english`, `jisho_common`, `jlpt`, `jisho_url`, `jisho_screenshot` | What Jisho says, plus the **screenshot to double-check against** |
| `page`, `page_image`, `tb_kanji`, `tb_kana` | Source page and exactly what the book printed |
| `kanji_source`, `reading_source` | `textbook`, `jisho` or `generated` - where each part came from |
| `audio_file`, `audio_source`, `audio_confidence`, `needs_audio_check`, `audio_sim`, `audio_t0/t1`, `audio_src_file` | The clip (`genki` = cut from the real audio, `tts`), how confident the match was, and where in the source audio it came from |

`verify_reasons` (semicolon separated; any reason → `needs_verification = TRUE`):

* `kanji_from_jisho` - the book printed only kana; the kanji spelling came from Jisho.
* `reading_generated` - the book printed only kanji; the reading was generated by a tokenizer.
* `jisho_not_found`, `jisho_form_mismatch` - Jisho has no entry, or its spellings don't match what the book printed (also catches OCR slips).
* `meaning_mismatch` - the Jisho entry shares no words with the book's English (likely the wrong homophone).
* `not_checked` - Jisho stage hasn't run (offline, `--no-jisho`, or still in progress).

Words Jisho marks *usually written using kana alone* get a kana front even if the book prints kanji, and
a kana-only word that Jisho confirms as kana is **not** flagged.

### `grammar.csv`
One row per example sentence: `sentence` (front; most common script), `sentence_textbook` (as printed),
`hiragana`, `furigana`, `romaji`, `english`, `grammar_title`, `grammar_number`, `explanation_text` /
`explanation_html`, `page_image` (+ `other_page_images` for points that span pages), audio columns
(empty unless you add TTS or `--align-examples`), and the same `needs_verification` family.
Sentences are rewritten into standard orthography by the local LLM *only if the rewritten sentence has
exactly the same kana reading* (checked in code); changed ones are flagged `script_changed`, rejected ones
keep the textbook text and are flagged `script_unverified`. Sentence readings/furigana/romaji are generated
by a tokenizer and are **not** flagged individually - skim `hiragana` against the page image.

## Audio matching (short version)
Whisper → one timed kana stream → lessons assigned by vocabulary overlap (monotone) → fuzzy match of each
book word, best *ordered* chain, order-free recovery for the rest → clip edges snapped to real silence.
Rows whose match is weak or not cleanly bounded get `needs_audio_check = TRUE`; `work/report/alignment.tsv`
has the details. Tuning: `align.min_similarity`, `align.require_tight`, `asr.model`.

## Honest limitations
* Tested here with a mock VLM server, synthetic audio, a stubbed Whisper, a mock Jisho server and a real
  headless Chromium - **not** against real Genki scans, real recordings, a real GPU, or the live jisho.org.
  The Jisho JSON fields (`japanese`, `senses`, tags) follow its public API; the screenshot step looks for
  its result blocks (`.concept_light`) and falls back to a viewport screenshot if the page layout differs.
  Do the 3-page trial and `doctor` first.
* Jisho is a free site: the stage is rate-limited (`jisho.delay_s`) and cached; please keep it that way.
* The page image saved per row is the **whole textbook page** (not a crop of the entry).
* Vision-model transcription of printed Japanese is good but not perfect - that is what `jisho_form_mismatch`
  and the page images are for.
* Multi-line dialogue examples are joined into one sentence; pitch accent, furigana and exercises are ignored.
