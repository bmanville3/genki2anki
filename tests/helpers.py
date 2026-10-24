"""Test helpers: synthetic audio, fake ASR output, mock HTTP servers."""
from __future__ import annotations

import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

SR = 16000


def write_speechlike_wav(path: Path, bursts: list[tuple[float, float]], total_s: float) -> None:
    """16 kHz mono wav: faint noise floor + loud tone bursts at the given (start, end) times."""
    rng = np.random.default_rng(0)
    n = int(total_s * SR)
    x = rng.normal(0, 0.0005, n)
    for a, b in bursts:
        i, j = int(a * SR), int(b * SR)
        t = np.arange(j - i) / SR
        env = np.minimum(1.0, np.minimum(t / 0.02, (b - a - t) / 0.02))
        x[i:j] += 0.3 * np.sin(2 * np.pi * 220 * t) * env
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def fake_asr(words: list[tuple[str, float, float]], wav: Path, source: Path | None = None) -> dict:
    """One segment per word, one whisper-style 'word' each."""
    segs = [{"start": a, "end": b, "text": t,
             "words": [{"word": t, "start": a, "end": b, "probability": 0.9}]}
            for t, a, b in words]
    return {"source": str(source or wav), "wav": str(wav), "duration": 0.0, "segments": segs}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _send(self, obj, code=200):
        if isinstance(obj, str):
            body, ctype = obj.encode(), "text/html; charset=utf-8"
        else:
            body, ctype = json.dumps(obj).encode(), "application/json"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import unquote
        self.server.requests.append((self.path, None))
        code, obj = self.server.respond(unquote(self.path), None)
        self._send(obj, code)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        self.server.requests.append((self.path, payload))
        code, obj = self.server.respond(self.path, payload)
        self._send(obj, code)


def start_server(respond) -> tuple[HTTPServer, str]:
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    srv.requests = []
    srv.respond = respond
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"
