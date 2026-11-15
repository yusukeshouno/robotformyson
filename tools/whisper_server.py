"""イモの耳を正確にする書き起こしサーバー（Mac mini などで動かす）。

Pi は Vosk で「話しはじめ・話し終わり」を見つけ、その間の音（WAV）をここに送る。
ここで Whisper が日本語の文字にして返す。家の中だけで動き、音は外に出ない。

    # Apple シリコンの Mac（いちばん速い）
    python3 -m venv ~/whisper-venv && ~/whisper-venv/bin/pip install mlx-whisper numpy
    ~/whisper-venv/bin/python tools/whisper_server.py

    # ほかのパソコン（Linux / Windows / Intel Mac）
    pip install faster-whisper numpy && python3 tools/whisper_server.py --engine faster

    curl http://localhost:8765/health   # {"ok": true, ...} なら動いている

Pi 側は イモの起動に --whisper-url http://Mac-mini.local:8765 を付ける（または ROBOT_WHISPER_URL）。
"""
import argparse
import io
import json
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

RATE = 16000
# 音がほとんど無いときに Whisper がでっちあげやすい文（動画の字幕で覚えたもの）
HALLUCINATIONS = ("ご視聴ありがとうございました", "ご視聴ありがとうございます", "チャンネル登録",
                  "おやすみなさい。おやすみなさい", "字幕", "ありがとうございました。ありがとうございました",
                  "次回予告", "お疲れ様でした。お疲れ様でした")
PROMPT = "イモ、こんにちは。うん、いいよ。ねえねえ、きいて。"  # 名前と、子どもの話し言葉に寄せる


def read_wav(data):
    """WAV を 16kHz・モノラル・float32 にする。"""
    with wave.open(io.BytesIO(data)) as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        pcm = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("16bit の WAV だけ受けつけます")
    x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != RATE and len(x):
        n = int(len(x) * RATE / rate)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def clean(text):
    t = text.strip()
    for h in HALLUCINATIONS:
        if h in t:
            return ""
    return t


class Engine:
    def __init__(self, kind, model):
        self.kind = kind
        self.lock = threading.Lock()
        if kind == "mlx":
            import mlx_whisper
            self.mlx = mlx_whisper
            self.model = model or "mlx-community/whisper-large-v3-turbo"
            self.transcribe(np.zeros(RATE, np.float32))  # 最初の1回でモデルをダウンロード・読みこみ
        else:
            from faster_whisper import WhisperModel
            self.model = model or "large-v3-turbo"
            self.fw = WhisperModel(self.model, compute_type="int8")

    def transcribe(self, audio):
        with self.lock:
            if self.kind == "mlx":
                r = self.mlx.transcribe(audio, path_or_hf_repo=self.model, language="ja",
                                        temperature=0.0, condition_on_previous_text=False,
                                        initial_prompt=PROMPT, verbose=None)
                segs = r.get("segments", [])
                return "".join(s["text"] for s in segs if s.get("no_speech_prob", 0) < 0.6)
            segs, _ = self.fw.transcribe(audio, language="ja", beam_size=1, temperature=0.0,
                                         condition_on_previous_text=False, initial_prompt=PROMPT,
                                         vad_filter=False)
            return "".join(s.text for s in segs if s.no_speech_prob < 0.6)


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/health"):
                self._send(200, {"ok": True, "engine": engine.kind, "model": engine.model})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith("/transcribe"):
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0 or n > 20_000_000:
                return self._send(400, {"error": "音がありません"})
            try:
                audio = read_wav(self.rfile.read(n))
            except (ValueError, wave.Error, EOFError) as e:
                return self._send(400, {"error": str(e)})
            t0 = time.monotonic()
            text = clean(engine.transcribe(audio))
            print(f"[{len(audio) / RATE:4.1f}秒 → {time.monotonic() - t0:4.2f}秒] {text}", flush=True)
            self._send(200, {"text": text})

        def log_message(self, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description="イモの耳（Whisper 書き起こしサーバー）")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--engine", choices=["mlx", "faster"], default="mlx",
                    help="mlx = Apple シリコンの Mac / faster = ほかのパソコン")
    ap.add_argument("--model", default=None, help="モデル名（ふつうは large-v3-turbo）")
    args = ap.parse_args()
    print("モデルを読みこんでいます（初回はダウンロードで数分かかります）…", flush=True)
    engine = Engine(args.engine, args.model)
    print(f"イモの耳: http://{args.host}:{args.port}  ({engine.kind}, {engine.model})", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(engine)).serve_forever()


if __name__ == "__main__":
    main()
