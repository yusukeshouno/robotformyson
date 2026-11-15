"""Whisper で書き起こす経路（Ears の transcriber・WhisperClient・サーバーの WAV 読みこみ）のテスト。"""
import importlib.util
import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

from robot import conversation as cv
from robot.audio import MicGate


class FakeRec:
    def __init__(self, finals):
        self.finals = list(finals)   # AcceptWaveform を呼ぶたびに 1 つ取り出す（None=途中）
        self.cur = None

    def AcceptWaveform(self, data):
        self.cur = self.finals.pop(0)
        return self.cur is not None

    def Result(self):
        return json.dumps({"text": self.cur})

    def Reset(self):
        pass


def test_ears_uses_transcriber_text_and_whole_utterance():
    got = []

    def tr(pcm, rate):
        got.append((len(pcm), rate))
        return "イモ りんご すき"

    ears = cv.Ears(rate=16000, gate=MicGate(tail=0), recognizer=FakeRec([None, None, "いも りん"]),
                   transcriber=tr)
    chunk = b"\x01\x00" * 1600
    assert ears.accept(chunk) is None and ears.accept(chunk) is None
    assert ears.accept(chunk) == "イモ りんご すき"
    assert got == [(len(chunk) * 3, 16000)]     # 話しはじめから終わりまでを送る
    assert len(ears.buf) == 0


def test_ears_sends_only_preroll_before_speech():
    class PartialRec(FakeRec):
        def PartialResult(self):
            return json.dumps({"partial": "" if self.cur is None and self.n < 20 else "あ"})

        def AcceptWaveform(self, data):
            self.n = getattr(self, "n", 0) + 1
            return super().AcceptWaveform(data)

    got = []
    ears = cv.Ears(rate=16000, gate=MicGate(tail=0),
                   recognizer=PartialRec([None] * 25 + ["あいう"]),
                   transcriber=lambda pcm, rate: got.append(len(pcm)) or "あいう")
    chunk = b"\x01\x00" * 1600          # 0.1 秒
    for _ in range(26):
        ears.accept(chunk)
    # 静かな 19 個ぶんは捨て、話しはじめ前の 0.6 秒 ＋ 話している 0.7 秒くらいだけ送る
    assert got and got[0] <= len(chunk) * 14


def test_ears_skips_transcriber_on_silence_and_falls_back():
    calls = []
    ears = cv.Ears(rate=16000, gate=MicGate(tail=0), recognizer=FakeRec(["", "こんにちは"]),
                   transcriber=lambda pcm, rate: calls.append(1) or None)
    assert ears.accept(b"\x00\x00" * 10) is None   # Vosk が何も聞いていない → 送らない
    assert calls == []
    assert ears.accept(b"\x00\x00" * 10) == "こんにちは"  # Whisper がだめなら Vosk の文字
    assert calls == [1]


def _serve(handler_cls):
    srv = HTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_whisper_client_roundtrip_and_backoff():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            data = self.rfile.read(int(self.headers["Content-Length"]))
            with wave.open(io.BytesIO(data)) as w:
                seen["rate"] = w.getframerate()
            body = json.dumps({"text": " うん、 いいよ "}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = _serve(H)
    c = cv.WhisperClient(f"http://127.0.0.1:{srv.server_port}")
    assert c(b"\x00\x00" * 100, 16000) == "うん、いいよ"
    assert seen["rate"] == 16000
    srv.shutdown()

    now = [0.0]
    down = cv.WhisperClient("http://127.0.0.1:9", timeout=0.5, clock=lambda: now[0])
    assert down(b"\x00\x00", 16000) is None
    assert down.down_until == down.RETRY        # しばらく送らない
    now[0] = 1.0
    assert down(b"\x00\x00", 16000) is None


def test_server_reads_and_resamples_wav():
    path = Path(__file__).resolve().parent.parent / "tools" / "whisper_server.py"
    spec = importlib.util.spec_from_file_location("ws", path)
    ws = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ws)
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes((np.ones(44100) * 1000).astype(np.int16).tobytes())
    audio = ws.read_wav(out.getvalue())
    assert audio.dtype == np.float32 and abs(len(audio) - 16000) <= 1
    assert ws.clean("ご視聴ありがとうございました") == ""
    assert ws.clean(" うん ") == "うん"
    assert ws.clean("おいしいよ。おいしいよ。おいしいよ。おいしいよ。") == ""
    assert ws.clean("うれば、うれば、うれば、うれば、うれば") == ""
    assert ws.clean("イモ。") == ""
    assert ws.clean("イモ、りんごはすき？") == "イモ、りんごはすき？"
    assert ws.voiced_seconds(np.zeros(16000, np.float32)) == 0.0
    tone = np.zeros(32000, np.float32)
    tone[8000:16000] = 0.3 * np.sin(np.arange(8000) / 5)
    assert 0.4 < ws.voiced_seconds(tone) < 0.6
