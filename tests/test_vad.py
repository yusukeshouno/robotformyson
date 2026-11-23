"""話の区切り（Silero VAD）のテスト。"""
import json

import numpy as np
import pytest

from robot import conversation as cv
from robot import vad
from robot.audio import MicGate


class LoudVAD:
    """音の大きさで「声らしさ」を決めるにせもの。"""

    def reset(self):
        pass

    def prob(self, frame):
        return 1.0 if np.abs(frame).mean() > 0.01 else 0.0


def chunk(level, secs=0.1, rate=16000):
    return (np.ones(int(rate * secs)) * level).astype(np.int16).tobytes()


def test_segmenter_returns_utterance_with_preroll():
    seg = vad.Segmenter(LoudVAD())
    out = [seg.feed(chunk(50)) for _ in range(10)]          # 静か
    out += [seg.feed(chunk(2000)) for _ in range(5)]        # 0.5 秒の声
    out += [seg.feed(chunk(50)) for _ in range(10)]         # 0.7 秒だまると終わり
    got = [a for a in out if a]
    assert len(got) == 1
    secs = len(got[0]) / 2 / 16000
    assert 0.5 + 0.7 <= secs <= 0.5 + 0.7 + vad.Segmenter.PRE_ROLL + 0.2


def test_segmenter_drops_clicks_and_handles_48k():
    seg = vad.Segmenter(LoudVAD())
    out = [seg.feed(chunk(2000, 0.05))] + [seg.feed(chunk(50)) for _ in range(10)]
    assert not any(out)                                     # 0.05 秒の物音は捨てる
    seg = vad.Segmenter(LoudVAD(), rate=48000)
    out = [seg.feed(chunk(2000, rate=48000)) for _ in range(5)]
    out += [seg.feed(chunk(50, rate=48000)) for _ in range(10)]
    assert sum(1 for a in out if a) == 1


class FakeRec:
    def __init__(self):
        self.final = "うん"

    def AcceptWaveform(self, data):
        return False

    def FinalResult(self):
        return json.dumps({"text": self.final})

    def Reset(self):
        pass


def test_ears_with_vad_sends_quiet_voice_and_keeps_rough_text():
    sent, unclear = [], []
    ears = cv.Ears(rate=16000, gate=MicGate(tail=0), recognizer=FakeRec(),
                   transcriber=lambda pcm, rate: sent.append(len(pcm)) or "",
                   segmenter=vad.Segmenter(LoudVAD()))
    ears.on_unclear = unclear.append
    for c in [chunk(50)] * 5 + [chunk(500)] * 4 + [chunk(50)] * 8:
        assert ears.accept(c) is None
    assert len(sent) == 1 and unclear == ["うん"]           # Whisper が空でも Vosk の当て推量を渡す


def test_ears_with_vad_falls_back_to_vosk_when_whisper_down():
    ears = cv.Ears(rate=16000, gate=MicGate(tail=0), recognizer=FakeRec(),
                   transcriber=lambda pcm, rate: None, segmenter=vad.Segmenter(LoudVAD()))
    out = [ears.accept(c) for c in [chunk(500)] * 4 + [chunk(50)] * 8]
    assert "うん" in out


def test_real_silero_model_is_quiet_on_noise():
    pytest.importorskip("onnxruntime")
    seg = vad.Segmenter.create(16000)
    assert seg is not None
    rng = np.random.default_rng(0)
    out = [seg.feed((rng.standard_normal(1600) * 150).astype(np.int16).tobytes()) for _ in range(30)]
    assert not any(out) and seg.vad.prob(np.zeros(512, np.float32)) < 0.2


def test_create_without_model_returns_none(tmp_path):
    assert vad.Segmenter.create(16000, path=str(tmp_path / "none.onnx")) is None


def test_listen_reopens_stream_when_audio_stops(monkeypatch):
    """実機: 数時間後に何も聞かなくなった。音がこなければ録音を開きなおす。"""
    import sys
    import types

    opened = []

    class Stream:
        def __init__(self, callback=None, **kw):
            self.cb = callback
            opened.append(self)

        def __enter__(self):
            if len(opened) >= 2:            # 2回目に開いたときだけ音がくる
                self.cb(b"\x00\x00" * 1600, 1600, None, None)
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(RawInputStream=Stream))
    ears = cv.Ears(rate=16000, gate=MicGate(tail=0), recognizer=FakeRec(), transcriber=None)
    ears.STALL = 0.05
    ears.process = lambda item: "きこえた"
    assert next(ears.listen()) == "きこえた"
    assert len(opened) == 2


def test_vad_state_is_reset_while_idle():
    from robot.vad import Segmenter

    class V:
        resets = 0
        def reset(self):
            V.resets += 1
        def prob(self, frame):
            return 0.0

    seg = Segmenter(V(), rate=16000)
    start = V.resets
    chunk = np.zeros(1600, np.int16).tobytes()   # 0.1秒
    for _ in range(int(Segmenter.IDLE_RESET / 0.1) + 5):
        seg.feed(chunk)
    assert V.resets > start                      # 静かなあいだに記憶を消した
    assert seg.stats() == (0.0, 0.0)
