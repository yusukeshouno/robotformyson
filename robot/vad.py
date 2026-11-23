"""声の区切りを見つける（Silero VAD）。

Vosk の「話しはじめ・話し終わり」は、声が小さいと気づかずに音を捨ててしまう。
Silero VAD は「人の声かどうか」だけを見る小さなモデル（2MB、MIT ライセンス、robot/data/ に同梱）で、
小さな声でも区切りを見つけやすい。onnxruntime で動かす（Pi では pip install onnxruntime）。

    seg = Segmenter.create(rate=16000)   # 使えなければ None
    for chunk in マイクの音:
        audio = seg.feed(chunk)          # 話し終わったら、その発話の音（bytes）。それ以外は None
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "data", "silero_vad.onnx")
VAD_RATE = 16000
FRAME = 512      # Silero VAD が一度に見る長さ（16kHz で 32ms）
CONTEXT = 64     # 前の枠の終わりを少し足して渡す（Silero VAD v5 の決まり）


class SileroVAD:
    """32ms ごとに「人の声らしさ」（0〜1）を返す。"""

    def __init__(self, path=MODEL):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1   # Pi の他の仕事（顔・カメラ）をじゃましない
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path, sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros((1, CONTEXT), np.float32)

    def prob(self, frame):
        """frame: float32（-1〜1）、16kHz、512 サンプル。"""
        x = np.concatenate([self.context, frame.reshape(1, -1).astype(np.float32)], axis=1)
        out, self.state = self.session.run(
            None, {"input": x, "state": self.state, "sr": np.array(VAD_RATE, np.int64)})
        self.context = x[:, -CONTEXT:]
        return float(out[0][0])


class Segmenter:
    """マイクの音を少しずつ受けとり、ひとつの発話が終わったらその音を返す。

    - 声らしさが START を超えたら話しはじめ（その前 PRE_ROLL 秒も入れる。言いはじめが切れないように）
    - END より下が END_SILENCE 秒つづいたら話し終わり
    - 声だった長さが MIN_SPEECH 秒より短ければ捨てる（せき・物音）
    """

    START, END = 0.5, 0.35
    PRE_ROLL = 0.5
    END_SILENCE = 0.7
    MIN_SPEECH = 0.2
    MAX_UTTERANCE = 15.0
    IDLE_RESET = 60.0   # 話していない時間がこの秒数つづいたら、VAD の中の記憶を消す
    # （実機で、起動して1時間ほどすると声らしさが 0.01 未満のまま上がらなくなり、何も聞かなくなった。
    #   Silero VAD は1つの音声ごとに記憶を消して使う作りなので、静かなあいだにときどき消しておく）

    def __init__(self, vad, rate=16000):
        self.vad, self.rate = vad, rate
        self.reset()

    @classmethod
    def create(cls, rate=16000, path=MODEL):
        """使えれば Segmenter、onnxruntime かモデルがなければ None（そのときは Vosk で区切る）。"""
        if not os.path.exists(path):
            return None
        try:
            return cls(SileroVAD(path), rate)
        except Exception as e:  # onnxruntime がない・読めない
            print(f"[耳] Silero VAD を使えません（Vosk で区切ります）: {e}")
            return None

    def reset(self):
        self.vad.reset()
        self.pending = np.zeros(0, np.float32)   # VAD にまだ渡していない音（16kHz）
        self.ring = []                           # 話しはじめる前の音（bytes のかたまり）
        self.buf = []                            # 話している間の音
        self.active = False
        self.speech = self.silence = 0.0
        self.last_prob = 0.0
        self.idle = 0.0
        self.max_prob = self.max_level = 0.0   # stats() で読む（ログで「聞こえているか」を見る）

    def stats(self):
        """前に呼んでからの (声らしさの最大, 音の大きさの最大 0〜1)。呼ぶと0にもどる。"""
        out = (self.max_prob, self.max_level)
        self.max_prob = self.max_level = 0.0
        return out

    @property
    def speaking(self):
        return self.active

    def _probs(self, pcm_bytes):
        x = np.frombuffer(pcm_bytes[:len(pcm_bytes) // 2 * 2], np.int16).astype(np.float32) / 32768
        if self.rate != VAD_RATE and len(x):
            n = int(round(len(x) * VAD_RATE / self.rate))
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
        self.pending = np.concatenate([self.pending, x])
        out = []
        while len(self.pending) >= FRAME:
            out.append(self.vad.prob(self.pending[:FRAME]))
            self.pending = self.pending[FRAME:]
        return out

    def feed(self, pcm_bytes):
        probs = self._probs(pcm_bytes)
        p = max(probs) if probs else self.last_prob
        self.last_prob = p
        dur = len(pcm_bytes) / 2 / self.rate
        x = np.frombuffer(pcm_bytes[:len(pcm_bytes) // 2 * 2], np.int16)
        if len(x):
            self.max_level = max(self.max_level, float(np.abs(x).max()) / 32768)
        self.max_prob = max(self.max_prob, p)
        if not self.active:
            self.idle += dur
            if self.idle >= self.IDLE_RESET:
                self.vad.reset()
                self.idle = 0.0
            self.ring.append(pcm_bytes)
            while sum(len(c) for c in self.ring) / 2 / self.rate > self.PRE_ROLL + dur:
                self.ring.pop(0)
            if p >= self.START:
                self.active, self.buf, self.ring = True, self.ring, []
                self.idle = 0.0
                self.speech, self.silence = dur, 0.0
            return None
        self.buf.append(pcm_bytes)
        if p >= self.END:
            self.speech += dur
            self.silence = 0.0
        else:
            self.silence += dur
        total = sum(len(c) for c in self.buf) / 2 / self.rate
        if self.silence >= self.END_SILENCE or total >= self.MAX_UTTERANCE:
            audio, speech = b"".join(self.buf), self.speech
            self.active, self.buf = False, []
            self.speech = self.silence = 0.0
            if speech >= self.MIN_SPEECH:
                return audio
        return None
