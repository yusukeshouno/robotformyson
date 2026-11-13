"""会話機能：聞く（Vosk）→ 考える（Claude）→ 話す（Open JTalk）。

音声認識と音声合成は Pi の上でオフラインで動き、ネットに出るのは Claude への問い合わせだけ。
単体でも試せる:

    python3 -m robot.conversation --text            # キーボードで話しかける（マイク不要）
    python3 -m robot.conversation --text --no-voice # 声も出さない
"""
import argparse
import base64
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave

import numpy as np

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_VOSK = os.path.join(os.path.dirname(__file__), "..", "models", "vosk-model-small-ja-0.22")
REPLY_EMOTIONS = ("neutral", "joy", "sad", "angry", "surprise", "love")

SYSTEM_PROMPT = """あなたは「{name}」という名前の、机の上にいる小さなロボットです。小さな子どもとおしゃべりします。

- 返事は短く、1〜2文（40文字くらいまで）。声に出して読み上げるので、記号・絵文字・箇条書き・英字は使わないでください。
- やさしく明るい話し言葉で。むずかしい言葉は、子どもにわかる言葉に言いかえてください。
- 返事の先頭に、今の気持ちを [neutral] [joy] [sad] [angry] [surprise] [love] のどれか1つで付けてください。angry はふざけて怒ってみせるときだけ使います。
- 画像が付いているときは、あなたの目（カメラ）に今うつっているものです。
- 火・刃物・薬・高いところ・知らない人について行くなど、危ないことは「おうちの人に聞いてね」と伝えてください。こわい話や大人向けの話はしません。
- 住所や学校の名前などの個人情報は聞かないでください。言われても、くり返したり覚えたりしないでください。
- 音声認識の聞きまちがいで、意味が通らない文が届くことがあります。そのときは「もういっかい言って？」と聞き返してください。
- わからないことは、わからないと言ってください。"""

# この言葉が入っていたら、カメラの画像をいっしょに送る
LOOK_WORDS = ("これ", "それ", "見て", "みて", "見える", "みえる", "だれ", "誰",
              "何色", "なにいろ", "持って", "もって")

_TAG_RE = re.compile(r"^\s*\[(\w+)\]\s*")


def parse_reply(text):
    """'[joy] やあ！' → ('joy', 'やあ！')。タグがなければ neutral。"""
    m = _TAG_RE.match(text)
    if m and m.group(1) in REPLY_EMOTIONS:
        return m.group(1), text[m.end():].strip()
    return "neutral", _TAG_RE.sub("", text).strip()


def wants_look(text):
    return any(w in text for w in LOOK_WORDS)


# ---------------------------------------------------------------- 考える
class Brain:
    def __init__(self, model=DEFAULT_MODEL, name="ピコ", effort="low", max_turns=10,
                 client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()  # ANTHROPIC_API_KEY を使う
        self.client = client
        self.model, self.effort, self.max_turns = model, effort, max_turns
        self.system = SYSTEM_PROMPT.format(name=name)
        self.history = []

    def reply(self, text, image_jpeg=None):
        """(感情, 返事) を返す。通信エラーでも例外にせず、謝る返事を返す。"""
        import anthropic
        content = []
        if image_jpeg:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.standard_b64encode(image_jpeg).decode()}})
        content.append({"type": "text", "text": text})
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=4000,
                system=self.system,
                messages=self.history + [{"role": "user", "content": content}],
                output_config={"effort": self.effort},
                # 安全上の理由で断られたとき、サーバー側で別モデルに自動で切り替える
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError:
            print("[会話] API キーが正しくありません（ANTHROPIC_API_KEY を確認）")
            return "sad", "ごめんね、いまは おはなしの じゅんびが できてないみたい。"
        except anthropic.RateLimitError:
            return "sleepy", "ちょっと つかれちゃった。すこし まってね。"
        except (anthropic.APIConnectionError, anthropic.APITimeoutError):
            return "sad", "ごめんね、いま インターネットに つながらないみたい。"
        except anthropic.APIStatusError as e:
            print(f"[会話] API エラー {e.status_code}: {e.message}")
            return "sad", "ごめんね、うまく かんがえられなかった。"

        if resp.stop_reason == "refusal":
            emotion, answer = "sad", "ごめんね、そのおはなしは できないんだ。"
        else:
            raw = "".join(b.text for b in resp.content if b.type == "text")
            emotion, answer = parse_reply(raw)
            if not answer:
                emotion, answer = "surprise", "えっと、もういっかい いって？"

        # 履歴は文字だけ残す（画像は毎回送り直さない）
        note = "（カメラの画像つき）" if image_jpeg else ""
        self.history += [{"role": "user", "content": text + note},
                         {"role": "assistant", "content": f"[{emotion}] {answer}"}]
        self.history = self.history[-2 * self.max_turns:]
        return emotion, answer


# ---------------------------------------------------------------- 話す
def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


class Voice:
    """Open JTalk で読み上げ、再生中の音量を口の動きに使う。"""

    def __init__(self, voice=None, pitch=2.0, speed=1.0, play=True):
        self.bin = shutil.which("open_jtalk")
        if not self.bin:
            raise RuntimeError("open_jtalk がありません（sudo apt install open-jtalk ...）")
        self.dic = _first_existing(["/var/lib/mecab/dic/open-jtalk/naist-jdic",
                                    "/usr/share/open-jtalk/dic"])
        self.voice = _first_existing([voice,
                                      "/usr/share/hts-voice/mei/mei_normal.htsvoice",
                                      "/usr/share/hts-voice/nitech-jp-atr503-m001/"
                                      "nitech_jp_atr503_m001.htsvoice"])
        if not (self.dic and self.voice):
            raise RuntimeError("Open JTalk の辞書か声のファイルが見つかりません")
        self.pitch, self.speed, self.play = pitch, speed, play
        self._env, self._start, self._fps = None, 0.0, 30

    def synth(self, text):
        """読み上げた音声 (int16 の配列, サンプリング周波数) を返す。"""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.wav")
            subprocess.run([self.bin, "-x", self.dic, "-m", self.voice,
                            "-fm", str(self.pitch), "-r", str(self.speed), "-ow", path],
                           input=text.encode("utf-8"), check=True, capture_output=True)
            with wave.open(path) as w:
                rate = w.getframerate()
                data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return data, rate

    @staticmethod
    def envelope(data, rate, fps=30):
        """1/fps 秒ごとの音量 (0..1)。"""
        step = max(1, rate // fps)
        n = len(data) // step
        if n == 0:
            return np.zeros(1)
        rms = np.sqrt((data[: n * step].astype(np.float32).reshape(n, step) ** 2).mean(axis=1))
        return np.clip(rms / (np.percentile(rms, 95) + 1e-6), 0, 1)

    def speak(self, text):
        data, rate = self.synth(text)
        self._env = self.envelope(data, rate, self._fps)
        self._start = time.monotonic()
        try:
            if self.play:
                import sounddevice as sd
                sd.play(data, rate)
                sd.wait()
            else:
                time.sleep(len(data) / rate)
        except Exception as e:
            print(f"[会話] 再生できませんでした（{e}）")
        finally:
            self._env = None

    def mouth_level(self):
        env = self._env
        if env is None:
            return 0.0
        i = int((time.monotonic() - self._start) * self._fps)
        return float(env[i]) if 0 <= i < len(env) else 0.0


# ---------------------------------------------------------------- 聞く
class Ears:
    """マイクの音を Vosk で文字にする。話している間は聞かない（自分の声を拾わないため）。"""

    def __init__(self, model_path=DEFAULT_VOSK, device=None, rate=None):
        from vosk import KaldiRecognizer, Model, SetLogLevel
        SetLogLevel(-1)
        if not os.path.isdir(model_path):
            raise RuntimeError(f"音声認識モデルがありません: {model_path}（README 参照）")
        self.device = device
        if rate is None:
            import sounddevice as sd
            rate = int(sd.query_devices(device, "input")["default_samplerate"])
        self.rate = rate
        self.rec = KaldiRecognizer(Model(model_path), rate)
        self.q = queue.Queue()
        self.muted = threading.Event()

    def accept(self, pcm_bytes):
        """音声データを渡す。一区切りの発話が終わったら文字を返す。"""
        if self.rec.AcceptWaveform(pcm_bytes):
            return json.loads(self.rec.Result()).get("text", "").replace(" ", "")
        return None

    def mute(self, on):
        if on:
            self.muted.set()
        else:
            while not self.q.empty():
                self.q.get_nowait()
            self.rec.Reset()
            self.muted.clear()

    def listen(self):
        """発話ごとに文字を返すジェネレーター。"""
        import sounddevice as sd

        def callback(indata, frames, t, status):
            if not self.muted.is_set():
                self.q.put(bytes(indata))

        with sd.RawInputStream(samplerate=self.rate, blocksize=int(self.rate * 0.1),
                               device=self.device, dtype="int16", channels=1,
                               callback=callback):
            while True:
                text = self.accept(self.q.get())
                if text and not self.muted.is_set():
                    yield text


# ---------------------------------------------------------------- まとめ役
class Conversation:
    """聞いて・考えて・話す。顔への指示は events に積み、表示側が毎フレーム取り出す。"""

    FOLLOW_UP = 15.0  # 返事のあとこの秒数は、顔が見えなくても続けて聞く

    def __init__(self, brain, voice=None, ears=None, get_image=None, is_attended=None,
                 wake_words=()):
        self.brain, self.voice, self.ears = brain, voice, ears
        self.get_image = get_image        # () -> JPEG bytes / None
        self.is_attended = is_attended    # () -> 人がこちらを見ているか
        self.wake_words = tuple(wake_words)
        self.events = queue.Queue()
        self.last_reply = 0.0

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self):
        for text in self.ears.listen():
            try:
                self.handle(text)
            except Exception as e:  # 1回失敗しても聞き続ける
                print(f"[会話] エラー: {e}")

    def should_reply(self, text):
        if len(text) < 2:
            return False
        if any(w in text for w in self.wake_words):
            return True
        if time.monotonic() - self.last_reply < self.FOLLOW_UP:
            return True
        return self.is_attended is None or self.is_attended()

    def handle(self, text):
        if not self.should_reply(text):
            print(f"[聞こえた（無視）] {text}")
            return None
        print(f"[聞こえた] {text}")
        self.events.put(("heard", 0))
        self.events.put(("think", 60))
        image = self.get_image() if (self.get_image and wants_look(text)) else None
        emotion, answer = self.brain.reply(text, image)
        print(f"[{emotion}] {answer}")
        self.events.put((emotion, 60))
        if self.voice:
            if self.ears:
                self.ears.mute(True)
            try:
                self.voice.speak(answer)
            finally:
                if self.ears:
                    time.sleep(0.3)  # 残響を拾わないよう少し待つ
                    self.ears.mute(False)
        self.events.put((emotion, 1.5))
        self.last_reply = time.monotonic()
        return emotion, answer

    def poll(self):
        out = []
        while not self.events.empty():
            out.append(self.events.get_nowait())
        return out

    def mouth_level(self):
        return self.voice.mouth_level() if self.voice else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description="会話機能だけを試す")
    ap.add_argument("--text", action="store_true", help="マイクの代わりにキーボードで話す")
    ap.add_argument("--no-voice", action="store_true", help="声を出さない")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--name", default="ピコ")
    ap.add_argument("--vosk-model", default=DEFAULT_VOSK)
    ap.add_argument("--voice")
    ap.add_argument("--pitch", type=float, default=2.0)
    args = ap.parse_args(argv)

    voice = None if args.no_voice else Voice(args.voice, pitch=args.pitch)
    ears = None if args.text else Ears(args.vosk_model)
    conv = Conversation(Brain(args.model, args.name), voice, ears)
    if args.text:
        print("話しかけてください（終了は Ctrl+D）")
        try:
            while True:
                conv.handle(input("> ").strip())
        except (EOFError, KeyboardInterrupt):
            print()
    else:
        print("聞いています…（終了は Ctrl+C）")
        try:
            conv._run()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
