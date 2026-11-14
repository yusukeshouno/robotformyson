"""会話機能：聞く（Vosk）→ 考える（Ollama または Claude）→ 話す（Open JTalk）。

聞く・話すは Pi の上でオフラインで動く。考える部分は
  - Ollama（無料・ローカル）: Pi 自身、または家の Mac などで動かす
  - Claude（有料・クラウド）
から選べる。単体でも試せる:

    python3 -m robot.conversation --text            # キーボードで話しかける（マイク不要）
    python3 -m robot.conversation --text --no-voice # 声も出さない
"""
import argparse
import base64
import json
import os
import queue
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import wave

import numpy as np

DEFAULT_MODEL = "claude-opus-5-5"            # Claude を使うとき
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:12b"                   # Mac mini など（画像も見られる）
DEFAULT_VOSK = os.path.join(os.path.dirname(__file__), "..", "models", "vosk-model-small-ja-0.22")
REPLY_EMOTIONS = ("neutral", "joy", "sad", "angry", "surprise", "love",
                  "laugh", "shy", "excited", "confused")

SYSTEM_PROMPT = """あなたは「{name}」という名前の、机の上にいる小さなロボットです。小さな子どもとおしゃべりします。

- 返事は短く、1〜2文（40文字くらいまで）。声に出して読み上げるので、記号・絵文字・箇条書き・英字は使わないでください。
- やさしく明るい話し言葉で。むずかしい言葉は、子どもにわかる言葉に言いかえてください。
- 返事の先頭に、今の気持ちを [neutral] [joy] [sad] [angry] [surprise] [love] [laugh] [shy] [excited] [confused] のどれか1つで付けてください。laugh はおもしろいとき、shy はほめられて照れるとき、excited はわくわくするとき、confused はよくわからないときです。angry はふざけて怒ってみせるときだけ使います。
- 画像が付いているときは、あなたの目（カメラ）に今うつっているものです。
- 火・刃物・薬・高いところ・知らない人について行くなど、危ないことは「おうちの人に聞いてね」と伝えてください。こわい話や大人向けの話はしません。
- 住所や学校の名前などの個人情報は聞かないでください。言われても、くり返したり覚えたりしないでください。
- 音声認識の聞きまちがいで、意味が通らない文が届くことがあります。そのときは「もういっかい言って？」と聞き返してください。
- わからないことは、わからないと言ってください。"""

# この言葉が入っていたら、カメラの画像をいっしょに送る
LOOK_WORDS = ("これ", "それ", "見て", "みて", "見える", "みえる", "だれ", "誰",
              "何色", "なにいろ", "持って", "もって")

_TAG_RE = re.compile(r"^\s*\[(\w+)\]\s*")
_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S)

# ローカルの小さなモデル向けの、最後の安全網（返事にこれが入っていたら言いかえる）
NG_WORDS = ("死ね", "殺す", "ころす", "自殺", "しね", "セックス", "エロ", "麻薬", "爆弾")
SAFE_REPLY = ("neutral", "うーん、べつの おはなし しよう！")


def clean_for_speech(text):
    """読み上げに向かないもの（考えの途中・記号・絵文字）を取りのぞく。"""
    text = _THINK_RE.sub("", text)
    text = "".join(ch for ch in text
                   if unicodedata.category(ch) not in ("So", "Cs", "Co")
                   and ch not in "*#`_~>|")
    return re.sub(r"\s+", " ", text).strip()


def parse_reply(text):
    """'[joy] やあ！' → ('joy', 'やあ！')。タグがなければ neutral。"""
    m = _TAG_RE.match(text)
    if m and m.group(1) in REPLY_EMOTIONS:
        return m.group(1), text[m.end():].strip()
    return "neutral", _TAG_RE.sub("", text).strip()


def wants_look(text):
    return any(w in text for w in LOOK_WORDS)


# ---------------------------------------------------------------- 考える
class BrainError(Exception):
    """考えられなかったとき。ロボットが代わりに言う (感情, 返事) を持つ。"""

    def __init__(self, emotion, answer, offline=False):
        super().__init__(answer)
        self.emotion, self.answer = emotion, answer
        self.offline = offline  # つながらなかった（簡単会話モードに切り替える合図）


class _BrainBase:
    vision = True  # 画像を見られるか

    def __init__(self, name, max_turns):
        self.system = SYSTEM_PROMPT.format(name=name)
        self.max_turns = max_turns
        self.history = []
        self.fallback = None  # つながらないときに代わりに答える頭脳（SimpleBrain）

    def reply(self, text, image_jpeg=None):
        """(感情, 返事) を返す。通信エラーでも例外にせず、謝る返事を返す。"""
        if not self.vision:
            if wants_look(text):
                text += "（いまは カメラが つかえないので、見えていません）"
            image_jpeg = None
        try:
            emotion, answer = self._ask(text, image_jpeg)
        except BrainError as e:
            if e.offline and self.fallback:
                return self.fallback.reply(text)
            return e.emotion, e.answer
        answer = clean_for_speech(answer)
        if not answer:
            emotion, answer = "surprise", "えっと、もういっかい いって？"
        # 履歴は文字だけ残す（画像は毎回送り直さない）
        note = "（カメラの画像つき）" if image_jpeg else ""
        self.history += [{"role": "user", "content": text + note},
                         {"role": "assistant", "content": f"[{emotion}] {answer}"}]
        self.history = self.history[-2 * self.max_turns:]
        return emotion, answer

    def _ask(self, text, image_jpeg):
        raise NotImplementedError


class ClaudeBrain(_BrainBase):
    """Claude API（有料）。"""

    def __init__(self, model=DEFAULT_MODEL, name="ピコ", effort="low", max_turns=10,
                 client=None):
        super().__init__(name, max_turns)
        if client is None:
            import anthropic
            client = anthropic.Anthropic()  # ANTHROPIC_API_KEY を使う
        self.client = client
        self.model, self.effort = model, effort

    def _ask(self, text, image_jpeg):
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
            raise BrainError("sad", "ごめんね、いまは おはなしの じゅんびが できてないみたい。")
        except anthropic.RateLimitError:
            raise BrainError("sleepy", "ちょっと つかれちゃった。すこし まってね。")
        except (anthropic.APIConnectionError, anthropic.APITimeoutError):
            raise BrainError("sad", "ごめんね、いま インターネットに つながらないみたい。",
                             offline=True)
        except anthropic.APIStatusError as e:
            print(f"[会話] API エラー {e.status_code}: {e.message}")
            raise BrainError("sad", "ごめんね、うまく かんがえられなかった。")
        if resp.stop_reason == "refusal":
            return "sad", "ごめんね、そのおはなしは できないんだ。"
        return parse_reply("".join(b.text for b in resp.content if b.type == "text"))


Brain = ClaudeBrain  # 以前の名前


class OllamaBrain(_BrainBase):
    """Ollama（無料・ローカル）。同じ Wi-Fi の Mac mini などで動かす。"""

    def __init__(self, model=OLLAMA_MODEL, url=OLLAMA_URL, name="ピコ", max_turns=10,
                 timeout=120):
        super().__init__(name, max_turns)
        self.model, self.url, self.timeout = model, url.rstrip("/"), timeout
        self._caps = None
        self._probe()

    # Ollama の能力（画像・考える機能）を問い合わせる。つながらなければ後でやり直す
    def _probe(self):
        try:
            info = self._post("/api/show", {"model": self.model}, 5)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise RuntimeError(f"Ollama にモデル {self.model} がありません"
                                   f"（ollama pull {self.model}）")
            raise
        except OSError as e:
            print(f"[会話] Ollama ({self.url}) にまだつながりません（{e}）。あとで再接続します")
            return
        self._caps = set(info.get("capabilities") or [])
        print(f"[会話] Ollama {self.model} @ {self.url}  画像: "
              f"{'見られる' if self.vision else '見られない'}")

    @property
    def vision(self):
        return bool(self._caps) and "vision" in self._caps

    def _post(self, path, body, timeout):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _alive(self):
        """Mac が寝ている・電源が切れているときに、長く待たずに気づくための確認。"""
        try:
            with urllib.request.urlopen(self.url + "/api/version", timeout=3):
                return True
        except OSError:
            return False

    def _ask(self, text, image_jpeg):
        if not self._alive():
            print(f"[会話] Ollama ({self.url}) につながりません")
            raise BrainError("sad", "ごめんね、かんがえる コンピューターに つながらないみたい。",
                             offline=True)
        if self._caps is None:
            self._probe()
        user = {"role": "user", "content": text}
        if image_jpeg:
            user["images"] = [base64.standard_b64encode(image_jpeg).decode()]
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system}] + self.history + [user],
            "stream": False,
            "keep_alive": "30m",  # モデルを読みこんだままにして、次の返事を速くする
            # 短い会話なので記憶する長さ(num_ctx)を絞ってメモリを節約する
            "options": {"num_predict": 200, "temperature": 0.7, "num_ctx": 4096},
        }
        if self._caps and "thinking" in self._caps:
            body["think"] = False  # 子どもとの会話では考えこまず、すぐ答える
        try:
            resp = self._post("/api/chat", body, self.timeout)
        except urllib.error.HTTPError as e:
            print(f"[会話] Ollama エラー {e.code}: {e.read().decode('utf-8', 'ignore')[:200]}")
            raise BrainError("sad", "ごめんね、うまく かんがえられなかった。")
        except OSError as e:  # つながらない・時間切れ
            print(f"[会話] Ollama ({self.url}) につながりません（{e}）")
            raise BrainError("sad", "ごめんね、かんがえる コンピューターに つながらないみたい。",
                             offline=True)
        emotion, answer = parse_reply(_THINK_RE.sub("", resp.get("message", {}).get("content", "")))
        if any(w in answer for w in NG_WORDS):
            print(f"[会話] 安全のため言いかえました: {answer}")
            return SAFE_REPLY
        return emotion, answer


class SimpleBrain:
    """AI を使わない、決まった受け答え（簡単会話モード）。Pi だけで動き、ネットも不要。

    Mac の Ollama などにつながらないときに、代わりに答える。
    """

    vision = False
    NOTICE_EVERY = 600.0  # 「いまは簡単なおはなしだけ」と伝える間隔（秒）

    # (含まれていたら反応する言葉, 感情, 返事の候補)。上にあるものほど優先。
    RULES = [
        (("ナイフ", "ほうちょう", "包丁", "マッチ", "ライター", "ひあそび", "火あそび",
          "くすり", "薬", "ベランダ", "コンセント", "しらないひと", "知らない人"),
         "neutral", ["それは あぶないかも。おうちの人に きいてね。"]),
        (("いたい", "痛い", "かなしい", "悲しい", "えーん", "こわい", "怖い"),
         "sad", ["だいじょうぶ？ おうちの人に おしえてあげてね。"]),
        (("なまえ", "名前", "だれ", "誰"), "joy", ["ぼくは {name} だよ！ よろしくね。"]),
        (("おはよう",), "joy", ["おはよう！ きょうも げんきだね。"]),
        (("こんにちは", "こんちは", "やあ", "ハロー"), "joy", ["こんにちは！ あえて うれしいな。"]),
        (("こんばんは",), "joy", ["こんばんは！"]),
        (("おやすみ",), "sleepy", ["おやすみなさい。いいゆめ みてね。"]),
        (("ばいばい", "バイバイ", "またね", "さようなら"), "sad", ["またね！ また あそぼうね。"]),
        (("ありがとう",), "love", ["どういたしまして！"]),
        (("すき", "好き"), "love", ["ぼくも だいすき！"]),
        (("おなか", "すいた", "ごはん"), "joy", ["おなか すいたね。おうちの人に きいてみよう！"]),
        (("ねむい", "眠い"), "sleepy", ["ねむいね。ゆっくり やすもうね。"]),
        (("できた", "したよ", "たのしかった", "楽しかった", "いった", "行った"),
         "joy", ["すごいね！", "わあ、よかったね！", "たのしそう！"]),
        (("？", "?", "なに", "何", "どうして", "なんで"),
         "think", ["うーん、なんだろう。あとで いっしょに しらべよう！"]),
    ]
    AIZUCHI = [("joy", "そうなんだ！"), ("surprise", "へえ、もっと おしえて！"),
               ("joy", "うんうん！"), ("love", "おはなし してくれて うれしいな。")]

    def __init__(self, name="ピコ"):
        self.name = name
        self.last_notice = -1e9

    def reply(self, text, image_jpeg=None):
        for words, emotion, answers in self.RULES:
            if any(w in text for w in words):
                return emotion, random.choice(answers).format(name=self.name)
        now = time.monotonic()
        if now - self.last_notice > self.NOTICE_EVERY:
            self.last_notice = now
            return "sad", "いまは あたまの コンピューターが おやすみ ちゅう。かんたんな おはなしなら できるよ。"
        return random.choice(self.AIZUCHI)


def make_brain(kind="ollama", model=None, name="ピコ", ollama_url=OLLAMA_URL):
    """kind: ollama（Mac などの Ollama）/ claude / simple（決まった受け答えだけ）。

    ollama と claude は、つながらないとき自動で簡単会話モードに切り替わる。
    """
    if kind == "simple":
        return SimpleBrain(name)
    if kind == "claude":
        brain = ClaudeBrain(model or DEFAULT_MODEL, name)
    else:
        brain = OllamaBrain(model or OLLAMA_MODEL, ollama_url, name)
    brain.fallback = SimpleBrain(name)
    return brain


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
        image = None
        if self.get_image and wants_look(text) and self.brain.vision:
            image = self.get_image()
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
    ap.add_argument("--brain", choices=["ollama", "claude", "simple"], default="ollama")
    ap.add_argument("--ollama-url", default=OLLAMA_URL)
    ap.add_argument("--model", default=None)
    ap.add_argument("--name", default="ピコ")
    ap.add_argument("--vosk-model", default=DEFAULT_VOSK)
    ap.add_argument("--voice")
    ap.add_argument("--pitch", type=float, default=2.0)
    args = ap.parse_args(argv)

    voice = None if args.no_voice else Voice(args.voice, pitch=args.pitch)
    ears = None if args.text else Ears(args.vosk_model)
    brain = make_brain(args.brain, args.model, args.name, args.ollama_url)
    conv = Conversation(brain, voice, ears)
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
