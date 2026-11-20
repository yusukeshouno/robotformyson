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

from . import expression
from .arms import gesture_for
from .audio import (AudioSettings, MicGate, add_audio_args, output_channels, resolve_device,
                    to_channels)
from .faces import is_no, is_yes
from .memory import wants_forget

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
- 「はい」か「いいえ」で答えられる質問には、返事の最初に「うん」か「ううん」をはっきり言ってください。
- わからないことは、わからないと言ってください。"""

# この言葉が入っていたら、カメラの画像をいっしょに送る
LOOK_WORDS = ("これ", "それ", "見て", "みて", "見える", "みえる", "だれ", "誰",
              "何色", "なにいろ", "持って", "もって")

# [joy] のような感情タグ。小さなモデルは全角の［］【】や大文字で書くこともある
_TAG_RE = re.compile(r"[\[［【]\s*([A-Za-z]+)\s*[\]］】]")
_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S)
_SENTENCE_RE = re.compile(r"[^。！？!?]*[。！？!?]+|[^。！？!?]+$")
MAX_SENTENCES = 2   # 声に出すのは最初の2文まで（長い返事は子どもが飽きてしまう）
MAX_CHARS = 80

# ローカルの小さなモデル向けの、最後の安全網（返事にこれが入っていたら言いかえる）
NG_WORDS = ("死ね", "殺す", "ころす", "自殺", "しね", "セックス", "エロ", "麻薬", "爆弾")
# NG_WORDS を含むが、ふつうの言葉（「むかしね」「イエロー」など）。先に取りのぞいてから調べる
NG_OK_WORDS = ("むかしね", "おかしね", "たのしね", "うれしね", "かなしね", "さびしね", "ほしね",
               "はなしね", "わたしね", "あしね", "どうしね", "しねん", "イエロー", "エロン")
SAFE_REPLY = ("neutral", "うーん、べつの おはなし しよう！")


def normalize(text):
    """聞き取った文字をくらべやすくする（全角半角をそろえ、カタカナをひらがなに）。

    音声認識は「ピコ」を「ぴこ」と書くこともあるので、名前や言葉の判定はこれを通してから行う。
    """
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch for ch in text)


# 名前の書き方のちがい（Whisper は「芋」と漢字で書くこともある）
NAME_ALIASES = {"いも": ("芋",), "ぴこ": ()}


def wake_variants(name):
    """呼ばれたと分かる言葉: 名前（カタカナ・ひらがなは normalize でそろう）と、その漢字など。"""
    return (name,) + NAME_ALIASES.get(normalize(name), ())


def is_just_call(text, words):
    """「イモ」「いもー」「ねえ、イモ」のように、名前を呼んだだけか。"""
    t = re.sub(r"[\s、。,.!?！？ー〜~っ]", "", normalize(text))
    for w in words:
        t = t.replace(normalize(w), "")
    return t in ("", "ねえ", "ねぇ", "おい", "あの", "あのね", "ちゃん", "くん", "さん")


def contains_any(text, words):
    t = normalize(text)
    return any(normalize(w) in t for w in words)


def is_unsafe(text):
    t = normalize(text)
    for w in NG_OK_WORDS:
        t = t.replace(normalize(w), "")
    return any(normalize(w) in t for w in NG_WORDS)


def shorten(text, max_sentences=MAX_SENTENCES, max_chars=MAX_CHARS):
    """読み上げる長さに切りつめる（文の区切りで切る）。"""
    out = ""
    for s in _SENTENCE_RE.findall(text)[:max_sentences]:
        if out and len(out) + len(s) > max_chars:
            break
        out += s
    out = out.strip() or text
    if len(out) > max_chars:  # 句点のない長い文
        out = out[:max_chars].rstrip("、, ") + "…"
    return out


def clean_for_speech(text):
    """読み上げに向かないもの（考えの途中・記号・絵文字）を取りのぞく。"""
    text = _THINK_RE.sub("", text)
    text = "".join(ch for ch in text
                   if unicodedata.category(ch) not in ("So", "Cs", "Co")
                   and ch not in "*#`_~>|")
    return re.sub(r"\s+", " ", text).strip()


def parse_reply(text):
    """'[joy] やあ！' → ('joy', 'やあ！')。タグがなければ neutral。

    タグが先頭以外にあっても読み、どこにあっても読み上げる文からは取りのぞく。
    """
    emotion = "neutral"
    for m in _TAG_RE.finditer(text):
        if m.group(1).lower() in REPLY_EMOTIONS:
            emotion = m.group(1).lower()
            break
    return emotion, _TAG_RE.sub("", text).strip()


# はい・いいえ。返事の始まりだけを見る（「いいよ」は はい、「いいえ」は いいえ）
YES_STARTS = ("はい", "うん", "そうだよ", "そうそう", "そうなの", "そう！", "そう!", "そうだね",
              "いいよ", "いいね", "あたり", "せいかい", "正解", "もちろん", "ええ", "あってる", "できるよ")
NO_STARTS = ("いいえ", "ううん", "うーうん", "ちがう", "違う", "いや", "だめ", "ダメ", "ざんねん",
             "できないよ", "ないよ", "しらない", "知らない")


def yes_no(answer):
    """返事が「はい」か「いいえ」で始まっていれば 'yes' / 'no'。どちらでもなければ None。"""
    t = normalize(answer).lstrip("、。,. 　")
    for w in NO_STARTS:  # 「いいえ」を「いい」より先に見る
        if t.startswith(normalize(w)):
            return "no"
    for w in YES_STARTS:
        if t.startswith(normalize(w)):
            return "yes"
    return None


def wants_look(text):
    return contains_any(text, LOOK_WORDS)


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
        self.extra_system = None  # () -> 指示に足す文（成長のレベルや相手の名前など）
        self.max_turns = max_turns
        self.history = []
        self.fallback = None  # つながらないときに代わりに答える頭脳（SimpleBrain）
        self.face_parts = True  # 返事のたびに、顔の部品の組み合わせも考えてもらう
        self.last_face = None   # 直前の返事で AI が選んだ顔の部品（expression.parse の辞書）

    def reply(self, text, image_jpeg=None):
        """(感情, 返事) を返す。通信エラーでも例外にせず、謝る返事を返す。"""
        if not self.vision:
            if wants_look(text):
                text += "（いまは カメラが つかえないので、見えていません）"
            image_jpeg = None
        self.last_face = None
        try:
            emotion, answer = self._ask(text, image_jpeg)
        except BrainError as e:
            if e.offline and self.fallback:
                return self.fallback.reply(text)
            return e.emotion, e.answer
        spec, answer = expression.parse(answer)
        self.last_face = spec if self.face_parts else None
        answer = shorten(clean_for_speech(answer))
        if not answer:
            emotion, answer = "surprise", "えっと、もういっかい いって？"
        # 履歴は文字だけ残す（画像は毎回送り直さない）
        note = "（カメラの画像つき）" if image_jpeg else ""
        self.history += [{"role": "user", "content": text + note},
                         {"role": "assistant", "content": f"[{emotion}] "
                          + (expression.to_text(self.last_face) + " " if self.last_face else "")
                          + answer}]
        self.history = self.history[-2 * self.max_turns:]
        return emotion, answer

    def system_prompt(self):
        extra = self.extra_system() if self.extra_system else ""
        base = self.system + ("\n" + expression.PROMPT_HINT if self.face_parts else "")
        return base + ("\n" + extra if extra else "")

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
                system=self.system_prompt(),
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

    def summarize(self, prompt):
        """記憶のまとめ用。会話の履歴は使わず、1回だけ聞く。家の中の Ollama にだけ送る。"""
        body = {"model": self.model, "stream": False, "options": {"temperature": 0.2},
                "messages": [{"role": "user", "content": prompt}]}
        if self._caps and "thinking" in self._caps:
            body["think"] = False
        resp = self._post("/api/chat", body, self.timeout)
        return _THINK_RE.sub("", resp.get("message", {}).get("content", ""))

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
            "messages": [{"role": "system", "content": self.system_prompt()}] + self.history + [user],
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
        if is_unsafe(answer):
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
            if contains_any(text, words):
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

    def __init__(self, voice=None, pitch=2.0, speed=1.0, play=True, device=None):
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
        self.device = device      # None はシステム標準（setup_audio.sh 後はエコー除去の出口）
        self._channels = None
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
                if self._channels is None:
                    self._channels = output_channels(self.device)
                sd.play(to_channels(data, self._channels), rate, device=self.device)
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
class WhisperClient:
    """Mac などで動かす Whisper の書き起こしサーバー（tools/whisper_server.py）に音を送る。

    声ではなかったときは ""、つながらないときは None を返し、30秒は送らない
    （そのあいだは Vosk の文字を使う）。
    """

    RETRY = 30.0

    def __init__(self, url, timeout=10.0, clock=time.monotonic):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.clock = clock
        self.down_until = 0.0

    def __call__(self, pcm_bytes, rate):
        if not pcm_bytes or self.clock() < self.down_until:
            return None
        import io
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm_bytes)
        req = urllib.request.Request(self.url + "/transcribe", data=out.getvalue(),
                                     headers={"Content-Type": "audio/wav"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                text = json.loads(r.read().decode("utf-8")).get("text", "")
        except (OSError, ValueError) as e:
            print(f"[書き起こし] つながらないので Vosk を使います（{e}）")
            self.down_until = self.clock() + self.RETRY
            return None
        return text.replace(" ", "").replace("　", "").strip()  # "" = 声ではなかった


def pick_rate(device=None, want=16000):
    """録音のサンプリングレート。音声認識モデルは 16kHz 用なので、録れるなら 16kHz で録る
    （44.1kHz を渡すと Vosk の中で変換が1回ふえ、聞き取りが少し悪くなる）。"""
    import sounddevice as sd
    try:
        sd.check_input_settings(device=device, samplerate=want, channels=1, dtype="int16")
        return want
    except Exception:
        return int(sd.query_devices(device, "input")["default_samplerate"])


class Ears:
    """マイクの音を Vosk で文字にする。

    ロボットが話している間（と話し終えて少しの間）は gate が閉じ、その間の音と、
    閉じる前に録りかけていた言葉の切れ端は捨てる（自分の声を拾わないため）。
    """

    MAX_UTTERANCE = 15.0   # 書き起こしに送る音の長さの上限（秒）
    PRE_ROLL = 1.0         # 話しはじめの前に残しておく音（秒）。言いはじめが切れないように
    LOUD_HOLD = 2.0        # 雑音より大きい音のあと、この秒数は音を捨てない（Vosk が気づかない小さな声のため）
    LOUD_RATIO = 3.0       # 部屋の雑音の何倍で「声かも」とみるか

    def __init__(self, model_path=DEFAULT_VOSK, device=None, rate=None, gate=None,
                 recognizer=None, transcriber=None, segmenter=None):
        self.device = device
        # segmenter: 話しはじめ・終わりを見つける係（robot.vad.Segmenter、Silero VAD）。
        # あれば Vosk より小さな声に強い。Vosk は Whisper が使えないときの文字の係になる。
        self.segmenter = segmenter
        self._rough = []
        self.gate = gate or MicGate()
        # transcriber(pcm_bytes, rate) -> 文字 / ""（声ではない）/ None（使えない）。Mac の Whisper など。
        # Vosk は「話しはじめ・話し終わり」を見つける係になり、文字は transcriber のものを使う。
        self.transcriber = transcriber
        self.buf = bytearray()
        if recognizer is None:
            from vosk import KaldiRecognizer, Model, SetLogLevel
            SetLogLevel(-1)
            if not os.path.isdir(model_path):
                raise RuntimeError(f"音声認識モデルがありません: {model_path}（README 参照）")
            if rate is None:
                rate = pick_rate(device)
            recognizer = KaldiRecognizer(Model(model_path), rate)
        self.rate = rate or 16000
        self.rec = recognizer
        self.q = queue.Queue()
        self._epoch = self.gate.epoch
        # on_listen(vosk の文字): 話し終わったのが分かった瞬間（書き起こしを待つ前）に呼ぶ。考える顔を早く出すため
        # on_unclear(): 声はしたのに書き起こしが「声ではない」と言ったときに呼ぶ
        self.on_listen = None
        self.on_unclear = None
        self.on_speech = None  # on_speech(): 話しはじめたのが分かった瞬間（すぐ「聞いてるよ」の顔に）
        self.lag = 0.0         # 録ってから処理するまでのおくれ（秒）。Pi が忙しいと大きくなる
        self.timing = {}       # 直前の発話の時間（話し終わり・書き起こしにかかった秒）
        self.floor = 0.0       # 部屋の雑音の大きさ（RMS、int16）
        self._quiet_left = 0   # あと何バイト「静か」が続けば、前の音を捨ててよいか
        self._loud_bytes = 0   # この発話で「声かも」だった音の長さ（バイト）

    def mute(self, on):
        """on=True で閉じる。False で開く（実際に聞きはじめるのは余韻の時間のあと）。"""
        if on:
            self.gate.close()
        else:
            self.gate.open()

    def on_audio(self, pcm_bytes):
        """録音スレッドから呼ばれる。閉じている間の音は捨てる。"""
        if self.gate.is_open():
            self.q.put((self.gate.epoch, pcm_bytes, time.monotonic()))

    def process(self, item):
        """録った音を1つ認識器に渡す。一区切りの発話が終わったら文字を返す。"""
        epoch, pcm_bytes = item[0], item[1]
        if len(item) > 2:
            self.lag = time.monotonic() - item[2]   # 録ってから処理するまでのおくれ（秒）
        if epoch != self.gate.epoch or not self.gate.is_open():
            return None                       # 話す前に録った音・話している間の音
        if epoch != self._epoch:              # 話したあと最初の音: 途中経過を捨てる
            self.rec.Reset()
            self.buf.clear()
            self._rough = []
            if self.segmenter:
                self.segmenter.reset()
            self._epoch = epoch
        if self.segmenter:
            return self._process_vad(epoch, pcm_bytes)
        if self.transcriber:
            self.buf += pcm_bytes
            limit = int(self.rate * self.MAX_UTTERANCE) * 2
            if len(self.buf) > limit:
                del self.buf[:len(self.buf) - limit]
            self._watch_level(pcm_bytes)
        if not self.rec.AcceptWaveform(pcm_bytes):
            if self.transcriber and not self._partial() and self._quiet_left <= 0:
                # まだだれも話していない: 雑音を送らないよう、直前の少しだけ残す
                keep = int(self.rate * self.PRE_ROLL) * 2
                if len(self.buf) > keep:
                    del self.buf[:len(self.buf) - keep]
            return None
        text = json.loads(self.rec.Result()).get("text", "").replace(" ", "")
        audio, self.buf = bytes(self.buf), bytearray()
        # Vosk が文字にできなくても、雑音よりはっきり大きい音が 0.3 秒以上あれば、小さな声かもしれない
        rough = text  # Vosk の当て推量（「うん」のような短い返事は Whisper が空にしやすい）
        maybe_voice = self._loud_bytes >= int(self.rate * 0.3) * 2
        self._loud_bytes = 0
        if text and self.on_listen:
            self.on_listen(text)
        if (text or maybe_voice) and self.transcriber:  # 声らしいときだけ送る（雑音は送らない）
            better = self.transcriber(audio, self.rate)
            if better is not None:  # "" は「声ではなかった」。Vosk の当て推量は使わない
                text = better
                if not text and self.on_unclear:
                    self.on_unclear(rough)
        if text and self.gate.is_open() and self.gate.epoch == epoch:
            return text
        return None

    def _process_vad(self, epoch, pcm_bytes):
        """Silero VAD で区切る。Vosk は横で聞いて、当て推量の文字（rough）を作るだけ。"""
        was_speaking = self.segmenter.speaking
        audio = self.segmenter.feed(pcm_bytes)
        if self.segmenter.speaking and not was_speaking and self.on_speech:
            self.on_speech()
        # Vosk は重いので、だれかが話している間だけ動かす（Pi 4 の CPU が足りず、音の処理がおくれていた）
        if self.segmenter.speaking or audio is not None:
            chunks = list(self.segmenter.buf) if not was_speaking else [pcm_bytes]  # 話しはじめは前の音から
            for c in chunks:
                if self.rec.AcceptWaveform(c):
                    self._rough.append(json.loads(self.rec.Result()).get("text", ""))
        if audio is None:
            if not was_speaking and not self.segmenter.speaking:
                self._rough = []
            return None
        self._rough.append(json.loads(self.rec.FinalResult()).get("text", ""))
        rough, self._rough = "".join(self._rough).replace(" ", ""), []
        t_end = time.monotonic()
        self.timing = {"end": t_end, "lag": self.lag, "stt": 0.0}
        if self.on_listen:
            self.on_listen(rough)
        text = rough
        if self.transcriber:
            better = self.transcriber(audio, self.rate)
            self.timing["stt"] = time.monotonic() - t_end
            if better is not None:  # "" は「声ではなかった」。Vosk の当て推量は使わない
                text = better
                if not text and self.on_unclear:
                    self.on_unclear(rough)
        if text and self.gate.is_open() and self.gate.epoch == epoch:
            return text
        return None

    def _watch_level(self, pcm_bytes):
        """音の大きさを見て、部屋の雑音の大きさを覚え、声らしい音のあとはしばらく音を捨てない。"""
        x = np.frombuffer(pcm_bytes[:len(pcm_bytes) // 2 * 2], np.int16).astype(np.float32)
        if not len(x):
            return
        rms = float(np.sqrt((x * x).mean()))
        if self.floor <= 0:
            self.floor = rms
        elif rms < self.floor * 1.5:          # 静かなときだけ、ゆっくり雑音の大きさを合わせる
            self.floor += (rms - self.floor) * 0.05
        elif rms > self.floor * 6:            # 大きな音が続いても、少しずつ上がる
            self.floor += (rms - self.floor) * 0.002
        if rms > max(self.floor * self.LOUD_RATIO, 150):
            self._loud_bytes += len(pcm_bytes)
            self._quiet_left = int(self.rate * self.LOUD_HOLD) * 2
        else:
            self._quiet_left -= len(pcm_bytes)

    def _partial(self):
        """Vosk が今なにか聞き取りかけているか。"""
        try:
            return bool(json.loads(self.rec.PartialResult()).get("partial", "").strip())
        except (AttributeError, ValueError):
            return True

    def accept(self, pcm_bytes):
        """（互換用）今の状態で音を1つ渡す。"""
        return self.process((self.gate.epoch, pcm_bytes))

    STALL = 10.0        # マイクの音がこの秒数こなければ、録音を開きなおす
    BACKLOG = 30        # 処理が追いつかず、たまった音がこれ以上なら古い音を捨てる（3秒ぶん）
    HEARTBEAT = 600.0   # この秒数ごとに「耳は動いている」とログに出す

    def listen(self):
        """発話ごとに文字を返すジェネレーター。

        実機で、数時間たつと何も聞かなくなることがあった（原因は調査中）。そこで
        - マイクの音が STALL 秒こなければ、録音を開きなおす
        - 処理がおくれて音がたまったら、古い音を捨てて今の音から聞く
        - ときどき、耳が動いていることと、たまっている音の数をログに出す
        """
        import sounddevice as sd

        def callback(indata, frames, t, status):
            self.on_audio(bytes(indata))

        last_beat = time.monotonic()
        while True:
            try:
                with sd.RawInputStream(samplerate=self.rate, blocksize=int(self.rate * 0.1),
                                       device=self.device, dtype="int16", channels=1,
                                       callback=callback):
                    while True:
                        try:
                            item = self.q.get(timeout=self.STALL)
                        except queue.Empty:
                            if self.gate.is_open():
                                print(f"[耳] {self.STALL:.0f}秒 マイクの音がこないので、開きなおします")
                                break
                            continue          # 話している間（ゲートが閉じている）は音がこなくてよい
                        backlog = self.q.qsize()
                        if backlog > self.BACKLOG:
                            print(f"[耳] 処理がおくれています（{backlog} 個たまった）。古い音を捨てます")
                            while self.q.qsize() > 2:
                                try:
                                    self.q.get_nowait()
                                except queue.Empty:
                                    break
                        now = time.monotonic()
                        if now - last_beat > self.HEARTBEAT:
                            last_beat = now
                            print(f"[耳] 聞いています（たまった音 {backlog}）")
                        text = self.process(item)
                        if text:
                            yield text
            except Exception as e:  # 録音が止まった・デバイスが抜けた
                print(f"[耳] 録音のエラー: {e}。2秒後に開きなおします")
                time.sleep(2)


# ---------------------------------------------------------------- まとめ役
class Conversation:
    """聞いて・考えて・話す。顔への指示は events に積み、表示側が毎フレーム取り出す。"""

    FOLLOW_UP = 15.0  # 返事のあとこの秒数は、顔が見えなくても続けて聞く

    def __init__(self, brain, voice=None, ears=None, get_image=None, is_attended=None,
                 wake_words=(), faces=None, memory=None, language=None):
        self.brain, self.voice, self.ears = brain, voice, ears
        self.memory = memory              # Memory（前に話したことをおぼえる）/ None
        self.language = language          # Language（話し方に合わせる。数字だけ数える）/ None
        self.get_image = get_image        # () -> JPEG bytes / None
        self.is_attended = is_attended    # () -> 人がこちらを見ているか
        self.wake_words = tuple(v for w in wake_words for v in wake_variants(w))
        self.faces = faces                # FaceRecognizer（顔をおぼえる機能）/ None
        self.enroll = None
        if faces is not None:
            from .faces import EnrollDialog
            self.enroll = EnrollDialog(faces, show=lambda t: self.events.put(("caption", t)))
        self.artist = None                # drawing.Artist（お絵かき）/ None
        self.events = queue.Queue()
        self.last_reply = 0.0
        self._speak_lock = threading.Lock()
        self._thinking = False
        if ears is not None:
            ears.on_listen = self._on_listen
            ears.on_unclear = self._on_unclear
            ears.on_speech = self._on_speech

    def _expecting(self):
        """いま話しかけられていそうか（顔が見える・返事のすぐあと・顔の登録中）。"""
        return bool((self.enroll and self.enroll.active)
                    or time.monotonic() - self.last_reply < self.FOLLOW_UP
                    or (self.is_attended and self.is_attended()))

    def _on_speech(self):
        """声がしはじめた瞬間。耳をすます顔（話しかけられていそうなときだけ）。"""
        if self._expecting():
            self.events.put(("hearing", 0))

    def _on_listen(self, rough_text):
        """話し終わった直後（文字になる前）。返事をしそうなら、すぐ考える顔にする。"""
        if rough_text:
            self._thinking = self.should_reply(rough_text)
        else:  # Vosk は文字にできなかったが、声はした（Silero VAD）。待っている相手なら考える顔に
            self._thinking = self._expecting()
        if self._thinking:
            # 「聞いて考えはじめた」顔。まだ返事するか分からないので、前の返事の字幕は消さない
            self.events.put(("listen", 4))

    def _on_unclear(self, rough=""):
        """声はしたけれど聞き取れなかった。待っている相手に分かるようにする。

        顔の登録で「あってる？」と聞いたあとは、Vosk が「うん」「ちがう」と聞いていればそれを使う。
        """
        thinking, self._thinking = self._thinking, False
        if (rough and self.wake_words and contains_any(rough, self.wake_words)
                and not (self.enroll and self.enroll.active)):
            # 「イモ…」と呼ばれたのに、そのあとが聞き取れなかった: 呼ばれたことには答えて、もう一度聞く
            self.events.put(("heard", 0))
            self._answer("confused", "ん？ よんだ？ もういっかい いって？")
        elif (self.enroll and self.enroll.active and self.enroll.state == "confirm" and rough
                and (is_yes(rough) or is_no(rough))):
            self.handle(rough)
        elif self.enroll and self.enroll.active:  # 名前を待っている: 声と字幕で聞きなおす（何回かまで）
            self.enroll.on_unclear(self._answer)
        elif thinking:  # 聞き取れなかったことを、はっきり見せる（？の顔と字幕）
            print("[聞きとれず] もういっかい たのむ")
            self.events.put(("unclear", 0))
            self.events.put(("confused", 2.0))
            self.events.put(("caption", "ん？ もういっかい いって？"))

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
        if self.enroll and self.enroll.active:  # 顔の登録の途中（「うん」も聞く）
            return True
        if contains_any(text, self.wake_words):  # 「芋」1文字でも呼ばれたことになる
            return True
        if len(text) < 2:
            return False
        if time.monotonic() - self.last_reply < self.FOLLOW_UP:
            return True
        return self.is_attended is None or self.is_attended()

    def handle(self, text):
        if not self.should_reply(text):
            print(f"[聞こえた（無視）] {text}")
            if self._thinking:
                self._thinking = False
                self.events.put(("neutral", 0.5))
            return None
        print(f"[聞こえた] {text}")
        self._thinking = False
        self.events.put(("ack", 0))  # 聞きとれたよ（小さくうなずく）
        if (self.wake_words and is_just_call(text, self.wake_words)
                and not (self.enroll and self.enroll.active)):
            # 名前を呼ばれただけ: 考えずにすぐ「なあに？」。そのあと少しのあいだは顔が見えなくても聞く
            self.events.put(("heard", 0))
            self._answer("excited", random.choice(("なあに？", "はーい！", "よんだ？")))
            return "excited", "なあに？"
        self.events.put(("heard", 0))
        gesture = gesture_for(text)
        if gesture:
            self.events.put(("gesture", gesture))
        if self.enroll and self.enroll.handle(text, self._answer):
            return None
        if self.memory and wants_forget(text):  # 「いまの わすれて」：この会話はおぼえない
            self.memory.drop_session()
            self._answer("neutral", "わかった。いまの おはなしは わすれるね。")
            return None
        if self.artist and self._maybe_draw(text):
            return None
        self.events.put(("think", 60))
        image = None
        if self.get_image and wants_look(text) and self.brain.vision:
            image = self.get_image()
        t_ask = time.monotonic()
        emotion, answer = self.brain.reply(self._with_speaker(text), image)
        print(f"[{emotion}] {answer}")
        self._log_timing(t_ask)
        if self.memory:
            self.memory.add_turn(text, answer)
        if self.language:
            self.language.on_turn(text, answer)
        params = expression.compose(getattr(self.brain, "last_face", None))
        if params:  # AI が組み立てた、その場の顔
            print(f"[顔の部品] {expression.to_text(self.brain.last_face)}")
            self.events.put(("compose", (emotion, params)))
        self.events.put((emotion, 60))
        sign = yes_no(answer)
        if sign:  # はい・いいえは顔でもはっきり見せる（うなずく・首をふる、〇・×）
            self.events.put(("sign", sign))
        self._speak(answer)
        if params:
            self.events.put(("uncompose", 1.5))
        self.events.put((emotion, 1.5))
        self.last_reply = time.monotonic()
        return emotion, answer

    def _family_names(self):
        if not self.faces:
            return ()
        people = getattr(getattr(self.faces, "db", None), "people", {}) or {}
        return tuple(n for p in people.values() for n in (p.get("name"), p.get("call")) if n)

    def _maybe_draw(self, text):
        """「えを かいて」「ぼくの かお かいて」と頼まれたら、お絵かきを始める。"""
        from .drawing import wants_drawing
        req = wants_drawing(text, self._family_names())
        if not req:
            return False
        kind, subject = req
        image = call = None
        if kind == "portrait":
            image = self.get_image() if self.get_image else None
            if not image:
                self._answer("confused", "かおが みえないよ。カメラの まえに きてね。")
                return True
            call = self.who()
        else:
            subject = (f"子どもの たのみ「{subject}」（なにを描いてほしいか読みとって描く。"
                       "とくになければ、すきなものを描く）")
        if self.artist.request(kind, subject=subject, image=image, call=call, asked=True):
            self._answer("excited", "いいよ！ どんなふうに かこうかな。ちょっと まってね。")
        else:
            self._answer("joy", "いま べつの えを かいてるところ。まっててね。")
        return True

    def _log_timing(self, t_ask):
        """どこで時間がかかっているかをログに出す（返事がおそいときの調べもの用）。"""
        tm = getattr(self.ears, "timing", None) or {}
        now = time.monotonic()
        if not tm.get("end") or now - tm["end"] > 120:
            return
        print(f"[時間] おくれ {tm.get('lag', 0):.1f}秒 / 書き起こし {tm.get('stt', 0):.1f}秒"
              f" / 考える {now - t_ask:.1f}秒 / 話し終わりから返事まで {now - tm['end']:.1f}秒")

    def _speak(self, text):
        self.events.put(("caption", text))  # 字幕（スピーカーがなくても読める）
        if not self.voice:
            return
        with self._speak_lock:  # 返事と あいさつが重ならないように
            # 話している間はマイクを閉じる。開くのは、話し終えて余韻の時間がたってから
            if self.ears:
                self.ears.mute(True)
            try:
                self.voice.speak(text)
            finally:
                if self.ears:
                    self.ears.mute(False)

    def say(self, text):
        """自分から話す（あいさつ・レベルアップなど）。裏で話すのですぐ戻る。"""
        print(f"[ひとりごと] {text}")
        threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _answer(self, emotion, text):
        """表情を出しながら話し、話し終わるまで待つ（顔の登録のやりとりで使う）。"""
        print(f"[{emotion}] {text}")
        self.events.put((emotion, 60))
        self._speak(text)
        self.events.put((emotion, 1.5))
        self.last_reply = time.monotonic()

    def who(self):
        """いま話している（見えている）人の呼び名。わからなければ None。"""
        if not self.faces:
            return None
        pid, name = self.faces.current()
        if pid is None:
            return None
        return (self.faces.person(pid) or {}).get("call") or name

    def _with_speaker(self, text):
        if getattr(self.brain, "speaker_in_system", False):  # 成長の記録が相手の名前を指示に入れている
            return text
        call = self.who()
        return f"{text}（話しているのは {call}）" if call else text

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
    add_audio_args(ap)
    args = ap.parse_args(argv)

    audio = AudioSettings.load(args.mic, args.speaker, args.echo_tail)
    voice = ears = None
    if not args.no_voice:
        voice = Voice(args.voice, pitch=args.pitch,
                      device=resolve_device(audio.speaker, "output"))
    if not args.text:
        ears = Ears(args.vosk_model, device=resolve_device(audio.mic, "input"),
                    gate=MicGate(audio.echo_tail))
    brain = make_brain(args.brain, args.model, args.name, args.ollama_url)
    conv = Conversation(brain, voice, ears)
    if args.text:
        print("話しかけてください（終了は Ctrl+D）")
        try:
            while True:
                text = input("> ").strip()
                if text:
                    conv.handle(text)
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
