"""会話機能のテスト（マイク・スピーカー・API キー・Ollama がなくても動く）。

    python3 -m pytest tests
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from robot import conversation as cv


# ---------------------------------------------------------------- 文字の処理
@pytest.mark.parametrize("raw, expected", [
    ("[joy] やあ！", ("joy", "やあ！")),
    ("［surprise］わあ", ("surprise", "わあ")),
    ("【love】すき", ("love", "すき")),
    ("[Joy] やあ", ("joy", "やあ")),
    ("やあ [sad] ざんねん", ("sad", "やあ  ざんねん")),
    ("[think] うーん", ("neutral", "うーん")),   # 知らないタグは neutral、文からは消す
    ("タグなし", ("neutral", "タグなし")),
])
def test_parse_reply(raw, expected):
    assert cv.parse_reply(raw) == expected


def test_normalize_katakana_and_width():
    assert cv.normalize("ピコ") == cv.normalize("ぴこ") == "ぴこ"
    assert cv.normalize("ＡＢＣ") == "abc"


def test_is_unsafe_has_no_false_positives():
    assert cv.is_unsafe("しね")
    assert cv.is_unsafe("エロい はなし")
    assert not cv.is_unsafe("むかしね、きょうりゅうが いたんだよ")
    assert not cv.is_unsafe("きいろは えいごで イエロー だよ")


def test_shorten_keeps_first_two_sentences():
    assert cv.shorten("いち。に！さん。") == "いち。に！"
    long = "あ" * 200
    assert len(cv.shorten(long)) <= cv.MAX_CHARS + 1


def test_clean_for_speech_removes_symbols_and_thinking():
    assert cv.clean_for_speech("<think>考え中</think>**やあ** 😀") == "やあ"


def test_wants_look():
    assert cv.wants_look("これ なに？")
    assert cv.wants_look("ミテ ミテ")
    assert not cv.wants_look("おはよう")


# ---------------------------------------------------------------- 簡単会話モード
def test_simple_brain_rules():
    b = cv.SimpleBrain("ピコ")
    assert b.reply("おなまえは？") == ("joy", "ぼくは ピコ だよ！ よろしくね。")
    assert b.reply("ナイフ つかっていい？")[1].endswith("おうちの人に きいてね。")
    assert b.reply("オハヨウ")[0] == "joy"   # カタカナで聞き取っても反応する


def test_simple_brain_notice_then_aizuchi():
    b = cv.SimpleBrain()
    emotion, answer = b.reply("きょうは こうえん")
    assert "かんたんな おはなし" in answer
    assert b.reply("きょうは こうえん") in b.AIZUCHI


# ---------------------------------------------------------------- Ollama（偽サーバー）
class FakeOllama:
    def __init__(self, reply="[joy] こんにちは！", caps=("completion", "vision")):
        self.reply, self.caps, self.requests = reply, list(caps), []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send({"version": "0.0.0"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, body))
                if self.path == "/api/show":
                    self._send({"capabilities": outer.caps})
                else:
                    self._send({"message": {"role": "assistant", "content": outer.reply}})

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def ollama():
    srv = FakeOllama()
    yield srv
    srv.close()


def test_ollama_reply_and_history(ollama):
    brain = cv.make_brain("ollama", "test-model", "ピコ", ollama.url)
    assert brain.vision
    assert brain.reply("こんにちは") == ("joy", "こんにちは！")
    path, body = ollama.requests[-1]
    assert path == "/api/chat"
    assert body["model"] == "test-model"
    assert body["messages"][0]["role"] == "system" and "ピコ" in body["messages"][0]["content"]
    assert brain.history[-1] == {"role": "assistant", "content": "[joy] こんにちは！"}


def test_ollama_sends_image(ollama):
    brain = cv.make_brain("ollama", "m", ollama_url=ollama.url)
    brain.reply("これ なに？", b"\xff\xd8jpeg")
    user = ollama.requests[-1][1]["messages"][-1]
    assert user["images"]
    assert brain.history[-2]["content"].endswith("（カメラの画像つき）")


def test_ollama_without_vision_does_not_send_image():
    srv = FakeOllama(caps=("completion",))
    try:
        brain = cv.make_brain("ollama", "m", ollama_url=srv.url)
        brain.reply("これ なに？", b"jpeg")
        user = srv.requests[-1][1]["messages"][-1]
        assert "images" not in user and "見えていません" in user["content"]
    finally:
        srv.close()


def test_ollama_unsafe_reply_is_replaced():
    srv = FakeOllama(reply="[angry] しね")
    try:
        brain = cv.make_brain("ollama", "m", ollama_url=srv.url)
        assert brain.reply("ねえ") == cv.SAFE_REPLY
    finally:
        srv.close()


def test_ollama_long_reply_is_shortened():
    srv = FakeOllama(reply="[joy] いち。に。さん。し。")
    try:
        brain = cv.make_brain("ollama", "m", ollama_url=srv.url)
        assert brain.reply("ねえ") == ("joy", "いち。に。")
    finally:
        srv.close()


def test_ollama_offline_falls_back_to_simple_brain():
    brain = cv.make_brain("ollama", "m", "ピコ", "http://127.0.0.1:9")  # だれもいないポート
    assert isinstance(brain.fallback, cv.SimpleBrain)
    assert brain.reply("おなまえは？") == ("joy", "ぼくは ピコ だよ！ よろしくね。")
    assert brain.history == []


# ---------------------------------------------------------------- Claude（偽クライアント）
class FakeMessages:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return self.resp


def _resp(text, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason,
                           content=[SimpleNamespace(type="thinking", thinking=""),
                                    SimpleNamespace(type="text", text=text)])


def _claude(messages):
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    brain = cv.ClaudeBrain(client=client)
    brain.fallback = cv.SimpleBrain()
    return brain


def test_claude_reply_with_image():
    msgs = FakeMessages(_resp("[surprise] わあ、りんごだね！"))
    brain = _claude(msgs)
    assert brain.reply("これ なに？", b"jpeg") == ("surprise", "わあ、りんごだね！")
    kw = msgs.calls[0]
    assert kw["model"] == cv.DEFAULT_MODEL
    assert kw["fallbacks"] == "default" and kw["output_config"] == {"effort": "low"}
    content = kw["messages"][-1]["content"]
    assert content[0]["type"] == "image" and content[1]["text"] == "これ なに？"


def test_claude_refusal():
    brain = _claude(FakeMessages(_resp("", stop_reason="refusal")))
    assert brain.reply("ねえ") == ("sad", "ごめんね、そのおはなしは できないんだ。")


def test_claude_offline_falls_back():
    import anthropic
    import httpx2
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://example.invalid"))
    brain = _claude(FakeMessages(exc=err))
    assert brain.reply("ありがとう") == ("love", "どういたしまして！")


# ---------------------------------------------------------------- まとめ役
class FakeVoice:
    def __init__(self):
        self.said = []

    def speak(self, text):
        self.said.append(text)

    def mouth_level(self):
        return 0.0


class FakeEars:
    def __init__(self):
        self.mutes = []

    def mute(self, on):
        self.mutes.append(on)


class FixedBrain:
    vision = True

    def __init__(self):
        self.calls = []

    def reply(self, text, image=None):
        self.calls.append((text, image))
        return "joy", "やあ！"


def test_conversation_flow(monkeypatch):
    monkeypatch.setattr(cv.time, "sleep", lambda s: None)
    voice, ears, brain = FakeVoice(), FakeEars(), FixedBrain()
    conv = cv.Conversation(brain, voice, ears, get_image=lambda: b"jpeg")
    assert conv.handle("これ みて") == ("joy", "やあ！")
    assert brain.calls == [("これ みて", b"jpeg")]
    assert voice.said == ["やあ！"] and ears.mutes == [True, False]
    assert [e for e, _ in conv.poll()] == ["ack", "heard", "think", "joy", "caption", "joy"]


def test_conversation_only_replies_when_attended_or_called():
    brain = FixedBrain()
    attended = [False]
    conv = cv.Conversation(brain, is_attended=lambda: attended[0], wake_words=("ピコ",))
    assert conv.handle("テレビの おと") is None
    assert conv.handle("ぴこ、あそぼう") == ("joy", "やあ！")   # 名前（ひらがな）で呼ぶ
    assert conv.handle("つづきの おはなし") is not None        # 返事のあとは続けて聞く
    conv.last_reply -= cv.Conversation.FOLLOW_UP + 1
    assert conv.handle("ひとりごと") is None
    attended[0] = True
    assert conv.handle("こっちを みてるよ") is not None
    assert conv.handle("あ") is None                            # 短すぎる


def test_text_mode_main(monkeypatch, capsys):
    lines = iter(["", "おはよう"])

    def fake_input(prompt=""):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)
    cv.main(["--text", "--no-voice", "--brain", "simple"])
    out = capsys.readouterr().out
    assert "[joy] おはよう！ きょうも げんきだね。" in out
    assert "無視" not in out   # 空の行は黙って読みとばす


def test_curious_personality_hint():
    import datetime as dt
    from robot import personality
    assert "年下の友達" in personality.hint("curious")
    assert personality.hint("plain") == ""
    even = dt.date(2026, 11, 9)   # ISO 第46週
    odd = dt.date(2026, 11, 16)   # ISO 第47週
    assert personality.is_curious("weekly", even)
    assert not personality.is_curious("weekly", odd)
    # 本当のことは、わざとまちがえない
    assert "わざとまちがえない" in personality.CURIOUS_STYLE


def test_speaker_name_kept_when_only_personality_in_system():
    from robot.conversation import Conversation, SimpleBrain

    class Faces:
        def current(self):
            return "p1", "ゆうと"

        def person(self, pid):
            return {"call": "ゆうくん"}

        def on_change(self, cb):
            pass
    brain = SimpleBrain("ピコ")
    brain.extra_system = lambda: "性格"
    conv = Conversation(brain)
    conv.faces = Faces()
    assert "ゆうくん" in conv._with_speaker("こんにちは")
    brain.speaker_in_system = True
    assert conv._with_speaker("こんにちは") == "こんにちは"


def test_ears_keep_running_while_replying():
    """返事を考えているあいだも耳は止まらない（前は音がたまって捨てられていた）。"""
    import threading
    import time as _t
    from robot.conversation import Conversation
    gate = threading.Event()
    listened = []

    class Ears:
        def listen(self):
            for t in ("イモ ひとつめ", "イモ ふたつめ"):
                listened.append(t)
                yield t
            gate.wait(2)

    c = Conversation(brain=None, ears=Ears())
    handled = []
    c.handle = lambda t: (handled.append(t), _t.sleep(0.5))
    c.start()
    _t.sleep(0.2)
    assert listened == ["イモ ひとつめ", "イモ ふたつめ"]  # 1つめの返事の途中でも2つめを聞けた
    gate.set()


def test_voicevox_voice_and_fallback(monkeypatch):
    import numpy as np
    import io as _io
    import wave as _wave
    buf = _io.BytesIO()
    with _wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(np.zeros(2400, np.int16).tobytes())
    calls = []

    def post(self, path, params, body=None, timeout=10):
        calls.append((path, params))
        if path == "/audio_query":
            return b'{"speedScale": 1, "pitchScale": 0}'
        return buf.getvalue()

    monkeypatch.setattr(cv.VoicevoxVoice, "_post", post)
    v = cv.VoicevoxVoice("http://mac:50021", speaker=3, play=False)
    data, rate = v.synth("こんにちは")
    assert rate == 24000 and len(data) == 2400
    assert calls[0] == ("/audio_query", {"text": "こんにちは", "speaker": 3})

    class JTalk:
        def synth(self, text):
            return np.ones(10, np.int16), 48000

    def down(self, *a, **k):
        raise OSError("no route")

    monkeypatch.setattr(cv.VoicevoxVoice, "_post", down)
    v = cv.VoicevoxVoice("http://mac:50021", fallback=JTalk(), play=False)
    assert v.synth("やあ")[1] == 48000          # つながらなければ Open JTalk
    assert v.synth("やあ")[1] == 48000          # しばらくは VOICEVOX を試さない


def test_fix_name_accent():
    q = {"accent_phrases": [
        {"moras": [{"text": "イ"}, {"text": "モ"}, {"text": "ダ"}, {"text": "ヨ"}], "accent": 2},
        {"moras": [{"text": "イ"}, {"text": "ヌ"}], "accent": 2}]}
    cv.fix_name_accent(q, "イモ")
    assert q["accent_phrases"][0]["accent"] == 1   # イ↘モだよ
    assert q["accent_phrases"][1]["accent"] == 2   # ほかの言葉はそのまま
