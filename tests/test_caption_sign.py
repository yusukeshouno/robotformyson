"""はい・いいえの見分け、顔の〇×、字幕の折り返しのテスト。"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from robot import conversation as cv
from robot.caption import Caption, wrap
from robot.face import Face


def test_yes_no():
    assert cv.yes_no("うん、すきだよ！") == "yes"
    assert cv.yes_no("はい、そうです。") == "yes"
    assert cv.yes_no("いいよ！あそぼう") == "yes"
    assert cv.yes_no("いいえ、ちがうよ。") == "no"
    assert cv.yes_no("ううん、しらないんだ。") == "no"
    assert cv.yes_no("ちがうよ、それは ねこだよ") == "no"
    assert cv.yes_no("うーん、どうかな") is None
    assert cv.yes_no("きょうは はれだよ") is None


def test_reply_events_include_sign_and_caption(monkeypatch):
    class Brain:
        vision = False

        def reply(self, text, image=None):
            return "joy", "うん、すきだよ！"

    conv = cv.Conversation(Brain(), None, None)
    conv.handle("イモ、りんご すき？")
    events = conv.poll()
    assert ("sign", "yes") in events
    assert not any(e == "caption_heard" for e, _ in events)  # 字幕は返事だけ
    assert ("caption", "うん、すきだよ！") in events


def test_face_sign_moves_and_draws():
    f = Face(96)
    f.update(0.016)
    before = f.render().copy()
    f.sign("no")
    shifts = set()
    for _ in range(40):
        f.update(0.016)
        shifts.add(f._motion()[0])
    assert len(shifts) > 1                      # 首を横にふる
    assert (f.render() != before).any()         # ×が出る
    f.update(3.0)
    assert f.sign_kind == "no" and f.t - f.sign_start > 2.6  # 時間がたてば消える


def test_wrap_japanese():
    lines = wrap("あいうえおかきくけこさしすせそ", lambda s: len(s) * 10, 50)
    assert lines == ["あいうえお", "かきくけこ", "さしすせそ"]
    lines = wrap("あ" * 40, lambda s: len(s) * 10, 50)
    assert len(lines) == 3 and lines[-1].endswith("…")


def test_caption_timing():
    c = Caption(480)
    c.say("うん！", now=0.0)
    c.update(1.0)
    assert c.text == "うん！"
    c.update(10.0)
    assert c.text == ""


def test_mouth_never_crashes():
    """口の端で 1 - u*u がごくわずかに負になっても落ちない（実機で TypeError: complex）。"""
    import numpy as np
    from robot.face import EMOTION_NAMES
    f = Face(96)
    for emo in EMOTION_NAMES:
        f.set_emotion(emo)
        for w in np.linspace(0.3, 1.4, 23):
            f.update(0.033)
            f.params["mouth_w"] = float(w)
            f.set_talk(float(w) % 1)
            f.render()
