import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from robot import expression as ex  # noqa: E402


def test_parse_and_strip():
    spec, rest = ex.parse("{目:star 口:laugh 眉:up ほっぺ:1 色:yellow 動き:bounce 小物:note} すごい！")
    assert spec == {"eyes": "star", "mouth": "laugh", "brows": "up", "cheeks": "1",
                    "color": "yellow", "motion": "bounce", "fx": "note"}
    assert rest == "すごい！"


def test_parse_tolerates_japanese_and_fullwidth():
    spec, rest = ex.parse("やった｛目＝ハート, 色：ピンク｝ね")
    assert spec == {"eyes": "heart", "color": "pink"}
    assert "｛" not in rest and rest.startswith("やった")


def test_no_spec():
    assert ex.parse("こんにちは！") == (None, "こんにちは！")


def test_compose_ignores_unknown():
    p = ex.compose({"eyes": "laser", "mouth": "grin", "fx": "fire"})
    assert p["mouth_open"] == 0.5 and "fx" not in p
    assert ex.compose({"eyes": "laser"}) is None
    assert ex.compose(None) is None


def test_compose_values():
    p = ex.compose({"eyes": "star", "cheeks": "2", "color": "green", "motion": "none",
                    "fx": "sweat"})
    assert p["eyes"] == ("star", "star") and p["blush"] == 1.0
    assert p["motion"] is None and p["fx"] == ("sweat",)


def test_brain_strips_spec_and_keeps_it(monkeypatch):
    from robot import conversation as cv
    brain = cv.ClaudeBrain.__new__(cv.ClaudeBrain)
    cv._BrainBase.__init__(brain, "イモ", 10)
    brain._ask = lambda text, img: ("joy", "{目:happy 口:grin} やったね！")
    emo, answer = brain.reply("できた")
    assert (emo, answer) == ("joy", "やったね！")
    assert brain.last_face == {"eyes": "happy", "mouth": "grin"}
    assert "{目:happy 口:grin}" in brain.history[-1]["content"]
    assert "部品" in brain.system_prompt()


def test_face_custom_lasts_for_the_reply():
    import pygame
    pygame.display.init()
    from robot.face import Face
    f = Face(48)
    f.set_custom("joy", ex.compose({"eyes": "star", "color": "green"}))
    f.set_emotion("think")
    f.update(0.05)
    assert f.custom  # 考え中には消えない
    f.set_emotion("joy")
    for _ in range(30):
        f.update(0.05)
    assert f.eyes == ("star", "star")
    f.render()
    f.end_custom(0.5)
    for _ in range(20):
        f.update(0.05)
    assert f.custom is None


def test_conversation_sends_compose():
    from robot import conversation as cv

    class B:
        vision = False
        last_face = {"eyes": "heart"}

        def reply(self, text, image=None):
            return "love", "だいすき！"

    conv = cv.Conversation(B())
    conv.handle("すき")
    events = conv.poll()
    kinds = [e[0] for e in events]
    assert "compose" in kinds and "uncompose" in kinds
    assert kinds.index("compose") < kinds.index("love")
