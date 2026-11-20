import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402

from robot import drawing as dr  # noqa: E402


def test_wants_drawing():
    assert dr.wants_drawing("ねこ かいて") == ("picture", "ねこ かいて")
    assert dr.wants_drawing("イモ、おえかきしよう")[0] == "picture"
    assert dr.wants_drawing("ぼくの かお かいて") == ("portrait", None)
    assert dr.wants_drawing("ママ かいて", names=("ママ",)) == ("portrait", None)
    assert dr.wants_drawing("なんて かいてある？") is None
    assert dr.wants_drawing("こんにちは") is None


def test_shapes_render_and_bad_shapes_are_skipped():
    d = dr.Drawing({"title": "テスト", "background": "white", "shapes": [
        {"type": "rect", "color": "red", "x": 0, "y": 0, "w": 10, "h": 10},
        {"type": "circle", "color": "blue", "x": 30, "y": 30, "r": 5},
        {"type": "line", "color": "black", "x": 0, "y": 39, "x2": 39, "y2": 39},
        {"type": "laser", "color": "red"}, "ごみ",
        {"type": "triangle", "color": "green", "x": 20, "y": 0, "x2": 25, "y2": 8, "x3": 15, "y3": 8},
        {"type": "rect", "color": "nope", "x": "a", "y": 50, "w": 3, "h": 3},
    ]})
    assert len(d.strokes) == 4
    assert tuple(d.img[5, 5]) == dr.PALETTE["red"]
    assert tuple(d.img[30, 30]) == dr.PALETTE["blue"]
    assert tuple(d.img[39, 20]) == dr.PALETTE["black"]
    assert tuple(d.img[6, 20]) == dr.PALETTE["green"]


def test_portraits_for_all_styles():
    for hair in dr.FEATURES_SCHEMA["properties"]["hair"]["enum"]:
        f = dict(hair=hair, hair_color="brown", skin="tan", glasses=True, beard=True,
                 hat="cap", smile=True, shirt="green")
        d = dr.Drawing({"title": "x", "background": "pink", "shapes": dr.portrait_shapes(f)})
        assert d.ok and len(d.strokes) == len(d.data["shapes"])


def test_show_draws_then_ends():
    d = next(dr.demo_drawings())
    show = dr.DrawingShow(d, rows=48)
    seen_partial = False
    for _ in range(int((show.INTRO + show.MAX_DRAW + show.DONE + show.OUTRO + 2) * 20)):
        img = show.update(0.05)
        assert img.shape == (48, 48, 3)
        if show.phase == "draw" and not np.array_equal(show.paper, d.img):
            seen_partial = True
        if show.ended:
            break
    assert seen_partial and show.ended
    assert show.draw_time <= show.MAX_DRAW + 1e-6


def test_show_stop():
    show = dr.DrawingShow(next(dr.demo_drawings()))
    show.update(0.05)
    show.stop()
    for _ in range(20):
        show.update(0.05)
    assert show.ended


class FakeArtist(dr.Artist):
    def __init__(self, answers):
        super().__init__()
        self.answers = list(answers)
        self.prompts = []

    def _chat(self, prompt, schema, image=None, **kw):
        self.prompts.append((prompt, image))
        return self.answers.pop(0)


def wait(artist):
    import time
    for _ in range(100):
        got = artist.poll()
        if got:
            return got
        time.sleep(0.01)


def test_artist_picture_and_portrait():
    a = FakeArtist([dr.EXAMPLE, dict(person=True, hair="bob", hair_color="black",
                                      skin="light", glasses=False, smile=True, shirt="red")])
    assert a.request("picture", subject="いえ", asked=True)
    d, asked = wait(a)
    assert d.title == "おひさまと いえ" and asked
    assert a.request("portrait", image=b"jpg", call="ママ")
    d, asked = wait(a)
    assert d.title == "ママ の かお" and d.kind == "portrait" and not asked
    assert a.prompts[1][1] == b"jpg"


def test_artist_failure_reported_only_when_asked():
    a = FakeArtist([{"title": "x", "background": "white", "shapes": []}])
    a.request("picture", asked=False)
    assert wait(a) is None
    a = FakeArtist([])  # 例外（つながらない）
    a.request("picture", asked=True)
    assert wait(a) == (None, True)


def test_doodler_waits_for_idle_and_presence():
    t = [0.0]
    a = FakeArtist([dr.EXAMPLE] * 3)
    dd = dr.Doodler(a, every_min=10, clock=lambda: t[0])
    t[0] = 10 * 60
    assert not dd.tick(False, 999, False)     # だれもいない
    assert not dd.tick(True, 10, False)       # さっき話したばかり
    assert not dd.tick(True, 999, True)       # ねている
    assert dd.tick(True, 999, False)
    assert wait(a)
    assert not dd.tick(True, 999, False)      # すぐには描かない


def test_conversation_draw_request():
    from robot import conversation as cv

    class B:
        vision = False

        def reply(self, text, image=None):
            raise AssertionError("お絵かきは返事を考えない")

    conv = cv.Conversation(B(), get_image=lambda: b"jpg")
    conv.artist = FakeArtist([dr.EXAMPLE])
    conv.handle("ねこ かいて")
    got = wait(conv.artist)
    assert got and got[1] is True
    assert "ねこ かいて" in conv.artist.prompts[0][0]


def test_extract_json():
    assert dr.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert dr.extract_json('はい！ {"a": {"b": 2}} どうぞ') == {"a": {"b": 2}}


def test_falls_back_when_structured_output_is_unavailable():
    import io
    import urllib.error
    a = dr.Artist()
    calls = []

    def post(prompt, image, temperature, num_predict, schema):
        calls.append(schema)
        if schema is not None:
            raise urllib.error.HTTPError("u", 501, "structured output is unavailable", {},
                                         io.BytesIO(b""))
        assert "JSON スキーマ" in prompt
        return "```json\n" + __import__("json").dumps(dr.EXAMPLE) + "\n```"

    a._post = post
    assert a.picture("いえ").ok
    assert a.picture("いえ").ok
    assert calls == [dr.SCHEMA, None, None]  # 2回目からは最初から format なし
