"""返事の余韻（mood）と、話すときの目そらしのテスト。"""
import os
import random

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from robot.emotion import EmotionEngine  # noqa: E402
from robot.face import Face, _target  # noqa: E402


def test_mood_fades_out():
    e = EmotionEngine(0.0)
    e.set_mood("joy", 0.0)
    name, w0 = e.mood(0.0)
    assert name == "joy" and w0 == EmotionEngine.MOOD_WEIGHT
    _, w1 = e.mood(10.0)
    assert 0 < w1 < w0
    assert e.mood(120.0) == ("neutral", 0.0)


def test_sad_mood_is_short():
    e = EmotionEngine(0.0)
    e.set_mood("sad", 0.0)
    _, sad = e.mood(8.0)
    e.set_mood("joy", 0.0)
    _, joy = e.mood(8.0)
    assert sad < joy


def test_think_does_not_leave_mood():
    e = EmotionEngine(0.0)
    e.set_mood("think", 0.0)
    assert e.mood(0.0) == ("neutral", 0.0)


def test_no_mood_while_sleeping():
    e = EmotionEngine(0.0)
    e.set_mood("joy", 0.0)
    e.sleeping = True
    assert e.mood(1.0) == ("neutral", 0.0)


def test_face_blends_mood_into_neutral():
    f = Face(96)
    f.set_emotion("neutral")
    f.set_mood("joy", 0.5)
    for _ in range(200):
        f.update(1 / 30)
    n, j = _target("neutral")["mouth"], _target("joy")["mouth"]
    assert abs(f.params["mouth"] - (n + (j - n) * 0.5)) < 0.02
    assert f.eyes == _target("neutral")["eyes"]  # 目の形は変えない


def _run(face, seconds, talk):
    face.set_talk(talk)
    for _ in range(int(seconds * 30)):
        face.update(1 / 30)


def test_looks_away_when_starting_to_talk_then_back():
    random.seed(1)
    f = Face(96)
    f.look_at((0.0, 0.0))
    _run(f, 2.0, 0.0)
    assert abs(f.gaze[0]) < 0.05
    _run(f, 0.4, 0.5)                 # 話し始め：目をそらす
    assert abs(f.gaze[0]) > 0.3
    _run(f, 2.0, 0.5)                 # しばらくすると相手を見る
    assert abs(f.gaze[0]) < 0.05


def test_no_averting_while_listening():
    f = Face(96)
    f.look_at((0.2, 0.1))
    _run(f, 10.0, 0.0)
    assert abs(f.gaze[0] - 0.2) < 0.02 and abs(f.gaze[1] - 0.1) < 0.02
