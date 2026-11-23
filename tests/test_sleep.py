"""人がいないと眠り、顔が見えても起きない（「イモ」と呼ばれる・さわられると起きる）。"""
from robot.emotion import EmotionEngine


def test_sleeps_when_nobody_and_face_does_not_wake():
    e = EmotionEngine(0.0)
    assert e.update(EmotionEngine.SLEEP_AFTER + 1) == "sleepy" and e.sleeping
    e.on_face(True, EmotionEngine.SLEEP_AFTER + 2)
    assert e.sleeping and e.update(EmotionEngine.SLEEP_AFTER + 3) == "sleepy"
    assert e.notice(EmotionEngine.SLEEP_AFTER + 4)   # 呼ばれた → 起きる
    assert not e.sleeping


def test_stays_awake_while_face_visible():
    e = EmotionEngine(0.0)
    for t in range(0, 300, 10):
        e.on_face(True, float(t))
        e.update(float(t))
    assert not e.sleeping
