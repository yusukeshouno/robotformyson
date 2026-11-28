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


def test_presence_ignores_flicker_and_still_photo():
    from robot.presence import PresenceFilter
    p = PresenceFilter()
    assert not p.update(True, 0.1, 0.1, 0.0)      # 一瞬だけでは「いる」にしない
    assert p.update(True, 0.1, 0.1, 2.0)
    t = 2.0
    while t < 120:                                 # 90秒以上まったく動かない顔 → 写真など
        t += 1
        ok = p.update(True, 0.1, 0.1, t)
    assert not ok
    q = PresenceFilter()
    t, ok = 0.0, False
    while t < 120:                                 # 人は少し動く → いる
        t += 1
        ok = q.update(True, 0.1 + (t % 10) * 0.01, 0.1, t)
    assert ok


def test_sleeps_after_quiet_even_with_face():
    e = EmotionEngine(0.0)
    t = 0.0
    while t < EmotionEngine.QUIET_SLEEP_AFTER + 15:
        e.on_face(True, t)
        e.update(t)
        t += 5
    assert e.sleeping                         # 顔は見えていても、話しかけられなければ眠る
    e.notice(t)
    assert not e.sleeping
