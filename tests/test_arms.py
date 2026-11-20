"""手の動き（ArmMotion）・Pi との送受信（ArmLink）・XIAO のファームのテスト。"""
import importlib.util
import os
import random
from pathlib import Path

from robot.arms import ARM_MAX, ArmLink, ArmMotion, gesture_for

DT = 1 / 30


def run(m, secs, **kw):
    out = (0.0, 0.0)
    for _ in range(int(secs / DT)):
        out = m.update(DT, **kw)
    return out


def test_rest_is_down():
    left, right = run(ArmMotion(rng=random.Random(0)), 3.0)
    assert left < 8 and right < 8


def test_surprise_raises_both():
    left, right = run(ArmMotion(rng=random.Random(0)), 2.0, emotion="surprise")
    assert left > 120 and right > 120


def test_sleepy_and_sad_hang_down():
    for emo in ("sleepy", "sad"):
        left, right = run(ArmMotion(rng=random.Random(0)), 2.0, emotion=emo)
        assert left < 3 and right < 3


def test_think_uses_one_hand():
    left, right = run(ArmMotion(rng=random.Random(0)), 2.0, emotion="think")
    assert left > 70 and right < 10


def test_wave_raises_right_then_returns():
    m = ArmMotion(rng=random.Random(0))
    m.wave()
    _, mid = run(m, 1.2)
    assert mid > 100
    _, end = run(m, 3.0)
    assert end < 10 and m.gesture is None


def test_banzai_both_up():
    m = ArmMotion(rng=random.Random(0))
    m.banzai()
    peak = 0.0
    for _ in range(int(2.0 / DT)):
        left, right = m.update(DT)
        peak = max(peak, min(left, right))
    assert peak > 140


def test_talking_moves_hands_a_little():
    left, right = run(ArmMotion(rng=random.Random(0)), 2.0, talking=1.0)
    assert 10 < left < 60 and 10 < right < 60


def test_always_in_range():
    m = ArmMotion(rng=random.Random(1))
    emos = ["excited", "dizzy", "cry", "angry", "confused", "joy", "love"]
    for i in range(600):
        if i % 40 == 0:
            m.do(["wave", "banzai", "perk", "nuzzle"][i // 40 % 4])
        left, right = m.update(DT, emotion=emos[i // 90 % len(emos)], talking=(i % 7) / 7,
                               present=i % 200 < 100)
        assert 0 <= left <= ARM_MAX and 0 <= right <= ARM_MAX


def test_gesture_words():
    assert gesture_for("イモ、バイバイ") == "wave"
    assert gesture_for("ばんざーい") is None
    assert gesture_for("ばんざい！") == "banzai"
    assert gesture_for("きょうは あめ") is None


def test_link_sends_and_swaps(tmp_path):
    out = tmp_path / "tty"
    out.write_text("")
    fds = []

    def opener(path):
        fd = os.open(path, os.O_WRONLY | os.O_APPEND)
        fds.append(fd)
        return fd

    link = ArmLink(str(out), swap=True, opener=opener)
    assert link.send(10, 90, now=0.0)
    assert not link.send(10, 90, now=0.01)   # 速すぎる
    assert not link.send(10, 90, now=0.5)    # 同じ角度
    assert link.send(10, 90, now=1.2)        # 1秒たてば同じでも送る
    link.relax()
    link.close()
    assert out.read_text() == "A 90 10\nA 90 10\nR\n"


def test_link_without_device(monkeypatch):
    monkeypatch.setattr("robot.arms.find_device", lambda: None)
    link = ArmLink()
    assert not link.send(0, 0, now=0.0)


def _firmware():
    path = Path(__file__).resolve().parent.parent / "firmware" / "arms_xiao" / "main.py"
    spec = importlib.util.spec_from_file_location("arms_fw", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # machine が無いので main() は動かない
    return mod


def test_firmware_moves_smoothly_and_relaxes_when_down():
    fw = _firmware()
    left = fw.Arm(0, fw.LEFT_DOWN_US, fw.LEFT_DIR)
    right = fw.Arm(1, fw.RIGHT_DOWN_US, fw.RIGHT_DIR)
    assert fw.handle("A 150 40", left, right)
    last = 0.0
    for _ in range(10):
        left.step(0.01)
        assert left.pos - last <= fw.MAX_SPEED * 0.01 + 1e-6
        last = left.pos
    for _ in range(300):
        left.step(0.01)
        right.step(0.01)
    assert abs(left.pos - 150) < 0.1 and abs(right.pos - 40) < 0.1 and left.on
    fw.handle("A 0 0", left, right)
    for _ in range(300):
        left.step(0.01)
    assert not left.on                 # おろして止まったら力を抜く
    assert fw.handle("A 999 -5", left, right)
    assert left.goal == fw.RANGE[1] and right.goal == fw.RANGE[0]
    assert fw.handle("H", left, right) and left.on
    assert fw.handle("R", left, right) and not left.on
    assert not fw.handle("A x y", left, right)
