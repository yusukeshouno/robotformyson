"""首の動き（NeckMotion）・Pi との送受信（NeckLink）・XIAO のファームのテスト。"""
import importlib.util
import os
import random
from pathlib import Path

from robot.neck import PAN_LIMIT, TILT_DOWN, TILT_UP, NeckLink, NeckMotion

DT = 1 / 30


def run(m, secs, **kw):
    out = (0.0, 0.0)
    for _ in range(int(secs / DT)):
        out = m.update(DT, **kw)
    return out


def test_turns_toward_person_slowly():
    m = NeckMotion(rng=random.Random(0))
    pan1, _ = run(m, 0.3, target=(0.8, 0.0), eye_gaze=(0.8, 0.0))
    pan2, _ = run(m, 1.5, target=(0.8, 0.0), eye_gaze=(0.8, 0.0))
    assert 0 < pan1 < pan2


def test_person_in_center_keeps_head_still():
    m = NeckMotion(rng=random.Random(0))
    pan, tilt = run(m, 3.0, target=(0.1, 0.05), eye_gaze=(0.1, 0.05))
    assert abs(pan) < 1.0 and abs(tilt) < 1.0


def test_body_camera_points_at_person():
    m = NeckMotion(camera_on_head=False, rng=random.Random(0))
    pan, _ = run(m, 4.0, target=(-0.6, 0.0), eye_gaze=(-0.6, 0.0))
    assert pan < -10


def test_sleepy_droops():
    m = NeckMotion(rng=random.Random(0))
    _, tilt = run(m, 4.0, emotion="sleepy", sleeping=True)
    assert tilt < -15


def test_nod_goes_down_and_comes_back():
    m = NeckMotion(rng=random.Random(0))
    target = (0.0, 0.0)
    run(m, 1.0, target=target)
    m.nod()
    lows = [m.update(DT, target=target)[1] for _ in range(int(0.45 / DT))]
    assert min(lows) < -4
    _, tilt = run(m, 0.5, target=target)
    assert abs(tilt) < 1.0 and not m.gestures


def test_averting_eyes_turns_head_a_little():
    m = NeckMotion(rng=random.Random(0))
    pan, tilt = run(m, 2.0, target=(0.0, 0.0), eye_gaze=(0.6, -0.4))
    assert 3 < pan < 10 and tilt > 2


def test_always_within_limits():
    m = NeckMotion(rng=random.Random(1))
    for emo in ("dizzy", "excited", "think", "shy", "confused", "cry"):
        for _ in range(200):
            m.nod()
            m.shake()
            pan, tilt = m.update(DT, emotion=emo, target=(1.0, 1.0), eye_gaze=(1, 1), talking=1)
            assert -PAN_LIMIT <= pan <= PAN_LIMIT and -TILT_DOWN <= tilt <= TILT_UP


class FakeDev:
    def __init__(self):
        r, w = os.pipe()
        self.r, self.w = r, w

    def read(self):
        return os.read(self.r, 4096).decode()


def test_link_sends_angles_and_skips_repeats():
    dev = FakeDev()
    link = NeckLink("/dev/fake", invert_pan=True, opener=lambda p: dev.w)
    assert link.send(10.0, -5.0, now=0.0)
    assert not link.send(10.05, -5.0, now=0.1)    # ほとんど同じ
    assert link.send(12.0, -5.0, now=0.2)
    assert dev.read() == "N -10.0 -5.0\nN -12.0 -5.0\n"


def test_link_without_device_does_nothing(monkeypatch):
    monkeypatch.setattr("robot.neck.find_device", lambda: None)
    link = NeckLink()
    assert not link.send(0, 0, now=0.0)


def test_link_reconnects_after_unplug():
    dev = FakeDev()
    calls = []

    def opener(p):
        calls.append(p)
        return dev.w
    link = NeckLink("/dev/fake", opener=opener)
    assert link.send(1, 1, now=0.0)
    os.close(dev.w)                      # 抜けた
    assert not link.send(5, 5, now=0.1)
    assert link.fd is None
    dev2 = FakeDev()
    link.opener = lambda p: dev2.w
    assert link.send(5, 5, now=5.0)      # さしなおした
    assert dev2.read() == "N 5.0 5.0\n"


def _firmware():
    path = Path(__file__).parent.parent / "firmware" / "neck_xiao" / "main.py"
    spec = importlib.util.spec_from_file_location("neck_fw", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # machine がないので main() は動かない
    return mod


def test_firmware_moves_smoothly_and_stops_on_target():
    fw = _firmware()
    ax = fw.Axis(0, 1500, 1, (-40, 40))
    fw.handle("N 30 0", ax, fw.Axis(1, 1500, 1, (-22, 18)))
    prev_v, peak = 0.0, 0.0
    for _ in range(200):
        ax.step(0.01)
        assert abs(ax.vel - prev_v) <= fw.MAX_ACCEL * 0.01 + 1e-6   # 急に速くならない
        assert abs(ax.vel) <= fw.MAX_SPEED + 1e-6
        prev_v, peak = ax.vel, max(peak, ax.pos)
    assert ax.pos == 30.0 and peak <= 30.5


def test_firmware_clamps_and_relaxes():
    fw = _firmware()
    pan = fw.Axis(0, 1500, 1, (-40, 40), relax_after=0.5)
    tilt = fw.Axis(1, 1500, 1, (-22, 18))
    assert fw.handle("N 90 -90", pan, tilt)
    assert pan.goal == 40 and tilt.goal == -22
    for _ in range(300):
        pan.step(0.01)
    assert not pan.on
    assert not fw.handle("hello", pan, tilt)
    assert fw.handle("R", pan, tilt) and not tilt.on
