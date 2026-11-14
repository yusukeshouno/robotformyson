# イモの首（XIAO RP2040 用 MicroPython）
#
# Pi から USB で 1 行ずつ受け取る:
#   N <左右> <上下>   角度（度）。真ん中が 0、左右は +が右、上下は +が上
#   R                 力を抜く（次の N でまた動く）
# 受け取った角度へ、速さと加速度を抑えて、なめらかに動かす。
# 3秒なにも来なければ、ゆっくり真ん中に戻って力を抜く（Pi が止まったとき用）。
import sys
import time

try:
    import select
    from machine import PWM, Pin
except ImportError:  # PC でのテスト用
    PWM = Pin = select = None

# ---- 組み立てに合わせて変える所 ----
PAN_PIN, TILT_PIN = 26, 27        # XIAO の D0, D1
PAN_CENTER_US, TILT_CENTER_US = 1500, 1500  # 真ん中のパルス幅。ずれていたら ±50 ずつ直す
US_PER_DEG = 2000 / 180           # MG90S: 500〜2500us で約180度
PAN_DIR, TILT_DIR = 1, 1          # 向きが逆なら -1
PAN_RANGE = (-40, 40)
TILT_RANGE = (-22, 18)
MAX_SPEED = 160.0                 # 度/秒
MAX_ACCEL = 900.0                 # 度/秒^2
RELAX_PAN_AFTER = 1.5             # 止まって この秒数たったら左右は力を抜く（ジーという音を止める）
TIMEOUT = 3.0


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class Axis:
    def __init__(self, pin, center_us, direction, rng, relax_after=None):
        self.pwm = None
        if PWM:
            self.pwm = PWM(Pin(pin))
            self.pwm.freq(50)
        self.center_us, self.dir, self.rng = center_us, direction, rng
        self.relax_after = relax_after
        self.pos = 0.0
        self.vel = 0.0
        self.goal = 0.0
        self.still = 0.0
        self.on = False

    def set(self, deg):
        g = clamp(deg, *self.rng)
        if abs(g - self.goal) > 0.3 or not self.on:
            self.still = 0.0
            self.on = True
        self.goal = g

    def relax(self):
        self.on = False
        self.vel = 0.0
        if self.pwm:
            self.pwm.duty_ns(0)

    def step(self, dt):
        if not self.on:
            return
        err = self.goal - self.pos
        # 止まれる速さ（v^2 = 2 a d）を超えないように加速・減速する
        stop_v = max(0.0, (2 * MAX_ACCEL * abs(err)) ** 0.5 - MAX_ACCEL * dt)
        want = clamp(err / dt if dt > 0 else 0, -min(MAX_SPEED, stop_v), min(MAX_SPEED, stop_v))
        dv = clamp(want - self.vel, -MAX_ACCEL * dt, MAX_ACCEL * dt)
        self.vel += dv
        self.pos += self.vel * dt
        if abs(self.goal - self.pos) < 0.05 and abs(self.vel) < 1:
            self.pos, self.vel = self.goal, 0.0
            self.still += dt
        if self.relax_after is not None and self.still > self.relax_after:
            self.relax()
            return
        if self.pwm:
            us = self.center_us + self.dir * self.pos * US_PER_DEG
            self.pwm.duty_ns(int(clamp(us, 500, 2500) * 1000))


def handle(line, pan, tilt):
    parts = line.split()
    if not parts:
        return False
    if parts[0] == "N" and len(parts) >= 3:
        try:
            p, t = float(parts[1]), float(parts[2])
        except ValueError:
            return False
        pan.set(p)
        tilt.set(t)
        return True
    if parts[0] == "R":
        pan.relax()
        tilt.relax()
        return True
    return False


def main():
    pan = Axis(PAN_PIN, PAN_CENTER_US, PAN_DIR, PAN_RANGE, RELAX_PAN_AFTER)
    tilt = Axis(TILT_PIN, TILT_CENTER_US, TILT_DIR, TILT_RANGE)
    pan.set(0)
    tilt.set(0)
    poll = select.poll()
    poll.register(sys.stdin, select.POLLIN)
    buf = ""
    last_cmd = time.ticks_ms()
    last = time.ticks_us()
    homing = False
    while True:
        while poll.poll(0):
            ch = sys.stdin.read(1)
            if ch in ("\n", "\r"):
                if handle(buf, pan, tilt):
                    last_cmd = time.ticks_ms()
                    homing = False
                buf = ""
            elif len(buf) < 40:
                buf += ch
        if time.ticks_diff(time.ticks_ms(), last_cmd) > TIMEOUT * 1000 and not homing:
            homing = True
            pan.set(0)
            tilt.set(0)
        if homing and tilt.on and tilt.still > 1.0:
            tilt.relax()
        now = time.ticks_us()
        dt = time.ticks_diff(now, last) / 1e6
        last = now
        pan.step(dt)
        tilt.step(dt)
        time.sleep_ms(10)


if PWM:
    main()
