# イモの手（XIAO RP2040 用 MicroPython）
#
# Pi から USB で 1 行ずつ受け取る:
#   A <左手> <右手>   角度（度）。0 = 手をおろした所、+ で前から上へあがる（150 くらいでバンザイ）
#   H                 手をおろした位置で1分間 力を入れたままにする（ホーンを付けるとき）
#   R                 力を抜く（次の A でまた動く）
# 受け取った角度へ、速さと加速度を抑えて、なめらかに動かす。
# 手をおろして止まったら力を抜く（ジーという音を止める・電池の節約）。
# 3秒なにも来なければ、ゆっくり手をおろして力を抜く（Pi が止まったとき用）。
import sys
import time

try:
    import select
    from machine import PWM, Pin
except ImportError:  # PC でのテスト用
    PWM = Pin = select = None

# ---- 組み立てに合わせて変える所 ----
LEFT_PIN, RIGHT_PIN = 26, 27      # XIAO の D0（イモの左手）, D1（イモの右手）
# 手をおろした所のパルス幅。左右の箱の壁に向かい合わせに付けるので、回る向きが逆になる
LEFT_DOWN_US, RIGHT_DOWN_US = 700, 2300
LEFT_DIR, RIGHT_DIR = 1, -1       # 手をあげると下がってしまう方は、向きを逆に（1 ⇔ -1）
US_PER_DEG = 2000 / 180           # SG90 / MG90S: 500〜2500us で約180度
RANGE = (0, 160)                  # 手が箱や机にぶつかるなら、せまくする
MAX_SPEED = 220.0                 # 度/秒
MAX_ACCEL = 1400.0                # 度/秒^2
RELAX_DOWN_AFTER = 1.0            # 手をおろして この秒数たったら力を抜く
TIMEOUT = 3.0


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class Arm:
    def __init__(self, pin, down_us, direction):
        self.pwm = None
        if PWM:
            self.pwm = PWM(Pin(pin))
            self.pwm.freq(50)
        self.down_us, self.dir = down_us, direction
        self.pos = 0.0
        self.vel = 0.0
        self.goal = 0.0
        self.still = 0.0
        self.on = False

    def set(self, deg):
        g = clamp(deg, *RANGE)
        if abs(g - self.goal) > 0.3 or (not self.on and g > 1.0):
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
        if abs(self.goal - self.pos) < 0.5 and abs(self.vel) < 15:  # ほぼ着いたら ぴたっと止める
            self.pos, self.vel = self.goal, 0.0
            self.still += dt
        if self.goal < 1.0 and self.still > RELAX_DOWN_AFTER:
            self.relax()  # おろした手は重さでそのまま。力はいらない
            return
        if self.pwm:
            us = self.down_us + self.dir * self.pos * US_PER_DEG
            self.pwm.duty_ns(int(clamp(us, 500, 2500) * 1000))


def handle(line, left, right):
    parts = line.split()
    if not parts:
        return False
    if parts[0] == "A" and len(parts) >= 3:
        try:
            a, b = float(parts[1]), float(parts[2])
        except ValueError:
            return False
        left.set(a)
        right.set(b)
        return True
    if parts[0] == "H":  # 手をおろした位置で力を入れる（ホーンを付けるとき）
        left.set(0)
        right.set(0)
        left.on = right.on = True
        left.still = right.still = -60.0
        return True
    if parts[0] == "R":
        left.relax()
        right.relax()
        return True
    return False


def main():
    left = Arm(LEFT_PIN, LEFT_DOWN_US, LEFT_DIR)
    right = Arm(RIGHT_PIN, RIGHT_DOWN_US, RIGHT_DIR)
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
                if handle(buf, left, right):
                    last_cmd = time.ticks_ms()
                    homing = False
                buf = ""
            elif len(buf) < 40:
                buf += ch
        if time.ticks_diff(time.ticks_ms(), last_cmd) > TIMEOUT * 1000 and not homing:
            homing = True
            left.set(0)
            right.set(0)
        now = time.ticks_us()
        dt = time.ticks_diff(now, last) / 1e6
        last = now
        left.step(dt)
        right.step(dt)
        time.sleep_ms(10)


if PWM:
    main()
