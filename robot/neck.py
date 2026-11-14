"""イモの首（左右のパン・上下のチルトの2軸）。

サーボは XIAO RP2040（firmware/neck_xiao/main.py）が動かし、Pi からは USB で
「N <左右の角度> <上下の角度>」を送るだけ。角度は真ん中が 0、左右は +が右、上下は +が上（度）。
首の基板がつながっていなくても、そのまま動く（何もしない）。

    python3 -m robot.neck center   # サーボを真ん中にする（ホーンを付ける前に）
    python3 -m robot.neck test     # 左右・上下・うなずき・首ふりを順に試す
    python3 -m robot.neck relax    # 力を抜く

首の動きは NeckMotion が決める:
- 人がいるとき: 目で先に見て、首はゆっくりその人の方を向く
- 目をそらすとき（話しはじめ・照れ）: 首も少しそらす
- 聞こえたとき: 小さくうなずく。話している間は声に合わせて少しゆれる
- だれもいないとき: ゆっくりゆらゆら、ときどきキョロキョロ
- 眠いとき: こくんと下を向いて、寝息のようにゆれる
- うれしい・わくわく: 左右にふりふり。悲しい: うつむく。考えちゅう: 上を向く
"""
import glob
import math
import os
import random
import sys
import time

PAN_LIMIT = 40.0     # 左右に動かす最大の角度
TILT_UP = 18.0       # 上を向く最大
TILT_DOWN = 22.0     # 下を向く最大

# 感情ごとの首の構え: (左右, 上下, ゆれの軸, ゆれの大きさ, ゆれの速さ Hz)
POSES = {
    "neutral": (0, 0, None, 0, 0),
    "joy": (0, 4, "pan", 5, 2.2),
    "laugh": (0, 6, "tilt", 3, 4.0),
    "excited": (0, 6, "pan", 6, 2.8),
    "love": (0, -3, "pan", 3, 0.8),
    "wink": (6, 2, None, 0, 0),
    "sad": (0, -12, None, 0, 0),
    "cry": (0, -14, "tilt", 2, 3.0),
    "angry": (0, -4, "pan", 2, 6.0),
    "surprise": (0, 10, None, 0, 0),
    "sleepy": (0, -20, "tilt", 2, 0.25),
    "think": (12, 10, None, 0, 0),
    "shy": (-14, -10, "pan", 2, 0.7),
    "confused": (0, 3, "pan", 10, 0.35),
    "dizzy": (0, 0, "circle", 8, 0.8),
}


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _follow(cur, goal, dt, tau):
    """cur を goal に、時定数 tau 秒でなめらかに近づける。"""
    return cur + (goal - cur) * (1 - math.exp(-dt / tau)) if tau > 0 else goal


class NeckMotion:
    """表情・視線・会話から、首の角度 (左右, 上下) を毎フレーム決める。ハードウェアは触らない。"""

    TRACK_RATE = 35.0    # 人が画面の端にいるとき、首を回す速さ（度/秒）
    DEADBAND = 0.18      # 人がこれより真ん中に近ければ首は回さない（目だけで追う）
    FOLLOW = 25.0        # カメラが体に付いているとき、人の位置 1.0 で向く角度
    GLANCE = 10.0        # 目の動き 1.0 につき、首も一緒に動く角度
    SMOOTH = 0.35        # 首のなめらかさ（秒）

    def __init__(self, camera_on_head=True, rng=None):
        self.camera_on_head = camera_on_head
        self.rng = rng or random.Random()
        self.t = 0.0
        self.base = [0.0, 0.0]      # 首の向き（人の方・あたりを見る）
        self.out = [0.0, 0.0]       # なめらかにした角度
        self.pose = [0.0, 0.0]
        self.idle_goal = (0.0, 0.0)
        self.next_idle = 0.0
        self.gestures = []          # (始まった時刻, 種類, 長さ, 大きさ)
        self.bob = 0.0

    # ---- しぐさ ----
    def nod(self, times=1, depth=7.0):
        """うなずく（あいづち）。"""
        self._gesture("nod", 0.45 * times, depth, times)

    def shake(self, times=2, width=8.0):
        """首を横にふる（いやいや・ちがうよ）。"""
        self._gesture("shake", 0.4 * times, width, times)

    def perk(self):
        """はっと顔を上げる（名前を呼ばれた・目が覚めた）。"""
        self._gesture("perk", 0.7, 8.0, 1)

    def nuzzle(self):
        """なでられて、首をかしげるように寄せる。"""
        self._gesture("nuzzle", 1.6, 8.0, 1)

    def _gesture(self, kind, length, size, times):
        self.gestures = [g for g in self.gestures if g[1] != kind]
        self.gestures.append((self.t, kind, length, size, times))

    def _gesture_offset(self):
        pan = tilt = 0.0
        alive = []
        for start, kind, length, size, times in self.gestures:
            p = (self.t - start) / length
            if p >= 1:
                continue
            alive.append((start, kind, length, size, times))
            env = math.sin(math.pi * p)  # 始めと終わりは 0（なめらかにつながる）
            if kind == "nod":
                tilt -= size * abs(math.sin(math.pi * p * times)) * (0.6 + 0.4 * env)
            elif kind == "shake":
                pan += size * math.sin(2 * math.pi * p * times) * env
            elif kind == "perk":
                tilt += size * env
            elif kind == "nuzzle":
                pan += size * 0.6 * env
                tilt -= size * 0.5 * env
        self.gestures = alive
        return pan, tilt

    # ---- 毎フレーム ----
    def update(self, dt, emotion="neutral", eye_gaze=(0.0, 0.0), target=None,
               talking=0.0, sleeping=False):
        """首の角度 (左右, 上下) を返す。

        eye_gaze: いまの目の向き（Face.gaze。-1..1、y は下が +）。
        target: 見えている人の位置（-1..1）か None。
        talking: 口の開き 0..1（話しているときに少しゆれる）。
        """
        self.t += dt
        gx, gy = eye_gaze
        if target is not None:
            tx, ty = target
            if self.camera_on_head:  # 首を回すと人が真ん中に寄るので、少しずつ回す
                for i, v in enumerate((tx, -ty)):
                    if abs(v) > self.DEADBAND:
                        step = (v - math.copysign(self.DEADBAND, v)) * self.TRACK_RATE * dt
                        self.base[i] += step
            else:
                self.base[0] = _follow(self.base[0], tx * self.FOLLOW, dt, 0.8)
                self.base[1] = _follow(self.base[1], -ty * self.FOLLOW * 0.6, dt, 0.8)
            look = ((gx - tx) * self.GLANCE, -(gy - ty) * self.GLANCE)  # 目そらしの分だけ
        else:
            if sleeping:
                self.idle_goal = (0.0, 0.0)
            elif self.t >= self.next_idle:  # ときどき、あたりを見回す
                self.idle_goal = (self.rng.uniform(-22, 22), self.rng.uniform(-4, 8))
                self.next_idle = self.t + self.rng.uniform(4.0, 9.0)
            for i in (0, 1):
                self.base[i] = _follow(self.base[i], self.idle_goal[i], dt, 1.5)
            look = (0.0, 0.0) if sleeping else (gx * self.GLANCE, -gy * self.GLANCE)
        self.base[0] = _clamp(self.base[0], -PAN_LIMIT, PAN_LIMIT)
        self.base[1] = _clamp(self.base[1], -TILT_DOWN, TILT_UP)

        pp, pt, axis, amp, hz = POSES.get("sleepy" if sleeping else emotion, POSES["neutral"])
        for i, v in enumerate((pp, pt)):
            self.pose[i] = _follow(self.pose[i], v, dt, 0.6)
        wave = math.sin(2 * math.pi * hz * self.t) * amp if axis else 0.0
        sway = 0.0
        if target is None and not sleeping:  # だれもいないとき、ゆらゆら
            sway = 3.0 * math.sin(self.t * 0.5)
        self.bob = _follow(self.bob, talking, dt, 0.12)
        goal = [self.base[0] + look[0] + self.pose[0] + sway,
                self.base[1] + look[1] + self.pose[1] + self.bob * 3.0]
        if axis == "pan":
            goal[0] += wave
        elif axis == "tilt":
            goal[1] += wave
        elif axis == "circle":
            goal[0] += wave
            goal[1] += math.cos(2 * math.pi * hz * self.t) * amp * 0.6
        for i in (0, 1):
            self.out[i] = _follow(self.out[i], goal[i], dt, self.SMOOTH)
        gp, gt = self._gesture_offset()  # しぐさは形を崩さないよう、なめらかにしない
        return (_clamp(self.out[0] + gp, -PAN_LIMIT, PAN_LIMIT),
                _clamp(self.out[1] + gt, -TILT_DOWN, TILT_UP))


def find_device():
    """首の基板（XIAO RP2040 の MicroPython）の場所を探す。なければ None。"""
    env = os.environ.get("ROBOT_NECK")
    if env:
        return env
    ids = sorted(glob.glob("/dev/serial/by-id/*"))
    for key in ("MicroPython", "Seeed", "XIAO", "RP2040", "Pico"):
        for p in ids:
            if key.lower() in os.path.basename(p).lower():
                return p
    acm = sorted(glob.glob("/dev/ttyACM*"))
    return acm[0] if acm else None


class NeckLink:
    """首の基板に角度を送る。抜けても、さしなおせばつなぎなおす。"""

    RATE = 30.0   # 1秒に送る回数の上限
    RETRY = 2.0

    def __init__(self, device=None, invert_pan=False, invert_tilt=False, opener=None):
        self.device = device
        self.sign = (-1.0 if invert_pan else 1.0, -1.0 if invert_tilt else 1.0)
        self.opener = opener or self._open
        self.fd = None
        self.last_try = -1e9
        self.last_send = -1e9
        self.last = None
        self.label = None

    @staticmethod
    def _open(path):
        fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            import termios
            import tty
            tty.setraw(fd)
            attrs = termios.tcgetattr(fd)
            attrs[3] &= ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
        except (ImportError, OSError):
            pass
        return fd

    def connect(self, now=None):
        now = time.monotonic() if now is None else now
        if self.fd is not None:
            return True
        if now - self.last_try < self.RETRY:
            return False
        self.last_try = now
        path = self.device or find_device()
        if not path:
            return False
        try:
            self.fd = self.opener(path)
        except OSError:
            return False
        self.label = path
        self.last = None
        return True

    def _write(self, line):
        try:
            os.write(self.fd, line.encode())
            return True
        except BlockingIOError:
            return True  # 今回はあきらめて、次に送る
        except OSError:
            self.close()
            return False

    def send(self, pan, tilt, now=None):
        """角度を送る。送ったら True。"""
        now = time.monotonic() if now is None else now
        if not self.connect(now) or now - self.last_send < 1.0 / self.RATE:
            return False
        p, t = round(pan * self.sign[0], 1), round(tilt * self.sign[1], 1)
        if self.last and abs(p - self.last[0]) < 0.2 and abs(t - self.last[1]) < 0.2 \
                and now - self.last_send < 1.0:
            return False  # ほとんど同じなら送らない（1秒に1回は送って、生きていると知らせる）
        self.last_send = now
        if self._write(f"N {p} {t}\n"):
            self.last = (p, t)
            return True
        return False

    def relax(self):
        if self.connect():
            self._write("R\n")

    def close(self):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
        self.fd = None


def add_neck_args(ap):
    ap.add_argument("--no-neck", action="store_true", help="首を動かさない")
    ap.add_argument("--neck-device", default=None,
                    help="首の基板の場所（ふつうは自動。/dev/ttyACM0 など）")
    ap.add_argument("--neck-invert-pan", action="store_true",
                    help="首が人と逆の方へ向くときに指定")
    ap.add_argument("--neck-invert-tilt", action="store_true",
                    help="うなずくと上を向いてしまうときに指定")
    ap.add_argument("--neck-camera", choices=["head", "body"], default="head",
                    help="カメラが頭（画面）に付いているか、体に付いているか")
    ap.add_argument("--neck-preview", action="store_true",
                    help="首の動きを画面の顔の位置で見せる（首の部品がなくても試せる）")


def _demo(link, steps):
    for pan, tilt, secs in steps:
        end = time.monotonic() + secs
        while time.monotonic() < end:
            link.last_send = -1e9
            link.send(pan, tilt)
            time.sleep(0.05)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "test"
    link = NeckLink(argv[1] if len(argv) > 1 else None)
    if not link.connect():
        print("首の基板が見つかりません。USB をさして、firmware/neck_xiao/README.md の手順で書きこんでください。")
        return 1
    print(f"首の基板: {link.label}")
    if cmd == "center":
        _demo(link, [(0, 0, 1.0)])
        print("真ん中にしました。この向きでホーンを付けてください。")
    elif cmd == "relax":
        link.relax()
        print("力を抜きました。")
    else:
        m = NeckMotion()
        steps = [(0, 0, 1), (-30, 0, 1.2), (30, 0, 1.5), (0, 0, 1), (0, 15, 1), (0, -18, 1.2), (0, 0, 1)]
        print("左 → 右 → 上 → 下 → うなずき → 首ふり")
        _demo(link, steps)
        for gesture in (m.nod, m.shake):
            gesture()
            t0 = time.monotonic()
            while m.gestures:
                now = time.monotonic()
                pan, tilt = m.update(now - t0)
                t0 = now
                link.last_send = -1e9
                link.send(pan, tilt)
                time.sleep(0.03)
        _demo(link, [(0, 0, 0.8)])
        print("おわり。向きが逆なら --neck-invert-pan / --neck-invert-tilt を付けます。")
    link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
