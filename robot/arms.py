"""イモの手（左右の2本。箱の横の壁にサーボが1個ずつ）。

サーボは XIAO RP2040（firmware/arms_xiao/main.py）が動かし、Pi からは USB で
「A <左手の角度> <右手の角度>」を送るだけ。角度は 0 が手をおろした所、+ で前から上へあがる（度）。
左右は「イモから見て」の左右（向かい合うと、こちらの右側がイモの左手）。
手の基板がつながっていなくても、そのまま動く（何もしない）。

    python3 -m robot.arms down    # 手をおろした位置で止める（ホーンを付けるとき）
    python3 -m robot.arms test    # 左 → 右 → バンザイ → バイバイ を順に試す
    python3 -m robot.arms relax   # 力を抜く

手の動きは ArmMotion が決める:
- うれしい: 両手をぱたぱた。わくわく: バンザイしてゆらす。びっくり: ぱっと両手をあげる
- かなしい・ねむい: だらんとおろす。てれる: 両手で顔をかくす。考えちゅう: 片手をあごに
- 話している間: 声に合わせて手を少しずつ動かす（身ぶり）
- 「バイバイ」「こんにちは」: 手をふる。「バンザイ」: バンザイ。なでられる: もじもじ
- だれもいないとき: ほとんどおろしたまま（電池の節約）、ときどき片手を少しあげる
"""
import glob
import math
import os
import random
import sys
import time

ARM_MAX = 160.0   # 手をあげる最大の角度（真上が 180）

# 感情ごとの手の構え: (左手, 右手, ゆらし方, ゆれの大きさ, ゆれの速さ Hz)
# ゆらし方: both=両手いっしょ / alt=交互 / left・right=片手だけ
POSES = {
    "neutral": (4, 4, None, 0, 0),
    "joy": (55, 55, "alt", 18, 2.0),
    "laugh": (40, 40, "both", 12, 4.0),
    "excited": (150, 150, "alt", 10, 2.5),
    "love": (70, 70, "both", 6, 0.8),
    "wink": (4, 120, None, 0, 0),
    "sad": (0, 0, None, 0, 0),
    "cry": (110, 110, "alt", 10, 3.0),     # 目をこする
    "angry": (65, 65, "alt", 18, 5.0),     # ぷんぷん
    "surprise": (150, 150, None, 0, 0),
    "sleepy": (0, 0, None, 0, 0),
    "think": (100, 4, None, 0, 0),         # 片手をあごに
    "shy": (100, 100, "both", 5, 0.7),     # 顔をかくす
    "confused": (4, 145, "right", 10, 2.0),  # 頭をかく
    "dizzy": (60, 60, "alt", 25, 0.8),
}

# 会話の言葉から出すしぐさ
GESTURE_WORDS = {
    "wave": ("ばいばい", "バイバイ", "またね", "こんにちは", "おはよう", "やっほー", "ただいま", "おかえり"),
    "banzai": ("ばんざい", "バンザイ", "万歳", "やったー", "やった"),
}


def gesture_for(text):
    """言葉から手のしぐさを選ぶ。なければ None。"""
    for name, words in GESTURE_WORDS.items():
        if any(w in text for w in words):
            return name
    return None


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _follow(cur, goal, dt, tau):
    """cur を goal に、時定数 tau 秒でなめらかに近づける。"""
    return cur + (goal - cur) * (1 - math.exp(-dt / tau)) if tau > 0 else goal


class ArmMotion:
    """表情・会話から、手の角度 (左手, 右手) を毎フレーム決める。ハードウェアは触らない。"""

    SMOOTH = 0.22   # 手のなめらかさ（秒）

    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.t = 0.0
        self.out = [0.0, 0.0]
        self.pose = [0.0, 0.0]
        self.idle = [0.0, 0.0]
        self.next_idle = 5.0
        self.idle_until = 0.0
        self.bob = 0.0
        self.gesture = None   # (始まった時刻, 種類, 長さ)

    # ---- しぐさ ----
    def wave(self):
        """右手をあげて ふる（バイバイ・こんにちは）。"""
        self._start("wave", 2.6)

    def banzai(self):
        """両手をあげて バンザイ（2回）。"""
        self._start("banzai", 2.0)

    def perk(self):
        """はっと両手を少しあげる（名前を呼ばれた・目が覚めた）。"""
        self._start("perk", 0.8)

    def nuzzle(self):
        """なでられて、もじもじ。"""
        self._start("nuzzle", 1.8)

    def do(self, name):
        """名前でしぐさをする（wave / banzai / perk / nuzzle）。"""
        getattr(self, name, lambda: None)()

    def _start(self, kind, length):
        self.gesture = (self.t, kind, length)

    def _gesture(self):
        """しぐさの (左, 右, 重み)。重み 0 ならしぐさなし。"""
        if not self.gesture:
            return 0.0, 0.0, 0.0
        start, kind, length = self.gesture
        p = (self.t - start) / length
        if p >= 1:
            self.gesture = None
            return 0.0, 0.0, 0.0
        # 最初と最後の 15% でなめらかに入って出る
        w = min(1.0, p / 0.15, (1 - p) / 0.15)
        s = self.t - start
        if kind == "wave":
            return 4.0, 135 + 22 * math.sin(2 * math.pi * 2.0 * s), w
        if kind == "banzai":
            up = 155 if (p % 0.5) < 0.32 else 40
            return up, up, w
        if kind == "perk":
            return 45.0, 45.0, w
        if kind == "nuzzle":
            return 85 + 8 * math.sin(2 * math.pi * 1.5 * s), 85 - 8 * math.sin(2 * math.pi * 1.5 * s), w
        return 0.0, 0.0, 0.0

    # ---- 毎フレーム ----
    def update(self, dt, emotion="neutral", talking=0.0, sleeping=False, present=True):
        """手の角度 (左手, 右手) を返す。talking は口の開き 0..1。"""
        self.t += dt
        name = "sleepy" if sleeping else emotion
        pl, pr, mode, amp, hz = POSES.get(name, POSES["neutral"])
        for i, v in enumerate((pl, pr)):
            self.pose[i] = _follow(self.pose[i], v, dt, 0.3)
        wave = math.sin(2 * math.pi * hz * self.t) * amp if mode else 0.0
        goal = list(self.pose)
        if mode == "both":
            goal = [goal[0] + wave, goal[1] + wave]
        elif mode == "alt":
            goal = [goal[0] + wave, goal[1] - wave]
        elif mode == "left":
            goal[0] += wave
        elif mode == "right":
            goal[1] += wave

        # 話している間の身ぶり: 口の動きに合わせて、左右交互に少し
        self.bob = _follow(self.bob, talking, dt, 0.15)
        if not sleeping:
            sway = math.sin(2 * math.pi * 0.6 * self.t)
            goal[0] += self.bob * (18 + 10 * sway)
            goal[1] += self.bob * (18 - 10 * sway)

        # だれもいないとき: ときどき片手を少しあげる。ほかは おろしたまま
        if not present and not sleeping and name == "neutral":
            if self.t >= self.next_idle:
                side = self.rng.randrange(2)
                self.idle = [0.0, 0.0]
                self.idle[side] = self.rng.uniform(20, 45)
                self.idle_until = self.t + self.rng.uniform(0.8, 1.6)
                self.next_idle = self.t + self.rng.uniform(8.0, 16.0)
            if self.t >= self.idle_until:
                self.idle = [0.0, 0.0]
        else:
            self.idle = [0.0, 0.0]
        goal = [goal[0] + self.idle[0], goal[1] + self.idle[1]]

        for i in (0, 1):
            self.out[i] = _follow(self.out[i], goal[i], dt, self.SMOOTH)
        gl, gr, w = self._gesture()
        left = self.out[0] * (1 - w) + gl * w
        right = self.out[1] * (1 - w) + gr * w
        return _clamp(left, 0.0, ARM_MAX), _clamp(right, 0.0, ARM_MAX)


def find_device():
    """手の基板（XIAO RP2040 の MicroPython）の場所を探す。なければ None。"""
    env = os.environ.get("ROBOT_ARMS")
    if env:
        return env
    ids = sorted(glob.glob("/dev/serial/by-id/*"))
    for key in ("MicroPython", "Seeed", "XIAO", "RP2040", "Pico"):
        for p in ids:
            if key.lower() in os.path.basename(p).lower():
                return p
    acm = sorted(glob.glob("/dev/ttyACM*"))
    return acm[0] if acm else None


class ArmLink:
    """手の基板に角度を送る。抜けても、さしなおせばつなぎなおす。"""

    RATE = 30.0   # 1秒に送る回数の上限
    RETRY = 2.0

    def __init__(self, device=None, swap=False, opener=None):
        self.device = device
        self.swap = swap
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

    def send(self, left, right, now=None):
        """角度を送る。送ったら True。"""
        now = time.monotonic() if now is None else now
        if not self.connect(now) or now - self.last_send < 1.0 / self.RATE:
            return False
        if self.swap:
            left, right = right, left
        a, b = round(left, 1), round(right, 1)
        if self.last and abs(a - self.last[0]) < 0.3 and abs(b - self.last[1]) < 0.3 \
                and now - self.last_send < 1.0:
            return False  # ほとんど同じなら送らない（1秒に1回は送って、生きていると知らせる）
        self.last_send = now
        if self._write(f"A {a} {b}\n"):
            self.last = (a, b)
            return True
        return False

    def hold_down(self):
        if self.connect():
            self._write("H\n")

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


def add_arm_args(ap):
    ap.add_argument("--no-arms", action="store_true", help="手を動かさない")
    ap.add_argument("--arms-device", default=None,
                    help="手の基板の場所（ふつうは自動。/dev/ttyACM0 など）")
    ap.add_argument("--arms-swap", action="store_true",
                    help="左手と右手が逆に動くときに指定")


def _play(link, steps):
    for left, right, secs in steps:
        end = time.monotonic() + secs
        while time.monotonic() < end:
            link.last_send = -1e9
            link.send(left, right)
            time.sleep(0.05)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "test"
    link = ArmLink(argv[1] if len(argv) > 1 else None)
    if not link.connect():
        print("手の基板が見つかりません。USB をさして、firmware/arms_xiao/README.md の手順で書きこんでください。")
        return 1
    print(f"手の基板: {link.label}")
    if cmd == "down":
        link.hold_down()
        print("手をおろした位置で1分間止めます。この間に、手がまっすぐ下を向くようにホーンを付けてください。")
    elif cmd == "relax":
        link.relax()
        print("力を抜きました。")
    else:
        print("左手 → 右手 → バンザイ → バイバイ")
        _play(link, [(0, 0, 0.8), (90, 0, 1.2), (0, 0, 0.8), (0, 90, 1.2), (0, 0, 0.8)])
        m = ArmMotion()
        for gesture in (m.banzai, m.wave):
            gesture()
            t0 = time.monotonic()
            while m.gesture:
                now = time.monotonic()
                left, right = m.update(now - t0)
                t0 = now
                link.last_send = -1e9
                link.send(left, right)
                time.sleep(0.03)
        _play(link, [(0, 0, 0.8)])
        print("おわり。左右が逆なら --arms-swap、あがらずに下がる手は main.py の *_DIR を逆にします。")
    link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
