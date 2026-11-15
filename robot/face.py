"""ピクセルアートの顔。

48×48 の小さな「LED マトリクス」に顔を描き、拡大してドットの粒が見えるように表示する。
感情ごとのパラメータ（目の開き・まぶた・眉・口など）を目標値へなめらかに補間し、
まばたき・視線・呼吸・感情ごとの動き（はねる・ふるえる・ゆれる）・小物（涙・Z・ハート・？など）を自動で付ける。
目の形が変わるとき（ふつうの目 → ハート目など）は、一度まばたきして入れかわる。
"""
import math
import random

import numpy as np
import pygame

N = 48                    # ドットの数（一辺）
BG = (6, 7, 12)           # ドットのすきま・画面の外
OFF = (17, 20, 33)        # 消えているドット
PINK = (255, 105, 160)
TEAR = (120, 200, 255)
GOLD = (255, 230, 120)
RED = (255, 70, 70)

# 感情ごとの目標パラメータ。省略したキーは DEFAULTS の値になる。
DEFAULTS = {
    "open": 1.0,        # 目の開き 0..1
    "scale": 1.0,       # 目の大きさ
    "lid_top": 0.0,     # 上まぶたが覆う割合
    "tilt": 0.0,        # 上まぶたの傾き（+ で内側が下がる＝怒り、- で外側が下がる＝悲しみ）
    "lid_bot": 0.0,     # 下まぶたが覆う割合（笑い目）
    "skew": 0.0,        # 左右の目の大きさの差（+ で左が大きい。困り顔）
    "brow": 0.0,        # 眉の濃さ 0..1
    "brow_y": 0.0,      # 眉の高さ（+ で上がる）
    "brow_tilt": 0.0,   # 眉の傾き（+ で内側が下がる＝怒り）
    "brow_skew": 0.0,   # 眉の左右差（+ で左が上がる）
    "mouth": 0.3,       # 口の曲がり（+ 笑顔、- への字）
    "mouth_open": 0.0,  # 口の開き
    "mouth_w": 1.0,     # 口の幅
    "round": 0.0,       # 丸い「o」の口
    "wavy": 0.0,        # 波うつ口（もじもじ・こまった）
    "blush": 0.0,       # ほっぺの斜線
    "color": (90, 220, 255),
}
# 数値にならない設定（目の形・動き・小物）。補間せず切りかえる。
STYLE_DEFAULTS = {"eyes": ("normal", "normal"), "motion": None, "fx": ()}

EMOTIONS = {
    "neutral": {},
    "joy": {"scale": 1.05, "lid_bot": 0.5, "mouth": 0.9, "mouth_open": 0.45,
            "blush": 0.6, "color": (120, 235, 255), "motion": "bounce"},
    "sad": {"open": 0.85, "scale": 0.95, "lid_top": 0.3, "tilt": -0.5,
            "brow": 0.8, "brow_tilt": -0.6, "brow_y": 0.3,
            "mouth": -0.6, "mouth_w": 0.8, "color": (80, 140, 255),
            "motion": "droop", "fx": ("tear",)},
    "angry": {"open": 0.9, "lid_top": 0.2, "tilt": 0.6, "lid_bot": 0.1,
              "brow": 1.0, "brow_tilt": 1.0, "brow_y": -0.4,
              "mouth": -0.45, "mouth_open": 0.15, "color": (255, 90, 80),
              "motion": "shake", "fx": ("vein",)},
    "surprise": {"scale": 1.2, "mouth": 0.0, "round": 1.0, "brow": 0.9, "brow_y": 1.0,
                 "color": (150, 240, 255), "motion": "jump", "fx": ("exclaim",)},
    "sleepy": {"open": 0.0, "mouth": 0.0, "mouth_w": 0.6, "color": (70, 160, 200),
               "motion": "breathe", "fx": ("zzz",)},
    "love": {"scale": 1.1, "mouth": 0.8, "mouth_open": 0.25, "blush": 1.0,
             "color": (255, 120, 190), "eyes": ("heart", "heart"), "fx": ("hearts",)},
    # 返事を考えているあいだ：目を細めて、上の右・左をきょろきょろ見ながら、首をかしげてゆれる
    "think": {"open": 0.85, "scale": 0.95, "lid_bot": 0.2, "mouth": -0.1, "mouth_w": 0.6,
              "brow": 0.7, "brow_skew": 0.6, "color": (110, 200, 255), "motion": "ponder",
              "fx": ("dots",)},
    # ここから下はピクセルアート化で増やした表情
    "shy": {"open": 0.8, "scale": 0.92, "lid_bot": 0.35, "mouth": 0.2, "wavy": 1.0,
            "mouth_w": 0.6, "blush": 1.0, "color": (150, 210, 255), "motion": "fidget"},
    "laugh": {"mouth": 1.0, "mouth_open": 1.0, "mouth_w": 1.2, "blush": 0.7,
              "color": (110, 245, 220), "eyes": ("happy", "happy"), "motion": "giggle"},
    "excited": {"scale": 1.1, "mouth": 1.0, "mouth_open": 0.6, "color": (255, 215, 90),
                "eyes": ("star", "star"), "motion": "bounce", "fx": ("sparkle",)},
    "confused": {"skew": 0.22, "brow": 0.9, "brow_skew": 1.0, "mouth": -0.1, "wavy": 1.0,
                 "mouth_w": 0.8, "color": (170, 195, 255), "motion": "tilt",
                 "fx": ("question",)},
    "cry": {"mouth": -0.8, "mouth_open": 0.5, "mouth_w": 0.9,
            "color": (90, 150, 255), "eyes": ("squeeze", "squeeze"), "motion": "sob",
            "fx": ("tears",)},
    "wink": {"mouth": 0.9, "mouth_open": 0.2, "blush": 0.5, "color": (120, 235, 255),
             "eyes": ("normal", "happy"), "fx": ("sparkle",)},
    "dizzy": {"scale": 1.05, "mouth": -0.2, "wavy": 1.0, "color": (200, 160, 255),
              "eyes": ("spiral", "spiral"), "motion": "sway", "fx": ("orbit",)},
}

EMOTION_NAMES = list(EMOTIONS)
_NUM_KEYS = [k for k in DEFAULTS if k != "color"]
NO_BLINK = {"happy", "squeeze", "spiral"}  # もともと閉じている／回っている目はまばたきしない


def _target(name):
    t = dict(DEFAULTS)
    t.update(STYLE_DEFAULTS)
    t.update(EMOTIONS.get(name, {}))
    return t


def _lerp(a, b, t):
    return a + (b - a) * t


def _mix(c1, c2, t):
    return tuple(int(_lerp(a, b, t)) for a, b in zip(c1, c2))


# ---------------------------------------------------------------- ドット絵の部品
def _sprite(*rows):
    """文字で書いたドット絵を配列に。'#' 本体色 / 'w' 光 / 's' 影 / '.' なし。"""
    return np.array([list(r) for r in rows])


SPR_HEART = _sprite(
    ".###...###.",
    "#####.#####",
    "#ww########",
    "#w#########",
    "###########",
    ".#########.",
    "..#######..",
    "...#####...",
    "....###....",
    ".....#.....",
)
SPR_STAR = _sprite(
    ".....#.....",
    "....###....",
    "....#w#....",
    "...##w##...",
    "###########",
    ".#########.",
    "..#######..",
    "...#####...",
    "..###.###..",
    ".##.....##.",
    "##.......##",
)
SPR_HAPPY = _sprite(       # ^ の形の笑い目
    "...#####...",
    "..#######..",
    ".###...###.",
    "###.....###",
    "##.......##",
)
SPR_SQUEEZE = _sprite(     # > の形のぎゅっとつぶった目（右目は左右反転）
    "##.......",
    "####.....",
    ".#####...",
    "...######",
    ".#####...",
    "####.....",
    "##.......",
)
SPR_Z = _sprite("#####", "...#.", "..#..", ".#...", "#####")
SPR_Z_SMALL = _sprite("###", ".#.", "###")
SPR_HEART_SMALL = _sprite("##.##", "#####", ".###.", "..#..")
SPR_SPARKLE = _sprite("..#..", "..#..", "##w##", "..#..", "..#..")
SPR_SPARKLE_SMALL = _sprite(".#.", "#w#", ".#.")
SPR_DROP = _sprite(".#.", "###", "###", ".#.")
SPR_QUESTION = _sprite(".###.", "#...#", "....#", "..##.", "..#..", ".....", "..#..")
SPR_EXCLAIM = _sprite("##", "##", "##", "##", "..", "##")
SPR_VEIN = _sprite(".#..#.", "##..##", "......", "##..##", ".#..#.")
SPR_MARU = _sprite(        # 〇（はい）
    "...#####...",
    ".##.....##.",
    ".#.......#.",
    "#.........#",
    "#.........#",
    "#.........#",
    "#.........#",
    "#.........#",
    ".#.......#.",
    ".##.....##.",
    "...#####...",
)
SPR_BATSU = _sprite(       # ×（いいえ）
    "##.......##",
    "###.....###",
    ".###...###.",
    "..###.###..",
    "...#####...",
    "....###....",
    "...#####...",
    "..###.###..",
    ".###...###.",
    "###.....###",
    "##.......##",
)
YES_COLOR = (120, 255, 150)
NO_COLOR = (255, 110, 110)
SIGN_SECONDS = 2.6


def dot_grid(w, h, nx, ny):
    """ドットのすきまと角を描いた重ね絵（w×h ピクセルを nx×ny ドットに分ける）。"""
    key = (255, 0, 255)
    g = pygame.Surface((w, h))
    g.fill(key)
    g.set_colorkey(key)
    xs = [round(i * w / nx) for i in range(nx + 1)]
    ys = [round(j * h / ny) for j in range(ny + 1)]
    gap = max(1, round(w / nx * 0.14))
    for e in xs:
        g.fill(BG, (e - gap // 2, 0, gap, h))
    for e in ys:
        g.fill(BG, (0, e - gap // 2, w, gap))
    c = max(1, gap // 2 + (1 if w / nx >= 10 else 0))  # 角を少し落として丸いドットに
    for ex in xs:
        for ey in ys:
            g.fill(BG, (ex - gap // 2 - c, ey - gap // 2 - c, gap + 2 * c, gap + 2 * c))
    return g


class Face:
    def __init__(self, size):
        self.size = size
        self.emotion = "neutral"
        self.params = _target("neutral")
        self.t = 0.0
        self.gaze = [0.0, 0.0]
        self.idle_gaze = [0.0, 0.0]
        self.external_gaze = None
        self.talk = 0.0  # 話しているときの口の開き 0..1
        self.mood, self.mood_w = "neutral", 0.0  # 返事の余韻（ふだんの顔にうっすら混ぜる）
        # 話し始めや話の途中で、ときどき相手から目をそらす（考えながら話している感じ）
        self.avert_until = -1.0
        self.avert_goal = (0.0, 0.0)
        self.last_talk = -10.0        # 最後に口が動いていた時刻
        self.next_talk_avert = 0.0
        self.next_saccade = 0.0
        self.next_blink = random.uniform(1.0, 3.0)
        self.blink_start = -1.0
        self.entered = 0.0            # 今の感情になった時刻
        self.eyes = ("normal", "normal")  # いま描いている目の形
        self.swap_start = -1.0        # 目の形を入れかえるまばたきの開始時刻
        self.sign_kind, self.sign_start = None, -10.0  # はい・いいえ（うなずく / 首をふる、〇×）
        self.lo = pygame.Surface((N, N))
        self.grid = None
        self._ys, self._xs = np.mgrid[0:N, 0:N] + 0.5

    def set_emotion(self, name):
        if name in EMOTIONS and name != self.emotion:
            self.emotion = name
            self.entered = self.t

    def sign(self, kind):
        """kind='yes' でうなずいて〇、'no' で首をふって×を出す。"""
        if kind in ("yes", "no"):
            self.sign_kind, self.sign_start = kind, self.t

    def set_talk(self, level):
        self.talk = level

    def set_mood(self, name, weight):
        """ふだんの顔（neutral）に、name の表情を weight（0..1）だけ混ぜる。"""
        self.mood = name if name in EMOTIONS else "neutral"
        self.mood_w = max(0.0, min(1.0, weight))

    def look_at(self, target):
        """target: (x, y) を -1..1 で。None なら自由に視線を泳がせる。"""
        self.external_gaze = target

    # ---- 更新 ----
    def _update_avert(self):
        """話し始めに 1 秒ほど、長く話すときは数秒ごとに一瞬、目をそらす。
        Andrist ら (HRI 2014) の会話での目そらしをもとにした、簡単な決まり。"""
        speaking = self.talk > 0.05
        if speaking:
            if self.t - self.last_talk > 1.0:      # 新しく話し始めた
                self._avert(random.uniform(0.8, 1.5))
                self.next_talk_avert = self.t + random.uniform(4.0, 7.0)
            elif self.t >= self.next_talk_avert and self.t >= self.avert_until:
                self._avert(random.uniform(0.4, 0.7))
                self.next_talk_avert = self.t + random.uniform(4.0, 7.0)
            self.last_talk = self.t

    def _avert(self, seconds):
        side = random.choice((-1, 1))
        self.avert_goal = (side * random.uniform(0.45, 0.65), -random.uniform(0.2, 0.45))
        self.avert_until = self.t + seconds

    def update(self, dt):
        self.t += dt
        k = 1 - math.exp(-dt * 8)
        target = _target(self.emotion)
        if self.emotion == "neutral" and self.mood != "neutral" and self.mood_w > 0:
            m = _target(self.mood)  # 目の形や動きは neutral のまま、数値だけ寄せる
            for key in _NUM_KEYS:
                target[key] = _lerp(target[key], m[key], self.mood_w)
            target["color"] = tuple(_lerp(a, b, self.mood_w)
                                    for a, b in zip(target["color"], m["color"]))
        for key in _NUM_KEYS:
            self.params[key] = _lerp(self.params[key], target[key], k)
        self.params["color"] = tuple(
            _lerp(a, b, k) for a, b in zip(self.params["color"], target["color"]))
        for key in ("motion", "fx"):
            self.params[key] = target[key]

        # 目の形が変わるときは、まばたきの途中（閉じた瞬間）で入れかえる
        if target["eyes"] != self.eyes:
            if self.swap_start < 0:
                self.swap_start = self.t
            elif self.t - self.swap_start >= 0.09:
                self.eyes = target["eyes"]
        elif self.swap_start >= 0 and self.t - self.swap_start >= 0.2:
            self.swap_start = -1.0

        if self.t >= self.next_saccade:
            self.idle_gaze = [random.uniform(-0.6, 0.6), random.uniform(-0.4, 0.4)]
            self.next_saccade = self.t + random.uniform(1.2, 3.5)
        e = self.emotion
        if e == "sleepy":
            goal = (0.0, 0.3)
        elif e == "think":  # 「うーん」と上の右と左を交互に見る
            side = 0.55 if (self.t // 1.1) % 2 else -0.55
            goal = (side, -0.65 + 0.08 * math.sin(self.t * 2))
        elif e == "shy":
            goal = (-0.55 + 0.15 * math.sin(self.t * 1.3), 0.6)
        elif e == "dizzy":
            goal = (0.5 * math.cos(self.t * 5), 0.4 * math.sin(self.t * 5))
        elif e == "confused":
            goal = (0.5 if (self.t // 1.4) % 2 else -0.5, -0.2)
        elif self.external_gaze is not None:
            self._update_avert()
            goal = self.avert_goal if self.t < self.avert_until else self.external_gaze
        else:
            goal = self.idle_gaze
        kg = 1 - math.exp(-dt * 14)
        self.gaze[0] = _lerp(self.gaze[0], goal[0], kg)
        self.gaze[1] = _lerp(self.gaze[1], goal[1], kg)

        if self.t >= self.next_blink and self.emotion != "sleepy":
            self.blink_start = self.t
            # たまに二回まばたき
            self.next_blink = self.t + (0.35 if random.random() < 0.15
                                        else random.uniform(2.5, 6.0))

    def _blink_factor(self):
        """まばたき（すばやく閉じて、少しゆっくり開く）と、目の形の入れかえ。"""
        f = 1.0
        ph = self.t - self.blink_start
        if 0 <= ph < 0.06:
            f = 1 - ph / 0.06
        elif 0.06 <= ph < 0.16:
            f = (ph - 0.06) / 0.10
        if self.swap_start >= 0:
            ps = (self.t - self.swap_start) / 0.2
            f = min(f, abs(1 - 2 * min(ps, 1.0)))
        return f

    def _motion(self):
        """感情ごとの体の動き。(dx, dy, 目の大きさの倍率) をドット単位で返す。"""
        t, te = self.t, self.t - self.entered
        m = self.params["motion"]
        dx = dy = 0.0
        # 感情が変わった瞬間は、目がぽよんとはずむ
        pop = 1 + 0.16 * math.exp(-te * 9) * math.cos(te * 24)
        dy += 0.6 * math.sin(t * 1.6)  # 呼吸
        if m == "bounce":
            dy -= abs(math.sin(t * 5.5)) * 1.6
        elif m == "giggle":
            dy -= abs(math.sin(t * 11)) * 1.8
        elif m == "shake":
            if te < 0.6 or (t % 2.0) < 0.25:
                dx += 1 if int(t * 30) % 2 else -1
        elif m == "jump":
            dy -= 4 * math.exp(-te * 6) * abs(math.cos(te * 9))
        elif m == "droop":
            dy += min(te * 3, 1.2)
        elif m == "breathe":
            dy += 0.6 * math.sin(t * 1.0)
        elif m == "fidget":
            dx += 0.9 * math.sin(t * 2.2)
        elif m == "tilt":
            dx += 0.8 * math.sin(t * 1.2)
        elif m == "sob":
            dy += 0.8 * abs(math.sin(t * 9)) if (t % 1.6) < 0.8 else 0
        elif m == "ponder":  # 考え中: ゆっくり左右にかしげて、ときどき小さくうなる
            dx += 1.2 * math.sin(t * 1.4)
            dy -= 0.6 * abs(math.sin(t * 2.8))
        elif m == "sway":
            dx += 1.6 * math.sin(t * 2.6)
        ts = t - self.sign_start
        if self.sign_kind and 0 <= ts < 1.2:  # 2回うなずく / 2回首をふる
            env = math.sin(math.pi * ts / 1.2)
            if self.sign_kind == "yes":
                dy += 2.5 * abs(math.sin(ts * 2 * math.pi / 1.2)) * env
            else:
                dx += 2.5 * math.sin(ts * 2 * math.pi * 2 / 1.2) * env
        return round(dx), round(dy), pop

    # ---- 描画 ----
    def draw(self, surf, backdrop=None):
        """顔を surf いっぱいに描く。backdrop は render() と同じ。"""
        S = surf.get_width()
        self.render(backdrop)
        pygame.surfarray.blit_array(self.lo, self.img.swapaxes(0, 1))
        pygame.transform.scale(self.lo, (S, S), surf)
        surf.blit(self._grid(S), (0, 0))

    def render(self, backdrop=None):
        """self.img（48×48×3 のドット）に顔を描く。

        backdrop（48×48×3 の明るさ）を渡すと、消えているドットに足して模様を透かす。
        """
        p = self.params
        color = tuple(int(c) for c in p["color"])
        self.c_main = color
        self.c_hi = _mix(color, (255, 255, 255), 0.7)
        self.c_shade = _mix(color, OFF, 0.45)
        self.c_dark = _mix(color, BG, 0.8)

        if backdrop is None:
            self.img = np.empty((N, N, 3), np.uint8)
            self.img[:] = OFF
        else:
            self.img = np.clip(backdrop + OFF, 0, 255).astype(np.uint8)
        dx, dy, pop = self._motion()
        blink = self._blink_factor()

        self._draw_blush(dx, dy)
        for i, side in enumerate((-1, 1)):  # -1 = 画面左の目
            self._draw_eye(side, self.eyes[i], dx, dy, pop, blink)
        self._draw_brows(dx, dy)
        self._draw_mouth(dx, dy)
        self._draw_fx(dx, dy)
        return self.img

    def _grid(self, S):
        if self.grid is None or self.grid.get_width() != S:
            self.grid = dot_grid(S, S, N, N)
        return self.grid

    # ---- 部品の描きこみ（座標はすべてドット単位） ----
    def _put(self, x, y, color):
        x, y = int(x), int(y)
        if 0 <= x < N and 0 <= y < N:
            self.img[y, x] = color

    def _stamp(self, spr, cx, cy, colors, sx=1.0, sy=1.0, flip=False):
        """ドット絵を (cx, cy) を中心に置く。sx, sy で拡大縮小（最近傍）。"""
        if flip:
            spr = spr[:, ::-1]
        h, w = spr.shape
        tw, th = max(1, round(w * sx)), max(1, round(h * sy))
        x0, y0 = round(cx - tw / 2), round(cy - th / 2)
        for j in range(th):
            row = spr[min(h - 1, int(j * h / th))]
            for i in range(tw):
                ch = row[min(w - 1, int(i * w / tw))]
                if ch in colors:
                    self._put(x0 + i, y0 + j, colors[ch])

    def _draw_eye(self, side, shape, dx, dy, pop, blink):
        p = self.params
        gx, gy = self.gaze
        cx = 24 + side * 10 + round(gx * 3) + dx
        cy = 20 + round(gy * 2) + dy
        size = p["scale"] * pop * (1 + p["skew"] * -side) * (1 + 0.08 * gx * side)
        cols = {"#": self.c_main, "w": self.c_hi, "s": self.c_shade}

        if shape != "normal":
            squash = blink if shape not in NO_BLINK else 1.0
            if shape == "spiral":
                self._draw_spiral(cx, cy, size)
                return
            spr = {"heart": SPR_HEART, "star": SPR_STAR, "happy": SPR_HAPPY,
                   "squeeze": SPR_SQUEEZE}[shape]
            s = size
            if shape == "heart":   # どきどき
                s *= 1 + 0.12 * max(0.0, math.sin(self.t * 7)) ** 8
            if shape == "star":    # きらきら
                s *= 1 + 0.06 * math.sin(self.t * 4)
            if shape in ("happy", "squeeze"):
                s = 1.0 + (size - 1) * 0.5
                cy += 1
            self._stamp(spr, cx, cy, cols, s, s * max(squash, 0.12), flip=(side > 0))
            return

        b_full = 7 * size
        b = b_full * p["open"] * blink
        a = 5.5 * size
        if b < 0.9:  # 閉じた目は横一本線（少し下に弧）
            for x in range(round(cx - a + 1), round(cx + a - 1)):
                u = (x + 0.5 - cx) / a
                self._put(x, cy + (1 if abs(u) < 0.6 else 0), self.c_main)
            return
        X, Y = self._xs, self._ys
        ddx, ddy = (X - cx) / a, (Y - cy) / b
        mask = np.abs(ddx) ** 2.6 + np.abs(ddy) ** 2.6 <= 1
        top = cy - b
        h = 2 * b
        u_in = ddx * -side  # 内側ほど +1
        lt, tilt = p["lid_top"], p["tilt"]
        if lt > 0.01 or abs(tilt) > 0.01:
            mask &= Y >= top + h * (lt + tilt * 0.5 * u_in)
        lb = p["lid_bot"]
        if lb > 0.01:
            ey = cy + b - lb * h + 0.65 * h
            mask &= ((X - cx) / (0.8 * 2 * a)) ** 2 + ((Y - ey) / (0.65 * h)) ** 2 > 1
        self.img[mask] = self.c_main
        self.img[mask & (Y > cy + b * 0.5)] = self.c_shade
        if b >= 3.5:  # 光（四角いハイライト）
            hx, hy = int(cx - a * 0.5), int(cy - b * 0.55)
            for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 2)):
                if 0 <= hy + oy < N and 0 <= hx + ox < N and mask[hy + oy, hx + ox]:
                    self.img[hy + oy, hx + ox] = self.c_hi

    def _draw_spiral(self, cx, cy, size):
        rot = self.t * 6
        turns = 2.6
        r_max = 5.5 * size
        th = 0.0
        while th < turns * 2 * math.pi:
            r = 0.6 + r_max * th / (turns * 2 * math.pi)
            self._put(math.floor(cx + r * math.cos(th + rot)),
                      math.floor(cy + r * math.sin(th + rot)), self.c_main)
            th += 0.12

    def _draw_brows(self, dx, dy):
        p = self.params
        if p["brow"] < 0.05:
            return
        col = _mix(OFF, self.c_main, min(1.0, p["brow"]))
        gx, gy = self.gaze
        for side in (-1, 1):
            cx = 24 + side * 10 + round(gx * 3) + dx
            base = 20 + round(gy * 2) + dy - 7 * p["scale"] - 3 - p["brow_y"] * 2
            base -= p["brow_skew"] * 1.5 * -side
            for i in range(-3, 4):
                u_in = i / 3 * -side
                y = base + p["brow_tilt"] * 1.5 * u_in
                self._put(cx + i, round(y), col)

    def _draw_mouth(self, dx, dy):
        p = self.params
        gx, gy = self.gaze
        mx = 24 + round(gx * 1.5) + dx
        my = 35 + round(gy) + dy
        if p["round"] > 0.5:   # o の口
            r = p["round"]
            X, Y = self._xs, self._ys
            rx, ry = 2.4 * r + 0.3, 3.0 * r + 0.3
            d = ((X - mx) / rx) ** 2 + ((Y - my) / ry) ** 2
            self.img[d <= 1] = self.c_main
            self.img[d <= 0.35] = self.c_dark
            return
        hw = 5.5 * p["mouth_w"]
        curve = p["mouth"] * 3.2
        opening = (p["mouth_open"] + self.talk * 0.9) * 4.5
        prev = None
        for x in range(round(mx - hw), round(mx + hw)):
            u = (x + 0.5 - mx) / hw
            bend = max(0.0, 1 - u * u)  # 誤差で少しでも負になると 0.6 乗が複素数になって落ちる
            y_up = my + curve * bend + p["wavy"] * 0.9 * math.sin(u * math.pi * 2)
            y_lo = y_up + opening * bend ** 0.6
            yu, yl = round(y_up), round(y_lo)
            for y in range(yu + 1, yl):  # 口の中（暗く、下のほうに舌）
                self._put(x, y, PINK if (y == yl - 1 and yl - yu >= 4 and abs(u) < 0.5)
                          else self.c_dark)
            lo, hi = (yu, yu) if prev is None else (min(prev, yu), max(prev, yu))
            for y in range(lo, hi + 1):  # 線がとぎれないようにつなぐ
                self._put(x, y, self.c_main)
            if yl > yu:
                self._put(x, yl, self.c_main)
            prev = yu

    def _draw_blush(self, dx, dy):
        b = self.params["blush"]
        if b < 0.05:
            return
        col = _mix(OFF, PINK, min(1.0, b))
        for side in (-1, 1):
            x0 = 24 + side * 13 - 3 + dx
            y0 = 28 + dy
            for i in range(3):  # ／／／ の斜線
                self._put(x0 + 2 * i + 1, y0, col)
                self._put(x0 + 2 * i, y0 + 1, col)

    def _draw_sign(self):
        ts = self.t - self.sign_start
        if not self.sign_kind or not 0 <= ts < SIGN_SECONDS:
            return
        spr, col = (SPR_MARU, YES_COLOR) if self.sign_kind == "yes" else (SPR_BATSU, NO_COLOR)
        pop = 1 + 0.35 * math.exp(-ts * 8)
        fade = min(1.0, (SIGN_SECONDS - ts) / 0.4)
        self._stamp(spr, 41, 40, {"#": _mix(OFF, col, fade)}, pop, pop)  # 右下（字幕・目・口にかからない）

    def _draw_fx(self, dx, dy):
        self._draw_sign()
        t, te = self.t, self.t - self.entered
        fade_in = min(1.0, te / 0.3)
        for fx in self.params["fx"]:
            if fx == "zzz":
                for i, spr in enumerate((SPR_Z_SMALL, SPR_Z, SPR_Z)):
                    ph = (t * 0.35 + i / 3) % 1.0
                    if 0.1 < ph < 0.9:
                        self._stamp(spr, 37 + ph * 6, 15 - ph * 10,
                                    {"#": _mix(OFF, self.c_main, math.sin(ph * math.pi))},
                                    0.8 + ph * 0.5, 0.8 + ph * 0.5)
            elif fx == "hearts":
                for i in range(3):
                    ph = (t * 0.45 + i / 3) % 1.0
                    x = (8, 40, 24)[i] + math.sin(ph * 9 + i) * 1.5
                    self._stamp(SPR_HEART_SMALL, x, 44 - ph * 40,
                                {"#": _mix(OFF, PINK, math.sin(ph * math.pi) * fade_in)})
            elif fx == "sparkle":
                for i, (x, y) in enumerate(((6, 8), (42, 6), (43, 30), (5, 33))):
                    ph = (t * 1.3 + i * 0.37) % 1.0
                    spr = SPR_SPARKLE if ph < 0.35 else SPR_SPARKLE_SMALL
                    if ph < 0.7:
                        self._stamp(spr, x, y, {"#": GOLD, "w": (255, 255, 255)})
            elif fx == "tears":
                for side in (-1, 1):
                    x = 24 + side * 10 + dx
                    for k in range(4):
                        ph = (t * 1.6 + k / 4 + (0.12 if side > 0 else 0)) % 1.0
                        self._stamp(SPR_DROP, x + side * (2 + ph * 3), 25 + dy + ph * 20,
                                    {"#": TEAR})
            elif fx == "tear":   # 片目からときどき一粒
                ph = (te * 0.5) % 1.0
                if te > 1.0 and ph < 0.8:
                    self._stamp(SPR_DROP, 24 - 10 - 4 + dx, 26 + dy + ph * 12, {"#": TEAR})
            elif fx == "vein":
                c = RED if (t * 3) % 1 < 0.6 else _mix(RED, OFF, 0.5)
                self._stamp(SPR_VEIN, 42 + dx, 7 + dy, {"#": c})
            elif fx == "question":
                self._stamp(SPR_QUESTION, 42 + dx, 8 + round(math.sin(t * 3)) + dy,
                            {"#": self.c_main})
            elif fx == "exclaim":
                if te < 1.2 and (te > 0.15):
                    self._stamp(SPR_EXCLAIM, 42, 7 - min(te * 20, 2), {"#": GOLD})
            elif fx == "dots":   # 考え中の …
                n = int(t * 2.5) % 4
                for i in range(n):
                    self._stamp(np.array([["#", "#"], ["#", "#"]]), 39 + i * 3,
                                4 + dy, {"#": self.c_main})
            elif fx == "orbit":   # 頭の上を回るお星さま
                for i in range(3):
                    a = t * 3 + i * 2 * math.pi / 3
                    self._stamp(SPR_SPARKLE_SMALL, 24 + math.cos(a) * 14 + dx,
                                6 + math.sin(a) * 2.5 + dy, {"#": GOLD, "w": (255, 255, 255)})
