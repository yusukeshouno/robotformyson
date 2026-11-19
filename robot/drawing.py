"""イモのお絵かき。ときどき（または「えをかいて」と頼まれたら）ドット絵を描いて見せる。

- 絵の中身は Mac の Ollama が考える。「まる・しかく・線・三角」などの図形のリスト（JSON）で
  返してもらい、Pi はそれを 40×40 ドットに塗るだけ（1枚に1回、とても軽い）。
- 家族の顔: カメラの画像を Mac の Ollama に見せて「髪の長さ・色・めがね」などの特徴だけを
  聞き、こちらで決まった描き方の似顔絵にする。写真は保存しないし、家の外にも出さない。
- 描くところも見せる: 図形を1つずつ、クレヨンでぬるように左右に往復しながら塗っていく。
  画面の下からイモの目がのぞいて、ペンの先を目で追う。できあがったら「○○を かいたよ！」。

単体で試す（Ollama がなくても、用意した絵で動きが見られる）:

    python3 -m robot.drawing --demo            # ウィンドウで描くところを見る
    python3 -m robot.drawing --ollama-url http://Mac-mini.local:11434 --subject ねこ
"""
import argparse
import base64
import json
import math
import os
import queue
import random
import threading
import time
import urllib.request

import numpy as np

from .conversation import _THINK_RE, contains_any

C = 40                      # 絵のドットの数（一辺）
N = 48                      # 画面のドットの数（一辺）
X0, Y0 = 4, 2               # 画面の中の絵の左上
OFF = (17, 20, 33)
FRAME = (150, 100, 60)      # 額ぶち
EYE = (90, 220, 255)        # のぞいているイモの目

PALETTE = {
    "black": (30, 30, 40), "white": (245, 245, 245), "gray": (140, 140, 150),
    "red": (235, 60, 60), "orange": (255, 150, 50), "yellow": (255, 220, 60),
    "lightgreen": (150, 230, 100), "green": (50, 170, 80), "skyblue": (130, 200, 255),
    "blue": (50, 100, 230), "navy": (30, 40, 110), "purple": (150, 90, 210),
    "pink": (255, 150, 190), "brown": (140, 85, 45), "beige": (240, 215, 170),
    "skin": (250, 205, 170), "skin_medium": (220, 165, 120), "skin_tan": (185, 130, 90),
    "skin_dark": (120, 80, 55), "gold": (240, 190, 60),
}
SHAPES = ("rect", "circle", "ellipse", "ring", "line", "triangle", "arc", "dot")
MAX_SHAPES = 40

# Ollama に、この形の JSON だけを返してもらう（structured outputs）
_NUM = {"type": "number"}
SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "background": {"type": "string", "enum": list(PALETTE)},
        "shapes": {"type": "array", "items": {
            "type": "object",
            "properties": {"type": {"type": "string", "enum": list(SHAPES)},
                           "color": {"type": "string", "enum": list(PALETTE)},
                           "x": _NUM, "y": _NUM, "w": _NUM, "h": _NUM, "r": _NUM,
                           "rx": _NUM, "ry": _NUM, "x2": _NUM, "y2": _NUM,
                           "x3": _NUM, "y3": _NUM, "width": _NUM},
            "required": ["type", "color"]}},
    },
    "required": ["title", "background", "shapes"],
}

DRAW_PROMPT = """あなたは小さな子どもと遊ぶロボットです。40×40 マスのドット絵を、図形を重ねて描きます。
座標は左上が (0,0)、右下が (39,39)。x が横、y が縦（下ほど大きい）。
図形は後に書いたものほど上に重なります。まず大きな形（地面・からだ）、最後に細かい所（目・模様）。
使える図形:
- rect: x,y（左上）, w,h        - circle: x,y（中心）, r       - ellipse: x,y（中心）, rx,ry
- ring: x,y,r（丸の線だけ）     - line: x,y → x2,y2, width     - triangle: x,y / x2,y2 / x3,y3
- arc: x,y,rx,ry（下半分の弧。にっこりした口など）  - dot: x,y（1マス）
色は {colors} から選びます。図形は 8〜{max} 個。子どもがひと目で何か分かる、かわいくて大きな絵にしてください。
title は ひらがなで短く（例「ねこ」「ロケット」）。
例（おひさまと いえ）:
{example}

描くもの: {subject}"""

EXAMPLE = {"title": "おひさまと いえ", "background": "skyblue", "shapes": [
    {"type": "rect", "color": "lightgreen", "x": 0, "y": 31, "w": 40, "h": 9},
    {"type": "circle", "color": "yellow", "x": 32, "y": 7, "r": 5},
    {"type": "rect", "color": "beige", "x": 8, "y": 18, "w": 18, "h": 14},
    {"type": "triangle", "color": "red", "x": 5, "y": 19, "x2": 17, "y2": 8, "x3": 29, "y3": 19},
    {"type": "rect", "color": "brown", "x": 14, "y": 24, "w": 6, "h": 8},
    {"type": "rect", "color": "skyblue", "x": 21, "y": 21, "w": 3, "h": 3},
]}

# 自分から描くときの「なにを描こうかな」
SUBJECTS = ("ねこ", "いぬ", "きょうりゅう", "ロケット", "おはな", "くるま", "でんしゃ", "くじら",
            "りんご", "にじ", "おばけ", "ケーキ", "うさぎ", "ペンギン", "おひさま", "さかな",
            "ちょうちょ", "アイスクリーム", "ふね", "かたつむり", "ロボット", "おしろ", "きのこ")

FEATURES_PROMPT = """この画像の、いちばん大きく写っている人の見た目を、似顔絵を描くために教えてください。
分からないところは、いちばん近いものを選んでください。"""
FEATURES_SCHEMA = {
    "type": "object",
    "properties": {
        "person": {"type": "boolean"},
        "hair": {"type": "string", "enum": ["short", "long", "bob", "ponytail", "twintails",
                                            "bun", "curly", "spiky", "bald"]},
        "hair_color": {"type": "string", "enum": ["black", "brown", "blonde", "gray", "white",
                                                  "red"]},
        "skin": {"type": "string", "enum": ["light", "medium", "tan", "dark"]},
        "glasses": {"type": "boolean"},
        "beard": {"type": "boolean"},
        "hat": {"type": "string", "enum": ["none", "cap", "beanie"]},
        "smile": {"type": "boolean"},
        "shirt": {"type": "string", "enum": ["red", "orange", "yellow", "green", "blue", "navy",
                                             "purple", "pink", "white", "gray", "black"]},
    },
    "required": ["person", "hair", "hair_color", "skin", "glasses", "smile", "shirt"],
}
HAIR_COLOR = {"black": "black", "brown": "brown", "blonde": "gold", "gray": "gray",
              "white": "white", "red": "orange"}
SKIN = {"light": "skin", "medium": "skin_medium", "tan": "skin_tan", "dark": "skin_dark"}

# 「えを かいて」「ねこ かいて」「パパの かお かいて」
DRAW_WORDS = ("かいて", "描いて", "えかいて", "おえかき", "お絵かき", "お絵描き", "かける？",
              "描ける？", "えがいて")
NOT_DRAW = ("書いて", "かいてある", "かいてる", "字", "じを", "てがみ", "手紙")
FACE_WORDS = ("かお", "顔", "にがおえ", "似顔絵", "ぼくを", "わたしを", "おれを", "ぼくのこと",
              "わたしのこと")


def wants_drawing(text, names=()):
    """頼まれた絵: None / ("portrait", None) / ("picture", 頼まれた文)。"""
    if not contains_any(text, DRAW_WORDS) or contains_any(text, NOT_DRAW):
        return None
    if contains_any(text, FACE_WORDS) or (names and contains_any(text, names)):
        return "portrait", None
    return "picture", text


# ---------------------------------------------------------------- 図形 → ドット
def _num(s, k, default=0.0):
    try:
        return float(s.get(k, default))
    except (TypeError, ValueError):
        return default


def shape_mask(s):
    """図形1つ → (40×40 の True/False, 塗る順番に並べた (y, x) の配列)。"""
    Y, X = np.mgrid[0:C, 0:C].astype(np.float32)
    t = s.get("type")
    x, y = _num(s, "x"), _num(s, "y")
    line_t = None
    if t == "rect":
        w, h = max(1.0, _num(s, "w", 1)), max(1.0, _num(s, "h", 1))
        m = (X >= round(x)) & (X < round(x + w)) & (Y >= round(y)) & (Y < round(y + h))
    elif t == "circle":
        r = max(0.5, _num(s, "r", 1))
        m = (X - x) ** 2 + (Y - y) ** 2 <= r * r + r * 0.6
    elif t == "ellipse":
        rx, ry = max(0.5, _num(s, "rx", 1)), max(0.5, _num(s, "ry", 1))
        m = ((X - x) / (rx + 0.3)) ** 2 + ((Y - y) / (ry + 0.3)) ** 2 <= 1
    elif t in ("ring", "arc"):
        rx = max(0.5, _num(s, "rx", _num(s, "r", 1)))
        ry = max(0.5, _num(s, "ry", _num(s, "r", 1)))
        d = np.sqrt(((X - x) / rx) ** 2 + ((Y - y) / ry) ** 2)
        m = np.abs(d - 1) * min(rx, ry) < 0.62
        if t == "arc":
            m &= Y >= y
        line_t = np.arctan2(Y - y, -(X - x))   # 弧にそって順に
    elif t == "line":
        x2, y2 = _num(s, "x2", x), _num(s, "y2", y)
        w = max(1.0, _num(s, "width", 1))
        dx, dy = x2 - x, y2 - y
        L2 = dx * dx + dy * dy
        u = np.clip(((X - x) * dx + (Y - y) * dy) / L2, 0, 1) if L2 else np.zeros_like(X)
        dist = np.hypot(X - (x + u * dx), Y - (y + u * dy))
        m = dist <= w / 2 + 0.2
        line_t = u
    elif t == "triangle":
        pts = [(x, y), (_num(s, "x2"), _num(s, "y2")), (_num(s, "x3"), _num(s, "y3"))]

        def side(a, b):
            return (X - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (Y - b[1])
        d1, d2, d3 = side(pts[0], pts[1]), side(pts[1], pts[2]), side(pts[2], pts[0])
        neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        m = ~(neg & pos)
    elif t == "dot":
        m = (np.round(X) == round(x)) & (np.round(Y) == round(y))
    else:
        return None, None
    ys, xs = np.nonzero(m)
    if not len(ys):
        return None, None
    if line_t is not None:   # 線は端から順に
        order = np.argsort(line_t[ys, xs], kind="stable")
    else:                    # 面はクレヨンのように、上から左右に往復しながら
        key = ys * 100 + np.where(ys % 2 == 0, xs, 99 - xs)
        order = np.argsort(key, kind="stable")
    return m, np.stack([ys[order], xs[order]], 1)


class Drawing:
    """できあがった絵。img は (40, 40, 3)、strokes は (色, 塗る順の点) のリスト。"""

    def __init__(self, data, kind="picture"):
        self.kind = kind
        self.title = str(data.get("title") or "え")[:16]
        self.data = data
        bg = PALETTE.get(data.get("background"), PALETTE["white"])
        self.bg = bg
        self.img = np.empty((C, C, 3), np.uint8)
        self.img[:] = bg
        self.strokes = []
        for s in (data.get("shapes") or [])[:MAX_SHAPES]:
            if not isinstance(s, dict):
                continue
            m, pts = shape_mask(s)
            if m is None:
                continue
            col = PALETTE.get(s.get("color"), PALETTE["black"])
            self.img[m] = col
            self.strokes.append((col, pts))

    @property
    def ok(self):
        return len(self.strokes) >= 2

    def save(self, folder):
        """絵の図形（JSON）と、できあがり（PNG）を残す。写真は残さない。"""
        os.makedirs(folder, exist_ok=True)
        stem = os.path.join(folder, time.strftime("%Y%m%d-%H%M%S"))
        with open(stem + ".json", "w", encoding="utf-8") as f:
            json.dump(dict(self.data, kind=self.kind), f, ensure_ascii=False)
        try:
            from PIL import Image
            Image.fromarray(self.img).resize((C * 8, C * 8), Image.NEAREST).save(stem + ".png")
        except ImportError:
            pass
        return stem


# ---------------------------------------------------------------- 似顔絵
def portrait_shapes(f, seed=None):
    """顔の特徴 → 似顔絵の図形。決まった描き方なので、いつも同じくらいかわいく描ける。"""
    rnd = random.Random(seed)
    hair = HAIR_COLOR.get(f.get("hair_color"), "black")
    skin = SKIN.get(f.get("skin"), "skin")
    style = f.get("hair", "short")
    shirt = f.get("shirt") if f.get("shirt") in PALETTE else "blue"
    S = []

    def add(t, color, **kw):
        S.append(dict(type=t, color=color, **kw))

    if style in ("long", "bob"):          # うしろの髪
        add("rect", hair, x=8, y=12, w=24, h=20 if style == "long" else 12)
        add("ellipse", hair, x=20, y=13, rx=12, ry=8)
    add("ellipse", shirt, x=20, y=40, rx=15, ry=8)                 # からだ
    add("rect", skin, x=17, y=27, w=6, h=6)                        # くび
    add("ellipse", skin, x=20, y=19, rx=10, ry=11)                 # かお
    add("ellipse", skin, x=9, y=20, rx=2, ry=3)                    # みみ
    add("ellipse", skin, x=31, y=20, rx=2, ry=3)
    if style == "ponytail":
        add("ellipse", hair, x=33, y=14, rx=3, ry=6)
    elif style == "twintails":
        add("ellipse", hair, x=7, y=18, rx=3, ry=6)
        add("ellipse", hair, x=33, y=18, rx=3, ry=6)
    elif style == "bun":
        add("circle", hair, x=20, y=5, r=4)
    if style == "curly":
        for cx, cy in ((11, 13), (15, 9), (20, 8), (25, 9), (29, 13), (13, 11), (27, 11)):
            add("circle", hair, x=cx, y=cy, r=3.2)
    elif style == "spiky":
        add("ellipse", hair, x=20, y=12, rx=11, ry=5)
        for i in range(5):
            bx = 9 + i * 4.5
            add("triangle", hair, x=bx, y=12, x2=bx + 2.5, y2=4 + (i % 2) * 2, x3=bx + 5, y3=12)
    elif style != "bald":                                          # 前髪
        add("ellipse", hair, x=20, y=11, rx=11, ry=5)
        add("triangle", hair, x=12, y=12, x2=21, y2=12, x3=14, y3=17)
    hat = f.get("hat", "none")
    hat_col = rnd.choice(("red", "blue", "green", "orange", "purple"))
    if hat == "cap":
        add("ellipse", hat_col, x=20, y=9, rx=11, ry=5)
        add("rect", hat_col, x=20, y=11, w=13, h=2)
    elif hat == "beanie":
        add("ellipse", hat_col, x=20, y=10, rx=11, ry=6)
        add("rect", "white", x=9, y=12, w=22, h=2)
    if f.get("beard"):
        add("ellipse", hair, x=20, y=27, rx=8, ry=4)
    for ex in (16, 24):                                            # 目（光つき）
        add("rect", "black", x=ex - 1, y=19, w=2, h=3)
        add("dot", "white", x=ex - 1, y=19)
    if f.get("glasses"):
        add("ring", "navy", x=15.5, y=20, r=3.5)
        add("ring", "navy", x=24.5, y=20, r=3.5)
        add("line", "navy", x=19, y=20, x2=21, y2=20)
    add("circle", "pink", x=12.5, y=24, r=1.6)                    # ほっぺ
    add("circle", "pink", x=27.5, y=24, r=1.6)
    if f.get("smile", True):
        add("arc", "red", x=20, y=24, rx=3.5, ry=2.5)
    else:
        add("line", "red", x=18, y=26, x2=22, y2=26)
    return S


# ---------------------------------------------------------------- AI に描いてもらう
class Artist:
    """Mac の Ollama に絵を考えてもらう。裏で動くので、顔の動きは止まらない。"""

    def __init__(self, url="http://localhost:11434", model="gemma4:12b", timeout=120):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self.ready = queue.Queue()     # (Drawing, 頼まれたか) ができたら入る
        self.busy = False

    def _chat(self, prompt, schema, image=None, temperature=0.8, num_predict=1500):
        msg = {"role": "user", "content": prompt}
        if image:
            msg["images"] = [base64.standard_b64encode(image).decode()]
        body = {"model": self.model, "messages": [msg], "stream": False, "format": schema,
                "think": False, "keep_alive": "30m",
                "options": {"temperature": temperature, "num_predict": num_predict,
                            "num_ctx": 4096}}
        req = urllib.request.Request(self.url + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            text = json.loads(r.read().decode("utf-8")).get("message", {}).get("content", "")
        return json.loads(_THINK_RE.sub("", text))

    def picture(self, subject):
        prompt = DRAW_PROMPT.format(colors=" ".join(PALETTE), max=MAX_SHAPES,
                                    example=json.dumps(EXAMPLE, ensure_ascii=False),
                                    subject=subject)
        return Drawing(self._chat(prompt, SCHEMA), "picture")

    def portrait(self, image_jpeg, call=None):
        f = self._chat(FEATURES_PROMPT, FEATURES_SCHEMA, image=image_jpeg, temperature=0.1,
                       num_predict=200)
        if not f.get("person", True):
            return None
        print(f"[お絵かき] 似顔絵の特徴: {f}")
        title = f"{call} の かお" if call else "きみの かお"
        return Drawing({"title": title, "background": random.choice(  # はだの色と分かれる色
            ("skyblue", "lightgreen", "yellow", "purple")),
            "shapes": portrait_shapes(f)}, "portrait")

    def request(self, kind="picture", subject=None, image=None, call=None, asked=False):
        """裏で描きはじめる。描いている途中なら False。"""
        if self.busy:
            return False
        self.busy = True
        threading.Thread(target=self._work, args=(kind, subject, image, call, asked),
                         daemon=True).start()
        return True

    def _work(self, kind, subject, image, call, asked):
        t0 = time.monotonic()
        try:
            if kind == "portrait" and image:
                d = self.portrait(image, call)
            else:
                d = self.picture(subject or random.choice(SUBJECTS))
            if d is not None and d.ok:
                print(f"[お絵かき] 「{d.title}」 {len(d.strokes)}こ の形（{time.monotonic() - t0:.0f}秒）")
                self.ready.put((d, asked))
            else:
                print("[お絵かき] うまく描けなかったので、やめました")
                if asked:
                    self.ready.put((None, asked))
        except Exception as e:  # つながらない・JSON がこわれていた
            print(f"[お絵かき] 描けませんでした（{e}）")
            if asked:
                self.ready.put((None, asked))
        finally:
            self.busy = False

    def poll(self):
        try:
            return self.ready.get_nowait()
        except queue.Empty:
            return None


class Doodler:
    """「たまに」自分から描く係。人がいて、しばらく話していないときだけ。

    家族（顔をおぼえた人）が見えているときは、ときどきその人の似顔絵にする。
    """

    def __init__(self, artist, every_min=30.0, portrait_chance=0.4, clock=time.monotonic):
        self.artist, self.every = artist, every_min * 60
        self.portrait_chance = portrait_chance
        self.clock = clock
        self.next_at = clock() + min(self.every, 5 * 60) if self.every > 0 else float("inf")

    def tick(self, present, idle_secs, sleeping, who=None, get_image=None, topic=None):
        """自分から描きはじめたら True。who は顔をおぼえた人の呼び名（いなければ None）。

        get_image は () -> JPEG。似顔絵にすると決めたときだけ呼ぶ（毎コマは撮らない）。"""
        now = self.clock()
        if now < self.next_at or not present or sleeping or idle_secs < 60 or self.artist.busy:
            return False
        self.next_at = now + self.every * random.uniform(0.7, 1.3)
        if who and get_image and random.random() < self.portrait_chance:
            image = get_image()
            if image:
                return self.artist.request("portrait", image=image, call=who)
        subject = random.choice(SUBJECTS)
        if topic:  # さっき話したことを描くと、子どもがよろこぶ
            subject = (f"さいきん子どもと話したこと「{topic}」に出てきたもの。"
                       f"なければ {subject}")
        return self.artist.request("picture", subject=subject)


# ---------------------------------------------------------------- 描くところを見せる
class DrawingShow:
    """48×48 ドットの画面に、絵を描いていくようすを出す。update(dt) で1コマ進む。

    段階: intro（白い紙が出る）→ draw（図形を1つずつ塗る）→ done（できあがり）→ outro。
    """

    INTRO, DONE, OUTRO = 1.0, 9.0, 0.6
    MAX_DRAW = 14.0      # 描く時間の上限（秒）

    def __init__(self, drawing, rows=N, top=0):
        self.d = drawing
        self.rows, self.top = rows, top
        self.t = 0.0
        self.paper = np.empty((C, C, 3), np.uint8)
        self.paper[:] = drawing.bg
        # 図形ごとの時間（大きい図形ほど少し長く）
        secs = [min(1.4, max(0.25, 0.09 * math.sqrt(len(p)))) for _, p in drawing.strokes]
        k = min(1.0, self.MAX_DRAW / max(sum(secs), 1e-6))
        self.secs = [s * k for s in secs]
        self.draw_time = sum(self.secs)
        self.stroke, self.done_px = 0, 0
        self.pen = None
        self.phase = "intro"
        self.ended = False
        self.finish_early = False

    @property
    def finished_drawing(self):
        return self.phase in ("done", "outro")

    def stop(self):
        """だれかが話しかけた: 顔にもどる。"""
        if self.phase != "outro":
            self.phase, self.t = "outro", 0.0

    def _advance(self, dt):
        while dt > 0 and self.stroke < len(self.d.strokes):
            col, pts = self.d.strokes[self.stroke]
            rate = len(pts) / self.secs[self.stroke]
            n = min(len(pts) - self.done_px, max(1, int(math.ceil(rate * dt))))
            seg = pts[self.done_px:self.done_px + n]
            self.paper[seg[:, 0], seg[:, 1]] = col
            self.pen = tuple(seg[-1])
            self.done_px += n
            dt -= n / rate
            if self.done_px >= len(pts):
                self.stroke, self.done_px = self.stroke + 1, 0

    def update(self, dt):
        """1コマ進めて、画面の絵 (rows, 48, 3) を返す。"""
        self.t += dt
        if self.phase == "intro" and self.t >= self.INTRO:
            self.phase, self.t = "draw", 0.0
        elif self.phase == "draw":
            self._advance(dt)
            if self.stroke >= len(self.d.strokes):
                self.phase, self.t, self.pen = "done", 0.0, None
                self.paper[:] = self.d.img
        elif self.phase == "done" and self.t >= self.DONE:
            self.phase, self.t = "outro", 0.0
        elif self.phase == "outro" and self.t >= self.OUTRO:
            self.ended = True
        return self.render()

    def render(self):
        out = np.empty((self.rows, N, 3), np.float32)
        out[:] = OFF
        g = out[self.top:self.top + N]
        fade = 1.0
        if self.phase == "intro":
            fade = min(1.0, self.t / 0.4)
        elif self.phase == "outro":
            fade = max(0.0, 1 - self.t / self.OUTRO)
        off = np.array(OFF, np.float32)

        def mix(c, a=1.0):
            return off + (np.asarray(c, np.float32) - off) * a * fade

        g[Y0 - 1:Y0 + C + 1, X0 - 1:X0 + C + 1] = mix(FRAME, 0.9)          # 額ぶち
        g[Y0:Y0 + C, X0:X0 + C] = mix(self.paper)
        if self.pen is not None and int(self.t * 8) % 2 == 0:             # ペンの先がちかちか
            py, px = self.pen
            g[Y0 + py, X0 + px] = mix((255, 255, 255))
        if self.phase == "done":                                           # できた！のきらきら
            for i, (x, y) in enumerate(((1, 3), (46, 8), (2, 30), (45, 36), (24, 0))):
                if ((self.t * 1.5 + i * 0.37) % 1.0) < 0.5:
                    g[y, x] = mix((255, 230, 120))
        self._peek(g, mix)
        return out

    def _peek(self, g, mix):
        """画面の下から、イモの目がのぞいてペンを追う。できたら ^ ^ の目。"""
        if self.pen is not None:
            look = (self.pen[1] - C / 2) / (C / 2)
        else:
            look = 0.35 * math.sin(self.t * 1.3)
        cx = 24 + round(look * 3)
        happy = self.phase in ("done", "outro")
        blink = (self.t % 3.2) < 0.12 and not happy
        for side in (-1, 1):
            ex = cx + side * 5
            if happy:   # ^ ^
                for dx, dy in ((-2, 46), (-1, 45), (0, 44), (1, 45), (2, 46)):
                    g[dy, ex + dx] = mix(EYE)
            elif blink:
                g[46, ex - 1:ex + 2] = mix(EYE)
            else:
                g[44:47, ex - 1:ex + 2] = mix(EYE)
                g[44, ex - 1] = mix((220, 250, 255))
        if self.phase == "draw" and self.pen is not None:   # ペンを持った手（右下）
            g[43:46, 41] = mix((240, 200, 60))
            g[46, 40:43] = mix(EYE)


# ---------------------------------------------------------------- 試す
def demo_drawings():
    yield Drawing(EXAMPLE)
    yield Drawing({"title": "ねこ", "background": "pink", "shapes": [
        {"type": "ellipse", "color": "orange", "x": 20, "y": 30, "rx": 11, "ry": 8},
        {"type": "circle", "color": "orange", "x": 20, "y": 17, "r": 9},
        {"type": "triangle", "color": "orange", "x": 11, "y": 13, "x2": 13, "y2": 3, "x3": 19, "y3": 9},
        {"type": "triangle", "color": "orange", "x": 21, "y": 9, "x2": 27, "y2": 3, "x3": 29, "y3": 13},
        {"type": "line", "color": "orange", "x": 31, "y": 32, "x2": 37, "y2": 22, "width": 2},
        {"type": "rect", "color": "black", "x": 15, "y": 15, "w": 2, "h": 3},
        {"type": "rect", "color": "black", "x": 23, "y": 15, "w": 2, "h": 3},
        {"type": "dot", "color": "pink", "x": 20, "y": 19},
        {"type": "arc", "color": "black", "x": 20, "y": 20, "rx": 2, "ry": 1.5},
        {"type": "line", "color": "black", "x": 8, "y": 19, "x2": 14, "y2": 20},
        {"type": "line", "color": "black", "x": 26, "y": 20, "x2": 32, "y2": 19},
    ]})


def main(argv=None):
    ap = argparse.ArgumentParser(description="イモのお絵かきを試す")
    ap.add_argument("--demo", action="store_true", help="Ollama なしで、用意した絵を描く")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--model", default="gemma4:12b")
    ap.add_argument("--subject", default=None, help="描くもの（例 ねこ）")
    ap.add_argument("--size", type=int, default=720)
    args = ap.parse_args(argv)
    import pygame
    from .face import dot_grid

    if args.demo:
        drawings = list(demo_drawings())
    else:
        artist = Artist(args.ollama_url, args.model)
        print("Ollama に描いてもらっています…")
        drawings = [artist.picture(args.subject or random.choice(SUBJECTS))]
        print(json.dumps(drawings[0].data, ensure_ascii=False))
    pygame.display.init()
    screen = pygame.display.set_mode((args.size, args.size))
    lo = pygame.Surface((N, N))
    grid = dot_grid(args.size, args.size, N, N)
    clock = pygame.time.Clock()
    for d in drawings:
        show = DrawingShow(d)
        while not show.ended:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return
            img = show.update(clock.tick(20) / 1000.0)
            pygame.surfarray.blit_array(lo, img.astype(np.uint8).swapaxes(0, 1))
            pygame.transform.scale(lo, screen.get_size(), screen)
            screen.blit(grid, (0, 0))
            pygame.display.flip()


if __name__ == "__main__":
    main()
