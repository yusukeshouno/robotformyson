"""ピコの庭：家族ひとりに1本の植物が、なかよし度に合わせて育つ。

- 植物の形は人ごとに決まっている（同じ人ならいつも同じ形）。なかよし度がたまるほど伸びる。
- レベル2で葉、レベル4でつぼみ、レベル5で花。レベル5のあとも少しずつ大きくなる。
- 今いっしょにいる人の植物が明るくなり、レベル5ならまわりに光の粒が飛ぶ。
- 顔の上に空（昼・夕方・夜で変わる）、下に地面。正方形の画面では眉より上のすきまが空になる。
  背の高い植物は顔のうしろにうっすら見える（花は弱めない）。

成長の記録（growth.json）を読むだけで、新しく保存するものはない。

    python3 -m robot.garden             # PC で庭だけ見る（数字キーで なかよし度を変える）
"""
import math
import random
import time
import zlib

import numpy as np

from .face import N
from .growth import FAMILY, LEVELS, level_for

MAX_PLANTS = 4
SQUARE_SKY = 7         # 正方形の画面で空にする上の行数
SQUARE_PLANT_TOP = 14  # 正方形の画面で植物がのびる上の限り（口のあたりまで）
# なかよし度 → 植物がどこまで伸びたか（全体を 1 として）。レベルの変わり目で区切る
GROW_POINTS = [need for need, _ in LEVELS] + [LEVELS[-1][0] + 600]
GROW_FRAC = [0.06, 0.15, 0.3, 0.5, 0.7, 1.0]
GROW_SPEED = 0.18      # 1秒にのびる割合（起動したときや、なかよし度が上がったときの動き）

STEM_LOW, STEM_HIGH = (60, 150, 80), (110, 200, 110)
LEAF = (120, 225, 120)
GOLD = (255, 235, 140)
SOIL, GRASS = (110, 70, 40), (70, 150, 80)
FLOWER_COLORS = [(255, 170, 60), (255, 100, 180), (110, 170, 255), (190, 120, 255),
                 (255, 90, 90), (255, 220, 80)]
SKY = {  # (上の色, 下の色, 明るさ)
    "day": ((70, 130, 210), (150, 200, 230), 0.55),
    "dusk": ((60, 50, 120), (240, 140, 90), 0.6),
    "night": ((18, 22, 70), (8, 10, 30), 1.0),
}


def growth_frac(points):
    return float(np.interp(points, GROW_POINTS, GROW_FRAC))


def time_of_day(hour):
    if 6 <= hour < 17:
        return "day"
    if 17 <= hour < 19 or 5 <= hour < 6:
        return "dusk"
    return "night"


class Plant:
    """1本の植物。ドットの位置と「何番目にのびるか」(t) を最初に全部決めておく。"""

    def __init__(self, seed, x0, ground, top):
        r = random.Random(seed)
        cells, tips = [], []   # (x, y, t, 種類 0=茎 1=葉) / (x, y, t, 深さ)

        def grow(x, y, length, depth, t0, slope):
            drift = 0.0
            for i in range(1, length + 1):
                y -= 1
                drift += slope + (r.random() - 0.5) * 0.5
                if abs(drift) >= 1:
                    s = 1 if drift > 0 else -1
                    x += s
                    drift -= s
                if y < top:
                    break
                t = t0 + i
                cells.append((x, y, t, 0))
                if r.random() < 0.32:
                    cells.append((x + r.choice((-1, 1)), y, t + 1, 1))
                if depth < 2 and i > 3 and r.random() < (0.09 if depth else 0.13):
                    s = r.choice((-1, 1))
                    grow(x, y, int(length * (0.35 + r.random() * 0.3)), depth + 1, t,
                         s * (0.6 + r.random() * 0.5))
            tips.append((x, y, t0 + length, depth))

        span = ground - top
        # 大きく育っても顔の口のあたりまで（目のうしろは茂らせない）
        grow(x0, ground, int(span * (0.45 + r.random() * 0.17)), 0, 0, (r.random() - 0.5) * 0.15)
        c = np.array(cells, np.int32).reshape(-1, 4)
        self.x, self.y, self.t, self.kind = c[:, 0], c[:, 1], c[:, 2].astype(np.float32), c[:, 3]
        tp = np.array(tips, np.int32).reshape(-1, 4)
        self.tx, self.ty, self.tt, self.tdepth = tp[:, 0], tp[:, 1], tp[:, 2].astype(np.float32), tp[:, 3]
        self.t_max = float(max(self.t.max(initial=1), self.tt.max(initial=1)))
        h = np.clip((ground - self.y) / max(1, span), 0, 1)[:, None]
        self.colors = np.where(self.kind[:, None] == 1, np.array(LEAF, np.float32),
                               np.array(STEM_LOW) + (np.array(STEM_HIGH) - np.array(STEM_LOW)) * h)
        self.phase = r.random() * 6.28
        self.color = FLOWER_COLORS[seed % len(FLOWER_COLORS)]
        self.x0, self.ground, self.span = x0, ground, span
        self.g = 0.0   # 今どこまでのびて見えているか（t の単位）


class Garden:
    def __init__(self, growth, rows, top, clock=time.time):
        """rows: 画面の行数（ドット）、top: 顔の上の帯の行数。"""
        self.growth = growth
        self.rows, self.top = rows, top
        self.tall = top >= 8
        # 空の行数。縦長なら顔の上の帯ぜんぶ、正方形なら顔の上のすきま（眉より上）
        self.sky_rows = top if self.tall else SQUARE_SKY
        self.ground = rows - 1
        self.clock = clock
        self.plants = {}     # 人の id → Plant
        self.order = []      # 左から並べる人の id
        self._check = 0.0
        self.t = 0.0
        self._sky_key, self._sky = None, None
        self.keep = None
        self._grass = None

    # ---- だれの植物を植えるか ----
    def _people(self):
        people = self.growth.data["people"]
        named = [pid for pid in people if pid != FAMILY]
        if not named:
            return [FAMILY] if FAMILY in people or not people else []
        # なかよし度の高い順に MAX_PLANTS 人まで。並びは出会った順
        named.sort(key=lambda pid: -people[pid].get("points", 0))
        return sorted(named[:MAX_PLANTS], key=lambda pid: people[pid].get("first_met", 0))

    def _replant(self):
        order = self._people()
        if order == self.order:
            return
        n = max(1, len(order))
        old = self.plants
        self.plants = {}
        for i, pid in enumerate(order):
            x0 = int(round((i + 0.5) * N / n))
            seed = zlib.crc32(pid.encode("utf-8"))
            p = old.get(pid)
            if p is None or p.x0 != x0:
                p = Plant(seed, x0, self.ground, top=18 if self.tall else SQUARE_PLANT_TOP)
            self.plants[pid] = p
        self.order = order

    # ---- 毎フレーム ----
    def update(self, dt):
        """(rows, 48, 3) の明るさを返す。"""
        self.t += dt
        self._check -= dt
        if self._check <= 0:
            self._replant()
            self._check = 2.0
        art = self._sky_layer().copy()
        self.keep = np.zeros(art.shape[:2], bool)   # 顔のうしろでも弱めないドット（花・光の粒）
        self._ground(art)
        people = self.growth.data["people"]
        current = self.growth.person
        for pid, plant in self.plants.items():
            p = people.get(pid) or {"points": 0}
            pts = p.get("points", 0)
            lv = level_for(pts)
            goal = growth_frac(pts) * plant.t_max
            step = GROW_SPEED * plant.t_max * dt
            plant.g = goal if abs(goal - plant.g) <= step else plant.g + math.copysign(step, goal - plant.g)
            bright = 1.0 if pid == current else (0.85 if current not in self.plants else 0.7)
            self._draw_plant(art, plant, lv, bright)
            if lv >= 5 and pid == current:
                self._motes(art, plant)
        return art

    def _sway(self, plant, y):
        return np.round(np.sin(self.t * 1.1 + plant.phase + y * 0.12)
                        * (plant.ground - y) / max(1, plant.span) * 1.3).astype(np.int32)

    def _draw_plant(self, art, plant, lv, bright):
        show = plant.t <= plant.g
        if lv < 2:
            show &= plant.kind == 0
        x = plant.x[show] + self._sway(plant, plant.y[show])
        colors = plant.colors[show]
        if lv >= 5:  # だいしんゆう：葉のあいだにも小さな花が咲く
            bloom = (plant.kind[show] == 1) & ((plant.x[show] + plant.y[show]) % 3 == 0)
            colors = np.where(bloom[:, None], np.array(plant.color, np.float32), colors)
            _put(art, x[bloom], plant.y[show][bloom], colors[bloom] * bright, self.keep)
        _put(art, x, plant.y[show], colors * bright)
        if lv < 4:
            return
        reach = plant.tt <= plant.g + 0.5
        tx = plant.tx[reach] + self._sway(plant, plant.ty[reach])
        ty = plant.ty[reach]
        col = np.array(plant.color, np.float32)
        if lv == 4:  # つぼみ
            _put(art, tx, ty, np.broadcast_to(col * 0.9 * bright, (tx.size, 3)))
            return
        for u, v in ((0, -1), (1, 0), (0, 1), (-1, 0)):   # 花びら
            _put(art, tx + u, ty + v, np.broadcast_to(col * bright, (tx.size, 3)), self.keep)
        main = plant.tdepth[reach] == 0   # いちばん上の花は大きく
        for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            _put(art, tx[main] + u, ty[main] + v,
                 np.broadcast_to(col * 0.6 * bright, (int(main.sum()), 3)), self.keep)
        _put(art, tx, ty, np.broadcast_to(np.array(GOLD, np.float32) * bright, (tx.size, 3)), self.keep)

    def _motes(self, art, plant):
        """だいしんゆうの植物のまわりを飛ぶ光の粒（夜はほたる、昼はちょうちょ）。"""
        night = self._sky_key == "night"
        mid = plant.ground - plant.span * 0.45
        for f in range(4):
            x = plant.x0 + math.sin(self.t * 0.7 + f * 1.7) * 7
            y = mid + math.cos(self.t * 0.5 + f * 2.3) * plant.span * 0.35
            a = 0.5 + 0.5 * math.sin(self.t * 4 + f)
            c = np.array((220, 255, 120) if night else (255, 255, 255), np.float32) * a
            _put(art, np.array([round(x)]), np.array([round(y)]), c[None], self.keep)

    def _sky_layer(self):
        """上の帯の空と星。時間帯が変わったときだけ作り直す（星のまたたきは毎回）。"""
        key = time_of_day(time.localtime(self.clock()).tm_hour)
        if key != self._sky_key:
            self._sky_key = key
            sky = np.zeros((self.rows, N, 3), np.float32)
            h = self.sky_rows
            hi, lo, a = SKY[key]
            f = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
            sky[:h] = (np.array(hi) + (np.array(lo) - np.array(hi)) * f) * a
            if not self.tall:  # 正方形の画面：顔のおでこに向かって空をうすくする
                sky[:h] *= np.linspace(1.0, 0.25, h, dtype=np.float32)[:, None, None]
            cy = min(5, h // 2)
            if key == "day":    # おひさま
                yy, xx = np.mgrid[0:h, 0:N]
                sky[:h][(xx - 8) ** 2 + (yy - cy) ** 2 <= 5] = (255, 230, 140)
            elif key == "night":  # おつきさま
                for u, v in ((0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1)):
                    sky[max(1, cy - 1) + v, 40 + u] = (255, 240, 190)
            self._sky = sky
            r = random.Random(5)
            n = (22 if self.tall else 12) if key == "night" else 0
            self._stars = (np.array([r.randrange(N) for _ in range(n)], np.int32),
                           np.array([r.randrange(max(1, h - 2)) for _ in range(n)], np.int32),
                           np.array([r.random() * 6 for _ in range(n)], np.float32))
        sx, sy, ph = self._stars
        if sx.size == 0:
            return self._sky
        out = self._sky.copy()
        b = 0.3 + 0.7 * np.maximum(0, np.sin(self.t * 1.5 + ph)) ** 2
        _put(out, sx, sy, np.array((230, 230, 255), np.float32) * b[:, None])
        return out

    def _ground(self, art):
        """地面と、風にゆれる草むら（だれの植物でもない、いつもある草）。"""
        g = self.ground
        art[g] = SOIL
        if self._grass is None:
            r = random.Random(17)
            tufts = []
            for x in range(N):
                h = r.choice((0, 0, 1, 1, 2))
                tufts += [(x, g - 1 - k, k) for k in range(h)]
            self._grass = np.array(tufts, np.int32).reshape(-1, 3)
        x, y, k = self._grass.T
        sway = np.round(np.sin(self.t * 1.3 + x * 0.4) * 0.6 * (k >= 1)).astype(np.int32)
        shade = (0.4 + 0.12 * k)[:, None] * np.array(GRASS, np.float32)
        _put(art, x + sway, y, shade)

def _put(art, x, y, colors, keep=None):
    """(x, y) のドットに色を置く（はみ出したぶんは捨てる、明るいほうを残す）。"""
    x, y = np.asarray(x), np.asarray(y)
    ok = (x >= 0) & (x < art.shape[1]) & (y >= 0) & (y < art.shape[0])
    if not ok.any():
        return
    c = np.asarray(colors, np.float32).reshape(-1, 3)
    if c.shape[0] != ok.shape[0]:
        c = np.broadcast_to(c, (ok.shape[0], 3))
    ys, xs, c = y[ok], x[ok], c[ok]
    art[ys, xs] = np.maximum(art[ys, xs], c)
    if keep is not None:
        keep[ys, xs] = True


def main(argv=None):
    """PC で庭だけを見る。1〜5 キーで今の人のレベルを変える、Tab で人を切りかえる。"""
    import argparse
    import os

    import pygame

    from .face import Face
    from .growth import Growth
    from .screen import DotScreen

    ap = argparse.ArgumentParser(description="ピコの庭を PC で見る")
    ap.add_argument("--size", default="720x720", help="画面の大きさ（例 480x800, 720x720）")
    ap.add_argument("--frames", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    w, h = (int(v) for v in args.size.lower().split("x"))

    g = Growth(os.devnull, autosave=1e9)
    g.data = {"version": 1, "people": {}}
    for pid, name, pts in (("p1", "ぼく", 300), ("p2", "ママ", 1200), ("p3", "パパ", 120)):
        g._get(pid, name)["points"] = pts
        g._get(pid)["level"] = level_for(pts)
    g.person = "p1"

    pygame.display.init()
    screen = DotScreen(w, h)
    win = pygame.display.set_mode(screen.size)
    pygame.display.set_caption("ピコの庭")
    face = Face(screen.face_size)
    garden = Garden(g, screen.rows, screen.top)
    clock = pygame.time.Clock()
    frame = 0
    while not args.frames or frame < args.frames:
        dt = clock.tick(30) / 1000
        frame += 1
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_q, pygame.K_ESCAPE)):
                return
            if ev.type == pygame.KEYDOWN and pygame.K_1 <= ev.key <= pygame.K_5:
                g.current["points"] = GROW_POINTS[ev.key - pygame.K_1]
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_TAB:
                ids = list(g.data["people"])
                g.person = ids[(ids.index(g.person) + 1) % len(ids)]
        face.update(dt)
        art = garden.update(dt)
        screen.draw(win, face, art, garden.keep)
        pygame.display.flip()


if __name__ == "__main__":
    main()
