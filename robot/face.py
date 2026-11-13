"""感情を表す顔の描画。

パラメータ（目の開き・まぶたの傾き・口の曲がり等）を感情ごとに定義し、
現在値を目標値へなめらかに補間して描く。まばたき・視線のゆらぎ・呼吸も自動で入る。
"""
import math
import random

import pygame

BG = (8, 10, 18)

# 感情ごとの目標パラメータ。省略したキーは DEFAULTS の値になる。
DEFAULTS = {
    "open": 1.0,        # 目の開き 0..1
    "scale": 1.0,       # 目の大きさ
    "lid_top": 0.0,     # 上まぶたが覆う割合
    "tilt": 0.0,        # 上まぶたの傾き（+ で内側が下がる＝怒り、- で外側が下がる＝悲しみ）
    "lid_bot": 0.0,     # 下まぶたが覆う割合（笑い目）
    "mouth": 0.15,      # 口の曲がり（+ 笑顔、- への字）
    "mouth_open": 0.0,  # 口の開き
    "round": 0.0,       # 丸い「o」の口
    "blush": 0.0,       # ほっぺ
    "heart": 0.0,       # ハート目
    "zzz": 0.0,         # 寝息
    "color": (90, 220, 255),
}

EMOTIONS = {
    "neutral": {},
    "joy": {"scale": 1.05, "lid_bot": 0.55, "mouth": 0.9, "mouth_open": 0.35,
            "blush": 0.6, "color": (120, 235, 255)},
    "sad": {"open": 0.85, "scale": 0.95, "lid_top": 0.3, "tilt": -0.5,
            "mouth": -0.6, "color": (80, 140, 255)},
    "angry": {"open": 0.9, "lid_top": 0.2, "tilt": 0.6, "lid_bot": 0.1,
              "mouth": -0.45, "mouth_open": 0.1, "color": (255, 90, 80)},
    "surprise": {"scale": 1.25, "mouth": 0.0, "round": 1.0,
                 "color": (150, 240, 255)},
    "sleepy": {"open": 0.06, "mouth": 0.0, "zzz": 1.0, "color": (70, 160, 200)},
    "love": {"scale": 1.1, "heart": 1.0, "mouth": 0.8, "mouth_open": 0.2,
             "blush": 1.0, "color": (255, 120, 190)},
    # 返事を考えているあいだ：目を細めて上を見る
    "think": {"open": 0.85, "scale": 0.95, "lid_bot": 0.2, "mouth": -0.1,
              "color": (110, 200, 255)},
}

EMOTION_NAMES = list(EMOTIONS)
_NUM_KEYS = [k for k in DEFAULTS if k != "color"]


def _target(name):
    t = dict(DEFAULTS)
    t.update(EMOTIONS.get(name, {}))
    return t


def _lerp(a, b, t):
    return a + (b - a) * t


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
        self.next_saccade = 0.0
        self.next_blink = random.uniform(1.0, 3.0)
        self.blink_start = -1.0
        self.font = pygame.font.Font(None, max(12, int(size * 0.13)))

    def set_emotion(self, name):
        if name in EMOTIONS:
            self.emotion = name

    def set_talk(self, level):
        self.talk = level

    def look_at(self, target):
        """target: (x, y) を -1..1 で。None なら自由に視線を泳がせる。"""
        self.external_gaze = target

    # ---- 更新 ----
    def update(self, dt):
        self.t += dt
        k = 1 - math.exp(-dt * 8)
        target = _target(self.emotion)
        for key in _NUM_KEYS:
            self.params[key] = _lerp(self.params[key], target[key], k)
        self.params["color"] = tuple(
            _lerp(a, b, k) for a, b in zip(self.params["color"], target["color"]))

        if self.t >= self.next_saccade:
            self.idle_gaze = [random.uniform(-0.6, 0.6), random.uniform(-0.4, 0.4)]
            self.next_saccade = self.t + random.uniform(1.2, 3.5)
        if self.emotion == "sleepy":
            goal = (0.0, 0.3)
        elif self.emotion == "think":
            goal = (0.45 + 0.1 * math.sin(self.t * 0.8), -0.6)
        elif self.external_gaze is not None:
            goal = self.external_gaze
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
        phase = (self.t - self.blink_start) / 0.16
        if 0 <= phase < 1:
            return max(abs(1 - 2 * phase), 0.05)
        return 1.0

    # ---- 描画 ----
    def draw(self, surf):
        S = self.size
        p = self.params
        gx, gy = self.gaze
        bob = math.sin(self.t * 1.6) * S * 0.006
        color = tuple(int(c) for c in p["color"])
        blink = self._blink_factor()

        surf.fill(BG)
        for side in (-1, 1):  # -1 = 画面左の目
            self._draw_eye(surf, side, color, gx, gy, bob, blink)
        self._draw_blush(surf, gx, bob)
        self._draw_mouth(surf, color, gx, gy, bob)
        self._draw_zzz(surf, color)

    def _draw_eye(self, surf, side, color, gx, gy, bob, blink):
        S, p = self.size, self.params
        cx = S * 0.5 + side * S * 0.2 + gx * S * 0.07
        cy = S * 0.42 + gy * S * 0.06 + bob
        persp = 1 + 0.08 * gx * side  # 見ている側の目を少し大きく
        w = S * 0.22 * p["scale"] * persp
        h_full = S * 0.28 * p["scale"] * persp
        h = max(h_full * p["open"] * blink, S * 0.012)
        rect = pygame.Rect(0, 0, int(w), int(h))
        rect.center = (int(cx), int(cy))

        if p["heart"] > 0.5:
            self._draw_heart(surf, color, cx, cy, w, h, h / h_full)
        else:
            pygame.draw.rect(surf, color, rect, border_radius=int(min(w, h) * 0.38))
            if h > S * 0.08:
                hl = tuple(int(_lerp(c, 255, 0.6)) for c in color)
                pygame.draw.circle(surf, hl, (int(rect.left + w * 0.28),
                                              int(rect.top + h * 0.25)), int(w * 0.08))

        # まぶたは背景色で上書きして表現する
        surf.set_clip(rect.inflate(6, 6))
        lt, tilt = p["lid_top"], p["tilt"]
        if lt > 0.01 or abs(tilt) > 0.01:
            inner_x = rect.right + 4 if side < 0 else rect.left - 4
            outer_x = rect.left - 4 if side < 0 else rect.right + 4
            y_in = rect.top + h * (lt + tilt * 0.5)
            y_out = rect.top + h * (lt - tilt * 0.5)
            pygame.draw.polygon(surf, BG, [(inner_x, rect.top - 6), (outer_x, rect.top - 6),
                                           (outer_x, y_out), (inner_x, y_in)])
        lb = p["lid_bot"]
        if lb > 0.01:
            ew, eh = w * 1.6, h * 1.3
            pygame.draw.ellipse(surf, BG, pygame.Rect(int(cx - ew / 2),
                                                      int(rect.bottom - lb * h),
                                                      int(ew), int(eh)))
        surf.set_clip(None)

    @staticmethod
    def _draw_heart(surf, color, cx, cy, w, h, sy):
        rx = w * 0.27
        ry = rx * sy
        for dx in (-1, 1):
            lobe = pygame.Rect(0, 0, int(rx * 2), int(ry * 2))
            lobe.center = (int(cx + dx * w * 0.23), int(cy - h * 0.15))
            pygame.draw.ellipse(surf, color, lobe)
        pygame.draw.polygon(surf, color, [(cx - w * 0.49, cy - h * 0.08),
                                          (cx + w * 0.49, cy - h * 0.08),
                                          (cx, cy + h * 0.5)])

    def _draw_blush(self, surf, gx, bob):
        S, b = self.size, self.params["blush"]
        if b < 0.02:
            return
        bs = pygame.Surface((int(S * 0.12), int(S * 0.05)), pygame.SRCALPHA)
        pygame.draw.ellipse(bs, (255, 110, 150, int(150 * b)), bs.get_rect())
        for side in (-1, 1):
            r = bs.get_rect(center=(int(S * 0.5 + side * S * 0.27 + gx * S * 0.05),
                                    int(S * 0.62 + bob)))
            surf.blit(bs, r)

    def _draw_mouth(self, surf, color, gx, gy, bob):
        S, p = self.size, self.params
        mx = S * 0.5 + gx * S * 0.035
        my = S * 0.73 + gy * S * 0.02 + bob
        mw = S * 0.13 * (1 - 0.75 * p["round"])
        curve = p["mouth"] * S * 0.045
        opening = (p["mouth_open"] + self.talk * 0.9) * S * 0.07
        n = 16
        upper, lower = [], []
        for i in range(n + 1):
            u = 2 * i / n - 1
            x = mx + u * mw / 2
            y = my + curve * (1 - u * u)
            upper.append((x, y))
            lower.append((x, y + opening * (1 - u * u)))
        width = max(2, int(S * 0.014))
        if p["round"] > 0.05:
            o = pygame.Rect(0, 0, int(S * 0.07 * p["round"]), int(S * 0.09 * p["round"]))
            o.center = (int(mx), int(my + S * 0.01))
            pygame.draw.ellipse(surf, color, o)
        if opening > S * 0.008:
            pygame.draw.polygon(surf, color, upper + lower[::-1])
        pygame.draw.lines(surf, color, False, upper, width)
        for pt in (upper[0], upper[-1]):
            pygame.draw.circle(surf, color, (int(pt[0]), int(pt[1])), width // 2)

    def _draw_zzz(self, surf, color):
        S, z = self.size, self.params["zzz"]
        if z < 0.3:
            return
        for i in range(3):
            phase = (self.t * 0.4 + i / 3) % 1.0
            img = self.font.render("Z", True, color)
            img = pygame.transform.rotozoom(img, 15, 0.5 + phase * 0.7)
            img.set_alpha(int(255 * z * math.sin(phase * math.pi)))
            surf.blit(img, img.get_rect(center=(int(S * 0.74 + phase * S * 0.1),
                                                int(S * 0.3 - phase * S * 0.2))))
