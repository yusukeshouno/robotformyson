"""画面の上に字幕を出す（イモの返事）。

スピーカーがなくても、何と言ったかが読める。耳の聞こえにくい人や、まだ字を練習中の子にも。
日本語のフォントが見つからないときは字幕を出さない（Pi では `sudo apt install fonts-noto-cjk`）。
"""
import os

import pygame

FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
)
MATCH_NAMES = "notosanscjkjp,notosanscjk,ipagothic,takaogothic,droidsansfallback,hiraginosans,meiryo"

WHITE = (240, 245, 255)
BAND = (8, 10, 20, 200)


def find_font():
    """日本語が書けるフォントのファイルを探す。なければ None。"""
    env = os.environ.get("ROBOT_FONT")
    if env and os.path.exists(env):
        return env
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    try:
        return pygame.font.match_font(MATCH_NAMES)
    except Exception:
        return None


def wrap(text, width_of, max_width, max_lines=3):
    """日本語は単語の区切りがないので、1文字ずつ足して幅で折り返す。"""
    lines, cur = [], ""
    for ch in text:
        if ch == "\n" or (cur and width_of(cur + ch) > max_width):
            lines.append(cur)
            cur = "" if ch == "\n" else ch
            if len(lines) == max_lines:
                break
        else:
            cur += ch
    else:
        if cur:
            lines.append(cur)
    if len(lines) == max_lines and "".join(lines) != text.replace("\n", ""):
        last = lines[-1]
        while last and width_of(last + "…") > max_width:
            last = last[:-1]
        lines[-1] = last + "…"
    return lines


class Caption:
    """字幕。say() で返事を出す。update() で時間がたつと消える。"""

    def __init__(self, width, font_path=None, size=None):
        pygame.font.init()
        self.width = width
        self.size = size or max(18, width // 22)
        self.path = font_path if font_path is not None else find_font()
        self.font = pygame.font.Font(self.path, self.size) if self.path else None
        self.text, self.color, self.until = "", WHITE, 0.0
        self._cache = None

    @property
    def ok(self):
        return self.font is not None

    def say(self, text, now):
        """イモの返事。読む時間を見て、しばらく出しておく。"""
        self._set(text, WHITE, now + 4.0 + 0.18 * len(text))

    def hold(self, now, seconds=2.0):
        """話しているあいだは消さない。"""
        if self.text and self.color == WHITE:
            self.until = max(self.until, now + seconds)

    def clear(self):
        self.text = ""
        self._cache = None

    def _set(self, text, color, until):
        self.text, self.color, self.until = text.strip(), color, until
        self._cache = None

    def update(self, now):
        if self.text and now > self.until:
            self.clear()

    def render(self):
        """字幕の絵（帯つき）。出すものがなければ None。"""
        if not (self.ok and self.text):
            return None
        if self._cache is None:
            pad = self.size // 2
            lines = wrap(self.text, lambda s: self.font.size(s)[0], self.width - pad * 4)
            lh = self.font.get_linesize()
            h = lh * len(lines) + pad * 2
            surf = pygame.Surface((self.width, h), pygame.SRCALPHA)
            pygame.draw.rect(surf, BAND, (pad, 0, self.width - pad * 2, h), border_radius=pad)
            for i, line in enumerate(lines):
                img = self.font.render(line, True, self.color)
                surf.blit(img, ((self.width - img.get_width()) // 2, pad + i * lh))
            self._cache = surf
        return self._cache

    def draw(self, canvas, top_margin=None):
        """顔の上のあいている所（おでこ）に出す。口や目にかからない。"""
        img = self.render()
        if img is None:
            return
        m = self.size // 3 if top_margin is None else top_margin
        canvas.blit(img, (0, m))
