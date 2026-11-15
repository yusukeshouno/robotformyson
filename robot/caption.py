"""画面の上に字幕を出す（イモの返事）。

スピーカーがなくても、何と言ったかが読める。耳の聞こえにくい人や、まだ字を練習中の子にも。
文字は顔と同じドット絵の雰囲気にするため、12ドットのピクセルフォント（PixelMplus12、
M+ FONT LICENSE、robot/fonts/ に同梱）を小さく描いてから、ぼかさずに大きくする。
"""
import os

import pygame

HERE = os.path.dirname(os.path.abspath(__file__))
PIXEL_FONT = os.path.join(HERE, "fonts", "PixelMplus12-Bold.ttf")
PIXEL_SIZE = 12          # このフォントのドットの格子（12ドット）

WHITE = (240, 245, 255)
BAND = (8, 10, 20, 210)


def find_font():
    """字幕のフォント。ROBOT_FONT で変えられる。なければ None。"""
    env = os.environ.get("ROBOT_FONT")
    if env and os.path.exists(env):
        return env
    return PIXEL_FONT if os.path.exists(PIXEL_FONT) else None


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
    """字幕。say() で返事を出す。update() で時間がたつと消える。

    scale: 1ドットを何ピクセルにするか。720 の画面なら 4（1文字 48 ピクセル、1行 14 文字くらい）。
    """

    def __init__(self, width, font_path=None, scale=None):
        pygame.font.init()
        self.width = width
        self.scale = scale or max(2, width // 180)
        self.path = font_path if font_path is not None else find_font()
        self.font = pygame.font.Font(self.path, PIXEL_SIZE) if self.path else None
        self.text, self.until = "", 0.0
        self._cache = None

    @property
    def ok(self):
        return self.font is not None

    def say(self, text, now):
        """イモの返事。読む時間を見て、しばらく出しておく。"""
        self.text, self.until = text.strip(), now + 4.0 + 0.18 * len(text)
        self._cache = None

    def hold(self, now, seconds=2.0):
        """話しているあいだは消さない。"""
        if self.text:
            self.until = max(self.until, now + seconds)

    def clear(self):
        self.text = ""
        self._cache = None

    def update(self, now):
        if self.text and now > self.until:
            self.clear()

    def render(self):
        """字幕の絵（帯つき）。出すものがなければ None。"""
        if not (self.ok and self.text):
            return None
        if self._cache is None:
            k = self.scale
            pad = k * 3
            inner = (self.width - pad * 4) // k                       # ドット単位の幅
            lines = wrap(self.text, lambda s: self.font.size(s)[0], inner)
            lh = PIXEL_SIZE + 2                                        # 行の高さ（ドット）
            h = lh * len(lines) * k + pad * 2
            surf = pygame.Surface((self.width, h), pygame.SRCALPHA)
            pygame.draw.rect(surf, BAND, (pad, 0, self.width - pad * 2, h), border_radius=pad)
            for i, line in enumerate(lines):
                img = self.font.render(line, False, WHITE)            # ぼかさない
                img = pygame.transform.scale(img, (img.get_width() * k, img.get_height() * k))
                surf.blit(img, ((self.width - img.get_width()) // 2, pad + i * lh * k))
            self._cache = surf
        return self._cache

    def draw(self, canvas, top_margin=None):
        """顔の上のあいている所（おでこ）に出す。"""
        img = self.render()
        if img is None:
            return
        canvas.blit(img, (0, self.scale * 2 if top_margin is None else top_margin))
