"""画面全体を、顔と同じ「LED のドット」で描く。

顔（48×48 ドット）を真ん中に置き、まわりと、顔の消えているドットに背景の模様を描く。
- 縦長の画面（480×800 など）: 横 48 ドット × 縦 80 ドットくらいになり、顔の上下に帯ができる。
- 正方形の画面（720×720 など）: 48×48。模様は顔のうしろにだけ透ける。

毎フレームの計算は数千ドット分だけなので、Pi 4 でも軽い。
"""
import numpy as np
import pygame

from .face import N, OFF, dot_grid

TALL_RATIO = 1.3   # 縦がこれ以上長ければ、顔の上下にも模様を描く


class DotScreen:
    def __init__(self, width, height):
        """width, height は回転させる前（顔がまっすぐに見える向き）の画面の大きさ。"""
        if height >= width * TALL_RATIO:
            self.dot = width / N
            self.rows = max(N, int(height / self.dot))
            self.size = (width, round(self.rows * self.dot))
        else:
            S = min(width, height)
            self.dot = S / N
            self.rows = N
            self.size = (S, S)
        self.top = (self.rows - N) // 2   # 顔の上の帯の行数
        self.face_size = round(N * self.dot)
        self.lo = pygame.Surface((N, self.rows))
        self.grid = dot_grid(self.size[0], self.size[1], N, self.rows)
        self.dim = face_dim(N)[:, None, None]

    @property
    def tall(self):
        return self.rows > N

    def draw(self, surf, face, art=None, keep=None):
        """surf（self.size の大きさ）に、背景の模様 art と顔を描く。

        art は (rows, 48, 3) の明るさ（0..255 くらい、OFF の色に足す）。None なら模様なし。
        keep は (rows, 48) の True/False。True のドットは顔のうしろでも弱めない（花など）。
        """
        if art is None:
            full = np.empty((self.rows, N, 3), np.uint8)
            full[:] = OFF
            face.render()
        else:
            full = np.clip(art + OFF, 0, 255).astype(np.uint8)
            back = art[self.top:self.top + N]
            dim = self.dim if keep is None else np.where(
                keep[self.top:self.top + N, :, None], 1.0, self.dim)
            face.render(back * dim)
        full[self.top:self.top + N] = face.img
        pygame.surfarray.blit_array(self.lo, full.swapaxes(0, 1))
        pygame.transform.scale(self.lo, self.size, surf)
        surf.blit(self.grid, (0, 0))


    def draw_image(self, surf, img):
        """顔の代わりに、(rows, 48, 3) のドット絵をそのまま出す（お絵かきを見せるとき）。"""
        full = np.clip(img, 0, 255).astype(np.uint8)
        pygame.surfarray.blit_array(self.lo, full.swapaxes(0, 1))
        pygame.transform.scale(self.lo, self.size, surf)
        surf.blit(self.grid, (0, 0))


def face_dim(n=N):
    """顔のうしろの模様の明るさ（行ごと）。目と口のあたりは弱く、顔のふちは強く。"""
    y = np.arange(n) + 0.5
    edge = np.minimum(y, n - y)            # 上下のふちからの距離
    return np.clip(0.55 + (6.0 - edge) / 6.0 * 0.45, 0.55, 1.0).astype(np.float32)
