"""全感情の表情を1枚の画像に並べて保存する（動作確認用）。

    python3 tools/snapshot.py expressions.png
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pygame  # noqa: E402

from robot.face import EMOTION_NAMES, Face  # noqa: E402


def main(out="expressions.png", cell=320, cols=4):
    pygame.display.init()
    pygame.font.init()
    pygame.display.set_mode((1, 1))
    rows = (len(EMOTION_NAMES) + cols - 1) // cols
    sheet = pygame.Surface((cols * cell, rows * cell))
    sheet.fill((40, 40, 48))
    font = pygame.font.Font(None, cell // 9)
    for i, name in enumerate(EMOTION_NAMES):
        face = Face(cell - 8)
        face.next_blink = face.next_saccade = 1e9  # 撮影中はまばたき・視線移動なし
        face.set_emotion(name)
        face.look_at((0.0, 0.0))
        for _ in range(60):
            face.update(1 / 30)
        surf = pygame.Surface((cell - 8, cell - 8))
        face.draw(surf)
        surf.blit(font.render(name, True, (220, 220, 220)), (8, 8))
        sheet.blit(surf, ((i % cols) * cell + 4, (i // cols) * cell + 4))
    pygame.image.save(sheet, out)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:2])
