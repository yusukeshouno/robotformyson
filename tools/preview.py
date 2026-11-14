"""表情が動くようすを GIF に書き出す（README・確認用。Pillow が必要）。

    pip install pillow
    python3 tools/preview.py docs/face_demo.gif
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pygame  # noqa: E402
from PIL import Image  # noqa: E402

from robot.face import Face  # noqa: E402

# (感情, 秒数, 視線) の台本。視線が None なら自由に泳ぐ
SCRIPT = [
    ("neutral", 2.6, None), ("joy", 2.0, None), ("laugh", 1.8, None),
    ("wink", 1.6, None), ("love", 2.2, None), ("shy", 2.2, None),
    ("surprise", 1.6, (0.0, 0.0)), ("excited", 2.0, None), ("think", 2.2, None),
    ("confused", 2.4, None), ("sad", 2.4, (0.0, 0.3)), ("cry", 2.0, None),
    ("angry", 2.0, (0.0, 0.0)), ("dizzy", 2.4, None), ("sleepy", 2.6, None),
]


def main(out="face_demo.gif", size=360, fps=15):
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    face = Face(size)
    surf = pygame.Surface((size, size))
    frames = []
    for emotion, secs, gaze in SCRIPT:
        face.set_emotion(emotion)
        face.look_at(gaze)
        talk_from = secs * 0.35 if emotion in ("joy", "excited") else None
        for i in range(int(secs * fps)):
            # うれしいときは少しおしゃべり（口パク）
            t = i / fps
            face.set_talk(abs(((t * 7) % 2) - 1) if talk_from and t > talk_from else 0.0)
            for _ in range(2):
                face.update(0.5 / fps)
            face.draw(surf)
            raw = pygame.image.tobytes(surf, "RGB")
            frames.append(Image.frombytes("RGB", (size, size), raw).quantize(64))
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    print(out, len(frames), "frames")


if __name__ == "__main__":
    main(*sys.argv[1:2])
