"""ロボットの顔デモ。

    python3 -m robot.main              # ウィンドウ表示（カメラがあれば顔を追う）
    python3 -m robot.main --fullscreen # Pi のディスプレイで全画面
    python3 -m robot.main --mouse      # カメラなし：マウスを「人の顔」として扱う

キー: 1-7 感情を出す / 0 自動に戻す / c カメラ映像 / d デバッグ表示 / q・Esc 終了
画面クリック(タップ)で喜ぶ、連打で怒る、なでる(ドラッグ)と照れる。
"""
import argparse
import socket
import time

import pygame

from .emotion import EmotionEngine
from .face import BG, EMOTION_NAMES, Face


class EmotionListener:
    """UDP で "joy 3" のような感情タグを受け取る（会話プログラムとの連携口）。"""

    def __init__(self, port, host="127.0.0.1"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)

    def poll(self):
        msgs = []
        while True:
            try:
                data, _ = self.sock.recvfrom(256)
            except BlockingIOError:
                return msgs
            parts = data.decode("utf-8", "ignore").split()
            if parts and parts[0] in EMOTION_NAMES:
                try:
                    dur = float(parts[1]) if len(parts) > 1 else 3.0
                except ValueError:
                    dur = 3.0
                msgs.append((parts[0], dur))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="エンタメロボットの顔デモ")
    ap.add_argument("--size", type=int, default=720, help="ウィンドウの一辺(px)")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mouse", action="store_true", help="マウスを顔として扱う（カメラ不要）")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--camera", choices=["auto", "picamera", "opencv"], default="auto")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                    help="カメラの取り付け向きの補正")
    ap.add_argument("--no-mirror", action="store_true",
                    help="目が人と逆を向くときに指定")
    ap.add_argument("--udp", type=int, default=5005, help="感情タグ受信ポート（0で無効）")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--run-seconds", type=float, default=0, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    pygame.display.init()
    pygame.font.init()
    if args.fullscreen:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.mouse.set_visible(False)
    else:
        screen = pygame.display.set_mode((args.size, args.size))
    pygame.display.set_caption("robot face")
    sw, sh = screen.get_size()
    S = min(sw, sh)
    offset = ((sw - S) // 2, (sh - S) // 2)
    canvas = pygame.Surface((S, S))

    face = Face(S)
    engine = EmotionEngine(time.monotonic())
    font = pygame.font.Font(None, max(16, S // 24))

    tracker = None
    if not (args.no_camera or args.mouse):
        try:
            from .vision import FaceTracker
            tracker = FaceTracker(args.camera, args.camera_index,
                                  mirror=not args.no_mirror, rotate=args.rotate).start()
            print(f"カメラ: {tracker.backend}")
        except Exception as e:  # カメラがなくても顔だけは動かす
            print(f"カメラを使わずに起動します（{e}）")

    listener = None
    if args.udp:
        try:
            listener = EmotionListener(args.udp)
        except OSError as e:
            print(f"UDP {args.udp} を開けませんでした（{e}）")

    show_cam, debug = False, args.debug
    drag, start = None, time.monotonic()
    clock = pygame.time.Clock()
    running = True
    while running:
        dt = clock.tick(args.fps) / 1000.0
        now = time.monotonic()
        if args.run_seconds and now - start > args.run_seconds:
            break

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif pygame.K_1 <= ev.key <= pygame.K_7:
                    engine.express(EMOTION_NAMES[ev.key - pygame.K_1], 4.0, now)
                elif ev.key == pygame.K_0:
                    engine.express("neutral", 0.0, now)
                elif ev.key == pygame.K_c:
                    show_cam = not show_cam
                elif ev.key == pygame.K_d:
                    debug = not debug
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                drag = 0.0
            elif ev.type == pygame.MOUSEMOTION and drag is not None and ev.buttons[0]:
                drag += abs(ev.rel[0]) + abs(ev.rel[1])
                if drag > S * 0.6:
                    engine.on_stroke(now)
                    drag = -1e9  # 1回のなでで1度だけ
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                if drag is not None and 0 <= drag < S * 0.05:
                    engine.on_tap(now)
                drag = None

        present, fx, fy = False, 0.0, 0.0
        if tracker:
            present, fx, fy = tracker.get()
            fx, fy = fx * 1.3, fy * 1.3
        elif args.mouse and pygame.mouse.get_focused():
            mx, my = pygame.mouse.get_pos()
            fx = (mx - offset[0]) / S * 2 - 1
            fy = (my - offset[1]) / S * 2 - 1
            present = abs(fx) <= 1 and abs(fy) <= 1
        engine.on_face(present, now)
        if listener:
            for emo, dur in listener.poll():
                engine.express(emo, dur, now)

        emotion = engine.update(now)
        face.set_emotion(emotion)
        face.look_at((max(-1, min(1, fx)), max(-1, min(1, fy))) if present else None)
        face.update(dt)
        face.draw(canvas)

        if show_cam and tracker:
            img = tracker.preview()
            if img is not None:
                surf = pygame.image.frombuffer(img.tobytes(), (img.shape[1], img.shape[0]), "RGB")
                w = S // 3
                canvas.blit(pygame.transform.smoothscale(
                    surf, (w, w * img.shape[0] // img.shape[1])), (8, 8))
        if debug:
            label = f"{emotion}  face={'yes' if present else 'no'}  {clock.get_fps():.0f}fps"
            canvas.blit(font.render(label, True, (200, 200, 200)), (8, S - font.get_height() - 8))

        screen.fill(BG)
        screen.blit(canvas, offset)
        pygame.display.flip()

    if tracker:
        tracker.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
