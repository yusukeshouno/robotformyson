"""ロボットの顔デモ。

    python3 -m robot.main              # ウィンドウ表示（カメラがあれば顔を追う）
    python3 -m robot.main --fullscreen # Pi のディスプレイで全画面
    python3 -m robot.main --mouse      # カメラなし：マウスを「人の顔」として扱う

    python3 -m robot.main --talk       # 会話機能つき（README の準備が必要）

キー: 1-9 感情を出す / ←→ 全部の表情を順に出す / 0 自動に戻す / c カメラ映像 / d デバッグ表示 / q・Esc 終了
画面クリック(タップ)で喜ぶ、連打で怒る、なでる(ドラッグ)と照れる。
"""
import argparse
import os
import socket
import sys
import time

import pygame

from .arms import ArmLink, ArmMotion, add_arm_args
from .audio import add_audio_args
from .camera import add_camera_args, settings_from_args
from .emotion import EmotionEngine
from .growth import Growth
from .memory import Memory
from .language import Language
from . import personality
from .face import BG, EMOTION_NAMES, Face
from .screen import DotScreen


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
    ap.add_argument("--size", default="720",
                    help="ウィンドウの大きさ(px)。一辺だけ（720）か 幅x高さ（480x800）")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--fps", type=int, default=20)  # Pi 4 では 30 だと耳（音の処理）の時間がなくなる
    ap.add_argument("--mouse", action="store_true", help="マウスを顔として扱う（カメラ不要）")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--camera", choices=["auto", "picamera", "opencv"], default="auto")
    add_camera_args(ap)
    ap.add_argument("--screen-rotate", type=int, choices=[0, 90, 180, 270], default=0,
                    help="画面の回転（時計回り）。ケースに入れた向きに合わせる")
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                    help="カメラの取り付け向きの補正")
    ap.add_argument("--no-mirror", action="store_true",
                    help="目が人と逆を向くときに指定")
    ap.add_argument("--gaze-gain", type=float, default=1.3,
                    help="目の動きの大きさ（カメラの写る範囲が広いときは大きく）")
    ap.add_argument("--gaze-offset-y", type=float, default=0.0,
                    help="目の上下のずれの補正（カメラが画面の上にあって目が下を向きすぎるときは -0.2 など）")
    ap.add_argument("--udp", type=int, default=5005, help="感情タグ受信ポート（0で無効）")
    ap.add_argument("--talk", action="store_true", help="会話機能を使う")
    ap.add_argument("--no-faces", action="store_true",
                    help="家族の顔を見分ける機能を使わない")
    ap.add_argument("--name", default="ピコ", help="ロボットの名前")
    ap.add_argument("--brain", choices=["ollama", "claude", "simple"], default="ollama",
                    help="考える部分: ollama（無料・Mac など）/ claude（有料）/ simple（決まった受け答えだけ）")
    ap.add_argument("--ollama-url", default="http://localhost:11434",
                    help="Ollama の場所（Mac で動かすなら http://<Macの名前>.local:11434）")
    ap.add_argument("--model", default=None, help="使うモデル名")
    ap.add_argument("--vosk-model", default=None, help="音声認識モデルのフォルダ")
    ap.add_argument("--always-reply", action="store_true",
                    help="カメラに顔が見えていなくても返事する（ふつうは顔が見えるか「イモ」と呼んだときだけ）")
    ap.add_argument("--no-vad", action="store_true",
                    help="話の区切りを Silero VAD ではなく Vosk で見つける")
    ap.add_argument("--whisper-url", default=None,
                    help="Whisper の書き起こしサーバー（例 http://Mac-mini.local:8765）。"
                         "指定すると聞き取りが正確になる。ROBOT_WHISPER_URL でも可")
    ap.add_argument("--voice", default=None, help="Open JTalk の声ファイル (.htsvoice)")
    ap.add_argument("--pitch", type=float, default=2.0, help="声の高さ（半音単位）")
    add_audio_args(ap)
    add_arm_args(ap)
    ap.add_argument("--no-growth", action="store_true", help="成長の記録をしない")
    ap.add_argument("--no-garden", action="store_true",
                    help="背景の「ピコの庭」を出さない（なかよし度で育つ植物）")
    ap.add_argument("--personality", choices=personality.MODES, default="curious",
                    help="curious=好奇心いっぱいの年下の友達 / plain=前と同じ / weekly=週ごとに入れかえて比べる")
    ap.add_argument("--no-language", action="store_true",
                    help="話し方に合わせて言葉を変えない（話した長さも数えない）")
    ap.add_argument("--no-memory", action="store_true",
                    help="前に話したことをおぼえない（小さい子には無い方が楽しいこともある）")
    ap.add_argument("--no-caption", action="store_true",
                    help="画面の上に字幕（イモの返事）を出さない")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--run-seconds", type=float, default=0, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def screen_to_face(x, y, rotate):
    """画面上の座標(-1..1)を、回転前の顔の座標に戻す。"""
    for _ in range(rotate // 90):
        x, y = y, -x
    return x, y


def start_faces(args, tracker):
    """家族の顔を見分ける機能を起動する。モデルがなければ使わずに動く。"""
    if not tracker or args.no_faces:
        return None, None
    try:
        from .faces import Embedder, FaceDB, FaceRecognizer, Greeter
        faces = FaceRecognizer(FaceDB(), Embedder()).start(tracker.latest)
    except Exception as e:
        print(f"顔を見分ける機能を使わずに起動します（{e}）")
        return None, None
    print(f"顔をおぼえる: {len(faces.db.people)}人 登録ずみ")
    return faces, Greeter(faces)


def start_conversation(args, tracker, last_face, growth=None, faces=None, memory=None,
                       language=None):
    """会話機能を起動する。準備が足りなければ理由を表示して顔だけで動く。"""
    try:
        from . import conversation as cv
        brain = cv.make_brain(args.brain, args.model, args.name, args.ollama_url)
        hints = [lambda: personality.hint(args.personality)]  # 好奇心・まちがえても大丈夫
        print(f"性格: {'好奇心いっぱい' if personality.is_curious(args.personality) else 'ふつう'}"
              f"（--personality {args.personality}）")
        if growth:  # なかよしになるほど話し方が変わる
            hints.append(growth.prompt_hint)
            brain.speaker_in_system = True
        if language:  # 話し方のむずかしさ・今週のことば
            hints.append(language.prompt_hint)
        if memory:  # 前に話したこと。まとめは家の中の Ollama だけ（Claude には送らない）
            hints.append(memory.prompt_hint)
            memory.summarize = getattr(brain, "summarize", None)
        if hasattr(brain, "extra_system"):
            brain.extra_system = lambda: "\n".join(h for h in (f() for f in hints) if h)
        audio = cv.AudioSettings.load(args.mic, args.speaker, args.echo_tail)
        mic, speaker = audio.resolve()
        voice = cv.Voice(args.voice, pitch=args.pitch, device=speaker)
        whisper = args.whisper_url or os.environ.get("ROBOT_WHISPER_URL")
        ears = cv.Ears(args.vosk_model or cv.DEFAULT_VOSK, device=mic,
                       gate=cv.MicGate(audio.echo_tail),
                       transcriber=cv.WhisperClient(whisper) if whisper else None)
        if not args.no_vad:  # 話しはじめ・終わりは Silero VAD で（小さな声に強い）
            from .vad import Segmenter
            ears.segmenter = Segmenter.create(ears.rate)
        print(f"聞き取り: {'Whisper（' + whisper + '）、つながらないときは Vosk' if whisper else 'Vosk'}"
              f"、区切り: {'Silero VAD' if ears.segmenter else 'Vosk'}")
    except Exception as e:
        print(f"会話機能を使わずに起動します（{e}）")
        return None
    attended = None
    if tracker and not args.always_reply:  # カメラがあれば、顔が見えている（30秒以内）ときだけ返事する
        attended = lambda: time.monotonic() - last_face[0] < 30.0  # noqa: E731
    conv = cv.Conversation(brain, voice, ears, get_image=tracker.jpeg if tracker else None,
                           is_attended=attended, wake_words=(args.name,),
                           faces=faces, memory=memory, language=language).start()
    print("会話機能: 起動しました")
    return conv


def start_arms(args):
    """手の動き。基板がなくても動きの計算はする（させば自動でつながる）。"""
    if args.no_arms:
        return None, None
    link = ArmLink(args.arms_device, args.arms_swap)
    if link.connect():
        print(f"手: {link.label}")
    else:
        print("手: 基板が見つからないので動かしません（させば自動でつながります）")
    return ArmMotion(), link


def main(argv=None):
    try:  # ログに1行ずつすぐ書く（ファイルに出すと、ためこまれて落ちたときに消えるため）
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    args = parse_args(argv)
    pygame.display.init()
    pygame.font.init()
    if args.fullscreen:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.mouse.set_visible(False)
    else:
        w, _, h = args.size.lower().partition("x")
        screen = pygame.display.set_mode((int(w), int(h or w)))
    pygame.display.set_caption("robot face")
    sw, sh = screen.get_size()
    # 顔がまっすぐ見える向きでの画面の大きさ。縦長なら顔の上下にも背景を描く
    dots = DotScreen(*((sh, sw) if args.screen_rotate in (90, 270) else (sw, sh)))
    S = dots.face_size
    canvas = pygame.Surface(dots.size)
    cw, ch = dots.size

    face = Face(S)
    engine = EmotionEngine(time.monotonic())
    font = pygame.font.Font(None, max(16, S // 24))

    tracker = None
    if not (args.no_camera or args.mouse):
        try:
            from .vision import FaceTracker
            tracker = FaceTracker(args.camera, settings_from_args(args),
                                  mirror=not args.no_mirror, rotate=args.rotate).start()
            print(f"カメラ: {tracker.label}")
        except Exception as e:  # カメラがなくても顔だけは動かす
            print(f"カメラを使わずに起動します（{e}）")

    faces, greeter = start_faces(args, tracker)

    listener = None
    if args.udp:
        try:
            listener = EmotionListener(args.udp)
        except OSError as e:
            print(f"UDP {args.udp} を開けませんでした（{e}）")

    growth = None if args.no_growth else Growth()
    garden = None
    if growth and not args.no_garden:
        from .garden import Garden
        garden = Garden(growth, dots.rows, dots.top)
    memory = None if (args.no_memory or not args.talk or not faces) else Memory(robot=args.name)
    language = None if (args.no_language or not args.talk) else Language()
    if faces:  # 見分けた人ごとに、なかよし度と記憶を切りかえる
        def call_name(pid, name):
            return (faces.person(pid) or {}).get("call") or name if pid else None
        if growth:
            growth.greet_on_switch = greeter is None  # あいさつは顔をおぼえる機能にまかせる
            faces.on_change(lambda pid, name: growth.set_person(pid, call_name(pid, name)))
        if language:
            faces.on_change(lambda pid, name: language.set_person(pid, call_name(pid, name)))
            language.set_person(*faces.current())
        if memory:
            faces.on_change(lambda pid, name: memory.set_person(pid, call_name(pid, name)))
            memory.set_person(*faces.current())
            memory.start()

    arms, arm_link = start_arms(args)

    caption = None
    if args.talk and not args.no_caption:
        from .caption import Caption
        caption = Caption(cw)
        if not caption.ok:
            print("字幕: 日本語のフォントがないので出しません（sudo apt install fonts-noto-cjk）")
            caption = None

    last_face = [0.0]
    conv = None
    pending_caption = None
    if args.talk:
        conv = start_conversation(args, tracker, last_face, growth, faces, memory, language)

    def react(reactions):  # 成長からの反応を顔と声に出す
        for i, (emo, dur, text) in enumerate(reactions):
            engine.express(emo, dur, now, interrupt=i == 0)
            if text and conv:
                conv.say(text)

    show_cam, debug = False, args.debug
    shown = 0  # ←→ キーで出している表情の番号
    talk = 0.0
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
                elif pygame.K_1 <= ev.key <= pygame.K_9:
                    engine.express(EMOTION_NAMES[ev.key - pygame.K_1], 4.0, now)
                elif ev.key in (pygame.K_RIGHT, pygame.K_LEFT):
                    step = 1 if ev.key == pygame.K_RIGHT else -1
                    shown = (shown + step) % len(EMOTION_NAMES)
                    engine.express(EMOTION_NAMES[shown], 4.0, now)
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
                    if arms:
                        arms.nuzzle()
                    if growth:
                        react(growth.on_stroke())
                    drag = -1e9  # 1回のなでで1度だけ
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                if drag is not None and 0 <= drag < S * 0.05:
                    engine.on_tap(now)
                    if growth:
                        react(growth.on_tap())
                drag = None

        present, fx, fy = False, 0.0, 0.0
        if tracker:
            present, fx, fy = tracker.get()
            fx, fy = fx * args.gaze_gain, fy * args.gaze_gain + args.gaze_offset_y
        elif args.mouse and pygame.mouse.get_focused():
            mx, my = pygame.mouse.get_pos()
            fx, fy = screen_to_face((mx - sw / 2) / S * 2, (my - sh / 2) / S * 2,
                                    args.screen_rotate)
            present = abs(fx) <= 1 and abs(fy) <= 1
        if present:
            if now - last_face[0] > 30.0:
                print("[顔] 見えた")  # 返事をする・しないの理由がログで分かるように
            last_face[0] = now
        engine.on_face(present, now)
        if growth:
            react(growth.on_face(present, dt))
        if listener:
            for emo, dur in listener.poll():
                engine.express(emo, dur, now)
        if greeter:  # 家族が来たら、名前を呼んであいさつ
            for emo, text in greeter.poll():
                engine.express(emo if emo in EMOTION_NAMES else "joy", 3.0, now)
                if arms:
                    arms.wave()
                if conv:
                    conv.say(text)
                else:
                    print(f"[あいさつ] {text}")
        if conv:
            for emo, dur in conv.poll():
                if emo == "gesture":  # 「バイバイ」「バンザイ」など
                    if arms:
                        arms.do(dur)
                    continue
                if emo == "sign":  # はい・いいえ
                    face.sign(dur)
                    if arms and dur == "yes":
                        arms.perk()
                    continue
                if emo == "caption":  # 字幕はイモの返事だけ
                    if caption:  # 答えが出た電球（頭の上）が見えるよう、少しおくらせて出す
                        pending_caption = (dur, now + 0.8)
                    continue
                if emo == "listen":  # 話し終わってすぐの考える顔（字幕はそのまま）
                    emo = "think"
                elif emo == "think" and caption:
                    caption.clear()  # 返事を考えはじめた: 前の返事の字幕で、考えている顔（上の …）が隠れないように
                if emo == "heard":
                    if engine.notice(now) and arms:
                        arms.perk()  # 眠っていたら、はっと手をあげる
                    if growth:
                        react(growth.on_talk())
                else:
                    engine.express(emo, dur, now)
                    engine.set_mood(emo, now)
            if pending_caption and now >= pending_caption[1]:
                caption.say(pending_caption[0], now)
                pending_caption = None
            talk += (conv.mouth_level() - talk) * 0.6
            face.set_talk(talk)
            if caption:
                if talk > 0.05 or (conv.enroll and conv.enroll.active):
                    caption.hold(now)  # 話している間と、顔の登録で答えを待っている間は消さない
                caption.update(now)

        emotion = engine.update(now)
        face.set_emotion(emotion)
        face.set_mood(*engine.mood(now))
        face.look_at((max(-1, min(1, fx)), max(-1, min(1, fy))) if present else None)
        face.update(dt)
        hands = (0.0, 0.0)
        if arms:
            hands = arms.update(dt, emotion, talk, engine.sleeping, present)
            arm_link.send(*hands)
        if garden:
            dots.draw(canvas, face, garden.update(dt), garden.keep)
        else:
            dots.draw(canvas, face)

        if caption:
            caption.draw(canvas)
        if show_cam and tracker:
            img = tracker.preview()
            if img is not None:
                surf = pygame.image.frombuffer(img.tobytes(), (img.shape[1], img.shape[0]), "RGB")
                w = S // 3
                canvas.blit(pygame.transform.smoothscale(
                    surf, (w, w * img.shape[0] // img.shape[1])), (8, 8))
        if debug:
            label = f"{emotion}  face={'yes' if present else 'no'}  {clock.get_fps():.0f}fps"
            if arms:
                label += f"  arms={hands[0]:.0f},{hands[1]:.0f}{'' if arm_link.fd is not None else ' (off)'}"
            if growth:
                label += f"  Lv{growth.level} {growth.current['points']}pt"
            if faces:
                label += f"  who={faces.current()[0] or '-'}"  # 日本語は既定のフォントで出ないので id
            canvas.blit(font.render(label, True, (200, 200, 200)), (8, ch - font.get_height() - 8))

        screen.fill(BG)
        out = pygame.transform.rotate(canvas, -args.screen_rotate) if args.screen_rotate else canvas
        screen.blit(out, ((sw - out.get_width()) // 2, (sh - out.get_height()) // 2))
        pygame.display.flip()

    if arm_link:
        arm_link.relax()
        arm_link.close()
    if memory:
        try:
            memory.stop()
        except OSError as e:
            print(f"記憶を保存できませんでした（{e}）")
    if faces:
        faces.stop()
    if tracker:
        tracker.stop()
    if growth:
        try:
            growth.save()
        except OSError as e:
            print(f"成長の記録を保存できませんでした（{e}）")
    pygame.quit()


if __name__ == "__main__":
    main()
