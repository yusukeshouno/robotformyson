"""ピコの「目」になるカメラを探して開く。USB カメラ（UVC）が第一候補。

想定: サンワサプライ 400-CAM103 などの USB Web カメラを Pi 4 につなぐ。
Pi には USB カメラ以外にも /dev/video* が並ぶ（動画の変換用 bcm2835-codec など）ので、
番号 0 決め打ちではなく、USB のカメラを名前で探す。

カメラの指定は コマンドの引数 > 環境変数 > 自動 の順:
    --camera-device 1 / /dev/video2 / CAM103   （番号・パス・名前の一部）
    ROBOT_CAMERA=CAM103
    ROBOT_CAMERA_SIZE=640x480   ROBOT_CAMERA_FPS=15   ROBOT_CAMERA_FORMAT=MJPG

Pi 4 向けの標準は 640x480・15fps・MJPG。顔を追うのにはこれで十分で、
残りの力を画面と音声認識に回せる。

カメラが届いたら、まずこれで確認:
    python3 -m robot.camera list      # カメラとマイクの一覧（どれを使うか）
    python3 -m robot.camera check     # 5秒うつして、速さ・顔が見えるかを調べて写真を保存
"""
import argparse
import glob
import os
import re
import sys
import time
from dataclasses import dataclass

ENV_CAMERA = "ROBOT_CAMERA"
ENV_SIZE = "ROBOT_CAMERA_SIZE"
ENV_FPS = "ROBOT_CAMERA_FPS"
ENV_FORMAT = "ROBOT_CAMERA_FORMAT"
DEFAULT_SIZE = (640, 480)
DEFAULT_FPS = 15
DEFAULT_FORMAT = "MJPG"   # YUYV より USB の帯域が少なくてすむ。うまく出ないときは YUYV

SYSFS = "/sys/class/video4linux"
BY_ID = "/dev/v4l/by-id"

# Pi の中にある「カメラではない」映像デバイス（動画の変換・画像処理など）
_INTERNAL = ("bcm2835-codec", "bcm2835-isp", "rpivid", "rpi-hevc", "pispbe", "hevc")
# CSI のカメラモジュール（picamera2 で使う）
_CSI = ("unicam", "rp1-cfe")

_CAMERA_WORDS = re.compile(r"camera|webcam|cam\d|\bcam\b|cam103|video", re.I)


def looks_like_camera(name):
    """デバイス名がカメラらしいか（カメラ内蔵マイクを見分けるのに使う）。"""
    return bool(_CAMERA_WORDS.search(name or ""))


@dataclass
class VideoDevice:
    path: str          # /dev/video0
    number: int        # 0
    name: str          # カメラが名乗る名前
    driver: str = ""   # uvcvideo など
    usb: bool = False
    by_id: str = ""    # /dev/v4l/by-id/...（差しなおしても変わらない名前）

    @property
    def kind(self):
        n = self.name.lower()
        if self.driver == "uvcvideo" or (self.usb and not any(k in n for k in _INTERNAL)):
            return "usb"
        if any(k in n or k in self.driver for k in _CSI):
            return "csi"
        return "internal"

    def open_path(self):
        return self.by_id or self.path


def _read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return ""


def list_devices(sysfs=SYSFS, by_id=BY_ID):
    """映像デバイスの一覧。1台のカメラが2つ出る（映像とメタデータ）ので、映像の方だけ。"""
    ids = {}
    for link in glob.glob(os.path.join(by_id, "*")):
        if link.endswith("-index0"):
            ids[os.path.realpath(link)] = link
    out = []
    for entry in glob.glob(os.path.join(sysfs, "video*")):
        m = re.fullmatch(r"video(\d+)", os.path.basename(entry))
        if not m:
            continue
        if _read(os.path.join(entry, "index")) not in ("", "0"):
            continue  # メタデータ用のノード
        dev = os.path.join(entry, "device")
        real = os.path.realpath(dev)
        driver = os.path.basename(os.path.realpath(os.path.join(dev, "driver")))
        path = f"/dev/video{m.group(1)}"
        out.append(VideoDevice(path=path, number=int(m.group(1)),
                               name=_read(os.path.join(entry, "name")),
                               driver=driver if os.path.exists(os.path.join(dev, "driver")) else "",
                               usb="/usb" in real, by_id=ids.get(os.path.realpath(path), "")))
    return sorted(out, key=lambda d: d.number)


def find_camera(spec=None, devices=None):
    """使う USB カメラを決める。見つからなければ None（カメラモジュールか、PC の 0 番を試す）。

    spec: None/"auto" → 最初の USB カメラ。数字 → /dev/videoN。/dev/... → そのパス。
          それ以外 → 名前の一部（大文字小文字は区別しない）。
    """
    devices = list_devices() if devices is None else devices
    spec = None if spec is None else str(spec).strip()
    if not spec or spec.lower() == "auto":
        usb = [d for d in devices if d.kind == "usb"]
        return usb[0] if usb else None
    if spec.isdigit():
        n = int(spec)
        return next((d for d in devices if d.number == n),
                    VideoDevice(f"/dev/video{n}", n, f"video{n}"))
    if spec.startswith("/"):
        real = os.path.realpath(spec)
        return next((d for d in devices if d.path in (spec, real) or d.by_id == spec),
                    VideoDevice(spec, -1, os.path.basename(spec)))
    want = spec.lower()
    for d in devices:
        if want in d.name.lower() or want in d.by_id.lower():
            return d
    names = "\n".join(f"  {d.path}: {d.name}（{d.kind}）" for d in devices) or "  （ありません）"
    raise RuntimeError(f"カメラ「{spec}」が見つかりません。あるもの:\n{names}")


def has_csi_camera(devices=None):
    devices = list_devices() if devices is None else devices
    return any(d.kind == "csi" for d in devices)


@dataclass
class CameraSettings:
    device: object = None          # None（自動）/ 番号 / パス / 名前
    width: int = DEFAULT_SIZE[0]
    height: int = DEFAULT_SIZE[1]
    fps: int = DEFAULT_FPS
    fourcc: str = DEFAULT_FORMAT

    @classmethod
    def load(cls, device=None, size=None, fps=None, fourcc=None, env=None):
        """引数 > 環境変数 > 標準 の順で決める。"""
        env = os.environ if env is None else env
        device = device if device not in (None, "") else (env.get(ENV_CAMERA) or None)
        size = size or env.get(ENV_SIZE) or ""
        w, h = DEFAULT_SIZE
        if size:
            m = re.fullmatch(r"\s*(\d+)\s*[xX×]\s*(\d+)\s*", str(size))
            if not m:
                raise ValueError(f"カメラの大きさ「{size}」は 640x480 のように書いてください")
            w, h = int(m.group(1)), int(m.group(2))
        fps = int(fps or env.get(ENV_FPS) or DEFAULT_FPS)
        fourcc = (fourcc or env.get(ENV_FORMAT) or DEFAULT_FORMAT).strip().upper()
        if fourcc not in ("MJPG", "YUYV", "ANY"):
            raise ValueError(f"カメラの形式「{fourcc}」は MJPG / YUYV / ANY のどれかにしてください")
        return cls(device, w, h, max(1, fps), fourcc)


def add_camera_args(ap):
    ap.add_argument("--camera-device", "--camera-index", dest="camera_device", default=None,
                    help=f"カメラの番号・パス・名前の一部（環境変数 {ENV_CAMERA}。標準は USB カメラを自動で探す）")
    ap.add_argument("--camera-size", default=None,
                    help=f"カメラの解像度（環境変数 {ENV_SIZE}。標準 640x480）")
    ap.add_argument("--camera-fps", type=int, default=None,
                    help=f"カメラのフレーム数（環境変数 {ENV_FPS}。標準 15）")
    ap.add_argument("--camera-format", default=None,
                    help=f"MJPG / YUYV / ANY（環境変数 {ENV_FORMAT}。標準 MJPG）")


def settings_from_args(args):
    return CameraSettings.load(getattr(args, "camera_device", None),
                               getattr(args, "camera_size", None),
                               getattr(args, "camera_fps", None),
                               getattr(args, "camera_format", None))


def open_opencv(settings, devices=None):
    """OpenCV で USB カメラを開く。(読む関数, 閉じる関数, 説明) を返す。"""
    import cv2
    dev = find_camera(settings.device, devices)
    if dev is not None and sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(dev.open_path(), cv2.CAP_V4L2)
        label = f"{dev.name} ({dev.path})"
    else:  # Mac・Windows や、Linux で見つからないときは 0 番（か指定の番号）
        n = dev.number if dev is not None and dev.number >= 0 else 0
        cap = cv2.VideoCapture(n)
        label = f"カメラ {n}"
    if not cap.isOpened():
        raise RuntimeError(f"カメラを開けませんでした（{label}）")
    if settings.fourcc != "ANY":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    cap.set(cv2.CAP_PROP_FPS, settings.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 古い画像をためない（目の動きが遅れないように）
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    label += f" {w}x{h} {fps:.0f}fps"

    def read():
        ok, frame = cap.read()
        return frame if ok else None
    return read, cap.release, label


# ---------------------------------------------------------------- 確認用のコマンド
def _audio_inputs():
    try:
        import sounddevice as sd
        return [(i, d["name"]) for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0]
    except Exception as e:  # sounddevice が無いなど
        return e


def cmd_list(settings):
    devices = list_devices()
    print("映像デバイス:")
    if not devices:
        print("  （ありません。USB をさしなおして、lsusb で見えるか確認してください）")
    for d in devices:
        tag = {"usb": "USB カメラ", "csi": "カメラモジュール", "internal": "Pi の内部（使わない）"}[d.kind]
        print(f"  {d.path}: {d.name}  [{tag}]" + (f"\n      {d.by_id}" if d.by_id else ""))
    try:
        cam = find_camera(settings.device, devices)
    except RuntimeError as e:
        print(f"\n{e}")
        return 1
    if cam:
        print(f"\n→ 目に使うカメラ: {cam.path}（{cam.name}）")
    elif has_csi_camera(devices):
        print("\n→ USB カメラが無いので、カメラモジュール（picamera2）を使います")
    else:
        print("\n→ 使えるカメラが見つかりません")
    inputs = _audio_inputs()
    print("\nマイク:")
    if isinstance(inputs, Exception):
        print(f"  （一覧を出せません: {inputs}）")
    else:
        for i, name in inputs:
            note = "  ← カメラのマイク（使わない）" if looks_like_camera(name) else ""
            print(f"  {i}: {name}{note}")
        print("  ロボットの耳は USB ミニマイクにします（ROBOT_MIC で名前を指定。"
              "名前があいまいでもカメラのマイクは選びません）")
    return 0


def cmd_check(settings, seconds=5.0, out=None):
    import cv2
    from .vision import find_cascade
    out = out or os.path.expanduser("~/camera_check.jpg")
    try:
        read, close, label = open_opencv(settings)
    except RuntimeError as e:
        print(f"NG: {e}")
        print("  → python3 -m robot.camera list でカメラが見えるか確認してください")
        return 1
    print(f"カメラ: {label}")
    cascade = cv2.CascadeClassifier(find_cascade())
    print(f"{seconds:.0f}秒うつします。カメラの前に顔を出してください…")
    frames = faces = fails = 0
    shape, best, best_box = None, None, None
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        frame = read()
        if frame is None:
            fails += 1
            if fails > 50:
                break
            time.sleep(0.02)
            continue
        frames += 1
        shape = frame.shape
        small = cv2.resize(frame, (320, 320 * frame.shape[0] // frame.shape[1]))
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        found = cascade.detectMultiScale(gray, 1.15, 5, minSize=(32, 32))
        if len(found):
            faces += 1
            s = frame.shape[1] / 320
            best_box = [int(v * s) for v in max(found, key=lambda f: f[2] * f[3])]
            best = frame
        elif best_box is None:
            best = frame
    elapsed = time.monotonic() - t0
    close()
    if not frames:
        print("NG: 画像が1枚も来ませんでした")
        print("  → ROBOT_CAMERA_FORMAT=YUYV python3 -m robot.camera check を試してください")
        return 1
    fps = frames / elapsed
    brightness = float(cv2.cvtColor(best, cv2.COLOR_BGR2GRAY).mean())
    img = best.copy()
    if best_box:
        x, y, w, h = best_box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imwrite(out, img)
    ok = True
    print(f"大きさ: {shape[1]}x{shape[0]}  速さ: {fps:.1f}fps  明るさ: {brightness:.0f}/255")
    if fps < max(5, settings.fps * 0.5):
        ok = False
        print("  △ 遅いです。部屋が暗いとカメラが自動で遅くなります。明るくするか、"
              "ROBOT_CAMERA_FORMAT=YUYV も試してください")
    if brightness < 50:
        print("  △ 暗いです。顔が見つけにくくなります")
    print(f"顔が見えた割合: {faces}/{frames}" + ("" if faces else "  （顔が写っていません）"))
    print(f"写真: {out}")
    print("OK: ピコの目に使えます" if ok and faces else
          "カメラは動いています" + ("。顔を写してもう一度どうぞ" if not faces else ""))
    return 0


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    add_camera_args(common)
    ap = argparse.ArgumentParser(description="ピコの目（カメラ）の確認")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list", parents=[common], help="カメラとマイクの一覧")
    c = sub.add_parser("check", parents=[common], help="数秒うつして、速さ・顔が見えるかを調べる")
    c.add_argument("--seconds", type=float, default=5.0)
    c.add_argument("--out", default=None, help="写真の保存先（標準 ~/camera_check.jpg）")
    args = ap.parse_args(argv)
    try:
        settings = settings_from_args(args)
    except ValueError as e:
        print(e)
        return 2
    if args.cmd == "check":
        return cmd_check(settings, args.seconds, args.out)
    return cmd_list(settings)


if __name__ == "__main__":
    sys.exit(main())
