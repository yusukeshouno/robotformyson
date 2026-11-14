"""カメラで顔を見つけ、その位置を -1..1 の座標で返す（別スレッドで動く）。

USB カメラ（Pi なら名前で探す。robot/camera.py）を第一に、なければカメラモジュール
（picamera2）、それも無ければ PC の 0 番カメラを OpenCV で使う。
USB を抜いても、さしなおせば自動でつなぎなおす。
"""
import os
import threading
import time

import cv2

from .camera import CameraSettings, find_camera, open_opencv

_CASCADE = "haarcascade_frontalface_default.xml"


def find_cascade():
    dirs = []
    if hasattr(cv2, "data"):
        dirs.append(cv2.data.haarcascades)
    dirs += ["/usr/share/opencv4/haarcascades", "/usr/share/opencv/haarcascades"]
    for d in dirs:
        path = os.path.join(d, _CASCADE)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{_CASCADE} が見つかりません")


class FaceTracker:
    def __init__(self, source="auto", settings=None, mirror=True, rotate=0,
                 detect_width=320):
        self.source = source
        self.settings = settings or CameraSettings.load()
        self.mirror, self.rotate = mirror, rotate
        self.width, self.height = self.settings.width, self.settings.height
        self.detect_width = detect_width
        self._full = None
        self.backend = None
        self.label = ""
        self._lock = threading.Lock()
        self._running = False
        self._pos = (0.0, 0.0)
        self._last_seen = 0.0
        self._frame = None
        self._box = None
        self.cascade = cv2.CascadeClassifier(find_cascade())

    def start(self):
        self._read, self._close = self._open()
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def stop(self):
        self._running = False

    def _open(self):
        # 自動のときは、USB カメラがあればそれを先に（カメラモジュールより優先）
        usb = None
        if self.source in ("auto", "opencv"):
            usb = find_camera(self.settings.device)
        if self.source == "picamera" or (self.source == "auto" and usb is None):
            try:
                from picamera2 import Picamera2
                cam = Picamera2()
                cam.configure(cam.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}))
                cam.start()
                self.backend = self.label = "picamera2"
                return cam.capture_array, cam.stop
            except Exception:
                if self.source == "picamera":
                    raise
        read, close, self.label = open_opencv(self.settings)
        self.backend = "opencv"
        return read, close

    def _reopen(self):
        """USB が抜けたあと、さしなおされたらつなぎなおす。"""
        try:
            self._close()
        except Exception:
            pass
        try:
            self._read, self._close = self._open()
            print(f"カメラにつなぎなおしました: {self.label}")
            return True
        except Exception:
            self._read, self._close = (lambda: None), (lambda: None)
            return False

    def _loop(self):
        rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
               270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(self.rotate)
        misses, retry_at = 0, 0.0
        while self._running:
            frame = self._read()
            if frame is None:
                misses += 1
                now = time.monotonic()
                if misses > 40 and now >= retry_at:  # 2秒ほど何も来ない＝抜けたかも
                    if self._reopen():
                        misses = 0
                    retry_at = now + 3.0
                time.sleep(0.05)
                continue
            misses = 0
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if rot is not None:
                frame = cv2.rotate(frame, rot)
            full = frame
            if frame.shape[1] > self.detect_width:  # 検出は小さい画像で（速さのため）
                scale = self.detect_width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            fh, fw = gray.shape
            faces = self.cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=5,
                minSize=(int(fw * 0.1), int(fw * 0.1)))
            box = None
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                nx = (x + w / 2) / fw * 2 - 1
                ny = (y + h / 2) / fh * 2 - 1
                if self.mirror:  # カメラの左に写った人 = 画面上では右を見る
                    nx = -nx
                box = (x, y, w, h)
                with self._lock:
                    self._pos = (nx, ny)
                    self._last_seen = time.monotonic()
            with self._lock:
                self._frame, self._box, self._full = frame, box, full
        self._close()

    def get(self):
        """(顔がいるか, x, y) を返す。一瞬見失っても 0.6 秒は「いる」扱い。"""
        with self._lock:
            present = time.monotonic() - self._last_seen < 0.6
            return present, self._pos[0], self._pos[1]

    def latest(self):
        """今のカメラ画像（回転済み・元の大きさの BGR）。顔の見分けなどで使う。"""
        with self._lock:
            return None if self._full is None else self._full.copy()

    def jpeg(self, quality=80):
        """今のカメラ画像を JPEG で返す（会話で「これなに？」と聞かれたとき用）。"""
        with self._lock:
            img = self._full
        if img is None:
            return None
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    def preview(self):
        """デバッグ用のカメラ映像（RGB の numpy 配列）。"""
        with self._lock:
            if self._frame is None:
                return None
            img = self._frame.copy()
            box = self._box
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
