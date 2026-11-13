"""カメラで顔を見つけ、その位置を -1..1 の座標で返す（別スレッドで動く）。

Pi のカメラモジュールなら picamera2、それ以外（USB カメラ・PC）は OpenCV を使う。
"""
import os
import threading
import time

import cv2

_CASCADE = "haarcascade_frontalface_default.xml"


def _find_cascade():
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
    def __init__(self, source="auto", index=0, mirror=True, rotate=0,
                 width=640, height=480, detect_width=320):
        self.source, self.index = source, index
        self.mirror, self.rotate = mirror, rotate
        self.width, self.height = width, height
        self.detect_width = detect_width
        self._full = None
        self.backend = None
        self._lock = threading.Lock()
        self._running = False
        self._pos = (0.0, 0.0)
        self._last_seen = 0.0
        self._frame = None
        self._box = None
        self.cascade = cv2.CascadeClassifier(_find_cascade())

    def start(self):
        self._read, self._close = self._open()
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def stop(self):
        self._running = False

    def _open(self):
        if self.source in ("auto", "picamera"):
            try:
                from picamera2 import Picamera2
                cam = Picamera2()
                cam.configure(cam.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}))
                cam.start()
                self.backend = "picamera2"
                return cam.capture_array, cam.stop
            except Exception:
                if self.source == "picamera":
                    raise
        cap = cv2.VideoCapture(self.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            raise RuntimeError("カメラを開けませんでした")
        self.backend = "opencv"

        def read():
            ok, frame = cap.read()
            return frame if ok else None
        return read, cap.release

    def _loop(self):
        rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
               270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(self.rotate)
        while self._running:
            frame = self._read()
            if frame is None:
                time.sleep(0.05)
                continue
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
