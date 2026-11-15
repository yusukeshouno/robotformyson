"""家族の顔をおぼえて、だれが来たかを見分ける。

すべて Pi の中だけで動く。顔の写真は保存せず、顔の特徴（128個の数字）だけを
~/.local/share/robot/faces.json に保存する。ネットワークには何も送らない。

顔の検出は YuNet、見分けは SFace（どちらも OpenCV に入っている小さなモデル）。
1秒に1回くらい調べるだけなので、Pi 4 でも顔追従の動きは重くならない。

    python3 -m robot.faces download          # モデルを models/ に入れる（最初に1回）
    python3 -m robot.faces add ゆうと         # カメラを見てもらって登録
    python3 -m robot.faces list              # 登録した人の一覧
    python3 -m robot.faces set ゆうと --call ゆうとくん --emotion love
    python3 -m robot.faces forget ゆうと      # 消す（forget --all で全部）

ほかの機能から「いま見えている人」を使うとき:

    rec.current()                 # -> (person_id, name) / (None, None)
    rec.on_change(callback)       # 見えている人が変わったら callback(person_id, name)

person_id は "p1" のような、登録したときに決まって変わらない文字列。name は呼び名。
"""
import argparse
import hashlib
import json
import os
import re
import threading
import time
import unicodedata
import urllib.request
from collections import deque

import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
# 古い OpenCV（4.8 より前。Pi OS Bookworm は 4.6）は 2023mar を読めないので、消される前の 2022mar を使う。
_ZOO_2022 = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/"
             "7e062e54cf5410c09b795ff71b4a255e58498c79/models/")
MODELS = {  # ファイル名: (取ってくる場所, sha256)
    "face_detection_yunet_2022mar.onnx": (
        _ZOO_2022 + "face_detection_yunet/face_detection_yunet_2022mar.onnx",
        "50ef07f702a31741ca46a4c0d947773b64143b9362780237bf0d427d6c79bab7"),
    "face_detection_yunet_2023mar.onnx": (
        _ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "face_recognition_sface_2021dec.onnx": (
        _ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}
YUNET_OLD, YUNET, SFACE = list(MODELS)


def yunet_for(cv2_version):
    """この OpenCV で動く顔検出モデルの名前。4.8 より前なら古い 2022mar。"""
    try:
        major, minor = (int(x) for x in cv2_version.split(".")[:2])
    except ValueError:
        return YUNET
    return YUNET_OLD if (major, minor) < (4, 8) else YUNET

MATCH = 0.40        # これ以上似ていたら同じ人（SFace のコサイン類似度。OpenCV の目安は 0.363）
SAME_AS_OTHER = 0.50  # 登録しようとした顔が、別の人にこれ以上似ていたら登録しない
MAX_SAMPLES = 20    # 1人あたり覚えておく顔の数（古いものから捨てる）


def default_db_path():
    base = os.environ.get("ROBOT_DATA") or os.path.join(
        os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), "robot")
    return os.path.join(base, "faces.json")


def _key(name):
    """名前をくらべやすくする（全角半角・カタカナひらがなの違いをなくす）。"""
    t = unicodedata.normalize("NFKC", name).strip().lower()
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in t)


def _unit(v):
    v = np.asarray(v, np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


# ---------------------------------------------------------------- 保存
class FaceDB:
    """登録した人と、その顔の特徴。ファイルは本人（Pi のユーザー）だけが読める。"""

    def __init__(self, path=None):
        self.path = path or default_db_path()
        self.people = {}
        self.next_id = 1   # 消した人の id は使いまわさない（成長記録などがまざらないように）
        self._lock = threading.Lock()
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self.people = data.get("people", {})
            self.next_id = data.get("next_id", 1)
        nums = [int(k[1:]) for k in self.people if k[1:].isdigit()]
        self.next_id = max([self.next_id] + [n + 1 for n in nums])

    def save(self):
        with self._lock:
            data = json.dumps({"version": 1, "next_id": self.next_id, "people": self.people},
                              ensure_ascii=False)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, self.path)
        os.chmod(self.path, 0o600)

    def find(self, name_or_id):
        if name_or_id in self.people:
            return name_or_id
        k = _key(name_or_id)
        for pid, p in self.people.items():
            if _key(p["name"]) == k or _key(p.get("call", "")) == k:
                return pid
        return None

    def get(self, pid):
        return self.people.get(pid)

    def add(self, name, embeddings, now=None):
        """顔を登録する。同じ名前の人がいれば、その人の顔として足す。person_id を返す。"""
        now = time.time() if now is None else now
        embs = [_unit(e).round(5).tolist() for e in embeddings]
        with self._lock:
            pid = self.find(name)
            if pid is None:
                pid = f"p{self.next_id}"
                self.next_id += 1
                self.people[pid] = {"name": name, "call": name, "emotion": "joy",
                                    "added": now, "last_seen": None, "embeddings": []}
            p = self.people[pid]
            p["embeddings"] = (p["embeddings"] + embs)[-MAX_SAMPLES:]
        return pid

    def update(self, pid, **fields):
        with self._lock:
            self.people[pid].update(fields)

    def forget(self, pid):
        with self._lock:
            return self.people.pop(pid, None) is not None

    def forget_all(self):
        with self._lock:
            self.people = {}

    def match(self, emb):
        """いちばん似ている人を (person_id, 似ている度合い) で返す。だれもいなければ (None, 0)。"""
        emb = _unit(emb)
        best, score = None, 0.0
        with self._lock:
            for pid, p in self.people.items():
                if not p["embeddings"]:
                    continue
                s = float(np.max(np.asarray(p["embeddings"], np.float32) @ emb))
                if s > score:
                    best, score = pid, s
        return best, score


# ---------------------------------------------------------------- 顔の特徴
class Embedder:
    """画像から顔を見つけ（YuNet）、顔の特徴を取り出す（SFace）。"""

    def __init__(self, model_dir=MODEL_DIR, detect_width=320):
        import cv2
        paths = [os.path.join(model_dir, n) for n in (yunet_for(cv2.__version__), SFACE)]
        for p in paths:
            if not os.path.exists(p):
                raise RuntimeError(f"顔を見分けるモデルがありません: {p}"
                                   "（python3 -m robot.faces download）")
        self.cv2 = cv2
        self.detect_width = detect_width
        self.detector = cv2.FaceDetectorYN.create(paths[0], "", (320, 320), 0.8)
        self.recognizer = cv2.FaceRecognizerSF.create(paths[1], "")

    def faces(self, frame):
        """見えている顔を大きい順に返す（座標は元の画像のもの）。"""
        h, w = frame.shape[:2]
        scale = min(1.0, self.detect_width / w)
        small = self.cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, found = self.detector.detect(small)
        if found is None:
            return []
        found = found.copy()
        found[:, :14] /= scale
        return sorted(found, key=lambda f: -f[2] * f[3])

    def embed(self, frame, face):
        aligned = self.recognizer.alignCrop(frame, face)
        return _unit(self.recognizer.feature(aligned))


# ---------------------------------------------------------------- 見分ける
class FaceRecognizer:
    """カメラの画像をときどき調べて、いま見えている人を決める（別スレッドで動く）。

    一度の見まちがいで呼び名がころころ変わらないよう、直近3回のうち2回同じ人だったら
    その人にする。顔が見えなくなって forget_after 秒たったら「だれもいない」にする。
    """

    def __init__(self, db, embedder, interval=1.0, forget_after=4.0, min_face=60,
                 threshold=MATCH):
        self.db, self.embedder = db, embedder
        self.interval, self.forget_after = interval, forget_after
        self.min_face, self.threshold = min_face, threshold
        self._lock = threading.Lock()
        self._votes = deque(maxlen=3)
        self._current = None
        self._last_known = 0.0
        self._listeners = []
        self._enroll = None
        self._enroll_result = None
        self.enrolled_at = {}   # person_id -> 登録した時刻（time.time）
        self._running = False

    # ---- ほかの機能から使う ----
    def current(self):
        """(person_id, name)。だれもいない・知らない人なら (None, None)。"""
        with self._lock:
            pid = self._current
        p = self.db.get(pid) if pid else None
        return (pid, p["name"]) if p else (None, None)

    def person(self, pid):
        """登録内容（name, call, emotion, last_seen など）。顔の特徴は含めない。"""
        p = self.db.get(pid)
        return {k: v for k, v in p.items() if k != "embeddings"} if p else None

    def on_change(self, callback):
        """見えている人が変わったら callback(person_id, name) を呼ぶ（だれもいなければ None, None）。

        callback はカメラのスレッドから呼ばれるので、重い処理はしないこと。
        """
        self._listeners.append(callback)

    # ---- 動かす ----
    def start(self, get_frame):
        """get_frame() -> 今のカメラ画像（BGR）/ None。"""
        self._running = True

        def loop():
            while self._running:
                t0 = time.monotonic()
                frame = get_frame()
                if frame is not None:
                    try:
                        self.step(frame, t0)
                    except Exception as e:  # 1回失敗しても続ける
                        print(f"[顔] エラー: {e}")
                wait = (0.25 if self._enroll else self.interval) - (time.monotonic() - t0)
                time.sleep(max(0.05, wait))
        threading.Thread(target=loop, daemon=True).start()
        return self

    def stop(self):
        self._running = False

    def step(self, frame, now):
        """1枚の画像を調べる。テストからも直接呼べる。"""
        faces = [f for f in self.embedder.faces(frame) if f[2] >= self.min_face]
        if self._enroll is not None:
            self._enroll_step(frame, faces, now)
        if not faces:
            self._decide(None, now, seen=False)
            return
        emb = self.embedder.embed(frame, faces[0])  # いちばん近く（大きく写った）の人
        pid, score = self.db.match(emb)
        self._decide(pid if score >= self.threshold else None, now, seen=True)

    def _decide(self, pid, now, seen):
        if seen:
            self._votes.append(pid)
        new = self._current
        if pid is not None and (pid == self._current or list(self._votes).count(pid) >= 2):
            new = pid
            self._last_known = now
        if new is not None and now - self._last_known > self.forget_after:
            new = None
            self._votes.clear()
        if new == self._current:
            return
        old = self._current
        with self._lock:
            self._current = new
        name = self.db.get(new)["name"] if new and self.db.get(new) else None
        for cb in list(self._listeners):
            try:
                cb(new, name)
            except Exception as e:
                print(f"[顔] on_change エラー: {e}")
        for p in (old, new):  # 最後に見た時刻を残す（成長記録などで使える）
            if p and self.db.get(p):
                self.db.update(p, last_seen=time.time())
        try:
            self.db.save()
        except OSError as e:
            print(f"[顔] 保存できませんでした: {e}")

    # ---- 登録 ----
    def enroll(self, name, samples=5, timeout=8.0):
        """name の顔をおぼえる。カメラを見てもらっている間に samples 枚集める。

        結果は enroll_result() で取る: ("ok", person_id) / ("noface", None) /
        ("other", 似ている人の person_id)
        """
        done = threading.Event()
        with self._lock:
            self._enroll = {"name": name, "need": samples, "embs": [], "until": None,
                            "timeout": timeout, "done": done, "result": None, "last": 0.0}
        return done

    def enroll_progress(self):
        e = self._enroll
        return (len(e["embs"]), e["need"]) if e else None

    def enroll_result(self):
        return self._enroll_result

    def wait_enroll(self, timeout=None):
        e = self._enroll
        if e is None:
            return self.enroll_result()
        e["done"].wait(timeout if timeout is not None else e["timeout"] + 2)
        return self.enroll_result()

    def _enroll_step(self, frame, faces, now):
        e = self._enroll
        if e["until"] is None:
            e["until"] = now + e["timeout"]
        if len(faces) == 1 and now - e["last"] >= 0.2:  # 1人だけ写っているときに集める
            e["embs"].append(self.embedder.embed(frame, faces[0]))
            e["last"] = now
        if len(e["embs"]) >= e["need"]:
            self._finish_enroll(self._save_enroll(e))
        elif now > e["until"]:
            self._finish_enroll(("noface", None))

    def _save_enroll(self, e):
        own = self.db.find(e["name"])
        votes = {}
        for emb in e["embs"]:
            pid, score = self.db.match(emb)
            if pid and pid != own and score >= SAME_AS_OTHER:
                votes[pid] = votes.get(pid, 0) + 1
        if votes:
            other = max(votes, key=votes.get)
            if votes[other] * 2 > len(e["embs"]):  # 半分以上が別の人にそっくり
                return ("other", other)
        pid = self.db.add(e["name"], e["embs"])
        self.db.save()
        self.enrolled_at[pid] = time.time()
        self._votes.clear()
        return ("ok", pid)

    def _finish_enroll(self, result):
        e = self._enroll
        self._enroll_result = result
        self._enroll = None
        e["done"].set()


# ---------------------------------------------------------------- あいさつ
class Greeter:
    """登録した人が来たら、あいさつの言葉と表情を決める。

    同じ人には GREET_AGAIN 秒（30分）たつまで、もう一度はあいさつしない。
    """

    GREET_AGAIN = 30 * 60
    WELCOME_BACK = 2 * 60 * 60   # これより長く会っていなければ「おかえり」

    def __init__(self, recognizer, clock=time.time, hour=None):
        self.rec = recognizer
        self.clock = clock
        self.hour = hour or (lambda: time.localtime(self.clock()).tm_hour)
        self._greeted = {}
        self._left = {}       # person_id -> 見えなくなった時刻
        self._present = None
        self.events = deque()
        recognizer.on_change(self._on_change)

    def _on_change(self, pid, name):
        now = self.clock()
        if self._present:
            self._left[self._present] = now
        self._present = pid
        if pid is None:
            return
        p = self.rec.person(pid) or {}
        last_seen = self._left.get(pid, p.get("last_seen"))
        greeted = max(self._greeted.get(pid, -1e12), self.rec.enrolled_at.get(pid, -1e12))
        if now - greeted < self.GREET_AGAIN:  # さっきあいさつした・いま登録したばかり
            return
        self._greeted[pid] = now
        self.events.append(self.greeting(p, last_seen, now))

    def greeting(self, p, last_seen, now):
        """(感情, あいさつ) を返す。"""
        call = p.get("call") or p.get("name", "")
        emotion = p.get("emotion") or "joy"
        h = self.hour()
        away = None if last_seen is None else now - last_seen
        if 5 <= h < 10:
            text = f"おはよう、{call}！"
        elif away is None or away >= self.WELCOME_BACK:
            text = f"おかえり、{call}！"
        elif h >= 19:
            text = f"{call}、こんばんは！"
        else:
            text = f"あ、{call}だ！ また あえたね。"
        return emotion, text

    def poll(self):
        out = []
        while self.events:
            out.append(self.events.popleft())
        return out


# ---------------------------------------------------------------- 声で登録するときの言葉
_ENROLL_FACE = ("かお", "顔")
_ENROLL_REMEMBER = ("おぼえ", "覚え", "とうろく", "登録")
_YES = ("うん", "はい", "いいよ", "そう", "あってる", "あたり", "おっけー", "ok")
_NO = ("ちがう", "違う", "いいえ", "ううん", "だめ", "ちゃう")
_NAME_PREFIX = re.compile(r"^(えっと|えーと|あのね|ぼくの|わたしの|おれの|うちの)?"
                          r"(なまえは|名前は|ぼくは|わたしは|おれは|僕は|私は|俺は)?")
_NAME_SUFFIX = re.compile(r"(っていうの|っていうよ|っていいます|といいます|です|だよ|だ|よ|ね|)$")


def wants_enroll(text):
    t = _key(text)
    return any(_key(w) in t for w in _ENROLL_FACE) and any(_key(w) in t for w in _ENROLL_REMEMBER)


def is_yes(text):
    t = _key(text)
    return any(_key(w) in t for w in _YES) and not is_no(text)


def is_no(text):
    t = _key(text)
    return any(_key(w) in t for w in _NO)


def extract_name(text):
    """「ぼくは ゆうと だよ」→「ゆうと」。名前らしくなければ None。"""
    t = unicodedata.normalize("NFKC", text).replace(" ", "").replace("　", "")
    t = _NAME_PREFIX.sub("", t)
    t = _NAME_SUFFIX.sub("", t)
    if not t or len(t) > 8 or not re.fullmatch(r"[ぁ-んァ-ヶー一-龯a-zA-Z]+", t):
        return None
    return t


_CANCEL = ("やめる", "やめて", "やっぱりいい", "おしまい", "ストップ")


class EnrollDialog:
    """声で顔を登録する、子ども向けのやりとり。

        子「ピコ、かおを おぼえて」 → 「いいよ！ おなまえ おしえて？」
        子「ゆうと」               → 「ゆうと、で あってる？」
        子「うん」                 → 「ゆうと、カメラを みて にっこりしてね！」（顔を集める）
                                   → 「おぼえたよ！ よろしくね、ゆうと！」

    handle(text, speak) は、このやりとりの途中なら返事を speak(感情, 言葉) で言って True を返す。
    """

    TIMEOUT = 40.0   # この秒数だまっていたら、やりとりをやめる
    MAX_UNCLEAR = 3  # 聞き取れないのがこの回数つづいたら、やめる

    def __init__(self, recognizer, clock=time.monotonic, show=None):
        self.rec, self.clock = recognizer, clock
        self.show = show   # show(字幕)。撮っている途中の進み具合を画面に出す（スピーカーがなくても分かる）
        self.state, self.name, self.tries, self.last = None, None, 0, 0.0
        self.unclear = 0

    @property
    def active(self):
        if self.state and self.clock() - self.last > self.TIMEOUT:
            self.state = None
        return self.state is not None

    def handle(self, text, speak):
        self.unclear = 0
        if not self.active:
            if not wants_enroll(text):
                return False
            self._go("name")
            speak("excited", "いいよ！ おなまえ おしえて？")
            return True
        self.last = self.clock()
        if wants_enroll(text):  # もう一度「かおを おぼえて」: 名前とまちがえず、はじめから
            self._go("name")
            self.tries = 0
            speak("excited", "いいよ！ おなまえ おしえて？")
            return True
        if any(_key(w) in _key(text) for w in _CANCEL):
            self.state = None
            speak("neutral", "わかった、また こんどね。")
        elif self.state == "name":
            self._on_name(text, speak)
        elif self.state == "confirm":
            if is_yes(text):
                self._capture(speak)
            elif is_no(text):
                self._go("name")
                speak("confused", "ごめんね。もういっかい おなまえ いって？")
            else:
                speak("neutral", f"{self.name}、で あってたら「うん」って いってね。")
        return True

    def on_unclear(self, speak):
        """声はしたが聞き取れなかった。何回か聞きなおし、だめならやめる。"""
        self.unclear += 1
        if self.unclear >= self.MAX_UNCLEAR:
            self.state, self.unclear = None, 0
            speak("sad", "ごめんね、うまく きこえなかった。また あとで やろうね。")
        else:
            self.last = self.clock()
            speak("confused", "ん？ きこえなかった。もういっかい いって？")

    def _go(self, state):
        if state == "name" and self.state != "name":
            self.tries = 0
        self.state, self.last = state, self.clock()

    def _on_name(self, text, speak):
        name = extract_name(text)
        if name is None:
            self.tries += 1
            if self.tries >= 3:
                self.state = None
                speak("sad", "ごめんね、うまく きこえなかった。また あとで やろうね。")
            else:
                speak("confused", "もういっかい、おなまえ だけ いって？")
            return
        self.name = name
        self._go("confirm")
        speak("joy", f"{name}、で あってる？")

    def _wait_with_progress(self, name, done):
        """撮りおわるまで待つ。そのあいだ「カメラを みてね ●●○○○」を出す。"""
        progress = getattr(self.rec, "enroll_progress", None)
        if self.show is None or progress is None or done is None:
            return self.rec.wait_enroll() or ("noface", None)
        shown, until = None, time.monotonic() + 15
        while not done.wait(0.3) and time.monotonic() < until:
            p = progress()
            if p and p != shown:
                shown = p
                self.show(f"{name}、カメラを みてね {'●' * p[0]}{'○' * (p[1] - p[0])}")
        return self.rec.enroll_result() or ("noface", None)

    def _capture(self, speak):
        name, self.state = self.name, None
        speak("excited", f"{name}、カメラを みて にっこりしてね！")
        done = self.rec.enroll(name)
        result = self._wait_with_progress(name, done)
        if result[0] == "ok":
            speak("love", f"おぼえたよ！ よろしくね、{name}！")
        elif result[0] == "other":
            other = self.rec.person(result[1]) or {}
            speak("confused", f"あれ？ {other.get('call', 'だれか')}に そっくりだよ。"
                              "おうちの人と いっしょに やってみてね。")
        else:
            speak("sad", "おかおが よく みえなかったよ。あかるいところで、もういっかい やろうね。")


# ---------------------------------------------------------------- コマンド
def download(model_dir=MODEL_DIR):
    os.makedirs(model_dir, exist_ok=True)
    for name, (url, sha) in MODELS.items():
        path = os.path.join(model_dir, name)
        if os.path.exists(path) and _sha256(path) == sha:
            print(f"ある: {name}")
            continue
        print(f"ダウンロード中: {name}")
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        if _sha256(tmp) != sha:
            os.remove(tmp)
            raise RuntimeError(f"{name} の中身がちがいます（ダウンロードしなおしてください）")
        os.replace(tmp, path)
    print("OK")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description="家族の顔をおぼえる")
    ap.add_argument("--db", default=None, help="保存ファイル（ふつうは指定しない）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="顔を見分けるモデルを models/ に入れる")
    sub.add_parser("list", help="登録した人の一覧")
    a = sub.add_parser("add", help="カメラを見てもらって登録する")
    a.add_argument("name")
    from .camera import add_camera_args
    add_camera_args(a)
    a.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0)
    s = sub.add_parser("set", help="呼び方・あいさつの表情を変える")
    s.add_argument("name")
    s.add_argument("--call", help="呼びかけるときの名前（例: ゆうとくん、ママ）")
    s.add_argument("--rename", help="名前を変える")
    s.add_argument("--emotion", help="あいさつの表情（joy, love, excited など）")
    f = sub.add_parser("forget", help="登録を消す")
    f.add_argument("name", nargs="?")
    f.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "download":
        download()
        return
    db = FaceDB(args.db)
    if args.cmd == "list":
        if not db.people:
            print("まだだれも登録されていません")
        for pid, p in db.people.items():
            seen = (time.strftime("%m/%d %H:%M", time.localtime(p["last_seen"]))
                    if p.get("last_seen") else "-")
            print(f"{pid}\t{p['name']}\t呼び方:{p.get('call')}\t表情:{p.get('emotion')}"
                  f"\t顔:{len(p['embeddings'])}\t最後に見た:{seen}")
        print(f"（保存場所: {db.path}）")
        return
    if args.cmd == "add":
        from .camera import settings_from_args
        from .vision import FaceTracker
        tracker = FaceTracker("auto", settings_from_args(args), rotate=args.rotate).start()
        rec = FaceRecognizer(db, Embedder()).start(tracker.latest)
        print(f"{args.name} さん、カメラを見てください…")
        rec.enroll(args.name)
        result = rec.wait_enroll()
        tracker.stop()
        rec.stop()
        if result and result[0] == "ok":
            print(f"おぼえました（{result[1]}）")
        elif result and result[0] == "other":
            print(f"{db.get(result[1])['name']} さんにそっくりなので、登録しませんでした")
        else:
            print("顔がよく見えませんでした。明るいところで、1人だけ写るようにしてください")
        return
    pid = db.find(args.name) if args.name else None
    if args.cmd == "forget" and args.all:
        db.forget_all()
        db.save()
        print("全員の登録を消しました")
        return
    if pid is None:
        raise SystemExit(f"{args.name} は登録されていません")
    if args.cmd == "forget":
        db.forget(pid)
        print(f"{args.name} の登録を消しました")
    else:
        fields = {k: v for k, v in (("call", args.call), ("name", args.rename),
                                    ("emotion", args.emotion)) if v}
        db.update(pid, **fields)
        print(f"{pid}: {fields}")
    db.save()


if __name__ == "__main__":
    main()
