import time
"""顔をおぼえる機能のテスト（カメラ・モデルがなくても動く）。

本物の顔の代わりに、FakeEmbedder が「画像に写っている人の番号」から決まった特徴を返す。
"""
import os
import stat
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from robot import conversation as cv
from robot import faces as fc

DIM = 128


def vec(person, noise=0.0, seed=0):
    """person ごとにほぼ直交する特徴。noise で同じ人の写りのちがいをまねる。"""
    v = np.random.default_rng(person).normal(size=DIM)
    if noise:
        v = v + np.random.default_rng(1000 + seed).normal(size=DIM) * noise
    return fc._unit(v)


def frame(*people, size=120):
    """写っている人の番号を持つにせの画像。"""
    return SimpleNamespace(people=list(people), size=size)


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def faces(self, img):
        return [np.array([0, 0, img.size, img.size, p], np.float32) for p in img.people]

    def embed(self, img, face):
        self.calls += 1
        return vec(int(face[4]), noise=0.3, seed=self.calls)


@pytest.fixture
def db(tmp_path):
    return fc.FaceDB(str(tmp_path / "robot" / "faces.json"))


@pytest.fixture
def rec(db):
    return fc.FaceRecognizer(db, FakeEmbedder(), forget_after=4.0)


def enroll_now(rec, name, person, t0=0.0):
    rec.enroll(name, samples=3)
    for i in range(10):
        if rec.enroll_progress() is None:
            break
        rec.step(frame(person), t0 + i * 0.3)
    return rec.enroll_result()


# ---------------------------------------------------------------- 保存
def test_db_add_match_and_save_private(db):
    pid = db.add("ユウト", [vec(1), vec(1, 0.3)])
    assert pid == "p1"
    assert db.find("ゆうと") == pid                # カタカナ・ひらがなの違いは同じ名前
    assert db.add("ゆうと", [vec(1, 0.3, 5)]) == pid  # 同じ名前なら同じ人に足す
    mama = db.add("ママ", [vec(2)])
    assert db.match(vec(1, 0.3, 9))[0] == pid
    assert db.match(vec(2, 0.3, 9))[0] == mama
    _, score = db.match(vec(3))
    assert score < fc.MATCH                        # 知らない人は似ていない
    db.save()
    mode = stat.S_IMODE(os.stat(db.path).st_mode)
    assert mode == 0o600                           # 本人以外は読めない
    again = fc.FaceDB(db.path)
    assert again.people[pid]["name"] == "ユウト" and len(again.people[pid]["embeddings"]) == 3


def test_db_ids_are_never_reused(db):
    a = db.add("あ", [vec(1)])
    db.forget(a)
    db.save()
    b = fc.FaceDB(db.path).add("い", [vec(2)])
    assert a == "p1" and b == "p2"


def test_db_keeps_only_recent_samples(db):
    pid = db.add("あ", [vec(1, 0.3, i) for i in range(fc.MAX_SAMPLES + 5)])
    assert len(db.people[pid]["embeddings"]) == fc.MAX_SAMPLES


def test_db_stores_no_images(db, rec):
    enroll_now(rec, "ゆうと", 1)
    with open(db.path, encoding="utf-8") as f:
        text = f.read()
    assert "jpeg" not in text.lower() and "image" not in text.lower()
    assert set(db.people["p1"]) == {"name", "call", "emotion", "added", "last_seen",
                                    "embeddings"}


# ---------------------------------------------------------------- 見分ける
def test_recognizer_needs_two_votes_and_reports_changes(db, rec):
    yuto = db.add("ゆうと", [vec(1)])
    mama = db.add("ママ", [vec(2)])
    seen = []
    rec.on_change(lambda pid, name: seen.append((pid, name)))

    rec.step(frame(1), 0.0)
    assert rec.current() == (None, None)           # 1回だけではまだ決めない
    rec.step(frame(1), 1.0)
    assert rec.current() == (yuto, "ゆうと")
    rec.step(frame(2), 2.0)                        # 一度の見まちがいでは変わらない
    assert rec.current()[0] == yuto
    rec.step(frame(2), 3.0)
    assert rec.current() == (mama, "ママ")
    assert seen == [(yuto, "ゆうと"), (mama, "ママ")]


def test_recognizer_forgets_after_face_leaves(db, rec):
    yuto = db.add("ゆうと", [vec(1)])
    seen = []
    rec.on_change(lambda pid, name: seen.append(pid))
    rec.step(frame(1), 0.0)
    rec.step(frame(1), 1.0)
    for t in (2.0, 3.0, 4.0):                      # 少し見えなくても、まだいる
        rec.step(frame(), t)
    assert rec.current()[0] == yuto
    rec.step(frame(), 5.5)
    assert rec.current() == (None, None)
    assert seen == [yuto, None]
    assert db.get(yuto)["last_seen"] is not None   # 最後に見た時刻が残る


def test_unknown_face_is_nobody(db, rec):
    db.add("ゆうと", [vec(1)])
    for t in range(4):
        rec.step(frame(7), float(t))
    assert rec.current() == (None, None)


def test_largest_face_wins_and_small_faces_are_ignored(db, rec):
    db.add("ゆうと", [vec(1)])
    rec.step(frame(1, size=30), 0.0)               # 遠くて小さい顔は見分けない
    rec.step(frame(1, size=30), 1.0)
    assert rec.current() == (None, None)


def test_person_info_has_no_embeddings(db, rec):
    pid = db.add("ゆうと", [vec(1)])
    assert "embeddings" not in rec.person(pid)
    assert rec.person("p99") is None


def test_listener_errors_do_not_stop_recognition(db, rec):
    db.add("ゆうと", [vec(1)])
    rec.on_change(lambda *a: 1 / 0)
    ok = []
    rec.on_change(lambda pid, name: ok.append(pid))
    rec.step(frame(1), 0.0)
    rec.step(frame(1), 1.0)
    assert ok == ["p1"]


def test_background_thread_uses_latest_frame(db):
    db.add("ゆうと", [vec(1)])
    rec = fc.FaceRecognizer(db, FakeEmbedder(), interval=0.01)
    changed = threading.Event()
    rec.on_change(lambda pid, name: changed.set())
    rec.start(lambda: frame(1))
    try:
        assert changed.wait(2.0)
        assert rec.current()[1] == "ゆうと"
    finally:
        rec.stop()


# ---------------------------------------------------------------- 登録
def test_enroll_ok(db, rec):
    result = enroll_now(rec, "ゆうと", 1)
    assert result == ("ok", "p1")
    assert len(db.people["p1"]["embeddings"]) == 3
    assert fc.FaceDB(db.path).find("ゆうと") == "p1"   # ファイルにも保存された
    rec.step(frame(1), 10.0)
    rec.step(frame(1), 11.0)
    assert rec.current()[1] == "ゆうと"


def test_enroll_without_face_times_out(db, rec):
    rec.enroll("ゆうと", samples=3, timeout=2.0)
    for t in (0.0, 1.0, 2.5):
        rec.step(frame(), t)
    assert rec.enroll_result() == ("noface", None)
    assert not db.people


def test_enroll_ignores_frames_with_two_people(db, rec):
    rec.enroll("ゆうと", samples=2, timeout=5.0)
    for t in (0.0, 0.5, 1.0):
        rec.step(frame(1, 2), t)
    assert rec.enroll_progress() == (0, 2)


def test_enroll_refuses_someone_elses_face(db, rec):
    mama = db.add("ママ", [vec(2, 0.3, i) for i in range(3)])
    assert enroll_now(rec, "ゆうと", 2) == ("other", mama)   # ママの顔を「ゆうと」にしない
    assert db.find("ゆうと") is None
    assert enroll_now(rec, "ママ", 2)[0] == "ok"             # 本人の追加はできる


# ---------------------------------------------------------------- あいさつ
class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def arrive(rec, person, t):
    rec.step(frame(person), t)
    rec.step(frame(person), t + 1)


def leave(rec, t):
    rec.step(frame(), t)
    rec.step(frame(), t + 10)


def test_greeter_greets_by_name_once(db, rec):
    db.add("ゆうと", [vec(1)])
    db.update("p1", call="ゆうとくん", emotion="excited")
    clock = Clock()
    g = fc.Greeter(rec, clock=clock, hour=lambda: 16)
    arrive(rec, 1, 0)
    assert g.poll() == [("excited", "おかえり、ゆうとくん！")]
    leave(rec, 2)
    clock.t += 60
    arrive(rec, 1, 20)
    assert g.poll() == []                          # さっき会ったばかり
    leave(rec, 30)
    clock.t += 31 * 60
    arrive(rec, 1, 50)
    assert g.poll() == [("excited", "あ、ゆうとくんだ！ また あえたね。")]


def test_greeter_time_of_day(db, rec):
    db.add("ママ", [vec(2)])
    clock = Clock()
    hour = [7]
    g = fc.Greeter(rec, clock=clock, hour=lambda: hour[0])
    arrive(rec, 2, 0)
    assert g.poll() == [("joy", "おはよう、ママ！")]
    leave(rec, 2)
    clock.t += 3 * 60 * 60
    hour[0] = 18
    arrive(rec, 2, 20)
    assert g.poll() == [("joy", "おかえり、ママ！")]


def test_greeter_skips_just_enrolled(db, rec):
    g = fc.Greeter(rec)
    enroll_now(rec, "ゆうと", 1)
    arrive(rec, 1, 10)
    assert rec.current()[0] == "p1"
    assert g.poll() == []


# ---------------------------------------------------------------- 声で登録
@pytest.mark.parametrize("heard, name", [
    ("ゆうと", "ゆうと"), ("ぼくはゆうとだよ", "ゆうと"), ("なまえはゆうとです", "ゆうと"),
    ("えっと ゆうと", "ゆうと"), ("ママ", "ママ"), ("はなこっていうの", "はなこ"),
    ("", None), ("きょうはこうえんでたくさんあそんだよ", None),
])
def test_extract_name(heard, name):
    assert fc.extract_name(heard) == name


def test_wants_enroll():
    assert fc.wants_enroll("ピコ かおを おぼえて")
    assert fc.wants_enroll("ぼくの顔を覚えて")
    assert not fc.wants_enroll("おぼえてる？")
    assert fc.is_yes("うん") and not fc.is_yes("ううん") and fc.is_no("ちがうよ")


class FakeRec:
    def __init__(self, result=("ok", "p1")):
        self.result, self.enrolled, self.enrolled_at = result, [], {}

    def enroll(self, name):
        self.enrolled.append(name)

    def wait_enroll(self):
        return self.result

    def person(self, pid):
        return {"name": "ママ", "call": "ママ"}

    def current(self):
        return (None, None)


def talk(dialog, *lines):
    said = []
    for line in lines:
        assert dialog.handle(line, lambda e, t: said.append((e, t)))
    return said


def test_enroll_dialog_happy_path():
    r = FakeRec()
    said = talk(fc.EnrollDialog(r), "ピコ かおを おぼえて", "ゆうと", "うん")
    assert [t for _, t in said] == ["いいよ！ おなまえ おしえて？", "ゆうと、で あってる？",
                                    "ゆうと、カメラを みて にっこりしてね！",
                                    "おぼえたよ！ よろしくね、ゆうと！"]
    assert r.enrolled == ["ゆうと"]


def test_enroll_dialog_retry_and_cancel():
    r = FakeRec()
    d = fc.EnrollDialog(r)
    said = talk(d, "かおおぼえて", "ゆうと", "ちがう", "ゆうき", "はい")
    assert "ゆうき、で あってる？" in [t for _, t in said]
    assert r.enrolled == ["ゆうき"]
    talk(d, "かおおぼえて", "やめる")
    assert not d.active and r.enrolled == ["ゆうき"]


def test_enroll_dialog_gives_up_and_times_out():
    clock = Clock(0)
    d = fc.EnrollDialog(FakeRec(), clock=clock)
    said = talk(d, "かおおぼえて", "", "", "")
    assert said[-1][0] == "sad" and not d.active
    talk(d, "かおおぼえて")
    clock.t += 60
    assert not d.active
    assert not d.handle("きょうは はれ", lambda *a: None)   # ふつうの会話にもどる


@pytest.mark.parametrize("result, emotion", [(("noface", None), "sad"),
                                             (("other", "p2"), "confused")])
def test_enroll_dialog_failures(result, emotion):
    said = talk(fc.EnrollDialog(FakeRec(result)), "かおおぼえて", "ゆうと", "うん")
    assert said[-1][0] == emotion
    if emotion == "confused":
        assert "ママ" in said[-1][1]


# ---------------------------------------------------------------- 会話とつなぐ
class RecordingBrain:
    vision = False

    def __init__(self):
        self.heard = []

    def reply(self, text, image=None):
        self.heard.append(text)
        return "joy", "やあ！"


def test_conversation_tells_brain_who_is_talking(db, rec):
    db.add("ゆうと", [vec(1)])
    db.update("p1", call="ゆうとくん")
    arrive(rec, 1, 0)
    brain = RecordingBrain()
    conv = cv.Conversation(brain, faces=rec)
    conv.handle("ピコ こんにちは")
    assert brain.heard == ["ピコ こんにちは（話しているのは ゆうとくん）"]
    assert conv.who() == "ゆうとくん"


def test_conversation_leaves_name_to_growth_hint(db, rec):
    db.add("ゆうと", [vec(1)])
    arrive(rec, 1, 0)
    brain = RecordingBrain()
    brain.extra_system = lambda: "今話している相手は「ゆうと」です。"
    brain.speaker_in_system = True  # 成長の指示が名前を伝えている
    cv.Conversation(brain, faces=rec).handle("こんにちは")
    assert brain.heard == ["こんにちは"]           # 名前を二重に伝えない


def test_conversation_runs_enroll_dialog_instead_of_brain():
    brain = RecordingBrain()
    r = FakeRec()
    conv = cv.Conversation(brain, faces=r, is_attended=lambda: False, wake_words=("ピコ",))
    conv.handle("ピコ かおを おぼえて")
    conv.handle("ゆうと")      # 呼ばれていなくても、登録の途中なら聞く
    conv.handle("うん")
    assert brain.heard == [] and r.enrolled == ["ゆうと"]
    events = [e for e, _ in conv.poll()]
    assert "love" in events


def test_conversation_without_faces_is_unchanged():
    brain = RecordingBrain()
    conv = cv.Conversation(brain)
    conv.handle("こんにちは")
    assert brain.heard == ["こんにちは"] and conv.who() is None


# ---------------------------------------------------------------- コマンド
def test_cli_list_set_forget(tmp_path, capsys):
    path = str(tmp_path / "faces.json")
    db = fc.FaceDB(path)
    db.add("ゆうと", [vec(1)])
    db.save()
    fc.main(["--db", path, "set", "ゆうと", "--call", "ゆうとくん", "--emotion", "love"])
    fc.main(["--db", path, "list"])
    out = capsys.readouterr().out
    assert "ゆうとくん" in out and "love" in out
    fc.main(["--db", path, "forget", "ゆうと"])
    assert fc.FaceDB(path).people == {}
    with pytest.raises(SystemExit):
        fc.main(["--db", path, "forget", "だれか"])


def test_default_db_path_is_local(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBOT_DATA", str(tmp_path))
    assert fc.default_db_path() == str(tmp_path / "faces.json")


def test_main_start_faces_without_camera_or_models(tmp_path, monkeypatch):
    from robot import main as rm
    args = rm.parse_args(["--no-faces"])
    assert rm.start_faces(args, object()) == (None, None)
    assert rm.start_faces(rm.parse_args([]), None) == (None, None)
    monkeypatch.setattr(fc, "MODEL_DIR", str(tmp_path))       # モデルがなくても落ちない
    monkeypatch.setattr(fc.Embedder.__init__, "__defaults__", (str(tmp_path), 320))
    assert rm.start_faces(rm.parse_args([]), SimpleNamespace(latest=lambda: None)) == (None, None)


# ---------------------------------------------------------------- 本物のモデル（あれば）
@pytest.mark.skipif(not all(os.path.exists(os.path.join(fc.MODEL_DIR, n)) for n in fc.MODELS),
                    reason="モデルがない（python3 -m robot.faces download）")
def test_real_models_load_and_find_no_face_in_blank_image():
    e = fc.Embedder()
    assert e.faces(np.zeros((480, 640, 3), np.uint8)) == []


def test_old_opencv_uses_2022mar_model():
    """Pi の OpenCV 4.6 は 2023mar を読めない（Layer id=-1 not found）ので 2022mar を使う。"""
    assert fc.yunet_for("4.6.0") == "face_detection_yunet_2022mar.onnx"
    assert fc.yunet_for("4.7.0") == "face_detection_yunet_2022mar.onnx"
    assert fc.yunet_for("4.8.0") == "face_detection_yunet_2023mar.onnx"
    assert fc.yunet_for("4.10.0") == "face_detection_yunet_2023mar.onnx"
    assert fc.yunet_for("5.0.0-pre") == "face_detection_yunet_2023mar.onnx"


def test_detector_keeps_image_size_and_scales_back():
    """黒でうめると、ない顔を見つけてしまう（実機で 320x256 にしたら誤検出）ので、そのまま渡す。"""
    import cv2
    import numpy as np

    sizes = []

    class Det:
        def setInputSize(self, size):
            sizes.append(size)

        def detect(self, img):
            assert (img.shape[1], img.shape[0]) == sizes[-1]
            f = np.zeros((1, 15), np.float32)
            f[0, :4] = (10, 20, 30, 40)
            return 1, f

    e = object.__new__(fc.Embedder)
    e.cv2, e.detector, e.detect_width = cv2, Det(), 320
    found = e.faces(np.zeros((480, 640, 3), np.uint8))
    assert sizes == [(320, 240)]
    assert list(found[0][:4]) == [20, 40, 60, 80]   # 元の画像の座標に戻っている


def test_enroll_dialog_shows_progress_while_capturing():
    import threading as th

    class Rec:
        def __init__(self):
            self.done, self.n = th.Event(), 0

        def enroll(self, name):
            def run():
                for i in range(1, 4):
                    time.sleep(0.35)
                    self.n = i
                self.done.set()
            th.Thread(target=run, daemon=True).start()
            return self.done

        def enroll_progress(self):
            return None if self.done.is_set() else (self.n, 3)

        def enroll_result(self):
            return ("ok", "p1")

    shown, said = [], []
    d = fc.EnrollDialog(Rec(), show=shown.append)
    d.name = "ゆうと"
    d._capture(lambda emo, text: said.append(text))
    assert shown[0] == "ゆうと、カメラを みてね ○○○"
    assert any("●○○" in s for s in shown)
    assert said[-1] == "おぼえたよ！ よろしくね、ゆうと！"
