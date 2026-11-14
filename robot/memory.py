"""記憶：前に話したことを、人ごとに少しだけおぼえておく。

会話が一区切りついたら（しばらく話さない・相手が入れかわった）、その会話から
「好きなもの・遊びの話題・これからの予定」を最大3つの短いメモにして保存する。
次にその人の顔を見分けたら、新しいメモを会話の指示に入れて、ピコが
「きのうの恐竜、名前きまった？」のように自然に触れられるようにする。

子どものための決まり:
- 保存するのは短いメモだけ。声や会話の全文は保存しない。
- 顔をおぼえた人（顔をおぼえる機能が見分けた人）のメモだけ作る。知らない人の話はおぼえない。
- 住所・電話・学校や園の名前・体や病気・家族のけんか・ひみつ などは、
  まとめるときに除くよう指示し、さらに言葉で見つけたら捨てる。
- 古いメモは KEEP_DAYS 日で消える。1人 MAX_NOTES 件まで。
- 「わすれて」と言われたら、いまの会話はおぼえない。
- まとめるのは Ollama（家の中の Mac mini など）だけ。--brain claude のときは
  外に送らないよう、メモを作らない。

ファイルはこの端末の中だけ:
    ~/.local/share/robot/memory.json   （ROBOT_MEMORY_FILE で変更可）

    python3 -m robot.memory              # 家族向け：おぼえていることの一覧
    python3 -m robot.memory forget ゆうと  # その人のメモを消す
    python3 -m robot.memory reset        # ぜんぶ消す

もとにした研究: Ligthart ら (HRI 2022) 前回の話を覚えているロボットで子どもの関心が続いた。
Leite ら (IDC 2017) 年少の子は記憶なしの方を好んだ → --no-memory で切れるようにしてある。
"""
import argparse
import json
import os
import re
import tempfile
import threading
import time

KEEP_DAYS = 90
MAX_NOTES = 20
IN_PROMPT = 3          # 会話の指示に入れるメモの数
IDLE = 60.0            # 最後のやりとりからこの秒数たったら、まとめる
MIN_TURNS = 2          # これより短い会話はおぼえない
NOTE_MAX_LEN = 40

SUMMARY_PROMPT = """次は、ロボットの{robot}と子ども（{name}）の会話です。
次に会ったときに話題にできることを、最大3つ、短いメモにしてください。

おぼえてよいもの: 好きなもの・きらいな食べもの・はまっている遊び・作っているもの・これからの楽しみな予定。
おぼえてはいけないもの: 住所・電話番号・学校や園や先生の名前・体や病気やけが・家族のけんか・お金・ひみつ・こわかった話。

書き方:
- 1行に1つ、「- 」で始める。{robot}から見た言い方で、30文字以内。例: 「- 恐竜のティラノサウルスが好き」
- おぼえることがなければ「なし」とだけ書く。

会話:
{dialog}"""

# まとめに混ざっても保存しない言葉（子どものプライバシーと安全のため）
_NG = re.compile(
    r"住所|じゅうしょ|電話|でんわ|番号|丁目|番地|マンション|パスワード|"
    r"学校|がっこう|幼稚園|ようちえん|保育園|ほいくえん|先生|せんせい|"
    r"病気|びょうき|病院|びょういん|けが|ケガ|薬|くすり|痛|いたい|"
    r"けんか|ケンカ|喧嘩|離婚|叩|たたか|ぶたれ|なぐ|殴|"
    r"お金|おかね|ひみつ|秘密|ないしょ|内緒")
_DIGITS = re.compile(r"[0-9０-９]{3,}")
_FORGET = ("わすれて", "忘れて")


def default_path():
    return os.environ.get("ROBOT_MEMORY_FILE") or os.path.join(
        os.path.expanduser("~"), ".local", "share", "robot", "memory.json")


def wants_forget(text):
    """「いまのは わすれて」のような言葉か。"""
    return any(w in text for w in _FORGET)


def clean_notes(raw):
    """モデルのまとめから、保存してよいメモだけを取り出す。"""
    out = []
    for line in (raw or "").splitlines():
        s = line.strip()
        if not s.startswith(("-", "・", "*", "－")):
            continue
        s = s.lstrip("-・*－ ").strip().strip("「」")
        if not s or s == "なし" or _NG.search(s) or _DIGITS.search(s):
            continue
        out.append(s[:NOTE_MAX_LEN])
    return out[:3]


def _when(ts, now):
    days = int((now - ts) // 86400)
    return {0: "きょう", 1: "きのう", 2: "おととい"}.get(days, f"{days}日前")


class Memory:
    def __init__(self, path=None, clock=time.time, summarize=None, robot="ピコ"):
        self.path = path or default_path()
        self.clock = clock
        self.summarize = summarize    # (prompt) -> str / None。None なら新しいメモは作らない
        self.robot = robot
        self._lock = threading.RLock()
        self.data = {"people": {}}
        self.person, self.name = None, None
        self._session = None          # {"pid", "name", "turns": [(子ども, ピコ)], "last"}
        self._todo = []               # まとめ待ちの会話
        self._running = False
        self.load()
        self.prune()

    # ---- ファイル ----
    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data.get("people"), dict):
                self.data = data
        except (OSError, ValueError):
            pass

    def save(self):
        with self._lock:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d or ".", prefix=".memory-", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)  # 書き込み中に電源が切れても壊れない

    def prune(self):
        """古いメモと、多すぎるメモを消す。"""
        limit = self.clock() - KEEP_DAYS * 86400
        changed = False
        with self._lock:
            for p in self.data["people"].values():
                keep = [n for n in p.get("notes", []) if n["t"] >= limit][-MAX_NOTES:]
                if len(keep) != len(p.get("notes", [])):
                    p["notes"], changed = keep, True
        return changed

    # ---- 誰と話しているか（顔をおぼえる機能の on_change から。別スレッドでもよい） ----
    def set_person(self, person_id, name=None):
        with self._lock:
            pid = str(person_id) if person_id else None
            if pid == self.person:
                return
            self._close_session()
            self.person, self.name = pid, name
            if pid and name:
                self.data["people"].setdefault(pid, {"name": name, "notes": []})["name"] = name

    def set_call(self, call):
        """呼び名（「ゆうくん」など）があれば、こちらを使う。"""
        with self._lock:
            if call:
                self.name = call

    # ---- 会話から ----
    def add_turn(self, child, answer):
        """1回のやりとりを、いまの会話に足す（メモにするのは後で）。"""
        with self._lock:
            if not self.person:
                return
            s = self._session
            if not s or s["pid"] != self.person:
                self._close_session()
                s = self._session = {"pid": self.person, "name": self.name or "",
                                     "turns": [], "last": 0.0}
            s["turns"].append((child, answer))
            s["last"] = self.clock()

    def drop_session(self):
        """「わすれて」と言われたとき。いまの会話はメモにしない。"""
        with self._lock:
            self._session = None

    def _close_session(self):
        s, self._session = self._session, None
        if s and len(s["turns"]) >= MIN_TURNS:
            self._todo.append(s)

    def step(self, now=None):
        """まとめ待ちの会話をメモにする。時間がかかるので裏のスレッドから呼ぶ。"""
        now = self.clock() if now is None else now
        with self._lock:
            s = self._session
            if s and now - s["last"] >= IDLE:
                self._close_session()
            todo, self._todo = self._todo, []
        saved = 0
        for s in todo:
            saved += self._summarize(s)
        changed = self.prune()
        if saved or changed:
            self.save()
        return saved

    def _summarize(self, s):
        if not self.summarize:
            return 0
        dialog = "\n".join(f"{s['name'] or '子ども'}: {c}\n{self.robot}: {a}"
                           for c, a in s["turns"])
        try:
            raw = self.summarize(SUMMARY_PROMPT.format(robot=self.robot, name=s["name"] or "子ども",
                                                       dialog=dialog))
        except Exception as e:  # つながらなくても会話はそのまま続ける
            print(f"[記憶] まとめられませんでした（{e}）")
            return 0
        notes = clean_notes(raw)
        with self._lock:
            p = self.data["people"].setdefault(s["pid"], {"name": s["name"], "notes": []})
            have = {n["text"] for n in p["notes"]}
            now = self.clock()
            new = [{"t": now, "text": t} for t in notes if t not in have]
            p["notes"] = (p["notes"] + new)[-MAX_NOTES:]
        for n in new:
            print(f"[記憶] {s['name']}: {n['text']}")
        return len(new)

    def start(self, interval=5.0):
        self._running = True

        def loop():
            while self._running:
                time.sleep(interval)
                try:
                    self.step()
                except Exception as e:
                    print(f"[記憶] エラー: {e}")
        threading.Thread(target=loop, daemon=True).start()
        return self

    def stop(self):
        """終わるときに、まだメモにしていない会話もまとめて保存する。"""
        self._running = False
        with self._lock:
            self._close_session()
        self.step()

    # ---- 会話の指示へ ----
    def prompt_hint(self):
        with self._lock:
            p = self.data["people"].get(self.person) if self.person else None
            notes = list(p["notes"][-IN_PROMPT:]) if p else []
            name = self.name or (p or {}).get("name", "")
        if not notes:
            return ""
        now = self.clock()
        lines = [f"前に{name}と話したこと（自然なときだけ1つ触れてください。"
                 "しつこく聞いたり、おぼえていることを自慢したりしないこと）:"]
        lines += [f"  ・{n['text']}（{_when(n['t'], now)}）" for n in reversed(notes)]
        return "- " + "\n".join(lines)

    # ---- 家族向け ----
    def report(self):
        rows = []
        now = self.clock()
        with self._lock:
            for p in self.data["people"].values():
                rows.append(f"{p['name']}（{len(p['notes'])}件）")
                rows += [f"  ・{n['text']}（{_when(n['t'], now)}）" for n in p["notes"]]
        return "\n".join(rows) or "まだ何もおぼえていません。"

    def forget(self, name_or_id):
        with self._lock:
            for pid, p in list(self.data["people"].items()):
                if name_or_id in (pid, p.get("name")):
                    del self.data["people"][pid]
                    if self._session and self._session["pid"] == pid:
                        self._session = None
                    return True
        return False

    def forget_all(self):
        with self._lock:
            self.data = {"people": {}}
            self._session, self._todo = None, []


def main(argv=None):
    ap = argparse.ArgumentParser(description="ピコがおぼえていること（家族向け）")
    ap.add_argument("command", nargs="?", default="list", choices=["list", "forget", "reset"])
    ap.add_argument("name", nargs="?", help="forget のとき：消す人の名前")
    args = ap.parse_args(argv)
    m = Memory()
    if args.command == "list":
        print(m.report())
    elif args.command == "forget":
        if not args.name:
            ap.error("消す人の名前を指定してください")
        print("消しました。" if m.forget(args.name) else f"{args.name} の記録はありません。")
        m.save()
    else:
        m.forget_all()
        m.save()
        print("ぜんぶ消しました。")


if __name__ == "__main__":
    main()
