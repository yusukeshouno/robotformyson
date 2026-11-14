"""ことば：話し方に合わせて、ピコの言葉のむずかしさを少しずつ変える。

人ごとに、1日に何回話したか・聞き取れた文字数・長めに話した回数（数字だけ）を数える。
直近7日の「1回あたりの文字数」で、ピコの話し方を3段階から選ぶ。
親が「今週のことば」を決めると、ピコが会話の中で自然に使い、意味もそっと伝える。
子どもがそのことばを自分で言った回数も数える。

保存するのは数字と「今週のことば」だけ。声や会話の全文は保存しない。
ファイルはこの端末の中だけ:
    ~/.local/share/robot/language.json   （ROBOT_LANGUAGE_FILE で変更可）

    python3 -m robot.language                         # 家族向け：ことばの記録
    python3 -m robot.language word とうめい "すけて むこうが見えること"   # 今週のことば（7日間）
    python3 -m robot.language word --clear            # 今週のことばをやめる
    python3 -m robot.language reset                   # 記録を消す

子どもの声の聞き取りはまちがいが多いので（Janssens ら 2024）、数字は大まかな目安。
もとにした研究: Kory-Westlund & Breazeal (2019) 言葉の力に合わせた物語で語彙の伸びが大きかった。
ELLA (Antony ら 2026) 親が選んだ言葉の目標に合わせて生成 AI が対話する。
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import tempfile
import threading
import time

WINDOW_DAYS = 7
MIN_UTTERANCES = 20          # これより少ないうちは、ふつう（レベル2）のまま
LEVEL_CHARS = (6.0, 12.0)    # 1回あたりの文字数: 6未満→1、12未満→2、それ以上→3
LONG_CHARS = 15              # これ以上を「長めに話した」とする
WORD_DAYS = 7
KEEP_DAYS = 120
FAMILY = "family"

STYLE = {
    1: "話し方: 1文を10文字くらいまでの、短くやさしい言葉にしてください。ひらがなで言える言葉だけを使います。",
    2: "",  # ふつう（会話のもとの指示どおり）
    3: ("話し方: 相手はたくさん話せます。ときどき少しだけむずかしい言葉を1つ使い、"
        "「〇〇って、〜ってことだよ」と意味もそっと添えてください。"),
}

_NOT_SPOKEN = re.compile(r"[\s、。！？!?…ー〜・「」]")


def default_path():
    return os.environ.get("ROBOT_LANGUAGE_FILE") or os.path.join(
        os.path.expanduser("~"), ".local", "share", "robot", "language.json")


def spoken_length(text):
    return len(_NOT_SPOKEN.sub("", text or ""))


def level_for(avg_chars):
    if avg_chars < LEVEL_CHARS[0]:
        return 1
    if avg_chars < LEVEL_CHARS[1]:
        return 2
    return 3


class Language:
    def __init__(self, path=None, clock=time.time):
        self.path = path or default_path()
        self.clock = clock
        self._lock = threading.RLock()
        self.data = {"people": {}, "word": None}
        self.person, self.name = FAMILY, "みんな"
        self.load()

    # ---- ファイル ----
    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data.get("people"), dict):
                self.data = {"people": data["people"], "word": data.get("word")}
        except (OSError, ValueError):
            pass

    def save(self):
        with self._lock:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d or ".", prefix=".language-", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)

    def _day(self, ts=None):
        return dt.date.fromtimestamp(self.clock() if ts is None else ts)

    # ---- 誰と話しているか（顔をおぼえる機能の on_change から） ----
    def set_person(self, person_id, name=None):
        with self._lock:
            self.person = str(person_id) if person_id else FAMILY
            self.name = name or ("みんな" if self.person == FAMILY else self.person)

    def _p(self):
        p = self.data["people"].setdefault(self.person, {"name": self.name, "days": {}})
        if self.person != FAMILY and self.name:
            p["name"] = self.name
        return p

    # ---- 会話から ----
    def on_turn(self, child, answer):
        """1回のやりとりを数える。保存するのは数字だけ。"""
        n = spoken_length(child)
        if n < 2:
            return
        with self._lock:
            p = self._p()
            key = self._day().isoformat()
            d = p["days"].setdefault(key, {"n": 0, "chars": 0, "long": 0, "word_said": 0,
                                           "word_heard": 0})
            d["n"] += 1
            d["chars"] += n
            d["long"] += n >= LONG_CHARS
            w = self.word()
            if w:
                d["word_said"] += w["word"] in (child or "")     # 子どもが言った
                d["word_heard"] += w["word"] in (answer or "")   # ピコが使った
            self._prune(p)
        try:
            self.save()
        except OSError as e:
            print(f"[ことば] 保存できませんでした（{e}）")

    def _prune(self, p):
        limit = (self._day() - dt.timedelta(days=KEEP_DAYS)).isoformat()
        for k in [k for k in p["days"] if k < limit]:
            del p["days"][k]

    # ---- 話し方 ----
    def stats(self, pid=None, days=WINDOW_DAYS):
        """(回数, 1回あたりの文字数) を直近 days 日で。"""
        with self._lock:
            p = self.data["people"].get(pid or self.person)
            if not p:
                return 0, 0.0
            start = (self._day() - dt.timedelta(days=days - 1)).isoformat()
            rows = [d for k, d in p["days"].items() if k >= start]
        n = sum(d["n"] for d in rows)
        chars = sum(d["chars"] for d in rows)
        return n, (chars / n if n else 0.0)

    def level(self, pid=None):
        n, avg = self.stats(pid)
        return level_for(avg) if n >= MIN_UTTERANCES else 2

    # ---- 今週のことば ----
    def set_word(self, word, meaning="", days=WORD_DAYS):
        with self._lock:
            self.data["word"] = ({"word": word, "meaning": meaning,
                                  "until": self.clock() + days * 86400} if word else None)

    def word(self):
        w = self.data.get("word")
        return w if w and w.get("until", 0) > self.clock() else None

    # ---- 会話の指示へ ----
    def prompt_hint(self):
        lines = []
        style = STYLE[self.level()]
        if style:
            lines.append(style)
        w = self.word()
        if w:
            meaning = f"（意味: {w['meaning']}）" if w.get("meaning") else ""
            lines.append(f"今週のことばは「{w['word']}」{meaning}です。話に自然に合うときだけ使い、"
                         "はじめて使うときは意味もやさしく伝えてください。毎回は使わないでください。")
        return "\n".join("- " + s for s in lines)

    # ---- 家族向け ----
    def report(self, days=WINDOW_DAYS):
        out = []
        w = self.word()
        if w:
            left = max(0, math.ceil((w["until"] - self.clock()) / 86400))
            out.append(f"今週のことば: 「{w['word']}」（あと{left}日）")
        with self._lock:
            people = list(self.data["people"].items())
        start = (self._day() - dt.timedelta(days=days - 1)).isoformat()
        for pid, p in people:
            n, avg = self.stats(pid, days)
            rows = [d for k, d in p["days"].items() if k >= start]
            said = sum(d.get("word_said", 0) for d in rows)
            heard = sum(d.get("word_heard", 0) for d in rows)
            lv = self.level(pid)
            note = "" if n >= MIN_UTTERANCES else f"（{MIN_UTTERANCES}回たまるまでは ふつう）"
            out.append(f"{p['name']}: この{days}日で {n}回 話した・1回 平均 {avg:.1f}文字・"
                       f"話し方 レベル{lv}{note}")
            if w:
                out.append(f"  「{w['word']}」 ピコが使った {heard}回 / 自分で言った {said}回")
        out.append("※ 子どもの声の聞き取りはまちがいが多いので、数字は大まかな目安です。")
        return "\n".join(out)

    def reset(self):
        with self._lock:
            self.data["people"] = {}


def main(argv=None):
    ap = argparse.ArgumentParser(description="ピコのことばの記録（家族向け）")
    sub = ap.add_subparsers(dest="command")
    sub.add_parser("report")
    w = sub.add_parser("word", help="今週のことばを決める")
    w.add_argument("word", nargs="?")
    w.add_argument("meaning", nargs="?", default="")
    w.add_argument("--clear", action="store_true")
    sub.add_parser("reset")
    args = ap.parse_args(argv)
    lang = Language()
    if args.command == "word":
        if args.clear or not args.word:
            lang.set_word(None)
            print("今週のことばをやめました。")
        else:
            lang.set_word(args.word, args.meaning)
            print(f"今週のことば: 「{args.word}」（{WORD_DAYS}日間）")
        lang.save()
    elif args.command == "reset":
        lang.reset()
        lang.save()
        print("ことばの記録を消しました（今週のことばは残します）。")
    else:
        print(lang.report())


if __name__ == "__main__":
    main()
