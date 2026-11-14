"""成長：遊んでもらうほど、ピコと「なかよし」になっていく。

人ごとに「なかよし度」をためて、5段階のレベルで反応や話し方が変わる。
誰と遊んでいるかは顔をおぼえる機能が set_person() で教える（わからなければ「みんな」）。

保存するのは日ごとの回数（話した・なでた・タップ・顔を見ていた秒数）だけ。
会話の中身は保存しない。ファイルはこの端末の中だけ（外には送らない）:
    ~/.local/share/robot/growth.json   （ROBOT_GROWTH_FILE で変更可）

    python3 -m robot.growth             # 今のレベル
    python3 -m robot.growth report      # 家族向け：この1週間のまとめ
    python3 -m robot.growth reset       # 記録を消して最初から
"""
import argparse
import datetime as dt
import json
import os
import tempfile
import time

FAMILY = "family"      # 誰かわからないときの記録先
FAMILY_NAME = "みんな"

# (必要な なかよし度, 名前)。1日の上限 DAILY_CAP と合わせて、毎日10分くらい遊ぶと
# 2: 2〜3日 / 3: 1週間 / 4: 2週間 / 5: 1か月 くらいで上がる
LEVELS = (
    (0, "はじめまして"),
    (80, "ともだち"),
    (250, "なかよし"),
    (500, "しんゆう"),
    (1100, "だいしんゆう"),
)
DAILY_CAP = 40         # 1日にたまる上限（長く遊んでも一気に育たない）
POINTS = {"talk": 2, "stroke": 3, "tap": 1, "first_today": 5}
FACE_SECONDS_PER_POINT = 60   # 顔を見ていた1分ごとに1
LONELY_AFTER = 24 * 3600      # これだけ会わないと、次に会ったとき少しすねる
SESSION_GAP = 10 * 60         # これだけ見えなかったら「また来た」とみなしてあいさつ
KEEP_DAYS = 120               # 日ごとの記録を残す日数

# レベルごとの話し方（会話の指示に足す）
LEVEL_STYLE = {
    1: "相手とは まだ会ったばかりです。ていねいで、少し照れた話し方をしてください。",
    2: "相手とは ともだちです。明るく親しみのある話し方をしてください。",
    3: "相手とは なかよしです。くだけた話し方で、ときどき「だいすき」などの気持ちも伝えてください。",
    4: "相手とは しんゆうです。くだけた話し方で、前に遊んだことにも ときどきふれてください。",
    5: "相手とは だいしんゆうです。とても親しく話し、ときどき自分から なぞなぞ や しりとりに さそってください。",
}


def default_path():
    return os.environ.get("ROBOT_GROWTH_FILE") or os.path.join(
        os.path.expanduser("~"), ".local", "share", "robot", "growth.json")


def level_for(points):
    level = 1
    for i, (need, _) in enumerate(LEVELS):
        if points >= need:
            level = i + 1
    return level


def level_name(level):
    return LEVELS[level - 1][1]


def _new_person(name, now):
    return {"name": name, "points": 0, "level": 1, "first_met": now, "last_seen": 0,
            "days": {}}


def _new_day():
    return {"talks": 0, "strokes": 0, "taps": 0, "seconds": 0.0, "points": 0}


class Growth:
    """なかよし度の記録。イベントを渡すと、顔や声で出すこと（反応）のリストを返す。

    反応は (感情, 秒数, 言うこと or None)。main.py が顔と声に流す。
    """

    def __init__(self, path=None, clock=time.time, autosave=30.0):
        self.path = path or default_path()
        self.clock = clock
        self.autosave = autosave
        self.data = {"version": 1, "people": {}}
        self.person = FAMILY
        self.present = False
        self._face_carry = 0.0
        self._pending = None   # 別スレッドから届いた「誰か」
        self.greet_on_switch = True  # 人が入れかわったとき、ここで あいさつの言葉を出すか
        self._dirty = False
        self._saved_at = clock()
        self.load()

    # ---- 保存 ----
    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data.get("people"), dict):
                self.data = data
        except (OSError, ValueError):
            pass  # まだない・壊れている → 最初から
        return self

    def save(self):
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d or ".", prefix=".growth-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)  # 書き込み中に電源が切れても壊れない
        self._dirty = False
        self._saved_at = self.clock()

    def maybe_save(self):
        if self._dirty and self.clock() - self._saved_at >= self.autosave:
            try:
                self.save()
            except OSError as e:
                print(f"[成長] 保存できませんでした（{e}）")

    # ---- 誰と遊んでいるか ----
    def set_person(self, person_id, name=None):
        """顔をおぼえる機能から呼ぶ（別スレッドからでもよい）。person_id=None は「誰かわからない」。

        実際に切りかえるのは、次の on_face()（表示のループ）のとき。
        """
        self._pending = (person_id, name)

    def switch_person(self, person_id, name=None):
        """すぐに切りかえて、あいさつなどの反応を返す。"""
        pid = str(person_id) if person_id else FAMILY
        p = self._get(pid, name)
        if name and pid != FAMILY:
            p["name"] = name
        if pid == self.person:
            return []
        self.person = pid
        # 人が入れかわったら、その人へのあいさつ（ほかの機能があいさつするなら言葉は出さない）
        if not self.present:
            return []
        out = self._greet(p)
        if not self.greet_on_switch:  # 表情とポイントだけ。言葉は顔をおぼえる機能のあいさつにまかせる
            out = [(emo, dur, None) for emo, dur, _ in out]
        return out

    def _get(self, pid, name=None):
        people = self.data["people"]
        if pid not in people:
            people[pid] = _new_person(name or (FAMILY_NAME if pid == FAMILY else pid),
                                      self.clock())
            self._dirty = True
        return people[pid]

    @property
    def current(self):
        return self._get(self.person)

    @property
    def level(self):
        return self.current["level"]

    def is_named(self):
        return self.person != FAMILY

    # ---- 入力 ----
    def on_face(self, present, seconds=0.0):
        """毎フレーム呼ぶ。seconds は前のフレームからの時間。"""
        out = []
        pending, self._pending = self._pending, None
        if pending:
            out += self.switch_person(*pending)
        if present and not self.present:
            out += self._greet(self.current)
        self.present = present
        if present:
            p = self.current
            p["last_seen"] = self.clock()
            day = self._today(p)
            day["seconds"] += seconds
            self._face_carry += seconds
            while self._face_carry >= FACE_SECONDS_PER_POINT:
                self._face_carry -= FACE_SECONDS_PER_POINT
                out += self._add(1)
            self._dirty = True
        self.maybe_save()
        return out

    def on_talk(self):
        self._today(self.current)["talks"] += 1
        return self._touch() + self._add(POINTS["talk"])

    def on_stroke(self):
        self._today(self.current)["strokes"] += 1
        out = self._touch() + self._add(POINTS["stroke"])
        if self.level >= 3:  # なかよしになると、なでられたとき特にうれしそう
            out.append(("love", 2.0, None))
        return out

    def on_tap(self):
        self._today(self.current)["taps"] += 1
        return self._touch() + self._add(POINTS["tap"])

    # ---- 中身 ----
    def _touch(self):
        """顔が見えなくても（カメラなしでも）、遊んでもらえたら会ったことにする。"""
        out = []
        if not self.present:
            out = self._greet(self.current, quiet=True)
        self.current["last_seen"] = self.clock()
        return out

    def _today_key(self):
        return dt.datetime.fromtimestamp(self.clock()).strftime("%Y-%m-%d")

    def _today(self, p):
        key = self._today_key()
        days = p["days"]
        if key not in days:
            days[key] = _new_day()
            for old in sorted(days)[:-KEEP_DAYS]:
                del days[old]
        self._dirty = True
        return days[key]

    def _add(self, points):
        p = self.current
        day = self._today(p)
        points = max(0, min(points, DAILY_CAP - day["points"]))
        if not points:
            return []
        day["points"] += points
        p["points"] += points
        self._dirty = True
        new = level_for(p["points"])
        if new > p["level"]:
            p["level"] = new
            self.save_soon()
            who = f"{p['name']}と " if self.is_named() else ""
            return [("surprise", 1.0, None),
                    ("joy", 3.0, f"やったあ！ {who}{level_name(new)}に なれたよ！")]
        return []

    def save_soon(self):
        self._saved_at = -1e18  # 次の maybe_save ですぐ保存

    def _greet(self, p, quiet=False):
        """会ったときの反応。しばらくぶりなら あいさつ、1日以上ぶりなら少しすねる。"""
        now = self.clock()
        gap = now - p["last_seen"] if p["last_seen"] else None
        out = []
        first_today = self._today_key() not in p["days"]
        named = self.is_named() and p["level"] >= 2
        name = f"{p['name']}、" if named else ""
        if gap is not None and gap >= LONELY_AFTER:
            out += [("sad", 2.0, None),
                    ("joy", 2.5, f"{name}ひさしぶり！ さみしかったよ。")]
        elif not quiet and (gap is None or gap >= SESSION_GAP):
            if gap is None:
                out.append(("joy", 2.5, "はじめまして！ よろしくね。"))
            elif named:
                out.append(("joy", 2.5, f"{name}おかえり！"))
        if first_today:
            out += self._add(POINTS["first_today"])
        p["last_seen"] = now
        self._dirty = True
        return out

    # ---- 会話へ ----
    def prompt_hint(self):
        """会話の指示に足す文。誰と話しているか・どれくらい なかよしか。"""
        p = self.current
        lines = [LEVEL_STYLE[p["level"]]]
        if self.is_named():
            lines.append(f"今話している相手は「{p['name']}」です。ときどき名前で呼んでください。")
        if p["level"] >= 4:
            days = sorted(k for k in p["days"] if k != self._today_key())
            if days:
                last = dt.date.fromisoformat(days[-1])
                ago = (dt.date.fromisoformat(self._today_key()) - last).days
                when = {1: "きのう", 2: "おととい"}.get(ago, f"{ago}日前")
                lines.append(f"前に遊んだのは{when}です。")
        return "\n".join("- " + s for s in lines)

    # ---- 家族向け ----
    def status(self):
        rows = []
        for p in self.data["people"].values():
            lv = p["level"]
            nxt = LEVELS[lv][0] - p["points"] if lv < len(LEVELS) else None
            more = f"（つぎまで あと{nxt}）" if nxt else "（さいこう！）"
            rows.append(f"{p['name']}: レベル{lv} {level_name(lv)}  なかよし度 {p['points']}{more}")
        return "\n".join(rows) or "まだ記録がありません。"

    def report(self, days=7):
        today = dt.date.fromisoformat(self._today_key())
        keys = [(today - dt.timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]
        out = [f"ピコの成長日記（{keys[0]} 〜 {keys[-1]}）", ""]
        for p in self.data["people"].values():
            got = [(k, p["days"][k]) for k in keys if k in p["days"]]
            if not got:
                continue
            talks = sum(d["talks"] for _, d in got)
            strokes = sum(d["strokes"] for _, d in got)
            minutes = sum(d["seconds"] for _, d in got) / 60
            out.append(f"■ {p['name']}  レベル{p['level']} {level_name(p['level'])}")
            out.append(f"  遊んだ日: {len(got)}日 / 話した: {talks}回 / なでた: {strokes}回 / "
                       f"顔を見せてくれた: {minutes:.0f}分")
            for k, d in got:
                bar = "■" * min(20, d["points"] // 2)
                out.append(f"  {k[5:]} {bar} 話{d['talks']} なで{d['strokes']} "
                           f"{d['seconds'] / 60:.0f}分")
            out.append("")
        if len(out) == 2:
            out.append("この期間は まだ遊んでいません。")
        return "\n".join(out).rstrip()


def main(argv=None):
    ap = argparse.ArgumentParser(description="ピコの成長の記録を見る")
    ap.add_argument("cmd", nargs="?", default="status", choices=["status", "report", "reset"])
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--file", default=None, help="記録ファイル")
    args = ap.parse_args(argv)
    g = Growth(args.file)
    if args.cmd == "report":
        print(g.report(args.days))
    elif args.cmd == "reset":
        if input("成長の記録を全部消します。よいですか？ [y/N] ").strip().lower() == "y":
            g.data = {"version": 1, "people": {}}
            g.save()
            print("消しました。")
    else:
        print(g.status())


if __name__ == "__main__":
    main()
