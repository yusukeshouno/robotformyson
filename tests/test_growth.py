"""成長（なかよし度）のテスト。時計を進めて、何日分もの遊びをまねる。"""
import json

import pytest

from robot import growth as gr


class Clock:
    def __init__(self, t=1_800_000_000.0):  # 2027-01-15 ごろ
        self.t = t

    def __call__(self):
        return self.t

    def days(self, n):
        self.t += n * 86400


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def g(tmp_path, clock):
    return gr.Growth(str(tmp_path / "growth.json"), clock=clock)


def texts(reactions):
    return [t for _, _, t in reactions if t]


def play_one_day(g, clock, minutes=10):
    out = g.on_face(True, 0.1)
    for _ in range(minutes):
        out += g.on_face(True, 60)
        out += g.on_talk()
    out += g.on_stroke()
    g.on_face(False, 0.1)
    clock.t += 3600
    return out


def test_levels_from_points():
    assert gr.level_for(0) == 1
    assert gr.level_for(79) == 1
    assert gr.level_for(80) == 2
    assert gr.level_for(10_000) == len(gr.LEVELS)


def test_first_meeting_says_hello(g):
    assert "はじめまして！ よろしくね。" in texts(g.on_face(True, 0.1))


def test_daily_cap_keeps_growth_slow(g, clock):
    g.on_face(True, 0.1)
    for _ in range(500):
        g.on_talk()
    assert g.current["points"] == gr.DAILY_CAP
    assert g.level == 1


def test_grows_over_weeks(g, clock):
    reached = {}
    for day in range(40):
        for _ in play_one_day(g, clock):
            pass
        reached.setdefault(g.level, day + 1)
        clock.days(1)
        clock.t -= 3600
    # 毎日10分遊ぶと: 2〜3日でともだち、1週間でなかよし、2週間でしんゆう、1か月でだいしんゆう
    assert 2 <= reached[2] <= 3
    assert 6 <= reached[3] <= 8
    assert 12 <= reached[4] <= 15
    assert 25 <= reached[5] <= 31


def test_level_up_is_announced(g, clock):
    g.current["points"] = 79
    out = g.on_face(True, 0.1) + g.on_talk()
    assert any("ともだち" in t for t in texts(out))
    assert out[-2][0] == "surprise"


def test_lonely_after_a_day_but_points_stay(g, clock):
    play_one_day(g, clock)
    before = g.current["points"]
    clock.days(2)
    out = g.on_face(True, 0.1)
    assert out[0][0] == "sad"
    assert any("さみしかった" in t for t in texts(out))
    assert g.current["points"] >= before


def test_person_from_face_recognition(g, clock):
    g.set_person("p1", "ゆうと")      # 別スレッドから届く想定
    g.on_face(True, 0.1)              # 表示のループで反映
    assert g.person == "p1"
    g.on_talk()
    g.set_person(None, None)          # 知らない顔 / 誰もいない
    g.on_face(True, 0.1)
    g.on_talk()
    people = g.data["people"]
    assert people["p1"]["name"] == "ゆうと"
    assert people["p1"]["days"] and people[gr.FAMILY]["days"]
    assert people["p1"]["points"] > 0 and people[gr.FAMILY]["points"] > 0


def test_greets_by_name_when_friends(g, clock):
    g.switch_person("p1", "ゆうと")
    g.current.update(points=100, level=2, last_seen=clock() - 3600)
    out = g.on_face(True, 0.1)
    assert "ゆうと、おかえり！" in texts(out)


def test_prompt_hint_changes_with_level(g, clock):
    g.switch_person("p1", "ゆうと")
    assert "会ったばかり" in g.prompt_hint()
    play_one_day(g, clock)
    clock.days(1)
    g.current.update(points=600, level=4)
    hint = g.prompt_hint()
    assert "しんゆう" in hint and "ゆうと" in hint and "きのう" in hint


def test_saves_only_counts(g, clock, tmp_path):
    play_one_day(g, clock)
    g.save()
    raw = (tmp_path / "growth.json").read_text(encoding="utf-8")
    day = next(iter(json.loads(raw)["people"][gr.FAMILY]["days"].values()))
    assert set(day) == {"talks", "strokes", "taps", "seconds", "points"}
    again = gr.Growth(str(tmp_path / "growth.json"), clock=clock)
    assert again.current["points"] == g.current["points"]


def test_broken_file_starts_fresh(tmp_path, clock):
    p = tmp_path / "growth.json"
    p.write_text("{こわれた", encoding="utf-8")
    assert gr.Growth(str(p), clock=clock).level == 1


def test_report(g, clock):
    g.switch_person("p1", "ゆうと")
    for _ in range(3):
        play_one_day(g, clock)
        clock.days(1)
    rep = g.report()
    assert "ゆうと" in rep and "遊んだ日: 3日" in rep
    assert "まだ記録" in gr.Growth("/nonexistent/x.json", clock=clock).status()


def test_brain_gets_hint():
    from robot import conversation as cv
    brain = cv.ClaudeBrain.__new__(cv.ClaudeBrain)
    cv._BrainBase.__init__(brain, "ピコ", 10)
    assert brain.system_prompt() == brain.system
    brain.extra_system = lambda: "- テスト"
    assert brain.system_prompt().endswith("\n- テスト")


def test_switch_greeting_can_be_left_to_face_greeter(tmp_path):
    from robot.growth import Growth
    g = Growth(path=str(tmp_path / "g.json"))
    g.on_face(True, 0.1)
    g.greet_on_switch = False
    g.set_person("p1", "ゆうくん")
    out = g.on_face(True, 0.1)
    assert all(text is None for _, _, text in out)
