"""ことば（話し方に合わせる）のテスト。"""
from robot.language import MIN_UTTERANCES, Language, level_for, spoken_length


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def make(tmp_path):
    clock = Clock()
    return Language(path=str(tmp_path / "language.json"), clock=clock), clock


def talk(lang, text, n=MIN_UTTERANCES):
    for _ in range(n):
        lang.on_turn(text, "そうなんだ！")


def test_spoken_length_ignores_spaces_and_marks():
    assert spoken_length("きょう こうえん いった！") == 10
    assert spoken_length("") == 0


def test_levels():
    assert level_for(3) == 1 and level_for(8) == 2 and level_for(20) == 3


def test_stays_normal_until_enough_talk(tmp_path):
    lang, _ = make(tmp_path)
    talk(lang, "うん", MIN_UTTERANCES - 1)
    assert lang.level() == 2
    assert lang.prompt_hint() == ""


def test_short_talk_gives_easy_style(tmp_path):
    lang, _ = make(tmp_path)
    talk(lang, "わんわん")
    assert lang.level() == 1
    assert "短くやさしい" in lang.prompt_hint()


def test_long_talk_gives_richer_style_and_is_per_person(tmp_path):
    lang, _ = make(tmp_path)
    lang.set_person("p1", "ゆうくん")
    talk(lang, "きょうは こうえんで おおきな すべりだいを すべったよ")
    assert lang.level() == 3
    lang.set_person("p2", "ママ")
    assert lang.level() == 2


def test_old_days_drop_out_of_window(tmp_path):
    lang, clock = make(tmp_path)
    talk(lang, "わんわん")
    clock.t += 8 * 86400
    assert lang.level() == 2


def test_word_of_the_week(tmp_path):
    lang, clock = make(tmp_path)
    lang.set_word("とうめい", "すけて むこうが見えること")
    hint = lang.prompt_hint()
    assert "とうめい" in hint and "すけて" in hint
    lang.on_turn("この コップ とうめい", "そうだね、とうめい だね")
    report = lang.report()
    assert "ピコが使った 1回 / 自分で言った 1回" in report
    clock.t += 8 * 86400
    assert lang.word() is None and "とうめい" not in lang.prompt_hint()


def test_saves_only_numbers(tmp_path):
    lang, _ = make(tmp_path)
    lang.on_turn("ひみつの あいことば は りんご", "そっか")
    text = open(lang.path, encoding="utf-8").read()
    assert "りんご" not in text and "ひみつ" not in text
    again = Language(path=lang.path, clock=lang.clock)
    assert again.stats()[0] == 1


def test_conversation_counts_turns(tmp_path):
    from robot.conversation import Conversation, SimpleBrain
    lang, _ = make(tmp_path)
    Conversation(SimpleBrain("ピコ"), language=lang).handle("こんにちは ピコ")
    assert lang.stats()[0] == 1
