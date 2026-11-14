"""記憶（前に話したことをおぼえる）のテスト。"""
from robot.memory import IDLE, KEEP_DAYS, Memory, clean_notes, wants_forget


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def make(tmp_path, reply="- 恐竜のティラノサウルスが好き\n- 週末に公園へ行く"):
    clock = Clock()
    asked = []

    def summarize(prompt):
        asked.append(prompt)
        return reply
    m = Memory(path=str(tmp_path / "memory.json"), clock=clock, summarize=summarize)
    return m, clock, asked


def chat(m, n=2):
    for i in range(n):
        m.add_turn(f"きょうりゅう すき {i}", "ぼくも すき！")


def test_remembers_after_idle_and_puts_in_prompt(tmp_path):
    m, clock, asked = make(tmp_path)
    m.set_person("p1", "ゆうくん")
    chat(m)
    assert m.step() == 0              # まだ会話中
    clock.t += IDLE + 1
    assert m.step() == 2
    assert "ゆうくん" in asked[0] and "きょうりゅう" in asked[0]
    hint = m.prompt_hint()
    assert "ティラノサウルス" in hint and "きょう" in hint
    clock.t += 86400
    assert "きのう" in m.prompt_hint()
    # ファイルに残り、読み直せる
    m2 = Memory(path=m.path, clock=clock)
    m2.set_person("p1", "ゆうくん")
    assert "ティラノサウルス" in m2.prompt_hint()


def test_unknown_person_is_not_remembered(tmp_path):
    m, clock, asked = make(tmp_path)
    m.set_person(None, None)
    chat(m, 3)
    clock.t += IDLE + 1
    assert m.step() == 0 and not asked


def test_short_chat_is_not_remembered(tmp_path):
    m, clock, asked = make(tmp_path)
    m.set_person("p1", "ゆうくん")
    chat(m, 1)
    clock.t += IDLE + 1
    assert m.step() == 0 and not asked


def test_switching_person_closes_the_chat(tmp_path):
    m, clock, asked = make(tmp_path)
    m.set_person("p1", "ゆうくん")
    chat(m)
    m.set_person("p2", "ママ")
    assert m.step() == 2
    assert m.prompt_hint() == ""      # ママとはまだ話していない
    m.set_person("p1", "ゆうくん")
    assert "ティラノサウルス" in m.prompt_hint()


def test_forget_words_drop_the_chat(tmp_path):
    assert wants_forget("いまの わすれて")
    m, clock, asked = make(tmp_path)
    m.set_person("p1", "ゆうくん")
    chat(m)
    m.drop_session()
    clock.t += IDLE + 1
    assert m.step() == 0 and not asked


def test_private_things_are_not_saved():
    raw = """- 恐竜が好き
- 住所は○○町
- 電話は 090-1234-5678
- ひまわり幼稚園に行っている
- きのう病院に行った
- パパとママがけんかした
- ないしょの話をした
- レゴでお城を作っている
なし"""
    assert clean_notes(raw) == ["恐竜が好き", "レゴでお城を作っている"]
    assert clean_notes("なし") == []
    assert clean_notes(None) == []


def test_old_notes_expire_and_forget(tmp_path):
    m, clock, asked = make(tmp_path)
    m.set_person("p1", "ゆうくん")
    chat(m)
    m.set_person(None)
    m.step()
    clock.t += (KEEP_DAYS + 1) * 86400
    m.step()
    m.set_person("p1", "ゆうくん")
    assert m.prompt_hint() == ""
    chat(m)
    m.set_person(None)
    m.step()
    assert "ゆうくん" in m.report()
    assert m.forget("ゆうくん")
    assert "まだ何も" in m.report()


def test_no_summarizer_means_no_new_notes(tmp_path):
    m = Memory(path=str(tmp_path / "m.json"), clock=Clock(), summarize=None)
    m.set_person("p1", "ゆうくん")
    chat(m)
    m.set_person(None)
    assert m.step() == 0


def test_summarizer_error_does_not_crash(tmp_path):
    def boom(prompt):
        raise OSError("Mac mini が寝ている")
    m = Memory(path=str(tmp_path / "m.json"), clock=Clock(), summarize=boom)
    m.set_person("p1", "ゆうくん")
    chat(m)
    m.set_person(None)
    assert m.step() == 0


def test_conversation_records_turns_and_forgets(tmp_path):
    from robot.conversation import Conversation, SimpleBrain
    m, clock, asked = make(tmp_path)
    m.set_person("p1", "ゆうくん")
    conv = Conversation(SimpleBrain("ピコ"), memory=m)
    conv.handle("こんにちは")
    conv.handle("きょうりゅう すき")
    assert len(m._session["turns"]) == 2
    conv.handle("いまの わすれて")
    assert m._session is None
