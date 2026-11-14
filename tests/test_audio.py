"""音声まわりのテスト（マイク・スピーカー・PipeWire がなくても動く）。"""
import json

import numpy as np
import pytest

from robot import audio
from robot import conversation as cv

DEVICES = [
    {"name": "bcm2835 Headphones: - (hw:0,0)", "max_input_channels": 0, "max_output_channels": 8},
    {"name": "USB PnP Sound Device: Audio (hw:2,0)", "max_input_channels": 1,
     "max_output_channels": 0},
    {"name": "pipewire", "max_input_channels": 64, "max_output_channels": 64},
    {"name": "default", "max_input_channels": 64, "max_output_channels": 64},
]


# ---------------------------------------------------------------- デバイスの選択
@pytest.mark.parametrize("spec, expected", [
    (None, None), ("", None), ("  ", None), ("default", None), ("3", 3), (3, 3),
    (" USB ", "USB"),
])
def test_parse_device(spec, expected):
    assert audio.parse_device(spec) == expected


def test_resolve_device_by_name_and_number():
    assert audio.resolve_device("usb", "input", DEVICES) == 1          # 部分一致・大小無視
    assert audio.resolve_device("Headphones", "output", DEVICES) == 0
    assert audio.resolve_device("pipewire", "input", DEVICES) == 2      # 完全一致
    assert audio.resolve_device("2", "output", DEVICES) == 2
    assert audio.resolve_device(None, "input", DEVICES) is None         # システム標準


def test_resolve_device_prefers_exact_match():
    devices = [{"name": "pipewire-extra", "max_input_channels": 1, "max_output_channels": 0},
               {"name": "pipewire", "max_input_channels": 1, "max_output_channels": 0}]
    assert audio.resolve_device("pipewire", "input", devices) == 1


def test_resolve_device_wrong_direction_or_missing_lists_choices():
    with pytest.raises(audio.DeviceNotFound) as e:
        audio.resolve_device("Headphones", "input", DEVICES)   # 出力専用をマイクに
    assert "USB PnP" in str(e.value) and "bcm2835" not in str(e.value)
    with pytest.raises(audio.DeviceNotFound):
        audio.resolve_device("1", "output", DEVICES)           # マイク専用をスピーカーに
    with pytest.raises(audio.DeviceNotFound):
        audio.resolve_device("bluetooth", "output", DEVICES)


def test_settings_priority_args_over_env():
    env = {"ROBOT_MIC": "USB", "ROBOT_SPEAKER": "0", "ROBOT_ECHO_TAIL": "1.2"}
    s = audio.AudioSettings.load(env=env)
    assert (s.mic, s.speaker, s.echo_tail) == ("USB", 0, 1.2)
    s = audio.AudioSettings.load(mic="pipewire", echo_tail=0.2, env=env)
    assert (s.mic, s.speaker, s.echo_tail) == ("pipewire", 0, 0.2)
    s = audio.AudioSettings.load(env={})
    assert (s.mic, s.speaker, s.echo_tail) == (None, None, audio.DEFAULT_ECHO_TAIL)
    assert s.resolve(DEVICES) == (None, None)
    with pytest.raises(ValueError):
        audio.AudioSettings.load(echo_tail=-1, env={})


def test_to_channels():
    mono = np.array([1, 2, 3], dtype=np.int16)
    assert audio.to_channels(mono, 1) is not None and audio.to_channels(mono, 1).ndim == 1
    st = audio.to_channels(mono, 2)
    assert st.shape == (3, 2) and (st[:, 0] == st[:, 1]).all()


# ---------------------------------------------------------------- 話している間は聞かない
class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_gate_closes_while_speaking_and_for_tail():
    clock = Clock()
    gate = audio.MicGate(tail=0.5, clock=clock)
    assert gate.is_open()
    with gate.speaking():
        assert not gate.is_open()
        clock.t += 3
        assert not gate.is_open()
    assert not gate.is_open()          # 余韻の時間
    clock.t += 0.49
    assert not gate.is_open()
    clock.t += 0.02
    assert gate.is_open()


def test_gate_nested_speaking():
    clock = Clock()
    gate = audio.MicGate(tail=0, clock=clock)
    gate.close()
    gate.close()
    gate.open()
    assert not gate.is_open()          # まだ片方が話している
    gate.open()
    assert gate.is_open()
    gate.open()                        # 多く開いても壊れない
    assert gate.is_open()


class FakeRecognizer:
    """b"." を受け取ると文字を1つ足し、b"|" で発話の区切りとして結果を返す。"""

    def __init__(self):
        self.buf, self.resets = "", 0

    def AcceptWaveform(self, data):
        if data == b"|":
            return True
        self.buf += data.decode()
        return False

    def Result(self):
        text, self.buf = self.buf, ""
        return json.dumps({"text": " ".join(text)})

    def Reset(self):
        self.buf = ""
        self.resets += 1


def make_ears(tail=0.5):
    clock = Clock()
    ears = cv.Ears(gate=audio.MicGate(tail=tail, clock=clock), recognizer=FakeRecognizer())
    return ears, clock


def drain(ears):
    out = []
    while not ears.q.empty():
        t = ears.process(ears.q.get_nowait())
        if t:
            out.append(t)
    return out


def test_ears_hears_normally():
    ears, _ = make_ears()
    for chunk in (b"a", b"b", b"|"):
        ears.on_audio(chunk)
    assert drain(ears) == ["ab"]


def test_ears_drops_audio_while_speaking_and_during_tail():
    ears, clock = make_ears(tail=0.5)
    ears.mute(True)
    ears.on_audio(b"x")                 # 自分の声
    ears.mute(False)
    ears.on_audio(b"y")                 # 余韻（残響）
    ears.on_audio(b"|")
    assert ears.q.empty() and drain(ears) == []
    clock.t += 0.6
    for chunk in (b"o", b"k", b"|"):
        ears.on_audio(chunk)
    assert drain(ears) == ["ok"]


def test_ears_discards_words_cut_by_speaking():
    """話しはじめる前に録りかけていた言葉と、まだ処理していない音は捨てる。"""
    ears, clock = make_ears(tail=0.5)
    ears.on_audio(b"h")
    assert drain(ears) == []            # 認識の途中（"h" が途中経過に残る）
    ears.on_audio(b"e")                 # キューに残ったまま話しはじめる
    ears.mute(True)
    ears.mute(False)
    clock.t += 1
    assert drain(ears) == []            # 古い音は捨てる
    for chunk in (b"h", b"i", b"|"):
        ears.on_audio(chunk)
    assert drain(ears) == ["hi"]        # "h" の切れ端は混ざらない
    assert ears.rec.resets == 1


def test_conversation_closes_mic_while_speaking():
    ears, clock = make_ears(tail=0.5)
    states = []

    class Voice:
        def speak(self, text):
            states.append(ears.gate.is_open())

        def mouth_level(self):
            return 0.0

    class Brain:
        vision = False

        def reply(self, text, image=None):
            return "joy", "やあ"

    conv = cv.Conversation(Brain(), Voice(), ears)
    conv.handle("こんにちは")
    assert states == [False]            # 話している間は閉じている
    assert not ears.gate.is_open()      # 余韻の時間
    clock.t += 0.6
    assert ears.gate.is_open()


def test_conversation_reopens_mic_even_if_speaking_fails():
    ears, clock = make_ears(tail=0)

    class Voice:
        def speak(self, text):
            raise RuntimeError("スピーカーがない")

    class Brain:
        vision = False

        def reply(self, text, image=None):
            return "joy", "やあ"

    with pytest.raises(RuntimeError):
        cv.Conversation(Brain(), Voice(), ears).handle("こんにちは")
    assert ears.gate.is_open()


# ---------------------------------------------------------------- PipeWire の設定づくり
PW_DUMP = [
    {"id": 31, "type": "PipeWire:Interface:Node", "info": {"props": {
        "media.class": "Audio/Sink", "node.name": "alsa_output.platform-fef00700.hdmi.hdmi-stereo",
        "node.description": "Built-in Audio Digital Stereo (HDMI)"}}},
    {"id": 40, "type": "PipeWire:Interface:Node", "info": {"props": {
        "media.class": "Audio/Sink", "node.name": "alsa_output.platform-bcm2835_audio.stereo-fallback",
        "node.description": "Built-in Audio Stereo"}}},
    {"id": 52, "type": "PipeWire:Interface:Node", "info": {"props": {
        "media.class": "Audio/Source",
        "node.name": "alsa_input.usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device-00.mono-fallback",
        "node.description": "USB PnP Sound Device Mono"}}},
    {"id": 60, "type": "PipeWire:Interface:Node", "info": {"props": {
        "media.class": "Audio/Source", "node.name": "robot_aec_source"}}},
    {"id": 61, "type": "PipeWire:Interface:Link", "info": {"props": {}}},
    {"id": 62, "type": "PipeWire:Interface:Node", "info": None},
]


def test_pick_node():
    assert audio.pick_node(PW_DUMP, "Audio/Sink", "bcm2835").endswith("bcm2835_audio.stereo-fallback")
    assert audio.pick_node(PW_DUMP, "Audio/Source", "usb").startswith("alsa_input.usb-")
    assert audio.pick_node(PW_DUMP, "Audio/Source", "pnp sound").startswith("alsa_input.usb-")
    assert audio.pick_node(PW_DUMP, "Audio/Source", "robot") is None   # 自分の出口は選ばない
    assert audio.pick_node(PW_DUMP, "Audio/Sink", "usb") is None


def test_node_id():
    assert audio.node_id(PW_DUMP, "robot_aec_source") == 60
    assert audio.node_id(PW_DUMP, "nothing") is None


def test_render_echo_cancel_conf():
    text = audio.render_echo_cancel_conf("alsa_input.usb-x", "alsa_output.platform-bcm2835_audio.y")
    assert "@" not in text   # 置き換え忘れがない
    assert 'target.object  = "alsa_input.usb-x"' in text
    assert 'target.object  = "alsa_output.platform-bcm2835_audio.y"' in text
    assert f'node.name        = "{audio.AEC_SOURCE}"' in text
    assert "aec/libspa-aec-webrtc" in text
    assert text.count("{") == text.count("}") and text.count("[") == text.count("]")
    with pytest.raises(ValueError):
        audio.render_echo_cancel_conf('bad"name', "x")


def test_pipewire_conf_command(monkeypatch, capsys):
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(PW_DUMP)))
    assert audio.main(["pipewire-conf"]) == 0
    out = capsys.readouterr()
    assert "alsa_input.usb-C-Media" in out.out and "bcm2835_audio" in out.out
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(PW_DUMP)))
    assert audio.main(["pipewire-conf", "--mic-match", "bluetooth"]) == 1
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(PW_DUMP)))
    assert audio.main(["node-id", "robot_aec_source"]) == 0
    assert capsys.readouterr().out.strip() == "60"
