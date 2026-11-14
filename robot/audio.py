"""音声デバイスの選び方と、ロボットが話している間マイクを止める仕組み。

構成（安く組む版）:
    USB ミニマイク ──▶ PipeWire（WebRTC エコー除去）──▶ 音声認識
    Pi の 3.5mm 出力 ◀─ PipeWire（エコー除去の参照）◀── 読み上げ
         └─▶ PAM8403 アンプ ─▶ 3W スピーカー

エコー除去はソフトなので完全ではない。そこで「話している間＋話し終えて少しの間」は
マイクの音を捨てる（半二重）。エコー除去は、それ以外の時間の雑音を減らす役。

デバイスは、コマンドの引数 > 環境変数 > システム標準 の順で決まる。
    ROBOT_MIC=USB ROBOT_SPEAKER=Headphones python3 -m robot.main --talk
名前は部分一致（大文字小文字は区別しない）、または番号。

単体でも使える（機材が届いたら、まずこれで確認）:
    python3 -m robot.audio list                  # デバイスの一覧
    python3 -m robot.audio check                 # 音を鳴らしながら録音して、回り込みを測る
"""
import argparse
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

ENV_MIC = "ROBOT_MIC"
ENV_SPEAKER = "ROBOT_SPEAKER"
ENV_ECHO_TAIL = "ROBOT_ECHO_TAIL"
DEFAULT_ECHO_TAIL = 0.6   # 話し終えてから、マイクを戻すまでの秒数（部屋の残響ぶん）

# PipeWire のエコー除去が作る入口・出口の名前（tools/setup_audio.sh と合わせる）
AEC_SOURCE = "robot_aec_source"
AEC_SINK = "robot_aec_sink"


class DeviceNotFound(RuntimeError):
    pass


# ---------------------------------------------------------------- デバイスの選択
def parse_device(spec):
    """'' / None → None（システム標準）、'3' → 3、それ以外は名前の文字列。"""
    if spec is None:
        return None
    if isinstance(spec, int):
        return spec
    spec = str(spec).strip()
    if not spec or spec.lower() == "default":
        return None
    return int(spec) if spec.isdigit() else spec


def _channels_key(kind):
    if kind not in ("input", "output"):
        raise ValueError(kind)
    return "max_input_channels" if kind == "input" else "max_output_channels"


def device_list_text(devices, kind):
    key = _channels_key(kind)
    rows = [f"  {i}: {d['name']}" for i, d in enumerate(devices) if d[key] > 0]
    return "\n".join(rows) or "  （ありません）"


def resolve_device(spec, kind, devices=None):
    """spec を sounddevice のデバイス番号にする。None はシステム標準のまま。

    名前は 完全一致 → 部分一致 の順に探す（大文字小文字は区別しない）。
    見つからなければ、使えるデバイスの一覧つきで DeviceNotFound を出す。
    """
    spec = parse_device(spec)
    if spec is None:
        return None
    if devices is None:
        import sounddevice as sd
        devices = sd.query_devices()
    key = _channels_key(kind)
    usable = [(i, d) for i, d in enumerate(devices) if d[key] > 0]
    if isinstance(spec, int):
        if any(i == spec for i, _ in usable):
            return spec
    else:
        want = spec.lower()
        for match in (lambda n: n == want, lambda n: want in n):
            for i, d in usable:
                if match(d["name"].lower()):
                    return i
    label = "マイク" if kind == "input" else "スピーカー"
    raise DeviceNotFound(f"{label}「{spec}」が見つかりません。使えるもの:\n"
                         + device_list_text(devices, kind))


@dataclass
class AudioSettings:
    mic: object = None        # None / 番号 / 名前
    speaker: object = None
    echo_tail: float = DEFAULT_ECHO_TAIL

    @classmethod
    def load(cls, mic=None, speaker=None, echo_tail=None, env=None):
        """引数 > 環境変数 > 標準 の順で決める。"""
        env = os.environ if env is None else env
        if echo_tail is None:
            raw = env.get(ENV_ECHO_TAIL, "").strip()
            echo_tail = float(raw) if raw else DEFAULT_ECHO_TAIL
        if echo_tail < 0:
            raise ValueError("echo_tail は 0 以上にしてください")
        return cls(mic=parse_device(mic if mic is not None else env.get(ENV_MIC)),
                   speaker=parse_device(speaker if speaker is not None else env.get(ENV_SPEAKER)),
                   echo_tail=float(echo_tail))

    def resolve(self, devices=None):
        """(マイク番号, スピーカー番号) を返す。None はシステム標準。"""
        return (resolve_device(self.mic, "input", devices),
                resolve_device(self.speaker, "output", devices))


def add_audio_args(ap):
    ap.add_argument("--mic", default=None,
                    help=f"マイクの名前の一部か番号（環境変数 {ENV_MIC} でも可。標準はシステムの既定）")
    ap.add_argument("--speaker", default=None,
                    help=f"スピーカーの名前の一部か番号（環境変数 {ENV_SPEAKER} でも可）")
    ap.add_argument("--echo-tail", type=float, default=None,
                    help=f"話し終えてからマイクを戻すまでの秒数（環境変数 {ENV_ECHO_TAIL}。標準 0.6）")


# ---------------------------------------------------------------- 話している間は聞かない
class MicGate:
    """ロボットが話している間と、話し終えてから tail 秒は「閉じる」。

    epoch は閉じるたびに増える。聞く側は、閉じる前に録った音（古い epoch）を捨て、
    認識の途中経過もリセットする。自分の声の切れ端を、子どもの言葉と取りちがえないため。
    """

    def __init__(self, tail=DEFAULT_ECHO_TAIL, clock=time.monotonic):
        self.tail = tail
        self.clock = clock
        self._lock = threading.Lock()
        self._speaking = 0
        self._open_at = 0.0
        self.epoch = 0

    def close(self):
        with self._lock:
            self._speaking += 1
            self.epoch += 1

    def open(self):
        with self._lock:
            self._speaking = max(0, self._speaking - 1)
            if self._speaking == 0:
                self._open_at = self.clock() + self.tail

    def is_open(self):
        with self._lock:
            return self._speaking == 0 and self.clock() >= self._open_at

    @contextmanager
    def speaking(self):
        self.close()
        try:
            yield
        finally:
            self.open()


# ---------------------------------------------------------------- 再生の形
def to_channels(data, channels):
    """モノラルの音を、出力の数（スピーカー2個なら2）にそろえる。"""
    import numpy as np
    data = np.asarray(data)
    if channels <= 1 or data.ndim != 1:
        return data
    return np.repeat(data[:, None], channels, axis=1)


def output_channels(device):
    """出力デバイスのチャンネル数（2 まで）。わからなければ 1。"""
    try:
        import sounddevice as sd
        return max(1, min(2, int(sd.query_devices(device, "output")["max_output_channels"])))
    except Exception:
        return 1


# ---------------------------------------------------------------- PipeWire の設定づくり
def pick_node(pw_dump, media_class, match, exclude=("robot_aec",)):
    """pw-dump の結果から、media.class が一致し名前に match を含むノード名を選ぶ。"""
    want = match.lower()
    for obj in pw_dump:
        props = (obj.get("info") or {}).get("props") or {}
        if props.get("media.class") != media_class:
            continue
        name = props.get("node.name", "")
        text = " ".join(str(props.get(k, "")) for k in
                        ("node.name", "node.description", "node.nick")).lower()
        if any(x in name for x in exclude):
            continue
        if want in text:
            return name
    return None


def node_id(pw_dump, name):
    for obj in pw_dump:
        props = (obj.get("info") or {}).get("props") or {}
        if obj.get("type", "").endswith(":Node") and props.get("node.name") == name:
            return obj.get("id")
    return None


ECHO_CANCEL_TEMPLATE = os.path.join(os.path.dirname(__file__), "..", "deploy", "pipewire",
                                    "60-robot-echo-cancel.conf.in")


def render_echo_cancel_conf(mic_node, speaker_node, template=ECHO_CANCEL_TEMPLATE):
    with open(template, encoding="utf-8") as f:
        text = f.read()
    for key, val in (("@MIC_NODE@", mic_node), ("@SPEAKER_NODE@", speaker_node),
                     ("@AEC_SOURCE@", AEC_SOURCE), ("@AEC_SINK@", AEC_SINK)):
        if '"' in val or "\n" in val:
            raise ValueError(f"使えない文字を含むノード名です: {val!r}")
        text = text.replace(key, val)
    return text


# ---------------------------------------------------------------- 確認用コマンド
def _db(x):
    import numpy as np
    rms = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float32) ** 2)))
    return 20 * np.log10(rms + 1e-9)


def check(settings, seconds=2.0, rate=48000):
    """無音で録音 → 音を鳴らしながら録音、の差を出す。差が小さいほど回り込みが少ない。"""
    import numpy as np
    import sounddevice as sd
    mic, spk = settings.resolve()
    t = np.arange(int(seconds * rate)) / rate
    # 300Hz→3kHz のスイープ（声の帯域）。いきなり大きな音が出ないよう前後をなめらかに
    tone = 0.3 * np.sin(2 * np.pi * (300 * t + (2700 / (2 * seconds)) * t ** 2))
    fade = np.minimum(1, np.minimum(t, t[-1] - t) / 0.05)
    tone = (tone * fade).astype(np.float32)
    quiet = sd.rec(len(tone), samplerate=rate, channels=1, device=mic, dtype="float32")
    sd.wait()
    out = to_channels(tone, output_channels(spk))
    loud = sd.playrec(out, samplerate=rate, input_mapping=[1], device=(mic, spk),
                      dtype="float32")
    sd.wait()
    q, l_ = _db(quiet), _db(loud)
    print(f"静かなとき   : {q:6.1f} dB")
    print(f"鳴らしたとき : {l_:6.1f} dB（差 {l_ - q:+.1f} dB）")
    print("差が 10dB 以下なら、エコー除去がよく効いています。"
          "20dB を超えるなら、マイクとスピーカーを離すか、音量を下げてください。")
    return q, l_


def main(argv=None):
    ap = argparse.ArgumentParser(description="音声デバイスの確認と PipeWire 設定の手伝い")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="マイクとスピーカーの一覧")
    c = sub.add_parser("check", help="鳴らしながら録音して回り込みを測る")
    c.add_argument("--mic")
    c.add_argument("--speaker")
    p = sub.add_parser("pipewire-conf", help="pw-dump の JSON（標準入力）から設定を作る")
    p.add_argument("--mic-match", default="usb")
    p.add_argument("--speaker-match", default="bcm2835")
    n = sub.add_parser("node-id", help="pw-dump の JSON（標準入力）からノード番号を出す")
    n.add_argument("name")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        import sounddevice as sd
        devices = sd.query_devices()
        print("マイク:\n" + device_list_text(devices, "input"))
        print("スピーカー:\n" + device_list_text(devices, "output"))
        print(f"\n指定は --mic / --speaker か、環境変数 {ENV_MIC} / {ENV_SPEAKER} で"
              "（名前の一部か番号）。")
    elif args.cmd == "check":
        check(AudioSettings.load(args.mic, args.speaker))
    elif args.cmd == "pipewire-conf":
        dump = json.load(sys.stdin)
        mic = pick_node(dump, "Audio/Source", args.mic_match)
        spk = pick_node(dump, "Audio/Sink", args.speaker_match)
        if not mic or not spk:
            print(f"見つかりません: マイク={mic} スピーカー={spk}"
                  "（wpctl status で名前を確認し、--mic-match / --speaker-match で指定）",
                  file=sys.stderr)
            return 1
        print(f"マイク: {mic}\nスピーカー: {spk}", file=sys.stderr)
        sys.stdout.write(render_echo_cancel_conf(mic, spk))
    elif args.cmd == "node-id":
        i = node_id(json.load(sys.stdin), args.name)
        if i is None:
            return 1
        print(i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
