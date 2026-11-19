"""AI が顔の部品を選んで、その場に合った新しい表情を組み立てる。

返事の感情タグのあとに、AI が部品の組み合わせを書く:

    [joy] {目:star 口:grin 眉:up ほっぺ:2 色:yellow 動き:bounce 小物:note} きょうりゅう だいすき！

決まった表情（face.EMOTIONS）を土台に、書かれた部品だけを差しかえる。
知らない部品は無視するので、AI が少しまちがえても顔はこわれない。
計算は返事ごとに1回だけなので、Pi 4 の負担はほぼない。
"""
import re

# 部品の名前 → 顔のパラメータ（face.DEFAULTS / STYLE_DEFAULTS のキー）
EYES = {
    "normal": {"eyes": ("normal", "normal")},
    "big": {"scale": 1.2, "open": 1.0},
    "small": {"scale": 0.82},
    "happy": {"eyes": ("happy", "happy")},              # ^ ^
    "smile": {"lid_bot": 0.5},                           # 下から笑う目
    "heart": {"eyes": ("heart", "heart")},
    "star": {"eyes": ("star", "star")},
    "squeeze": {"eyes": ("squeeze", "squeeze")},         # > <
    "spiral": {"eyes": ("spiral", "spiral")},
    "wink": {"eyes": ("normal", "happy")},
    "sleepy": {"open": 0.45, "lid_top": 0.45},
    "angry": {"lid_top": 0.2, "tilt": 0.6},
    "sad": {"lid_top": 0.3, "tilt": -0.5},
    "uneven": {"skew": 0.22},                             # 左右で大きさがちがう（？の顔）
    "closed": {"open": 0.0},
}
MOUTH = {
    "smile": {"mouth": 0.8, "mouth_open": 0.0},
    "grin": {"mouth": 0.9, "mouth_open": 0.5},
    "laugh": {"mouth": 1.0, "mouth_open": 1.0, "mouth_w": 1.2},
    "o": {"round": 1.0, "mouth": 0.0},
    "small_o": {"round": 0.6, "mouth": 0.0},
    "flat": {"mouth": 0.0, "mouth_w": 0.7},
    "frown": {"mouth": -0.6, "mouth_w": 0.8},
    "wavy": {"wavy": 1.0, "mouth": 0.0, "mouth_w": 0.75},
    "pout": {"mouth": -0.15, "mouth_w": 0.45},           # むー
    "open": {"mouth": 0.1, "mouth_open": 0.8},           # あーっ
}
BROWS = {
    "none": {"brow": 0.0},
    "up": {"brow": 0.9, "brow_y": 1.0, "brow_tilt": 0.0},
    "angry": {"brow": 1.0, "brow_tilt": 1.0, "brow_y": -0.4},
    "worried": {"brow": 0.8, "brow_tilt": -0.6, "brow_y": 0.3},
    "uneven": {"brow": 0.9, "brow_skew": 1.0},
    "flat": {"brow": 0.7, "brow_tilt": 0.0, "brow_y": 0.0},
}
COLORS = {
    "blue": (90, 220, 255), "cyan": (110, 245, 220), "sky": (150, 210, 255),
    "pink": (255, 120, 190), "yellow": (255, 215, 90), "orange": (255, 160, 70),
    "green": (120, 240, 140), "purple": (200, 160, 255), "red": (255, 90, 80),
    "white": (225, 235, 255),
}
MOTIONS = ("none", "bounce", "jump", "shake", "droop", "sway", "tilt", "giggle", "fidget",
           "breathe")
FX = ("none", "hearts", "sparkle", "tear", "question", "exclaim", "zzz", "note", "sweat",
      "orbit")

# AI が日本語で書いても読めるように
KEY_ALIASES = {
    "目": "eyes", "め": "eyes", "eye": "eyes",
    "口": "mouth", "くち": "mouth",
    "眉": "brows", "まゆ": "brows", "brow": "brows", "eyebrows": "brows",
    "ほっぺ": "cheeks", "頬": "cheeks", "blush": "cheeks", "cheek": "cheeks",
    "色": "color", "いろ": "color", "colour": "color",
    "動き": "motion", "うごき": "motion", "move": "motion",
    "小物": "fx", "こもの": "fx", "effect": "fx", "effects": "fx",
}
VALUE_ALIASES = {
    "ハート": "heart", "星": "star", "ほし": "star", "にっこり": "smile", "えがお": "grin",
    "まる": "o", "ぐるぐる": "spiral", "ウインク": "wink", "ねむい": "sleepy",
    "音符": "note", "おんぷ": "note", "汗": "sweat", "あせ": "sweat", "涙": "tear",
    "なみだ": "tear", "きらきら": "sparkle", "はてな": "question", "びっくり": "exclaim",
    "あお": "blue", "青": "blue", "ピンク": "pink", "きいろ": "yellow", "黄色": "yellow",
    "みどり": "green", "緑": "green", "むらさき": "purple", "紫": "purple", "あか": "red",
    "赤": "red", "オレンジ": "orange", "しろ": "white", "白": "white",
    "smiling": "smile", "sparkles": "sparkle", "heart_eyes": "heart", "music": "note",
    "notes": "note", "tears": "tear",
}

# {目:star 口:grin ...}。全角の ｛｝ や、= の書き方も読む
_SPEC_RE = re.compile(r"[{｛]([^{}｛｝]*)[}｝]")
_PAIR_RE = re.compile(r"([^\s:：=＝,、，]+)\s*[:：=＝]\s*([^\s,、，]+)")

PROMPT_HINT = """- 感情タグのすぐあとに、そのときの顔を部品で組み立ててください。決まった顔でなく、話の内容に合った顔を考えてください:
  {目:? 口:? 眉:? ほっぺ:0〜2 色:? 動き:? 小物:?}
  目: normal big small happy smile heart star squeeze spiral wink sleepy angry sad uneven closed
  口: smile grin laugh o small_o flat frown wavy pout open
  眉: none up angry worried uneven flat
  色: blue cyan sky pink yellow orange green purple red white
  動き: none bounce jump shake droop sway tilt giggle fidget breathe
  小物: none hearts sparkle tear question exclaim zzz note sweat orbit
  例: [excited] {目:star 口:laugh 眉:up ほっぺ:1 色:yellow 動き:bounce 小物:sparkle} すごい！ きょうりゅう だいすき！
  例: [confused] {目:uneven 口:pout 眉:uneven ほっぺ:0 色:purple 動き:tilt 小物:sweat} あれれ、どっちかな？"""


def parse(text):
    """返事から部品の指定を取り出す。(部品の辞書 または None, 指定を取りのぞいた文)。"""
    m = _SPEC_RE.search(text)
    if not m:
        return None, text
    spec = {}
    for k, v in _PAIR_RE.findall(m.group(1)):
        key = KEY_ALIASES.get(k.strip().lower(), k.strip().lower())
        val = v.strip().lower().strip("\"'")
        spec[key] = VALUE_ALIASES.get(val, val)
    rest = (text[:m.start()] + text[m.end():]).strip()
    return (spec or None), _SPEC_RE.sub("", rest).strip()


def compose(spec, base=None):
    """部品の指定 → face.EMOTIONS と同じ形の辞書。base（感情の土台）の上に重ねる。

    使える部品が1つもなければ None（ふつうの感情の顔のままにする）。
    """
    if not spec:
        return None
    out = dict(base or {})
    used = 0
    for key, table in (("eyes", EYES), ("mouth", MOUTH), ("brows", BROWS)):
        part = table.get(spec.get(key, ""))
        if part:
            if key == "eyes" and "eyes" not in part:
                out["eyes"] = ("normal", "normal")   # 数値だけの目は、ふつうの目の形で
            if key == "mouth":
                out["round"] = 0.0                   # o の口から戻す
                out["wavy"] = 0.0
            out.update(part)
            used += 1
    if "cheeks" in spec:
        try:
            out["blush"] = max(0.0, min(1.0, float(spec["cheeks"]) / 2))
            used += 1
        except ValueError:
            pass
    if spec.get("color") in COLORS:
        out["color"] = COLORS[spec["color"]]
        used += 1
    if spec.get("motion") in MOTIONS:
        out["motion"] = None if spec["motion"] == "none" else spec["motion"]
        used += 1
    if spec.get("fx") in FX:
        out["fx"] = () if spec["fx"] == "none" else (spec["fx"],)
        used += 1
    return out if used else None


def to_text(spec):
    """履歴にのこす書き方（次の返事でも同じ書き方をしてもらうため）。"""
    names = {"eyes": "目", "mouth": "口", "brows": "眉", "cheeks": "ほっぺ", "color": "色",
             "motion": "動き", "fx": "小物"}
    return "{" + " ".join(f"{names[k]}:{spec[k]}" for k in names if k in spec) + "}"
