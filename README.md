# robotformyson

Raspberry Pi 4 ＋小型正方形ディスプレイ＋カメラで動く、エンタメ用ロボットの「顔」デモ。

![表情一覧](docs/expressions.png)

- 8つの表情（ふつう・うれしい・かなしい・おこる・びっくり・ねむい・だいすき・かんがえちゅう）
- まばたき、視線のゆらぎ、呼吸のような揺れを自動で付与
- カメラで顔を見つけて、目で追いかける
- 人が来ると喜び、去ると悲しみ、40秒放置すると眠る（人が来ると起きる）
- タップで喜び、5連打で怒り、なでる（ドラッグ）と照れる
- 会話：話しかけると Claude が考えて、感情つきで声で答える（話している間は口が動く）
- 「これなに？」と聞くと、カメラの画像を見て答える
- UDP で感情タグを受け付ける（外部プログラムとの連携口）

## Raspberry Pi で動かす

Raspberry Pi OS（Bookworm）を想定しています。

```bash
sudo apt install -y python3-pygame python3-opencv python3-picamera2
git clone <このリポジトリ> && cd robotformyson
python3 -m robot.main --fullscreen
```

- カメラモジュールは picamera2、USB カメラは OpenCV で自動的に選ばれます。
- 目が人と逆の方向を見るときは `--no-mirror` を付けてください。
- カメラを横向き・逆さに付けたときは `--rotate 90`（または 180, 270）で補正します。

## 会話機能を使う

聞く（Vosk）と話す（Open JTalk）は Pi の上でオフラインで動き、ネットに出るのは Claude への問い合わせだけです。
必要なもの：USB スピーカーフォン（マイクとスピーカー）と、Claude の API キー。

```bash
# 1. 音声合成と音声入出力のライブラリ
sudo apt install -y open-jtalk open-jtalk-mecab-naist-jdic hts-voice-nitech-jp-atr503-m001 libportaudio2

# 2. Python のライブラリ（apt で入れた pygame などもそのまま使える仮想環境）
python3 -m venv --system-site-packages .venv && . .venv/bin/activate
pip install anthropic vosk sounddevice

# 3. 日本語の音声認識モデル（約50MB）
mkdir -p models && cd models
wget https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip && unzip vosk-model-small-ja-0.22.zip && cd ..

# 4. API キー（https://platform.claude.com で発行）
export ANTHROPIC_API_KEY=...

python3 -m robot.main --fullscreen --talk
```

- **いつ返事をするか：** カメラに顔が見えているとき（10秒以内）か、名前（標準は「ピコ」）を呼ばれたときです。返事のあと15秒は、続けて話しかけられます。テレビの音などに反応しすぎないためです。
- **名前と声：** `--name ロボ` で名前、`--pitch 3` で声の高さを変えられます。もっとかわいい声にしたいときは、MMDAgent の「mei」の声ファイルを `/usr/share/hts-voice/mei/` に置くと自動で使います（`--voice` でファイルを指定することもできます）。
- **マイクが違うとき：** `python3 -m sounddevice` で一覧を出し、`--mic 番号` で指定します。
- **会話だけ試す：** `python3 -m robot.conversation --text` でキーボードから話しかけられます（マイク不要）。
- **モデル：** 標準は `claude-opus-5-5`（考える深さは low）。返事を速く・安くしたいときは `--model claude-haiku-5-5` を試してください。
- **費用の目安：** Opus 5.5 で1往復およそ1〜2円（100往復で100〜200円程度）の見込みです。実際の請求は Claude Console で確認してください。

### 子ども向けの安全とプライバシー

- システムプロンプト（`robot/conversation.py` の `SYSTEM_PROMPT`）で、短くやさしい返事・危ないことはおうちの人に聞く・個人情報を聞かない、と指示しています。
- Claude が安全上の理由で答えを断ったときは、サーバー側で別のモデルに自動で切り替えます（`fallbacks="default"`）。それでも断られた場合は「そのおはなしはできない」と答えます。
- 送るのは聞き取った文字だけで、音声そのものは送りません。カメラの画像は「これ」「見て」などの言葉があったときだけ、その1枚を送ります。
- 会話の履歴は直近10往復だけをメモリに置き、ファイルには保存しません。

## PC で試す

```bash
pip install -r requirements.txt
python3 -m robot.main            # Web カメラがあれば顔を追う
python3 -m robot.main --mouse    # カメラなし：マウスを人の顔として扱う
```

## 操作

| 入力 | 動作 |
|---|---|
| `1`〜`8` | neutral / joy / sad / angry / surprise / sleepy / love / think を4秒出す |
| `0` | 自動モードに戻す |
| `c` | カメラ映像と検出枠を表示 |
| `d` | 感情名・顔検出・FPS を表示 |
| `q` / `Esc` | 終了 |
| タップ・ドラッグ | 喜ぶ・照れる（タッチ対応ディスプレイならそのまま使える） |

## 外部プログラムとの連携口

ポート 5005（localhost）に `感情名 秒数` を UDP で送ると、その表情を出します。

```bash
echo "joy 3" | nc -u -w0 127.0.0.1 5005
```

別のプログラムから表情を動かしたいときに使います。

## 構成

| ファイル | 役割 |
|---|---|
| `robot/face.py` | 表情の描画。感情ごとのパラメータを補間して描く |
| `robot/emotion.py` | 感情エンジン。顔・タッチ・外部指示から今の感情を決める |
| `robot/vision.py` | カメラと顔検出（別スレッド） |
| `robot/conversation.py` | 会話：聞く（Vosk）・考える（Claude）・話す（Open JTalk） |
| `robot/main.py` | 起動・入力処理・画面更新 |
| `tools/snapshot.py` | 全表情を1枚の画像に書き出す（確認用） |
