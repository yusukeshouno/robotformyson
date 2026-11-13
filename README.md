# robotformyson

Raspberry Pi 4 ＋小型正方形ディスプレイ＋カメラで動く、エンタメ用ロボットの「顔」デモ。

![表情一覧](docs/expressions.png)

- 7つの表情（ふつう・うれしい・かなしい・おこる・びっくり・ねむい・だいすき）
- まばたき、視線のゆらぎ、呼吸のような揺れを自動で付与
- カメラで顔を見つけて、目で追いかける
- 人が来ると喜び、去ると悲しみ、40秒放置すると眠る（人が来ると起きる）
- タップで喜び、5連打で怒り、なでる（ドラッグ）と照れる
- UDP で感情タグを受け付ける（会話 AI との連携口）

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

## PC で試す

```bash
pip install -r requirements.txt
python3 -m robot.main            # Web カメラがあれば顔を追う
python3 -m robot.main --mouse    # カメラなし：マウスを人の顔として扱う
```

## 操作

| 入力 | 動作 |
|---|---|
| `1`〜`7` | neutral / joy / sad / angry / surprise / sleepy / love を4秒出す |
| `0` | 自動モードに戻す |
| `c` | カメラ映像と検出枠を表示 |
| `d` | 感情名・顔検出・FPS を表示 |
| `q` / `Esc` | 終了 |
| タップ・ドラッグ | 喜ぶ・照れる（タッチ対応ディスプレイならそのまま使える） |

## 会話 AI との連携口

ポート 5005（localhost）に `感情名 秒数` を UDP で送ると、その表情を出します。

```bash
echo "joy 3" | nc -u -w0 127.0.0.1 5005
```

LLM の返答に感情タグを付けさせ、それをここへ送る想定です。

## 構成

| ファイル | 役割 |
|---|---|
| `robot/face.py` | 表情の描画。感情ごとのパラメータを補間して描く |
| `robot/emotion.py` | 感情エンジン。顔・タッチ・外部指示から今の感情を決める |
| `robot/vision.py` | カメラと顔検出（別スレッド） |
| `robot/main.py` | 起動・入力処理・画面更新 |
| `tools/snapshot.py` | 全表情を1枚の画像に書き出す（確認用） |
