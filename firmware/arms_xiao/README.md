# イモの手（XIAO RP2040）

箱の左右の壁にサーボを1個ずつ入れて、手を2本動かします。XIAO RP2040 が Pi から USB で角度を受け取り、サーボを動かします（HyperPixel が Pi のピンをほとんど使うため）。

## 部品

| 部品 | メモ |
|---|---|
| Seeed XIAO RP2040 | USB-C で Pi につなぐ |
| SG90 か MG90S サーボ ×2 | 手は段ボールで軽いので SG90 でも足ります |
| 470µF の電解コンデンサ（あると安心） | サーボが動いた瞬間に Pi が再起動するとき用 |
| ジャンパー線（オス-メス）少し | サーボのコネクタは メス なので、XIAO 側にピンヘッダか線をはんだ付け |

## 配線

```
XIAO 5V ──────┬─────────── 左右サーボの 赤
              └─ コンデンサ ＋
XIAO GND ─────┬─────────── 左右サーボの 茶
              └─ コンデンサ −（帯の印の方）
XIAO D0 ────────────────── イモの左手のサーボの オレンジ（イモから見て左 = 向かい合うと右側）
XIAO D1 ────────────────── イモの右手のサーボの オレンジ
XIAO USB-C ─────────────── Pi の USB
```

- サーボの電気は XIAO の 5V ピン（= Pi の USB）からもらいます。電池の UPS HAT から Pi を通って来ます。
- 手は軽いので2個で足りますが、動きはじめに Pi が再起動するようならコンデンサを足します。

## 書きこみ（最初の1回だけ・Pi で）

1. MicroPython を入れる（Mac でやるのが簡単）。
   https://micropython.org/download/SEEED_XIAO_RP2040/ から最新の `.uf2` をダウンロード。
   XIAO の **B ボタンを押したまま** USB で Mac にさすと、`RPI-RP2` というドライブが出るので、Finder で `.uf2` をそこへドラッグ。ドライブが消えたら完了。XIAO を Pi の USB にさしかえます。

2. 手のプログラムを入れる（イモを止めてから）。

   ```bash
   pkill -f run-robot.sh; pkill -f robot.main
   cd ~/robotformyson && . .venv/bin/activate
   pip install mpremote
   mpremote cp firmware/arms_xiao/main.py :main.py && mpremote reset
   ```

3. 手をおろした位置で止めてから、ホーン（白い羽根）を付けます。1分間その位置で止まっています。

   ```bash
   python3 -m robot.arms down
   ```

4. 動きを試します。

   ```bash
   python3 -m robot.arms test
   ```

   左手 → 右手 → バンザイ → バイバイ（右手）の順に動きます。

## 調整

- 左手と右手が逆: イモの起動に `--arms-swap`
- 手をあげると下（後ろ）に回ってしまう: `main.py` のその手の `LEFT_DIR` / `RIGHT_DIR` を `1 ⇔ -1` に。同じ手の `*_DOWN_US` も反対側の端（700 ⇔ 2300）に変える
- おろした手がまっすぐ下を向かない: `LEFT_DOWN_US` / `RIGHT_DOWN_US` を ±50 ずつ変える
- 手が箱や机にぶつかる: `RANGE` をせまく（例 `(0, 140)`）
- 速すぎる・こわい: `MAX_SPEED` を小さく

変えたら、もう一度 `mpremote cp firmware/arms_xiao/main.py :main.py && mpremote reset`。

## しくみ

Pi は 1秒に最大30回、`A <左手> <右手>`（度。0 = おろした所、150 くらいでバンザイ）を送るだけです。XIAO は速さと加速度を抑えて、なめらかに動かします。
手をおろして1秒止まると力を抜きます（ジーという音を止め、電池も節約）。
Pi から3秒なにも来なければ、ゆっくり手をおろして力を抜きます。
