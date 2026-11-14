# イモの首（XIAO RP2040）

## 部品

| 部品 | 目安 |
|---|---|
| Seeed XIAO RP2040 | 997円 |
| MG90S サーボ ×2（左右用・上下用） | 1,500円前後（4個入り） |
| SG90/MG90S 用 2軸ブラケット | 数百円 |
| 5V 2A 以上の AC アダプタ＋DC ジャック（サーボ専用） | 手持ちがあれば 0円 |
| 470µF〜1000µF の電解コンデンサ（あると安心） | 数十円 |

## 配線

```
5V アダプタ ＋ ───────────── サーボ2個の 赤
5V アダプタ − ──┬────────── サーボ2個の 茶
                └────────── XIAO の GND
XIAO D0 ─────────────────── 左右サーボの オレンジ
XIAO D1 ─────────────────── 上下サーボの オレンジ
XIAO USB-C ──────────────── Pi の USB
（コンデンサはサーボの 赤・茶 の間に。− の印を 茶 側に）
```

- サーボの電源は Pi から取りません。動きはじめに電気をたくさん使い、Pi の電圧が下がって再起動することがあるためです。
- アダプタの 5V を XIAO の 5V ピンにはつながないでください（USB とぶつかります）。GND だけ共通にします。
- 頭（Pi＋HyperPixel）は100g ほどあります。上下サーボの軸の真上に重心が来るように載せると、ジーという音が出にくくなります。

## 書きこみ（最初の1回だけ・Pi で）

1. MicroPython を入れる。XIAO の B ボタンを押したまま USB で Pi にさすと、`RPI-RP2` というドライブが出ます。
   https://micropython.org/download/SEEED_XIAO_RP2040/ から `.uf2` を落として、そのドライブにコピーします。

   ```bash
   cp ~/Downloads/SEEED_XIAO_RP2040-*.uf2 /media/$USER/RPI-RP2/
   ```

2. 首のプログラムを入れる（イモを止めてから）。

   ```bash
   cd ~/robotformyson && . .venv/bin/activate
   pip install mpremote
   mpremote cp firmware/neck_xiao/main.py :main.py && mpremote reset
   ```

3. サーボを真ん中にしてから、ホーン（腕）を付けます。

   ```bash
   python3 -m robot.neck center
   ```

4. 動きを試します。

   ```bash
   python3 -m robot.neck test
   ```

   左 → 右 → 上 → 下 → うなずき → 首ふり の順に動きます。

## 調整

- 人と逆の方を向く: イモの起動に `--neck-invert-pan`
- うなずくと上を向く: `--neck-invert-tilt`
- 真ん中がずれている: `main.py` の `PAN_CENTER_US` / `TILT_CENTER_US` を ±50 ずつ変えて、もう一度 `mpremote cp`
- 動きが速すぎる・こわい: `MAX_SPEED`（度/秒）を小さく
- カメラを頭ではなく体に付けた: `--neck-camera body`

## しくみ

Pi は 1秒に最大30回、`N <左右> <上下>`（度）を送るだけです。XIAO は速さと加速度を抑えて、なめらかに動かします。
左右のサーボは止まって 1.5秒たつと力を抜きます（ジーという音を止めるため）。
Pi から3秒なにも来なければ、ゆっくり真ん中に戻って力を抜きます。
