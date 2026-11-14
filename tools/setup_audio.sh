#!/usr/bin/env bash
# Pi OS (Bookworm) の PipeWire で、USB マイク＋3.5mm 出力のエコー除去を設定する。
#
#   bash tools/setup_audio.sh            # 設定して、標準のマイク・スピーカーにする
#   bash tools/setup_audio.sh --dry-run  # 作る設定を表示するだけ
#   bash tools/setup_audio.sh --undo     # 元に戻す
#
# マイクやスピーカーの名前がちがうとき（wpctl status で確認）:
#   MIC_MATCH=pnp SPEAKER_MATCH=headphones bash tools/setup_audio.sh
set -euo pipefail

cd "$(dirname "$0")/.."
MIC_MATCH="${MIC_MATCH:-usb}"
SPEAKER_MATCH="${SPEAKER_MATCH:-bcm2835}"
CONF_DIR="$HOME/.config/pipewire/pipewire.conf.d"
CONF="$CONF_DIR/60-robot-echo-cancel.conf"
MODE="${1:-}"

say() { printf '\n== %s\n' "$*"; }
restart_audio() {
  systemctl --user restart pipewire pipewire-pulse wireplumber
  sleep 3
}

if [ "$MODE" = "--undo" ]; then
  rm -f "$CONF"
  restart_audio
  echo "エコー除去の設定を消しました。"
  exit 0
fi

say "1. PipeWire があるか"
if ! command -v pw-dump >/dev/null || ! command -v wpctl >/dev/null; then
  echo "PipeWire を入れます"
  sudo apt install -y pipewire pipewire-pulse wireplumber
fi
if ! systemctl --user is-active --quiet pipewire; then
  systemctl --user enable --now pipewire pipewire-pulse wireplumber
  sleep 3
fi
pipewire --version | head -n 2

say "2. WebRTC のエコー除去が入っているか"
if ! find /usr/lib -path '*spa-0.2/aec/libspa-aec-webrtc.so' 2>/dev/null | grep -q .; then
  echo "libspa-aec-webrtc.so が見つかりません。sudo apt install libspa-0.2-modules を試してください。" >&2
  exit 1
fi
echo "OK"

say "3. マイク（$MIC_MATCH）と 3.5mm 出力（$SPEAKER_MATCH）を探す"
if ! grep -q '^dtparam=audio=on' /boot/firmware/config.txt 2>/dev/null; then
  echo "注意: /boot/firmware/config.txt に dtparam=audio=on がありません（3.5mm 出力が出ないかもしれません）"
fi
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
if ! pw-dump | python3 -m robot.audio pipewire-conf \
      --mic-match "$MIC_MATCH" --speaker-match "$SPEAKER_MATCH" > "$TMP"; then
  echo
  wpctl status | sed -n '/Audio/,/Video/p'
  exit 1
fi

if [ "$MODE" = "--dry-run" ]; then
  cat "$TMP"
  exit 0
fi

say "4. 設定を置いて PipeWire を再起動"
mkdir -p "$CONF_DIR"
cp "$TMP" "$CONF"
echo "$CONF"
restart_audio

say "5. エコー除去したマイク・スピーカーを標準にする"
DUMP="$(pw-dump)"
if ! SRC_ID="$(printf '%s' "$DUMP" | python3 -m robot.audio node-id robot_aec_source)" ||
   ! SINK_ID="$(printf '%s' "$DUMP" | python3 -m robot.audio node-id robot_aec_sink)"; then
  echo "エコー除去のノードができていません。journalctl --user -u pipewire -n 50 を確認してください。" >&2
  echo "元に戻すときは: bash tools/setup_audio.sh --undo" >&2
  exit 1
fi
wpctl set-default "$SRC_ID"
wpctl set-default "$SINK_ID"
wpctl status | sed -n '/Audio/,/Video/p'

cat <<'EOF'

できました。ロボットは何も指定しなければ、エコー除去したマイク・スピーカーを使います。
確認:
  python3 -m robot.audio list     # 一覧
  python3 -m robot.audio check    # 鳴らしながら録音して、回り込みを測る
EOF
