#!/bin/bash
# イモのカメラ（Camera Module 3 / imx708）が起動時に見つからなかったら、ドライバーにもう一度探させる。
# 起動の途中だとカメラの返事が間に合わず「failed to read chip id 708, error -5」になるため
# （Pi 4 ＋ HyperPixel 4.0 Square で確認。起動後に探し直せば見つかる）。
DEV=10-001a
DRV=/sys/bus/i2c/drivers/imx708
for i in 1 2 3 4 5; do
  if [ -e "$DRV/$DEV" ]; then
    echo "imo-camera: ok ($DEV bound)"; exit 0
  fi
  [ -e "/sys/bus/i2c/devices/$DEV" ] || { echo "imo-camera: no $DEV (camera not configured)"; exit 0; }
  echo "imo-camera: try $i: binding $DEV"
  echo "$DEV" > "$DRV/bind" 2>/dev/null
  sleep 2
done
[ -e "$DRV/$DEV" ] && { echo "imo-camera: ok after retry"; exit 0; }
echo "imo-camera: camera still not found"; exit 1
