# カメラが起動のたびに見えるようにする（Camera Module 3）

Pi 4 で Camera Module 3 をつなぐと、起動の途中でカメラの返事が間に合わず、
`imx708 ... failed to read chip id 708, with error -5` となってカメラが使えないことがあります。
起動が終わってからドライバーに探し直させれば見つかるので、そのための小さな仕組みです。

```bash
sudo install -m 755 deploy/camera/imo-camera-rebind.sh /usr/local/bin/
sudo install -m 644 deploy/camera/imo-camera.service /etc/systemd/system/
sudo systemctl enable --now imo-camera.service
rpicam-hello --list-cameras     # imx708_wide などが出れば OK
```

イモを自動で起動するスクリプトから先に呼ぶなら、`sudo -n /usr/local/bin/imo-camera-rebind.sh` を
イモを起動する前に1行入れます（sudo をパスワードなしで使える pi ユーザーの場合）。
