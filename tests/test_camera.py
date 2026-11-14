"""カメラ選びのテスト（本物のカメラがなくても動く。Pi 4 の /sys を小さくまねする）。"""
import os

import pytest

from robot import audio, camera


def _video(root, n, name, driver, usb, index=0):
    """/sys/class/video4linux/videoN を作る。device は本物と同じく別の場所へのリンク。"""
    bus = root / "devices" / ("usb1/1-1/1-1.3:1.0" if usb else f"platform/soc/codec{n}")
    bus.mkdir(parents=True, exist_ok=True)
    drv = root / "drivers" / driver
    drv.mkdir(parents=True, exist_ok=True)
    if not (bus / "driver").exists():
        os.symlink(drv, bus / "driver")
    entry = root / "class" / f"video{n}"
    entry.mkdir(parents=True)
    (entry / "name").write_text(name + "\n")
    (entry / "index").write_text(f"{index}\n")
    os.symlink(bus, entry / "device")


@pytest.fixture
def pi_sysfs(tmp_path):
    """Pi 4 ＋ USB カメラ（400-CAM103）をさしたときの様子。"""
    _video(tmp_path, 10, "bcm2835-codec-decode", "bcm2835-codec", usb=False)
    _video(tmp_path, 11, "bcm2835-codec-encode", "bcm2835-codec", usb=False)
    _video(tmp_path, 13, "bcm2835-isp", "bcm2835-isp", usb=False)
    _video(tmp_path, 0, "400-CAM103: 400-CAM103", "uvcvideo", usb=True)
    _video(tmp_path, 1, "400-CAM103: 400-CAM103", "uvcvideo", usb=True, index=1)  # メタデータ
    return str(tmp_path / "class")


def test_list_skips_metadata_node_and_classifies(pi_sysfs, tmp_path):
    devs = camera.list_devices(pi_sysfs, by_id=str(tmp_path / "none"))
    assert [d.number for d in devs] == [0, 10, 11, 13]
    assert [d.kind for d in devs] == ["usb", "internal", "internal", "internal"]
    assert devs[0].driver == "uvcvideo"


def test_find_camera_auto_picks_usb_not_pi_codec(pi_sysfs, tmp_path):
    devs = camera.list_devices(pi_sysfs, by_id=str(tmp_path / "none"))
    assert camera.find_camera(None, devs).path == "/dev/video0"
    assert camera.find_camera("auto", devs).path == "/dev/video0"
    assert camera.find_camera("cam103", devs).number == 0      # 名前の一部
    assert camera.find_camera("13", devs).name == "bcm2835-isp"  # 番号はそのまま
    assert camera.find_camera("/dev/video5", devs).path == "/dev/video5"
    with pytest.raises(RuntimeError, match="400-CAM103"):
        camera.find_camera("logitech", devs)                    # 一覧つきで教える


def test_find_camera_none_without_usb(tmp_path):
    _video(tmp_path, 10, "bcm2835-codec-decode", "bcm2835-codec", usb=False)
    _video(tmp_path, 0, "unicam-image", "unicam", usb=False)
    devs = camera.list_devices(str(tmp_path / "class"), by_id=str(tmp_path / "none"))
    assert camera.find_camera(None, devs) is None
    assert camera.has_csi_camera(devs)


def test_settings_defaults_env_and_args():
    s = camera.CameraSettings.load(env={})
    assert (s.device, s.width, s.height, s.fps, s.fourcc) == (None, 640, 480, 15, "MJPG")
    env = {"ROBOT_CAMERA": "CAM103", "ROBOT_CAMERA_SIZE": "320x240",
           "ROBOT_CAMERA_FPS": "10", "ROBOT_CAMERA_FORMAT": "yuyv"}
    s = camera.CameraSettings.load(env=env)
    assert (s.device, s.width, s.height, s.fps, s.fourcc) == ("CAM103", 320, 240, 10, "YUYV")
    s = camera.CameraSettings.load("1", "1280x720", 30, "MJPG", env=env)  # 引数が勝つ
    assert (s.device, s.width, s.fps, s.fourcc) == ("1", 1280, 30, "MJPG")
    with pytest.raises(ValueError):
        camera.CameraSettings.load(size="big", env={})
    with pytest.raises(ValueError):
        camera.CameraSettings.load(fourcc="H264", env={})


def test_old_camera_index_option_still_works():
    from robot.main import parse_args
    assert parse_args(["--camera-index", "2"]).camera_device == "2"
    assert parse_args(["--camera-device", "CAM103"]).camera_device == "CAM103"


@pytest.mark.parametrize("name, cam", [
    ("400-CAM103: USB Audio (hw:3,0)", True),
    ("USB2.0 PC CAMERA: Audio (hw:2,0)", True),
    ("HD Webcam C270: USB Audio", True),
    ("USB PnP Sound Device: Audio (hw:2,0)", False),
    ("pipewire", False),
])
def test_looks_like_camera(name, cam):
    assert camera.looks_like_camera(name) is cam


def test_mic_usb_skips_webcam_mic():
    devices = [  # カメラの方が先に並んでいても、USB ミニマイクを選ぶ
        {"name": "400-CAM103: USB Audio (hw:2,0)", "max_input_channels": 1, "max_output_channels": 0},
        {"name": "USB PnP Sound Device: Audio (hw:3,0)", "max_input_channels": 1,
         "max_output_channels": 0},
    ]
    assert audio.resolve_device("usb", "input", devices) == 1
    assert audio.resolve_device("CAM103", "input", devices) == 0   # 名指しならカメラのマイクも使える


def test_pipewire_mic_skips_webcam_mic():
    def node(name, desc):
        return {"type": "PipeWire:Interface:Node",
                "info": {"props": {"media.class": "Audio/Source", "node.name": name,
                                   "node.description": desc}}}
    dump = [node("alsa_input.usb-SANWA_400-CAM103-02.mono-fallback", "400-CAM103 Mono"),
            node("alsa_input.usb-C-Media_USB_PnP_Sound_Device-00.mono-fallback",
                 "USB PnP Sound Device Mono")]
    assert "PnP" in audio.pick_node(dump, "Audio/Source", "usb")
    assert "CAM103" in audio.pick_node(dump, "Audio/Source", "cam103")


def test_tracker_reconnects_after_unplug(monkeypatch):
    """USB を抜いて画像が来なくなったら、つなぎなおす。"""
    import time

    import numpy as np

    from robot import vision
    opened = []

    def fake_open(settings):
        opened.append(settings)
        if len(opened) == 1:
            return (lambda: None), (lambda: None), "fake (抜けた)"  # 何も来ない
        return (lambda: np.zeros((480, 640, 3), np.uint8)), (lambda: None), "fake"
    monkeypatch.setattr(vision, "open_opencv", fake_open)
    monkeypatch.setattr(vision, "find_camera", lambda spec: object())  # USB カメラあり
    t = vision.FaceTracker("auto", camera.CameraSettings.load(env={})).start()
    deadline = time.monotonic() + 5
    while t.latest() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    t.stop()
    assert len(opened) >= 2 and t.latest() is not None and t.label == "fake"
