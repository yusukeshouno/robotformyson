"""ピコの庭と、画面全体のドット表示のテスト。"""
import os
import random
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import pygame  # noqa: E402
import pytest  # noqa: E402

from robot import garden as gd  # noqa: E402
from robot.face import N, OFF, Face  # noqa: E402
from robot.growth import Growth, level_for  # noqa: E402
from robot.screen import DotScreen  # noqa: E402

NOON = time.mktime((2026, 11, 14, 12, 0, 0, 0, 0, -1))
NIGHT = time.mktime((2026, 11, 14, 22, 0, 0, 0, 0, -1))


def make_growth(tmp_path, people):
    g = Growth(str(tmp_path / "growth.json"), autosave=1e9)
    for pid, pts in people.items():
        p = g._get(pid, pid)
        p["points"], p["level"] = pts, level_for(pts)
    return g


def settle(garden, seconds=12.0):
    art = None
    for _ in range(int(seconds * 30)):
        art = garden.update(1 / 30)
    return art


def lit(art, rows=None):
    a = art if rows is None else art[rows]
    return int((a.sum(axis=2) > 1).sum())


@pytest.mark.parametrize("w, h, rows, top", [(480, 800, 80, 16), (720, 720, 48, 0), (800, 480, 48, 0)])
def test_screen_layout(w, h, rows, top):
    s = DotScreen(w, h)
    assert (s.rows, s.top) == (rows, top)
    assert s.size[0] <= w and s.size[1] <= h


def test_plant_shape_is_fixed_per_person(tmp_path):
    a = gd.Garden(make_growth(tmp_path, {"p1": 300}), 80, 16, clock=lambda: NOON)
    b = gd.Garden(make_growth(tmp_path, {"p1": 300}), 80, 16, clock=lambda: NOON)
    a.update(0), b.update(0)
    pa, pb = a.plants["p1"], b.plants["p1"]
    assert np.array_equal(pa.x, pb.x) and np.array_equal(pa.y, pb.y)


def test_plant_grows_with_points(tmp_path):
    sizes = []
    for pts in (0, 100, 300, 600, 1200, 1700):
        g = gd.Garden(make_growth(tmp_path, {"p1": pts}), 80, 16, clock=lambda: NOON)
        settle(g)
        sizes.append(lit(g.update(0), slice(g.top, g.ground - 1)))
    assert sizes == sorted(sizes) and sizes[-1] > sizes[0] * 4


def test_flowers_only_at_level5(tmp_path):
    gold = np.array(gd.GOLD, np.float32)

    def has_gold(pts):
        g = gd.Garden(make_growth(tmp_path, {"p1": pts}), 80, 16, clock=lambda: NOON)
        g.growth.person = "p1"
        art = settle(g)
        return bool(np.all(np.isclose(art, gold), axis=2).any())

    assert not has_gold(600)   # レベル4（しんゆう）はつぼみまで
    assert has_gold(1200)      # レベル5（だいしんゆう）で花


def test_family_plant_when_nobody_is_known(tmp_path):
    g = gd.Garden(make_growth(tmp_path, {}), 48, 0, clock=lambda: NOON)
    g.update(0)
    assert list(g.plants) == ["family"]


def test_at_most_four_plants_and_spread(tmp_path):
    people = {f"p{i}": i * 50 for i in range(1, 7)}
    g = gd.Garden(make_growth(tmp_path, people), 80, 16, clock=lambda: NOON)
    g.update(0)
    assert len(g.plants) == gd.MAX_PLANTS
    assert set(g.plants) == {"p3", "p4", "p5", "p6"}   # なかよし度の高い4人
    xs = [p.x0 for p in g.plants.values()]
    assert xs == sorted(xs) and len(set(xs)) == 4


def test_sky_changes_with_time(tmp_path):
    day = gd.Garden(make_growth(tmp_path, {"p1": 0}), 80, 16, clock=lambda: NOON).update(0)
    night = gd.Garden(make_growth(tmp_path, {"p1": 0}), 80, 16, clock=lambda: NIGHT).update(0)
    assert day[:16].mean() > night[:16].mean()
    square = gd.Garden(make_growth(tmp_path, {"p1": 0}), 48, 0, clock=lambda: NOON).update(0)
    rows = square.mean(axis=(1, 2))
    assert rows[0] > rows[gd.SQUARE_SKY - 1] > 0   # 正方形でも上に空。下へいくほどうすい
    assert square[gd.SQUARE_SKY:20].sum() == 0     # 目のまわりには空を描かない


def test_face_dots_win_over_garden(tmp_path):
    pygame.display.init()
    s = DotScreen(480, 800)
    surf = pygame.Surface(s.size)
    random.seed(3)
    face = Face(s.face_size)
    face.update(1 / 30)
    art = np.full((s.rows, N, 3), 200, np.float32)
    s.draw(surf, face, art)
    random.seed(3)
    plain = Face(s.face_size)
    plain.update(1 / 30)
    plain.render()
    lit_face = np.any(plain.img != np.array(OFF, np.uint8), axis=2)
    assert np.array_equal(face.img[lit_face], plain.img[lit_face])   # 顔の点いたドットはそのまま
    assert (face.img[~lit_face] > np.array(OFF)).all()               # 消えたドットに模様が透ける


def test_fast_enough_for_pi(tmp_path):
    """Pi 4 は PC よりだいぶ遅いので、PC で 1 フレーム 4ms 以内を目安にする（30fps は 33ms）。"""
    pygame.display.init()
    s = DotScreen(480, 800)
    surf = pygame.Surface(s.size)
    face = Face(s.face_size)
    g = gd.Garden(make_growth(tmp_path, {"p1": 1200, "p2": 300, "p3": 90, "p4": 600}),
                  s.rows, s.top, clock=lambda: NIGHT)
    g.growth.person = "p1"
    settle(g, 3)
    n, t0 = 120, time.perf_counter()
    for _ in range(n):
        art = g.update(1 / 30)
    per_garden = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for _ in range(n):
        face.update(1 / 30)
        s.draw(surf, face, art)
    per_draw = (time.perf_counter() - t0) / n
    print(f"庭 {per_garden * 1000:.2f}ms / 顔と合成 {per_draw * 1000:.2f}ms")
    assert per_garden < 0.004
