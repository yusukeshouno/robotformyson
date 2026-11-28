"""カメラの「顔がいる」を、本当に人がいるかに近づける。

実機で、だれもいないのに部屋の何か（写真・ポスター・ぬいぐるみなど）を顔とまちがえ続け、
イモが眠らず、ひとりで絵をかいたり話したりした。そこで
- 顔が CONFIRM 秒つづけて見えたら「いる」（一瞬の見まちがいは数えない）
- 顔の位置が STILL_SECONDS 秒ほとんど動かなければ、写真などの「動かない顔」として「いない」
とする。人はじっとしているつもりでも、少しは動く。
"""
from collections import deque


class PresenceFilter:
    CONFIRM = 1.5        # 秒
    STILL_SECONDS = 90.0
    STILL_RANGE = 0.03   # 顔の位置（-1〜1）がこの幅より動かなければ「動かない顔」

    def __init__(self):
        self.since = None
        self.track = deque()
        self.still = False

    def update(self, raw, x, y, now):
        if not raw:
            self.since, self.still = None, False
            self.track.clear()
            return False
        if self.since is None:
            self.since = now
        self.track.append((now, x, y))
        while self.track and now - self.track[0][0] > self.STILL_SECONDS:
            self.track.popleft()
        if now - self.since >= self.STILL_SECONDS:
            xs = [p[1] for p in self.track]
            ys = [p[2] for p in self.track]
            still = max(xs) - min(xs) < self.STILL_RANGE and max(ys) - min(ys) < self.STILL_RANGE
            if still and not self.still:
                print("[顔] ずっと動かない顔は、写真などとみなします")
            self.still = still
        return now - self.since >= self.CONFIRM and not self.still
