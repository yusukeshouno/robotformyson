"""感情エンジン：顔の検出・タッチ・外部からの指示を受けて、今の感情を決める。"""
import random
from collections import deque


class EmotionEngine:
    SLEEP_AFTER = 40.0   # 誰もいない状態がこの秒数続くと眠る
    LOVE_EVERY = 20.0    # 見つめられ続けると時々ハート目

    def __init__(self, now):
        self.queue = deque()        # (感情, 秒数) の予約
        self.current = None
        self.current_until = 0.0
        self.present = False
        self.present_since = now
        self.last_activity = now    # 最後に人の気配があった時刻
        self.last_love = now
        self.sleeping = False
        self.taps = deque()

    # ---- 外部から ----
    def express(self, emotion, duration, now, interrupt=True):
        """一時的に感情を出す。LLM の感情タグもここに渡す。"""
        if interrupt:
            self.queue.clear()
            self.current_until = now
        self.queue.append((emotion, duration))

    def _wake(self, now):
        if self.sleeping:
            self.sleeping = False
            self.express("surprise", 1.0, now)
            self.express("joy", 2.5, now, interrupt=False)
            return True
        return False

    # ---- 入力イベント ----
    def on_face(self, present, now):
        if present:
            self.last_activity = now
        if present == self.present:
            return
        self.present = present
        if present:
            self.present_since = now
            if not self._wake(now):
                self.express("joy", 2.0, now)
        elif now - self.present_since > 3.0:
            self.express("sad", 2.0, now)

    def on_tap(self, now):
        self.last_activity = now
        if self._wake(now):
            return
        self.taps.append(now)
        while self.taps and now - self.taps[0] > 2.0:
            self.taps.popleft()
        if len(self.taps) >= 5:          # 連打されると怒る
            self.taps.clear()
            self.express("angry", 2.5, now)
        else:
            self.express("joy", 1.5, now)

    def on_stroke(self, now):            # なでられると照れる
        self.last_activity = now
        if not self._wake(now):
            self.express("love", 3.0, now)

    # ---- 毎フレーム ----
    def update(self, now):
        if now >= self.current_until:
            if self.queue:
                emo, dur = self.queue.popleft()
                self.current, self.current_until = emo, now + dur
            else:
                self.current = None
        if self.current:
            return self.current

        if self.present:
            if now - self.last_love > self.LOVE_EVERY and now - self.present_since > 10:
                self.last_love = now
                self.express(random.choice(["love", "joy"]), 2.5, now)
            return "neutral"
        if now - self.last_activity > self.SLEEP_AFTER:
            self.sleeping = True
            return "sleepy"
        return "neutral"
