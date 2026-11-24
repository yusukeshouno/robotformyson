"""感情エンジン：顔の検出・タッチ・外部からの指示を受けて、今の感情を決める。"""
import math
import random
from collections import deque


class EmotionEngine:
    SLEEP_AFTER = 120.0  # 顔も見えず、話しかけられもしない状態がこの秒数続くと眠る（起きるのは「イモ」と呼ばれたとき）
    QUIET_SLEEP_AFTER = 300.0  # 顔が見えていても、だれも話しかけない状態がこの秒数続くと眠る
    LOVE_EVERY = 20.0    # 見つめられ続けると時々ハート目
    # 返事の気持ちの「余韻」。話し終えたあとも少しのあいだ顔に残り、ゆっくり消える。
    # 悲しい・怒りは子どもを引き止めないよう、短く・弱く残す。
    MOOD_WEIGHT = 0.45
    MOOD_FADE = {"sad": 4.0, "cry": 4.0, "angry": 3.0}  # 秒（この時間で約 1/3 になる）
    MOOD_FADE_DEFAULT = 15.0
    NO_MOOD = ("neutral", "think", "sleepy", "dizzy")

    def __init__(self, now):
        self.queue = deque()        # (感情, 秒数) の予約
        self.current = None
        self.current_until = 0.0
        self.present = False
        self.present_since = now
        self.last_activity = now    # 最後に人の気配があった時刻
        self.last_talk = now        # 最後に話しかけられた時刻
        self.last_love = now
        self.sleeping = False
        self.taps = deque()
        self.mood_name, self.mood_since = "neutral", now
        self.last_seen = now        # 最後に顔が見えていた時刻

    # ---- 外部から ----
    def express(self, emotion, duration, now, interrupt=True):
        """一時的に感情を出す。LLM の感情タグもここに渡す。"""
        if interrupt:
            self.queue.clear()
            self.current_until = now
        self.queue.append((emotion, duration))

    def set_mood(self, emotion, now):
        """会話の返事の気持ちを余韻として残す。"""
        if emotion not in self.NO_MOOD:
            self.mood_name, self.mood_since = emotion, now

    def mood(self, now):
        """(余韻の感情, 強さ 0..MOOD_WEIGHT)。眠っているときや何か表情を出しているときは 0。"""
        if self.mood_name == "neutral" or self.sleeping:
            return "neutral", 0.0
        fade = self.MOOD_FADE.get(self.mood_name, self.MOOD_FADE_DEFAULT)
        w = self.MOOD_WEIGHT * math.exp(-(now - self.mood_since) / fade)
        if w < 0.03:
            self.mood_name = "neutral"
            return "neutral", 0.0
        return self.mood_name, w

    def notice(self, now):
        """声をかけられたなど、人の気配があったとき。眠っていれば起きる。"""
        self.last_activity = now
        self.last_talk = now
        return self._wake(now)

    def _wake(self, now):
        if self.sleeping:
            self.sleeping = False
            self.express("surprise", 1.0, now)
            self.express("joy", 2.5, now, interrupt=False)
            return True
        return False

    # ---- 入力イベント ----
    def on_face(self, present, now):
        away = now - self.last_seen      # 顔が見えなくなってからの時間
        if present:
            self.last_activity = now
            self.last_seen = now
        if present == self.present:
            return
        self.present = present
        if self.sleeping:                # 眠っているあいだは、顔が見えても起きない（「イモ」で起きる）
            self.present_since = now
            return
        if present:
            self.present_since = now
            if not self._wake(now):
                # しばらくぶりに会えたら、目がきらきら
                self.express("excited" if away > 60.0 else "joy", 2.0, now)
        elif now - self.present_since > 3.0:
            self.express("sad", 2.0, now)

    def on_tap(self, now):
        self.last_activity = now
        if self._wake(now):
            return
        self.taps.append(now)
        while self.taps and now - self.taps[0] > 5.0:
            self.taps.popleft()
        recent = sum(1 for t in self.taps if now - t <= 2.0)
        if len(self.taps) >= 10:         # それでも連打されると目が回る
            self.taps.clear()
            self.express("dizzy", 3.0, now)
        elif recent >= 5:                # 連打されると怒る
            self.express("angry", 2.5, now)
        elif recent == 3:                # 3回目はくすぐったくて笑う
            self.express("laugh", 1.5, now)
        else:
            self.express("joy", 1.5, now)

    def on_stroke(self, now):            # なでられると照れて、それから大好き
        self.last_activity = now
        if not self._wake(now):
            self.express("shy", 1.5, now)
            self.express("love", 2.5, now, interrupt=False)

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

        if self.sleeping:
            return "sleepy"
        if self.QUIET_SLEEP_AFTER and now - self.last_talk > self.QUIET_SLEEP_AFTER:
            self.sleeping = True     # 顔は見えていても、しばらく話しかけられなければ眠る
            return "sleepy"
        if self.present:
            if now - self.last_love > self.LOVE_EVERY and now - self.present_since > 10:
                self.last_love = now
                self.express(random.choice(["love", "joy", "wink", "laugh"]), 2.5, now)
            return "neutral"
        if now - self.last_activity > self.SLEEP_AFTER:
            self.sleeping = True
            return "sleepy"
        return "neutral"
