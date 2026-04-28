"""
skills/human_mimicry.py -- Make the agent behave like a real human on WhatsApp.

Humans are messy, inconsistent, and unpredictable. This module simulates:

    1. Read-then-think delay: Humans don't reply instantly after reading.
       They read, pause, think, then start typing.

    2. Variable typing speed: Humans don't type at constant speed.
       They speed up on familiar words, slow down on complex ones,
       pause between thoughts, and occasionally hesitate.

    3. Message splitting: Humans often send multiple short messages
       instead of one long paragraph.

    4. Time-of-day awareness: Humans type faster during the day,
       slower at night. They don't reply at 3am in 2 seconds.

    5. Engagement decay: As conversations go longer, reply times
       increase (fatigue simulation).

    6. "Seen" delay: Sometimes humans read but don't reply immediately.
       They come back to it after a few minutes.

    7. Natural imperfections: Occasional lowercase, missing periods,
       casual abbreviations.
"""

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TypingProfile:
    """Parameters that define how human-like the typing feels."""
    # Characters per second (base rate) - increased for speed
    base_cps: float = 12.0
    # Variance in CPS (jitter)
    cps_jitter: float = 2.0
    # Probability of a mid-message pause (thinking break)
    think_pause_prob: float = 0.1
    # Duration range for think pauses (seconds)
    think_pause_range: Tuple[float, float] = (0.5, 1.5)
    # Probability of a word-boundary micro-pause
    word_pause_prob: float = 0.05
    # Duration range for word micro-pauses
    word_pause_range: Tuple[float, float] = (0.1, 0.3)


class HumanMimicry:
    """
    Simulates realistic human behavior patterns for WhatsApp interaction.

    Controls timing, typing speed, message splitting, and natural imperfections
    to make the agent indistinguishable from a real user.
    """

    # Time-of-day response speed multipliers (0-23 hours)
    # Flattened to keep the agent fast at all times
    HOUR_MULTIPLIERS = {
        0: 1.2, 1: 1.2, 2: 1.2, 3: 1.2, 4: 1.2, 5: 1.2,        # Late night
        6: 1.1, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0,      # Morning
        12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0,   # Afternoon
        18: 1.0, 19: 1.0, 20: 1.0, 21: 1.1, 22: 1.1, 23: 1.2,   # Evening
    }

    def __init__(self):
        self.profile = TypingProfile()
        self._exchange_count: int = 0
        self._last_reply_time: float = 0.0
        self._conversation_start: float = time.time()

    def record_exchange(self):
        """Track that a reply was sent (for fatigue simulation)."""
        self._exchange_count += 1
        self._last_reply_time = time.time()

    # ------------------------------------------------------------------ #
    # 1. Pre-Reply Delay (Read + Think)                                    #
    # ------------------------------------------------------------------ #

    def compute_read_think_delay(self, incoming_text: str, is_proactive: bool = False) -> float:
        """
        How long a human would take to read the message and start thinking
        about a reply. This happens BEFORE typing starts.

        Factors:
            - Message length (longer = more reading time)
            - Time of day (slower at night)
            - Engagement fatigue (longer conversations = slower)
            - Proactive messages need less "reading" time
        """
        if is_proactive:
            return random.uniform(0.5, 1.5) * self._time_of_day_factor()

        word_count = len(incoming_text.split())

        # Base reading time: ~500 words/minute
        read_time = (word_count / 500.0) * 60.0

        # Thinking time
        think_time = random.uniform(0.5, 1.5)

        # Questions get faster replies
        if incoming_text.strip().endswith("?"):
            think_time *= 0.5

        # Long messages get slightly more think time
        if word_count > 30:
            think_time += random.uniform(0.5, 1.5)

        # Time of day factor
        tod_factor = self._time_of_day_factor()

        # Fatigue factor
        fatigue = 1.0 + (self._exchange_count * 0.01)
        fatigue = min(fatigue, 1.2)

        total = (read_time + think_time) * tod_factor * fatigue

        # Floor and cap -- keep it snappy
        total = max(total, 0.5)
        total = min(total, 5.0)

        logger.info(f"HumanMimicry: read+think delay = {total:.1f}s "
                    f"(words={word_count}, tod={tod_factor:.1f}x)")
        return total

    # ------------------------------------------------------------------ #
    # 2. Typing Simulation                                                 #
    # ------------------------------------------------------------------ #

    def compute_typing_duration(self, text: str) -> float:
        """
        How long it would take a human to physically type this message.

        More realistic than the old flat delay -- accounts for:
            - Variable speed per character
            - Pauses between sentences/thoughts
            - Time of day
        """
        char_count = len(text)
        if char_count == 0:
            return 0.0

        # Base typing time
        cps = self.profile.base_cps + random.uniform(
            -self.profile.cps_jitter, self.profile.cps_jitter
        )
        cps = max(cps, 2.0)  # Floor at 2 CPS (very slow typer)
        base_time = char_count / cps

        # Add think pauses (between sentences)
        sentence_count = len(re.split(r'[.!?]+', text))
        pause_time = 0.0
        for _ in range(sentence_count - 1):
            if random.random() < self.profile.think_pause_prob:
                pause_time += random.uniform(*self.profile.think_pause_range)

        # Add word-boundary micro-pauses
        word_count = len(text.split())
        for _ in range(word_count):
            if random.random() < self.profile.word_pause_prob:
                pause_time += random.uniform(*self.profile.word_pause_range)

        # Time of day factor
        total = (base_time + pause_time) * self._time_of_day_factor()

        # Floor and cap
        total = max(total, 0.5)
        total = min(total, 8.0)

        logger.info(f"HumanMimicry: typing duration = {total:.1f}s ({char_count} chars)")
        return total

    async def simulate_typing(self, page, text: str):
        """
        Full human typing simulation with realistic character-by-character timing.

        Uses Playwright's keyboard.type with variable per-character delay.
        """
        # Variable delay per character (ms)
        base_delay_ms = int(1000.0 / self.profile.base_cps)
        min_delay = max(base_delay_ms - 30, 15)
        max_delay = base_delay_ms + 40

        await page.keyboard.type(text, delay=random.randint(min_delay, max_delay))

    # ------------------------------------------------------------------ #
    # 3. Message Splitting                                                 #
    # ------------------------------------------------------------------ #

    def should_split_message(self, text: str) -> bool:
        """
        Decide whether to split a long message into multiple shorter ones.

        Humans frequently send messages in bursts:
            "hey"
            "check this out"
            "what do you think?"
        """
        word_count = len(text.split())

        # Short messages: never split
        if word_count <= 10:
            return False

        # Medium messages: 20% chance of splitting
        if word_count <= 25:
            return random.random() < 0.20

        # Long messages: 50% chance
        return random.random() < 0.50

    def split_message(self, text: str) -> List[str]:
        """
        Split a message into natural chunks at sentence boundaries.

        Returns a list of 2-3 message parts.
        """
        if not self.should_split_message(text):
            return [text]

        # Split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return [text]

        # Group sentences into 2-3 chunks
        if len(sentences) == 2:
            return sentences

        # For 3+ sentences, group into 2-3 parts
        mid = len(sentences) // 2
        parts = [
            " ".join(sentences[:mid]),
            " ".join(sentences[mid:]),
        ]
        return [p for p in parts if p.strip()]

    def inter_message_delay(self) -> float:
        """Delay between split messages (simulates human multi-send behavior)."""
        return random.uniform(0.8, 2.5)

    # ------------------------------------------------------------------ #
    # 4. Natural Imperfections                                             #
    # ------------------------------------------------------------------ #

    def humanize_text(self, text: str) -> str:
        """
        Apply subtle natural imperfections to make text feel more human.

        Changes are probabilistic and subtle:
            - Occasional lowercase first letter (20%)
            - Drop trailing period in short messages (30%)
            - Common abbreviation substitutions (10%)
        """
        if not text:
            return text

        result = text

        # 20% chance: lowercase first letter (casual texting style)
        if random.random() < 0.20 and len(result) > 1:
            result = result[0].lower() + result[1:]

        # 30% chance: drop trailing period in short messages (< 10 words)
        if random.random() < 0.30 and len(result.split()) < 10:
            result = result.rstrip(".")

        # 10% chance: casual substitutions
        if random.random() < 0.10:
            casual_subs = [
                (r"\byou\b", "u"),
                (r"\bare\b", "r"),
                (r"\bthough\b", "tho"),
                (r"\bgoing to\b", "gonna"),
                (r"\bwant to\b", "wanna"),
            ]
            sub = random.choice(casual_subs)
            result = re.sub(sub[0], sub[1], result, count=1)

        return result

    # ------------------------------------------------------------------ #
    # 5. Polling Interval Variation                                        #
    # ------------------------------------------------------------------ #

    def compute_poll_interval(self, base_interval: float, is_active: bool) -> float:
        """
        Vary the polling interval to avoid robotic regularity.

        Active conversation: shorter, tighter intervals
        Idle/waiting: longer, more variable intervals
        """
        if is_active:
            # Active chat: very fast interval
            interval = base_interval + random.uniform(-0.5, 0.5)
            return max(interval, 1.0)
        else:
            # Idle: just base interval
            return base_interval

    # ------------------------------------------------------------------ #
    # Internal Helpers                                                     #
    # ------------------------------------------------------------------ #

    def _time_of_day_factor(self) -> float:
        """Get the response speed multiplier for the current hour."""
        hour = time.localtime().tm_hour
        return self.HOUR_MULTIPLIERS.get(hour, 1.0)
