"""
skills/silence_policy.py -- Smart silence handling to prevent token waste.

Controls WHEN the agent invokes LLM reasoning during idle cycles.

Policies:
    - After outgoing message: 30s cooldown before considering proactive
    - For business/bot contacts: never go proactive
    - Exponential backoff on repeated silence: check less frequently
    - Rate limit: max 6 LLM strategist calls per minute
    - outgoing_seen: fast-path only during cooldown, then allows initiation
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class SilencePolicy:
    """
    Determines whether the agent should invoke LLM reasoning during silence.
    """

    POST_SEND_COOLDOWN = 10.0          # Wait after sending before going proactive
    MIN_SILENCE_CHECK_INTERVAL = 5.0  # Minimum gap between LLM silence checks
    BACKOFF_MULTIPLIER = 1.5
    MAX_SILENCE_INTERVAL = 300.0       # Cap backoff at 5 minutes
    MAX_LLM_CALLS_PER_MINUTE = 20

    def __init__(self):
        self._last_outgoing_time: float = 0.0
        self._last_silence_check_time: float = 0.0
        self._consecutive_silence_noop: int = 0
        self._current_interval: float = self.MIN_SILENCE_CHECK_INTERVAL
        self._llm_call_timestamps: list = []

    def record_outgoing_message(self):
        """Call this after the agent sends a message."""
        self._last_outgoing_time = time.time()
        self._consecutive_silence_noop = 0
        self._current_interval = self.MIN_SILENCE_CHECK_INTERVAL

    def record_silence_noop(self):
        """Call this when a silence check resulted in no action."""
        self._consecutive_silence_noop += 1
        self._current_interval = min(
            self._current_interval * self.BACKOFF_MULTIPLIER,
            self.MAX_SILENCE_INTERVAL,
        )

    def record_llm_call(self):
        """Track an LLM call for rate limiting."""
        now = time.time()
        self._llm_call_timestamps.append(now)
        self._llm_call_timestamps = [
            t for t in self._llm_call_timestamps if (now - t) < 60.0
        ]

    def should_check_silence(self, is_business: bool = False) -> bool:
        """Should we invoke LLM reasoning for the current idle cycle?"""
        now = time.time()

        if is_business:
            return False

        # Post-send cooldown
        if self._last_outgoing_time > 0:
            if (now - self._last_outgoing_time) < self.POST_SEND_COOLDOWN:
                return False

        # Interval backoff
        if self._last_silence_check_time > 0:
            if (now - self._last_silence_check_time) < self._current_interval:
                return False

        # Rate limit
        if len(self._llm_call_timestamps) >= self.MAX_LLM_CALLS_PER_MINUTE:
            return False

        self._last_silence_check_time = now
        return True

    def should_skip_strategist(self, signal_type: str, signal_priority: str) -> bool:
        """
        Should we skip the LLM Strategist entirely for this signal?

        Mechanical signals (navigation, QR, loading) always skip.
        outgoing_seen and silence go through backoff -- they CAN reach the
        Strategist after cooldown so the agent can initiate conversations.
        """
        # Mechanical: always fast-path
        if signal_type in ("not_on_whatsapp", "qr_code", "ui_loading", "dashboard"):
            return True

        # outgoing_seen: fast-path during cooldown, then allow LLM
        if signal_type == "outgoing_seen":
            return not self.should_check_silence()

        # silence: same backoff logic
        if signal_type == "silence" and signal_priority == "low":
            return not self.should_check_silence()

        return False
