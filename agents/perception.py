"""
agents/perception.py -- Perception layer of the agentic system.

The PerceptionAgent is a pure-heuristic observer -- it reads the browser DOM,
classifies the environment into structured signals, and writes them to the
Blackboard.  No LLM calls are made here; speed is paramount.

Responsibilities:
    - Determine the current UI state (QR, loading, dashboard, chat open).
    - Detect new incoming messages and compute content hashes.
    - Detect silence intervals for proactive engagement.
    - Classify signal priority based on content heuristics.
"""

import hashlib
import logging
import time
from typing import Any, Optional

from agents.blackboard import (
    Blackboard, Signal, SignalType, Priority,
)
from whatsapp.selectors import SEL_QR_CANVAS, SEL_SEARCH_BOX, SEL_SEARCH_BOX2
import whatsapp.dom_reader as dom

logger = logging.getLogger(__name__)


class PerceptionAgent:
    """
    Observes the browser environment and produces classified signals.

    Stateless per-cycle -- all persistent state is stored on the Blackboard
    or passed in via constructor dependencies.
    """

    # Heuristic thresholds
    URGENCY_KEYWORDS = {"urgent", "help", "asap", "emergency", "please reply", "?"}
    SILENCE_THRESHOLD_S = 600  # 10 minutes default, overridden by persona

    def __init__(
        self,
        page: Any,
        blackboard: Blackboard,
        contact_name: str,
        silence_threshold: float = 600.0,
    ):
        self.page = page
        self.bb = blackboard
        self.contact_name = contact_name
        self.silence_threshold = silence_threshold

        # Internal dedup state
        self._last_seen_hash: Optional[str] = None

    def set_last_seen_hash(self, h: str):
        """Allow external components to seed the dedup hash (e.g., from DB)."""
        self._last_seen_hash = h

    async def perceive(self) -> Signal:
        """
        Run a full perception cycle and post the resulting signal to the Blackboard.

        Returns:
            The classified Signal (also written to the Blackboard).
        """
        signal = await self._classify_environment()
        await self.bb.post_signal(signal)
        return signal

    # ------------------------------------------------------------------ #
    # Classification Pipeline                                              #
    # ------------------------------------------------------------------ #

    async def _classify_environment(self) -> Signal:
        """Determine what is happening on screen right now."""
        url = self.page.url

        # Gate 1: Not on WhatsApp at all
        if "whatsapp.com" not in url:
            return Signal(
                signal_type=SignalType.NOT_ON_WHATSAPP,
                priority=Priority.CRITICAL,
                payload={"url": url},
            )

        # Gate 2: QR code visible (user not logged in)
        if await self.page.query_selector(SEL_QR_CANVAS):
            return Signal(
                signal_type=SignalType.QR_CODE,
                priority=Priority.HIGH,
                payload={"action_needed": "scan_qr"},
            )

        # Gate 3: Chat is open with target contact
        header = await self.page.query_selector("#main header")
        if header:
            header_text = await header.inner_text()
            name_tokens = [t.lower() for t in self.contact_name.split() if len(t) > 3]
            if not name_tokens or any(t in header_text.lower() for t in name_tokens):
                return await self._classify_chat_state()

        # Gate 4: Dashboard (search box present, no chat open)
        if await dom.find_element(self.page, SEL_SEARCH_BOX, SEL_SEARCH_BOX2):
            return Signal(
                signal_type=SignalType.DASHBOARD,
                priority=Priority.NORMAL,
                payload={"contact": self.contact_name},
            )

        # Gate 5: Still loading
        return Signal(
            signal_type=SignalType.UI_LOADING,
            priority=Priority.LOW,
        )

    async def _classify_chat_state(self) -> Signal:
        """Deeper classification when the target chat is already open."""
        activity = await dom.read_burst(self.page)

        if not activity:
            return self._build_silence_signal()

        if activity["direction"] == "incoming":
            content_hash = self._compute_hash(activity["text"])
            if content_hash != self._last_seen_hash:
                # Genuine new message
                priority = self._assess_priority(activity["text"])
                return Signal(
                    signal_type=SignalType.NEW_MESSAGE,
                    priority=priority,
                    payload={
                        "text": activity["text"],
                        "content_hash": content_hash,
                        "burst_count": activity.get("count", 1),
                    },
                )
            else:
                # Already seen this message -- treat as silence
                return self._build_silence_signal()

        if activity["direction"] == "outgoing":
            return Signal(
                signal_type=SignalType.OUTGOING_SEEN,
                priority=Priority.LOW,
                payload={
                    "text": activity["text"],
                    "content_hash": self._compute_hash(activity["text"]),
                },
            )

        return Signal(signal_type=SignalType.CHAT_OPEN, priority=Priority.LOW)

    # ------------------------------------------------------------------ #
    # Heuristics                                                           #
    # ------------------------------------------------------------------ #

    def _assess_priority(self, text: str) -> Priority:
        """Heuristic priority assessment based on message content."""
        lower = text.lower()
        word_count = len(text.split())

        # Question marks or urgency keywords → high
        if any(kw in lower for kw in self.URGENCY_KEYWORDS):
            return Priority.HIGH

        # Very short messages (single word / greeting) → normal
        if word_count <= 2:
            return Priority.NORMAL

        # Long messages (effort from user) → high
        if word_count > 30:
            return Priority.HIGH

        return Priority.NORMAL

    def _build_silence_signal(self) -> Signal:
        """Construct a silence signal with timing metadata."""
        # Check how long since last signal
        last_signals = self.bb.signal_history
        last_msg_time = None
        for sig in reversed(last_signals):
            if sig.signal_type in (SignalType.NEW_MESSAGE, SignalType.OUTGOING_SEEN):
                last_msg_time = sig.timestamp
                break

        silence_duration = (time.time() - last_msg_time) if last_msg_time else 0
        is_prolonged = silence_duration > self.silence_threshold

        return Signal(
            signal_type=SignalType.SILENCE,
            priority=Priority.NORMAL if is_prolonged else Priority.LOW,
            payload={
                "silence_seconds": round(silence_duration),
                "prolonged": is_prolonged,
            },
        )

    @staticmethod
    def _compute_hash(text: str) -> str:
        return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()
