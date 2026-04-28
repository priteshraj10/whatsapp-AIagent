"""
skills/message_guard.py -- Pre-send validation gate for outgoing messages.

Runs BEFORE the Executor sends any message to catch:
    - Replies that sound like an AI summary instead of a human message
    - Messages that leak the agent's internal state or system prompts
    - Replies that are too long, too formal, or off-tone for WhatsApp
    - Messages directed at business/spam contacts

This is the last line of defense before a message hits the wire.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GuardVerdict:
    """Result of a message guard check."""
    def __init__(self, allowed: bool, reason: str = "", severity: str = "info"):
        self.allowed = allowed
        self.reason = reason
        self.severity = severity  # "info", "warning", "block"

    def __bool__(self):
        return self.allowed

    def __repr__(self):
        return f"GuardVerdict(allowed={self.allowed}, reason='{self.reason}')"


class MessageGuard:
    """
    Pre-send validation gate. Blocks or flags problematic outgoing messages.

    Each check returns a GuardVerdict. If ANY check returns blocked,
    the message is suppressed.
    """

    # Patterns that indicate the LLM is summarizing instead of replying
    SUMMARY_PATTERNS = [
        r"(?i)^(the\s+user|the\s+contact|the\s+person)\s+(has|have|had|is|was|received)",
        r"(?i)^(this\s+is|these\s+are|it\s+appears|it\s+seems)\s+(a|an|that|like)",
        r"(?i)^(the\s+conversation|the\s+messages?|the\s+chat)\s+(is|are|contains?|shows?)",
        r"(?i)(user\s+has\s+received|contact\s+has\s+been|messages\s+indicate)",
        r"(?i)(based\s+on\s+the\s+(context|conversation|messages))",
        r"(?i)(in\s+summary|to\s+summarize|overall|in\s+conclusion)",
        r"(?i)(the\s+agent\s+should|i\s+should\s+respond\s+with)",
    ]

    # Patterns that leak system internals
    LEAK_PATTERNS = [
        r"(?i)(system\s+prompt|my\s+instructions|i\s+am\s+an?\s+ai\s+agent)",
        r"(?i)(i\s+am\s+programmed|my\s+programming|i\s+was\s+trained)",
        r"(?i)(as\s+an?\s+ai\s+(assistant|agent|model|language))",
        r"(?i)(blackboard|strategist|tactician|perception\s+layer)",
        r"(?i)(tool_?executor|llm_?client|contact_?profiler)",
        r"(?i)(json\s+response|should_reply|emotion_detected)",
    ]

    # Max acceptable message length for WhatsApp
    MAX_LENGTH = 500
    MAX_SENTENCES = 8

    def __init__(self):
        pass

    def validate(self, text: str, contact_type: str = "person") -> GuardVerdict:
        """
        Run all validation checks on a message before sending.

        Args:
            text: The message text to validate.
            contact_type: The classified type of the contact.

        Returns:
            A GuardVerdict indicating whether the message should be sent.
        """
        checks = [
            self._check_empty(text),
            self._check_business_target(contact_type),
            self._check_summary_pattern(text),
            self._check_internal_leak(text),
            self._check_length(text),
            self._check_tone(text),
        ]

        for verdict in checks:
            if not verdict.allowed:
                logger.warning(f"MessageGuard BLOCKED: {verdict.reason}")
                return verdict

        return GuardVerdict(True, "All checks passed.")

    def _check_empty(self, text: str) -> GuardVerdict:
        """Block empty or whitespace-only messages."""
        if not text or not text.strip():
            return GuardVerdict(False, "Empty message.", "block")
        return GuardVerdict(True)

    def _check_business_target(self, contact_type: str) -> GuardVerdict:
        """Block replies to business/bot contacts."""
        if contact_type in ("business", "bot"):
            return GuardVerdict(
                False,
                f"Target is a {contact_type} contact. Do not engage.",
                "block",
            )
        return GuardVerdict(True)

    def _check_summary_pattern(self, text: str) -> GuardVerdict:
        """Detect when the LLM produced a summary instead of a conversational reply."""
        for pattern in self.SUMMARY_PATTERNS:
            if re.search(pattern, text):
                return GuardVerdict(
                    False,
                    f"Message looks like a summary, not a reply: '{text[:60]}...'",
                    "block",
                )
        return GuardVerdict(True)

    def _check_internal_leak(self, text: str) -> GuardVerdict:
        """Detect leakage of system internals in the message."""
        for pattern in self.LEAK_PATTERNS:
            if re.search(pattern, text):
                return GuardVerdict(
                    False,
                    f"Message leaks internal system details: '{text[:60]}...'",
                    "block",
                )
        return GuardVerdict(True)

    def _check_length(self, text: str) -> GuardVerdict:
        """Flag overly long messages (unusual for WhatsApp)."""
        if len(text) > self.MAX_LENGTH:
            return GuardVerdict(
                False,
                f"Message too long ({len(text)} chars). Max {self.MAX_LENGTH}.",
                "warning",
            )
        # Count sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        if len(sentences) > self.MAX_SENTENCES:
            return GuardVerdict(
                False,
                f"Too many sentences ({len(sentences)}). Max {self.MAX_SENTENCES}.",
                "warning",
            )
        return GuardVerdict(True)

    def _check_tone(self, text: str) -> GuardVerdict:
        """Flag messages that are too formal for WhatsApp."""
        formal_indicators = [
            r"(?i)^(dear\s+(sir|madam|user|customer))",
            r"(?i)(sincerely|regards|respectfully|yours\s+truly)",
            r"(?i)(i\s+hope\s+this\s+(message|email)\s+finds\s+you)",
            r"(?i)(please\s+find\s+attached|as\s+per\s+our\s+discussion)",
        ]
        for pattern in formal_indicators:
            if re.search(pattern, text):
                return GuardVerdict(
                    False,
                    f"Too formal for WhatsApp: '{text[:60]}...'",
                    "warning",
                )
        return GuardVerdict(True)
