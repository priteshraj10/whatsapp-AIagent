"""
skills/contact_classifier.py -- Classifies contacts BEFORE the agent engages.

Prevents the agent from:
    - Replying to business/spam chats (KreditBee, OTP bots, banks)
    - Treating automated messages as human conversation
    - Wasting LLM calls on non-conversational contacts

Classification is cached per contact to avoid repeated analysis.
"""

import json
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ContactType(str, Enum):
    """Classification of a WhatsApp contact."""
    PERSON       = "person"         # Real human contact
    BUSINESS     = "business"       # Business/brand account
    BOT          = "bot"            # Automated bot (OTP, alerts)
    GROUP        = "group"          # Group chat
    UNKNOWN      = "unknown"        # Not enough data to classify


class ContactClassifier:
    """
    Classifies contacts using heuristics and optional LLM analysis.

    Runs once per contact and caches the result in the DB.
    """

    STATE_KEY = "contact_type"

    # Known business/bot patterns (case-insensitive)
    BUSINESS_PATTERNS = [
        r"(?i)\b(bank|finance|loan|credit|insurance|mutual\s*fund)\b",
        r"(?i)\b(hdfc|icici|sbi|axis|kotak|bajaj|paytm|phonepe|gpay)\b",
        r"(?i)\b(amazon|flipkart|swiggy|zomato|uber|ola|myntra)\b",
        r"(?i)\b(otp|verify|code|passcode|authentication)\b",
        r"(?i)\b(offer|discount|cashback|reward|deal|coupon)\b",
        r"(?i)\b(kreditbee|kbnbfc|cred|slice|lazypay|simpl)\b",
        r"(?i)\b(noreply|no-reply|donotreply|do-not-reply)\b",
    ]

    # Patterns that suggest automated messages
    AUTOMATED_PATTERNS = [
        r"(?i)(your\s+otp|verification\s+code|one[\s-]time\s+password)",
        r"(?i)(dear\s+(customer|user|member|sir|madam))",
        r"(?i)(t\s*&\s*c\s+apply|terms\s+and\s+conditions)",
        r"(?i)(click\s+here|tap\s+to|visit\s+https?://)",
        r"(?i)(unsubscribe|opt[\s-]?out|stop\s+to\s+cancel)",
        r"(?i)(loan\s+of\s+rs|credit\s+limit|emi\s+of|pre[\s-]?approved)",
        r"(?i)(congratulations!?\s+you|you\s+have\s+been\s+selected)",
    ]

    def __init__(self, db: Any, contact_id: int, contact_name: str):
        self.db = db
        self.contact_id = contact_id
        self.contact_name = contact_name
        self._cached_type: Optional[ContactType] = None

        # Try loading from DB
        saved = self.db.get_state(contact_id, self.STATE_KEY)
        if saved:
            try:
                self._cached_type = ContactType(saved)
            except ValueError:
                pass

    @property
    def contact_type(self) -> ContactType:
        """Get the cached contact type."""
        return self._cached_type or ContactType.UNKNOWN

    @property
    def is_human(self) -> bool:
        """Quick check: is this a real person?"""
        return self._cached_type == ContactType.PERSON

    @property
    def is_engageable(self) -> bool:
        """Should the agent engage with this contact at all?"""
        if self._cached_type is None:
            return True  # Unknown = give it a chance
        return self._cached_type in (ContactType.PERSON, ContactType.UNKNOWN)

    def classify(self, recent_messages: List[Dict[str, str]]) -> ContactType:
        """
        Classify the contact based on name + message content analysis.

        Uses a scoring system:
            - Business name patterns: +3 per match
            - Automated message patterns: +2 per match
            - Conversational indicators: -2 per match

        Threshold: score >= 3 = BUSINESS/BOT, score <= -3 = PERSON
        """
        if self._cached_type and self._cached_type != ContactType.UNKNOWN:
            return self._cached_type

        score = 0

        # 1. Analyze contact name
        name = self.contact_name.lower()
        for pattern in self.BUSINESS_PATTERNS:
            if re.search(pattern, name):
                score += 3
                break

        # 2. Analyze message content
        all_text = " ".join(m.get("content", "") for m in recent_messages[:20])

        for pattern in self.AUTOMATED_PATTERNS:
            if re.search(pattern, all_text):
                score += 2

        for pattern in self.BUSINESS_PATTERNS:
            if re.search(pattern, all_text):
                score += 1

        # 3. Conversational indicators (suggest human)
        conversational_patterns = [
            r"(?i)\b(haha|lol|lmao|rofl|xd)\b",
            r"(?i)\b(bro|dude|yaar|bhai|man|buddy)\b",
            r"(?i)\b(how\s+are\s+you|what'?s\s+up|sup|hey|hi\s+there)\b",
            r"(?i)\b(miss\s+you|love\s+you|good\s+night|good\s+morning)\b",
            r"(?i)(okay|ok\b|sure|yeah|yep|nope|nah|hmm)",
        ]
        for pattern in conversational_patterns:
            if re.search(pattern, all_text):
                score -= 2

        # 4. Message direction analysis
        user_msgs = [m for m in recent_messages if m.get("role") == "user"]
        if user_msgs:
            avg_len = sum(len(m.get("content", "")) for m in user_msgs) / len(user_msgs)
            # Automated messages tend to be long
            if avg_len > 200:
                score += 2
            # Short casual messages suggest human
            if avg_len < 50:
                score -= 1

        # 5. Classify
        if score >= 3:
            result = ContactType.BUSINESS
        elif score <= -3:
            result = ContactType.PERSON
        else:
            result = ContactType.UNKNOWN  # Not enough signal

        self._cached_type = result
        self.db.set_state(self.contact_id, self.STATE_KEY, result.value)

        # Auto-assign a relationship category for clearly non-human contacts
        # so the profiler doesn't waste LLM calls trying to befriend a spam bot.
        if result in (ContactType.BUSINESS, ContactType.BOT):
            self.db.set_state(self.contact_id, "relationship_category", "Business/Spam")
            logger.debug(f"Auto-assigned 'Business/Spam' category to '{self.contact_name}'")

        logger.info(
            f"Contact '{self.contact_name}' classified as {result.value} (score={score})"
        )
        return result

    async def classify_with_llm(self, llm_client: Any, recent_messages: List[Dict[str, str]]) -> ContactType:
        """
        Use LLM for ambiguous cases where heuristics are inconclusive.

        Only called when heuristic classification returns UNKNOWN.
        """
        heuristic_result = self.classify(recent_messages)
        if heuristic_result != ContactType.UNKNOWN:
            return heuristic_result

        # Build a compact context for LLM
        msg_sample = "\n".join(
            f"[{m['role'].upper()}]: {m['content'][:100]}"
            for m in recent_messages[:10]
        )

        prompt = [
            {
                "role": "system",
                "content": (
                    "Classify this WhatsApp contact. Is this a real person having a "
                    "conversation, or a business/bot sending automated messages?\n\n"
                    "Reply with ONLY one word: person, business, or bot"
                ),
            },
            {
                "role": "user",
                "content": f"Contact name: {self.contact_name}\n\nMessages:\n{msg_sample}",
            },
        ]

        try:
            raw = await llm_client.chat(prompt)
            result_str = raw.strip().lower().split()[0] if raw.strip() else "unknown"

            type_map = {
                "person": ContactType.PERSON,
                "business": ContactType.BUSINESS,
                "bot": ContactType.BOT,
            }
            result = type_map.get(result_str, ContactType.UNKNOWN)

            self._cached_type = result
            self.db.set_state(self.contact_id, self.STATE_KEY, result.value)

            # Sync category for non-human contacts confirmed by LLM
            if result in (ContactType.BUSINESS, ContactType.BOT):
                self.db.set_state(self.contact_id, "relationship_category", "Business/Spam")

            logger.info(f"LLM classified '{self.contact_name}' as {result.value}")
            return result

        except Exception as e:
            logger.warning(f"LLM contact classification failed: {e}")
            return heuristic_result
