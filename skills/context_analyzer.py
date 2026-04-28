"""
skills/context_analyzer.py -- Deep context analysis before the agent acts.

Provides pre-action intelligence:
    - Is there enough context to give a quality reply?
    - What is the conversation's current energy/vibe?
    - Is the user expecting a response or just venting?
    - What kind of response is appropriate (short, detailed, question)?

Runs between the Strategist and Tactician to provide richer context
for the LLM to work with.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContextInsight:
    """Structured analysis of the conversation context."""
    energy_level: str           # "high", "medium", "low", "dead"
    response_type: str          # "short", "detailed", "question", "acknowledge", "none"
    user_intent: str            # "asking", "sharing", "venting", "greeting", "closing", "spam"
    tone: str                   # "neutral", "urgent", "angry", "sad", "joyful", "sarcastic"
    time_of_day: str            # "morning", "afternoon", "evening", "late_night"
    context_quality: str        # "rich" (enough history), "thin" (sparse), "cold" (first message)
    language_detected: str      # "english", "hindi", "hinglish", "mixed"
    needs_research: bool        # True if the user asks a factual question
    suggested_constraints: List[str]  # Constraints to pass to the Tactician


class ContextAnalyzer:
    """
    Analyzes conversation context and produces actionable insights
    that guide the Tactician's LLM prompt.
    """

    def __init__(self, db: Any, contact_id: int):
        self.db = db
        self.contact_id = contact_id

    def analyze(self, incoming_text: str, recent_messages: List[Dict[str, str]]) -> ContextInsight:
        """
        Run a full context analysis on the current state.

        Returns:
            A ContextInsight with structured guidance for the Tactician.
        """
        energy = self._detect_energy(incoming_text, recent_messages)
        response_type = self._determine_response_type(incoming_text)
        user_intent = self._classify_intent(incoming_text)
        tone = self._detect_tone(incoming_text)
        time_of_day = self._get_time_of_day()
        context_quality = self._assess_context_quality(recent_messages)
        language = self._detect_language(incoming_text, recent_messages)
        needs_research = self._needs_research(incoming_text)
        
        # Pull relationship category from DB
        relationship = self.db.get_state(self.contact_id, "relationship_category") or "Unknown"
        
        constraints = self._build_constraints(
            energy, response_type, user_intent, tone, time_of_day, context_quality, language, relationship
        )

        insight = ContextInsight(
            energy_level=energy,
            response_type=response_type,
            user_intent=user_intent,
            tone=tone,
            time_of_day=time_of_day,
            context_quality=context_quality,
            language_detected=language,
            needs_research=needs_research,
            suggested_constraints=constraints,
        )

        logger.info(
            f"ContextAnalysis: energy={energy}, type={response_type}, "
            f"intent={user_intent}, quality={context_quality}, lang={language}"
        )
        return insight

    # ------------------------------------------------------------------ #
    # Analysis Components                                                  #
    # ------------------------------------------------------------------ #

    def _detect_energy(self, text: str, recent: List[Dict]) -> str:
        """Gauge the conversation's energy level."""
        if not text:
            return "dead"

        # High energy indicators
        high_patterns = [
            r"[!]{2,}",            # Multiple exclamation marks
            r"[A-Z]{3,}",         # CAPS
            r"(?i)(omg|wow|damn|bruh|crazy|insane|amazing)",
            r"(?i)(haha|lol|lmao|rofl)",
            r"\b[?!]{2,}",
        ]
        high_score = sum(1 for p in high_patterns if re.search(p, text))
        if high_score >= 2:
            return "high"

        # Low energy indicators
        low_patterns = [
            r"^(ok|k|hmm|ah|oh|fine|sure|ya|yep|nah)\.?$",
            r"^.{1,10}$",  # Very short messages
        ]
        if any(re.search(p, text.strip(), re.IGNORECASE) for p in low_patterns):
            return "low"

        # Check recent message frequency
        if len(recent) >= 6:
            recent_user = [m for m in recent[-6:] if m.get("role") == "user"]
            if len(recent_user) >= 4:
                return "high"  # User is actively chatting

        return "medium"

    def _determine_response_type(self, text: str) -> str:
        """Determine what kind of response the user expects."""
        if not text:
            return "none"

        # Questions need answers
        if text.strip().endswith("?") or re.search(r"(?i)^(what|how|why|when|where|who|can|do|is|are|will|should)\b", text):
            # Long questions need detailed answers
            if len(text.split()) > 10:
                return "detailed"
            return "short"

        # Sharing content (links, media descriptions)
        if re.search(r"https?://|www\.", text):
            return "acknowledge"

        # Greetings
        if re.search(r"(?i)^(hi|hey|hello|yo|sup|good\s+(morning|evening|night))", text.strip()):
            return "short"

        # Closings
        if re.search(r"(?i)(bye|good\s*night|ttyl|see\s+you|gtg|gotta\s+go)", text):
            return "acknowledge"

        # Venting (long messages without questions)
        if len(text.split()) > 30 and not text.strip().endswith("?"):
            return "acknowledge"

        return "short"

    def _classify_intent(self, text: str) -> str:
        """Classify what the user is trying to do."""
        if not text:
            return "spam"

        text_lower = text.lower().strip()

        # Greeting
        if re.search(r"(?i)^(hi|hey|hello|yo|sup|what'?s\s*up)", text_lower):
            return "greeting"

        # Closing
        if re.search(r"(?i)(bye|good\s*night|ttyl|see\s+you|take\s+care)", text_lower):
            return "closing"

        # Asking
        if "?" in text or re.search(r"(?i)^(what|how|why|when|where|who|can|do|is|tell)\b", text_lower):
            return "asking"

        # Sharing
        if re.search(r"(?i)(check\s+this|look\s+at|have\s+you\s+seen|did\s+you\s+know)", text_lower):
            return "sharing"

        # Venting
        if len(text.split()) > 20:
            return "venting"

        # Spam indicators
        spam_patterns = [
            r"(?i)(loan|emi|credit|cashback|offer|discount|otp|verification)",
            r"(?i)(dear\s+customer|click\s+here|apply\s+now|t&c\s+apply)",
        ]
        if any(re.search(p, text) for p in spam_patterns):
            return "spam"

        return "sharing"

    def _assess_context_quality(self, recent: List[Dict]) -> str:
        """How much context do we have to work with?"""
        msg_count = len(recent)
        if msg_count == 0:
            return "cold"
        if msg_count < 4:
            return "thin"
        return "rich"

    def _detect_language(self, text: str, recent: List[Dict]) -> str:
        """Detect the dominant language of the conversation."""
        all_text = text + " " + " ".join(m.get("content", "") for m in recent[-5:])

        # Hindi/Devanagari detection
        if re.search(r'[\u0900-\u097F]', all_text):
            return "hindi"

        # Hinglish patterns (Hindi words in English script)
        hinglish_words = [
            r"\b(kya|kaise|accha|theek|nahi|haan|aur|bhai|yaar|matlab|abhi|dekh)\b",
            r"\b(kuch|mujhe|tujhe|apna|kaisa|kaha|bahut|bohot|sahi|galat)\b",
        ]
        hinglish_score = sum(1 for p in hinglish_words if re.search(p, all_text, re.IGNORECASE))
        if hinglish_score >= 2:
            return "hinglish"

        return "english"

    def _needs_research(self, text: str) -> bool:
        """Detect if the text contains a factual query requiring a web search."""
        if not text:
            return False
            
        text_lower = text.lower()
        factual_patterns = [
            r"(?i)^(who\s+is|what\s+is|where\s+is|when\s+is|how\s+to)\b",
            r"(?i)\b(weather\s+in|score|match|news|price\s+of|stock|latest)\b",
            r"(?i)\b(capital\s+of|population\s+of|president\s+of|ceo\s+of)\b"
        ]
        
        return any(re.search(p, text_lower) for p in factual_patterns)

    def _detect_tone(self, text: str) -> str:
        """Detect the emotional tone of the message."""
        if not text:
            return "neutral"
            
        text_lower = text.lower()
        
        # Urgent
        if re.search(r"(?i)\b(asap|urgent|hurry|quick|emergency|help)\b", text_lower):
            return "urgent"
            
        # Angry
        if re.search(r"(?i)\b(wtf|hate|stupid|idiot|annoying|mad|angry|bullshit)\b", text_lower) or text.isupper():
            return "angry"
            
        # Sad
        if re.search(r"(?i)\b(sad|depressed|crying|upset|sorry|bad news|died)\b", text_lower):
            return "sad"
            
        # Joyful
        if re.search(r"(?i)\b(happy|yay|awesome|amazing|love|best|woo)\b", text_lower):
            return "joyful"
            
        return "neutral"
        
    def _get_time_of_day(self) -> str:
        """Determine the current time of day scenario, including day of week."""
        now = time.localtime()
        hour = now.tm_hour
        day_idx = now.tm_wday  # 0=Monday, 6=Sunday
        
        is_weekend = day_idx >= 5
        day_str = "weekend" if is_weekend else "weekday"
        
        if 5 <= hour < 12:
            time_str = "morning"
        elif 12 <= hour < 17:
            time_str = "afternoon"
        elif 17 <= hour < 22:
            time_str = "evening"
        else:
            time_str = "late_night"
            
        return f"{time_str}_{day_str}"

    def _build_constraints(
        self, energy: str, response_type: str, intent: str,
        tone: str, time_of_day: str, quality: str, language: str, relationship: str
    ) -> List[str]:
        """Build tactical constraints based on the analysis."""
        constraints = []

        # Energy matching
        if energy == "high":
            constraints.append("Match the user's high energy. Be enthusiastic.")
        elif energy == "low":
            constraints.append("Keep it brief. User is low-energy right now.")
        elif energy == "dead":
            constraints.append("Conversation is dead. Only engage if proactive is appropriate.")

        # Response type
        if response_type == "detailed":
            constraints.append("Give a thorough answer. The user asked a detailed question.")
        elif response_type == "acknowledge":
            constraints.append("Acknowledge briefly. No need for a long response.")
        elif response_type == "short":
            constraints.append("Keep reply to 1-2 sentences max.")

        # Intent
        if intent == "closing":
            constraints.append("The user is ending the conversation. Say goodbye naturally.")
        elif intent == "greeting":
            constraints.append("Greet back warmly and ask a follow-up.")
        elif intent == "venting":
            constraints.append("Be empathetic. Acknowledge their feelings first.")
        elif intent == "spam":
            constraints.append("This looks like spam/automated content. Do NOT reply.")

        # Context quality
        if quality == "cold":
            constraints.append("First interaction. Be warm but not presumptuous.")
        elif quality == "thin":
            constraints.append("Limited history. Don't reference past topics you don't have.")

        # Tone
        if tone == "urgent":
            constraints.append("USER IS IN A RUSH. Be extremely brief, helpful, and skip pleasantries.")
        elif tone == "angry":
            constraints.append("USER SEEMS FRUSTRATED. De-escalate, be polite, and do not use emojis.")
        elif tone == "sad":
            constraints.append("USER SEEMS SAD. Be empathetic, comforting, and supportive.")
        elif tone == "joyful":
            constraints.append("USER IS HAPPY. Match their joyful energy!")

        # Relationship
        if relationship == "Close Friend":
            constraints.append("Relationship: Close Friend. Be extremely casual, use slang, and act like best buddies.")
        elif relationship == "Family":
            constraints.append("Relationship: Family. Be respectful but warm and loving.")
        elif relationship == "Colleague":
            constraints.append("Relationship: Colleague. Be professional, polite, and helpful.")
        elif relationship == "Acquaintance":
            constraints.append("Relationship: Acquaintance. Be friendly but maintain appropriate boundaries.")

        # Time of Day
        current_time = time.strftime("%I:%M %p")
        if "late_night" in time_of_day:
            constraints.append(f"It is currently late at night ({current_time}). Keep it quiet, brief, or suggest talking tomorrow.")
        elif "morning" in time_of_day and quality == "cold":
            constraints.append(f"It is morning ({current_time}). Start with a good morning greeting.")
        else:
            # Just provide the time for context
            constraints.append(f"Current local time is {current_time}.")
            
        if "weekend" in time_of_day:
            constraints.append("It is the weekend. Keep the vibe relaxed.")

        # Language
        if language == "hinglish":
            constraints.append("CRITICAL: You MUST reply in conversational Hinglish (Hindi written in English alphabet, mixed with English words). DO NOT use pure English or Devanagari script.")
        elif language == "hindi":
            constraints.append("Reply in Hindi (Devanagari script) to match the user's language.")

        return constraints
