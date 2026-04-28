"""
agents/strategist.py -- Strategic planning layer of the agentic system.

The StrategistAgent is the "what should we do?" brain.  It reads the current
signal from the Blackboard, consults conversation context and persona, and
makes a single LLM call to determine the high-level intent.

Responsibilities:
    - Map perception signals to strategic intentions.
    - Apply persona constraints and contact profile context.
    - Integrate reflection notes from the metacognitive layer.
    - Produce a Strategy object for the Tactician.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from agents.blackboard import (
    Blackboard, Strategy, StrategyIntent, Signal,
    SignalType, Priority,
)

logger = logging.getLogger(__name__)


class StrategistAgent:
    """
    Determines the high-level intent for the current cycle.

    Uses a combination of rule-based fast-paths (for trivial signals like QR/loading)
    and LLM-based reasoning (for message/silence signals that require judgment).
    """

    def __init__(
        self,
        blackboard: Blackboard,
        llm_client: Any,
        persona: Any,
        profiler: Any,
        db: Any,
        contact_id: int,
    ):
        self.bb = blackboard
        self.llm = llm_client
        self.persona = persona
        self.profiler = profiler
        self.db = db
        self.contact_id = contact_id

    async def strategize(self) -> Strategy:
        """
        Produce a Strategy based on the current Blackboard signal.

        Fast-path rules handle mechanical states (QR, loading, navigation).
        Complex states (new message, silence) go through LLM reasoning.

        Returns:
            A Strategy object (also written to the Blackboard).
        """
        signal = self.bb.current_signal
        if not signal:
            strategy = Strategy(intent=StrategyIntent.WAIT, reasoning="No signal available.")
            await self.bb.post_strategy(strategy)
            return strategy

        # ---- Fast-path rules (no LLM needed) ---- #
        fast = self._try_fast_path(signal)
        if fast:
            await self.bb.post_strategy(fast)
            return fast

        # ---- LLM-assisted strategic reasoning ---- #
        strategy = await self._reason_with_llm(signal)
        await self.bb.post_strategy(strategy)
        
        # If the context analyzer flagged this as needing research and we intend to respond/initiate
        if strategy.intent in (StrategyIntent.RESPOND, StrategyIntent.INITIATE):
            insight = self.bb.get_context("context_insight")
            if insight and self.bb.get_context("needs_research", False):
                from agents.blackboard import AgentMessage
                # Clear previous research
                self.bb.set_context("research_results", "")
                await self.bb.post_message(
                    AgentMessage(
                        sender="Strategist",
                        receiver="ResearchAgent",
                        content="Factual query detected. Please fetch web data to assist the Tactician.",
                        msg_type="request"
                    )
                )
                
        return strategy

    # ------------------------------------------------------------------ #
    # Fast-Path Rules                                                      #
    # ------------------------------------------------------------------ #

    def _try_fast_path(self, signal: Signal) -> Optional[Strategy]:
        """Handle purely mechanical states without LLM.

        Only navigation/loading states are fast-pathed here.
        Message-related states (new_message, silence, outgoing_seen)
        always go through LLM so the agent can decide to act.
        """
        mapping = {
            SignalType.NOT_ON_WHATSAPP: Strategy(
                intent=StrategyIntent.NAVIGATE,
                reasoning="Not on WhatsApp. Navigate to web.whatsapp.com.",
            ),
            SignalType.QR_CODE: Strategy(
                intent=StrategyIntent.WAIT,
                reasoning="QR code displayed. Waiting for user to scan.",
            ),
            SignalType.UI_LOADING: Strategy(
                intent=StrategyIntent.WAIT,
                reasoning="WhatsApp UI is still loading.",
            ),
            SignalType.DASHBOARD: Strategy(
                intent=StrategyIntent.NAVIGATE,
                reasoning=f"On dashboard. Need to open chat with target contact.",
                context_notes=signal.payload.get("contact", ""),
            ),
            # NOTE: OUTGOING_SEEN is intentionally NOT here.
            # It goes through LLM so the agent can decide to initiate.
        }
        return mapping.get(signal.signal_type)

    # ------------------------------------------------------------------ #
    # LLM-Assisted Reasoning                                               #
    # ------------------------------------------------------------------ #

    async def _reason_with_llm(self, signal: Signal) -> Strategy:
        """Use the LLM to make a nuanced strategic decision."""
        # Build context for the strategist prompt
        context_parts = self._build_strategic_context(signal)

        prompt = [
            {"role": "system", "content": self._get_strategist_system_prompt()},
            {"role": "user", "content": "\n\n".join(context_parts)},
        ]

        try:
            raw = await self.llm.chat(prompt)
            return self._parse_strategy(raw, signal)
        except Exception as e:
            logger.error(f"Strategist LLM call failed: {e}")
            # Fallback: if we have a new message, always respond
            if signal.signal_type == SignalType.NEW_MESSAGE:
                return Strategy(
                    intent=StrategyIntent.RESPOND,
                    reasoning="LLM failed -- defaulting to respond for incoming message.",
                    confidence=0.5,
                )
            return Strategy(
                intent=StrategyIntent.WAIT,
                reasoning=f"LLM failed ({e}). Defaulting to wait.",
                confidence=0.3,
            )

    def _build_strategic_context(self, signal: Signal) -> List[str]:
        """Assemble the context pieces the Strategist needs.

        Reads from the Blackboard's shared_context (populated by the Orchestrator)
        so decisions are informed by ALL layers: DB, calendar, contact classifier,
        context analysis, and goals.
        """
        parts = []

        # 1. Current situation (human-readable, not raw signal dump)
        if signal.signal_type.value == "new_message":
            text = signal.payload.get("text", "")
            parts.append(
                f"## CURRENT SITUATION\n"
                f"New incoming message from the contact.\n"
                f"Message: \"{text[:200]}\""
            )
        elif signal.signal_type.value == "outgoing_seen":
            silence_duration = self.bb.get_context("silence_duration", 0)
            if silence_duration > 600:
                parts.append(
                    f"## CURRENT SITUATION\n"
                    f"The chat is idle. YOUR last message was the most recent one.\n"
                    f"The contact has NOT replied to you yet, and it has been {silence_duration/60:.1f} minutes.\n"
                    f"You may optionally INITIATE a follow-up or check-in if it's natural."
                )
            else:
                parts.append(
                    f"## CURRENT SITUATION\n"
                    f"The chat is idle. YOUR last message was the most recent one.\n"
                    f"The contact has NOT replied to you yet (silence: {silence_duration/60:.1f} mins).\n"
                    f"IMPORTANT: You should generally WAIT for them to reply. Do NOT double-text or spam them."
                )
        elif signal.signal_type.value == "silence":
            silence_s = signal.payload.get("silence_seconds", 0)
            parts.append(
                f"## CURRENT SITUATION\n"
                f"The chat has been silent for {silence_s}s.\n"
                f"Consider starting a new conversation topic."
            )
        else:
            parts.append(
                f"## CURRENT SITUATION\n"
                f"Signal: {signal.signal_type.value} (priority: {signal.priority.value})"
            )

        # 2. Conversation summary (from DB via Blackboard)
        summary = self.bb.get_context("conversation_summary")
        if not summary:
            summary = self.db.get_summary(self.contact_id)
        if summary:
            parts.append(f"## CONVERSATION HISTORY SUMMARY\n{summary}")

        # 3. Contact profile
        profile = self.profiler.get_context_snippet()
        if profile:
            parts.append(f"## CONTACT PROFILE\n{profile}")

        # 4. Contact classification
        contact_type = self.bb.get_context("contact_type")
        if contact_type:
            parts.append(f"## CONTACT TYPE\nClassified as: {contact_type}")

        # 5. Today's calendar (for time-aware decisions)
        calendar = self.bb.get_context("calendar_today")
        if calendar:
            parts.append(f"## TODAY'S CALENDAR\n{calendar}")

        # 6. Context analysis insights (energy, intent, language)
        insight = self.bb.get_context("context_insight")
        if insight:
            parts.append(
                f"## CONTEXT ANALYSIS\n"
                f"Energy: {insight.get('energy', 'unknown')}\n"
                f"User intent: {insight.get('user_intent', 'unknown')}\n"
                f"Expected response: {insight.get('response_type', 'unknown')}\n"
                f"Language: {insight.get('language', 'english')}\n"
                f"Context quality: {insight.get('quality', 'unknown')}"
            )

        # 7. Active goal
        goal = self.bb.get_context("goal")
        if goal:
            parts.append(f"## ACTIVE GOAL\n{goal}")

        # 8. Recent reflection notes
        reflections = self.bb.get_recent_reflections(limit=3)
        if reflections:
            notes_text = "\n".join(
                f"- [{n.severity}] {n.recommendation}" for n in reflections
            )
            parts.append(f"## SELF-REFLECTION NOTES\n{notes_text}")

        # 9. System health
        if self.bb.error_streak > 0:
            parts.append(f"## SYSTEM STATUS\nConsecutive errors: {self.bb.error_streak}")

        # 10. Message count (conversation depth)
        msg_count = self.bb.get_context("message_count", 0)
        parts.append(f"## CYCLE INFO\nCycle #{self.bb.cycle_count} | Messages in DB: {msg_count}")

        return parts

    def _get_strategist_system_prompt(self) -> str:
        """The system prompt that defines the Strategist's role."""
        return (
            "You are the STRATEGIC PLANNING layer of an autonomous WhatsApp agent.\n\n"
            "Your job is to decide the HIGH-LEVEL INTENT for the current situation.\n"
            "You do NOT write the reply -- you decide WHAT to do, not HOW.\n\n"
            "IMPORTANT RULES:\n"
            "- If there is a NEW INCOMING MESSAGE, choose 'respond'.\n"
            "- If the chat is IDLE and YOUR last message was the most recent one (outgoing_seen), choose 'wait'. Do NOT double-text or spam.\n"
            "- If the chat is IDLE and it has been a long time, you may choose 'initiate' ONLY if you have a strong reason to follow up.\n"
            "- Only choose 'wait' if you genuinely have nothing meaningful to say or are waiting for a reply.\n"
            "- You are talking to a REAL PERSON. Be proactive but respectful of their time.\n\n"
            "Possible intents:\n"
            '  - "respond": Reply to an incoming message.\n'
            '  - "initiate": Proactively start a new topic or follow up.\n'
            '  - "wait": Do nothing right now.\n'
            '  - "disengage": The conversation is naturally ending.\n'
            '  - "escalate": Something unusual needs attention.\n\n'
            "Reply with ONLY raw JSON (no code fences):\n"
            "{\n"
            '  "intent": "<respond|initiate|wait|disengage|escalate>",\n'
            '  "reasoning": "<1-2 sentence explanation>",\n'
            '  "constraints": ["<optional constraint for the Tactician>"],\n'
            '  "confidence": <0.0-1.0>\n'
            "}"
        )

    def _parse_strategy(self, raw: str, signal: Signal) -> Strategy:
        """Parse the LLM's JSON output into a Strategy object."""
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)

            intent_str = data.get("intent", "wait").lower()
            intent_map = {
                "respond": StrategyIntent.RESPOND,
                "initiate": StrategyIntent.INITIATE,
                "wait": StrategyIntent.WAIT,
                "disengage": StrategyIntent.DISENGAGE,
                "escalate": StrategyIntent.ESCALATE,
            }
            intent = intent_map.get(intent_str, StrategyIntent.WAIT)

            constraints = data.get("constraints", [])
            if isinstance(constraints, list):
                constraints = tuple(constraints)
            else:
                constraints = ()

            return Strategy(
                intent=intent,
                reasoning=data.get("reasoning", ""),
                constraints=constraints,
                confidence=float(data.get("confidence", 0.7)),
            )
        except Exception as e:
            logger.warning(f"Strategist parse failed: {e}. Raw: {raw[:100]}")
            # Sensible fallback based on signal type
            if signal.signal_type == SignalType.NEW_MESSAGE:
                return Strategy(
                    intent=StrategyIntent.RESPOND,
                    reasoning="Parse failed -- defaulting to respond for new message.",
                    confidence=0.5,
                )
            if signal.signal_type == SignalType.SILENCE and signal.payload.get("prolonged"):
                return Strategy(
                    intent=StrategyIntent.INITIATE,
                    reasoning="Parse failed -- defaulting to initiate for prolonged silence.",
                    confidence=0.4,
                )
            return Strategy(intent=StrategyIntent.WAIT, reasoning="Parse failed.", confidence=0.3)
