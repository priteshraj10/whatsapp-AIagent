"""
agents/reflection.py -- Metacognitive reflection layer of the agentic system.

The ReflectionAgent periodically reviews the system's recent performance and
produces adjustment notes that influence future strategic decisions.

This layer does NOT run every cycle -- it triggers based on thresholds:
    - Every N successful exchanges.
    - After error streaks.
    - After significant conversation shifts.

Responsibilities:
    - Evaluate recent conversation quality.
    - Detect repetition, tone mismatches, or engagement drops.
    - Produce ReflectionNotes with corrective recommendations.
    - Run asynchronously to avoid blocking the main cycle.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from agents.blackboard import Blackboard, ReflectionNote

logger = logging.getLogger(__name__)


class ReflectionAgent:
    """
    Metacognitive layer that evaluates and improves the system's behavior.

    Runs asynchronously and only when specific thresholds are met.
    Writes ReflectionNotes to the Blackboard that the Strategist consumes.
    """

    # Trigger thresholds
    EXCHANGE_INTERVAL = 8          # Reflect every N cycles with activity
    ERROR_STREAK_TRIGGER = 3       # Reflect after N consecutive errors
    MIN_CYCLES_BETWEEN = 15        # Don't reflect more often than this

    def __init__(
        self,
        blackboard: Blackboard,
        llm_client: Any,
        db: Any,
        contact_id: int,
    ):
        self.bb = blackboard
        self.llm = llm_client
        self.db = db
        self.contact_id = contact_id
        self._last_reflection_cycle: int = 0
        self._exchange_counter: int = 0

    def should_reflect(self) -> bool:
        """Determine if reflection should trigger this cycle."""
        cycle = self.bb.cycle_count

        # Minimum cooldown between reflections
        if (cycle - self._last_reflection_cycle) < self.MIN_CYCLES_BETWEEN:
            return False

        # Trigger on error streaks
        if self.bb.error_streak >= self.ERROR_STREAK_TRIGGER:
            return True

        # Trigger on exchange intervals
        recent_results = self.bb.get_recent_results(limit=self.EXCHANGE_INTERVAL)
        active_results = [r for r in recent_results if r.details and "Replied" in r.details]
        if len(active_results) >= self.EXCHANGE_INTERVAL:
            return True

        return False

    async def reflect(self) -> Optional[ReflectionNote]:
        """
        Run a reflection cycle if thresholds are met.

        Makes one LLM call to evaluate recent activity and produce recommendations.

        Returns:
            A ReflectionNote if reflection ran, None if skipped.
        """
        if not self.should_reflect():
            return None

        logger.info("ReflectionAgent: Initiating self-evaluation...")
        self._last_reflection_cycle = self.bb.cycle_count

        try:
            note = await self._evaluate_performance()
            if note:
                await self.bb.post_reflection(note)
            return note
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            return None

    async def _evaluate_performance(self) -> Optional[ReflectionNote]:
        """Run the LLM-based self-evaluation."""
        # Gather recent execution history
        context = self._build_reflection_context()

        prompt = [
            {"role": "system", "content": self._get_reflection_prompt()},
            {"role": "user", "content": context},
        ]

        try:
            raw = await self.llm.chat(prompt)
            return self._parse_reflection(raw)
        except Exception as e:
            logger.error(f"Reflection LLM call failed: {e}")
            return None

    def _build_reflection_context(self) -> str:
        """Assemble the performance data for the reflector to analyze."""
        parts = []

        # 1. Recent execution results
        results = self.bb.get_recent_results(limit=15)
        if results:
            results_text = "\n".join(
                f"  [{i+1}] {'OK' if r.success else 'FAIL'} | {r.action_type.value} | {r.details[:60]}"
                for i, r in enumerate(results)
            )
            parts.append(f"## RECENT ACTIONS\n{results_text}")

        # 2. Recent messages from DB
        recent_msgs = self.db.get_recent_messages(self.contact_id, limit=15)
        if recent_msgs:
            msgs_text = "\n".join(
                f"  [{m['role'].upper()}]: {m['content'][:80]}"
                for m in recent_msgs
            )
            parts.append(f"## RECENT CONVERSATION\n{msgs_text}")

        # 3. Error streak info
        parts.append(f"## SYSTEM STATUS\nError streak: {self.bb.error_streak}")
        parts.append(f"Total cycles: {self.bb.cycle_count}")

        # 4. Existing reflection notes
        existing = self.bb.get_recent_reflections(limit=3)
        if existing:
            prev_text = "\n".join(f"  - {n.recommendation}" for n in existing)
            parts.append(f"## PREVIOUS REFLECTIONS\n{prev_text}")

        return "\n\n".join(parts)

    @staticmethod
    def _get_reflection_prompt() -> str:
        """The system prompt for the reflective evaluation."""
        return (
            "You are the METACOGNITIVE REFLECTION layer of an autonomous WhatsApp agent.\n\n"
            "Your job is to evaluate the agent's recent performance and provide "
            "actionable recommendations to improve future interactions.\n\n"
            "Analyze:\n"
            "1. Is the conversation quality high? Are replies natural and engaging?\n"
            "2. Is the agent repeating itself or using the same patterns?\n"
            "3. Is the tone appropriate for the contact's energy level?\n"
            "4. Are there too many errors or failures?\n"
            "5. Should the agent be more/less proactive?\n\n"
            "Reply with ONLY raw JSON (no code fences):\n"
            "{\n"
            '  "observation": "<what you noticed about the agent\'s performance>",\n'
            '  "recommendation": "<specific, actionable advice for improvement>",\n'
            '  "severity": "<info|warning|critical>"\n'
            "}"
        )

    @staticmethod
    def _parse_reflection(raw: str) -> Optional[ReflectionNote]:
        """Parse the LLM's reflection output."""
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)
            return ReflectionNote(
                observation=data.get("observation", ""),
                recommendation=data.get("recommendation", ""),
                severity=data.get("severity", "info"),
            )
        except Exception as e:
            logger.warning(f"Reflection parse failed: {e}")
            # Best-effort: treat the raw response as a recommendation
            if raw.strip():
                return ReflectionNote(
                    observation="Raw reflection (unparsed)",
                    recommendation=raw.strip()[:200],
                    severity="info",
                )
            return None
