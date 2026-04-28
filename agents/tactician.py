"""
agents/tactician.py -- Tactical reasoning layer of the agentic system.

The TacticianAgent is the "how should we do it?" brain.  Given a Strategy
from the Strategist, it runs the tool-augmented reasoning loop and produces
a concrete Tactic with the actual reply text, tool calls, or wait parameters.

Responsibilities:
    - Translate Strategy intents into concrete actions.
    - Run the tool loop (web search, memory recall, profile).
    - Construct the full LLM prompt with conversation context.
    - Parse the LLM's output into a structured Tactic.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from agents.blackboard import (
    Blackboard, Tactic, TacticAction, Strategy, StrategyIntent,
)
from core.tools import TOOL_DESCRIPTIONS

logger = logging.getLogger(__name__)


class TacticianAgent:
    """
    Produces concrete actions given a strategic intent.

    For mechanical intents (NAVIGATE, WAIT), this is rule-based.
    For RESPOND/INITIATE, runs the full LLM + tool loop.
    """

    MAX_TOOL_ITERATIONS = 3

    def __init__(
        self,
        blackboard: Blackboard,
        llm_client: Any,
        persona: Any,
        profiler: Any,
        tool_executor: Any,
        db: Any,
        contact_id: int,
        contact_name: str,
    ):
        self.bb = blackboard
        self.llm = llm_client
        self.persona = persona
        self.profiler = profiler
        self.tools = tool_executor
        self.db = db
        self.contact_id = contact_id
        self.contact_name = contact_name

        # Context constraints injected by the Orchestrator from ContextAnalyzer
        self._context_constraints: List[str] = []

    def set_context_constraints(self, constraints: List[str]):
        """Set constraints from the ContextAnalyzer skill."""
        self._context_constraints = constraints

    async def plan(self) -> Tactic:
        """
        Produce a Tactic based on the active Strategy on the Blackboard.

        Returns:
            A Tactic object (also written to the Blackboard).
        """
        strategy = self.bb.active_strategy
        if not strategy:
            tactic = Tactic(action_type=TacticAction.WAIT, parameters={"seconds": 3})
            await self.bb.post_tactic(tactic)
            return tactic

        # ---- Rule-based fast paths ---- #
        fast = self._try_fast_path(strategy)
        if fast:
            await self.bb.post_tactic(fast)
            return fast

        # ---- LLM + Tool reasoning ---- #
        tactic = await self._reason_with_tools(strategy)
        await self.bb.post_tactic(tactic)
        return tactic

    # ------------------------------------------------------------------ #
    # Fast-Path Rules                                                      #
    # ------------------------------------------------------------------ #

    def _try_fast_path(self, strategy: Strategy) -> Optional[Tactic]:
        """Handle mechanical strategies without LLM."""
        if strategy.intent == StrategyIntent.NAVIGATE:
            contact = strategy.context_notes or self.contact_name
            if "whatsapp" in strategy.reasoning.lower() and "not on" in strategy.reasoning.lower():
                return Tactic(
                    action_type=TacticAction.NAVIGATE,
                    parameters={"url": "https://web.whatsapp.com"},
                    reasoning="Navigate to WhatsApp Web.",
                )
            return Tactic(
                action_type=TacticAction.OPEN_CHAT,
                parameters={"contact": contact},
                reasoning=f"Open chat with {contact}.",
            )

        if strategy.intent == StrategyIntent.WAIT:
            return Tactic(
                action_type=TacticAction.WAIT,
                parameters={"seconds": 5},
                reasoning=strategy.reasoning,
            )

        if strategy.intent == StrategyIntent.BOOTSTRAP:
            return Tactic(
                action_type=TacticAction.BOOTSTRAP,
                reasoning="Run one-time history bootstrap.",
            )

        if strategy.intent == StrategyIntent.DISENGAGE:
            return Tactic(
                action_type=TacticAction.WAIT,
                parameters={"seconds": 10},
                reasoning="Conversation ending naturally. Extended wait.",
            )

        return None  # Needs LLM reasoning

    # ------------------------------------------------------------------ #
    # LLM + Tool Loop                                                      #
    # ------------------------------------------------------------------ #

    async def _reason_with_tools(self, strategy: Strategy) -> Tactic:
        """Run the full tool-augmented LLM reasoning loop."""
        signal = self.bb.current_signal

        # Build trigger context
        trigger = self._build_trigger_note(strategy)

        # Build conversation context from DB
        incoming_text = ""
        if signal and signal.payload.get("text"):
            incoming_text = signal.payload["text"]

        context = self.db.build_llm_context(self.contact_id, incoming_text)

        # Build the full prompt
        profile_snippet = self.profiler.get_context_snippet()
        system_prompt = self.persona.build_system_prompt(profile_snippet)

        # Inject available tools
        if hasattr(self, 'tools') and self.tools:
            system_prompt += f"\n\n{self.tools.get_available_tools()}"

        # Inject strategy constraints
        constraints_text = ""
        all_constraints = list(strategy.constraints) + self._context_constraints
        if all_constraints:
            constraints_text = "\n## CONSTRAINTS (follow these strictly)\n" + "\n".join(
                f"- {c}" for c in all_constraints
            )

        # Inject pre-fetched research results
        research_text = ""
        research_results = self.bb.get_context("research_results")
        if research_results:
            research_text = f"\n## PRE-FETCHED RESEARCH DATA\n{research_results}\nUse this data to answer the user's question accurately."

        # Inject reflection guidance
        reflection_text = ""
        reflections = self.bb.get_recent_reflections(limit=3)
        if reflections:
            reflection_text = "\n## SELF-IMPROVEMENT NOTES\n" + "\n".join(
                f"- {n.recommendation}" for n in reflections
            )

        # Clear context constraints after use
        self._context_constraints = []

        messages = [
            {"role": "system", "content": system_prompt + constraints_text + research_text + reflection_text},
            {"role": "system", "content": f"[TRIGGER]: {trigger}"},
        ] + context

        # Tool loop
        tool_results: List[Dict[str, str]] = []
        for attempt in range(self.MAX_TOOL_ITERATIONS + 1):
            current_messages = messages.copy()
            if tool_results:
                results_text = "\n".join(
                    f"TOOL {r['name']}: {r['output']}" for r in tool_results
                )
                current_messages.append(
                    {"role": "system", "content": f"## TOOL RESULTS:\n{results_text}"}
                )

            try:
                raw_response = await self.llm.chat(current_messages)
                parsed = self._parse_json_response(raw_response)
            except Exception as e:
                logger.error(f"Tactician LLM call failed: {e}")
                return Tactic(
                    action_type=TacticAction.WAIT,
                    parameters={"seconds": 5},
                    reasoning=f"LLM failure: {e}",
                    confidence=0.1,
                )

            # Check for tool usage
            if "use_tool" in parsed and attempt < self.MAX_TOOL_ITERATIONS:
                tool_name = parsed["use_tool"]
                tool_args = parsed.get("tool_args", {})
                logger.info(f"Tactician requesting tool: {tool_name}")
                output = await self.tools.execute(tool_name, tool_args)
                tool_results.append({"name": tool_name, "output": output})
                continue

            # Final decision reached
            return self._build_tactic_from_decision(parsed, strategy)

        # Exhausted tool iterations
        return Tactic(
            action_type=TacticAction.WAIT,
            parameters={"seconds": 5},
            reasoning="Tool loop exhausted without final decision.",
            confidence=0.2,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _build_trigger_note(self, strategy: Strategy) -> str:
        """Describe the trigger for the LLM based on the strategy intent."""
        signal = self.bb.current_signal
        if strategy.intent == StrategyIntent.RESPOND:
            text_preview = ""
            if signal and signal.payload.get("text"):
                text_preview = f" Message: \"{signal.payload['text'][:200]}\""
            return (
                f"New incoming message from the contact.{text_preview}\n"
                f"You MUST compose a natural, conversational reply. "
                f"Set should_reply to true and provide your reply text."
            )

        if strategy.intent == StrategyIntent.INITIATE:
            # Get conversation summary for context
            summary = self.bb.get_context("conversation_summary", "")
            return (
                f"The chat has been idle. You should send a message to keep the "
                f"conversation alive. Use the conversation history to pick a natural "
                f"follow-up topic or ask about something they mentioned before.\n"
                f"Conversation context: {summary[:300]}\n"
                f"You MUST compose a message. Set should_reply to true and provide your reply."
            )

        if strategy.intent == StrategyIntent.ESCALATE:
            return f"Unusual situation detected. Reason: {strategy.reasoning}"

        return f"Strategy: {strategy.intent.value}. {strategy.reasoning}"

    def _build_tactic_from_decision(self, parsed: Dict, strategy: Strategy) -> Tactic:
        """Convert the LLM's parsed JSON into a Tactic object."""
        should_reply = parsed.get("should_reply", True)
        reply_text = parsed.get("reply", "").strip()
        action_type_str = parsed.get("action_type", "reply").lower()
        target_text = parsed.get("target_text", "").strip()
        reaction_emoji = parsed.get("reaction_emoji", "").strip()

        if not should_reply or (action_type_str == "reply" and not reply_text):
            return Tactic(
                action_type=TacticAction.NONE,
                reasoning=parsed.get("reasoning", "LLM chose not to reply."),
                intent_label=parsed.get("intent", "withheld"),
                emotion_detected=parsed.get("emotion_detected", "neutral"),
                confidence=float(parsed.get("confidence", 0.5)),
            )

        action_type = TacticAction.REPLY
        if action_type_str == "react" and target_text and reaction_emoji:
            action_type = TacticAction.REACT
        elif action_type_str == "delete" and target_text:
            action_type = TacticAction.DELETE

        return Tactic(
            action_type=action_type,
            reply_text=reply_text,
            target_text=target_text,
            reaction_emoji=reaction_emoji,
            emotion_detected=parsed.get("emotion_detected", "neutral"),
            intent_label=parsed.get("intent", ""),
            confidence=float(parsed.get("confidence", 0.7)),
            reasoning=parsed.get("reasoning", ""),
        )

    @staticmethod
    def _parse_json_response(raw: str) -> Dict[str, Any]:
        """Parse the LLM's JSON with robust fallback."""
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(clean)
        except Exception:
            return {
                "reply": raw.strip(),
                "should_reply": True,
                "intent": "fallback",
                "emotion_detected": "neutral",
                "confidence": 0.5,
            }
