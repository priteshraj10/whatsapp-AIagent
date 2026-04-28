"""
agents/orchestrator.py -- The central coordinator for the multi-layered agentic system.

The AgentOrchestrator implements BaseAgent and drives the full cognitive cycle:

    Perceive -> Classify -> GoalEval -> Strategize -> Analyze -> Plan -> Guard -> Execute -> Reflect

Each step is delegated to a specialized sub-agent or skill module.
All inter-layer communication flows through the Blackboard.

This class replaces the monolithic WhatsAppAgent while reusing all existing
infrastructure (DOMReader, ConversationDB, LLMClient, ToolExecutor, Persona, Profiler).
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from core.base_agent import BaseAgent
from agents.blackboard import (
    Blackboard, Signal, SignalType, Priority,
    StrategyIntent, TacticAction, Tactic
)
from agents.executor import ExecutionAgent
from agents.goal_manager import GoalManager
from agents.perception import PerceptionAgent
from agents.reflection import ReflectionAgent
from agents.knowledge import KnowledgeExtractionAgent
from agents.strategist import StrategistAgent
from agents.tactician import TacticianAgent
from agents.research import ResearchAgent
from integrations.calendar import GoogleCalendarService
from skills.contact_classifier import ContactClassifier
from skills.context_analyzer import ContextAnalyzer
from skills.human_mimicry import HumanMimicry
from skills.message_guard import MessageGuard
from skills.silence_policy import SilencePolicy
from integrations.database import ConversationDB
from core.llm_client import LLMClient
from core.profiler import ContactProfiler
from scripts.import_history import HistoryIngester
from core.tools import ToolExecutor
from core.persona import Persona

logger = logging.getLogger(__name__)


class AgentOrchestrator(BaseAgent):
    """
    Multi-layered autonomous agent orchestrator.

    Coordinates the Perception, Cognition, and Metacognition layers
    through a shared Blackboard, with skill-based pre-action intelligence.
    """

    def __init__(
        self,
        page: Any,
        contact_name: str = "Self",
        persona: Optional[Persona] = None,
        poll_interval: float = 5.0,
    ):
        super().__init__(page)
        self.contact_name = contact_name
        self.persona = persona or Persona()
        self.poll_interval = poll_interval

        # ---- Core Services ---- #
        self.db = ConversationDB()
        self.contact_id = self.db.get_or_create_contact(contact_name)
        self.llm = LLMClient()
        self.profiler = ContactProfiler(contact_name, db=self.db, contact_id=self.contact_id)
        self.calendar = GoogleCalendarService()
        self.tools = ToolExecutor(self.db, self.profiler, self.contact_id, self.calendar)

        # ---- Shared Working Memory ---- #
        self.blackboard = Blackboard()

        # ---- Skills (pre-action intelligence) ---- #
        self.classifier = ContactClassifier(self.db, self.contact_id, contact_name)
        self.guard = MessageGuard()
        self.context_analyzer = ContextAnalyzer(self.db, self.contact_id)
        self.silence_policy = SilencePolicy()

        # ---- Sub-Agents ---- #
        self.perception = PerceptionAgent(
            page=page,
            blackboard=self.blackboard,
            contact_name=contact_name,
            silence_threshold=self.persona.silence_threshold,
        )

        self.strategist = StrategistAgent(
            blackboard=self.blackboard,
            llm_client=self.llm,
            persona=self.persona,
            profiler=self.profiler,
            db=self.db,
            contact_id=self.contact_id,
        )
        
        self.researcher = ResearchAgent(
            blackboard=self.blackboard,
            llm_client=self.llm,
            tool_executor=self.tools,
        )

        self.tactician = TacticianAgent(
            blackboard=self.blackboard,
            llm_client=self.llm,
            persona=self.persona,
            profiler=self.profiler,
            tool_executor=self.tools,
            db=self.db,
            contact_id=self.contact_id,
            contact_name=contact_name,
        )

        self.executor = ExecutionAgent(
            page=page,
            blackboard=self.blackboard,
            db=self.db,
            profiler=self.profiler,
            contact_id=self.contact_id,
            contact_name=contact_name,
        )

        self.reflector = ReflectionAgent(
            blackboard=self.blackboard,
            llm_client=self.llm,
            db=self.db,
            contact_id=self.contact_id,
        )

        self.knowledge = KnowledgeExtractionAgent(
            blackboard=self.blackboard,
            llm_client=self.llm,
            db=self.db,
            contact_id=self.contact_id,
        )

        self.goal_manager = GoalManager(
            db=self.db,
            contact_id=self.contact_id,
        )

        # ---- Bootstrap State ---- #
        has_summary = bool(self.db.get_summary(self.contact_id))
        self._bootstrapped = has_summary
        self._contact_classified = False

        # Seed dedup hashes from DB
        last_hash = self.db.get_state(self.contact_id, "last_replied_hash")
        if last_hash:
            self.perception.set_last_seen_hash(last_hash)
            self.executor.set_last_replied_hash(last_hash)

        self._log_startup()

    def _log_startup(self):
        """Log initialisation details."""
        msg_count = self.db.get_message_count(self.contact_id)
        contact_type = self.classifier.contact_type
        cal_status = "Google" if self.calendar.is_google_connected else "Local"
        logger.info(
            f"AgentOrchestrator initialised for '{self.contact_name}'. "
            f"Messages: {msg_count} | Bootstrap: {'skipped' if self._bootstrapped else 'pending'} | "
            f"Contact: {contact_type} | Calendar: {cal_status} | "
            f"Layers: Perception | Skills | Strategist | Researcher | Tactician | Executor | Reflector"
        )
        profile = self.profiler.get_context_snippet()
        if profile:
            logger.info(f"Loaded contact profile:\n{profile}")

    # ------------------------------------------------------------------ #
    # BaseAgent Interface Implementation                                   #
    # ------------------------------------------------------------------ #

    async def observe(self) -> str:
        """Layer 1: Run the Perception agent and return the signal type."""
        signal = await self.perception.perceive()
        return signal.signal_type.value

    async def understand(self) -> str:
        """Layer 1: Alias for observe."""
        return self.blackboard.current_signal.signal_type.value if self.blackboard.current_signal else "unknown"

    async def plan(self) -> str:
        """Layer 2: Run the Strategist and return a human-readable plan."""
        strategy = await self.strategist.strategize()
        return f"[{strategy.intent.value}] {strategy.reasoning}"

    async def decide(self) -> Dict[str, Any]:
        """
        Full cognitive cycle with skill-based intelligence gates.

        Flow:
            Perceive -> Classify -> SilencePolicy -> ContextAnalyze -> Strategize
            -> Research -> Plan -> MessageGuard -> Execute -> Reflect
        """
        await self.blackboard.advance_cycle()
        cycle = self.blackboard.cycle_count

        # ---- Phase 1: Perception ---- #
        signal = await self.perception.perceive()
        logger.info(f"[Cycle {cycle}] Perception: {signal.signal_type.value} [{signal.priority.value}]")

        # ---- Phase 2: Contact Classification (one-time after bootstrap) ---- #
        if not self._contact_classified and self._bootstrapped:
            recent = self.db.get_recent_messages(self.contact_id, limit=20)
            contact_type = self.classifier.classify(recent)
            self._contact_classified = True
            # Publish to shared context so ALL layers see it
            self.blackboard.set_context("contact_type", contact_type.value)
            self.blackboard.set_context("is_engageable", self.classifier.is_engageable)

            if not self.classifier.is_engageable:
                logger.info(
                    f"[Cycle {cycle}] Contact '{self.contact_name}' classified as "
                    f"{contact_type.value}. Agent will NOT engage."
                )

        # ---- Phase 3: Bootstrap check (one-time) ---- #
        if (
            signal.signal_type in (
                SignalType.CHAT_OPEN, SignalType.NEW_MESSAGE,
                SignalType.SILENCE, SignalType.OUTGOING_SEEN,
            )
            and not self._bootstrapped
        ):
            await self._run_bootstrap()
            self._bootstrapped = True
            recent = self.db.get_recent_messages(self.contact_id, limit=20)
            self.classifier.classify(recent)
            self._contact_classified = True
            self.blackboard.set_context("contact_type", self.classifier.contact_type.value)
            self.blackboard.set_context("is_engageable", self.classifier.is_engageable)

            if not self.classifier.is_engageable:
                logger.info(f"[Cycle {cycle}] Non-engageable contact detected post-bootstrap.")
                return {"status": "active", "action": {"type": "wait", "value": str(self.poll_interval)}}

            return {"status": "active", "action": {"type": "wait", "value": str(self.poll_interval)}}

        # ---- Phase 4: Silence Policy Gate ---- #
        if self.silence_policy.should_skip_strategist(
            signal.signal_type.value, signal.priority.value
        ):
            fast_strategy = self.strategist._try_fast_path(signal)
            if fast_strategy:
                await self.blackboard.post_strategy(fast_strategy)
                fast_tactic = self.tactician._try_fast_path(fast_strategy)
                if fast_tactic:
                    await self.blackboard.post_tactic(fast_tactic)
                    # Only invoke executor for actionable fast-paths (navigate, open_chat).
                    # For wait/noop, skip executor -- the engine poll loop already sleeps.
                    if fast_tactic.action_type not in (TacticAction.WAIT, TacticAction.NONE):
                        await self.executor.execute()
                    logger.info(f"[Cycle {cycle}] Fast-path: {fast_strategy.intent.value} (no LLM)")
                    return {"status": "active", "action": {"type": "wait", "value": str(self.poll_interval)}}

            logger.debug(f"[Cycle {cycle}] SilencePolicy: skipped LLM (backoff/cooldown)")
            return {"status": "active", "action": {"type": "wait", "value": str(self.poll_interval)}}

        # ---- Phase 5: Business/Bot contact gate ---- #
        if self._contact_classified and not self.classifier.is_engageable:
            if signal.signal_type == SignalType.NEW_MESSAGE:
                text = signal.payload.get("text", "")
                if text:
                    self.db.add_message(self.contact_id, "user", text)
                    content_hash = signal.payload.get("content_hash", "")
                    if content_hash:
                        self.perception.set_last_seen_hash(content_hash)
                        self.executor._update_hash(content_hash)
                logger.info(f"[Cycle {cycle}] Skipped: non-engageable contact ({self.classifier.contact_type.value})")
            return {"status": "active", "action": {"type": "wait", "value": str(self.poll_interval)}}

        # ---- Phase 6: Publish Unified Context to Blackboard ---- #
        # This is the coordination hub: every layer reads from shared_context.

        # 6a. Goal state
        last_result = self.blackboard.last_result
        self.goal_manager.evaluate(
            execution_success=last_result.success if last_result else True,
            cycle_count=cycle,
        )
        if not self.goal_manager.get_active_goal() and self.goal:
            from agents.goal_manager import Goal, GoalStatus
            self.goal_manager.push_goal(
                Goal(description=self.goal, status=GoalStatus.ACTIVE, priority=10)
            )
        self.blackboard.set_context("goal", self.goal_manager.get_goal_context())

        # 6b. Conversation summary from DB
        summary = self.db.get_summary(self.contact_id)
        if summary:
            self.blackboard.set_context("conversation_summary", summary)

        # 6c. Message count and last message time
        msg_count = self.db.get_message_count(self.contact_id)
        self.blackboard.set_context("message_count", msg_count)

        # 6d. Calendar events (today's schedule for time-aware decisions)
        try:
            today_events = await self.calendar.get_today_events()
            if today_events:
                cal_text = self.calendar.format_for_llm(today_events)
                self.blackboard.set_context("calendar_today", cal_text)
            else:
                self.blackboard.set_context("calendar_today", "No events today.")
        except Exception as e:
            logger.debug(f"Calendar context fetch failed: {e}")

        # 6e. Context Analysis (energy, intent, language -- no LLM)
        incoming_text = signal.payload.get("text", "") if signal else ""
        recent_msgs = self.db.get_recent_messages(self.contact_id, limit=12)
        context_insight = self.context_analyzer.analyze(incoming_text, recent_msgs)
        self.blackboard.set_context("context_insight", {
            "energy": context_insight.energy_level,
            "response_type": context_insight.response_type,
            "user_intent": context_insight.user_intent,
            "language": context_insight.language_detected,
            "quality": context_insight.context_quality,
        })
        self.blackboard.set_context("needs_research", context_insight.needs_research)
        
        silence_duration = self.db.get_time_since_last_message(self.contact_id)
        self.blackboard.set_context("silence_duration", silence_duration)

        # ---- Phase 7: Strategic Planning (reads from shared_context) ---- #
        self.silence_policy.record_llm_call()
        strategy = await self.strategist.strategize()
        logger.info(f"[Cycle {cycle}] Strategy: {strategy.intent.value} -- {strategy.reasoning[:60]}")

        # ---- Phase 7.5: Collaborative Intelligence Gathering ---- #
        # The ResearchAgent listens to messages from the Strategist or the ContextAnalyzer's flags
        await self.researcher.research()

        # ---- Phase 8: Tactical Planning (LLM + Tools + context constraints) ---- #
        self.tactician.set_context_constraints(context_insight.suggested_constraints)
        self.silence_policy.record_llm_call()
        tactic = await self.tactician.plan()
        logger.info(f"[Cycle {cycle}] Tactic: {tactic.action_type.value} (conf={tactic.confidence:.2f})")

        # ---- Phase 10: Message Guard (pre-send validation) ---- #
        if tactic.action_type == TacticAction.REPLY and tactic.reply_text:
            verdict = self.guard.validate(
                tactic.reply_text,
                contact_type=self.classifier.contact_type.value,
            )
            if not verdict:
                logger.warning(f"[Cycle {cycle}] MessageGuard BLOCKED: {verdict.reason}")
                # Record the silence noop
                self.silence_policy.record_silence_noop()
                return {"status": "active", "action": {"type": "wait", "value": str(self.poll_interval)}}

        # ---- Phase 11: Execution ---- #
        result = await self.executor.execute()
        logger.info(f"[Cycle {cycle}] Execution: {'OK' if result.success else 'FAIL'} -- {result.details[:60]}")

        # Track send/silence for policy
        if tactic.action_type == TacticAction.REPLY and result.success:
            self.silence_policy.record_outgoing_message()
        elif tactic.action_type in (TacticAction.NONE, TacticAction.WAIT):
            self.silence_policy.record_silence_noop()

        # Sync hash between executor and perception
        executor_hash = self.executor.get_last_replied_hash()
        if executor_hash:
            self.perception.set_last_seen_hash(executor_hash)

        # ---- Phase 12: Reflection & Knowledge (async, periodic) ---- #
        asyncio.create_task(self._safe_reflect())
        asyncio.create_task(self._safe_extract_knowledge())

        # ---- Phase 13: Periodic maintenance ---- #
        await self._periodic_maintenance()

        logger.debug(f"Blackboard snapshot: {self.blackboard.snapshot()}")

        return {"status": "active", "action": {"type": "wait", "value": str(self.poll_interval)}}

    async def verify(self) -> bool:
        """Check system health based on Blackboard state."""
        return self.blackboard.error_streak < 10

    # ------------------------------------------------------------------ #
    # Bootstrap & Maintenance                                              #
    # ------------------------------------------------------------------ #

    async def _run_bootstrap(self):
        """One-time history ingestion and context building."""
        logger.info("Orchestrator: Starting one-time history bootstrap...")
        ingester = HistoryIngester(self.page, self.db, self.contact_id)
        await ingester.ingest()

        if self.db.get_message_count(self.contact_id) >= 3:
            await self._generate_summary()
            all_msgs = self.db.get_all_messages_for_summary(self.contact_id, limit=100)
            await self.profiler.update_with_llm(self.llm, all_msgs)

        logger.info("Bootstrap complete. Cognitive layers now active.")

    async def _generate_summary(self):
        """Aggregate history into a compact summary."""
        messages = self.db.get_all_messages_for_summary(self.contact_id, limit=200)
        if len(messages) < 3:
            return

        convo_text = "\n".join(f"[{m['role']}]: {m['content']}" for m in messages)
        prompt = [
            {
                "role": "system",
                "content": (
                    "Summarise the key topics, relationship dynamics, and ongoing "
                    "plans from this WhatsApp history. Be concise (max 8 sentences)."
                ),
            },
            {"role": "user", "content": convo_text},
        ]
        try:
            summary = await self.llm.chat(prompt)
            self.db.save_summary(self.contact_id, summary.strip())
            logger.info("Conversation summary refreshed.")
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")

    async def _periodic_maintenance(self):
        """Background tasks that run periodically."""
        if self.db.summary_needs_update(self.contact_id, threshold=25):
            asyncio.create_task(self._safe_summary())

        if self.blackboard.cycle_count % 20 == 0 and self.blackboard.cycle_count > 0:
            recent = self.db.get_recent_messages(self.contact_id, limit=20)
            asyncio.create_task(self.profiler.update_with_llm(self.llm, recent))

    async def _safe_reflect(self):
        """Run reflection safely without blocking the main loop."""
        try:
            await self.reflector.reflect()
        except Exception as e:
            logger.error(f"Async reflection failed: {e}")

    async def _safe_extract_knowledge(self):
        """Run knowledge extraction safely without blocking the main loop."""
        try:
            await self.knowledge.extract()
        except Exception as e:
            logger.error(f"Async knowledge extraction failed: {e}")

    async def _safe_summary(self):
        """Run summary generation in a safe wrapper."""
        try:
            await self._generate_summary()
        except Exception as e:
            logger.debug(f"Summary background task failed: {e}")
