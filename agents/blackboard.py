"""
agents/blackboard.py -- Shared working memory for the multi-layered agentic system.

The Blackboard is the central nervous system of the architecture.  Every layer
reads from and writes to this structure -- no direct inter-layer coupling.

Design:
    - Thread-safe via asyncio.Lock (single event-loop model).
    - Immutable dataclasses for signals/strategies/tactics to prevent mutation bugs.
    - Bounded history buffers to cap memory usage.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Enumerations                                                         #
# ------------------------------------------------------------------ #

class SignalType(str, Enum):
    """Classification of environmental signals from the Perception layer."""
    NEW_MESSAGE      = "new_message"
    SILENCE          = "silence"
    QR_CODE          = "qr_code"
    UI_LOADING       = "ui_loading"
    DASHBOARD        = "dashboard"
    NOT_ON_WHATSAPP  = "not_on_whatsapp"
    CHAT_OPEN        = "chat_open"
    ERROR            = "error"
    OUTGOING_SEEN    = "outgoing_seen"


class Priority(str, Enum):
    """Signal urgency levels."""
    CRITICAL = "critical"
    HIGH     = "high"
    NORMAL   = "normal"
    LOW      = "low"


class StrategyIntent(str, Enum):
    """High-level intentions produced by the Strategist."""
    RESPOND    = "respond"
    INITIATE   = "initiate"
    WAIT       = "wait"
    DISENGAGE  = "disengage"
    NAVIGATE   = "navigate"
    BOOTSTRAP  = "bootstrap"
    ESCALATE   = "escalate"


class TacticAction(str, Enum):
    REPLY       = "reply"
    REACT       = "react"
    DELETE      = "delete"
    USE_TOOL    = "use_tool"
    WAIT        = "wait"
    NAVIGATE    = "navigate"
    OPEN_CHAT   = "open_chat"
    BOOTSTRAP   = "bootstrap"
    NONE        = "none"


# ------------------------------------------------------------------ #
# Data Structures                                                      #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class Signal:
    """
    A classified observation from the Perception layer.

    Immutable to prevent accidental mutation after being placed on the board.
    """
    signal_type: SignalType
    priority: Priority
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Strategy:
    """
    A high-level plan produced by the Strategist.

    Tells the Tactician *what* to do, not *how*.
    """
    intent: StrategyIntent
    reasoning: str = ""
    constraints: tuple = ()          # Frozen-compatible (tuple vs list)
    context_notes: str = ""          # Extra context for the Tactician
    confidence: float = 1.0


@dataclass(frozen=True)
class Tactic:
    """
    A concrete action plan produced by the Tactician.

    Tells the Executor exactly what to do.
    """
    action_type: TacticAction
    parameters: Dict[str, Any] = field(default_factory=dict)
    reply_text: str = ""
    target_text: str = ""
    reaction_emoji: str = ""
    emotion_detected: str = "neutral"
    intent_label: str = ""
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class ExecutionResult:
    """Outcome of an action dispatched by the Executor."""
    success: bool
    action_type: TacticAction
    details: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReflectionNote:
    """An adjustment note from the Reflection layer."""
    observation: str
    recommendation: str
    severity: str = "info"           # "info", "warning", "critical"
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentMessage:
    """An inter-agent communication message."""
    sender: str
    receiver: str
    content: str
    msg_type: str = "request"
    timestamp: float = field(default_factory=time.time)


# ------------------------------------------------------------------ #
# Blackboard                                                           #
# ------------------------------------------------------------------ #

class Blackboard:
    """
    Thread-safe shared working memory for inter-layer communication.

    All cognitive layers read from and write to this structure.
    History buffers are bounded to prevent unbounded memory growth.
    """

    MAX_SIGNAL_HISTORY = 50
    MAX_EXECUTION_HISTORY = 30
    MAX_REFLECTION_NOTES = 20

    def __init__(self):
        self._lock = asyncio.Lock()

        # Current state
        self.current_signal: Optional[Signal] = None
        self.active_strategy: Optional[Strategy] = None
        self.active_tactic: Optional[Tactic] = None
        self.last_result: Optional[ExecutionResult] = None

        # Shared context -- cross-cutting knowledge accessible by ALL layers.
        # Keys are strings like "contact_type", "calendar_today", "context_insight".
        # Any layer can write here; all layers can read.
        self.shared_context: Dict[str, Any] = {}

        # History buffers (bounded)
        self.signal_history: List[Signal] = []
        self.execution_history: List[ExecutionResult] = []
        self.reflection_notes: List[ReflectionNote] = []
        self.messages: List[AgentMessage] = []

        # Cycle metadata
        self.cycle_count: int = 0
        self.error_streak: int = 0
        self.last_cycle_time: float = 0.0

        logger.debug("Blackboard initialised.")

    async def post_signal(self, signal: Signal):
        """Write a new perception signal to the board."""
        async with self._lock:
            self.current_signal = signal
            self.signal_history.append(signal)
            if len(self.signal_history) > self.MAX_SIGNAL_HISTORY:
                self.signal_history = self.signal_history[-self.MAX_SIGNAL_HISTORY:]
        logger.debug(f"Signal posted: {signal.signal_type.value} [{signal.priority.value}]")

    async def post_strategy(self, strategy: Strategy):
        """Write a new strategy to the board."""
        async with self._lock:
            self.active_strategy = strategy
        logger.debug(f"Strategy posted: {strategy.intent.value} (conf={strategy.confidence:.2f})")

    async def post_tactic(self, tactic: Tactic):
        """Write a new tactic to the board."""
        async with self._lock:
            self.active_tactic = tactic
        logger.debug(f"Tactic posted: {tactic.action_type.value} (conf={tactic.confidence:.2f})")

    async def post_result(self, result: ExecutionResult):
        """Write an execution result to the board."""
        async with self._lock:
            self.last_result = result
            self.execution_history.append(result)
            if len(self.execution_history) > self.MAX_EXECUTION_HISTORY:
                self.execution_history = self.execution_history[-self.MAX_EXECUTION_HISTORY:]
            # Track error streaks
            if result.success:
                self.error_streak = 0
            else:
                self.error_streak += 1
        logger.debug(f"Result posted: {'OK' if result.success else 'FAIL'} -- {result.details[:60]}")

    async def post_reflection(self, note: ReflectionNote):
        """Write a reflection note to the board."""
        async with self._lock:
            self.reflection_notes.append(note)
            if len(self.reflection_notes) > self.MAX_REFLECTION_NOTES:
                self.reflection_notes = self.reflection_notes[-self.MAX_REFLECTION_NOTES:]
        logger.info(f"Reflection [{note.severity}]: {note.observation[:80]}")

    async def post_message(self, message: AgentMessage):
        """Send a message to another agent via the Blackboard."""
        async with self._lock:
            self.messages.append(message)
            if len(self.messages) > 100:
                self.messages = self.messages[-100:]
        logger.debug(f"Message [{message.sender}->{message.receiver}]: {message.content[:40]}")

    def get_messages(self, receiver: str) -> List[AgentMessage]:
        """Read messages addressed to a specific agent."""
        return [m for m in self.messages if m.receiver == receiver]

    def consume_messages(self, receiver: str) -> List[AgentMessage]:
        """Read and remove messages addressed to a specific agent."""
        msgs = [m for m in self.messages if m.receiver == receiver]
        self.messages = [m for m in self.messages if m.receiver != receiver]
        return msgs

    async def advance_cycle(self):
        """Increment the cycle counter and timestamp."""
        async with self._lock:
            self.cycle_count += 1
            self.last_cycle_time = time.time()

    def get_recent_reflections(self, limit: int = 5) -> List[ReflectionNote]:
        """Read the most recent reflection notes."""
        return self.reflection_notes[-limit:]

    def get_recent_results(self, limit: int = 10) -> List[ExecutionResult]:
        """Read the most recent execution results."""
        return self.execution_history[-limit:]

    def set_context(self, key: str, value: Any):
        """Post a shared context entry visible to all layers."""
        self.shared_context[key] = value
        logger.debug(f"Context set: {key} = {str(value)[:80]}")

    def get_context(self, key: str, default: Any = None) -> Any:
        """Read a shared context entry."""
        return self.shared_context.get(key, default)

    def snapshot(self) -> Dict[str, Any]:
        """Produce a debug-friendly snapshot of the entire board state."""
        return {
            "cycle": self.cycle_count,
            "error_streak": self.error_streak,
            "signal": self.current_signal.signal_type.value if self.current_signal else None,
            "strategy": self.active_strategy.intent.value if self.active_strategy else None,
            "tactic": self.active_tactic.action_type.value if self.active_tactic else None,
            "last_result": self.last_result.success if self.last_result else None,
            "reflections_pending": len(self.reflection_notes),
            "shared_context_keys": list(self.shared_context.keys()),
        }
