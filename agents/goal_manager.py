"""
agents/goal_manager.py -- Goal tracking and decomposition for the agentic system.

The GoalManager maintains a stack of goals and sub-goals, evaluates completion
conditions, and persists goal state to the database.

Responsibilities:
    - Track the primary goal and any sub-goals.
    - Evaluate whether goals are met based on execution results.
    - Provide goal context to the Strategist.
    - Persist goal state across restarts via agent_state table.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GoalStatus(str, Enum):
    """Lifecycle states for a goal."""
    PENDING     = "pending"
    ACTIVE      = "active"
    COMPLETED   = "completed"
    FAILED      = "failed"
    ABANDONED   = "abandoned"


@dataclass
class Goal:
    """A single goal with metadata."""
    description: str
    status: GoalStatus = GoalStatus.PENDING
    priority: int = 0                         # Higher = more important
    sub_goals: List["Goal"] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority,
            "sub_goals": [sg.to_dict() for sg in self.sub_goals],
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Goal":
        return cls(
            description=data.get("description", ""),
            status=GoalStatus(data.get("status", "pending")),
            priority=data.get("priority", 0),
            sub_goals=[cls.from_dict(sg) for sg in data.get("sub_goals", [])],
            created_at=data.get("created_at", time.time()),
            completed_at=data.get("completed_at"),
            metadata=data.get("metadata", {}),
        )


class GoalManager:
    """
    Manages the agent's goal stack with persistence.

    The primary goal (e.g., "engage naturally with contact X") is always active.
    Sub-goals can be decomposed from the primary or injected dynamically.
    """

    STATE_KEY = "goal_stack"

    def __init__(self, db: Any, contact_id: int, primary_goal: str = ""):
        self.db = db
        self.contact_id = contact_id
        self._goal_stack: List[Goal] = []

        # Load persisted state or initialise with primary goal
        self._load_state()
        if not self._goal_stack and primary_goal:
            self._goal_stack.append(
                Goal(description=primary_goal, status=GoalStatus.ACTIVE, priority=10)
            )
            self._persist_state()

    # ------------------------------------------------------------------ #
    # Public Interface                                                     #
    # ------------------------------------------------------------------ #

    def get_active_goal(self) -> Optional[Goal]:
        """Return the highest-priority active goal."""
        active = [g for g in self._goal_stack if g.status == GoalStatus.ACTIVE]
        if not active:
            return None
        return max(active, key=lambda g: g.priority)

    def get_goal_context(self) -> str:
        """Generate a text summary of current goals for the Strategist prompt."""
        if not self._goal_stack:
            return ""

        lines = ["## ACTIVE GOALS"]
        for goal in self._goal_stack:
            if goal.status in (GoalStatus.ACTIVE, GoalStatus.PENDING):
                status_icon = ">" if goal.status == GoalStatus.ACTIVE else "-"
                lines.append(f"  {status_icon} [{goal.priority}] {goal.description}")
                for sg in goal.sub_goals:
                    if sg.status in (GoalStatus.ACTIVE, GoalStatus.PENDING):
                        lines.append(f"    - {sg.description} ({sg.status.value})")

        return "\n".join(lines) if len(lines) > 1 else ""

    def add_sub_goal(self, parent_desc: str, sub_goal: Goal):
        """Add a sub-goal under a parent goal."""
        for goal in self._goal_stack:
            if goal.description == parent_desc and goal.status == GoalStatus.ACTIVE:
                goal.sub_goals.append(sub_goal)
                self._persist_state()
                logger.info(f"Sub-goal added: '{sub_goal.description}' under '{parent_desc}'")
                return
        logger.warning(f"Parent goal not found or not active: '{parent_desc}'")

    def complete_goal(self, description: str):
        """Mark a goal as completed."""
        for goal in self._goal_stack:
            if goal.description == description:
                goal.status = GoalStatus.COMPLETED
                goal.completed_at = time.time()
                self._persist_state()
                logger.info(f"Goal completed: '{description}'")
                return
            for sg in goal.sub_goals:
                if sg.description == description:
                    sg.status = GoalStatus.COMPLETED
                    sg.completed_at = time.time()
                    self._persist_state()
                    logger.info(f"Sub-goal completed: '{description}'")
                    return

    def evaluate(self, execution_success: bool, cycle_count: int):
        """
        Evaluate goal progress based on execution results.

        Currently checks for sustained engagement (primary goal)
        and error-based abandonment.
        """
        active = self.get_active_goal()
        if not active:
            return

        # The primary conversational goal is perpetual -- it doesn't "complete"
        # Sub-goals may complete based on specific conditions
        for sg in active.sub_goals:
            if sg.status == GoalStatus.ACTIVE:
                # Sub-goal-specific completion logic can be extended here
                pass

    def push_goal(self, goal: Goal):
        """Push a new goal onto the stack."""
        self._goal_stack.append(goal)
        self._persist_state()
        logger.info(f"Goal pushed: '{goal.description}' (priority={goal.priority})")

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def _persist_state(self):
        """Save the goal stack to the database."""
        try:
            data = json.dumps([g.to_dict() for g in self._goal_stack])
            self.db.set_state(self.contact_id, self.STATE_KEY, data)
        except Exception as e:
            logger.error(f"Failed to persist goal state: {e}")

    def _load_state(self):
        """Load the goal stack from the database."""
        try:
            raw = self.db.get_state(self.contact_id, self.STATE_KEY)
            if raw:
                data = json.loads(raw)
                self._goal_stack = [Goal.from_dict(g) for g in data]
                logger.debug(f"Loaded {len(self._goal_stack)} goals from DB.")
        except Exception as e:
            logger.warning(f"Failed to load goal state: {e}")
            self._goal_stack = []
