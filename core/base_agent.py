"""
base_agent.py -- Standardised base class for autonomous web agents.

Defines the core Observe-Decide-Act lifecycle and provides common 
browser interaction primitives.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all web agents.
    
    Implements common browser interaction patterns and enforces the
    architectural lifecycle of an autonomous agent.
    """
    
    def __init__(self, page: Any):
        """
        Initialise the agent.
        
        Args:
            page: Initialised Playwright Page object.
        """
        self.page = page
        self.history: List[Dict[str, Any]] = []
        self.goal: Optional[str] = None

    @abstractmethod
    async def observe(self) -> str:
        """Analyze what is currently visible on the screen."""
        pass

    @abstractmethod
    async def understand(self) -> str:
        """Infer the state and meaning of the current page."""
        pass

    @abstractmethod
    async def plan(self) -> str:
        """Determine the next actionable step toward the goal."""
        pass

    @abstractmethod
    async def decide(self) -> Dict[str, Any]:
        """Choose the optimal action to perform next."""
        pass

    @abstractmethod
    async def verify(self) -> bool:
        """Check if the previous action achieved its intended result."""
        pass

    async def execute_action(self, action: Dict[str, Any]):
        """
        Execute a structured action on the browser page.
        
        Args:
            action: A dictionary defining the action type and parameters.
        """
        action_type = action.get("type")
        logger.debug(f"Executing action: {action_type} with payload: {action}")

        try:
            if action_type == "click":
                await self._action_click(action)
            elif action_type == "type":
                await self._action_type(action)
            elif action_type == "press":
                await self.page.keyboard.press(action["value"])
            elif action_type == "scroll":
                await self._action_scroll(action)
            elif action_type == "navigate":
                await self._action_navigate(action)
            elif action_type == "wait":
                await asyncio.sleep(float(action.get("value", 1.0)))
            else:
                logger.warning(f"Unsupported action type attempted: {action_type}")
        except Exception as e:
            logger.error(f"Action execution failed ({action_type}): {e}")

    # ------------------------------------------------------------------ #
    # Action Implementation Helpers                                        #
    # ------------------------------------------------------------------ #

    async def _action_click(self, action: Dict[str, Any]):
        target = action.get("target")
        if target and str(target).isdigit():
            # Legacy numeric mapping fallback
            await self.page.click(f'[data-agent-id="{target}"]')
        elif target:
            await self.page.click(target)
        else:
            await self.page.mouse.click(action.get("x", 0), action.get("y", 0))

    async def _action_type(self, action: Dict[str, Any]):
        target = action.get("target")
        value = str(action.get("value", ""))
        
        if target:
            # Handle both numeric IDs and standard selectors
            selector = f'[data-agent-id="{target}"]' if str(target).isdigit() else target
            
            # WhatsApp uses complex contenteditables, so we prefer focus + keyboard.type
            try:
                await self.page.focus(selector)
                await self.page.click(selector)
                await self.page.keyboard.type(value)
            except Exception:
                # Fallback to standard fill
                await self.page.fill(selector, value)
        else:
            # Global type
            await self.page.keyboard.type(value)

    async def _action_scroll(self, action: Dict[str, Any]):
        direction = action.get("value", "down")
        offset = 500 if direction == "down" else -500
        await self.page.mouse.wheel(0, offset)

    async def _action_navigate(self, action: Dict[str, Any]):
        url = action.get("target") or action.get("value", "")
        if url:
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            await self.page.goto(url)
