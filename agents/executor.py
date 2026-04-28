"""
agents/executor.py -- Execution layer of the agentic system.

The ExecutionAgent is the "do it and verify" layer.  It reads a Tactic from the
Blackboard, dispatches the appropriate browser action, verifies success via
DOM reflection, and records results to the database.

Now with HumanMimicry integration for realistic behavior:
    - Read-then-think delays before typing
    - Variable typing speed with micro-pauses
    - Message splitting into natural bursts
    - Text humanization (casual imperfections)
"""

import asyncio
import logging
from typing import Any, Optional

from agents.blackboard import (
    Blackboard, ExecutionResult, Tactic, TacticAction,
)
from skills.human_mimicry import HumanMimicry
import whatsapp.dom_reader as dom
from whatsapp.selectors import (
    SEL_MSG_ROW, SEL_MSG_DROPDOWN, SEL_REACT_MENU, 
    SEL_DELETE_OPTION, SEL_DELETE_FOR_EVERYONE, SEL_OK_BUTTON
)

logger = logging.getLogger(__name__)


class ExecutionAgent:
    """
    Dispatches concrete browser actions and verifies their outcomes.

    Integrates HumanMimicry for realistic typing, timing, and message delivery.
    """

    MAX_RETRIES = 2
    RETRY_DELAY = 2.0

    def __init__(
        self,
        page: Any,
        blackboard: Blackboard,
        db: Any,
        profiler: Any,
        contact_id: int,
        contact_name: str,
    ):
        self.page = page
        self.bb = blackboard
        self.db = db
        self.profiler = profiler
        self.contact_id = contact_id
        self.contact_name = contact_name

        # Human behavior simulation
        self.human = HumanMimicry()

        # Hash tracking for deduplication
        self._last_replied_hash: Optional[str] = None

    def set_last_replied_hash(self, h: Optional[str]):
        """Seed the dedup hash from DB state."""
        self._last_replied_hash = h

    def get_last_replied_hash(self) -> Optional[str]:
        """Get the current dedup hash."""
        return self._last_replied_hash

    async def execute(self) -> ExecutionResult:
        """
        Execute the active Tactic from the Blackboard.

        Returns:
            An ExecutionResult (also written to the Blackboard).
        """
        tactic = self.bb.active_tactic
        if not tactic:
            result = ExecutionResult(
                success=True,
                action_type=TacticAction.NONE,
                details="No tactic to execute.",
            )
            await self.bb.post_result(result)
            return result

        handler_map = {
            TacticAction.REPLY: self._execute_reply,
            TacticAction.REACT: self._execute_react,
            TacticAction.DELETE: self._execute_delete,
            TacticAction.NAVIGATE: self._execute_navigate,
            TacticAction.OPEN_CHAT: self._execute_open_chat,
            TacticAction.WAIT: self._execute_wait,
            TacticAction.BOOTSTRAP: self._execute_bootstrap,
            TacticAction.USE_TOOL: self._execute_wait,
            TacticAction.NONE: self._execute_noop,
        }

        handler = handler_map.get(tactic.action_type, self._execute_wait)

        try:
            result = await handler(tactic)
        except Exception as e:
            logger.error(f"Execution failed for {tactic.action_type.value}: {e}")
            result = ExecutionResult(
                success=False,
                action_type=tactic.action_type,
                details=f"Unhandled error: {e}",
            )

        await self.bb.post_result(result)
        return result

    # ------------------------------------------------------------------ #
    # Action Handlers                                                      #
    # ------------------------------------------------------------------ #

    async def _execute_reply(self, tactic: Tactic) -> ExecutionResult:
        """
        Send a reply with full human mimicry:
            1. Record incoming message
            2. Read + think delay (proportional to incoming length)
            3. Humanize text (subtle imperfections)
            4. Split into multiple messages if natural
            5. Type each part with variable speed
            6. Verify delivery
        """
        text = tactic.reply_text
        if not text:
            return ExecutionResult(
                success=False,
                action_type=TacticAction.REPLY,
                details="Empty reply text -- nothing to send.",
            )

        # Step 1: Record the incoming message
        signal = self.bb.current_signal
        incoming_text = ""
        if signal and signal.payload.get("text"):
            incoming_text = signal.payload["text"]
            self.db.add_message(self.contact_id, "user", incoming_text)

        # Step 2: Read + think delay (human reads before replying)
        is_proactive = not bool(incoming_text)
        read_think = self.human.compute_read_think_delay(incoming_text, is_proactive)
        logger.info(f"Simulating read+think pause: {read_think:.1f}s")
        await asyncio.sleep(read_think)

        # Step 3: Humanize the text
        text = self.human.humanize_text(text)

        # Step 4: Split message if appropriate
        message_parts = self.human.split_message(text)
        if len(message_parts) > 1:
            logger.info(f"Splitting reply into {len(message_parts)} parts")

        # Step 5: Send each part with realistic typing
        all_sent = True
        for i, part in enumerate(message_parts):
            # Typing delay (variable speed)
            type_duration = self.human.compute_typing_duration(part)
            logger.info(f"Typing part {i+1}/{len(message_parts)} ({len(part)} chars, {type_duration:.1f}s)...")
            await asyncio.sleep(type_duration)

            # Attempt send with retries
            part_sent = False
            for attempt in range(1, self.MAX_RETRIES + 2):
                success = await dom.send_message(self.page, part)
                if success:
                    part_sent = True
                    break
                logger.warning(f"Send attempt {attempt} failed for part {i+1}. Retrying...")
                await asyncio.sleep(self.RETRY_DELAY)

            if not part_sent:
                all_sent = False
                break

            # Inter-message delay for split messages
            if i < len(message_parts) - 1:
                gap = self.human.inter_message_delay()
                logger.info(f"Inter-message pause: {gap:.1f}s")
                await asyncio.sleep(gap)

        if all_sent:
            # Record the full message to DB
            self.db.add_message(
                contact_id=self.contact_id,
                role="assistant",
                content=text,
                intent=tactic.intent_label,
                emotion=tactic.emotion_detected,
                confidence=tactic.confidence,
            )

            # Update dedup hash
            content_hash = signal.payload.get("content_hash", "") if signal else ""
            self._update_hash(content_hash)

            # Update profiler
            self.profiler.update_from_exchange(tactic.emotion_detected)

            # Track exchange for fatigue simulation
            self.human.record_exchange()

            logger.info("Message delivery confirmed.")
            return ExecutionResult(
                success=True,
                action_type=TacticAction.REPLY,
                details=f"Replied ({len(message_parts)} parts): {text[:80]}",
            )

        return ExecutionResult(
            success=False,
            action_type=TacticAction.REPLY,
            details=f"Failed to deliver after {self.MAX_RETRIES + 1} attempts.",
        )

    async def _execute_react(self, tactic: Tactic) -> ExecutionResult:
        """React to a specific message using Playwright hover and click."""
        if not tactic.target_text or not tactic.reaction_emoji:
            return ExecutionResult(False, TacticAction.REACT, "Missing target or emoji")
            
        try:
            # 1. Find the message bubble containing the text
            escaped_text = tactic.target_text.replace('"', '\\"')
            msg_locator = self.page.locator(f'{SEL_MSG_ROW}:has-text("{escaped_text}")').last
            
            if not await msg_locator.is_visible(timeout=2000):
                return ExecutionResult(False, TacticAction.REACT, "Target message not found on screen")
                
            # 2. Hover over the message to reveal the dropdown arrow
            await msg_locator.hover()
            await asyncio.sleep(0.5)
            
            # 3. Click the dropdown arrow
            dropdown = msg_locator.locator(SEL_MSG_DROPDOWN)
            if not await dropdown.is_visible():
                 return ExecutionResult(False, TacticAction.REACT, "Dropdown arrow didn't appear")
            await dropdown.click()
            
            # 4. Click React from the menu
            react_btn = self.page.locator(SEL_REACT_MENU)
            if await react_btn.is_visible(timeout=1000):
                # Click the emoji from the quick tray
                emoji_btn = react_btn.locator(f'button:has-text("{tactic.reaction_emoji}")')
                if await emoji_btn.is_visible():
                    await emoji_btn.click()
                else:
                    return ExecutionResult(False, TacticAction.REACT, "Emoji not in quick menu")
            else:
                 li = self.page.locator('li:has-text("React")')
                 if await li.is_visible():
                     await li.click()
                     await asyncio.sleep(0.5)
                     emoji_btn = self.page.locator(f'[data-testid="reactions-tray"] button:has-text("{tactic.reaction_emoji}")')
                     if await emoji_btn.is_visible():
                         await emoji_btn.click()
                     else:
                         return ExecutionResult(False, TacticAction.REACT, "Emoji not found in full tray")
                 else:
                     return ExecutionResult(False, TacticAction.REACT, "React option not found")

            return ExecutionResult(True, TacticAction.REACT, f"Reacted with {tactic.reaction_emoji}")
        except Exception as e:
             return ExecutionResult(False, TacticAction.REACT, f"React failed: {e}")

    async def _execute_delete(self, tactic: Tactic) -> ExecutionResult:
        """Delete an outgoing message for everyone."""
        if not tactic.target_text:
            return ExecutionResult(False, TacticAction.DELETE, "Missing target text")
            
        try:
            escaped_text = tactic.target_text.replace('"', '\\"')
            msg_locator = self.page.locator(f'{SEL_MSG_ROW}.message-out:has-text("{escaped_text}")').last
            
            if not await msg_locator.is_visible(timeout=2000):
                return ExecutionResult(False, TacticAction.DELETE, "Target outgoing message not found")
                
            await msg_locator.hover()
            await asyncio.sleep(0.5)
            
            dropdown = msg_locator.locator(SEL_MSG_DROPDOWN)
            if not await dropdown.is_visible():
                 return ExecutionResult(False, TacticAction.DELETE, "Dropdown arrow didn't appear")
            await dropdown.click()
            
            delete_opt = self.page.locator(SEL_DELETE_OPTION)
            if not await delete_opt.is_visible(timeout=1000):
                return ExecutionResult(False, TacticAction.DELETE, "Delete option not found")
            await delete_opt.click()
            
            # Modal confirmation
            delete_everyone = self.page.locator(SEL_DELETE_FOR_EVERYONE)
            if await delete_everyone.is_visible(timeout=2000):
                await delete_everyone.click()
                
                # Confirm OK if it appears
                ok_btn = self.page.locator(SEL_OK_BUTTON)
                if await ok_btn.is_visible(timeout=1000):
                    await ok_btn.click()
                    
                return ExecutionResult(True, TacticAction.DELETE, "Deleted for everyone")
            else:
                return ExecutionResult(False, TacticAction.DELETE, "Delete for everyone not available")
                
        except Exception as e:
             return ExecutionResult(False, TacticAction.DELETE, f"Delete failed: {e}")

    async def _execute_navigate(self, tactic: Tactic) -> ExecutionResult:
        """Navigate to a URL."""
        url = tactic.parameters.get("url", "https://web.whatsapp.com")
        try:
            await self.page.goto(url)
            await asyncio.sleep(2)
            return ExecutionResult(
                success=True,
                action_type=TacticAction.NAVIGATE,
                details=f"Navigated to {url}",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                action_type=TacticAction.NAVIGATE,
                details=f"Navigation failed: {e}",
            )

    async def _execute_open_chat(self, tactic: Tactic) -> ExecutionResult:
        """Open a specific contact's chat."""
        contact = tactic.parameters.get("contact", self.contact_name)
        success = await dom.open_contact(self.page, contact)
        return ExecutionResult(
            success=success,
            action_type=TacticAction.OPEN_CHAT,
            details=f"{'Opened' if success else 'Failed to open'} chat with {contact}",
        )

    async def _execute_wait(self, tactic: Tactic) -> ExecutionResult:
        """Wait action — no-op since the engine poll loop already sleeps."""
        return ExecutionResult(
            success=True,
            action_type=TacticAction.WAIT,
            details="Waiting (engine poll handles timing).",
        )

    async def _execute_bootstrap(self, tactic: Tactic) -> ExecutionResult:
        """Trigger history bootstrap (delegated to orchestrator)."""
        return ExecutionResult(
            success=True,
            action_type=TacticAction.BOOTSTRAP,
            details="Bootstrap signal acknowledged. Orchestrator will handle.",
        )

    async def _execute_noop(self, tactic: Tactic) -> ExecutionResult:
        """Handle decisions where the agent chose not to act."""
        signal = self.bb.current_signal
        if signal and signal.payload.get("content_hash"):
            self._update_hash(signal.payload["content_hash"])

        if signal and signal.payload.get("text"):
            self.db.add_message(self.contact_id, "user", signal.payload["text"])

        logger.info(f"Agent chose no action. Reason: {tactic.reasoning[:80]}")
        return ExecutionResult(
            success=True,
            action_type=TacticAction.NONE,
            details=f"No action: {tactic.reasoning[:80]}",
        )

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _update_hash(self, h: str):
        """Update the dedup hash in both memory and persistent storage."""
        if h:
            self._last_replied_hash = h
            self.db.set_state(self.contact_id, "last_replied_hash", h)
