"""
core/history_ingester.py -- Senior-grade chat history ingestion service.

Handles the one-time operation of scrolling through a chat's history, 
reading all messages, and persisting them to the database.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any

from whatsapp.selectors import (
    SEL_MSG_CONTAINER, 
    SEL_MSG_PANEL, 
    SEL_MSG_PANEL_ALT
)

# Configure logging
logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Exception raised for errors during history ingestion."""
    pass


class HistoryIngester:
    """
    Automates the process of loading and reading historical chat messages.
    
    Uses multi-strategy scrolling to trigger WhatsApp's lazy-loading mechanism.
    """

    def __init__(self, page: Any, db: Any, contact_id: int):
        """
        Initialise history ingester.
        
        Args:
            page: Initialised Playwright Page instance.
            db: Initialised ConversationDB instance.
            contact_id: The ID of the contact whose history is being ingested.
        """
        self.page = page
        self.db = db
        self.contact_id = contact_id

    async def ingest(self, max_scroll_rounds: int = 60) -> int:
        """
        Main entry point: scroll to top, parse all messages, and store in DB.
        
        Args:
            max_scroll_rounds: Maximum number of scroll batches to attempt.
            
        Returns:
            Number of new (previously unseen) messages stored.
        """
        logger.info(f"Starting history ingestion for contact ID {self.contact_id}...")

        # 1. Scroll to the beginning of time (or max rounds)
        await self._scroll_to_top(max_scroll_rounds)

        # 2. Extract all messages currently in the DOM
        bubbles = await self._read_all_bubbles()
        if not bubbles:
            logger.warning("No message bubbles found in DOM after scrolling.")
            return 0

        logger.info(f"Extracted {len(bubbles)} message bubbles from DOM.")

        # 3. Filter and store unique messages
        new_count = 0
        for msg in bubbles:
            text = msg.get("text", "").strip()
            if not text:
                continue
                
            # Map DOM direction to DB role
            role = "user" if msg["direction"] == "incoming" else "assistant"
            
            if not self.db.has_message(self.contact_id, text):
                self.db.add_message(
                    contact_id=self.contact_id,
                    role=role,
                    content=text,
                    intent="history"
                )
                new_count += 1

        logger.info(f"Ingestion complete. Added {new_count} new messages to DB.")
        return new_count

    async def _scroll_to_top(self, max_rounds: int):
        """
        Repeatedly scrolls up to trigger lazy-loading of older messages.
        """
        stale_rounds = 0
        prev_count = 0
        STALE_LIMIT = 3  # Rounds with no new data before we assume top is reached

        for i in range(max_rounds):
            await self._execute_scroll_action()
            # Allow time for WhatsApp to fetch and render the batch
            await asyncio.sleep(1.8)

            current_count = await self._get_bubble_count()
            new_loaded = current_count - prev_count

            if new_loaded > 0:
                logger.debug(f"Scroll round {i+1}: loaded {new_loaded} more messages.")
                stale_rounds = 0
            else:
                stale_rounds += 1
                if stale_rounds >= STALE_LIMIT:
                    logger.info("Reached top of chat history (no new messages for 3 rounds).")
                    break

            prev_count = current_count
        else:
            logger.info("Reached maximum scroll rounds.")

    async def _execute_scroll_action(self):
        """Multi-technique scroll-up for maximum reliability."""
        # Method 1: Direct JS manipulation
        await self.page.evaluate(f"""() => {{
            const selectors = ['{SEL_MSG_PANEL}', '{SEL_MSG_PANEL_ALT}', '#main'];
            for (const sel of selectors) {{
                const el = document.querySelector(sel);
                if (el) {{
                    el.scrollTop = 0; // Jump to current batch top
                    return;
                }}
            }}
        }}""")
        
        # Method 2: PageUp keyboard events (simulates user interaction)
        try:
            panel = await self.page.query_selector(SEL_MSG_PANEL) or \
                    await self.page.query_selector(SEL_MSG_PANEL_ALT)
            if panel:
                await panel.focus()
                for _ in range(8):
                    await self.page.keyboard.press("PageUp")
                    await asyncio.sleep(0.05)
        except Exception as e:
            logger.debug(f"Keyboard scroll failed: {e}")

    async def _get_bubble_count(self) -> int:
        """Count the number of message containers currently in the DOM."""
        try:
            elements = await self.page.query_selector_all(SEL_MSG_CONTAINER)
            return len(elements)
        except Exception:
            return 0

    async def _read_all_bubbles(self) -> List[Dict[str, str]]:
        """Parse all message containers into a structured list."""
        containers = await self.page.query_selector_all(SEL_MSG_CONTAINER)
        messages = []

        for container in containers:
            try:
                # Primary text extraction (standard WhatsApp)
                text_el = await container.query_selector(".selectable-text span") or \
                          await container.query_selector(".selectable-text")
                
                if text_el:
                    text = (await text_el.inner_text()).strip()
                else:
                    # Fallback for updated/dynamic DOM structures
                    text = await container.evaluate("""el => {
                        const copyable = el.querySelector('[data-pre-plain-text]');
                        if (copyable) return copyable.innerText;
                        return el.innerText;
                    }""")
                    text = text.strip() if text else ""
                
                if not text:
                    continue

                # Determine direction (in vs out)
                cls = await container.evaluate(
                    "el => el.closest('.message-in,.message-out')?.className || ''"
                )
                direction = "incoming" if "message-in" in cls else "outgoing"
                
                messages.append({"direction": direction, "text": text})
            except Exception as e:
                logger.debug(f"Failed to parse bubble: {e}")
                continue

        return messages
