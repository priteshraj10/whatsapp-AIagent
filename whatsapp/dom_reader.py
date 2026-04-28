"""
whatsapp/dom_reader.py -- High-fidelity WhatsApp Web DOM interaction layer.

Encapsulates all direct browser interactions, providing a structured API
for reading message bursts, sending replies, and navigating the contact list.
"""

import asyncio
import logging
import random
from typing import Any, Dict, List, Optional

from whatsapp.selectors import (
    SEL_COMPOSE, SEL_COMPOSE2,
    SEL_MSG_CONTAINER,
    SEL_SEARCH_BOX, SEL_SEARCH_BOX2,
    SEL_FIRST_RESULT,
)

# Configure logging
logger = logging.getLogger(__name__)


class DOMError(Exception):
    """Exception raised for errors during browser/DOM interaction."""
    pass


async def find_element(page: Any, *selectors: str) -> Optional[Any]:
    """
    Find the first available element from a list of CSS selectors.
    
    Args:
        page: Playwright Page instance.
        selectors: Variadic list of selector strings.
        
    Returns:
        The first found element or None.
    """
    for selector in selectors:
        try:
            element = await page.query_selector(selector)
            if element:
                return element
        except Exception as e:
            logger.debug(f"Selector '{selector}' check failed: {e}")
    return None


async def read_burst(page: Any) -> Optional[Dict[str, Any]]:
    """
    Parse the most recent message thread activity.
    
    Identifies if the last activity was an incoming burst or an outgoing message.
    
    Returns:
        A dictionary describing the last activity, or None if empty.
    """
    try:
        containers = await page.query_selector_all(SEL_MSG_CONTAINER)
        if not containers:
            return None
            
        messages: List[Dict[str, str]] = []

        # Analyse the last 15 bubbles to find the active thread tail
        for container in containers[-15:]:
            text = await _extract_bubble_text(container)
            if not text:
                continue

            direction = await _get_bubble_direction(container)
            messages.append({"direction": direction, "text": text})

        if not messages:
            return None

        # Look for the last run of consecutive incoming messages
        incoming_burst = []
        for msg in reversed(messages):
            if msg["direction"] == "incoming":
                incoming_burst.insert(0, msg["text"])
            else:
                break

        if not incoming_burst:
            # Last activity was outgoing
            return {"direction": "outgoing", "text": messages[-1]["text"]}

        return {
            "direction": "incoming",
            "text":      "\n".join(incoming_burst),
            "count":     len(incoming_burst),
        }
        
    except Exception as e:
        logger.error(f"Failed to read message burst: {e}")
        return None


async def typing_delay(text: str):
    """
    Simulate a human typing pause.
    
    Calculation: Base delay + dynamic per-character delay.
    """
    base = random.uniform(3.0, 4.5)
    per_char = min(len(text) * 0.025, 4.0)
    total = base + per_char
    logger.info(f"Simulating typing delay: {total:.1f}s")
    await asyncio.sleep(total)


async def send_message(page: Any, text: str) -> bool:
    """
    Automate the typing and submission of a message.
    
    Args:
        page: Playwright Page instance.
        text: The message content to send.
        
    Returns:
        True if the message was confirmed as sent via DOM reflection.
    """
    for attempt in range(1, 4):
        try:
            compose_box = await find_element(page, SEL_COMPOSE, SEL_COMPOSE2)
            if not compose_box:
                logger.warning(f"Compose box not found (Attempt {attempt}/3).")
                await asyncio.sleep(1.0)
                continue

            await compose_box.click()
            await asyncio.sleep(0.2)
            
            # Clear any existing text and type new message
            await compose_box.click()
            await compose_box.press("Control+A")
            await compose_box.press("Meta+A")
            await compose_box.press("Backspace")
            
            await asyncio.sleep(0.1)
            await compose_box.type(text, delay=random.randint(15, 30))
            
            await asyncio.sleep(0.3)
            await page.keyboard.press("Enter")
            
            # Wait for DOM to reflect the outgoing message
            await asyncio.sleep(1.5)
            last_activity = await read_burst(page)
            if last_activity and last_activity["direction"] == "outgoing":
                logger.info("Message delivery confirmed via DOM.")
                return True
                
        except Exception as e:
            logger.error(f"Send attempt {attempt} failed: {e}")
            await asyncio.sleep(1.0)
            
    return False


async def open_contact(page: Any, contact_name: str) -> bool:
    """
    Search and navigate to a specific contact's chat.
    
    Args:
        page: Playwright Page instance.
        contact_name: Name of the contact to open.
        
    Returns:
        True if the chat appears to be open.
    """
    logger.info(f"Navigating to chat: {contact_name}")
    try:
        search_box = await find_element(page, SEL_SEARCH_BOX, SEL_SEARCH_BOX2)
        if not search_box:
            raise DOMError("Search box not found on dashboard.")

        await search_box.click()
        await search_box.press("Control+A")
        await search_box.press("Meta+A")
        await search_box.press("Backspace")
        await search_box.type(contact_name, delay=50)
        
        # Wait for search results to populate
        await asyncio.sleep(2.0)
        
        # Match the exact contact name in the results list using native Playwright
        try:
            # Scope search to the left sidebar (pane-side) to avoid clicking the search box itself
            pane_side = page.locator('#pane-side')
            exact_match = pane_side.get_by_text(contact_name, exact=True).first
            if await exact_match.count() > 0:
                await exact_match.click()
                await asyncio.sleep(1.0)
                return True
        except Exception as e:
            logger.debug(f"pane-side text match failed: {e}")
            
        # Keyboard Navigation Fallback
        # If selectors fail, pressing ArrowDown and Enter is the most reliable human way to open the first result
        logger.info("Selector match failed, attempting keyboard navigation fallback.")
        await search_box.click()
        await asyncio.sleep(0.5)
        await page.keyboard.press("ArrowDown")
        await asyncio.sleep(0.2)
        await page.keyboard.press("Enter")
        await asyncio.sleep(1.0)
        
        # Verify if a chat actually opened (the message compose box should be visible)
        compose_box = await page.query_selector(SEL_COMPOSE) or await page.query_selector(SEL_COMPOSE2)
        if compose_box:
            return True

        logger.warning(f"No results found for contact: {contact_name}")
        return False
        
    except Exception as e:
        logger.error(f"Failed to open contact '{contact_name}': {e}")
        return False


# ------------------------------------------------------------------ #
# Internal Helpers                                                     #
# ------------------------------------------------------------------ #

async def _extract_bubble_text(container: Any) -> str:
    """Extract text from a message bubble container with fallback logic."""
    # Delegate entirely to JS evaluation for robust multimedia and text extraction
    # Strategy B: Fallback JS evaluation with Multimedia Detection
    return await container.evaluate("""el => {
        let prefix = "";
        
        // Detect Multimedia
        if (el.querySelector('[data-icon="image"]') || el.querySelector('img[src^="blob:"]')) {
            prefix = "[Image Attached] ";
        } else if (el.querySelector('[data-icon="audio-play"]') || el.querySelector('[data-testid="audio-play"]')) {
            prefix = "[Voice Note Received] ";
        } else if (el.querySelector('[data-icon="document"]')) {
            prefix = "[Document Attached] ";
        } else if (el.querySelector('img[src*="sticker"]')) {
            prefix = "[Sticker] ";
        } else if (el.querySelector('[data-icon="video"]')) {
            prefix = "[Video Attached] ";
        }
        
        // Extract Text
        let text = "";
        const copyable = el.querySelector('[data-pre-plain-text]');
        if (copyable) {
            text = copyable.innerText;
        } else {
            text = el.innerText;
        }
        
        // Clean up text
        text = text.trim();
        // Sometimes WhatsApp appends timestamps to the innerText; try to strip common time formats at the end
        text = text.replace(/\\n?\\d{1,2}:\\d{2}\\s*(AM|PM)?$/i, '').trim();
        
        return (prefix + text).trim();
    }""")

async def _get_bubble_direction(container: Any) -> str:
    """Determine if a message is incoming or outgoing based on CSS classes."""
    cls = await container.evaluate(
        "el => el.closest('.message-in,.message-out')?.className || ''"
    )
    return "incoming" if "message-in" in cls else "outgoing"
