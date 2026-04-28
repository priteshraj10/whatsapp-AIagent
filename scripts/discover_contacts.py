"""
scripts/discover_contacts.py -- Autonomous Contact Discovery & Categorization.

This script:
  1. Opens WhatsApp Web and scrolls through ALL chats in the sidebar.
  2. Extracts every contact/group name found.
  3. For each contact, reads their last few messages and uses the ContactClassifier
     to determine if they are a Person, Business, or Bot.
  4. For human contacts, the profiler's LLM will guess the Relationship Category
     (Close Friend, Family, Colleague, Acquaintance).
  5. Stores everything in the SQLite database, queryable by category.

Usage:
    python -m scripts.discover_contacts

Output:
    Prints a summary table of all discovered contacts grouped by category.
    Also persists results to: data/conversations.db
"""

import asyncio
import logging
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from playwright.async_api import async_playwright, Page

from integrations.database import ConversationDB
from skills.contact_classifier import ContactClassifier, ContactType
from core.llm_client import LLMClient

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

WHATSAPP_URL = "https://web.whatsapp.com"
USER_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "browser_data")

# Relationship categories in display order
CATEGORIES = ["Close Friend", "Family", "Colleague", "Acquaintance", "Business/Spam", "unknown"]


# ------------------------------------------------------------------ #
# DOM Helpers                                                          #
# ------------------------------------------------------------------ #

async def wait_for_whatsapp(page: Page):
    """Block until the WhatsApp sidebar is visible."""
    logger.info("Waiting for WhatsApp Web to load...")
    await page.wait_for_url("https://web.whatsapp.com/", timeout=120_000)
    await page.wait_for_selector('#pane-side', timeout=60_000)
    logger.info("WhatsApp is ready.")


async def discover_chat_names(page: Page) -> list[str]:
    """
    Scroll the entire chat list and extract all unique contact/group names.
    Uses JS evaluation for maximum reliability.
    """
    logger.info("Scrolling chat list to discover all contacts...")
    names = []
    seen = set()
    scroll_container_sel = '#pane-side > div:first-child > div:first-child'

    # Scroll down repeatedly to load lazy-rendered chats
    last_count = 0
    no_change_rounds = 0

    while no_change_rounds < 3:
        # Extract all currently visible contact titles
        raw_names = await page.evaluate("""
            () => {
                const spans = document.querySelectorAll('#pane-side span[title]');
                return Array.from(spans).map(s => s.getAttribute('title')).filter(t => t && t.trim().length > 0);
            }
        """)

        for name in raw_names:
            name = name.strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)

        if len(names) == last_count:
            no_change_rounds += 1
        else:
            no_change_rounds = 0
            last_count = len(names)

        # Scroll down within the pane
        await page.evaluate("""
            () => {
                const pane = document.querySelector('#pane-side');
                if (pane) pane.scrollTop += 2000;
            }
        """)
        await asyncio.sleep(0.8)

    logger.info(f"Discovered {len(names)} unique contacts/groups.")
    return names


async def fetch_last_messages(page: Page, contact_name: str) -> list[dict]:
    """Open a contact's chat, read a few messages, then return to the sidebar."""
    try:
        # Search for the contact
        search_box = await page.query_selector('[data-testid="chat-list-search"]')
        if not search_box:
            return []

        await search_box.click()
        await search_box.press("Control+A")
        await search_box.press("Meta+A")
        await search_box.press("Backspace")
        await search_box.type(contact_name, delay=40)
        await asyncio.sleep(1.5)

        # Click first result
        await page.keyboard.press("ArrowDown")
        await asyncio.sleep(0.2)
        await page.keyboard.press("Enter")
        await asyncio.sleep(1.2)

        # Extract last 10 message bubbles
        messages = await page.evaluate("""
            () => {
                const bubbles = document.querySelectorAll('[data-testid="msg-container"]');
                const results = [];
                bubbles.forEach(b => {
                    const text = b.querySelector('[data-testid="balloon-text-content"]')?.innerText
                             || b.querySelector('span.copyable-text')?.innerText
                             || '';
                    const isOut = b.closest('[data-testid="msg-container"]')?.classList.contains('message-out');
                    if (text.trim()) {
                        results.push({ role: isOut ? 'assistant' : 'user', content: text.trim() });
                    }
                });
                return results.slice(-10);
            }
        """)

        # Go back to sidebar (press Escape to close chat or navigate back)
        await page.keyboard.press("Escape")
        await asyncio.sleep(0.5)

        return messages or []

    except Exception as e:
        logger.debug(f"Failed to fetch messages for '{contact_name}': {e}")
        return []


# ------------------------------------------------------------------ #
# Main Discovery Loop                                                  #
# ------------------------------------------------------------------ #

async def run_discovery():
    db = ConversationDB()
    llm = LLMClient()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR,
            headless=False,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        page = browser.pages[0] if browser.pages else await browser.new_page()

        await page.goto(WHATSAPP_URL, wait_until="domcontentloaded")
        await wait_for_whatsapp(page)

        # Step 1: Discover all chat names from the sidebar
        all_names = await discover_chat_names(page)

        logger.info("=" * 60)
        logger.info(f"Processing {len(all_names)} contacts...")
        logger.info("=" * 60)

        processed = 0
        for name in all_names:
            try:
                contact_id = db.get_or_create_contact(name)

                # Check if we already have a category for this contact
                existing_category = db.get_state(contact_id, "relationship_category")
                if existing_category and existing_category not in ("unknown", "Unknown"):
                    logger.info(f"[SKIP] '{name}' already categorized as '{existing_category}'")
                    processed += 1
                    continue

                # Fetch last few messages to assist classification
                recent_messages = await fetch_last_messages(page, name)

                # Step 2: Classify the contact (Person / Business / Bot)
                classifier = ContactClassifier(db, contact_id, name)
                contact_type = await classifier.classify_with_llm(llm, recent_messages)

                # Step 3: For human contacts, use LLM to guess relationship category
                if contact_type == ContactType.PERSON and recent_messages:
                    convo_text = "\n".join(
                        f"[{m['role'].upper()}]: {m['content'][:120]}"
                        for m in recent_messages[-10:]
                    )
                    prompt = [
                        {
                            "role": "system",
                            "content": (
                                "Based on this WhatsApp conversation, guess the relationship between the user and the contact named "
                                f"'{name}'. Choose EXACTLY ONE category from: "
                                "'Close Friend', 'Family', 'Colleague', 'Acquaintance'. "
                                "Reply with ONLY the category name, nothing else."
                            ),
                        },
                        {"role": "user", "content": f"Conversation:\n{convo_text}"},
                    ]
                    try:
                        raw = await llm.chat(prompt)
                        category = raw.strip().strip('"').strip("'")
                        # Validate the response
                        valid_cats = ["Close Friend", "Family", "Colleague", "Acquaintance"]
                        if category not in valid_cats:
                            category = "Acquaintance"  # safe default
                        db.set_state(contact_id, "relationship_category", category)
                        logger.info(f"[CATEGORIZED] '{name}' → {category}")
                    except Exception as e:
                        logger.warning(f"LLM category failed for '{name}': {e}")
                        db.set_state(contact_id, "relationship_category", "Acquaintance")

                processed += 1
                logger.info(f"[{processed}/{len(all_names)}] Done: {name}")
                await asyncio.sleep(0.3)  # Rate limit LLM calls

            except Exception as e:
                logger.error(f"Failed processing '{name}': {e}")

        await browser.close()

    # Step 4: Print summary
    _print_summary(db)


def _print_summary(db: ConversationDB):
    """Print a formatted summary of all contacts grouped by relationship category."""
    print("\n" + "=" * 60)
    print("  CONTACT RELATIONSHIP SUMMARY")
    print("=" * 60)

    total = 0
    for category in CATEGORIES:
        contacts = db.get_contacts_by_category(category)
        if contacts:
            print(f"\n📋 {category.upper()} ({len(contacts)})")
            for name in contacts:
                print(f"   • {name}")
            total += len(contacts)

    # Contacts with no category yet
    try:
        conn = db._conn
        all_contacts = conn.execute("SELECT name FROM contacts").fetchall()
        categorized = set()
        for cat in CATEGORIES:
            for name in db.get_contacts_by_category(cat):
                categorized.add(name)

        uncategorized = [r[0] for r in all_contacts if r[0] not in categorized]
        if uncategorized:
            print(f"\n❓ UNCATEGORIZED ({len(uncategorized)})")
            for name in uncategorized:
                print(f"   • {name}")
    except Exception:
        pass

    print(f"\n{'='*60}")
    print(f"  Total contacts: {total}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(run_discovery())
