"""
web_agent_engine.py -- High-availability Playwright browser orchestration engine.

Provides a robust execution environment for web agents, including:
- Graceful Chrome profile lock management.
- Automated UI debris cleanup (startup dialogs, popups).
- Fault-tolerant operational loop with error budgeting.
- Centralised browser lifecycle management.
- Multi-layered agentic system support with debug introspection.
"""

import asyncio
import logging
import os
import subprocess
import sys
from typing import Type, Dict, Any, Optional

from playwright.async_api import async_playwright, BrowserContext, Page, Playwright
from core.base_agent import BaseAgent

# Optional: import for type-hinting the orchestrator's blackboard
try:
    from agents.blackboard import Blackboard
except ImportError:
    Blackboard = None  # Graceful degradation if agents package not loaded

# Configure logging
logger = logging.getLogger(__name__)


class EngineError(Exception):
    """Exception raised for fatal browser engine errors."""
    pass


class WebAgentEngine:
    """
    The orchestration engine for browser-based agents.
    
    Handles persistent browser sessions, avoids startup crashes, and provides
    the operational loop that drives agent decision-making.
    """

    def __init__(self, user_data_dir: str = "./chrome_profile", headless: bool = False):
        """
        Initialise engine settings.
        
        Args:
            user_data_dir: Local path for the persistent Chrome profile.
            headless: Whether to run without a visible UI.
        """
        self.user_data_dir = os.path.abspath(user_data_dir)
        self.headless = headless
        self._ensure_profile_dir()

    def _ensure_profile_dir(self):
        """Verify the profile directory exists."""
        if not os.path.exists(self.user_data_dir):
            os.makedirs(self.user_data_dir, exist_ok=True)
            logger.info(f"Created new Chrome profile directory at {self.user_data_dir}")

    async def run_agent_loop(
        self,
        agent_class: Type[BaseAgent],
        goal: str,
        **agent_kwargs: Any,
    ):
        """
        Main operational loop. Executes the agent until termination or error budget exhaustion.
        
        Args:
            agent_class: The Agent class to instantiate.
            goal: The master objective for this run.
            **agent_kwargs: Configuration to pass to the agent constructor.
        """
        max_consecutive_errors = 10
        error_count = 0

        async with async_playwright() as playwright:
            context = await self._launch_browser(playwright)
            page = await context.new_page()

            # Force Chromium window to front on macOS
            await page.bring_to_front()
            if sys.platform == "darwin":
                try:
                    subprocess.Popen(
                        ['osascript', '-e',
                         'tell application "System Events" to set frontmost of '
                         '(first process whose name contains "Chromium") to true'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                except Exception:
                    pass  # Non-critical: agent works fine even if window stays behind

            # Sanitise UI state
            await self._dismiss_startup_popups(page)

            # Instantiate and configure agent
            agent = agent_class(page, **agent_kwargs)
            agent.goal = goal

            logger.info(f"Starting agent '{agent_class.__name__}' with goal: {goal}")

            try:
                while True:
                    try:
                        # Periodic popup cleanup
                        await self._dismiss_startup_popups(page, silent=True)

                        # Decision phase
                        decision = await agent.decide()
                        status = decision.get("status", "active")

                        if status == "stop":
                            logger.info(f"Agent requested stop: {decision.get('summary', 'Task complete.')}")
                            break

                        if status == "failed":
                            error_count += 1
                            logger.warning(f"Agent reported failure. Error count: {error_count}/{max_consecutive_errors}")
                        else:
                            error_count = 0  # Reset on success

                        # Execution phase
                        if "action" in decision:
                            await agent.execute_action(decision["action"])

                        if error_count >= max_consecutive_errors:
                            logger.critical("Error budget exhausted. Shutting down engine.")
                            break

                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        error_count += 1
                        logger.error(f"Loop runtime error ({error_count}/{max_consecutive_errors}): {e}")
                        if error_count >= max_consecutive_errors:
                            break
                        await asyncio.sleep(3)

            except asyncio.CancelledError:
                logger.info("Agent loop cancelled by system.")
            finally:
                logger.info("Closing browser engine...")
                try:
                    await context.close()
                except Exception:
                    pass

    async def _dismiss_startup_popups(self, page: Page, silent: bool = False):
        """Clean away common Chrome and WhatsApp UI debris."""
        dismiss_selectors = [
            'button:has-text("OK")',
            'button:has-text("Ok")',
            '[aria-label="Close"]',
            '[data-testid="popup-controls-close"]',
            'button[aria-label="Close"]'
        ]

        count = 0
        for selector in dismiss_selectors:
            try:
                # Short timeout to avoid stalling the loop
                element = await page.query_selector(selector, timeout=1500)
                if element and await element.is_visible():
                    await element.click()
                    count += 1
                    await asyncio.sleep(0.3)
            except Exception:
                continue

        if count > 0 and not silent:
            logger.info(f"Engine UI-Cleanup: Dismissed {count} popups.")

    async def _launch_browser(self, p: Playwright) -> BrowserContext:
        """Configure and launch the Chromium instance."""
        # Cleanup stale singleton locks (common source of "profile in use" errors)
        self._clear_stale_locks()

        logger.info("Launching persistent Chromium context...")
        return await p.chromium.launch_persistent_context(
            user_data_dir=self.user_data_dir,
            headless=self.headless,
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            args=[
                "--disable-notifications",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-restore-session-state",
                "--no-first-run",
                "--disable-crash-reporter",
                "--no-default-browser-check",
                "--disable-session-crashed-bubble",
                "--js-flags=--max-old-space-size=256",
                "--window-size=1280,800",
                "--window-position=100,50",
            ],
        )

    def _clear_stale_locks(self):
        """Remove SQLite and Singleton locks from the profile directory."""
        lock_files = ["SingletonLock", "SingletonCookie", "SingletonSocket"]
        for lock_name in lock_files:
            path = os.path.join(self.user_data_dir, lock_name)
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.debug(f"Removed stale lock: {lock_name}")
            except Exception as e:
                logger.warning(f"Failed to remove stale lock {lock_name}: {e}")
