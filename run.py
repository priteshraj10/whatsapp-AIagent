"""
run.py -- Entry point and CLI orchestrator for the Multi-Layered WhatsApp Agent.

Features:
- Configurable logging levels with per-layer debug.
- Graceful shutdown handling.
- Interactive and CLI argument support.
- Dependency injection via engine and agent configuration.
- Supports both legacy WhatsAppAgent and new AgentOrchestrator.
"""

import asyncio
import argparse
import logging
import sys
import os

# Set up logging before other imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Launcher")

# Project environment setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from engine.browser import WebAgentEngine
    from agents import AgentOrchestrator
    from core.persona import Persona
except ImportError as e:
    logger.critical(f"Critical dependency missing: {e}")
    sys.exit(1)

# Default configuration
DEFAULT_AGENT_NAME = "Aria"
DEFAULT_POLL_INTERVAL = 2


async def start_agent(contact: str, agent_name: str, poll: int, headless: bool = True):
    """
    Initialise the agent engine and start the operational loop.
    
    Args:
        contact: The name of the WhatsApp contact to chat with.
        agent_name: The display name of the AI persona.
        poll: Polling interval in seconds.
        headless: Whether to run the browser in headless mode.
    """
    persona = Persona(override={
        "name": agent_name,
        "role": "a friendly and professional AI assistant",
        "traits": ["helpful", "intelligent", "proactive", "natural"],
        "style": "professional yet warm text messaging style",
    })

    print(f"\n{'='*60}")
    print(f"  MULTI-LAYERED AGENTIC SYSTEM")
    print(f"{'='*60}")
    print(f"  Agent Name : {agent_name}")
    print(f"  Contact    : {contact}")
    print(f"  Polling    : {poll}s interval")
    print(f"  Headless   : {'Yes' if headless else 'No'}")
    print(f"  Architecture:")
    print(f"    Layer 1  : Perception + Execution")
    print(f"    Layer 2  : Strategist + Tactician")
    print(f"    Layer 3  : Reflection + GoalManager")
    print(f"    Backbone : Blackboard (shared working memory)")
    print(f"{'='*60}\n")

    engine = WebAgentEngine(user_data_dir="./chrome_profile", headless=headless)
    
    try:
        await engine.run_agent_loop(
            AgentOrchestrator,
            goal=f"Engage in a natural, helpful conversation with '{contact}'.",
            contact_name=contact,
            persona=persona,
            poll_interval=poll,
        )
    except Exception as e:
        logger.error(f"Engine crashed: {e}")
        raise


def main():
    """CLI Entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Layered Autonomous WhatsApp Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Architecture:\n"
            "  The agent runs a 3-layer cognitive architecture:\n"
            "    1. Perception  → classifies environment signals\n"
            "    2. Cognition   → strategist + tactician with tool loop\n"
            "    3. Metacognition → self-reflection + goal tracking\n"
        ),
    )
    parser.add_argument("--contact", help="WhatsApp contact name (interactive prompt if omitted)")
    parser.add_argument("--name", default=DEFAULT_AGENT_NAME, help="AI Agent display name")
    parser.add_argument("--poll", type=int, default=DEFAULT_POLL_INTERVAL, help="Polling interval (seconds)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--headful", action="store_true", help="Run browser with visible UI (for debugging)")
    parser.add_argument(
        "--debug-layers",
        action="store_true",
        help="Enable per-layer debug logging (Perception, Strategy, Tactic, Execution, Reflection)",
    )
    
    args = parser.parse_args()

    if args.verbose or args.debug_layers:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.debug_layers:
        # Enable debug for all agent layers specifically
        for layer in ["agents.perception", "agents.strategist", "agents.tactician",
                       "agents.executor", "agents.reflection", "agents.blackboard",
                       "agents.orchestrator", "agents.goal_manager"]:
            logging.getLogger(layer).setLevel(logging.DEBUG)

    contact_name = args.contact
    if not contact_name:
        try:
            contact_name = input("Target Contact Name: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

    if not contact_name:
        logger.error("No target contact specified. Aborting.")
        sys.exit(1)

    try:
        asyncio.run(start_agent(contact_name, args.name, args.poll, headless=not args.headful))
    except KeyboardInterrupt:
        logger.info("Termination signal received. Shutting down gracefully.")
    except Exception as e:
        logger.critical(f"Fatal system error: {e}")
        sys.exit(1)
    finally:
        print("\n[System] Operational loop terminated.")


if __name__ == "__main__":
    main()
