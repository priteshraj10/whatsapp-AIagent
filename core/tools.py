"""
core/tools.py -- Agent tool registry and execution engine.

Provides a pluggable architecture for agent tools:
    - web_search: Real-time web information via DuckDuckGo
    - recall_memory: Search past conversation history
    - get_profile: Access contact biographical data
    - get_calendar: View today's/upcoming calendar events
    - create_event: Schedule a new calendar event
    - check_availability: Check free/busy status for a date
"""

import asyncio
import json
import logging
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Individual Tools                                                     #
# ------------------------------------------------------------------ #

import traceback
from duckduckgo_search import DDGS

async def web_search(query: str, max_results: int = 3) -> str:
    """Search the internet for real-time information, news, and facts using DuckDuckGo."""
    if not query.strip():
        return "No query provided."

    try:
        loop = asyncio.get_event_loop()
        
        # Run search in thread pool to avoid blocking
        def _search():
            try:
                # 1. First, check if DuckDuckGo has an instant answer (like weather, math, simple facts)
                with DDGS() as ddgs:
                    answers = list(ddgs.answers(query))
                    if answers and answers[0].get("text"):
                        return f"Instant Answer for '{query}': {answers[0]['text']}"
                
                # 2. If no instant answer, do a text search
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=max_results))
                    if not results:
                        return f"No results found on the internet for '{query}'."
                        
                    formatted = []
                    for r in results:
                        formatted.append(f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}")
                        
                    return f"Search Results for '{query}':\n\n" + "\n\n".join(formatted)
            except Exception as e:
                logger.error(f"DDGS error: {traceback.format_exc()}")
                return f"Search error: {str(e)}"

        return await loop.run_in_executor(None, _search)
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"Error executing search: {str(e)}"


def recall_memory(db: Any, contact_id: int, query: str) -> str:
    """Search conversation history for relevant past messages."""
    try:
        results = db.search_relevant(contact_id, query, limit=5)
        if not results:
            return "No relevant past messages found."

        lines = [f"[{m['role'].upper()}]: {m['content']}" for m in results]
        return "Relevant history:\n" + "\n".join(lines)
    except Exception as e:
        logger.error(f"Memory recall failed: {e}")
        return "Failed to retrieve history."


def get_profile(profiler: Any) -> str:
    """Retrieve current contact profile facts."""
    try:
        snippet = profiler.get_context_snippet()
        return snippet if snippet else "No facts known about this contact yet."
    except Exception as e:
        logger.error(f"Profile retrieval failed: {e}")
        return "Failed to retrieve profile."


async def get_calendar(calendar_service: Any, days: int = 1) -> str:
    """Get calendar events for today or upcoming days."""
    try:
        if days <= 1:
            events = await calendar_service.get_today_events()
            label = "today"
        else:
            events = await calendar_service.get_upcoming_events(days=days)
            label = f"next {days} days"

        if not events:
            return f"No events {label}. Schedule is clear."

        formatted = calendar_service.format_for_llm(events)
        source = "Google Calendar" if calendar_service.is_google_connected else "Local calendar"
        return f"Events {label} ({source}):\n{formatted}"
    except Exception as e:
        logger.error(f"Calendar fetch failed: {e}")
        return f"Calendar unavailable: {e}"


async def create_event(
    calendar_service: Any,
    summary: str,
    date_str: str,
    time_str: str = "10:00",
    duration_hours: float = 1.0,
    description: str = "",
    location: str = "",
) -> str:
    """Create a calendar event from conversation context."""
    try:
        # Parse date and time
        dt_str = f"{date_str} {time_str}"
        for fmt in ["%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M", "%B %d %H:%M", "%b %d %H:%M"]:
            try:
                start = datetime.strptime(dt_str, fmt)
                # If no year, use current year
                if start.year == 1900:
                    start = start.replace(year=datetime.now().year)
                break
            except ValueError:
                continue
        else:
            return f"Could not parse date/time: '{dt_str}'. Use YYYY-MM-DD HH:MM format."

        end = start + timedelta(hours=duration_hours)

        event = await calendar_service.create_event(
            summary=summary,
            start_time=start,
            end_time=end,
            description=description,
            location=location,
        )

        if event:
            return f"Event created: '{summary}' on {start.strftime('%B %d at %I:%M %p')}"
        return "Failed to create event."
    except Exception as e:
        logger.error(f"Event creation failed: {e}")
        return f"Event creation failed: {e}"


async def update_event(
    calendar_service: Any,
    event_id: str,
    summary: str = "",
    date_str: str = "",
    time_str: str = "10:00",
    duration_hours: float = 1.0,
    description: str = "",
    location: str = "",
) -> str:
    """Update an existing calendar event."""
    try:
        start = None
        end = None
        
        if date_str:
            dt_str = f"{date_str} {time_str}"
            for fmt in ["%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M", "%B %d %H:%M", "%b %d %H:%M"]:
                try:
                    start = datetime.strptime(dt_str, fmt)
                    if start.year == 1900:
                        start = start.replace(year=datetime.now().year)
                    break
                except ValueError:
                    continue
            else:
                return f"Could not parse date/time: '{dt_str}'. Use YYYY-MM-DD HH:MM format."
            
            end = start + timedelta(hours=duration_hours)

        success = await calendar_service.update_event(
            event_id=event_id,
            summary=summary if summary else None,
            start_time=start,
            end_time=end,
            description=description if description else None,
            location=location if location else None,
        )

        if success:
            return f"Event updated successfully."
        return "Failed to update event. Event ID may be invalid."
    except Exception as e:
        logger.error(f"Event update failed: {e}")
        return f"Event update failed: {e}"


async def delete_event(calendar_service: Any, event_id: str) -> str:
    """Delete a calendar event."""
    try:
        success = await calendar_service.delete_event(event_id)
        if success:
            return f"Event deleted successfully."
        return "Failed to delete event. Event ID may be invalid."
    except Exception as e:
        logger.error(f"Event deletion failed: {e}")
        return f"Event deletion failed: {e}"


async def check_availability(calendar_service: Any, date_str: str) -> str:
    """Check schedule availability for a given date."""
    try:
        for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%B %d", "%b %d"]:
            try:
                date = datetime.strptime(date_str, fmt)
                if date.year == 1900:
                    date = date.replace(year=datetime.now().year)
                break
            except ValueError:
                continue
        else:
            return f"Could not parse date: '{date_str}'. Use YYYY-MM-DD format."

        return await calendar_service.check_availability(date)
    except Exception as e:
        logger.error(f"Availability check failed: {e}")
        return f"Availability check failed: {e}"

async def query_knowledge_graph(db: Any, entity: str) -> str:
    """Query the knowledge graph for a specific entity."""
    try:
        triples = db.query_knowledge_graph(entity)
        if not triples:
            return f"No known relationships found for '{entity}'."
        
        result = f"Knowledge Graph connections for '{entity}':\n"
        for t in triples:
            result += f"- {t['subject']} -> {t['predicate']} -> {t['object']}\n"
        return result
    except Exception as e:
        logger.error(f"Knowledge graph query failed: {e}")
        return f"Knowledge graph query failed: {e}"

# ------------------------------------------------------------------ #
# Tool Registry & Executor                                             #
# ------------------------------------------------------------------ #

TOOL_DESCRIPTIONS = """
AVAILABLE TOOLS (invoke by returning "use_tool" in JSON):

  - web_search(query): Look up real-time facts, news, or definitions.
    Example: {"use_tool": "web_search", "tool_args": {"query": "weather in hyderabad"}}

  - recall_memory(query): Search past conversations for specific details.
    Example: {"use_tool": "recall_memory", "tool_args": {"query": "birthday plans"}}

  - query_knowledge_graph(entity): Look up known relationships and facts about a specific person, place, or thing.
    Example: {"use_tool": "query_knowledge_graph", "tool_args": {"entity": "Rahul"}}

  - get_profile(): View known facts about this contact.
    Example: {"use_tool": "get_profile", "tool_args": {}}

  - get_calendar(days): View today's or upcoming calendar events.
    Example: {"use_tool": "get_calendar", "tool_args": {"days": 1}}

  - create_event(summary, date_str, time_str, duration_hours, description, location):
    Schedule a new event.
    Example: {"use_tool": "create_event", "tool_args": {"summary": "Coffee with Devika", "date_str": "2026-04-26", "time_str": "15:00"}}

  - update_event(event_id, summary, date_str, time_str, duration_hours, description, location):
    Update an existing event. Only provide fields you want to change.
    Example: {"use_tool": "update_event", "tool_args": {"event_id": "12345", "time_str": "16:00"}}

  - delete_event(event_id):
    Cancel or delete an event.
    Example: {"use_tool": "delete_event", "tool_args": {"event_id": "12345"}}

  - check_availability(date_str): Check if a date is free.
    Example: {"use_tool": "check_availability", "tool_args": {"date_str": "2026-04-26"}}
"""


class ToolExecutor:
    """
    Registry and execution handler for agent-facing tools.

    Manages both stateless tools (web search) and stateful tools
    (calendar, memory) with clean dependency injection.
    """

    def __init__(self, db: Any, profiler: Any, contact_id: int, calendar_service: Any = None):
        self.db = db
        self.profiler = profiler
        self.contact_id = contact_id
        self.calendar = calendar_service

        # Build registry
        self._registry: Dict[str, Callable] = {
            "web_search": lambda args: web_search(args.get("query", "")),
            "recall_memory": lambda args: recall_memory(self.db, self.contact_id, args.get("query", "")),
            "get_profile": lambda _: get_profile(self.profiler),
            "query_knowledge_graph": lambda args: query_knowledge_graph(self.db, args.get("entity", "")),
        }

        # Calendar tools (only if service is available)
        if self.calendar:
            self._registry.update({
                "get_calendar": lambda args: get_calendar(self.calendar, int(args.get("days", 1))),
                "create_event": lambda args: create_event(
                    self.calendar,
                    args.get("summary", ""),
                    args.get("date_str", ""),
                    args.get("time_str", "10:00"),
                    float(args.get("duration_hours", 1.0)),
                    args.get("description", ""),
                    args.get("location", ""),
                ),
                "update_event": lambda args: update_event(
                    self.calendar,
                    args.get("event_id", ""),
                    args.get("summary", ""),
                    args.get("date_str", ""),
                    args.get("time_str", "10:00"),
                    float(args.get("duration_hours", 1.0)),
                    args.get("description", ""),
                    args.get("location", ""),
                ),
                "delete_event": lambda args: delete_event(
                    self.calendar, args.get("event_id", "")
                ),
                "check_availability": lambda args: check_availability(
                    self.calendar, args.get("date_str", "")
                ),
            })

    def get_available_tools(self) -> str:
        """Return description of available tools for LLM context."""
        available = list(self._registry.keys())
        # Filter TOOL_DESCRIPTIONS to only show available tools
        return TOOL_DESCRIPTIONS

    async def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool by name."""
        if tool_name not in self._registry:
            available = ", ".join(self._registry.keys())
            logger.warning(f"Unknown tool requested: {tool_name}")
            return f"Error: Tool '{tool_name}' not found. Available: {available}"

        try:
            logger.info(f"Executing tool: {tool_name} | args: {args}")
            handler = self._registry[tool_name]
            result = handler(args)

            if asyncio.iscoroutine(result):
                return await result
            return str(result)

        except Exception as e:
            logger.error(f"Tool execution failed ({tool_name}): {e}")
            return f"Error: {e}"
