"""
services/google_calendar.py -- Google Calendar integration for the agent.

Provides the agent with real-time calendar awareness:
    - View today's/upcoming events
    - Create events from conversation context
    - Check availability for scheduling
    - Set reminders

Uses Google Calendar API v3 with OAuth2 service account or user credentials.
Falls back to a local calendar store if Google API is unavailable.

Setup:
    1. Enable Google Calendar API in Google Cloud Console
    2. Create OAuth2 credentials (Desktop app) or a Service Account
    3. Download credentials.json to the project root
    4. On first run, a browser window opens for OAuth consent
    5. Token is cached in data/calendar_token.json
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Calendar data directory
DATA_DIR = Path(__file__).parent.parent / "data"
TOKEN_PATH = DATA_DIR / "calendar_token.json"
CREDENTIALS_PATH = Path(__file__).parent.parent / "credentials.json"
LOCAL_EVENTS_PATH = DATA_DIR / "local_calendar.json"


@dataclass
class CalendarEvent:
    """A single calendar event."""
    summary: str
    start: str              # ISO format datetime
    end: str                # ISO format datetime
    description: str = ""
    location: str = ""
    event_id: str = ""
    source: str = "google"  # "google" or "local"

    def to_display(self) -> str:
        """Human-readable format for LLM context."""
        try:
            start_dt = datetime.fromisoformat(self.start.replace("Z", "+00:00"))
            time_str = start_dt.strftime("%I:%M %p")
            date_str = start_dt.strftime("%b %d")
        except Exception:
            time_str = self.start
            date_str = ""

        parts = [f"[{self.event_id}] {time_str} - {self.summary}"]
        if date_str:
            parts[0] = f"[{self.event_id}] [{date_str}] {time_str} - {self.summary}"
        if self.location:
            parts.append(f"  Location: {self.location}")
        if self.description:
            parts.append(f"  Note: {self.description[:100]}")
        return "\n".join(parts)


class GoogleCalendarService:
    """
    Google Calendar integration with graceful fallback.

    Attempts to use the Google Calendar API. If credentials are missing
    or authentication fails, falls back to a local JSON-based calendar.
    """

    def __init__(self):
        self._google_available = False
        self._service = None
        self._init_google_api()

    def _init_google_api(self):
        """Attempt to initialize the Google Calendar API client."""
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build

            SCOPES = ["https://www.googleapis.com/auth/calendar"]
            creds = None

            # Load existing token
            if TOKEN_PATH.exists():
                creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

            # Refresh or get new credentials
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                elif CREDENTIALS_PATH.exists():
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(CREDENTIALS_PATH), SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                else:
                    logger.info(
                        "Google Calendar: No credentials.json found. "
                        "Using local calendar. To enable Google Calendar, "
                        "place credentials.json in the project root."
                    )
                    return

                # Save token for future runs
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                with open(TOKEN_PATH, "w") as f:
                    f.write(creds.to_json())

            self._service = build("calendar", "v3", credentials=creds)
            self._google_available = True
            logger.info("Google Calendar API connected successfully.")

        except ImportError:
            logger.info(
                "Google Calendar: google-api-python-client not installed. "
                "Using local calendar. Install with: "
                "pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )
        except Exception as e:
            logger.warning(f"Google Calendar API init failed: {e}. Using local calendar.")

    @property
    def is_google_connected(self) -> bool:
        return self._google_available

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def get_today_events(self) -> List[CalendarEvent]:
        """Get all events for today."""
        now = datetime.now()
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        return await self.get_events(start_of_day, end_of_day)

    async def get_upcoming_events(self, days: int = 3, max_results: int = 10) -> List[CalendarEvent]:
        """Get upcoming events for the next N days."""
        now = datetime.now()
        end = now + timedelta(days=days)
        return await self.get_events(now, end, max_results)

    async def get_events(
        self, start: datetime, end: datetime, max_results: int = 20
    ) -> List[CalendarEvent]:
        """Fetch events within a time range."""
        if self._google_available:
            return await self._google_get_events(start, end, max_results)
        return self._local_get_events(start, end)

    async def create_event(
        self,
        summary: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        description: str = "",
        location: str = "",
        timezone: str = "Asia/Kolkata",
    ) -> Optional[CalendarEvent]:
        """Create a new calendar event."""
        if not end_time:
            end_time = start_time + timedelta(hours=1)

        if self._google_available:
            return await self._google_create_event(
                summary, start_time, end_time, description, location, timezone
            )
        return self._local_create_event(
            summary, start_time, end_time, description, location, timezone
        )

    async def update_event(
        self,
        event_id: str,
        summary: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        timezone: str = "Asia/Kolkata",
    ) -> bool:
        """Update an existing calendar event."""
        if self._google_available:
            return await self._google_update_event(
                event_id, summary, start_time, end_time, description, location, timezone
            )
        return self._local_update_event(
            event_id, summary, start_time, end_time, description, location, timezone
        )

    async def delete_event(self, event_id: str) -> bool:
        """Delete an existing calendar event."""
        if self._google_available:
            return await self._google_delete_event(event_id)
        return self._local_delete_event(event_id)

    async def check_availability(self, date: datetime) -> str:
        """Check free/busy status for a given date."""
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        events = await self.get_events(start, end)

        if not events:
            return f"No events on {date.strftime('%B %d')}. Fully available."

        busy_times = "\n".join(f"  - {e.to_display()}" for e in events)
        return f"Events on {date.strftime('%B %d')}:\n{busy_times}"

    def format_for_llm(self, events: List[CalendarEvent]) -> str:
        """Format events for injection into LLM context."""
        if not events:
            return "No upcoming events."

        lines = [e.to_display() for e in events]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Google Calendar API                                                  #
    # ------------------------------------------------------------------ #

    async def _google_get_events(
        self, start: datetime, end: datetime, max_results: int
    ) -> List[CalendarEvent]:
        """Fetch events from Google Calendar API."""
        loop = asyncio.get_event_loop()

        def _fetch():
            try:
                time_min = start.isoformat() + "Z" if not start.tzinfo else start.isoformat()
                time_max = end.isoformat() + "Z" if not end.tzinfo else end.isoformat()

                result = (
                    self._service.events()
                    .list(
                        calendarId="primary",
                        timeMin=time_min,
                        timeMax=time_max,
                        maxResults=max_results,
                        singleEvents=True,
                        orderBy="startTime",
                    )
                    .execute()
                )
                return result.get("items", [])
            except Exception as e:
                logger.error(f"Google Calendar fetch failed: {e}")
                return []

        items = await loop.run_in_executor(None, _fetch)

        events = []
        for item in items:
            start_str = item.get("start", {}).get("dateTime") or item.get("start", {}).get("date", "")
            end_str = item.get("end", {}).get("dateTime") or item.get("end", {}).get("date", "")
            events.append(
                CalendarEvent(
                    summary=item.get("summary", "Untitled"),
                    start=start_str,
                    end=end_str,
                    description=item.get("description", ""),
                    location=item.get("location", ""),
                    event_id=item.get("id", ""),
                    source="google",
                )
            )
        return events

    async def _google_create_event(
        self,
        summary: str,
        start_time: datetime,
        end_time: datetime,
        description: str,
        location: str,
        timezone: str,
    ) -> Optional[CalendarEvent]:
        """Create an event via Google Calendar API."""
        loop = asyncio.get_event_loop()

        def _create():
            body = {
                "summary": summary,
                "location": location,
                "description": description,
                "start": {"dateTime": start_time.isoformat(), "timeZone": timezone},
                "end": {"dateTime": end_time.isoformat(), "timeZone": timezone},
                "reminders": {"useDefault": True},
            }
            try:
                result = (
                    self._service.events()
                    .insert(calendarId="primary", body=body)
                    .execute()
                )
                return result
            except Exception as e:
                logger.error(f"Google Calendar create failed: {e}")
                return None

        result = await loop.run_in_executor(None, _create)
        if result:
            logger.info(f"Created Google Calendar event: {summary}")
            return CalendarEvent(
                summary=summary,
                start=start_time.isoformat(),
                end=end_time.isoformat(),
                description=description,
                location=location,
                event_id=result.get("id", ""),
                source="google",
            )
        return None

    async def _google_update_event(
        self,
        event_id: str,
        summary: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        description: Optional[str],
        location: Optional[str],
        timezone: str,
    ) -> bool:
        """Update an event via Google Calendar API."""
        loop = asyncio.get_event_loop()

        def _update():
            try:
                # First fetch the existing event
                event = self._service.events().get(calendarId="primary", eventId=event_id).execute()
                
                if summary is not None:
                    event["summary"] = summary
                if location is not None:
                    event["location"] = location
                if description is not None:
                    event["description"] = description
                if start_time is not None:
                    event["start"] = {"dateTime": start_time.isoformat(), "timeZone": timezone}
                if end_time is not None:
                    event["end"] = {"dateTime": end_time.isoformat(), "timeZone": timezone}
                
                self._service.events().update(calendarId="primary", eventId=event_id, body=event).execute()
                return True
            except Exception as e:
                logger.error(f"Google Calendar update failed: {e}")
                return False

        return await loop.run_in_executor(None, _update)

    async def _google_delete_event(self, event_id: str) -> bool:
        """Delete an event via Google Calendar API."""
        loop = asyncio.get_event_loop()

        def _delete():
            try:
                self._service.events().delete(calendarId="primary", eventId=event_id).execute()
                return True
            except Exception as e:
                logger.error(f"Google Calendar delete failed: {e}")
                return False

        return await loop.run_in_executor(None, _delete)

    # ------------------------------------------------------------------ #
    # Local Calendar Fallback                                              #
    # ------------------------------------------------------------------ #

    def _local_get_events(self, start: datetime, end: datetime) -> List[CalendarEvent]:
        """Get events from local JSON store."""
        events = self._load_local_events()
        filtered = []
        for e in events:
            try:
                event_start = datetime.fromisoformat(e["start"])
                if start <= event_start <= end:
                    filtered.append(CalendarEvent(**e))
            except Exception:
                continue
        return filtered

    def _local_create_event(
        self,
        summary: str,
        start_time: datetime,
        end_time: datetime,
        description: str,
        location: str,
        timezone: str,
    ) -> CalendarEvent:
        """Create an event in the local JSON store."""
        events = self._load_local_events()
        event_data = {
            "summary": summary,
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "description": description,
            "location": location,
            "event_id": f"local_{int(time.time())}",
            "source": "local",
        }
        events.append(event_data)
        self._save_local_events(events)
        logger.info(f"Created local calendar event: {summary}")
        return CalendarEvent(**event_data)

    def _local_update_event(
        self,
        event_id: str,
        summary: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        description: Optional[str],
        location: Optional[str],
        timezone: str,
    ) -> bool:
        """Update an event in the local JSON store."""
        events = self._load_local_events()
        updated = False
        for e in events:
            if e.get("event_id") == event_id:
                if summary is not None:
                    e["summary"] = summary
                if start_time is not None:
                    e["start"] = start_time.isoformat()
                if end_time is not None:
                    e["end"] = end_time.isoformat()
                if description is not None:
                    e["description"] = description
                if location is not None:
                    e["location"] = location
                updated = True
                break
        
        if updated:
            self._save_local_events(events)
        return updated

    def _local_delete_event(self, event_id: str) -> bool:
        """Delete an event from the local JSON store."""
        events = self._load_local_events()
        original_length = len(events)
        events = [e for e in events if e.get("event_id") != event_id]
        
        if len(events) < original_length:
            self._save_local_events(events)
            return True
        return False

    def _load_local_events(self) -> List[Dict]:
        """Load events from local JSON file."""
        if not LOCAL_EVENTS_PATH.exists():
            return []
        try:
            with open(LOCAL_EVENTS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_local_events(self, events: List[Dict]):
        """Save events to local JSON file."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOCAL_EVENTS_PATH, "w") as f:
            json.dump(events, f, indent=2, default=str)
