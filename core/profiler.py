"""
core/profiler.py -- Professional per-contact profile manager.

Maintains detailed JSON profiles containing facts, interests, and interaction
history extracted from conversations to personalize the agent's behavior.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_DIR = os.path.join(PROJECT_ROOT, "profiles")

DEFAULT_PROFILE: Dict[str, Any] = {
    "name": "",
    "college": "",
    "city": "",
    "interests": [],
    "topics_discussed": [],
    "personality_notes": "",
    "language_preference": "english",
    "last_mood": "neutral",
    "relationship_category": "unknown",
    "conversation_count": 0,
    "notes": [],
}


class ProfileError(Exception):
    """Exception raised for profile management errors."""
    pass


class ContactProfiler:
    """
    Manages persistent JSON profiles for individual contacts.
    
    Extracts and stores attributes using both immediate updates (after each message)
    and deep updates (using LLM-assisted fact extraction).
    """

    def __init__(self, contact_name: str, db: Optional[Any] = None, contact_id: Optional[int] = None):
        """
        Initialise profiler for a specific contact.
        
        Args:
            contact_name: The display name of the WhatsApp contact.
            db: Optional database connection to sync state.
            contact_id: Optional contact ID for DB sync.
        """
        self.contact_name = contact_name
        self.db = db
        self.contact_id = contact_id
        self._ensure_dir()
        self._path = os.path.join(PROFILES_DIR, f"{self._get_safe_filename(contact_name)}.json")
        self.profile = self._load_profile()

    def _ensure_dir(self):
        """Ensure the profiles directory exists."""
        try:
            os.makedirs(PROFILES_DIR, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create profiles directory: {e}")
            raise ProfileError(f"Directory creation failed: {e}")

    def get_context_snippet(self) -> str:
        """
        Generate a compact text representation of the profile for the LLM system prompt.
        
        Returns:
            A formatted string of known facts about the contact.
        """
        p = self.profile
        lines = []
        
        # Mapping of profile keys to display labels
        fields = [
            ("relationship_category", "Relationship"),
            ("college", "Education/Work"),
            ("city", "Location"),
            ("personality_notes", "Personality"),
            ("language_preference", "Communication Style"),
            ("last_mood", "Last Mood")
        ]
        
        for key, label in fields:
            val = p.get(key)
            if val and val not in ("", "neutral", "english"):
                lines.append(f"{label}: {val}")
        
        if p.get("interests"):
            lines.append(f"Interests: {', '.join(p['interests'][:5])}")
            
        if p.get("topics_discussed"):
            lines.append(f"Recent Topics: {', '.join(p['topics_discussed'][-5:])}")
            
        return "\n".join(lines)

    def update_from_exchange(self, emotion: str):
        """
        Perform a lightweight update after a message exchange.
        
        Args:
            emotion: The emotion detected in the latest message.
        """
        self.profile["last_mood"] = emotion or "neutral"
        self.profile["conversation_count"] = self.profile.get("conversation_count", 0) + 1
        self._save_profile()

    async def update_with_llm(self, llm_client: Any, recent_messages: List[Dict[str, str]]):
        """
        Perform a deep LLM-assisted fact extraction from recent conversation history.
        
        This is typically triggered every N exchanges to build a detailed contact profile.
        
        Args:
            llm_client: An initialised LLM client.
            recent_messages: A list of recent message objects.
        """
        count = self.profile.get("conversation_count", 0)
        # Only run every 10 exchanges to save tokens/cost
        if count == 0 or count % 10 != 0:
            return

        convo_slice = "\n".join(
            f"[{m['role'].upper()}]: {m['content']}"
            for m in recent_messages[-20:]
        )

        prompt = [
            {
                "role": "system",
                "content": (
                    "Extract biographical facts and interests about the USER (the person I'm chatting with). "
                    "Focus on: college/work, location, interests, personality traits, language preferences, and relationship_category. "
                    "For relationship_category, strictly choose ONE of: 'Close Friend', 'Family', 'Colleague', 'Acquaintance', 'Unknown'. "
                    "Return ONLY valid raw JSON without code blocks:\n"
                    '{"college":"","city":"","interests":[],"topics_discussed":[],'
                    '"personality_notes":"","language_preference":"","relationship_category":""}'
                ),
            },
            {"role": "user", "content": f"Conversation snippet:\n{convo_slice}"},
        ]

        try:
            raw_response = await llm_client.chat(prompt)
            # Remove possible markdown wrapper
            clean_json = re.sub(r"```(?:json)?\n?|\n?```", "", raw_response).strip()
            extracted_facts = json.loads(clean_json)

            self._merge_facts(extracted_facts)
            self._save_profile()
            logger.info(f"Deep profile update completed for {self.contact_name}")
            
        except Exception as e:
            logger.warning(f"Deep profile update failed for {self.contact_name}: {e}")

    def _merge_facts(self, facts: Dict[str, Any]):
        """Intelligently merge extracted facts into the existing profile."""
        for key, val in facts.items():
            if not val or key not in DEFAULT_PROFILE:
                continue
                
            if isinstance(val, list):
                # Merge lists, remove duplicates, keep order (newest last), cap at 10
                existing = self.profile.get(key, [])
                # Use dict.fromkeys to preserve order while deduping
                merged = list(dict.fromkeys(existing + val))[-10:]
                self.profile[key] = merged
            else:
                # Update scalar values
                self.profile[key] = val
                
                # Sync relationship category to database if available
                if key == "relationship_category" and self.db and self.contact_id is not None:
                    try:
                        self.db.set_state(self.contact_id, "relationship_category", val)
                        logger.info(f"Synced relationship_category '{val}' to database for {self.contact_name}")
                    except Exception as e:
                        logger.error(f"Failed to sync relationship category to DB: {e}")

    def _load_profile(self) -> Dict[str, Any]:
        """Load profile from disk or return default."""
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Merge with default to handle schema additions
                    return {**DEFAULT_PROFILE, **data}
            except Exception as e:
                logger.error(f"Failed to load profile for {self.contact_name}: {e}")
        
        return {**DEFAULT_PROFILE, "name": self.contact_name}

    def _save_profile(self):
        """Persist profile to disk."""
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self.profile, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save profile for {self.contact_name}: {e}")

    @staticmethod
    def _get_safe_filename(name: str) -> str:
        """Sanitize contact name for use as a filename."""
        safe = re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_")
        return safe[:60]
