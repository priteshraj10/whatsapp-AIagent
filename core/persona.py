"""
whatsapp/persona.py -- Senior-grade agent personality and prompt management.

Encapsulates the agent's identity, communication style, and behavioral rules.
Loads configuration from YAML and generates dynamic system prompts for the LLM.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from core.tools import TOOL_DESCRIPTIONS

# Configure logging
logger = logging.getLogger(__name__)

# Constants
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSONA_PATH = os.path.join(ROOT_DIR, "persona", "persona.yaml")
SKILLS_PATH = os.path.join(ROOT_DIR, "persona", "skills.md")


class PersonaError(Exception):
    """Exception raised for persona loading or prompt generation errors."""
    pass


class Persona:
    """
    Manages the AI agent's personality and prompt construction.
    
    Responsible for loading traits, styles, and skills from configuration files
    and assembling the master system prompt for LLM decision-making.
    """

    def __init__(self, override: Optional[Dict[str, Any]] = None):
        """
        Initialise persona from YAML config with optional runtime overrides.
        
        Args:
            override: Dictionary of values to override the YAML configuration.
        """
        config = self._load_config() or {}
        if override:
            config.update(override)

        # Core Identity
        self.name = config.get("name", "Aria")
        self.role = config.get("role", "a friendly and intelligent AI assistant")
        self.traits = config.get("traits", ["curious", "helpful", "empathetic", "honest"])
        self.style = config.get("style", "concise and natural like texting a smart friend")
        self.boundaries = config.get("boundaries", ["Do not pretend to be human if asked."])

        # Proactive Behavior Settings
        proactive_cfg = config.get("proactive", {})
        self.proactive_enabled = proactive_cfg.get("enabled", True)
        self.silence_threshold = proactive_cfg.get("silence_threshold_minutes", 10) * 60
        self.max_proactive_per_day = proactive_cfg.get("max_proactive_per_day", 5)

        # Content Assets
        self.skills_text = self._load_skills()
        self._cached_base_prompt: Optional[str] = None

    def build_system_prompt(self, profile_snippet: str = "", include_tools: bool = True) -> str:
        """
        Construct a comprehensive system prompt for the LLM.
        
        Args:
            profile_snippet: Contextual information about the current contact.
            include_tools: Whether to include tool descriptions in the prompt.
            
        Returns:
            A formatted system prompt string.
        """
        if self._cached_base_prompt is None:
            self._cached_base_prompt = self._assemble_base_identity()

        sections = [self._cached_base_prompt]

        if include_tools:
            sections.append(TOOL_DESCRIPTIONS)

        sections.append(self._get_interaction_rules())

        full_prompt = "\n\n".join(sections)

        if profile_snippet:
            full_prompt += f"\n\n## WHAT YOU KNOW ABOUT THIS PERSON\n{profile_snippet}"

        return full_prompt

    def _assemble_base_identity(self) -> str:
        """Assemble the static part of the agent's identity."""
        traits_str = ", ".join(self.traits) if isinstance(self.traits, list) else self.traits
        boundaries_list = self.boundaries if isinstance(self.boundaries, list) else [self.boundaries]
        boundaries_str = "\n".join(f"- {b}" for b in boundaries_list)
        
        skills_section = f"\n\n## YOUR SKILLS & KNOWLEDGE\n{self.skills_text}" if self.skills_text else ""
        
        return (
            f"You are {self.name}, {self.role}.\n\n"
            f"PERSONALITY TRAITS: {traits_str}\n"
            f"COMMUNICATION STYLE: {self.style}\n\n"
            f"BOUNDARIES:\n{boundaries_str}"
            f"{skills_section}"
        )

    @staticmethod
    def _get_interaction_rules() -> str:
        """Return the core rules for LLM output formatting and behavior."""
        return """
─────────────────────────────────────
FINAL REPLY FORMAT (when not using a tool)
─────────────────────────────────────
Return ONLY raw JSON -- no markdown, no code fences:
{
  "intent":           "<one-line summary of what is happening>",
  "emotion_detected": "<neutral|happy|sad|angry|confused|excited>",
  "should_reply":     true or false,
  "action_type":      "<reply|react|delete>",
  "reply":            "<message to send if action_type=reply>",
  "target_text":      "<exact text of the message to react to or delete (if action_type=react/delete)>",
  "reaction_emoji":   "<emoji to use if action_type=react (e.g. 👍, ❤️, 😂)>",
  "confidence":       <0.0-1.0>,
  "reasoning":        "<1 sentence why>"
}

RULES:
- Always try to keep the conversation going with real people.
- Ask follow-up questions or share related thoughts to maintain the flow.
- Match the user's language, tone, and energy. If they speak Hinglish, reply in native-sounding Hinglish (e.g., "Haan yaar, bilkul theek kaha tumne!").
- No emojis unless the user uses them first (or if you are using action_type=react).
- If you want to react to a specific message instead of replying, set action_type="react", target_text="exact text", reaction_emoji="😂".
- If you want to delete a previous message you sent, set action_type="delete", target_text="exact text of your message".
- Max 2 sentences unless detail is explicitly requested.
- should_reply=false for automated messages, passcodes, spam, or if the user explicitly ends the chat.
- Proactive trigger: pick an interesting topic from past context if there is silence.
"""

    @staticmethod
    def _load_config() -> Optional[Dict[str, Any]]:
        """Load YAML configuration file."""
        if not os.path.exists(PERSONA_PATH):
            logger.warning(f"Persona config not found at {PERSONA_PATH}. Using defaults.")
            return None
            
        try:
            import yaml
            with open(PERSONA_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load persona YAML: {e}")
            return None

    @staticmethod
    def _load_skills() -> str:
        """Load markdown skills file."""
        if not os.path.exists(SKILLS_PATH):
            return ""
            
        try:
            with open(SKILLS_PATH, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to load skills markdown: {e}")
            return ""
