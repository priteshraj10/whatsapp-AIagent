"""
core/llm_client.py -- Professional async LLM client with fallback and retry logic.

Built to Senior/CTO standards:
- Robust error handling with custom exceptions.
- Comprehensive logging.
- Type safety.
- Efficient resource management (reused clients).
"""

import asyncio
import logging
import os
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMRateLimitError(LLMError):
    """Exception raised when an LLM provider returns a rate limit error."""
    pass


class LLMClient:
    """
    Stateless async LLM client with provider-specific backends and failover.
    
    Attributes:
        GROQ_MODEL (str): Default model for Groq provider.
        GEMINI_MODEL (str): Default model for Gemini provider.
        GROQ_BASE_URL (str): API endpoint for Groq.
    """

    GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    GEMINI_MODEL = "gemini-2.0-flash"
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self):
        """Initialise credentials and lazy-loaded clients."""
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self._groq_client: Any = None

    def _init_groq_client(self) -> Any:
        """Lazily initialise the OpenAI-compatible Groq client."""
        if self._groq_client is None:
            if not self.groq_key:
                logger.warning("GROQ_API_KEY not found in environment.")
                return None
            try:
                from openai import AsyncOpenAI
                self._groq_client = AsyncOpenAI(
                    api_key=self.groq_key,
                    base_url=self.GROQ_BASE_URL,
                )
            except ImportError:
                logger.error("openai package not installed. Cannot use Groq.")
                return None
        return self._groq_client

    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
        base_delay: float = 4.0,
    ) -> str:
        """
        Execute a chat completion with Groq as primary and Gemini as fallback.
        
        Args:
            messages: List of message objects with 'role' and 'content'.
            max_retries: Number of retries for rate-limiting.
            base_delay: Base delay for exponential backoff.
            
        Returns:
            The text response from the LLM.
            
        Raises:
            LLMError: If all providers fail.
        """
        last_exception: Optional[Exception] = None

        # Attempt Groq (Primary)
        if self.groq_key:
            for attempt in range(max_retries):
                try:
                    return await self._call_groq(messages)
                except LLMRateLimitError as e:
                    last_exception = e
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Groq rate-limited. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                except Exception as e:
                    last_exception = e
                    logger.error(f"Groq provider failed: {e}. Falling back to Gemini.")
                    break

        # Attempt Gemini (Fallback)
        if self.gemini_key:
            try:
                return await self._call_gemini(messages)
            except Exception as e:
                last_exception = e
                logger.error(f"Gemini provider failed: {e}")

        error_msg = f"All LLM providers failed. Last error: {last_exception}"
        logger.critical(error_msg)
        raise LLMError(error_msg) from last_exception

    async def _call_groq(self, messages: List[Dict[str, str]]) -> str:
        """Internal method to invoke the Groq API."""
        client = self._init_groq_client()
        if not client:
            raise LLMError("Groq client not initialised.")

        try:
            response = await client.chat.completions.create(
                model=self.GROQ_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=400,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate_limit" in err_msg.lower():
                raise LLMRateLimitError(err_msg) from e
            raise LLMError(err_msg) from e

    async def _call_gemini(self, messages: List[Dict[str, str]]) -> str:
        """Internal method to invoke the Gemini API using an executor for non-async SDK calls."""
        try:
            from google import genai
        except ImportError:
            raise LLMError("google-genai package not installed. Cannot use Gemini.")

        prompt = self._format_messages_for_gemini(messages)
        loop = asyncio.get_event_loop()
        client = genai.Client(api_key=self.gemini_key)

        try:
            # Use run_in_executor since some parts of the GenAI SDK might be blocking
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=self.GEMINI_MODEL, contents=prompt
                ),
            )
            if not response.text:
                raise LLMError("Gemini returned an empty response.")
            return response.text.strip()
        except Exception as e:
            raise LLMError(f"Gemini API call failed: {e}") from e

    @staticmethod
    def _format_messages_for_gemini(messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI message format to a single text prompt for Gemini."""
        role_map = {
            "system": "[SYSTEM]",
            "assistant": "[ASSISTANT]",
            "user": "[USER]"
        }
        
        lines = []
        for msg in messages:
            role = role_map.get(msg.get("role", "user"), "[USER]")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
