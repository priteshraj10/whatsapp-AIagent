"""
agents/knowledge.py -- Knowledge Graph extraction layer of the agentic system.

This agent periodically reviews recent messages to extract relational facts
and builds a long-term knowledge graph.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from agents.blackboard import Blackboard

logger = logging.getLogger(__name__)


class KnowledgeExtractionAgent:
    """
    Extracts structural relational data from conversations to populate
    the agent's internal Knowledge Graph.
    """

    # Trigger threshold
    EXCHANGE_INTERVAL = 10  # Extract every 10 cycles with activity

    def __init__(
        self,
        blackboard: Blackboard,
        llm_client: Any,
        db: Any,
        contact_id: int,
    ):
        self.bb = blackboard
        self.llm = llm_client
        self.db = db
        self.contact_id = contact_id
        self._last_extraction_cycle: int = 0

    def should_extract(self) -> bool:
        """Determine if we should run extraction this cycle."""
        cycle = self.bb.cycle_count
        
        # Avoid running too frequently
        if (cycle - self._last_extraction_cycle) < self.EXCHANGE_INTERVAL:
            return False

        # Only run if there's been recent successful engagement
        recent_results = self.bb.get_recent_results(limit=self.EXCHANGE_INTERVAL)
        active_results = [r for r in recent_results if r.details and "Replied" in r.details]
        
        return len(active_results) >= (self.EXCHANGE_INTERVAL // 2)

    async def extract(self) -> None:
        """
        Run a knowledge extraction cycle.
        """
        if not self.should_extract():
            return

        logger.info("KnowledgeAgent: Initiating knowledge extraction...")
        self._last_extraction_cycle = self.bb.cycle_count

        try:
            triples = await self._run_llm_extraction()
            if triples:
                for triple in triples:
                    if len(triple) == 3:
                        subject, predicate, obj = triple
                        self.db.add_knowledge_triple(subject, predicate, obj)
                        logger.debug(f"Added to Knowledge Graph: {subject} -[{predicate}]-> {obj}")
                logger.info(f"KnowledgeAgent: Extracted {len(triples)} new relational facts.")
        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}")

    async def _run_llm_extraction(self) -> List[List[str]]:
        """Run the LLM prompt to extract triples."""
        recent_msgs = self.db.get_recent_messages(self.contact_id, limit=20)
        if not recent_msgs:
            return []
            
        msgs_text = "\n".join(
            f"[{m['role'].upper()}]: {m['content']}"
            for m in recent_msgs
        )

        prompt = [
            {"role": "system", "content": self._get_extraction_prompt()},
            {"role": "user", "content": f"Recent Conversation:\n{msgs_text}"},
        ]

        try:
            raw = await self.llm.chat(prompt)
            # Find JSON array in the response
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start != -1 and end != 0:
                json_str = raw[start:end]
                data = json.loads(json_str)
                if isinstance(data, list):
                    return data
            return []
        except Exception as e:
            logger.error(f"Knowledge LLM call failed: {e}")
            return []

    @staticmethod
    def _get_extraction_prompt() -> str:
        """The system prompt for knowledge extraction."""
        return (
            "You are the KNOWLEDGE EXTRACTION layer of an autonomous AI agent.\n\n"
            "Your job is to read the conversation and extract relational facts "
            "(triples) to build a Knowledge Graph.\n\n"
            "A triple consists of: [Subject, Predicate, Object].\n"
            "Example outputs:\n"
            '[["Alice", "works at", "Tech Corp"], ["Bob", "friends with", "User"], ["User", "lives in", "New York"]]\n\n'
            "Rules:\n"
            "1. ONLY extract meaningful facts about people, places, organizations, or projects.\n"
            "2. Keep the predicate simple (e.g. 'works at', 'likes', 'friends with', 'is a').\n"
            "3. If there are no clear facts to extract, return an empty array: []\n"
            "4. IMPORTANT: Output ONLY valid JSON array of string arrays. No markdown formatting, no explanations."
        )
