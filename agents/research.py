"""
agents/research.py -- Intelligence gathering layer of the agentic system.

The ResearchAgent runs between the Strategist and Tactician.
If the Strategist or ContextAnalyzer flags a message as needing research,
this agent performs a web search and posts the results to the Blackboard.
This offloads tool usage from the Tactician, saving LLM tokens and improving
the reliability of factual responses.
"""

import logging
from typing import Any, Optional

from agents.blackboard import Blackboard, AgentMessage, StrategyIntent

logger = logging.getLogger(__name__)


class ResearchAgent:
    """
    Dedicated intelligence-gathering sub-agent.
    
    Reads Blackboard messages requesting research, executes the search,
    and posts results back for the Tactician to consume.
    """

    def __init__(
        self,
        blackboard: Blackboard,
        llm_client: Any,
        tool_executor: Any,
    ):
        self.bb = blackboard
        self.llm = llm_client
        self.tools = tool_executor

    async def research(self) -> bool:
        """
        Check for research requests and execute them.
        
        Returns:
            True if research was performed, False if skipped.
        """
        # 1. Read messages addressed to ResearchAgent
        msgs = self.bb.consume_messages("ResearchAgent")
        
        # 2. Check ContextAnalyzer flags
        needs_research = self.bb.get_context("needs_research", False)
        
        if not msgs and not needs_research:
            return False
            
        strategy = self.bb.active_strategy
        if strategy and strategy.intent not in (StrategyIntent.RESPOND, StrategyIntent.INITIATE):
            return False

        logger.info("ResearchAgent: Initiating intelligence gathering...")
        
        queries = []
        for msg in msgs:
            queries.append(msg.content)
            
        if not queries and needs_research:
            # If flagged but no explicit query, ask LLM to extract the search query
            signal = self.bb.current_signal
            if signal and signal.payload.get("text"):
                text = signal.payload["text"]
                query = await self._extract_query(text)
                if query:
                    queries.append(query)
                    
        if not queries:
            logger.debug("ResearchAgent: No valid queries extracted.")
            return False
            
        # Execute the first query (keep it simple for now)
        primary_query = queries[0]
        logger.info(f"ResearchAgent: Searching web for '{primary_query}'")
        
        results = await self.tools.execute("web_search", {"query": primary_query})
        
        # Post results to shared context for the Tactician
        self.bb.set_context("research_results", f"Query: {primary_query}\n{results}")
        
        # Also post a message to the Tactician
        await self.bb.post_message(
            AgentMessage(
                sender="ResearchAgent",
                receiver="Tactician",
                content=f"I performed a web search. Check shared_context['research_results'].",
                msg_type="info"
            )
        )
        
        return True

    async def _extract_query(self, text: str) -> Optional[str]:
        """Use LLM to extract the best search query from a user's message."""
        prompt = [
            {
                "role": "system",
                "content": (
                    "Extract the optimal Google/DuckDuckGo search query to answer the user's message. "
                    "Return ONLY the search query text, nothing else. "
                    "If no clear query can be formulated, return 'NONE'."
                )
            },
            {"role": "user", "content": text}
        ]
        
        try:
            raw = await self.llm.chat(prompt)
            query = raw.strip().strip("'").strip('"')
            if query.upper() == "NONE":
                return None
            return query
        except Exception as e:
            logger.error(f"Failed to extract research query: {e}")
            return None
