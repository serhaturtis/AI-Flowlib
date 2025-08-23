"""Knowledge extraction and retrieval flows for the agent system.

This module contains flows that enable the agent to extract knowledge from
conversations and web research, and to retrieve domain-specific knowledge
using the plugin system.
"""

from .knowledge_extraction import KnowledgeExtractionFlow
from .knowledge_retrieval import KnowledgeRetrievalFlow

__all__ = [
    "KnowledgeExtractionFlow",
    "KnowledgeRetrievalFlow"
]