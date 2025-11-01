"""Unified Knowledge Component - Single source of truth.

This module provides the consolidated knowledge component that handles learning,
storage, and retrieval operations following flowlib's AgentComponent patterns.
"""

# Import prompts to register them with resource registry
from . import prompts
from .component import KnowledgeComponent
from .flows import (
    AgentKnowledgeExtractionFlow,
    AgentKnowledgeExtractionInput,
    AgentKnowledgeExtractionOutput,
    AgentKnowledgeRetrievalFlow,
    AgentKnowledgeRetrievalInput,
    AgentKnowledgeRetrievalOutput,
)
from .models import (
    Concept,
    ConfidenceLevel,
    Entity,
    KnowledgeComponentConfig,
    KnowledgeSet,
    KnowledgeType,
    LearningInput,
    LearningResult,
    Pattern,
    Relationship,
    RetrievalRequest,
    RetrievalResult,
    StorageRequest,
)

__all__ = [
    # Models
    "KnowledgeType",
    "ConfidenceLevel",
    "Entity",
    "Concept",
    "Relationship",
    "Pattern",
    "KnowledgeSet",
    "LearningInput",
    "LearningResult",
    "StorageRequest",
    "RetrievalRequest",
    "RetrievalResult",
    "KnowledgeComponentConfig",
    # Component
    "KnowledgeComponent",
    # Flows
    "AgentKnowledgeExtractionFlow",
    "AgentKnowledgeRetrievalFlow",
    "AgentKnowledgeExtractionInput",
    "AgentKnowledgeExtractionOutput",
    "AgentKnowledgeRetrievalInput",
    "AgentKnowledgeRetrievalOutput",
    # Prompt registration
    "prompts",
]
