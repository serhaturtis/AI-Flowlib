"""Context management system for agents.

This module provides unified context management for all agent operations,
replacing fragmented context assembly with a single source of truth.
"""

from .models import (
    ExecutionContext,
    SessionContext,
    TaskContext,
    ComponentContext,
    LearningContext,
    ConversationMessage,
    UserProfile,
    WorkspaceKnowledge,
    SuccessfulPattern,
    RecoveryStrategy
)
from .manager import AgentContextManager

__all__ = [
    "ExecutionContext",
    "SessionContext", 
    "TaskContext",
    "ComponentContext",
    "LearningContext",
    "ConversationMessage",
    "UserProfile",
    "WorkspaceKnowledge",
    "SuccessfulPattern",
    "RecoveryStrategy",
    "AgentContextManager"
]