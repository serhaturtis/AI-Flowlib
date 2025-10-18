"""Context management system for agents.

This module provides unified context management for all agent operations,
replacing fragmented context assembly with a single source of truth.
"""

from .manager import AgentContextManager
from .models import (
    ComponentContext,
    ContextManagerConfig,
    ConversationMessage,
    ExecutionContext,
    LearningContext,
    RecoveryStrategy,
    SessionContext,
    SuccessfulPattern,
    TaskContext,
    UserProfile,
    WorkspaceKnowledge,
)

__all__ = [
    "ContextManagerConfig",
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
