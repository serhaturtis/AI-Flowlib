"""Agent core components."""

from flowlib.agent.core.agent import AgentCore
from flowlib.agent.core.orchestrator import AgentOrchestrator
from flowlib.agent.core.config_manager import AgentConfigManager
from flowlib.agent.core.state_manager import AgentStateManager
from flowlib.agent.core.memory_manager import AgentMemoryManager
from flowlib.agent.core.flow_runner import AgentFlowRunner
from flowlib.agent.core.learning_manager import AgentLearningManager
from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.errors import (
    AgentError,
    ConfigurationError,
    ExecutionError,
    StatePersistenceError,
    NotInitializedError,
    ComponentError
)
from flowlib.agent.core.interfaces import ComponentInterface

# Single source of truth - no backwards compatibility
Agent = AgentCore

__all__ = [
    "Agent",
    "AgentCore",
    "AgentOrchestrator",
    "AgentConfigManager",
    "AgentStateManager", 
    "AgentMemoryManager",
    "AgentFlowRunner",
    "AgentLearningManager",
    "AgentComponent",
    "ComponentInterface",
    "AgentError",
    "ConfigurationError",
    "ExecutionError",
    "StatePersistenceError",
    "NotInitializedError",
    "ComponentError"
]