"""Agent components package.

This package contains all reusable agent components organized by functionality.
Each component has its own module with models, interfaces, and implementations.
"""

# Import submodules for backward compatibility
from flowlib.agent.components import (
    classification,
    conversation,
    decorators,
    discovery,
    engine,
    intelligence,
    knowledge_flows,
    memory,
    planning,
    reflection,
    remember,
    shell_command,
    tasks
)

# Core component interfaces
from flowlib.agent.components.memory.agent_memory import AgentMemory
from flowlib.agent.components.planning.planner import AgentPlanner  
from flowlib.agent.components.engine.engine import AgentEngine
from flowlib.agent.components.reflection.base import AgentReflection

__all__ = [
    "AgentMemory",
    "AgentPlanner", 
    "AgentEngine",
    "AgentReflection",
    "classification",
    "conversation",
    "decorators", 
    "discovery",
    "engine",
    "intelligence",
    "knowledge_flows",
    "memory",
    "planning",
    "reflection",
    "remember",
    "shell_command",
    "tasks"
]