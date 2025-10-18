"""Agent task execution system.

This module provides clean tool execution without decomposition concerns.
Following flowlib's single source of truth and no backward compatibility principles.

Architecture:
- ToolOrchestrator: Single orchestration approach
- Tool Registry: Discovery and instantiation
- Protocol interfaces: Clean contracts

Usage:
    from flowlib.agent.components.task.execution import tool_registry

    # Execute tool directly
    result = await tool_registry.execute_todo(todo, context)
"""

# Core system
# Import tool implementations to trigger registration
from . import tool_implementations
from .decorators import tool

# Import interfaces
from .interfaces import AgentToolFactory, AgentToolInterface, ToolFactory, ToolInterface
from .models import (
    AgentTaskRequest,
    TaskExecutionResult,
    TodoExecutionContext,
    TodoExecutionResult,
    ToolExecutionContext,
    ToolExecutionError,
    ToolMetadata,
    ToolParameters,
    ToolResult,
    ToolStatus,
)

# Tool orchestration
from .orchestration import (
    ToolExecutionRequest,
    ToolExecutionResponse,
    ToolOrchestrator,
    tool_orchestrator,
)
from .registry import ToolRegistry, tool_registry

__all__ = [
    # Tool implementations (imported for registration)
    "tool_implementations",

    # Core models
    "ToolParameters",
    "ToolResult",
    "ToolExecutionContext",
    "ToolStatus",
    "ToolMetadata",
    "ToolExecutionError",
    "AgentTaskRequest",
    "TaskExecutionResult",
    "TodoExecutionContext",
    "TodoExecutionResult",

    # Core interfaces
    "ToolInterface",
    "ToolFactory",
    "AgentToolInterface",
    "AgentToolFactory",

    # Registry system
    "ToolRegistry",
    "tool_registry",

    # Decorator
    "tool",

    # Orchestration
    "ToolOrchestrator",
    "tool_orchestrator",
    "ToolExecutionRequest",
    "ToolExecutionResponse",
]
