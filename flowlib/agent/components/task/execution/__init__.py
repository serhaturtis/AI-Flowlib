"""Agent task execution system.

This module provides clean tool execution without decomposition concerns.
Following flowlib's single source of truth and no backward compatibility principles.

Architecture:
- TaskExecutionFlow: Executes pre-decomposed TODO batches
- ToolOrchestrator: Single orchestration approach  
- Tool Registry: Discovery and instantiation
- Protocol interfaces: Clean contracts

Usage:
    from flowlib.agent.components.task_execution import TaskExecutionFlow
    
    # Execute pre-decomposed TODOs
    flow = TaskExecutionFlow()
    result = await flow.execute_todo_batch(todos, context)
"""

# Core system
from .models import (
    ToolParameters,
    ToolResult, 
    ToolExecutionContext,
    ToolStatus,
    ToolMetadata,
    ToolExecutionError,
    AgentTaskRequest,
    TaskExecutionResult,
)
from .interfaces import ToolInterface, ToolFactory
from .registry import ToolRegistry, tool_registry
from .decorators import tool

# Tool orchestration
from .orchestration import ToolOrchestrator, tool_orchestrator, ToolExecutionRequest, ToolExecutionResponse

# Import tool implementations to trigger registration
from .tool_implementations import *

# Import clean execution flows and coordinators
from .execution_flow import TaskExecutionFlow
from .component import TaskExecutionComponent

__all__ = [
    # Core models
    "ToolParameters",
    "ToolResult", 
    "ToolExecutionContext",
    "ToolStatus",
    "ToolMetadata",
    "ToolExecutionError",
    "AgentTaskRequest",
    "TaskExecutionResult",
    
    # Core interfaces  
    "ToolInterface",
    "ToolFactory",
    
    # Registry system
    "ToolRegistry", 
    "tool_registry",
    
    # Decorator
    "tool",
    
    # Flows
    "TaskExecutionFlow",
    
    # Executors
    "TaskExecutionComponent",
    
    # Orchestration
    "ToolOrchestrator",
    "tool_orchestrator", 
    "ToolExecutionRequest",
    "ToolExecutionResponse",
]