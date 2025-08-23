"""
Engine interface definitions.

This module defines the protocols and interfaces for execution engine operations.
"""

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class EngineInterface(Protocol):
    """Interface for execution engine.
    
    Defines the methods for executing cycles and managing flow execution.
    """
    
    async def execute_cycle(
        self,
        state: Any,  # AgentState
        memory_context: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Execute one planning-execution-reflection cycle.
        
        Args:
            state: Current agent state
            memory_context: Memory context for this cycle
            **kwargs: Additional execution arguments
            
        Returns:
            True if execution should continue, False if complete
        """
        ... 