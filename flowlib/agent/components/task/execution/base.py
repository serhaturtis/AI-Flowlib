"""
Base tool implementation classes following flowlib patterns.

This module provides concrete base classes for implementing agent tools
with proper validation, error handling, and execution patterns.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Type, Optional, ClassVar
from uuid import uuid4

from flowlib.core.models import StrictBaseModel
from .models import (
    ToolParameters, ToolResult, ToolExecutionContext, ToolMetadata, 
    ToolStatus, ToolExecutionError, ToolErrorContext
)
from .interfaces import ToolInterface, AgentToolInterface, ParameterFactoryInterface

logger = logging.getLogger(__name__)


class ToolExecutionException(Exception):
    """Exception raised during tool execution."""
    
    def __init__(self, message: str, error_type: str = "execution_error", 
                 error_code: Optional[str] = None, context: Optional[ToolErrorContext] = None,
                 recoverable: bool = False):
        super().__init__(message)
        self.error_info = ToolExecutionError(
            error_type=error_type,
            error_message=message,
            error_code=error_code,
            context=context or ToolErrorContext(operation="unknown"),
            recoverable=recoverable
        )


class Tool(ABC):
    """Abstract base class for all agent tools.
    
    Provides common tool functionality including parameter validation,
    result handling, error management, and execution patterns.
    
    Tools are stateless and created fresh for each execution.
    Tools can optionally implement parameter generation from content.
    """
    
    # Tool metadata (override in subclasses)
    TOOL_NAME: ClassVar[str] = ""
    TOOL_DESCRIPTION: ClassVar[str] = ""
    TOOL_CATEGORY: ClassVar[str] = "general"
    
    def __init__(self, metadata: Optional[ToolMetadata] = None):
        """Initialize tool with optional metadata override."""
        if metadata is None:
            raise ValueError("Metadata is required for tool initialization")
        self._metadata = metadata
        self._execution_id = str(uuid4())
        logger.debug(f"Created tool instance: {self.name} ({self._execution_id})")
    
    @property
    def name(self) -> str:
        """Tool name for identification."""
        return self._metadata.name
    
    @property
    def description(self) -> str:
        """Tool description for LLM integration."""
        return self._metadata.description
    
    @property
    def metadata(self) -> ToolMetadata:
        """Tool metadata."""
        return self._metadata
    
    @property
    def execution_id(self) -> str:
        """Current execution ID."""
        return self._execution_id
    
    @abstractmethod
    def get_parameter_model(self) -> Type[StrictBaseModel]:
        """Get the parameter model class for this tool.
        
        Must be implemented by concrete tool classes.
        """
        pass
    
    @abstractmethod
    def get_result_model(self) -> Type[StrictBaseModel]:
        """Get the result model class for this tool.
        
        Must be implemented by concrete tool classes.
        """
        pass
    
    @abstractmethod
    async def _execute_impl(self, parameters: StrictBaseModel, context: Optional[ToolExecutionContext] = None) -> StrictBaseModel:
        """Core tool execution implementation.
        
        This method contains the actual tool logic and must be
        implemented by concrete tool classes.
        
        Args:
            parameters: Validated parameters
            context: Execution context
            
        Returns:
            Tool execution result
        """
        pass
    
    def create_parameters(self, content: str, context: ToolExecutionContext) -> StrictBaseModel:
        """Create parameters from content and execution context.
        
        Default implementation that tools MUST override for their specific needs.
        This base implementation raises an error to force proper implementation.
        
        Args:
            content: User content/message
            context: Typed execution context with shared data and previous results
            
        Returns:
            Validated parameter instance
            
        Raises:
            NotImplementedError: Always - tools must implement parameter creation
        """
        raise NotImplementedError(
            f"Tool {self.name} must implement create_parameters() method. "
            "Do not use field introspection or generic parameter mapping."
        )
    
    async def execute(self, parameters: StrictBaseModel, context: Optional[ToolExecutionContext] = None) -> StrictBaseModel:
        """Execute the tool with full error handling and validation.
        
        This method wraps _execute_impl with common functionality:
        - Parameter validation
        - Timeout handling  
        - Error catching and formatting
        - Execution timing
        - Result validation
        
        Args:
            parameters: Tool parameters (pre-validated)
            context: Optional execution context
            
        Returns:
            Structured tool result
        """
        start_time = datetime.now()
        result_model = self.get_result_model()
        
        try:
            # Validate parameters against expected model
            parameter_model = self.get_parameter_model()
            if not isinstance(parameters, parameter_model):
                raise ToolExecutionException(
                    f"Parameters must be of type {parameter_model.__name__}",
                    error_type="parameter_validation_error"
                )
            
            # Execute with timeout if specified
            timeout = None
            if context and context.timeout_seconds:
                timeout = context.timeout_seconds
            elif self._metadata.max_execution_time:
                timeout = self._metadata.max_execution_time
            
            if timeout:
                result = await asyncio.wait_for(
                    self._execute_impl(parameters, context),
                    timeout=timeout
                )
            else:
                result = await self._execute_impl(parameters, context)
            
            # Validate result
            if not isinstance(result, result_model):
                raise ToolExecutionException(
                    f"Result must be of type {result_model.__name__}",
                    error_type="result_validation_error"
                )
            
            # Add execution timing if result supports it
            if hasattr(result, 'execution_time_ms'):
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                if hasattr(result, 'model_copy'):
                    result = result.model_copy(update={'execution_time_ms': execution_time})
            
            logger.debug(f"Tool {self.name} executed successfully in {(datetime.now() - start_time).total_seconds():.3f}s")
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Tool {self.name} timed out after {timeout}s")
            return self._create_error_result(
                result_model, 
                "Tool execution timed out", 
                ToolStatus.TIMEOUT,
                start_time
            )
        
        except ToolExecutionException as e:
            logger.error(f"Tool {self.name} execution failed: {e.error_info.error_message}")
            return self._create_error_result(
                result_model,
                e.error_info.error_message,
                ToolStatus.ERROR,
                start_time,
                e.error_info
            )
        
        except Exception as e:
            logger.error(f"Tool {self.name} unexpected error: {str(e)}", exc_info=True)
            return self._create_error_result(
                result_model,
                f"Unexpected error: {str(e)}",
                ToolStatus.ERROR, 
                start_time
            )
    
    def get_schema(self) -> Dict[str, Any]:
        """Generate LLM tool calling schema."""
        parameter_model = self.get_parameter_model()
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameter_model.model_json_schema()
            }
        }
    
    def _create_default_metadata(self) -> ToolMetadata:
        """Create default metadata from class attributes."""
        return ToolMetadata(
            name=self.TOOL_NAME or self.__class__.__name__.lower().replace('tool', ''),
            description=self.TOOL_DESCRIPTION or f"Tool: {self.__class__.__name__}",
            category=self.TOOL_CATEGORY
        )
    
    def _create_error_result(self, result_model: Type[StrictBaseModel], message: str, 
                           status: ToolStatus, start_time: datetime,
                           error_info: Optional[ToolExecutionError] = None) -> StrictBaseModel:
        """Create an error result using the tool's result model."""
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Try to create error result with common fields
        try:
            return result_model(
                status=status,
                message=message,
                execution_time_ms=execution_time,
                timestamp=datetime.now()
            )
        except Exception:
            # Fallback: create basic ToolResult if result model doesn't support common fields
            return ToolResult(
                status=status,
                message=message,
                execution_time_ms=execution_time,
                timestamp=datetime.now()
            )


class AgentTool(Tool):
    """Base class for agent-context aware tools.
    
    These tools can access agent state, memory, and other
    agent-specific capabilities during execution.
    """
    
    async def execute_with_agent(self, parameters: StrictBaseModel, agent_context: 'AgentExecutionContext') -> StrictBaseModel:
        """Execute tool with agent context access.
        
        Default implementation creates ToolExecutionContext from agent context
        and calls regular execute method. Override for agent-specific behavior.
        
        Args:
            parameters: Validated parameters
            agent_context: Agent execution context
            
        Returns:
            Tool execution result
        """
        # Convert agent context to tool context
        # This would be implemented based on actual AgentExecutionContext structure
        tool_context = ToolExecutionContext(
            working_directory=".",
            execution_id=self.execution_id,
            agent_id=getattr(agent_context, 'agent_id', None),
            session_id=getattr(agent_context, 'session_id', None)
        )
        
        return await self.execute(parameters, tool_context)


class AsyncTool(Tool):
    """Base class for tools that require async execution patterns.
    
    Provides additional async utilities and patterns for complex
    asynchronous tool operations.
    """
    
    def __init__(self, metadata: Optional[ToolMetadata] = None):
        super().__init__(metadata)
        self._background_tasks: list = []
    
    async def cleanup(self):
        """Cleanup any background tasks or resources."""
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
    
    def add_background_task(self, coro):
        """Add a background task to be managed by the tool."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        return task