"""Generic tool orchestration system for agent tool execution.

This module provides the orchestration layer that coordinates tool execution
through the registry system. Following flowlib's architectural patterns.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, cast

from pydantic import Field

from flowlib.core.models import StrictBaseModel

from .models import (
    ToolErrorContext,
    ToolExecutionContext,
    ToolExecutionError,
    ToolResult,
    ToolStatus,
)
from .registry import ToolNotFoundError, ToolRegistry, tool_registry

logger = logging.getLogger(__name__)


class ToolExecutionRequest(StrictBaseModel):
    """Request for tool execution through orchestration."""

    tool_name: str
    raw_parameters: Dict[str, Any]
    context: Optional[ToolExecutionContext] = None
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class ToolExecutionResponse(StrictBaseModel):
    """Response from tool execution orchestration."""

    execution_id: str
    tool_name: str
    status: ToolStatus
    result: Optional[ToolResult] = None
    error: Optional[ToolExecutionError] = None
    execution_time_ms: Optional[float] = None
    timestamp: datetime

    def get_display_content(self) -> str:
        """Get displayable content from the execution response."""
        if self.result:
            return self.result.get_display_content()
        elif self.error:
            return f"Error: {self.error.error_message}"
        return f"Tool {self.tool_name} execution {self.status.value}"

    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ToolStatus.SUCCESS

    def is_error(self) -> bool:
        """Check if execution failed."""
        return self.status == ToolStatus.ERROR


class ToolOrchestrator:
    """Generic orchestrator for agent tool execution.
    
    Provides coordination between tools and registry with:
    - Generic parameter validation and creation
    - Tool instance management
    - Execution context handling
    - Error handling and reporting
    - Execution tracking
    """

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self._registry = registry or tool_registry
        logger.debug("Tool orchestrator initialized")

    async def execute_tool(self, request: ToolExecutionRequest) -> ToolExecutionResponse:
        """Execute a tool through orchestration.
        
        Args:
            request: Tool execution request with parameters and context
            
        Returns:
            Tool execution response with result or error
        """
        start_time = datetime.now()

        try:
            logger.debug(f"Executing tool: {request.tool_name} ({request.execution_id})")

            # Create a proper TodoItem from raw parameters
            from ..models import TodoItem
            # Ensure content has a default if not in raw_parameters
            raw_params = dict(request.raw_parameters)
            if 'content' not in raw_params:
                raw_params['content'] = f"Execute {request.tool_name}"

            # Add assigned tool
            raw_params['assigned_tool'] = request.tool_name

            todo = TodoItem(**raw_params)

            # Execute tool using new architecture (tool handles its own parameters)
            from .models import ToolExecutionSharedData
            default_context = ToolExecutionContext(
                working_directory="/tmp",
                agent_id="system_agent",
                agent_persona="system",
                execution_id=f"exec_{int(datetime.now().timestamp())}",
                shared_data=ToolExecutionSharedData()
            )
            result = await self._registry.execute_todo(
                todo,
                request.context or default_context
            )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Check if result indicates success using safe method calls
            status = ToolStatus.SUCCESS  # Default to success
            try:
                # Try is_success method first
                if result.is_success():
                    status = ToolStatus.SUCCESS
                else:
                    status = ToolStatus.ERROR
            except (AttributeError, TypeError):
                # Try is_error method if is_success doesn't exist
                try:
                    if result.is_error():
                        status = ToolStatus.ERROR
                except (AttributeError, TypeError):
                    # Neither method exists, keep default success status
                    pass

            return ToolExecutionResponse(
                execution_id=request.execution_id,
                tool_name=request.tool_name,
                status=status,
                result=result,
                execution_time_ms=execution_time,
                timestamp=datetime.now()
            )

        except (ToolNotFoundError, KeyError) as e:
            logger.error(f"Tool not found: {request.tool_name}")
            return self._create_error_response(
                request, str(e), "tool_not_found", start_time
            )

        except Exception as e:
            logger.error(f"Tool execution failed: {request.tool_name} - {str(e)}", exc_info=True)
            return self._create_error_response(
                request, str(e), "execution_error", start_time
            )

    async def execute_multiple_tools(
        self,
        requests: List[ToolExecutionRequest]
    ) -> List[ToolExecutionResponse]:
        """Execute multiple tools concurrently.
        
        Args:
            requests: List of tool execution requests
            
        Returns:
            List of tool execution responses in same order
        """
        logger.debug(f"Executing {len(requests)} tools concurrently")

        # Execute all tools concurrently
        tasks = [self.execute_tool(request) for request in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that weren't caught
        final_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                error_response = self._create_error_response(
                    requests[i], str(response), "unexpected_error", datetime.now()
                )
                final_responses.append(error_response)
            else:
                final_responses.append(cast(ToolExecutionResponse, response))

        return final_responses

    def get_available_tools(self) -> List[str]:
        """Get list of available tools from registry.
        
        Returns:
            List of available tool names
        """
        return self._registry.list_tools()

    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get parameter schema for a specific tool.
        
        Args:
            tool_name: Name of tool to get schema for
            
        Returns:
            JSON schema for tool parameters
            
        Raises:
            ToolNotFoundError: If tool is not found
        """
        # For now, return basic schema - tools handle their own validation
        if not self._registry.contains(tool_name):
            raise ToolNotFoundError(f"Tool not found: {tool_name}")
        return {"type": "object", "properties": {"content": {"type": "string"}}}

    def get_all_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter schemas for all available tools.
        
        Returns:
            Dict mapping tool names to their parameter schemas
        """
        schemas = {}
        for tool_name in self._registry.list_tools():
            schemas[tool_name] = self.get_tool_schema(tool_name)
        return schemas

    def get_tools_by_category(self, category: str) -> List[str]:
        """Get tools by category.
        
        Args:
            category: Tool category to filter by
            
        Returns:
            List of tool names in the category
        """
        return self._registry.list({"category": category})

    def get_tools_by_capability(self, **capabilities: Union[str, int, bool]) -> List[str]:
        """Get tools by capabilities.
        
        Args:
            **capabilities: Capability requirements to match
            
        Returns:
            List of tool names matching capabilities
        """
        # For now, return all tools - capability filtering not implemented
        return self._registry.list_tools()

    def create_execution_context(
        self,
        agent_id: str,
        agent_persona: str,
        working_directory: str = ".",
        session_id: str = "",
        task_id: str = "",
        **kwargs: Any
    ) -> ToolExecutionContext:
        """Create a tool execution context.

        Args:
            working_directory: Working directory for tool execution
            agent_id: ID of the executing agent
            session_id: Agent session ID
            task_id: Current task ID
            **kwargs: Additional context data

        Returns:
            Tool execution context
        """
        from .models import ToolExecutionSharedData

        # Ensure shared_data is provided - either from kwargs or create new
        if 'shared_data' not in kwargs:
            kwargs['shared_data'] = ToolExecutionSharedData()

        return ToolExecutionContext(
            working_directory=working_directory,
            agent_id=agent_id,
            agent_persona=agent_persona,
            session_id=session_id,
            task_id=task_id,
            execution_id=str(uuid.uuid4()),
            **kwargs
        )

    def _create_error_response(
        self,
        request: ToolExecutionRequest,
        error_message: str,
        error_type: str,
        start_time: datetime
    ) -> ToolExecutionResponse:
        """Create an error response for failed tool execution."""
        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        error = ToolExecutionError(
            error_type=error_type,
            error_message=error_message,
            context=ToolErrorContext(
                operation="tool_execution",
                attempted_values={"tool_name": request.tool_name}
            )
        )

        return ToolExecutionResponse(
            execution_id=request.execution_id,
            tool_name=request.tool_name,
            status=ToolStatus.ERROR,
            error=error,
            execution_time_ms=execution_time,
            timestamp=datetime.now()
        )


# Global orchestrator instance
tool_orchestrator = ToolOrchestrator()
