"""Agent tool calling flow implementation.

This module provides the main flow for autonomous agent tool calling,
using structured generation and flowlib's provider system.
"""

import time
import logging
from typing import List, Dict, Any, Optional

from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.flows.base.base import Flow
from flowlib.core.container.container import get_container
from flowlib.providers.tools.models import (
    ToolCallRequest, ToolExecutionResult, ToolCall, ToolResult, ToolExecutionContext
)
from flowlib.resources.registry.registry import resource_registry
from .prompts import ToolSelectionPrompt

logger = logging.getLogger(__name__)


@flow(name="agent-tool-calling", description="Autonomous agent tool calling with structured generation")
class AgentToolCallingFlow(Flow):
    """Universal flow for agent tool calling.
    
    This flow handles all agent tool execution by:
    1. Discovering available tools
    2. Using LLM to select and parameterize tools
    3. Executing selected tools via provider system
    4. Aggregating and returning results
    """
    
    def __init__(self):
        super().__init__(
            name_or_instance="agent-tool-calling",
            input_schema=ToolCallRequest,
            output_schema=ToolExecutionResult
        )
    
    @pipeline(input_model=ToolCallRequest, output_model=ToolExecutionResult)
    async def execute_tool_calling(self, request: ToolCallRequest) -> ToolExecutionResult:
        """Main pipeline for tool calling execution.
        
        Args:
            request: Tool calling request
            
        Returns:
            Complete tool execution results
        """
        start_time = time.time()
        
        # Stage 1: Discover available tools
        available_tools = await self._discover_available_tools()
        logger.info(f"Discovered {len(available_tools)} available tools")
        
        # Stage 2: Use LLM to select and parameterize tools
        selected_tools = await self._select_tools_with_llm(request, available_tools)
        logger.info(f"LLM selected {len(selected_tools)} tools")
        
        # Stage 3: Execute selected tools
        execution_results = await self._execute_selected_tools(selected_tools, request)
        
        # Stage 4: Aggregate results
        total_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        success_count = sum(1 for result in execution_results if result.status == "success")
        error_count = len(execution_results) - success_count
        
        return ToolExecutionResult(
            request=request,
            selected_tools=selected_tools,
            results=execution_results,
            total_execution_time_ms=total_time,
            success_count=success_count,
            error_count=error_count
        )
    
    async def _discover_available_tools(self) -> List[Dict[str, Any]]:
        """Discover all available tool providers.
        
        Returns:
            List of tool schemas for LLM consumption
        """
        container = get_container()
        tool_schemas = []
        
        # Get all providers of type "tool"
        try:
            # Get all registered provider names
            provider_names = container.list_providers()
            
            for provider_name in provider_names:
                try:
                    # Get provider metadata to check if it's a tool
                    metadata = container.get_metadata("provider", provider_name)
                    
                    if metadata.get("provider_type") == "tool":
                        # Get the actual provider instance
                        provider = await container.get_provider_by_type("tool", provider_name)
                        
                        if hasattr(provider, 'get_tool_schema'):
                            schema = provider.get_tool_schema()
                            tool_schemas.append(schema)
                            logger.debug(f"Added tool schema for: {provider_name}")
                        
                except Exception as e:
                    logger.warning(f"Failed to get schema for tool provider '{provider_name}': {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to discover tool providers: {e}")
        
        return tool_schemas
    
    async def _select_tools_with_llm(self, request: ToolCallRequest, available_tools: List[Dict[str, Any]]) -> List[ToolCall]:
        """Use LLM to select and parameterize tools.
        
        Args:
            request: Tool calling request
            available_tools: Available tool schemas
            
        Returns:
            List of selected tool calls
        """
        if not available_tools:
            logger.warning("No tools available for selection")
            return []
        
        try:
            # Get LLM provider using configuration system
            from flowlib.providers import provider_registry
            # Use default LLM for tool selection
            llm_provider = await provider_registry.get_by_config("default-llm")
            
            # Get tool selection prompt
            tool_prompt = resource_registry.get("tool-selection")
            
            # Format available tools for prompt
            tools_text = self._format_tools_for_prompt(available_tools)
            
            # Prepare prompt variables
            prompt_variables = {
                "task_description": request.task_description,
                "available_tools": tools_text,
                "working_directory": request.working_directory,
                "context": str(request.context) if request.context else "None",
                "max_tools": request.max_tools
            }
            
            # Use structured generation to get tool calls
            tool_calls = await llm_provider.generate_structured(
                prompt=tool_prompt,
                output_type=List[ToolCall],
                prompt_variables=prompt_variables
            )
            
            # Validate tool calls
            validated_calls = []
            for call in tool_calls:
                if self._validate_tool_call(call, available_tools):
                    validated_calls.append(call)
                else:
                    logger.warning(f"Invalid tool call filtered out: {call.tool_name}")
            
            return validated_calls
            
        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            return []
    
    async def _execute_selected_tools(self, tool_calls: List[ToolCall], request: ToolCallRequest) -> List[ToolResult]:
        """Execute selected tools via provider system.
        
        Args:
            tool_calls: Selected tool calls
            request: Original request for context
            
        Returns:
            List of tool execution results
        """
        results = []
        container = get_container()
        
        # Create execution context
        context = ToolExecutionContext(
            working_directory=request.working_directory,
            metadata=request.context or {}
        )
        
        for call in tool_calls:
            start_time = time.time()
            
            try:
                # Get tool provider
                tool_provider = await container.get_provider_by_type("tool", call.tool_name)
                
                if not tool_provider:
                    results.append(ToolResult(
                        tool_name=call.tool_name,
                        status="error",
                        error=f"Tool provider '{call.tool_name}' not found",
                        execution_time_ms=(time.time() - start_time) * 1000,
                        call_id=call.call_id
                    ))
                    continue
                
                # Execute tool with validation
                result_data = await tool_provider.validate_and_execute(
                    parameters=call.parameters,
                    context=context
                )
                
                results.append(ToolResult(
                    tool_name=call.tool_name,
                    status="success",
                    result=result_data,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    call_id=call.call_id
                ))
                
                logger.info(f"Successfully executed tool: {call.tool_name}")
                
            except Exception as e:
                results.append(ToolResult(
                    tool_name=call.tool_name,
                    status="error",
                    error=str(e),
                    execution_time_ms=(time.time() - start_time) * 1000,
                    call_id=call.call_id
                ))
                
                logger.error(f"Tool execution failed for '{call.tool_name}': {e}")
        
        return results
    
    def _format_tools_for_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """Format tool schemas for LLM prompt.
        
        Args:
            tools: Tool schemas
            
        Returns:
            Formatted tools description
        """
        if not tools:
            return "No tools available"
        
        formatted_tools = []
        for tool in tools:
            function_info = tool.get("function", {})
            name = function_info.get("name", "unknown")
            description = function_info.get("description", "No description")
            parameters = function_info.get("parameters", {})
            
            # Format parameters
            params_info = []
            if "properties" in parameters:
                for param_name, param_info in parameters["properties"].items():
                    param_type = param_info.get("type", "unknown")
                    param_desc = param_info.get("description", "No description")
                    required = param_name in parameters.get("required", [])
                    required_mark = "*" if required else ""
                    
                    params_info.append(f"  - {param_name}{required_mark} ({param_type}): {param_desc}")
            
            params_text = "\n".join(params_info) if params_info else "  No parameters"
            
            formatted_tools.append(f"""
Tool: {name}
Description: {description}
Parameters:
{params_text}
""".strip())
        
        return "\n\n".join(formatted_tools)
    
    def _validate_tool_call(self, call: ToolCall, available_tools: List[Dict[str, Any]]) -> bool:
        """Validate that a tool call is valid.
        
        Args:
            call: Tool call to validate
            available_tools: Available tool schemas
            
        Returns:
            True if valid, False otherwise
        """
        # Check if tool exists
        tool_names = [
            tool.get("function", {}).get("name")
            for tool in available_tools
        ]
        
        if call.tool_name not in tool_names:
            return False
        
        # Additional validation could be added here (parameter validation, etc.)
        return True