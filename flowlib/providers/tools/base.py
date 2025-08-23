"""Base tool provider implementation following flowlib patterns.

This module provides the abstract base class for all tool providers,
following flowlib's provider pattern with strict validation and lifecycle management.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Optional
from pydantic import Field

from flowlib.core.interfaces.interfaces import ToolProvider
from flowlib.providers.core.base import ProviderSettings, Provider
from flowlib.core.models import StrictBaseModel
from .models import ToolExecutionContext


class ToolProviderSettings(ProviderSettings):
    """Settings for tool providers with strict validation."""
    # Inherits strict configuration from ProviderSettings
    
    # Tool-specific settings
    working_directory: str = Field(default=".", description="Default working directory for tool execution")
    permission_level: str = Field(default="ask", description="Permission level (ask/allow/deny)")
    enable_logging: bool = Field(default=True, description="Enable tool execution logging")
    max_execution_time: int = Field(default=300, description="Maximum execution time in seconds")


class BaseToolProvider(Provider[ToolProviderSettings], ABC):
    """Abstract base class for all tool providers.
    
    This class implements the ToolProvider interface and follows flowlib's
    provider pattern with strict validation and proper lifecycle management.
    """
    
    def __init__(self, name: str, settings: ToolProviderSettings):
        """Initialize tool provider with strict validation."""
        super().__init__(name=name, provider_type="tool", settings=settings)
        self._description: Optional[str] = None
    
    @property
    def description(self) -> str:
        """Get tool description for schema generation."""
        return self._description or f"Tool: {self.name}"
    
    @description.setter
    def description(self, value: str) -> None:
        """Set tool description."""
        self._description = value
    
    @abstractmethod
    def get_parameter_model(self) -> Type[StrictBaseModel]:
        """Get Pydantic model class for parameter validation.
        
        Returns:
            Pydantic model class that defines tool parameters
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Tool providers must implement get_parameter_model()")
    
    @abstractmethod
    async def execute_tool(self, parameters: Dict[str, Any], context: Optional[ToolExecutionContext] = None) -> Dict[str, Any]:
        """Execute the tool with validated parameters.
        
        Args:
            parameters: Pre-validated tool parameters
            context: Optional execution context
            
        Returns:
            Tool execution result
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Tool providers must implement execute_tool()")
    
    def get_tool_schema(self) -> Dict[str, Any]:
        """Generate JSON schema for LLM tool calling.
        
        This method automatically generates the schema from the parameter model,
        following OpenAI function calling format.
        
        Returns:
            Tool schema for LLM integration
        """
        parameter_model = self.get_parameter_model()
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameter_model.model_json_schema()
            }
        }
    
    def create_execution_context(self, **kwargs) -> ToolExecutionContext:
        """Create execution context for tool execution.
        
        Args:
            **kwargs: Additional context parameters
            
        Returns:
            Tool execution context
        """
        context_data = {
            "working_directory": self.settings.working_directory,
            **kwargs
        }
        
        return ToolExecutionContext(**context_data)
    
    async def validate_and_execute(self, parameters: Dict[str, Any], context: Optional[ToolExecutionContext] = None) -> Dict[str, Any]:
        """Validate parameters and execute tool with error handling.
        
        Args:
            parameters: Raw tool parameters
            context: Optional execution context
            
        Returns:
            Tool execution result
            
        Raises:
            ValidationError: If parameter validation fails
            ProviderError: If tool execution fails
        """
        # Validate parameters using the tool's parameter model
        parameter_model = self.get_parameter_model()
        validated_params = parameter_model(**parameters)
        
        # Create default context if none provided
        if context is None:
            context = self.create_execution_context()
        
        # Execute tool with validated parameters
        return await self.execute_tool(validated_params.model_dump(), context)
    
    async def _initialize(self) -> None:
        """Initialize tool provider."""
        # Tool-specific initialization can be added by subclasses
        pass
    
    async def _shutdown(self) -> None:
        """Shutdown tool provider."""
        # Tool-specific cleanup can be added by subclasses
        pass