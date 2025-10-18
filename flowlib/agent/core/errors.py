"""
Agent error classes.

This module defines the error classes used throughout the agent system.
All agent errors now inherit from the unified core error system.
"""

from typing import Any, Dict, Optional, Type, Union

from flowlib.core.errors.errors import BaseError
from flowlib.core.errors.errors import ErrorContext as CoreErrorContext


class ErrorContext:
    """Helper for creating error context dictionaries."""

    @staticmethod
    def create(**context: Union[str, int, bool, None]) -> Dict[str, Union[str, int, bool, None]]:
        """Create an error context dictionary."""
        return {k: v for k, v in context.items() if v is not None}


class AgentError(BaseError):
    """Base error class for agent errors.
    
    All errors in the agent system should inherit from this class.
    Now inherits from the unified core error system.
    """

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        component: str = "agent",
        operation: str = "unknown",
        **context: Union[str, int, bool, None]
    ):
        """Initialize agent error.
        
        Args:
            message: Error message
            cause: Original exception that caused this error
            component: Agent component name
            operation: Operation being performed
            **context: Additional context information
        """
        # Create core error context
        # flow_name is optional - many errors occur outside flow context
        flow_name_value = context.get("flow_name", "system")
        flow_name = str(flow_name_value) if flow_name_value is not None else "system"
        error_context = CoreErrorContext.create(
            flow_name=flow_name,
            error_type=self.__class__.__name__,
            error_location=f"{component}.{operation}",
            component=component,
            operation=operation
        )

        # Call parent constructor
        super().__init__(message, error_context, cause)

        # Store additional context as a separate field
        self.additional_context: Dict[str, Any] = dict(context)

    def __str__(self) -> str:
        """String representation - just the message for backward compatibility."""
        if self.cause:
            return f"{self.message} (caused by: {self.cause})"
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary.
        
        Returns:
            Dictionary representation of the error
        """
        result = {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.additional_context,
        }

        if self.cause:
            if hasattr(self.cause, "to_dict") and callable(self.cause.to_dict):
                result["cause"] = self.cause.to_dict()
            else:
                result["cause"] = {
                    "error_type": self.cause.__class__.__name__,
                    "message": str(self.cause),
                }

        return result


class NotInitializedError(AgentError):
    """Error raised when a component is used before initialization.
    
    Components must be initialized before they can be used.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        component_name: Optional[str] = None,
        operation: Optional[str] = None,
        **context: Union[str, int, bool, None]
    ):
        """Initialize not initialized error.
        
        Args:
            message: Optional custom error message
            component_name: Name of the component that was not initialized
            operation: Optional operation that was attempted
            **context: Additional context information
        """
        if message is None:
            message = f"Component '{component_name}' must be initialized before use"
            if operation:
                message += f" (attempted operation: {operation})"

        if component_name:
            context["component_name"] = component_name
        if operation:
            context["operation"] = operation

        # Remove operation from context since it's a parameter for parent class
        parent_context = {k: v for k, v in context.items() if k != "operation"}
        if not operation:
            raise ValueError("AgentError requires operation parameter - cannot be None or empty")
        if not component_name:
            raise ValueError("AgentError requires component_name parameter - cannot be None or empty")
        parent_operation = operation
        super().__init__(message, None, component_name, parent_operation, **parent_context)

        # Add operation back to context for backward compatibility
        if operation:
            self.additional_context["operation"] = operation


class ComponentError(AgentError):
    """Error in component operation.
    
    Raised when a component fails to initialize, operate, or shutdown.
    """

    def __init__(
        self,
        message: str,
        component_name: str,
        operation: Optional[str] = None,
        cause: Optional[Exception] = None,
        **context: Union[str, int, bool, None]
    ):
        """Initialize component error.
        
        Args:
            message: Error message
            component_name: Name of the component that failed
            operation: Optional name of the operation that failed
            cause: Original exception that caused this error
            **context: Additional context information
        """
        context["component_name"] = component_name
        if operation:
            context["operation"] = operation

        # Remove operation from context since it's a parameter for parent class
        parent_context = {k: v for k, v in context.items() if k != "operation"}
        parent_operation = operation if operation else "unknown"
        super().__init__(message, cause=cause, component=component_name, operation=parent_operation, **parent_context)

        # Add operation back to context for backward compatibility
        if operation:
            self.additional_context["operation"] = operation


class ConfigurationError(AgentError):
    """Error in agent configuration.
    
    Raised when there is an invalid configuration value or missing required config.
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        required_type: Optional[Type[Any]] = None,
        cause: Optional[Exception] = None,
        **context: Union[str, int, bool, None]
    ):
        """Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Key of the problematic configuration
            invalid_value: Invalid value that caused the error
            required_type: Type that was expected
            cause: Original exception that caused this error
            **context: Additional context information
        """
        self.config_key = config_key

        if config_key:
            context["config_key"] = config_key
        if invalid_value is not None:
            context["invalid_value"] = str(invalid_value)
        if required_type:
            context["required_type"] = str(required_type)

        super().__init__(message, cause, "config", "validation", **context)


class ExecutionError(AgentError):
    """Error during agent execution.
    
    Raised when there is a failure during the execution cycle.
    """

    def __init__(
        self,
        message: str,
        agent: Optional[str] = None,
        state: Optional[Any] = None,
        flow: Optional[str] = None,
        stage: Optional[str] = None,
        cause: Optional[Exception] = None,
        **context: Union[str, int, bool, None]
    ):
        """Initialize execution error.
        
        Args:
            message: Error message
            agent: Name of the agent that encountered the error
            state: State at the time of the error
            flow: Flow that was being executed
            stage: Stage in the execution cycle
            cause: Original exception that caused this error
            **context: Additional context information
        """
        if agent:
            context["agent"] = agent
        if flow:
            context["flow"] = flow
        if stage:
            context["stage"] = stage
        if state:
            # Add key state attributes but avoid huge serialization
            try:
                if hasattr(state, "task_id"):
                    context["task_id"] = state.task_id
                if hasattr(state, "is_complete"):
                    context["is_complete"] = state.is_complete
                if hasattr(state, "progress"):
                    context["progress"] = state.progress
            except Exception:
                # Ignore any errors in state extraction
                pass

        super().__init__(message, cause, "execution", "execute", **context)


class PlanningError(ExecutionError):
    """Error during agent planning.
    
    Raised when there is a failure during the planning phase.
    """

    def __init__(
        self,
        message: str,
        planning_type: str = "planning",  # "planning" or "input_generation"
        **kwargs: Any
    ):
        """Initialize planning error.
        
        Args:
            message: Error message
            planning_type: Type of planning that failed
            **kwargs: Additional arguments for ExecutionError
        """
        # Add planning type to context
        kwargs["planning_type"] = planning_type
        kwargs["stage"] = planning_type

        super().__init__(message, **kwargs)


class ReflectionError(ExecutionError):
    """Error during agent reflection.
    
    Raised when there is a failure during the reflection phase.
    """

    def __init__(
        self,
        message: str,
        **kwargs: Any
    ):
        """Initialize reflection error.
        
        Args:
            message: Error message
            **kwargs: Additional arguments for ExecutionError
        """
        kwargs["stage"] = "reflection"
        super().__init__(message, **kwargs)


class MemoryError(AgentError):
    """Error in memory operations.
    
    Raised when there is a failure during memory store, retrieve, or search.
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        key: Optional[str] = None,
        context: Optional[str] = None,
        cause: Optional[Exception] = None,
        **kwargs: Any
    ):
        """Initialize memory error.
        
        Args:
            message: Error message
            operation: Memory operation that failed (store, retrieve, search)
            key: Key involved in the operation
            context: Memory context name
            cause: Original exception that caused this error
            **kwargs: Additional context information
        """
        # Build context from parameters
        if operation:
            kwargs["operation"] = operation
        if key:
            kwargs["key"] = key
        if context:
            kwargs["memory_context"] = context

        # Remove operation from kwargs since it's a parameter for parent class
        parent_kwargs = {k: v for k, v in kwargs.items() if k != "operation"}
        parent_operation = operation if operation else "unknown"
        super().__init__(message, cause=cause, component="memory", operation=parent_operation, **parent_kwargs)

        # Add operation back to context for backward compatibility
        if operation:
            self.additional_context["operation"] = operation



class StatePersistenceError(AgentError):
    """Error in state persistence.
    
    Raised when there is a failure during state saving or loading.
    """

    def __init__(
        self,
        message: str,
        operation: str,  # "save", "load", "delete", "list"
        task_id: Optional[str] = None,
        cause: Optional[Exception] = None,
        **context: Union[str, int, bool, None]
    ):
        """Initialize state persistence error.
        
        Args:
            message: Error message
            operation: Persistence operation that failed
            task_id: ID of the task being persisted
            cause: Original exception that caused this error
            **context: Additional context information
        """
        self.operation = operation
        self.task_id = task_id

        context["operation"] = operation
        if task_id:
            context["task_id"] = task_id

        # Remove operation from context since it's a parameter for parent class
        parent_context = {k: v for k, v in context.items() if k != "operation"}
        super().__init__(message, cause=cause, component="persistence", operation=operation, **parent_context)

        # Add operation back to context for backward compatibility
        self.additional_context["operation"] = operation


class ProviderError(AgentError):
    """Error in provider operations.
    
    Raised when there is a failure in a provider operation.
    """

    def __init__(
        self,
        message: str,
        provider_name: Optional[str] = None,
        operation: Optional[str] = None,
        cause: Optional[Exception] = None,
        **kwargs: Any
    ):
        """Initialize provider error.
        
        Args:
            message: Error message
            provider_name: Name of the provider that failed
            operation: Provider operation that failed
            cause: Original exception that caused this error
            **kwargs: Additional context information
        """
        if provider_name:
            kwargs["provider_name"] = provider_name
        if operation:
            kwargs["operation"] = operation

        # Remove operation from kwargs since it's a parameter for parent class
        parent_kwargs = {k: v for k, v in kwargs.items() if k != "operation"}
        parent_operation = operation if operation else "unknown"
        super().__init__(message, cause=cause, component="provider", operation=parent_operation, **parent_kwargs)

        # Add operation back to context for backward compatibility
        if operation:
            self.additional_context["operation"] = operation
