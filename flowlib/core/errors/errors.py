"""Base error classes with enhanced error context and management.

This module provides the foundation for the error handling system,
including structured error types, error context, and error management.
"""

import logging
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import (
    Any,
    Protocol,
    TypeVar,
)

from .models import (
    ConfigurationErrorContext,
    ErrorContextData,
    ProviderErrorContext,
    ResourceErrorContext,
    StateErrorContext,
    ValidationErrorDetail,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorContext:
    """Enhanced error context with structured information.

    This class provides:
    1. Structured context information for errors
    2. Clean serialization for logging and reporting
    3. Strict typing with Pydantic models
    """

    def __init__(self, context_data: ErrorContextData):
        """Initialize error context.

        Args:
            context_data: Required error context data
        """
        self._data = context_data

    @classmethod
    def create(
        cls, flow_name: str, error_type: str, error_location: str, component: str, operation: str
    ) -> "ErrorContext":
        """Create a new error context with required data.

        Args:
            flow_name: Name of the flow
            error_type: Type of error
            error_location: Location in code
            component: Component raising error
            operation: Operation being performed

        Returns:
            New ErrorContext instance
        """
        context_data = ErrorContextData(
            flow_name=flow_name,
            error_type=error_type,
            error_location=error_location,
            component=component,
            operation=operation,
        )
        return cls(context_data)

    @property
    def data(self) -> ErrorContextData:
        """Get the context data."""
        return self._data

    @property
    def timestamp(self) -> datetime:
        """Get the context creation timestamp."""
        return self._data.timestamp

    def __str__(self) -> str:
        """String representation."""
        return f"ErrorContext({self._data.model_dump()})"


class BaseError(Exception):
    """Base class for all framework errors with enhanced context.

    This class provides:
    1. Structured error information with context
    2. Clean serialization for logging and reporting
    3. Cause tracking for nested errors
    """

    def __init__(self, message: str, context: ErrorContext, cause: Exception | None = None):
        """Initialize error.

        Args:
            message: Error message
            context: Required error context
            cause: Optional cause exception
        """
        self.message = message
        self.context = context
        self.cause = cause
        self.timestamp = datetime.now()
        self.traceback = self._capture_traceback()
        self.result: Any | None = None  # Can be set by error handlers

        # Initialize with message
        super().__init__(message)

    def _capture_traceback(self) -> str:
        """Capture the current traceback."""
        return traceback.format_exc()

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary.

        Returns:
            Dictionary representation of error
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context.data.model_dump(),
            "cause": str(self.cause) if self.cause else None,
            "traceback": self.traceback,
        }

    def __str__(self) -> str:
        """String representation."""
        cause_str = f" (caused by: {self.cause})" if self.cause else ""
        return f"{self.__class__.__name__}: {self.message}{cause_str}"


class ValidationError(BaseError):
    """Error raised when validation fails.

    This class provides:
    1. Structured validation error information
    2. Clean access to validation error details
    """

    def __init__(
        self,
        message: str,
        validation_errors: list[ValidationErrorDetail],
        context: ErrorContext,
        cause: Exception | None = None,
    ):
        """Initialize validation error.

        Args:
            message: Error message
            validation_errors: List of validation error details
            context: Required error context
            cause: Optional cause exception
        """
        self.validation_errors = validation_errors
        super().__init__(message, context, cause)

    def __str__(self) -> str:
        """String representation."""
        base_str = super().__str__()
        if self.validation_errors:
            errors_str = "; ".join(f"{e.location}: {e.message}" for e in self.validation_errors[:3])
            if len(self.validation_errors) > 3:
                errors_str += f" (and {len(self.validation_errors) - 3} more)"
            return f"{base_str} - {errors_str}"
        return base_str


class ExecutionError(BaseError):
    """Error raised when flow execution fails.

    This class provides:
    1. Structured execution error information
    2. Clean access to execution context
    """

    def __init__(self, message: str, context: ErrorContext, cause: Exception | None = None):
        """Initialize execution error.

        Args:
            message: Error message
            context: Required error context
            cause: Optional cause exception
        """
        super().__init__(message, context, cause)


class StateError(BaseError):
    """Error raised when state operations fail.

    This class provides:
    1. Structured state error information
    2. Clean access to state context
    """

    def __init__(
        self,
        message: str,
        context: ErrorContext,
        state_context: StateErrorContext,
        cause: Exception | None = None,
    ):
        """Initialize state error.

        Args:
            message: Error message
            context: Required error context
            state_context: Required state error context
            cause: Optional cause exception
        """
        self.state_context = state_context
        super().__init__(message, context, cause)


class ConfigurationError(BaseError):
    """Error raised when configuration is invalid.

    This class provides:
    1. Structured configuration error information
    2. Clean access to configuration context
    """

    def __init__(
        self,
        message: str,
        context: ErrorContext,
        config_context: ConfigurationErrorContext,
        cause: Exception | None = None,
    ):
        """Initialize configuration error.

        Args:
            message: Error message
            context: Required error context
            config_context: Required configuration error context
            cause: Optional cause exception
        """
        self.config_context = config_context
        super().__init__(message, context, cause)


class ResourceError(BaseError):
    """Error raised when resource operations fail.

    This class provides:
    1. Structured resource error information
    2. Clean access to resource context
    """

    def __init__(
        self,
        message: str,
        context: ErrorContext,
        resource_context: ResourceErrorContext,
        cause: Exception | None = None,
    ):
        """Initialize resource error.

        Args:
            message: Error message
            context: Required error context
            resource_context: Required resource error context
            cause: Optional cause exception
        """
        self.resource_context = resource_context
        super().__init__(message, context, cause)


class ProviderError(BaseError):
    """Error raised when provider operations fail.

    This class provides:
    1. Structured provider error information
    2. Clean access to provider context
    """

    def __init__(
        self,
        message: str,
        context: ErrorContext,
        provider_context: ProviderErrorContext,
        cause: Exception | None = None,
    ):
        """Initialize provider error.

        Args:
            message: Error message
            context: Required error context
            provider_context: Required provider error context
            cause: Optional cause exception
        """
        self.provider_context = provider_context
        super().__init__(message, context, cause)


# Placeholder for removed duplicate ErrorManager class

ErrorHandlerFunc = Callable[[BaseError, dict[str, Any]], None | dict[str, Any]]
AsyncErrorHandlerFunc = Callable[
    [BaseError, dict[str, Any]], Awaitable[None | dict[str, Any]]
]

ErrorHandler = ErrorHandlerFunc | AsyncErrorHandlerFunc


class ErrorManager:
    """Centralized error management with customizable handlers.

    This class provides:
    1. Error boundary capabilities
    2. Customizable error handlers
    3. Error logging and reporting
    4. Context preservation
    """

    def __init__(self) -> None:
        """Initialize error manager."""
        self._handlers: dict[type[BaseError], list[ErrorHandler]] = {}
        self._global_handlers: list[ErrorHandler] = []

    def register(self, error_type: type[BaseError], handler: ErrorHandler) -> None:
        """Register an error handler for a specific error type.

        Args:
            error_type: Type of error to handle
            handler: Handler function or coroutine
        """
        if error_type not in self._handlers:
            self._handlers[error_type] = []

        self._handlers[error_type].append(handler)

    def register_global(self, handler: ErrorHandler) -> None:
        """Register a global error handler that processes all errors.

        Args:
            handler: Handler function or coroutine
        """
        self._global_handlers.append(handler)

    async def _handle_error(self, error: BaseError, context: dict[str, Any]) -> None:
        """Handle an error with registered handlers.

        Args:
            error: Error to handle
            context: Context data

        Returns:
            Optional handler result
        """
        # Collect all applicable handlers
        handlers = []

        # Add specific handlers for this error type
        for error_type, type_handlers in self._handlers.items():
            if isinstance(error, error_type):
                handlers.extend(type_handlers)

        # Add global handlers
        handlers.extend(self._global_handlers)

        # Execute handlers
        for handler in handlers:
            try:
                import inspect

                if inspect.iscoroutinefunction(handler):
                    # Async handler
                    await handler(error, context)
                else:
                    # Sync handler
                    handler(error, context)

            except Exception as e:
                # Log handler errors but don't propagate
                logger.error(f"Error in error handler: {e}")
                logger.error(traceback.format_exc())

    @asynccontextmanager
    async def error_boundary(
        self, flow_name: str, component: str, operation: str
    ) -> AsyncIterator[None]:
        """Create an error boundary that handles errors with registered handlers.

        Args:
            flow_name: Name of the flow
            component: Component name
            operation: Operation being performed

        Yields:
            None

        Raises:
            BaseError: Propagated after handling
        """
        context = {"flow_name": flow_name, "component": component, "operation": operation}

        try:
            # Yield control to the wrapped code
            yield

        except BaseError as e:
            # Handle framework error
            await self._handle_error(e, context)

            # Re-raise after handling
            raise

        except Exception as e:
            error_context = ErrorContext.create(
                flow_name=flow_name,
                error_type=type(e).__name__,
                error_location=f"{component}.{operation}",
                component=component,
                operation=operation,
            )

            error = ExecutionError(message=str(e), context=error_context, cause=e)

            # Handle converted error
            await self._handle_error(error, context)

            # Re-raise converted error
            raise error from e


# Create default error manager instance
default_manager = ErrorManager()


class LoggingHandler:
    """Error handler that logs errors with configurable verbosity.

    This class provides detailed logging of errors, including
    context information and causal chains.
    """

    def __init__(
        self,
        level: int = logging.ERROR,
        include_context: bool = True,
        include_traceback: bool = True,
        logger_name: str | None = None,
    ):
        """Initialize logging handler.

        Args:
            level: Logging level
            include_context: Whether to include context in logs
            include_traceback: Whether to include traceback in logs
            logger_name: Optional custom logger name
        """
        self.level = level
        self.include_context = include_context
        self.include_traceback = include_traceback
        self.logger = logging.getLogger(logger_name or __name__)

    def __call__(self, error: BaseError, context: dict[str, Any]) -> None:
        """Handle error by logging it.

        Args:
            error: Error to handle
            context: Context data
        """
        # Format error message
        message = f"{type(error).__name__}: {error.message}"

        # Add context if enabled
        if self.include_context and error.context:
            message += f"\nContext: {error.context.data.model_dump()}"

        # Add traceback if enabled
        if self.include_traceback and error.cause:
            cause_tb = "".join(
                traceback.format_exception(
                    type(error.cause), error.cause, error.cause.__traceback__
                )
            )
            message += f"\nCaused by: {cause_tb}"

        # Log the error
        self.logger.log(self.level, message)


class MetricsClient(Protocol):
    """Protocol for metrics clients."""

    def increment(self, metric: str, tags: dict[str, str] | None = None) -> None:
        """Increment a metric counter."""
        ...


class MetricsHandler:
    """Error handler that records error metrics.

    This class provides error tracking for monitoring and alerting.
    """

    def __init__(self, metrics_client: MetricsClient):
        """Initialize metrics handler.

        Args:
            metrics_client: Client for recording metrics
        """
        self.metrics_client = metrics_client

    def __call__(self, error: BaseError, context: dict[str, str | int | bool | None]) -> None:
        """Handle error by recording metrics.

        Args:
            error: Error to handle
            context: Context data
        """
        # Record error count
        self.metrics_client.increment(
            "flow.errors",
            tags={"error_type": type(error).__name__, "flow_name": error.context.data.flow_name},
        )


# Create default logging handler
def default_logging_handler(error: BaseError, context: dict[str, Any]) -> None:
    """Default logging handler for errors.

    Args:
        error: Error to log
        context: Context data
    """
    logger.error(f"{error.__class__.__name__}: {error.message}")
    if error.cause:
        logger.debug(f"Caused by: {error.cause}")
    if error.context:
        logger.debug(f"Context: {error.context.data.model_dump()}")


# Register default handler
default_manager.register_global(default_logging_handler)
