"""
Error boundary system for GUI services following flowlib patterns.

Provides structured error handling, context preservation, and proper error propagation.
Now uses the unified core error system for consistency.
"""

import logging
import traceback
from typing import Any, Dict, Optional, Callable, Type, Union
from datetime import datetime
from contextlib import contextmanager
from functools import wraps

from .models import OperationResult
from flowlib.core.errors.errors import (
    BaseError, 
    ErrorContext as CoreErrorContext,
    ConfigurationError as CoreConfigurationError,
    ProviderError as CoreProviderError,
    ValidationError as CoreValidationError
)

logger = logging.getLogger(__name__)


class FlowlibError(BaseError):
    """Base exception for flowlib-related errors.
    
    Now inherits from the unified core error system.
    """
    
    def __init__(self, message: str, context: Optional[dict[str, Union[str, int, float, bool]]] = None, 
                 error_code: Optional[str] = None, component: str = "gui", 
                 operation: str = "unknown"):
        # Create core error context
        error_context = CoreErrorContext.create(
            flow_name=(context["flow_name"] if "flow_name" in context else "gui") if context else "gui",
            error_type=self.__class__.__name__,
            error_location=f"{component}.{operation}",
            component=component,
            operation=operation
        )
        
        # Call parent constructor
        super().__init__(message, error_context, None)
        
        # Keep additional GUI-specific attributes
        self.error_code = error_code
        self.gui_context = context or {}


class ConfigurationError(CoreConfigurationError):
    """GUI Configuration-related errors."""
    pass


class ProviderError(CoreProviderError):
    """GUI Provider-related errors."""
    pass


class ServiceError(FlowlibError):
    """GUI Service-related errors."""
    pass


class ValidationError(CoreValidationError):
    """GUI Validation errors."""
    pass


class ErrorBoundary:
    """Error boundary for capturing and handling errors in specific contexts."""
    
    def __init__(self, context_name: str):
        self.context_name = context_name
        self.errors = []
    
    def capture_error(self, error: Exception, additional_context: Optional[dict[str, Union[str, int, float, bool]]] = None):
        """Capture an error with context."""
        error_info = {
            'error': error,
            'context_name': self.context_name,
            'additional_context': additional_context or {},
            'timestamp': datetime.now(),
            'traceback': traceback.format_exc()
        }
        
        self.errors.append(error_info)
        logger.error(f"Error captured in {self.context_name}: {error}", exc_info=True)
    
    def has_errors(self) -> bool:
        """Check if any errors were captured."""
        return len(self.errors) > 0
    
    def get_errors(self) -> list:
        """Get all captured errors."""
        return self.errors.copy()
    
    def get_error_summary(self) -> dict[str, Union[str, int, float, bool]]:
        """Get summary of all errors."""
        return {
            'context': self.context_name,
            'error_count': len(self.errors),
            'errors': [
                {
                    'type': type(err['error']).__name__,
                    'message': str(err['error']),
                    'timestamp': err['timestamp'].isoformat(),
                    'context': err['additional_context']
                } for err in self.errors
            ]
        }


@contextmanager
def error_boundary(context_name: str):
    """Context manager for error boundary."""
    boundary = ErrorBoundary(context_name)
    try:
        yield boundary
    except Exception as e:
        boundary.capture_error(e)
        raise


def handle_service_errors(operation_name: str, 
                         fallback_result: Any = None,
                         raise_on_error: bool = False):
    """Decorator for handling service errors with proper context."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except FlowlibError as e:
                logger.error(f"Flowlib error in {operation_name}: {e.message}", 
                           extra={'context': e.context})
                if raise_on_error:
                    raise
                return fallback_result or OperationResult(
                    success=False, 
                    message=e.message,
                    data={'error_code': e.error_code, 'context': e.context}
                )
            except Exception as e:
                logger.error(f"Unexpected error in {operation_name}: {e}", exc_info=True)
                if raise_on_error:
                    raise
                return fallback_result or OperationResult(
                    success=False,
                    message=f"Unexpected error: {str(e)}",
                    data={'error_type': type(e).__name__}
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FlowlibError as e:
                logger.error(f"Flowlib error in {operation_name}: {e.message}", 
                           extra={'context': e.context})
                if raise_on_error:
                    raise
                return fallback_result or OperationResult(
                    success=False, 
                    message=e.message,
                    data={'error_code': e.error_code, 'context': e.context}
                )
            except Exception as e:
                logger.error(f"Unexpected error in {operation_name}: {e}", exc_info=True)
                if raise_on_error:
                    raise
                return fallback_result or OperationResult(
                    success=False,
                    message=f"Unexpected error: {str(e)}",
                    data={'error_type': type(e).__name__}
                )
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def validate_input(validation_func: Callable, error_message: str = "Invalid input"):
    """Decorator for input validation."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                if not validation_func(*args, **kwargs):
                    raise ValidationError(error_message, context={'args': args, 'kwargs': kwargs})
                return await func(*args, **kwargs)
            except ValidationError:
                raise
            except Exception as e:
                raise ValidationError(f"Validation failed: {str(e)}", context={'original_error': str(e)})
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                if not validation_func(*args, **kwargs):
                    raise ValidationError(error_message, context={'args': args, 'kwargs': kwargs})
                return func(*args, **kwargs)
            except ValidationError:
                raise
            except Exception as e:
                raise ValidationError(f"Validation failed: {str(e)}", context={'original_error': str(e)})
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class ErrorReporter:
    """Error reporter for collecting and reporting errors."""
    
    def __init__(self):
        self.errors = []
    
    def report_error(self, error: Union[Exception, FlowlibError], 
                    context: Optional[dict[str, Union[str, int, float, bool]]] = None):
        """Report an error with context."""
        error_info = {
            'error': error,
            'context': context or {},
            'timestamp': datetime.now(),
            'traceback': traceback.format_exc() if not isinstance(error, FlowlibError) else None
        }
        
        self.errors.append(error_info)
        
        # Log based on error type
        if isinstance(error, FlowlibError):
            logger.error(f"Flowlib error: {error.message}", extra={'context': error.context})
        else:
            logger.error(f"Unexpected error: {error}", exc_info=True)
    
    def get_error_report(self) -> dict[str, Union[str, int, float, bool]]:
        """Get comprehensive error report."""
        return {
            'total_errors': len(self.errors),
            'error_types': {
                error_type: len([e for e in self.errors if type(e['error']).__name__ == error_type])
                for error_type in set(type(e['error']).__name__ for e in self.errors)
            },
            'recent_errors': [
                {
                    'type': type(err['error']).__name__,
                    'message': str(err['error']),
                    'timestamp': err['timestamp'].isoformat(),
                    'context': err['context']
                } for err in self.errors[-10:]  # Last 10 errors
            ]
        }
    
    def clear_errors(self):
        """Clear all reported errors."""
        self.errors.clear()


# Global error reporter instance
error_reporter = ErrorReporter()


def create_error_context(operation_name: str, **context_data) -> dict[str, Union[str, int, float, bool]]:
    """Create error context for consistent error reporting."""
    return {
        'operation': operation_name,
        'timestamp': datetime.now().isoformat(),
        **context_data
    }


def handle_provider_error(provider_name: str, config_name: str, original_error: Exception) -> ProviderError:
    """Handle provider-specific errors."""
    context = create_error_context(
        'provider_access',
        provider_name=provider_name,
        config_name=config_name,
        original_error=str(original_error)
    )
    
    return ProviderError(
        f"Failed to access provider '{provider_name}' with config '{config_name}': {original_error}",
        context=context,
        error_code='PROVIDER_ACCESS_FAILED'
    )


def handle_configuration_error(config_name: str, operation: str, original_error: Exception) -> ConfigurationError:
    """Handle configuration-specific errors."""
    context = create_error_context(
        'configuration_operation',
        config_name=config_name,
        operation=operation,
        original_error=str(original_error)
    )
    
    return ConfigurationError(
        f"Configuration '{config_name}' {operation} failed: {original_error}",
        context=context,
        error_code='CONFIGURATION_OPERATION_FAILED'
    )


def handle_service_error(service_name: str, operation: str, original_error: Exception) -> ServiceError:
    """Handle service-specific errors."""
    context = create_error_context(
        'service_operation',
        service_name=service_name,
        operation=operation,
        original_error=str(original_error)
    )
    
    return ServiceError(
        f"Service '{service_name}' {operation} failed: {original_error}",
        context=context,
        error_code='SERVICE_OPERATION_FAILED'
    )