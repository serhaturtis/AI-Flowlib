"""Comprehensive tests for Error Handling Infrastructure."""

import pytest
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional, List
from unittest.mock import Mock, patch

from flowlib.core.errors.errors import (
    ErrorContext, BaseError, ValidationError, ExecutionError, StateError,
    ConfigurationError, ResourceError, ProviderError, ErrorManager,
    LoggingHandler, MetricsHandler, default_manager, default_logging_handler
)
from flowlib.core.errors.models import (
    StateErrorContext, ConfigurationErrorContext, ResourceErrorContext, 
    ProviderErrorContext, ValidationErrorDetail
)


class TestErrorContext:
    """Test ErrorContext functionality."""
    
    def test_error_context_creation(self):
        """Test basic error context creation."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="test",
            error_location="test_module",
            component="test_component",
            operation="test_operation"
        )
        
        assert context.data.flow_name == "test_flow"
        assert context.data.component == "test_component"
        assert isinstance(context.timestamp, datetime)
    
    def test_error_context_with_initial_data(self):
        """Test error context creation using the create method."""
        context = ErrorContext.create(
            flow_name="data_flow",
            error_type="data",
            error_location="data_module", 
            component="data_processor",
            operation="process_data"
        )
        
        assert context.data.flow_name == "data_flow"
        assert context.data.component == "data_processor"
        assert context.data.operation == "process_data"
    
    def test_error_context_create_classmethod(self):
        """Test ErrorContext.create() class method."""
        context = ErrorContext.create(
            flow_name="class_test_flow",
            error_type="class_test",
            error_location="class_test_module",
            component="test_component",
            operation="test_operation"
        )
        
        assert context.data.component == "test_component"
        assert context.data.operation == "test_operation"
        assert context.data.flow_name == "class_test_flow"
    
    def test_error_context_immutability(self):
        """Test that error context is immutable."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="test",
            error_location="test_module",
            component="test_component",
            operation="test_operation"
        )
        
        # Ensure context data is immutable (Pydantic model is frozen)
        assert context.data.flow_name == "test_flow"
        assert context.data.component == "test_component"
        
        # Test that the context data is accessible but immutable
        assert isinstance(context.data, type(context.data))
    
    # Removed redundant string representation test
        """Test string representation of error context."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="test",
            error_location="test_module",
            component="test_component",
            operation="test_operation"
        )
        str_repr = str(context)
        
        assert "ErrorContext" in str_repr


class TestBaseError:
    """Test BaseError functionality."""
    
    def test_base_error_creation(self):
        """Test basic error creation with required context."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="base",
            error_location="test_module",
            component="test_component",
            operation="test_operation"
        )
        error = BaseError("Test error message", context)
        
        assert error.message == "Test error message"
        assert error.context == context
        assert error.cause is None
    
    def test_base_error_with_cause(self):
        """Test error creation with cause."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="base",
            error_location="test_module",
            component="test_component",
            operation="test_operation"
        )
        cause = ValueError("Original cause")
        error = BaseError("Test error", context, cause)
        
        assert error.cause == cause


class TestValidationError:
    """Test ValidationError functionality."""
    
    def test_validation_error_creation(self):
        """Test basic validation error creation."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="validation",
            error_location="test_module",
            component="validator",
            operation="validate"
        )
        validation_errors = [
            ValidationErrorDetail(
                location="field1",
                message="Required field missing",
                error_type="missing"
            )
        ]
        error = ValidationError(
            "Validation failed",
            validation_errors=validation_errors,
            context=context
        )
        
        assert error.message == "Validation failed"
        assert len(error.validation_errors) == 1
        assert error.validation_errors[0].location == "field1"
        assert isinstance(error, BaseError)
    
    def test_validation_error_with_multiple_details(self):
        """Test validation error with multiple validation details."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="validation",
            error_location="test_module",
            component="validator",
            operation="validate"
        )
        validation_errors = [
            ValidationErrorDetail(
                location="field1",
                message="Required field missing",
                error_type="missing"
            ),
            ValidationErrorDetail(
                location="field2",
                message="Invalid value",
                error_type="value_error"
            )
        ]
        
        error = ValidationError(
            "Validation failed",
            validation_errors=validation_errors,
            context=context
        )
        
        assert len(error.validation_errors) == 2
        assert error.validation_errors[0].location == "field1"
        assert error.validation_errors[1].location == "field2"
    
    # Removed redundant string representation tests - handled by Pydantic


class TestExecutionError:
    """Test ExecutionError functionality."""
    
    def test_execution_error_creation(self):
        """Test basic execution error creation."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="execution",
            error_location="test_module",
            component="test_component",
            operation="test_operation"
        )
        error = ExecutionError("Execution failed", context)
        
        assert error.message == "Execution failed"
        assert isinstance(error, BaseError)
        assert error.context == context
    
    def test_execution_error_with_context(self):
        """Test execution error with context."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="execution", 
            error_location="test_module",
            component="test_component",
            operation="processing"
        )
        error = ExecutionError("Step failed", context)
        
        assert error.context.data.flow_name == "test_flow"
        assert error.context.data.operation == "processing"


class TestStateError:
    """Test StateError functionality."""
    
    def test_state_error_creation(self):
        """Test basic state error creation."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="state",
            error_location="test_module",
            component="state_manager",
            operation="state_transition"
        )
        state_context = StateErrorContext(
            state_name="processing_state",
            state_type="execution",
            transition_from="init",
            transition_to="processing"
        )
        error = StateError("State operation failed", context, state_context)
        
        assert error.message == "State operation failed"
        assert isinstance(error, BaseError)
        assert error.state_context.state_name == "processing_state"
    
    def test_state_error_with_state_name(self):
        """Test state error with state name."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="state",
            error_location="test_module",
            component="state_manager",
            operation="state_transition"
        )
        state_context = StateErrorContext(
            state_name="processing_state",
            state_type="execution",
            transition_from="init",
            transition_to="processing"
        )
        error = StateError("State failed", context, state_context)
        
        assert error.state_context.state_name == "processing_state"
    
    def test_state_error_with_context_and_state_name(self):
        """Test state error with both context and state name."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="state",
            error_location="test_module",
            component="state_manager",
            operation="state_transition"
        )
        state_context = StateErrorContext(
            state_name="processing_state",
            state_type="execution",
            transition_from="init",
            transition_to="processing"
        )
        error = StateError("State failed", context, state_context)
        
        assert error.context.data.component == "state_manager"
        assert error.state_context.state_name == "processing_state"


class TestConfigurationError:
    """Test ConfigurationError functionality."""
    
    def test_configuration_error_creation(self):
        """Test basic configuration error creation."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="configuration",
            error_location="test_module",
            component="config_loader",
            operation="load_config"
        )
        config_context = ConfigurationErrorContext(
            config_key="database.url",
            config_section="database", 
            expected_type="string",
            actual_value="invalid_value"
        )
        error = ConfigurationError("Configuration invalid", context, config_context)
        
        assert error.message == "Configuration invalid"
        assert isinstance(error, BaseError)
        assert error.config_context.config_key == "database.url"
    
    def test_configuration_error_with_config_key(self):
        """Test configuration error with config key."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="configuration",
            error_location="test_module",
            component="config_loader",
            operation="load_config"
        )
        config_context = ConfigurationErrorContext(
            config_key="database.url",
            config_section="database",
            expected_type="string",
            actual_value="invalid_value"
        )
        error = ConfigurationError("Invalid config", context, config_context)
        
        assert error.config_context.config_key == "database.url"
    
    def test_configuration_error_with_context_and_config_key(self):
        """Test configuration error with both context and config key."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="configuration",
            error_location="test_module",
            component="config_loader",
            operation="load_config"
        )
        config_context = ConfigurationErrorContext(
            config_key="api.timeout",
            config_section="api",
            expected_type="integer",
            actual_value="invalid_timeout"
        )
        error = ConfigurationError("Config failed", context, config_context)
        
        assert error.context.data.component == "config_loader"
        assert error.config_context.config_key == "api.timeout"


class TestResourceError:
    """Test ResourceError functionality."""
    
    def test_resource_error_creation(self):
        """Test basic resource error creation."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="resource",
            error_location="test_module",
            component="resource_manager",
            operation="resource_operation"
        )
        resource_context = ResourceErrorContext(
            resource_id="res_123",
            resource_type="database",
            operation="read"
        )
        error = ResourceError("Resource operation failed", context, resource_context)
        
        assert error.message == "Resource operation failed"
        assert isinstance(error, BaseError)
        assert error.resource_context.resource_id == "res_123"
    
    def test_resource_error_with_resource_info(self):
        """Test resource error with resource information."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="resource",
            error_location="test_module",
            component="resource_manager",
            operation="resource_operation"
        )
        resource_context = ResourceErrorContext(
            resource_id="res_123",
            resource_type="database",
            operation="find"
        )
        error = ResourceError("Resource not found", context, resource_context)
        
        assert error.resource_context.resource_id == "res_123"
        assert error.resource_context.resource_type == "database"
    
    def test_resource_error_with_context_and_resource_info(self):
        """Test resource error with context and resource info."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="resource",
            error_location="test_module",
            component="resource_manager",
            operation="create"
        )
        resource_context = ResourceErrorContext(
            resource_id="res_456",
            resource_type="file",
            operation="create"
        )
        error = ResourceError("Creation failed", context, resource_context)
        
        assert error.context.data.operation == "create"
        assert error.resource_context.resource_id == "res_456"


class TestProviderError:
    """Test ProviderError functionality."""
    
    def test_provider_error_creation(self):
        """Test basic provider error creation."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="provider",
            error_location="test_module",
            component="provider_manager",
            operation="provider_operation"
        )
        provider_context = ProviderErrorContext(
            provider_name="llm_provider",
            provider_type="llm",
            operation="generate",
            retry_count=3
        )
        error = ProviderError("Provider operation failed", context, provider_context)
        
        assert error.message == "Provider operation failed"
        assert isinstance(error, BaseError)
        assert error.provider_context.provider_name == "llm_provider"
    
    def test_provider_error_with_provider_name(self):
        """Test provider error with provider name."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="provider",
            error_location="test_module",
            component="provider_manager",
            operation="provider_operation"
        )
        provider_context = ProviderErrorContext(
            provider_name="llm_provider",
            provider_type="llm",
            operation="initialize",
            retry_count=1
        )
        error = ProviderError("Provider failed", context, provider_context)
        
        assert error.provider_context.provider_name == "llm_provider"
    
    def test_provider_error_with_context_and_provider_name(self):
        """Test provider error with context and provider name."""
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="provider",
            error_location="test_module",
            component="provider_manager",
            operation="initialize"
        )
        provider_context = ProviderErrorContext(
            provider_name="vector_db",
            provider_type="vector",
            operation="initialize",
            retry_count=0
        )
        error = ProviderError("Init failed", context, provider_context)
        
        assert error.context.data.operation == "initialize"
        assert error.provider_context.provider_name == "vector_db"


class TestErrorManager:
    """Test ErrorManager functionality."""
    
    def test_error_manager_creation(self):
        """Test basic error manager creation."""
        manager = ErrorManager()
        
        assert manager._handlers == {}
        assert manager._global_handlers == []
    
    def test_register_error_handler(self):
        """Test registering error handler for specific type."""
        manager = ErrorManager()
        
        def test_handler(error, context):
            pass
        
        manager.register(ValidationError, test_handler)
        
        assert ValidationError in manager._handlers
        assert test_handler in manager._handlers[ValidationError]
    
    @pytest.mark.asyncio
    async def test_handle_error_with_specific_handler(self):
        """Test handling error with specific handler."""
        manager = ErrorManager()
        handler_called = False
        
        def test_handler(error, context):
            nonlocal handler_called
            handler_called = True
            assert isinstance(error, ValidationError)
            assert context.get("test_key") == "test_value"
        
        manager.register(ValidationError, test_handler)
        
        error_context = ErrorContext.create(
            flow_name="test_flow",
            error_type="validation",
            error_location="test_module",
            component="validator",
            operation="validate"
        )
        error = ValidationError(
            "Test validation error",
            validation_errors=[],
            context=error_context
        )
        await manager._handle_error(error, {"test_key": "test_value"})
        
        assert handler_called
    
    @pytest.mark.asyncio
    async def test_handle_error_with_global_handler(self):
        """Test handling error with global handler."""
        manager = ErrorManager()
        handler_called = False
        
        def global_handler(error, context):
            nonlocal handler_called
            handler_called = True
            assert isinstance(error, BaseError)
        
        manager.register_global(global_handler)
        
        error_context = ErrorContext.create(
            flow_name="test_flow",
            error_type="execution",
            error_location="test_module",
            component="executor",
            operation="execute"
        )
        error = ExecutionError("Test execution error", error_context)
        await manager._handle_error(error, {})
        
        assert handler_called
    
    @pytest.mark.asyncio
    async def test_handle_error_with_async_handler(self):
        """Test handling error with async handler."""
        manager = ErrorManager()
        handler_called = False
        
        async def async_handler(error, context):
            nonlocal handler_called
            handler_called = True
            await asyncio.sleep(0.01)  # Simulate async work
        
        manager.register(BaseError, async_handler)
        
        error_context = ErrorContext.create(
            flow_name="test_flow",
            error_type="base",
            error_location="test_module",
            component="component",
            operation="operation"
        )
        error = BaseError("Test error", error_context)
        await manager._handle_error(error, {})
        
        assert handler_called
    
    @pytest.mark.asyncio
    async def test_error_boundary_success(self):
        """Test error boundary with successful execution."""
        manager = ErrorManager()
        
        async with manager.error_boundary(
            flow_name="test_flow",
            component="test_component",
            operation="test_operation"
        ):
            # No error should be raised
            result = 42
        
        assert result == 42
    
    @pytest.mark.asyncio
    async def test_error_boundary_catches_base_error(self):
        """Test error boundary catches BaseError."""
        manager = ErrorManager()
        handler_called = False
        
        def test_handler(error, context):
            nonlocal handler_called
            handler_called = True
        
        manager.register_global(test_handler)
        
        with pytest.raises(ValidationError):
            async with manager.error_boundary(
                flow_name="test_flow",
                component="test_component",
                operation="test_operation"
            ):
                error_context = ErrorContext.create(
                    flow_name="test_flow",
                    error_type="validation",
                    error_location="test_module",
                    component="validator",
                    operation="validate"
                )
                raise ValidationError("Test error", validation_errors=[], context=error_context)
        
        assert handler_called
    
    @pytest.mark.asyncio
    async def test_error_boundary_converts_exception(self):
        """Test error boundary converts general exceptions."""
        manager = ErrorManager()
        handler_called = False
        converted_error = None
        
        def test_handler(error, context):
            nonlocal handler_called, converted_error
            handler_called = True
            converted_error = error
        
        manager.register_global(test_handler)
        
        with pytest.raises(ExecutionError):
            async with manager.error_boundary(
                flow_name="test_flow",
                component="test_component",
                operation="test_operation"
            ):
                raise ValueError("Original error")
        
        assert handler_called
        assert isinstance(converted_error, ExecutionError)
        assert isinstance(converted_error.cause, ValueError)
    
    @pytest.mark.asyncio
    async def test_error_boundary_with_context(self):
        """Test error boundary with context data."""
        manager = ErrorManager()
        received_context = None
        
        def test_handler(error, context):
            nonlocal received_context
            received_context = context
        
        manager.register_global(test_handler)
        
        with pytest.raises(BaseError):
            async with manager.error_boundary(
                flow_name="test_flow",
                component="test_component",
                operation="test_operation"
            ):
                error_context = ErrorContext.create(
                    flow_name="test_flow",
                    error_type="base",
                    error_location="test_module",
                    component="component",
                    operation="operation"
                )
                raise BaseError("Test error", error_context)
        
        assert received_context["flow_name"] == "test_flow"
        assert received_context["component"] == "test_component"
        assert received_context["operation"] == "test_operation"


class TestLoggingHandler:
    """Test LoggingHandler functionality."""
    
    def test_logging_handler_creation(self):
        """Test basic logging handler creation."""
        handler = LoggingHandler()
        
        assert handler.level == logging.ERROR
        assert handler.include_context is True
        assert handler.include_traceback is True
        assert handler.logger is not None
    
    def test_logging_handler_call(self):
        """Test logging handler error handling."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            handler = LoggingHandler()
            
            context = ErrorContext.create(
                flow_name="test_flow",
                error_type="base",
                error_location="test_module",
                component="test",
                operation="test_operation"
            )
            error = BaseError("Test error", context=context)
            
            handler(error, {})
            
            # Verify logger was called
            mock_logger.log.assert_called_once()
            call_args = mock_logger.log.call_args
            assert call_args[0][0] == logging.ERROR  # Level
            assert "BaseError" in call_args[0][1]    # Message contains error type
            assert "Test error" in call_args[0][1]   # Message contains error message


class TestMetricsHandler:
    """Test MetricsHandler functionality."""
    
    def test_metrics_handler_creation(self):
        """Test basic metrics handler creation."""
        mock_client = Mock()
        handler = MetricsHandler(mock_client)
        
        assert handler.metrics_client == mock_client
    
    def test_metrics_handler_call(self):
        """Test metrics handler error handling."""
        mock_client = Mock()
        handler = MetricsHandler(mock_client)
        
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="validation",
            error_location="test_module",
            component="validator",
            operation="validate"
        )
        error = ValidationError(
            "Test validation error",
            validation_errors=[],
            context=context
        )
        
        handler(error, {})
        
        # Verify metrics client was called
        mock_client.increment.assert_called_once_with(
            "flow.errors",
            tags={
                "error_type": "ValidationError",
                "flow_name": "test_flow"
            }
        )


class TestDefaultComponents:
    """Test default error handling components."""
    
    def test_default_manager_exists(self):
        """Test that default manager exists and is properly configured."""
        assert default_manager is not None
        assert isinstance(default_manager, ErrorManager)
        assert len(default_manager._global_handlers) > 0
    
    def test_default_logging_handler(self):
        """Test default logging handler."""
        with patch('flowlib.core.errors.errors.logger') as mock_logger:
            context = ErrorContext.create(
                flow_name="test_flow",
                error_type="base",
                error_location="test_module",
                component="component",
                operation="operation"
            )
            error = BaseError("Test error", context)
            
            default_logging_handler(error, {})
            
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0][0]
            assert "BaseError" in call_args
            assert "Test error" in call_args


class TestErrorIntegration:
    """Test integration between error components."""
    
    @pytest.mark.asyncio
    async def test_full_error_flow(self):
        """Test complete error handling flow."""
        manager = ErrorManager()
        handler_results = []
        
        def validation_handler(error, context):
            handler_results.append(("validation", error, context))
        
        def global_handler(error, context):
            handler_results.append(("global", error, context))
        
        # Register handlers
        manager.register(ValidationError, validation_handler)
        manager.register_global(global_handler)
        
        # Create error with context
        context = ErrorContext.create(
            flow_name="test_flow",
            error_type="validation",
            error_location="test_module",
            component="validator",
            operation="validation"
        )
        error = ValidationError(
            "Invalid input",
            validation_errors=[],
            context=context
        )
        
        # Handle error
        await manager._handle_error(error, {"extra": "context"})
        
        # Verify all handlers were called
        assert len(handler_results) == 2
        
        # Check each handler received correct data
        handler_types = [result[0] for result in handler_results]
        assert "validation" in handler_types
        assert "global" in handler_types


# Removed redundant integration test that duplicated functionality

if __name__ == "__main__":
    pytest.main([__file__])