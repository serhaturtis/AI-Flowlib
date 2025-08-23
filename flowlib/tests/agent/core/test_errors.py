"""Tests for agent error classes."""

import pytest
from flowlib.agent.core.errors import (
    AgentError,
    NotInitializedError,
    ComponentError,
    ConfigurationError,
    ExecutionError,
    PlanningError,
    ReflectionError,
    MemoryError,
    StatePersistenceError,
    ProviderError
)


class TestAgentErrors:
    """Test agent error classes."""
    
    def test_base_agent_error(self):
        """Test base AgentError."""
        error = AgentError("Test error", test_context="value")
        
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.context == {"test_context": "value"}
        assert error.cause is None
    
    def test_agent_error_with_cause(self):
        """Test AgentError with cause."""
        cause = ValueError("Original error")
        error = AgentError("Wrapped error", cause=cause)
        
        assert "Wrapped error" in str(error)
        assert "Original error" in str(error)
        assert error.cause is cause
    
    def test_agent_error_to_dict(self):
        """Test converting error to dictionary."""
        cause = ValueError("Cause")
        error = AgentError(
            "Test error",
            cause=cause,
            key1="value1",
            key2="value2"
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error_type"] == "AgentError"
        assert error_dict["message"] == "Test error"
        assert error_dict["context"]["key1"] == "value1"
        assert error_dict["context"]["key2"] == "value2"
        assert error_dict["cause"]["error_type"] == "ValueError"
        assert error_dict["cause"]["message"] == "Cause"
    
    def test_not_initialized_error(self):
        """Test NotInitializedError."""
        error = NotInitializedError(
            component_name="TestComponent",
            operation="test_operation"
        )
        
        assert "TestComponent" in str(error)
        assert "must be initialized" in str(error)
        assert "test_operation" in str(error)
        assert error.context["component_name"] == "TestComponent"
        assert error.context["operation"] == "test_operation"
    
    def test_component_error(self):
        """Test ComponentError."""
        cause = RuntimeError("Component failed")
        error = ComponentError(
            "Failed to initialize",
            component_name="MemoryComponent",
            operation="initialize",
            cause=cause
        )
        
        assert error.message == "Failed to initialize"
        assert error.context["component_name"] == "MemoryComponent"
        assert error.context["operation"] == "initialize"
        assert error.cause is cause
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError(
            "Invalid configuration",
            config_key="llm_provider",
            invalid_value="unknown",
            required_type=str
        )
        
        assert error.message == "Invalid configuration"
        assert error.context["config_key"] == "llm_provider"
        assert error.context["invalid_value"] == "unknown"
        assert "str" in error.context["required_type"]
    
    def test_execution_error(self):
        """Test ExecutionError."""
        class MockState:
            task_id = "task_123"
            is_complete = False
            progress = 50
        
        error = ExecutionError(
            "Execution failed",
            agent="test_agent",
            state=MockState(),
            flow="test_flow",
            stage="planning"
        )
        
        assert error.message == "Execution failed"
        assert error.context["agent"] == "test_agent"
        assert error.context["flow"] == "test_flow"
        assert error.context["stage"] == "planning"
        assert error.context["task_id"] == "task_123"
        assert error.context["is_complete"] is False
        assert error.context["progress"] == 50
    
    def test_planning_error(self):
        """Test PlanningError."""
        error = PlanningError(
            "Failed to generate plan",
            planning_type="input_generation",
            agent="planner"
        )
        
        assert error.message == "Failed to generate plan"
        assert error.context["planning_type"] == "input_generation"
        assert error.context["agent"] == "planner"
        assert error.context["stage"] == "input_generation"
    
    def test_reflection_error(self):
        """Test ReflectionError."""
        error = ReflectionError(
            "Reflection failed",
            agent="reflector"
        )
        
        assert error.message == "Reflection failed"
        assert error.context["agent"] == "reflector"
        assert error.context["stage"] == "reflection"
    
    def test_memory_error(self):
        """Test MemoryError."""
        error = MemoryError(
            "Failed to store",
            operation="store",
            key="test_key",
            context="test_context"
        )
        
        assert error.message == "Failed to store"
        assert error.context["operation"] == "store"
        assert error.context["key"] == "test_key"
        assert error.context["memory_context"] == "test_context"
    
    def test_state_persistence_error(self):
        """Test StatePersistenceError."""
        error = StatePersistenceError(
            "Failed to save state",
            operation="save",
            task_id="task_456"
        )
        
        assert error.message == "Failed to save state"
        assert error.context["operation"] == "save"
        assert error.context["task_id"] == "task_456"
    
    def test_provider_error(self):
        """Test ProviderError."""
        cause = ConnectionError("Connection failed")
        error = ProviderError(
            "Provider operation failed",
            provider_name="openai",
            operation="generate",
            cause=cause
        )
        
        assert error.message == "Provider operation failed"
        assert error.context["provider_name"] == "openai"
        assert error.context["operation"] == "generate"
        assert error.cause is cause
    
    def test_error_inheritance(self):
        """Test error inheritance relationships."""
        # All errors should inherit from AgentError
        assert issubclass(ComponentError, AgentError)
        assert issubclass(ConfigurationError, AgentError)
        assert issubclass(ExecutionError, AgentError)
        assert issubclass(MemoryError, AgentError)
        assert issubclass(ProviderError, AgentError)
        
        # Planning and Reflection inherit from ExecutionError
        assert issubclass(PlanningError, ExecutionError)
        assert issubclass(ReflectionError, ExecutionError)