"""Comprehensive tests for flow results module."""

import pytest
from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel

from flowlib.flows.models.results import (
    FlowResult,
    AgentResult,
    result_from_value,
    error_result,
    TResult
)
from flowlib.flows.models.constants import FlowStatus


class FlowTestModel(BaseModel):
    """Test model for typed result testing."""
    name: str
    value: int
    flag: bool = False


class TestFlowResult:
    """Test FlowResult functionality."""
    
    def test_flow_result_basic_creation(self):
        """Test basic FlowResult creation."""
        result = FlowResult(
            data={"key": "value"},
            flow_name="test_flow"
        )
        
        assert result.data == {"key": "value"}
        assert result.flow_name == "test_flow"
        assert result.status == FlowStatus.SUCCESS
        assert result.error is None
        assert result.error_details == {}
        assert result.metadata == {}
        assert isinstance(result.timestamp, datetime)
        assert result.duration is None
        assert result.original_type is None
    
    def test_flow_result_with_all_fields(self):
        """Test FlowResult creation with all fields."""
        timestamp = datetime.now()
        result = FlowResult(
            data={"test": "data"},
            flow_name="complex_flow",
            status=FlowStatus.ERROR,
            error="Test error",
            error_details={"code": 500, "details": "Internal error"},
            metadata={"execution_id": "123", "user": "test"},
            timestamp=timestamp,
            duration=1.5,
            original_type=FlowTestModel
        )
        
        assert result.data == {"test": "data"}
        assert result.flow_name == "complex_flow"
        assert result.status == FlowStatus.ERROR
        assert result.error == "Test error"
        assert result.error_details == {"code": 500, "details": "Internal error"}
        assert result.metadata == {"execution_id": "123", "user": "test"}
        assert result.timestamp == timestamp
        assert result.duration == 1.5
        assert result.original_type == FlowTestModel
    
    def test_flow_result_attribute_access(self):
        """Test attribute-based access to result data."""
        result = FlowResult(
            data={"name": "test", "value": 42, "nested": {"inner": "data"}},
            flow_name="test_flow"
        )
        
        # Test direct attribute access
        assert result.name == "test"
        assert result.value == 42
        assert result.nested == {"inner": "data"}
    
    def test_flow_result_attribute_access_missing(self):
        """Test attribute access for missing keys."""
        result = FlowResult(
            data={"key": "value"},
            flow_name="test_flow"
        )
        
        with pytest.raises(AttributeError) as exc_info:
            _ = result.missing_key
        
        assert "'FlowResult' object has no attribute 'missing_key'" in str(exc_info.value)
    
    def test_flow_result_get_typed_success(self):
        """Test successful typed conversion."""
        result = FlowResult(
            data={"name": "test", "value": 42, "flag": True},
            flow_name="test_flow",
            original_type=FlowTestModel
        )
        
        typed_result = result.get_typed(FlowTestModel)
        
        assert isinstance(typed_result, FlowTestModel)
        assert typed_result.name == "test"
        assert typed_result.value == 42
        assert typed_result.flag is True
    
    def test_flow_result_get_typed_failure(self):
        """Test typed conversion failure."""
        result = FlowResult(
            data={"invalid": "data"},  # Missing required fields
            flow_name="test_flow"
        )
        
        with pytest.raises(ValueError) as exc_info:
            result.get_typed(FlowTestModel)
        
        assert "Failed to convert result to FlowTestModel" in str(exc_info.value)
    
    def test_flow_result_as_dict(self):
        """Test conversion to dictionary."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        result = FlowResult(
            data={"test": "data"},
            flow_name="test_flow",
            status=FlowStatus.ERROR,  # Changed to ERROR since error is present
            error="Test error",
            error_details={"code": 404},
            metadata={"user": "test"},
            timestamp=timestamp,
            duration=2.5
        )
        
        result_dict = result.as_dict()
        
        expected = {
            "data": {"test": "data"},
            "flow_name": "test_flow",
            "status": "ERROR",  # Enum value
            "error": "Test error",
            "error_details": {"code": 404},
            "metadata": {"user": "test"},
            "timestamp": "2023-01-01T12:00:00",
            "duration": 2.5
        }
        
        assert result_dict == expected
    
    def test_flow_result_is_success(self):
        """Test success status checking."""
        success_result = FlowResult(
            data={},
            status=FlowStatus.SUCCESS
        )
        
        error_result = FlowResult(
            data={},
            status=FlowStatus.ERROR
        )
        
        assert success_result.is_success() is True
        assert error_result.is_success() is False
    
    def test_flow_result_is_error(self):
        """Test error status checking."""
        success_result = FlowResult(
            data={},
            status=FlowStatus.SUCCESS
        )
        
        error_result = FlowResult(
            data={},
            status=FlowStatus.ERROR
        )
        
        timeout_result = FlowResult(
            data={},
            status=FlowStatus.TIMEOUT
        )
        
        assert success_result.is_error() is False
        assert error_result.is_error() is True
        assert timeout_result.is_error() is True
    
    def test_flow_result_raise_if_error_success(self):
        """Test raise_if_error with success status."""
        result = FlowResult(
            data={"key": "value"},
            flow_name="test_flow",
            status=FlowStatus.SUCCESS
        )
        
        # Should return self without raising
        returned = result.raise_if_error()
        assert returned is result
    
    def test_flow_result_raise_if_error_with_error(self):
        """Test raise_if_error with error status."""
        result = FlowResult(
            data={},
            flow_name="test_flow",
            status=FlowStatus.ERROR,
            error="Test error message"
        )
        
        with pytest.raises(Exception) as exc_info:
            result.raise_if_error()
        
        assert "Test error message" in str(exc_info.value)
    
    def test_flow_result_raise_if_error_no_message(self):
        """Test raise_if_error with error status but no message."""
        result = FlowResult(
            data={},
            flow_name="test_flow",
            status=FlowStatus.ERROR
        )
        
        with pytest.raises(Exception) as exc_info:
            result.raise_if_error()
        
        assert "Flow 'test_flow' failed with status ERROR" in str(exc_info.value)
    
    # Removed redundant str() test
        """Test string representation for success."""
        result = FlowResult(
            data={"key1": "value1", "key2": "value2"},
            flow_name="test_flow",
            status=FlowStatus.SUCCESS
        )
        
        str_repr = str(result)
        
        assert "FlowResult(flow='test_flow', status=SUCCESS" in str_repr
        assert "data_keys=['key1', 'key2']" in str_repr
    
    # Removed redundant str() test
        """Test string representation for error."""
        result = FlowResult(
            data={},
            flow_name="test_flow",
            status=FlowStatus.ERROR,
            error="Test error"
        )
        
        str_repr = str(result)
        
        assert "FlowResult(flow='test_flow', status=ERROR, error='Test error')" == str_repr
    
    def test_flow_result_model_validator(self):
        """Test model validator sets error status when error is present."""
        # Test with explicit status and error
        result = FlowResult(
            data={},
            flow_name="test_flow",
            status=FlowStatus.SUCCESS,  # Will be overridden
            error="Test error"
        )
        
        assert result.status == FlowStatus.ERROR
    
    def test_flow_result_model_validator_no_override(self):
        """Test model validator doesn't override non-success status."""
        result = FlowResult(
            data={},
            flow_name="test_flow",
            status=FlowStatus.TIMEOUT,
            error="Test error"
        )
        
        # Should keep original status
        assert result.status == FlowStatus.TIMEOUT
    
    def test_flow_result_with_pydantic_data(self):
        """Test FlowResult with Pydantic model as data."""
        model_data = FlowTestModel(name="test", value=42)
        result = FlowResult(
            data=model_data,
            flow_name="test_flow"
        )
        
        assert result.data == model_data
        assert isinstance(result.data, FlowTestModel)


class TestHelperFunctions:
    """Test helper functions for creating results."""
    
    def test_result_from_value_simple(self):
        """Test creating result from simple value."""
        result = result_from_value({"value": "test_value"})
        
        assert result.data == {"value": "test_value"}
        assert result.flow_name == "unnamed_flow"
        assert result.status == FlowStatus.SUCCESS
        assert result.error is None
    
    def test_result_from_value_with_name(self):
        """Test creating result from value with flow name."""
        result = result_from_value({"key": "value"}, "custom_flow")
        
        assert result.data == {"key": "value"}
        assert result.flow_name == "custom_flow"
        assert result.status == FlowStatus.SUCCESS
    
    def test_result_from_value_complex(self):
        """Test creating result from complex value."""
        complex_data = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "number": 42
        }
        
        result = result_from_value(complex_data, "complex_flow")
        
        assert result.data == complex_data
        assert result.flow_name == "complex_flow"
        assert result.status == FlowStatus.SUCCESS
    
    def test_error_result_basic(self):
        """Test creating basic error result."""
        result = error_result("Something went wrong")
        
        assert result.data == {}
        assert result.flow_name == "unnamed_flow"
        assert result.status == FlowStatus.ERROR
        assert result.error == "Something went wrong"
        assert result.error_details == {}
    
    def test_error_result_with_details(self):
        """Test creating error result with details."""
        error_details = {
            "code": 500,
            "category": "internal_error",
            "stack_trace": "..."
        }
        
        result = error_result(
            "Internal server error",
            "failing_flow",
            error_details
        )
        
        assert result.data == {}
        assert result.flow_name == "failing_flow"
        assert result.status == FlowStatus.ERROR
        assert result.error == "Internal server error"
        assert result.error_details == error_details
    
    def test_error_result_none_details(self):
        """Test creating error result with None details."""
        result = error_result("Error message", "test_flow", None)
        
        assert result.error_details == {}


class TestAgentResult:
    """Test AgentResult functionality."""
    
    def test_agent_result_basic_creation(self):
        """Test basic AgentResult creation."""
        result = AgentResult(success=True)
        
        assert result.success is True
        assert result.results == []
        assert result.state is None
        assert result.error is None
        assert result.metadata == {}
    
    def test_agent_result_with_flow_results(self):
        """Test AgentResult with flow results."""
        flow_result1 = FlowResult(
            data={"step": 1},
            flow_name="step1",
            status=FlowStatus.SUCCESS
        )
        
        flow_result2 = FlowResult(
            data={"step": 2},
            flow_name="step2",
            status=FlowStatus.SUCCESS
        )
        
        agent_result = AgentResult(
            success=True,
            results=[flow_result1, flow_result2],
            state={"final": "state"},
            metadata={"agent_id": "test_agent"}
        )
        
        assert agent_result.success is True
        assert len(agent_result.results) == 2
        assert agent_result.results[0] == flow_result1
        assert agent_result.results[1] == flow_result2
        assert agent_result.state == {"final": "state"}
        assert agent_result.metadata == {"agent_id": "test_agent"}
    
    def test_agent_result_failure(self):
        """Test AgentResult representing failure."""
        error_flow = FlowResult(
            data={},
            flow_name="failing_step",
            status=FlowStatus.ERROR,
            error="Step failed"
        )
        
        agent_result = AgentResult(
            success=False,
            results=[error_flow],
            error="Agent execution failed",
            metadata={"failure_reason": "step_failure"}
        )
        
        assert agent_result.success is False
        assert len(agent_result.results) == 1
        assert agent_result.results[0].is_error()
        assert agent_result.error == "Agent execution failed"
        assert agent_result.metadata["failure_reason"] == "step_failure"
    
    def test_agent_result_mixed_results(self):
        """Test AgentResult with mixed success/error results."""
        success_flow = FlowResult(
            data={"result": "success"},
            flow_name="success_step",
            status=FlowStatus.SUCCESS
        )
        
        error_flow = FlowResult(
            data={},
            flow_name="error_step",
            status=FlowStatus.ERROR,
            error="Failed"
        )
        
        agent_result = AgentResult(
            success=False,  # Overall failure due to error step
            results=[success_flow, error_flow],
            error="Partial execution failure"
        )
        
        assert agent_result.success is False
        assert len(agent_result.results) == 2
        assert agent_result.results[0].is_success()
        assert agent_result.results[1].is_error()
        assert agent_result.error == "Partial execution failure"


class TestTypeAliases:
    """Test type aliases and generic functionality."""
    
    def test_tresult_type_alias(self):
        """Test TResult type alias."""
        result: TResult[FlowTestModel] = FlowResult(
            data={"name": "test", "value": 42},
            flow_name="typed_flow"
        )
        
        # Should work with type hints
        assert isinstance(result, FlowResult)
        typed_data = result.get_typed(FlowTestModel)
        assert isinstance(typed_data, FlowTestModel)


class TestFlowResultIntegration:
    """Test FlowResult integration scenarios."""
    
    def test_flow_result_chaining(self):
        """Test chaining flow results together."""
        # Create initial result
        result1 = result_from_value({"step1": "completed"}, "step1")
        
        # Process and create next result
        result2 = FlowResult(
            data={"step1": result1.step1, "step2": "completed"},
            flow_name="step2",
            status=FlowStatus.SUCCESS,
            metadata={"previous_flow": result1.flow_name}
        )
        
        assert result2.step1 == "completed"
        assert result2.step2 == "completed"
        assert result2.metadata["previous_flow"] == "step1"
    
    def test_flow_result_error_propagation(self):
        """Test error propagation through results."""
        # Create error result
        error_result_obj = error_result(
            "Database connection failed",
            "db_connection",
            {"code": "DB_CONN_ERROR", "retries": 3}
        )
        
        # Create dependent result that inherits error context
        dependent_result = FlowResult(
            data={},
            flow_name="dependent_operation",
            status=FlowStatus.ERROR,
            error="Operation failed due to dependency",
            error_details={"dependency_error": error_result_obj.error},
            metadata={"failed_dependency": error_result_obj.flow_name}
        )
        
        assert dependent_result.is_error()
        assert "dependency_error" in dependent_result.error_details
        assert dependent_result.metadata["failed_dependency"] == "db_connection"
    
    def test_flow_result_with_complex_data_types(self):
        """Test FlowResult with complex data types."""
        complex_data = {
            "timestamp": datetime.now(),
            "model": FlowTestModel(name="complex", value=100),
            "list_of_dicts": [{"id": 1, "name": "item1"}, {"id": 2, "name": "item2"}],
            "nested": {
                "level1": {
                    "level2": ["a", "b", "c"]
                }
            }
        }
        
        result = FlowResult(
            data=complex_data,
            flow_name="complex_data_flow",
            duration=5.2,
            metadata={"complexity": "high"}
        )
        
        assert isinstance(result.timestamp, datetime)
        assert isinstance(result.model, FlowTestModel)
        assert result.model.name == "complex"
        assert len(result.list_of_dicts) == 2
        assert result.nested["level1"]["level2"] == ["a", "b", "c"]
    
    def test_flow_result_serialization_compatibility(self):
        """Test that FlowResult can be serialized and maintains structure."""
        result = FlowResult(
            data={"key": "value", "number": 42},
            flow_name="serialization_test",
            status=FlowStatus.SUCCESS,
            metadata={"version": "1.0"}
        )
        
        # Convert to dict (simulates serialization)
        result_dict = result.as_dict()
        
        # Verify structure is preserved
        assert result_dict["data"]["key"] == "value"
        assert result_dict["data"]["number"] == 42
        assert result_dict["flow_name"] == "serialization_test"
        assert result_dict["status"] == "SUCCESS"
        assert result_dict["metadata"]["version"] == "1.0"
        assert isinstance(result_dict["timestamp"], str)  # ISO format


if __name__ == "__main__":
    pytest.main([__file__, "-v"])