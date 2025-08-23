"""Tests for flowlib flows system."""

import pytest
import asyncio
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from flowlib.flows.base.base import Flow, FlowSettings
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.flows.models.results import FlowResult, result_from_value, error_result
from flowlib.flows.models.constants import FlowStatus
from flowlib.flows.registry.registry import stage_registry
from flowlib.flows.models.metadata import FlowMetadata
from flowlib.core.context.context import Context
from flowlib.core.errors.errors import ExecutionError


# Test models
class MockInput(BaseModel):
    value: int
    name: str = "test"


class MockOutput(BaseModel):
    result: str
    processed_value: int


class MockContext(BaseModel):
    intermediate: str


class TestFlowSettings:
    """Test FlowSettings functionality."""
    
    def test_default_settings(self):
        """Test default settings creation."""
        settings = FlowSettings()
        
        assert settings.timeout_seconds is None
        assert settings.max_retries == 0
        assert settings.validate_inputs is True
        assert settings.log_level == "INFO"
    
    def test_custom_settings(self):
        """Test custom settings."""
        settings = FlowSettings(
            timeout_seconds=30.0,
            max_retries=3,
            log_level="DEBUG",
            debug_mode=True
        )
        
        assert settings.timeout_seconds == 30.0
        assert settings.max_retries == 3
        assert settings.log_level == "DEBUG"
        assert settings.debug_mode is True
    
    def test_settings_validation(self):
        """Test settings validation."""
        with pytest.raises(ValueError):
            FlowSettings(log_level="INVALID")
        
        with pytest.raises(ValueError):
            FlowSettings(max_retries=-1)
    
    def test_settings_merge(self):
        """Test settings merging."""
        base = FlowSettings(timeout_seconds=10.0, max_retries=2)
        override = FlowSettings(max_retries=5, debug_mode=True)
        
        merged = base.merge(override)
        assert merged.timeout_seconds == 10.0
        assert merged.max_retries == 5
        assert merged.debug_mode is True


class TestFlowResults:
    """Test FlowResult functionality."""
    
    def test_success_result(self):
        """Test successful result creation."""
        data = {"key": "value"}
        result = FlowResult(data=data, flow_name="test_flow")
        
        assert result.is_success()
        assert not result.is_error()
        assert result.flow_name == "test_flow"
        assert result.data == data
    
    def test_error_result(self):
        """Test error result creation."""
        result = error_result("Something went wrong", "test_flow")
        
        assert result.is_error()
        assert not result.is_success()
        assert result.error == "Something went wrong"
        assert result.status == FlowStatus.ERROR
    
    def test_result_attribute_access(self):
        """Test attribute-based access to result data."""
        data = {"name": "test", "value": 42}
        result = FlowResult(data=data, flow_name="test")
        
        assert result.name == "test"
        assert result.value == 42
        
        with pytest.raises(AttributeError):
            _ = result.missing_attribute
    
    def test_typed_result_conversion(self):
        """Test typed result conversion."""
        data = {"result": "success", "processed_value": 100}
        result = FlowResult(data=data, flow_name="test")
        
        typed_result = result.get_typed(MockOutput)
        assert isinstance(typed_result, MockOutput)
        assert typed_result.result == "success"
        assert typed_result.processed_value == 100
    
    def test_result_raise_if_error(self):
        """Test raise_if_error method."""
        success_result = result_from_value({"data": "success"})
        error_result_obj = error_result("Test error")
        
        # Should not raise for success
        success_result.raise_if_error()
        
        # Should raise for error
        with pytest.raises(Exception):
            error_result_obj.raise_if_error()


class TestFlowDecorators:
    """Test flow decorators."""
    
    def test_flow_decorator(self):
        """Test @flow decorator."""
        @flow(name="test-flow", description="Test flow")
        class TestFlow:
            @pipeline(input_model=MockInput, output_model=MockOutput)
            async def run_pipeline(self, input_data: MockInput) -> MockOutput:
                return MockOutput(
                    result="success",
                    processed_value=input_data.value * 2
                )
        
        # Check that flow was registered
        assert hasattr(TestFlow, '__flow_metadata__')
        assert TestFlow.__flow_metadata__['name'] == "test-flow"
    
    def test_pipeline_decorator(self):
        """Test @pipeline decorator."""
        @flow(name="pipeline-test", description="Test pipeline")
        class PipelineTestFlow:
            @pipeline(input_model=MockInput, output_model=MockOutput)
            async def run_pipeline(self, input_data: MockInput) -> MockOutput:
                if input_data.value < 0:
                    raise ValueError("Value must be positive")
                return MockOutput(result="validated", processed_value=input_data.value)
        
        flow_instance = PipelineTestFlow()
        assert hasattr(flow_instance, 'run_pipeline')
        
        # Check pipeline method has expected attributes
        pipeline_method = flow_instance.run_pipeline
        assert hasattr(pipeline_method, '__pipeline__')
        assert pipeline_method.__pipeline__ is True


class TestFlowExecution:
    """Test flow execution."""
    
    async def test_simple_flow_execution(self):
        """Test simple flow execution."""
        @flow(name="simple-test", description="Simple test flow")
        class SimpleFlow:
            @pipeline(input_model=MockInput, output_model=MockOutput)
            async def run_pipeline(self, input_data: MockInput) -> MockOutput:
                return MockOutput(
                    result="doubled",
                    processed_value=input_data.value * 2
                )
        
        flow_instance = SimpleFlow()
        input_data = MockInput(value=5, name="test")
        result = await flow_instance.run_pipeline(input_data)
        
        assert isinstance(result, MockOutput)
        assert result.result == "doubled"
        assert result.processed_value == 10
    
    async def test_complex_flow(self):
        """Test flow with complex processing logic."""
        @flow(name="complex-test", description="Complex test flow")
        class ComplexFlow:
            @pipeline(input_model=MockInput, output_model=MockOutput)
            async def run_pipeline(self, input_data: MockInput) -> MockOutput:
                # Validate
                if input_data.value <= 0:
                    raise ValueError("Value must be positive")
                
                # Transform
                transformed_value = input_data.value * 3
                name_upper = input_data.name.upper()
                
                # Finalize
                return MockOutput(
                    result=f"processed_{name_upper}",
                    processed_value=transformed_value
                )
        
        flow_instance = ComplexFlow()
        input_data = MockInput(value=4, name="test")
        result = await flow_instance.run_pipeline(input_data)
        
        assert isinstance(result, MockOutput)
        assert result.result == "processed_TEST"
        assert result.processed_value == 12
    
    async def test_flow_error_handling(self):
        """Test flow error handling."""
        @flow(name="error-test", description="Error test flow")
        class ErrorFlow:
            @pipeline(input_model=MockInput, output_model=MockOutput)
            async def run_pipeline(self, input_data: MockInput) -> MockOutput:
                if input_data.value < 0:
                    raise ValueError("Negative values not allowed")
                return MockOutput(result="success", processed_value=input_data.value)
        
        flow_instance = ErrorFlow()
        
        # Valid input should work
        valid_input = MockInput(value=5)
        result = await flow_instance.run_pipeline(valid_input)
        assert result.result == "success"
        
        # Invalid input should raise error
        invalid_input = MockInput(value=-1)
        with pytest.raises(ValueError):
            await flow_instance.run_pipeline(invalid_input)
    
    async def test_flow_with_settings(self):
        """Test flow with custom settings."""
        custom_settings = FlowSettings(
            timeout_seconds=5.0,
            max_retries=2,
            debug_mode=True
        )
        
        @flow(name="settings-test", description="Settings test flow")
        class SettingsFlow:
            def __init__(self):
                self.settings = custom_settings
            
            @pipeline(input_model=MockInput, output_model=MockOutput)
            async def run_pipeline(self, input_data: MockInput) -> MockOutput:
                # Simulate processing time
                await asyncio.sleep(0.01)
                return MockOutput(
                    result="processed_with_settings",
                    processed_value=input_data.value
                )
        
        flow_instance = SettingsFlow()
        input_data = MockInput(value=10)
        result = await flow_instance.run_pipeline(input_data)
        
        assert result.result == "processed_with_settings"
        assert flow_instance.settings.debug_mode is True


class TestFlowRegistry:
    """Test flow registry functionality."""
    
    def test_flow_registration(self):
        """Test that flows are properly registered."""
        @flow(name="registry-test-flow", description="Registry test flow")
        class RegistryTestFlow:
            @pipeline(input_model=MockInput, output_model=MockOutput)
            async def run_pipeline(self, input_data: MockInput) -> MockOutput:
                return MockOutput(result="registered", processed_value=input_data.value)
        
        # Flow should be registered in the stage registry
        assert stage_registry.contains_flow("registry-test-flow")
    
    def test_flow_metadata(self):
        """Test that flow metadata is properly set."""
        @flow(name="metadata-test", description="Metadata test flow")
        class MetadataFlow:
            @pipeline(input_model=MockInput, output_model=MockOutput)
            async def run_pipeline(self, input_data: MockInput) -> MockOutput:
                return MockOutput(result="metadata", processed_value=input_data.value)
        
        flow_instance = MetadataFlow()
        assert hasattr(MetadataFlow, '__flow_metadata__')
        assert MetadataFlow.__flow_metadata__['name'] == "metadata-test"


class TestFlowMetadata:
    """Test flow metadata functionality."""
    
    def test_metadata_creation(self):
        """Test flow metadata creation."""
        metadata = FlowMetadata(
            name="test_flow",
            description="Test flow for metadata",
            input_model=MockInput,
            output_model=MockOutput,
            version="1.0.0"
        )
        
        assert metadata.name == "test_flow"
        assert metadata.description == "Test flow for metadata"
        assert metadata.input_model == MockInput
        assert metadata.output_model == MockOutput
        assert metadata.version == "1.0.0"
    
    def test_metadata_serialization(self):
        """Test metadata serialization."""
        metadata = FlowMetadata(
            name="test_flow",
            description="Test flow",
            input_model=MockInput,
            output_model=MockOutput,
            tags={"category": "test", "type": "demo"}
        )
        
        serialized = metadata.model_dump()
        assert serialized["name"] == "test_flow"
        assert serialized["tags"]["category"] == "test"


if __name__ == "__main__":
    pytest.main([__file__])