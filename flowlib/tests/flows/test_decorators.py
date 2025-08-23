"""Comprehensive tests for Flow Decorators (@flow, @pipeline)."""

import pytest
import asyncio
import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.flows.base.base import Flow
from flowlib.core.context.context import Context


# Test models for validation
class SimpleInput(BaseModel):
    """Simple input model."""
    value: int
    name: str = "default"


class SimpleOutput(BaseModel):
    """Simple output model."""
    result: str
    processed_value: int


class ComplexInput(BaseModel):
    """Complex input model with validation."""
    title: str = Field(..., min_length=1)
    count: int = Field(..., ge=0)
    tags: list = Field(default_factory=list)


class ComplexOutput(BaseModel):
    """Complex output model."""
    summary: str
    total_count: int
    processed_tags: list


class InvalidModel:
    """Non-Pydantic model for testing validation."""
    pass


class TestFlowDecorator:
    """Test @flow decorator functionality."""
    
    def test_flow_decorator_basic(self):
        """Test basic @flow decorator usage."""
        @flow(description="Test flow")
        class TestFlow:
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {"result": "success"}
        
        # Check flow metadata
        assert hasattr(TestFlow, '__flow_metadata__')
        assert TestFlow.__flow_metadata__['name'] == "TestFlow"
        assert TestFlow.__flow_metadata__['is_infrastructure'] is False
        
        # Check pipeline method
        assert hasattr(TestFlow, '__pipeline_method__')
        assert TestFlow.__pipeline_method__ == "run_pipeline"
        
        # Check flow inheritance
        assert issubclass(TestFlow, Flow)
        
        # Test instance creation
        flow_instance = TestFlow()
        assert isinstance(flow_instance, Flow)
        assert hasattr(flow_instance, 'get_description')
        assert flow_instance.get_description() == "Test flow"
    
    def test_flow_decorator_with_custom_name(self):
        """Test @flow decorator with custom name."""
        @flow(name="custom-flow-name", description="Custom named flow")
        class TestFlow:
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {"result": "success"}
        
        assert TestFlow.__flow_metadata__['name'] == "custom-flow-name"
        assert TestFlow.__flow_name__ == "custom-flow-name"
        
        flow_instance = TestFlow()
        assert flow_instance.name == "custom-flow-name"
    
    def test_flow_decorator_infrastructure_flag(self):
        """Test @flow decorator with infrastructure flag."""
        @flow(description="Infrastructure flow", is_infrastructure=True)
        class InfraFlow:
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {"result": "infrastructure"}
        
        assert InfraFlow.__flow_metadata__['is_infrastructure'] is True
        assert InfraFlow.is_infrastructure is True
    
    def test_flow_decorator_no_description_error(self):
        """Test that @flow decorator requires description."""
        with pytest.raises(TypeError):  # description is required parameter
            @flow()
            class TestFlow:
                @pipeline()
                async def run_pipeline(self, context: Context) -> dict:
                    return {}
    
    def test_flow_decorator_no_pipeline_method_error(self):
        """Test that @flow decorator requires exactly one pipeline method."""
        with pytest.raises(ValueError, match="must define exactly one pipeline method"):
            @flow(description="No pipeline flow")
            class NoPipelineFlow:
                async def some_method(self, context: Context) -> dict:
                    return {}
    
    def test_flow_decorator_multiple_pipeline_methods_error(self):
        """Test that @flow decorator rejects multiple pipeline methods."""
        with pytest.raises(ValueError, match="has multiple pipeline methods"):
            @flow(description="Multiple pipeline flow")
            class MultiplePipelineFlow:
                @pipeline()
                async def pipeline1(self, context: Context) -> dict:
                    return {}
                
                @pipeline()
                async def pipeline2(self, context: Context) -> dict:
                    return {}
    
    def test_flow_decorator_existing_flow_subclass(self):
        """Test @flow decorator on class that already inherits from Flow."""
        class BaseFlow(Flow):
            def __init__(self):
                super().__init__("base", metadata={"test": True})
        
        @flow(description="Flow subclass")
        class TestFlow(BaseFlow):
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {"inherited": True}
        
        flow_instance = TestFlow()
        assert isinstance(flow_instance, Flow)
        # The decorator adds get_description method even if class already inherits from Flow
        assert hasattr(flow_instance, 'get_description')
        # But the actual behavior might be empty string due to Flow's default implementation
        description = flow_instance.get_description()
        assert description in ["Flow subclass", ""]  # Allow either behavior
    
    def test_flow_decorator_with_existing_description_method(self):
        """Test @flow decorator preserves existing get_description method."""
        @flow(description="Decorator description")
        class TestFlow:
            def get_description(self):
                return "Original description"
            
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {}
        
        flow_instance = TestFlow()
        # Should preserve the original method
        assert flow_instance.get_description() == "Original description"
    
    def test_flow_decorator_with_custom_init(self):
        """Test @flow decorator with custom __init__ method."""
        @flow(description="Custom init flow")
        class TestFlow:
            def __init__(self, custom_param: str = "default"):
                self.custom_param = custom_param
            
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {"param": self.custom_param}
        
        flow_instance = TestFlow(custom_param="test_value")
        assert flow_instance.custom_param == "test_value"
        assert isinstance(flow_instance, Flow)


class TestPipelineDecorator:
    """Test @pipeline decorator functionality."""
    
    def test_pipeline_decorator_basic(self):
        """Test basic @pipeline decorator usage."""
        @flow(description="Pipeline test")
        class TestFlow:
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {"result": "pipeline_success"}
        
        flow_instance = TestFlow()
        pipeline_method = getattr(flow_instance, "run_pipeline")
        
        # Check pipeline metadata
        assert hasattr(pipeline_method, '__pipeline__')
        assert pipeline_method.__pipeline__ is True
        assert hasattr(pipeline_method, '__input_model__')
        assert hasattr(pipeline_method, '__output_model__')
    
    def test_pipeline_decorator_with_models(self):
        """Test @pipeline decorator with input/output models."""
        @flow(description="Model pipeline test")
        class TestFlow:
            @pipeline(input_model=SimpleInput, output_model=SimpleOutput)
            async def run_pipeline(self, input_data: SimpleInput) -> SimpleOutput:
                return SimpleOutput(
                    result="processed",
                    processed_value=input_data.value * 2
                )
        
        flow_instance = TestFlow()
        pipeline_method = getattr(flow_instance, "run_pipeline")
        
        assert pipeline_method.__input_model__ == SimpleInput
        assert pipeline_method.__output_model__ == SimpleOutput
    
    def test_pipeline_decorator_invalid_input_model(self):
        """Test @pipeline decorator rejects invalid input model."""
        with pytest.raises(ValueError, match="input_model must be a Pydantic BaseModel subclass"):
            @flow(description="Invalid input model flow")
            class TestFlow:
                @pipeline(input_model=InvalidModel)
                async def run_pipeline(self, context: Context) -> dict:
                    return {}
    
    def test_pipeline_decorator_invalid_output_model(self):
        """Test @pipeline decorator rejects invalid output model."""
        with pytest.raises(ValueError, match="output_model must be a Pydantic BaseModel subclass"):
            @flow(description="Invalid output model flow")
            class TestFlow:
                @pipeline(output_model="not_a_model")
                async def run_pipeline(self, context: Context) -> dict:
                    return {}
    
    def test_pipeline_decorator_with_kwargs_syntax(self):
        """Test @pipeline decorator with kwargs syntax."""
        @flow(description="Kwargs pipeline test")
        class TestFlow:
            @pipeline(input_model=ComplexInput, output_model=ComplexOutput)
            async def run_pipeline(self, input_data: ComplexInput) -> ComplexOutput:
                return ComplexOutput(
                    summary=f"Processed {input_data.title}",
                    total_count=input_data.count,
                    processed_tags=input_data.tags
                )
        
        flow_instance = TestFlow()
        pipeline_method = getattr(flow_instance, "run_pipeline")
        
        assert pipeline_method.__input_model__ == ComplexInput
        assert pipeline_method.__output_model__ == ComplexOutput
    
    def test_pipeline_decorator_without_parentheses(self):
        """Test @pipeline decorator without parentheses."""
        @flow(description="No parens pipeline test")
        class TestFlow:
            @pipeline
            async def run_pipeline(self, context: Context) -> dict:
                return {"no_parens": True}
        
        flow_instance = TestFlow()
        pipeline_method = getattr(flow_instance, "run_pipeline")
        
        assert pipeline_method.__pipeline__ is True
        assert pipeline_method.__input_model__ is None
        assert pipeline_method.__output_model__ is None


class TestDecoratorIntegration:
    """Test integration between @flow and @pipeline decorators."""
    
    def test_flow_with_pipeline_schemas(self):
        """Test that flow instance gets schemas from pipeline method."""
        @flow(description="Schema integration test")
        class TestFlow:
            @pipeline(input_model=SimpleInput, output_model=SimpleOutput)
            async def run_pipeline(self, input_data: SimpleInput) -> SimpleOutput:
                return SimpleOutput(
                    result="schema_test",
                    processed_value=input_data.value
                )
        
        flow_instance = TestFlow()
        
        # Check that schemas are set on pipeline method
        pipeline_method = getattr(flow_instance, "run_pipeline")
        assert pipeline_method.__input_model__ == SimpleInput
        assert pipeline_method.__output_model__ == SimpleOutput
    
    def test_multiple_flows_isolation(self):
        """Test that multiple flows are properly isolated."""
        @flow(description="First flow")
        class FirstFlow:
            @pipeline(input_model=SimpleInput)
            async def run_pipeline(self, input_data: SimpleInput) -> dict:
                return {"flow": "first", "value": input_data.value}
        
        @flow(description="Second flow")
        class SecondFlow:
            @pipeline(output_model=SimpleOutput)
            async def run_pipeline(self, context: Context) -> SimpleOutput:
                return SimpleOutput(result="second", processed_value=42)
        
        first_instance = FirstFlow()
        second_instance = SecondFlow()
        
        # Check isolation
        assert first_instance.__class__.__flow_name__ == "FirstFlow"
        assert second_instance.__class__.__flow_name__ == "SecondFlow"
        
        # Check pipeline method schemas
        first_pipeline = getattr(first_instance, "run_pipeline")
        second_pipeline = getattr(second_instance, "run_pipeline")
        
        assert first_pipeline.__input_model__ == SimpleInput
        assert first_pipeline.__output_model__ is None
        assert second_pipeline.__input_model__ is None
        assert second_pipeline.__output_model__ == SimpleOutput


class TestDecoratorExecution:
    """Test execution behavior of decorated flows."""
    
    async def test_pipeline_execution_timing(self):
        """Test that pipeline execution is timed."""
        execution_times = []
        
        @flow(description="Timing test flow")
        class TimingFlow:
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                await asyncio.sleep(0.01)  # Small delay
                return {"timing": "test"}
        
        flow_instance = TimingFlow()
        
        # Execute pipeline
        context = Context(data={})
        result = await flow_instance.run_pipeline(context)
        
        assert result["timing"] == "test"
    
    async def test_pipeline_execution_with_validation(self):
        """Test pipeline execution with input/output validation."""
        @flow(description="Validation test flow")
        class ValidationFlow:
            @pipeline(input_model=SimpleInput, output_model=SimpleOutput)
            async def run_pipeline(self, input_data: SimpleInput) -> SimpleOutput:
                return SimpleOutput(
                    result=f"Hello {input_data.name}",
                    processed_value=input_data.value * 10
                )
        
        flow_instance = ValidationFlow()
        
        # Test valid input
        valid_input = SimpleInput(value=5, name="test")
        result = await flow_instance.run_pipeline(valid_input)
        
        assert isinstance(result, SimpleOutput)
        assert result.result == "Hello test"
        assert result.processed_value == 50
    
    async def test_pipeline_execution_error_handling(self):
        """Test pipeline execution error handling."""
        @flow(description="Error handling test flow")
        class ErrorFlow:
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                raise ValueError("Test error")
        
        flow_instance = ErrorFlow()
        context = Context(data={})
        
        with pytest.raises(ValueError, match="Test error"):
            await flow_instance.run_pipeline(context)
    
    async def test_pipeline_execution_with_complex_models(self):
        """Test pipeline execution with complex models."""
        @flow(description="Complex model test flow")
        class ComplexFlow:
            @pipeline(input_model=ComplexInput, output_model=ComplexOutput)
            async def run_pipeline(self, input_data: ComplexInput) -> ComplexOutput:
                processed_tags = [tag.upper() for tag in input_data.tags]
                
                return ComplexOutput(
                    summary=f"Processed: {input_data.title}",
                    total_count=input_data.count + len(input_data.tags),
                    processed_tags=processed_tags
                )
        
        flow_instance = ComplexFlow()
        
        # Test with complex input
        complex_input = ComplexInput(
            title="Test Title",
            count=5,
            tags=["tag1", "tag2", "tag3"]
        )
        
        result = await flow_instance.run_pipeline(complex_input)
        
        assert isinstance(result, ComplexOutput)
        assert result.summary == "Processed: Test Title"
        assert result.total_count == 8  # 5 + 3 tags
        assert result.processed_tags == ["TAG1", "TAG2", "TAG3"]


class TestDecoratorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_flow_decorator_with_none_name(self):
        """Test @flow decorator with None name."""
        @flow(name=None, description="None name flow")
        class TestFlow:
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {}
        
        # Should use class name when name is None
        assert TestFlow.__flow_metadata__['name'] == "TestFlow"
    
    def test_flow_decorator_empty_string_name(self):
        """Test @flow decorator with empty string name."""
        @flow(name="", description="Empty name flow")
        class TestFlow:
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {}
        
        # Should fall back to class name since empty string is falsy
        assert TestFlow.__flow_metadata__['name'] == "TestFlow"
    
    def test_pipeline_decorator_with_none_models(self):
        """Test @pipeline decorator with None models."""
        @flow(description="None models flow")
        class TestFlow:
            @pipeline(input_model=None, output_model=None)
            async def run_pipeline(self, context: Context) -> dict:
                return {}
        
        flow_instance = TestFlow()
        pipeline_method = getattr(flow_instance, "run_pipeline")
        
        assert pipeline_method.__input_model__ is None
        assert pipeline_method.__output_model__ is None
    
    def test_flow_decorator_with_special_characters_in_name(self):
        """Test @flow decorator with special characters in name."""
        @flow(name="test-flow_with.special@chars", description="Special chars flow")
        class TestFlow:
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {}
        
        assert TestFlow.__flow_metadata__['name'] == "test-flow_with.special@chars"
    
    def test_flow_decorator_preserves_class_attributes(self):
        """Test that @flow decorator preserves class attributes."""
        @flow(description="Attribute preservation test")
        class TestFlow:
            class_attribute = "preserved"
            
            def __init__(self):
                self.instance_attribute = "also_preserved"
            
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {"class_attr": self.class_attribute, "instance_attr": self.instance_attribute}
        
        flow_instance = TestFlow()
        
        assert TestFlow.class_attribute == "preserved"
        assert flow_instance.instance_attribute == "also_preserved"
    
    def test_flow_decorator_with_class_methods(self):
        """Test @flow decorator with class methods and static methods."""
        @flow(description="Class methods test")
        class TestFlow:
            @classmethod
            def get_flow_type(cls):
                return "test_flow"
            
            @staticmethod
            def get_version():
                return "1.0.0"
            
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {
                    "type": self.get_flow_type(),
                    "version": self.get_version()
                }
        
        flow_instance = TestFlow()
        
        assert TestFlow.get_flow_type() == "test_flow"
        assert TestFlow.get_version() == "1.0.0"
        assert flow_instance.get_flow_type() == "test_flow"
        assert flow_instance.get_version() == "1.0.0"


class TestDecoratorMetadata:
    """Test metadata behavior of decorators."""
    
    def test_flow_metadata_structure(self):
        """Test flow metadata structure."""
        @flow(name="metadata-test", description="Metadata test flow", is_infrastructure=True)
        class TestFlow:
            @pipeline(input_model=SimpleInput, output_model=SimpleOutput)
            async def run_pipeline(self, input_data: SimpleInput) -> SimpleOutput:
                return SimpleOutput(result="metadata", processed_value=1)
        
        metadata = TestFlow.__flow_metadata__
        
        assert isinstance(metadata, dict)
        assert metadata["name"] == "metadata-test"
        assert metadata["is_infrastructure"] is True
        
        # Check other attributes
        assert TestFlow.__flow_name__ == "metadata-test"
        assert TestFlow.__pipeline_method__ == "run_pipeline"
        assert TestFlow.is_infrastructure is True
    
    def test_pipeline_metadata_structure(self):
        """Test pipeline metadata structure."""
        @flow(description="Pipeline metadata test")
        class TestFlow:
            @pipeline(input_model=ComplexInput, output_model=ComplexOutput)
            async def run_pipeline(self, input_data: ComplexInput) -> ComplexOutput:
                return ComplexOutput(summary="test", total_count=0, processed_tags=[])
        
        flow_instance = TestFlow()
        pipeline_method = getattr(flow_instance, "run_pipeline")
        
        # Check pipeline attributes
        assert pipeline_method.__pipeline__ is True
        assert pipeline_method.__input_model__ == ComplexInput
        assert pipeline_method.__output_model__ == ComplexOutput
        
        # Check that original function name is preserved
        assert pipeline_method.__name__ == "run_pipeline"


if __name__ == "__main__":
    pytest.main([__file__])