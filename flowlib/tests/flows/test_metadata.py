"""Comprehensive tests for flow metadata module."""

import pytest
from typing import Dict, Any
from pydantic import BaseModel

from flowlib.flows.models.metadata import FlowMetadata


class MockInputModel(BaseModel):
    """Mock input model for metadata testing."""
    text: str
    count: int = 1


class MockOutputModel(BaseModel):
    """Mock output model for metadata testing."""
    result: str
    success: bool = True


class InvalidModel:
    """Non-Pydantic model for testing validation."""
    pass


class MockFlowWithPipeline:
    """Mock flow class with pipeline method."""
    
    def get_description(self) -> str:
        return "Test flow for metadata testing"
    
    def get_pipeline_method(self):
        """Return a mock pipeline method with models."""
        pipeline = MockPipeline()
        return pipeline


class MockFlowClass:
    """Mock flow class for testing class-based extraction."""
    
    def get_description(self) -> str:
        return "Test flow class"
    
    @classmethod
    def get_pipeline_method_cls(cls):
        """Return a mock pipeline method with models."""
        return MockPipeline()


class MockPipeline:
    """Mock pipeline with input/output models."""
    
    def __init__(self):
        self.__input_model__ = MockInputModel
        self.__output_model__ = MockOutputModel


class MockPipelineInvalidInput:
    """Mock pipeline with invalid input model."""
    
    def __init__(self):
        self.__input_model__ = InvalidModel  # Not a Pydantic model
        self.__output_model__ = MockOutputModel


class MockPipelineInvalidOutput:
    """Mock pipeline with invalid output model."""
    
    def __init__(self):
        self.__input_model__ = MockInputModel
        self.__output_model__ = InvalidModel  # Not a Pydantic model


class MockPipelineNoInput:
    """Mock pipeline with no input model."""
    
    def __init__(self):
        self.__input_model__ = None
        self.__output_model__ = MockOutputModel


class MockPipelineNoOutput:
    """Mock pipeline with no output model."""
    
    def __init__(self):
        self.__input_model__ = MockInputModel
        self.__output_model__ = None


class MockFlowNoPipeline:
    """Mock flow without pipeline method."""
    
    def get_description(self) -> str:
        return "Flow without pipeline"


class MockFlowInvalidDescription:
    """Mock flow with invalid description."""
    
    def get_description(self):
        return 123  # Not a string


class MockFlowNoDescription:
    """Mock flow without get_description method."""
    pass


class MockFlowExceptionInDescription:
    """Mock flow that raises exception in get_description."""
    
    def get_description(self) -> str:
        raise RuntimeError("Description error")


class MockFlowExceptionInPipeline:
    """Mock flow that raises exception when getting pipeline."""
    
    def get_description(self) -> str:
        return "Test flow"
    
    def get_pipeline_method(self):
        raise RuntimeError("Pipeline error")


class TestFlowMetadata:
    """Test FlowMetadata functionality."""
    
    def test_flow_metadata_basic_creation(self):
        """Test basic FlowMetadata creation."""
        metadata = FlowMetadata(
            name="test_flow",
            description="A test flow",
            input_model=MockInputModel,
            output_model=MockOutputModel
        )
        
        assert metadata.name == "test_flow"
        assert metadata.description == "A test flow"
        assert metadata.input_model == MockInputModel
        assert metadata.output_model == MockOutputModel
        assert metadata.version == "1.0.0"
        assert metadata.tags == {}
    
    def test_flow_metadata_with_optional_fields(self):
        """Test FlowMetadata creation with optional fields."""
        tags = {"category": "test", "priority": "high"}
        
        metadata = FlowMetadata(
            name="advanced_flow",
            description="An advanced test flow",
            input_model=MockInputModel,
            output_model=MockOutputModel,
            version="2.1.0",
            tags=tags
        )
        
        assert metadata.name == "advanced_flow"
        assert metadata.description == "An advanced test flow"
        assert metadata.input_model == MockInputModel
        assert metadata.output_model == MockOutputModel
        assert metadata.version == "2.1.0"
        assert metadata.tags == tags
    
    def test_flow_metadata_model_validation_invalid_input(self):
        """Test model validation with invalid input model."""
        # Pydantic v2 raises ValidationError, not ValueError during model creation
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError) as exc_info:
            FlowMetadata(
                name="test_flow",
                description="Test flow",
                input_model=InvalidModel,  # Not a Pydantic model
                output_model=MockOutputModel
            )
        
        assert "Input should be a subclass of BaseModel" in str(exc_info.value)
    
    def test_flow_metadata_model_validation_invalid_output(self):
        """Test model validation with invalid output model."""
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError) as exc_info:
            FlowMetadata(
                name="test_flow",
                description="Test flow",
                input_model=MockInputModel,
                output_model=InvalidModel  # Not a Pydantic model
            )
        
        assert "Input should be a subclass of BaseModel" in str(exc_info.value)
    
    def test_flow_metadata_model_validation_non_type(self):
        """Test model validation with non-type objects."""
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError) as exc_info:
            FlowMetadata(
                name="test_flow",
                description="Test flow",
                input_model="not_a_type",  # String instead of type
                output_model=MockOutputModel
            )
        
        assert "Input should be a subclass of BaseModel" in str(exc_info.value)


class TestFlowMetadataFromFlow:
    """Test FlowMetadata.from_flow functionality."""
    
    def test_from_flow_instance_success(self):
        """Test creating metadata from flow instance."""
        flow = MockFlowWithPipeline()
        
        metadata = FlowMetadata.from_flow(flow, "test_flow")
        
        assert metadata.name == "test_flow"
        assert metadata.description == "Test flow for metadata testing"
        assert metadata.input_model == MockInputModel
        assert metadata.output_model == MockOutputModel
        assert metadata.version == "1.0.0"
    
    def test_from_flow_class_success(self):
        """Test creating metadata from flow class."""
        metadata = FlowMetadata.from_flow(MockFlowClass, "test_flow_class")
        
        assert metadata.name == "test_flow_class"
        assert metadata.description == "Test flow class"
        assert metadata.input_model == MockInputModel
        assert metadata.output_model == MockOutputModel
    
    def test_from_flow_class_with_docstring_fallback(self):
        """Test creating metadata from flow class with docstring fallback."""
        class FlowWithDocstring:
            """This is a flow with a docstring."""
            
            @classmethod
            def get_pipeline_method_cls(cls):
                return MockPipeline()
        
        metadata = FlowMetadata.from_flow(FlowWithDocstring, "docstring_flow")
        
        # Should use docstring as description since get_description() fails
        assert metadata.name == "docstring_flow"
        assert metadata.description == "This is a flow with a docstring."
    
    def test_from_flow_class_no_docstring(self):
        """Test creating metadata from flow class with no docstring."""
        class FlowNoDocstring:
            pass
        
        # This should raise an error because there are no models
        with pytest.raises(ValueError) as exc_info:
            FlowMetadata.from_flow(FlowNoDocstring, "no_docstring_flow")
        
        # The error should be about missing models, not description
        assert "does not have an input model" in str(exc_info.value)
    
    def test_from_flow_invalid_description_type(self):
        """Test from_flow with invalid description type."""
        flow = MockFlowInvalidDescription()
        
        with pytest.raises(ValueError) as exc_info:
            FlowMetadata.from_flow(flow, "invalid_desc_flow")
        
        assert "get_description() must return a string" in str(exc_info.value)
        assert "int" in str(exc_info.value)
    
    def test_from_flow_no_description_method(self):
        """Test from_flow with missing get_description method."""
        flow = MockFlowNoDescription()
        
        with pytest.raises(ValueError) as exc_info:
            FlowMetadata.from_flow(flow, "no_desc_flow")
        
        assert "Error getting description from flow no_desc_flow" in str(exc_info.value)
    
    def test_from_flow_exception_in_description(self):
        """Test from_flow when get_description raises exception."""
        flow = MockFlowExceptionInDescription()
        
        with pytest.raises(ValueError) as exc_info:
            FlowMetadata.from_flow(flow, "exception_desc_flow")
        
        assert "Error getting description from flow exception_desc_flow" in str(exc_info.value)
        assert "Description error" in str(exc_info.value)
    
    def test_from_flow_no_pipeline_method(self):
        """Test from_flow with missing pipeline method."""
        flow = MockFlowNoPipeline()
        
        with pytest.raises(ValueError) as exc_info:
            FlowMetadata.from_flow(flow, "no_pipeline_flow")
        
        assert "does not have an input model" in str(exc_info.value)
    
    def test_from_flow_exception_in_pipeline(self):
        """Test from_flow when pipeline method raises exception."""
        flow = MockFlowExceptionInPipeline()
        
        with pytest.raises(ValueError) as exc_info:
            FlowMetadata.from_flow(flow, "exception_pipeline_flow")
        
        assert "Error extracting models from flow exception_pipeline_flow" in str(exc_info.value)
        assert "Pipeline error" in str(exc_info.value)
    
    def test_from_flow_no_input_model(self):
        """Test from_flow with missing input model."""
        class FlowNoInputModel:
            def get_description(self) -> str:
                return "Flow without input model"
            
            def get_pipeline_method(self):
                return MockPipelineNoInput()
        
        flow = FlowNoInputModel()
        
        with pytest.raises(ValueError) as exc_info:
            FlowMetadata.from_flow(flow, "no_input_flow")
        
        assert "does not have an input model defined" in str(exc_info.value)
    
    def test_from_flow_no_output_model(self):
        """Test from_flow with missing output model."""
        class FlowNoOutputModel:
            def get_description(self) -> str:
                return "Flow without output model"
            
            def get_pipeline_method(self):
                return MockPipelineNoOutput()
        
        flow = FlowNoOutputModel()
        
        with pytest.raises(ValueError) as exc_info:
            FlowMetadata.from_flow(flow, "no_output_flow")
        
        assert "does not have an output model defined" in str(exc_info.value)
    
    def test_from_flow_invalid_input_model_type(self):
        """Test from_flow with invalid input model type."""
        class FlowInvalidInputModel:
            def get_description(self) -> str:
                return "Flow with invalid input model"
            
            def get_pipeline_method(self):
                return MockPipelineInvalidInput()
        
        flow = FlowInvalidInputModel()
        
        with pytest.raises(ValueError) as exc_info:
            FlowMetadata.from_flow(flow, "invalid_input_flow")
        
        assert "Input model must be a Pydantic BaseModel subclass" in str(exc_info.value)
    
    def test_from_flow_invalid_output_model_type(self):
        """Test from_flow with invalid output model type."""
        class FlowInvalidOutputModel:
            def get_description(self) -> str:
                return "Flow with invalid output model"
            
            def get_pipeline_method(self):
                return MockPipelineInvalidOutput()
        
        flow = FlowInvalidOutputModel()
        
        with pytest.raises(ValueError) as exc_info:
            FlowMetadata.from_flow(flow, "invalid_output_flow")
        
        assert "Output model must be a Pydantic BaseModel subclass" in str(exc_info.value)


class TestFlowMetadataEdgeCases:
    """Test edge cases and integration scenarios."""
    
    def test_metadata_with_complex_models(self):
        """Test metadata with complex Pydantic models."""
        class ComplexInputModel(BaseModel):
            data: Dict[str, Any]
            nested: Dict[str, str]
            list_field: list[int] = []
        
        class ComplexOutputModel(BaseModel):
            results: Dict[str, Any]
            metadata: Dict[str, str] = {}
            count: int
        
        metadata = FlowMetadata(
            name="complex_flow",
            description="Flow with complex models",
            input_model=ComplexInputModel,
            output_model=ComplexOutputModel,
            tags={"complexity": "high", "type": "advanced"}
        )
        
        assert metadata.input_model == ComplexInputModel
        assert metadata.output_model == ComplexOutputModel
        assert metadata.tags["complexity"] == "high"
    
    def test_metadata_equality(self):
        """Test metadata equality comparison."""
        metadata1 = FlowMetadata(
            name="test_flow",
            description="Test flow",
            input_model=MockInputModel,
            output_model=MockOutputModel
        )
        
        metadata2 = FlowMetadata(
            name="test_flow",
            description="Test flow",
            input_model=MockInputModel,
            output_model=MockOutputModel
        )
        
        metadata3 = FlowMetadata(
            name="different_flow",
            description="Test flow",
            input_model=MockInputModel,
            output_model=MockOutputModel
        )
        
        assert metadata1 == metadata2
        assert metadata1 != metadata3
    
    def test_metadata_serialization(self):
        """Test metadata serialization to dict."""
        metadata = FlowMetadata(
            name="serialization_flow",
            description="Flow for testing serialization",
            input_model=MockInputModel,
            output_model=MockOutputModel,
            version="1.2.3",
            tags={"category": "test"}
        )
        
        # Convert to dict (simulates serialization)
        metadata_dict = metadata.model_dump()
        
        assert metadata_dict["name"] == "serialization_flow"
        assert metadata_dict["description"] == "Flow for testing serialization"
        assert metadata_dict["version"] == "1.2.3"
        assert metadata_dict["tags"]["category"] == "test"
        # Note: model classes would need special handling for full serialization
    
    def test_metadata_with_inheritance(self):
        """Test metadata with inherited Pydantic models."""
        class BaseInputModel(BaseModel):
            base_field: str
        
        class ExtendedInputModel(BaseInputModel):
            extended_field: int
        
        class BaseOutputModel(BaseModel):
            base_result: str
        
        class ExtendedOutputModel(BaseOutputModel):
            extended_result: bool
        
        metadata = FlowMetadata(
            name="inheritance_flow",
            description="Flow with inherited models",
            input_model=ExtendedInputModel,
            output_model=ExtendedOutputModel
        )
        
        assert metadata.input_model == ExtendedInputModel
        assert metadata.output_model == ExtendedOutputModel
        assert issubclass(metadata.input_model, BaseModel)
        assert issubclass(metadata.output_model, BaseModel)
    
    def test_from_flow_comprehensive_pipeline_extraction(self):
        """Test comprehensive pipeline model extraction."""
        class ComprehensiveFlow:
            def get_description(self) -> str:
                return "Comprehensive flow with full pipeline"
            
            def get_pipeline_method(self):
                class FullPipeline:
                    def __init__(self):
                        self.__input_model__ = MockInputModel
                        self.__output_model__ = MockOutputModel
                        self.__version__ = "2.0.0"
                        self.__tags__ = {"tested": True}
                
                return FullPipeline()
        
        flow = ComprehensiveFlow()
        metadata = FlowMetadata.from_flow(flow, "comprehensive_flow")
        
        assert metadata.name == "comprehensive_flow"
        assert metadata.description == "Comprehensive flow with full pipeline"
        assert metadata.input_model == MockInputModel
        assert metadata.output_model == MockOutputModel
    
    def test_from_flow_pipeline_with_none_models(self):
        """Test pipeline extraction when models are explicitly None."""
        class FlowWithNoneModels:
            def get_description(self) -> str:
                return "Flow with None models"
            
            def get_pipeline_method(self):
                class NoneModelsPipeline:
                    def __init__(self):
                        self.__input_model__ = None
                        self.__output_model__ = None
                
                return NoneModelsPipeline()
        
        flow = FlowWithNoneModels()
        
        with pytest.raises(ValueError) as exc_info:
            FlowMetadata.from_flow(flow, "none_models_flow")
        
        assert "does not have an input model defined" in str(exc_info.value)
    
    def test_from_flow_pipeline_missing_attributes(self):
        """Test pipeline extraction when model attributes are missing."""
        class FlowWithMissingAttributes:
            def get_description(self) -> str:
                return "Flow with missing pipeline attributes"
            
            def get_pipeline_method(self):
                class EmptyPipeline:
                    pass
                
                return EmptyPipeline()
        
        flow = FlowWithMissingAttributes()
        
        with pytest.raises(ValueError) as exc_info:
            FlowMetadata.from_flow(flow, "missing_attrs_flow")
        
        assert "does not have an input model defined" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])