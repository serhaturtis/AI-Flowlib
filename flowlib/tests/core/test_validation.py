"""Comprehensive tests for Core Validation module."""

import pytest
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from flowlib.core.validation.validation import (
    validate_data,
    create_dynamic_model,
    validate_with_schema,
    validate_function,
    validate_input,
    validate_output,
    ValidationRule,
    _convert_type,
    _extract_constraints
)
from flowlib.core.errors.errors import ValidationError, ErrorContext


# Test models for validation testing
class SimpleModel(BaseModel):
    """Simple test model."""
    name: str
    age: int


class ComplexModel(BaseModel):
    """Complex test model with validation."""
    title: str = Field(..., min_length=1, max_length=100)
    count: int = Field(..., ge=0, le=1000)
    tags: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class NestedModel(BaseModel):
    """Model with nested structure."""
    id: int
    profile: SimpleModel
    settings: Optional[ComplexModel] = None


class ValidationTestModel(BaseModel):
    """Model for testing various validation constraints."""
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    score: float = Field(..., ge=0.0, le=100.0)
    category: str = Field(..., pattern=r'^[A-Z]+$')


class TestValidateData:
    """Test validate_data function."""
    
    def test_validate_data_with_valid_dict(self):
        """Test validating valid dictionary data."""
        data = {"name": "John", "age": 30}
        
        result = validate_data(data, SimpleModel)
        
        assert isinstance(result, SimpleModel)
        assert result.name == "John"
        assert result.age == 30
    
    def test_validate_data_with_model_instance(self):
        """Test validating with existing model instance."""
        model = SimpleModel(name="Jane", age=25)
        
        result = validate_data(model, SimpleModel)
        
        assert result is model  # Should return same instance
        assert result.name == "Jane"
        assert result.age == 25

    def test_multiple_validation_errors(self):
        """Test handling multiple validation errors."""
        data = {
            "title": "",  # Too short
            "count": -1,  # Below minimum
            "tags": "not_a_list"  # Wrong type
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_data(data, ComplexModel)
        
        error = exc_info.value
        assert len(error.validation_errors) >= 3  # Multiple errors
    
    def test_function_validation_error_context(self):
        """Test function validation error context enrichment."""
        @validate_function
        def test_function(data: SimpleModel) -> ComplexModel:
            return {"title": "test", "count": 5}  # Valid return
        
        with pytest.raises(ValidationError) as exc_info:
            test_function({"name": "John", "age": "invalid"})
        
        error = exc_info.value
        assert "test_function" in error.message
        assert "data" in error.message
    
    def test_return_value_validation_error_context(self):
        """Test return value validation error context."""
        @validate_function
        def test_function(name: str) -> SimpleModel:
            return {"name": name, "age": "invalid"}  # Invalid return
        
        with pytest.raises(ValidationError) as exc_info:
            test_function("John")
        
        error = exc_info.value
        assert "test_function" in error.message
        assert "SimpleModel" in error.message


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_data_validation(self):
        """Test validation with empty data."""
        with pytest.raises(ValidationError):
            validate_data({}, SimpleModel)
    
    def test_none_data_validation(self):
        """Test validation with None data."""
        with pytest.raises(ValidationError):
            validate_data(None, SimpleModel)
    
    def test_very_large_data_validation(self):
        """Test validation with very large data structures."""
        large_data = {
            "title": "x" * 50,  # Max length
            "count": 1000,  # Max value
            "tags": [f"tag{i}" for i in range(100)],
            "metadata": {f"key{i}": f"value{i}" for i in range(100)}
        }
        
        result = validate_data(large_data, ComplexModel)
        assert isinstance(result, ComplexModel)
        assert len(result.tags) == 100
        assert len(result.metadata) == 100
    
    def test_unicode_data_validation(self):
        """Test validation with Unicode data."""
        unicode_data = {
            "name": "José María 中文 🚀",
            "age": 30
        }
        
        result = validate_data(unicode_data, SimpleModel)
        assert result.name == "José María 中文 🚀"
    
    def test_function_with_no_return_annotation(self):
        """Test function validation with no return type annotation."""
        @validate_function
        def test_function(user: SimpleModel):
            return user.name  # No return annotation
        
        result = test_function({"name": "John", "age": 30})
        assert result == "John"  # Should work without return validation


if __name__ == "__main__":
    pytest.main([__file__])