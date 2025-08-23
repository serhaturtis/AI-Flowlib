"""Comprehensive tests for the Core Context System."""

import pytest
from copy import deepcopy
from typing import Optional, List
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from flowlib.core.context.context import Context


class SimpleModel(BaseModel):
    """Simple test model."""
    name: str
    value: int


class ComplexModel(BaseModel):
    """Complex test model with validation."""
    title: str = Field(..., min_length=1)
    count: int = Field(..., ge=0)
    tags: List[str] = Field(default_factory=list)
    metadata: Optional[dict] = None


class InvalidModel(BaseModel):
    """Model with strict validation for testing failures."""
    required_field: str = Field(..., pattern=r'^[A-Z]+$')
    positive_number: int = Field(..., gt=0)


class TestContextCreation:
    """Test Context creation with various inputs."""
    
    def test_empty_context_creation(self):
        """Test creating empty context."""
        context = Context()
        assert context.data == {}
        assert context.model_type is None
        assert context.keys() == []
    
    def test_context_with_model_type_only(self):
        """Test creating context with model type but no data."""
        context = Context(model_type=SimpleModel)
        assert context.data == {}
        assert context.model_type == SimpleModel
    
    def test_context_with_pydantic_model(self):
        """Test creating context with Pydantic model instance."""
        model = SimpleModel(name="test", value=42)
        context = Context(data=model)
        
        assert context.data == {"name": "test", "value": 42}
        assert context.model_type == SimpleModel
        assert context.name == "test"
        assert context.value == 42
    
    def test_context_with_dict_data(self):
        """Test creating context with dictionary data."""
        data = {"name": "test", "value": 42}
        context = Context(data=data, model_type=SimpleModel)
        
        assert context.data == data
        assert context.model_type == SimpleModel
        assert context.name == "test"
        assert context.value == 42
    
    def test_context_with_dict_no_model_type(self):
        """Test creating context with dict but no model type."""
        data = {"key": "value"}
        context = Context(data=data)
        
        assert context.data == data
        assert context.model_type is None
        assert context.key == "value"
    
    def test_invalid_data_type_raises_error(self):
        """Test that invalid data type raises TypeError."""
        with pytest.raises(TypeError, match="Context data must be a Pydantic BaseModel instance or dict"):
            Context(data="invalid_string")
    
    def test_invalid_model_data_raises_error(self):
        """Test that invalid model data raises ValueError."""
        # Create an invalid model instance that would fail validation
        with pytest.raises(PydanticValidationError):
            InvalidModel(required_field="lowercase", positive_number=-1)


class TestContextAttributeAccess:
    """Test attribute-based access to context data."""
    
    def test_attribute_access(self):
        """Test accessing attributes directly."""
        model = ComplexModel(title="Test", count=5, tags=["a", "b"])
        context = Context(data=model)
        
        assert context.title == "Test"
        assert context.count == 5
        assert context.tags == ["a", "b"]
        assert context.metadata is None
    
    def test_get_method(self):
        """Test strict get method - no fallbacks."""
        context = Context(data={"existing": "value"})
        
        assert context.get("existing") == "value"
        
        # Strict mode should raise KeyError for missing keys
        with pytest.raises(KeyError, match="Key 'missing' not found in context"):
            context.get("missing")
    
    def test_missing_attribute_raises_error(self):
        """Test that missing attribute raises AttributeError."""
        context = Context(data={"key": "value"})
        
        with pytest.raises(AttributeError, match="'Context' object has no attribute 'missing'"):
            _ = context.missing
    
    def test_keys_method(self):
        """Test keys method."""
        context = Context(data={"a": 1, "b": 2, "c": 3})
        keys = context.keys()
        
        assert set(keys) == {"a", "b", "c"}
        assert isinstance(keys, list)
    
    def test_contains_method(self):
        """Test __contains__ method."""
        context = Context(data={"key": "value"})
        
        assert "key" in context
        assert "missing" not in context


class TestContextModification:
    """Test context modification methods (deprecated but still available)."""
    
    def test_set_method(self):
        """Test set method."""
        context = Context(data={"initial": "value"})
        result = context.set("new_key", "new_value")
        
        assert result is context  # Returns self
        assert context.get("new_key") == "new_value"
        assert context.get("initial") == "value"
    
    def test_update_method(self):
        """Test update method."""
        context = Context(data={"initial": "value"})
        result = context.update({"key1": "value1", "key2": "value2"})
        
        assert result is context  # Returns self
        assert context.get("key1") == "value1"
        assert context.get("key2") == "value2"
        assert context.get("initial") == "value"
    
    def test_update_overwrites_existing(self):
        """Test that update overwrites existing keys."""
        context = Context(data={"key": "old_value"})
        context.update({"key": "new_value"})
        
        assert context.get("key") == "new_value"


class TestContextSnapshots:
    """Test context snapshot and rollback functionality."""
    
    def test_create_snapshot(self):
        """Test creating snapshots."""
        context = Context(data={"key": "value"})
        
        snapshot_id = context.create_snapshot()
        assert snapshot_id == 0
        
        snapshot_id2 = context.create_snapshot()
        assert snapshot_id2 == 1
    
    def test_rollback_to_latest_snapshot(self):
        """Test rollback to latest snapshot."""
        context = Context(data={"key": "original"})
        context.create_snapshot()
        
        context.set("key", "modified")
        context.set("new_key", "new_value")
        assert context.get("key") == "modified"
        assert context.get("new_key") == "new_value"
        
        result = context.rollback()
        assert result is context  # Returns self
        assert context.get("key") == "original"
        with pytest.raises(KeyError):
            context.get("new_key")
    
    def test_rollback_to_specific_snapshot(self):
        """Test rollback to specific snapshot."""
        context = Context(data={"key": "original"})
        snapshot_id = context.create_snapshot()
        
        context.set("key", "modified1")
        context.create_snapshot()
        
        context.set("key", "modified2")
        
        context.rollback(snapshot_id)
        assert context.get("key") == "original"
    
    def test_rollback_no_snapshots_raises_error(self):
        """Test that rollback without snapshots raises error."""
        context = Context(data={"key": "value"})
        
        with pytest.raises(ValueError, match="No snapshots available for rollback"):
            context.rollback()
    
    def test_rollback_invalid_snapshot_id_raises_error(self):
        """Test that invalid snapshot ID raises error."""
        context = Context(data={"key": "value"})
        context.create_snapshot()
        
        with pytest.raises(ValueError, match="Invalid snapshot ID: 5"):
            context.rollback(5)
        
        with pytest.raises(ValueError, match="Invalid snapshot ID: -1"):
            context.rollback(-1)
    
    def test_clear_snapshots(self):
        """Test clearing snapshots."""
        context = Context(data={"key": "value"})
        context.create_snapshot()
        context.create_snapshot()
        
        result = context.clear_snapshots()
        assert result is context  # Returns self
        
        with pytest.raises(ValueError, match="No snapshots available for rollback"):
            context.rollback()
    
    def test_snapshot_isolation(self):
        """Test that snapshots are isolated from each other."""
        context = Context(data={"key": "original"})
        
        snapshot1 = context.create_snapshot()
        context.set("key", "modified1")
        
        snapshot2 = context.create_snapshot()
        context.set("key", "modified2")
        
        # Rollback to snapshot2
        context.rollback(snapshot2)
        assert context.get("key") == "modified1"
        
        # Rollback to snapshot1
        context.rollback(snapshot1)
        assert context.get("key") == "original"


class TestContextCopy:
    """Test context copying functionality."""
    
    def test_copy_with_model_type(self):
        """Test copying context with model type."""
        model = SimpleModel(name="test", value=42)
        original = Context(data=model)
        
        copy = original.copy()
        
        assert copy is not original
        assert copy.data == original.data
        assert copy.model_type == original.model_type
        assert copy.name == "test"
        assert copy.value == 42
    
    def test_copy_without_model_type(self):
        """Test copying context without model type."""
        original = Context(data={"key": "value"})
        
        copy = original.copy()
        
        assert copy is not original
        assert copy.data == original.data
        assert copy.model_type is None
        assert copy.key == "value"
    
    def test_copy_isolation(self):
        """Test that copied contexts are isolated."""
        original = Context(data={"key": "original"})
        copy = original.copy()
        
        copy.set("key", "modified")
        copy.set("new_key", "new_value")
        
        assert original.get("key") == "original"
        with pytest.raises(KeyError):
            original.get("new_key")
        assert copy.get("key") == "modified"
        assert copy.get("new_key") == "new_value"
    
    def test_copy_with_invalid_data(self):
        """Test copying when internal data becomes invalid."""
        model = InvalidModel(required_field="VALID", positive_number=1)
        context = Context(data=model)
        
        # Corrupt the internal data to make it invalid
        context._data["required_field"] = "invalid_lowercase"
        context._data["positive_number"] = -1
        
        # Copy should handle invalid data gracefully
        copy = context.copy()
        assert copy.model_type is None  # Falls back to no model type


class TestContextAsModel:
    """Test converting context back to model."""
    
    def test_as_model_with_valid_data(self):
        """Test converting context to model with valid data."""
        model = ComplexModel(title="Test", count=5, tags=["a", "b"])
        context = Context(data=model)
        
        converted = context.as_model()
        assert isinstance(converted, ComplexModel)
        assert converted.title == "Test"
        assert converted.count == 5
        assert converted.tags == ["a", "b"]
    
    def test_as_model_without_model_type(self):
        """Test as_model returns None when no model type."""
        context = Context(data={"key": "value"})
        
        result = context.as_model()
        assert result is None
    
    def test_as_model_with_invalid_data(self):
        """Test as_model raises error when data is invalid."""
        model = InvalidModel(required_field="VALID", positive_number=1)
        context = Context(data=model)
        
        # Corrupt the data
        context.set("required_field", "invalid_lowercase")
        context.set("positive_number", -1)
        
        with pytest.raises(ValueError, match="Failed to convert context data"):
            context.as_model()
    
    def test_as_model_after_modifications(self):
        """Test as_model works after valid modifications."""
        model = ComplexModel(title="Original", count=1)
        context = Context(data=model)
        
        # Make valid modifications
        context.set("title", "Modified")
        context.set("count", 2)
        context.set("tags", ["new", "tags"])
        
        converted = context.as_model()
        assert isinstance(converted, ComplexModel)
        assert converted.title == "Modified"
        assert converted.count == 2
        assert converted.tags == ["new", "tags"]


class TestContextMerge:
    """Test context merging functionality."""
    
    def test_merge_basic(self):
        """Test basic context merging."""
        context1 = Context(data={"a": 1, "b": 2})
        context2 = Context(data={"b": 3, "c": 4})
        
        merged = context1.merge(context2)
        
        assert merged.get("a") == 1
        assert merged.get("b") == 3  # context2 overwrites
        assert merged.get("c") == 4
        
        # Original contexts unchanged
        assert context1.get("b") == 2
        with pytest.raises(KeyError):
            context1.get("c")
        with pytest.raises(KeyError):
            context2.get("a")
    
    def test_merge_with_model_types(self):
        """Test merging contexts with model types."""
        model1 = SimpleModel(name="first", value=1)
        model2 = SimpleModel(name="second", value=2)
        
        context1 = Context(data=model1)
        context2 = Context(data=model2)
        
        merged = context1.merge(context2)
        
        assert merged.get("name") == "second"
        assert merged.get("value") == 2
        # Note: merged context loses model type information
        assert merged.model_type is None
    
    def test_merge_empty_contexts(self):
        """Test merging empty contexts."""
        context1 = Context()
        context2 = Context()
        
        merged = context1.merge(context2)
        
        assert merged.data == {}
        assert merged.model_type is None


class TestContextStringRepresentation:
    """Test context string representation."""
    
    def test_context_str_basic(self):
        """Test basic string representation."""
        context = Context(data={"key": "value"})
        str_repr = str(context)
        assert "Context" in str_repr
        assert "key" in str_repr


class TestContextEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_deep_copy_in_snapshots(self):
        """Test that snapshots use deep copies."""
        context = Context(data={"nested": {"key": "value"}})
        context.create_snapshot()
        
        # Modify nested data
        context.data["nested"]["key"] = "modified"
        
        context.rollback()
        assert context.data["nested"]["key"] == "value"
    
    def test_context_with_none_values(self):
        """Test context with None values."""
        context = Context(data={"key": None, "other": "value"})
        
        # Removed redundant context.get() test - strict validation
        assert context.key is None
        assert context.get("other") == "value"
    
    def test_context_with_complex_data_types(self):
        """Test context with complex data types."""
        data = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "tuple": (1, 2, 3),
            "set": {1, 2, 3}
        }
        context = Context(data=data)
        
        assert context.list == [1, 2, 3]
        assert context.dict == {"nested": "value"}
        assert context.tuple == (1, 2, 3)
        assert context.get("set") == {1, 2, 3}
    
    def test_large_snapshot_count(self):
        """Test behavior with many snapshots."""
        context = Context(data={"counter": 0})
        
        # Create many snapshots
        for i in range(100):
            context.set("counter", i)
            context.create_snapshot()
        
        # Rollback to early snapshot  
        context.rollback(10)
        assert context.get("counter") == 10
        
        # Clear all snapshots
        context.clear_snapshots()
        with pytest.raises(ValueError):
            context.rollback()


if __name__ == "__main__":
    pytest.main([__file__])