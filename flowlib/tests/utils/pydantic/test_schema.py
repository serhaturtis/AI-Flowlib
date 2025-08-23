"""Tests for Pydantic schema utilities."""

import pytest
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from flowlib.utils.pydantic.schema import model_to_schema_dict, model_to_json_schema


# Test models
class SimpleModel(BaseModel):
    """Simple test model."""
    id: str
    name: str
    value: int = 0


class ComplexModel(BaseModel):
    """Complex test model with various field types."""
    id: str
    name: str = Field(..., description="The name of the item")
    value: int = Field(default=0, ge=0, le=100, description="A value between 0 and 100")
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[datetime] = None
    is_active: bool = True


class NestedModel(BaseModel):
    """Model with nested structure."""
    id: str
    simple: SimpleModel
    complex_list: List[ComplexModel] = Field(default_factory=list)
    optional_simple: Optional[SimpleModel] = None


class TestModelToSchemaDict:
    """Test model to schema dictionary conversion."""
    
    def test_model_to_schema_dict_simple(self):
        """Test converting simple model to schema dict."""
        schema = model_to_schema_dict(SimpleModel)
        
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "id" in schema["properties"]
        assert "name" in schema["properties"]
        assert "value" in schema["properties"]
        
        # Check field types
        assert schema["properties"]["id"]["type"] == "string"
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["value"]["type"] == "integer"
        
        # Check required fields
        assert "required" in schema
        assert "id" in schema["required"]
        assert "name" in schema["required"]
        assert "value" not in schema["required"]  # Has default
    
    def test_model_to_schema_dict_complex(self):
        """Test converting complex model to schema dict."""
        schema = model_to_schema_dict(ComplexModel)
        
        assert isinstance(schema, dict)
        assert "properties" in schema
        
        # Check field with constraints
        value_prop = schema["properties"]["value"]
        assert value_prop["type"] == "integer"
        assert "minimum" in value_prop or "ge" in value_prop
        assert "maximum" in value_prop or "le" in value_prop
        
        # Check array field
        tags_prop = schema["properties"]["tags"]
        assert tags_prop["type"] == "array"
        assert "items" in tags_prop
        
        # Check optional field
        timestamp_prop = schema["properties"]["timestamp"]
        assert "null" in timestamp_prop.get("type", []) or timestamp_prop.get("anyOf", [])
        
        # Check descriptions
        assert "description" in schema["properties"]["name"]
        assert "description" in schema["properties"]["value"]
    
    def test_model_to_schema_dict_nested(self):
        """Test converting nested model to schema dict."""
        schema = model_to_schema_dict(NestedModel)
        
        assert isinstance(schema, dict)
        assert "properties" in schema
        
        # Check nested object
        simple_prop = schema["properties"]["simple"]
        assert simple_prop["type"] == "object" or "$ref" in simple_prop
        
        # Check array of objects
        complex_list_prop = schema["properties"]["complex_list"]
        assert complex_list_prop["type"] == "array"
        
        # Check optional nested object
        optional_simple_prop = schema["properties"]["optional_simple"]
        assert "null" in str(optional_simple_prop) or "anyOf" in optional_simple_prop
    
    def test_model_to_schema_dict_edge_cases(self):
        """Test edge cases for schema dict conversion."""
        # Test with empty model
        class EmptyModel(BaseModel):
            pass
        
        schema = model_to_schema_dict(EmptyModel)
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert len(schema["properties"]) == 0


class TestModelToJsonSchema:
    """Test model to JSON schema conversion."""
    
    def test_model_to_json_schema_simple(self):
        """Test converting simple model to JSON schema."""
        schema = model_to_json_schema(SimpleModel)
        
        assert isinstance(schema, str)
        
        # Should be valid JSON
        import json
        parsed = json.loads(schema)
        assert isinstance(parsed, dict)
        assert "properties" in parsed
    
    def test_model_to_json_schema_complex(self):
        """Test converting complex model to JSON schema."""
        schema = model_to_json_schema(ComplexModel)
        
        assert isinstance(schema, str)
        
        # Should be valid JSON
        import json
        parsed = json.loads(schema)
        assert isinstance(parsed, dict)
        assert "properties" in parsed
        
        # Check that complex fields are properly represented
        assert "tags" in parsed["properties"]
        assert "metadata" in parsed["properties"]
        assert "timestamp" in parsed["properties"]
    
    def test_model_to_json_schema_nested(self):
        """Test converting nested model to JSON schema."""
        schema = model_to_json_schema(NestedModel)
        
        assert isinstance(schema, str)
        
        # Should be valid JSON
        import json
        parsed = json.loads(schema)
        assert isinstance(parsed, dict)
        assert "properties" in parsed
        
        # Check nested structures
        assert "simple" in parsed["properties"]
        assert "complex_list" in parsed["properties"]
        assert "optional_simple" in parsed["properties"]
    
    def test_model_to_json_schema_formatting(self):
        """Test JSON schema formatting options."""
        schema_compact = model_to_json_schema(SimpleModel, indent=None)
        schema_pretty = model_to_json_schema(SimpleModel, indent=2)
        
        assert isinstance(schema_compact, str)
        assert isinstance(schema_pretty, str)
        
        # Pretty version should have more whitespace
        assert len(schema_pretty) > len(schema_compact)
        assert "\n" not in schema_compact
        assert "\n" in schema_pretty
    
    def test_model_to_json_schema_validation(self):
        """Test that generated JSON schema is valid."""
        schema_str = model_to_json_schema(ComplexModel)
        
        import json
        schema = json.loads(schema_str)
        
        # Basic JSON Schema structure validation
        assert "$schema" in schema or "type" in schema
        assert "properties" in schema
        
        # Validate that we can use the schema
        for field_name, field_schema in schema["properties"].items():
            assert isinstance(field_name, str)
            assert isinstance(field_schema, dict)
            assert "type" in field_schema or "$ref" in field_schema or "anyOf" in field_schema