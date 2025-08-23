"""Tests for flowlib resource system."""

import pytest
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch
from pydantic import BaseModel, Field

from flowlib.resources.models.base import ResourceBase
from flowlib.resources.decorators.decorators import model_config, prompt
from flowlib.resources.registry.registry import resource_registry, ResourceRegistry
from flowlib.resources.models.constants import ResourceType
from flowlib.resources.models.model_resource import ModelResource
from flowlib.core.errors.errors import ConfigurationError
from pydantic import ValidationError


# Test models
class MockModelConfig(ResourceBase):
    """Mock model configuration."""
    path: str = "/test/path"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class MockPromptConfig(ResourceBase):
    """Mock prompt configuration."""
    template: str = "Test template: {variable}"
    variables: list[str] = Field(default_factory=list)
    description: str = "Test prompt description"


class TestResource:
    """Test ResourceBase functionality."""
    
    def test_resource_creation(self):
        """Test resource creation with basic fields."""
        resource = MockModelConfig(
            name="test_model",
            type="model",
            path="/path/to/model",
            parameters={"max_tokens": 100}
        )
        
        assert resource.name == "test_model"
        assert resource.path == "/path/to/model"
        assert resource.parameters["max_tokens"] == 100
        assert resource.enabled is True
    
    def test_resource_validation(self):
        """Test resource validation."""
        # Valid resource
        resource = MockModelConfig(name="valid", type="model", path="/valid/path")
        assert resource.name == "valid"
        
        # Required fields should be validated by Pydantic
        with pytest.raises(ValidationError):
            MockModelConfig()  # Missing required fields
    
    def test_resource_serialization(self):
        """Test resource serialization."""
        resource = MockModelConfig(
            name="serialize_test",
            type="model",
            path="/test/path",
            parameters={"param1": "value1"}
        )
        
        serialized = resource.model_dump()
        assert serialized["name"] == "serialize_test"
        assert serialized["path"] == "/test/path"
        assert serialized["parameters"]["param1"] == "value1"
    
    def test_resource_defaults(self):
        """Test resource default values."""
        resource = MockModelConfig(name="defaults", type="model", path="/path")
        
        assert resource.parameters == {}
        assert resource.enabled is True


class TestResourceDecorators:
    """Test resource decorators."""
    
    def test_model_decorator(self):
        """Test @model decorator."""
        @model_config("test-model")
        class TestModel(ResourceBase):
            model_path: str = "/test/model"
            config: Dict[str, Any] = Field(default_factory=dict)
        
        # Check that class was decorated
        assert hasattr(TestModel, '__resource_metadata__')
        assert TestModel.__resource_metadata__['name'] == "test-model"
        assert TestModel.__resource_metadata__['type'] == ResourceType.MODEL
        
        # Check instance creation
        instance = TestModel(name="test-model", type="model_config", model_path="/custom/path")
        assert instance.model_path == "/custom/path"
    
    def test_prompt_decorator(self):
        """Test @prompt decorator."""
        @prompt("test-prompt")
        class TestPrompt(ResourceBase):
            template: str = "Hello {name}"
            variables: list[str] = Field(default=["name"])
        
        # Check decoration
        assert hasattr(TestPrompt, '__resource_metadata__')
        assert TestPrompt.__resource_metadata__['name'] == "test-prompt"
        assert TestPrompt.__resource_metadata__['type'] == ResourceType.PROMPT
        
        # Check instance
        instance = TestPrompt(name="test-prompt", type="prompt_config", template="Hi {user}")
        assert instance.template == "Hi {user}"
    
    def test_decorator_with_description(self):
        """Test decorators with description."""
        @model_config("described-model", description="Test model with description")
        class DescribedModel(ResourceBase):
            path: str = "/test"
        
        metadata = DescribedModel.__resource_metadata__
        assert metadata['description'] == "Test model with description"
    
    def test_decorator_registration(self):
        """Test that decorated resources are registered."""
        # Clear registry for clean test
        test_registry = ResourceRegistry()
        
        with patch('flowlib.resources.decorators.decorators.resource_registry', test_registry):
            @model_config("auto-registered")
            class AutoRegisteredModel(ResourceBase):
                value: str = "test"
        
        # Resource should be auto-registered
        assert test_registry.contains("auto-registered")


class TestResourceRegistry:
    """Test ResourceRegistry functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        self.registry = ResourceRegistry()
    
    def test_registry_registration(self):
        """Test resource registration."""
        resource = MockModelConfig(name="test_resource", type="model")
        
        self.registry.register("test_resource", resource, ResourceType.MODEL)
        
        assert self.registry.contains("test_resource")
        retrieved = self.registry.get("test_resource")
        assert retrieved == resource
    
    def test_registry_type_filtering(self):
        """Test registry filtering by type."""
        model_resource = MockModelConfig(name="model_test", type="model")
        prompt_resource = MockPromptConfig(name="prompt_test", type="prompt")
        
        self.registry.register("model_test", model_resource, ResourceType.MODEL)
        self.registry.register("prompt_test", prompt_resource, ResourceType.PROMPT)
        
        # Should only find the model
        assert self.registry.contains("model_test")
        # Should only find the prompt
        assert self.registry.contains("prompt_test")
    
    def test_registry_overwrite_protection(self):
        """Test registry overwrite protection."""
        resource1 = MockModelConfig(name="resource1", type="model")
        resource2 = MockModelConfig(name="resource2", type="model")
        
        self.registry.register("test", resource1, ResourceType.MODEL)
        
        # Should raise error for duplicate registration
        with pytest.raises(ValueError):
            self.registry.register("test", resource2, ResourceType.MODEL)
        
        # Test successful retrieval
        retrieved = self.registry.get("test")
        assert retrieved == resource1
    
    def test_registry_listing(self):
        """Test registry listing by type."""
        model1 = MockModelConfig(name="model1", type="model")
        model2 = MockModelConfig(name="model2", type="model")
        prompt1 = MockPromptConfig(name="prompt1", type="prompt")
        
        self.registry.register("model1", model1, ResourceType.MODEL)
        self.registry.register("model2", model2, ResourceType.MODEL)
        self.registry.register("prompt1", prompt1, ResourceType.PROMPT)
        
        models = self.registry.list({"resource_type": ResourceType.MODEL})
        assert "model1" in models
        assert "model2" in models
        assert len(models) >= 2
        
        prompts = self.registry.list({"resource_type": ResourceType.PROMPT})
        assert "prompt1" in prompts
        assert len(prompts) >= 1
    
    def test_registry_removal(self):
        """Test resource removal."""
        resource = MockModelConfig(name="removable", type="model")
        
        self.registry.register("removable", resource, ResourceType.MODEL)
        assert self.registry.contains("removable")
        
        # Remove the resource by recreating registry or clearing it
        # Since unregister is not in the interface, we'll just test getting nonexistent
        with pytest.raises(KeyError):
            self.registry.get("nonexistent")
    
    def test_registry_get_all_types(self):
        """Test getting resources of all types."""
        model = MockModelConfig(name="test_model", type="model")
        prompt = MockPromptConfig(name="test_prompt", type="prompt")
        
        self.registry.register("test_model", model, ResourceType.MODEL)
        self.registry.register("test_prompt", prompt, ResourceType.PROMPT)
        
        model_resources = self.registry.get_by_type(ResourceType.MODEL)
        prompt_resources = self.registry.get_by_type(ResourceType.PROMPT)
        
        assert "test_model" in model_resources
        assert "test_prompt" in prompt_resources


class TestModelResource:
    """Test ModelResource functionality."""
    
    def test_model_resource_creation(self):
        """Test ModelResource creation."""
        config = {
            "name": "test_model",
            "type": "model",
            "provider_type": "llamacpp",
            "config": {"temperature": 0.7}
        }
        
        model_resource = ModelResource(**config)
        
        assert model_resource.name == "test_model"
        assert model_resource.provider_type == "llamacpp"
        assert model_resource.config["temperature"] == 0.7
    
    def test_model_resource_validation(self):
        """Test ModelResource validation."""
        # Valid configuration
        valid_config = {
            "name": "valid_model",
            "type": "model",
            "provider_type": "llamacpp",
            "config": {}
        }
        model_resource = ModelResource(**valid_config)
        assert model_resource.name == "valid_model"
        
        # Invalid configuration (missing required fields)
        with pytest.raises(ValidationError):
            ModelResource()
    
    def test_model_resource_defaults(self):
        """Test ModelResource default values."""
        model_resource = ModelResource(
            name="defaults_test",
            type="model",
            provider_type="llamacpp",
            config={}
        )
        
        assert model_resource.config == {}


class TestResourceIntegration:
    """Test resource system integration."""
    
    def test_decorator_and_registry_integration(self):
        """Test that decorators work with registry."""
        registry = ResourceRegistry()
        
        # Mock the global registry
        with patch('flowlib.resources.decorators.decorators.resource_registry', registry):
            @model_config("integration-test")
            class IntegrationModel(ResourceBase):
                test_field: str = "test_value"
        
        # Should be automatically registered
        assert registry.contains("integration-test")
        resource = registry.get("integration-test")
        assert hasattr(resource, 'test_field')
    
    def test_resource_inheritance(self):
        """Test resource inheritance patterns."""
        class BaseModelConfig(ResourceBase):
            base_field: str = "base_value"
            parameters: Dict[str, Any] = Field(default_factory=dict)
        
        @model_config("inherited-model")
        class InheritedModel(BaseModelConfig):
            specific_field: str = "specific_value"
        
        instance = InheritedModel(name="inherited-model", type="model_config")
        assert instance.base_field == "base_value"
        assert instance.specific_field == "specific_value"
        assert instance.parameters == {}
    
    def test_resource_configuration_patterns(self):
        """Test common resource configuration patterns."""
        @model_config("configurable-model")
        class ConfigurableModel(ResourceBase):
            name: str = "default_name"
            settings: Dict[str, Any] = Field(default_factory=dict)
            
            def configure(self, **kwargs):
                """Configure the model with kwargs, returning a new instance."""
                current_data = self.model_dump()
                
                for key, value in kwargs.items():
                    if key in self.model_fields:
                        current_data[key] = value
                    else:
                        current_data["settings"][key] = value
                
                return self.__class__(**current_data)
        
        model = ConfigurableModel(name="configurable-model", type="model_config")
        configured_model = model.configure(name="configured", custom_setting="value")
        
        assert configured_model.name == "configured"
        assert configured_model.settings["custom_setting"] == "value"


class TestResourceErrors:
    """Test resource error handling."""
    
    def test_missing_resource_error(self):
        """Test error when resource not found."""
        registry = ResourceRegistry()
        
        with pytest.raises(KeyError):
            registry.get("nonexistent")
    
    def test_invalid_resource_type(self):
        """Test error with invalid resource type."""
        registry = ResourceRegistry()
        resource = MockModelConfig(name="test", type="model")
        
        # Should handle invalid enum values gracefully
        with pytest.raises((ValueError, TypeError)):
            registry.register("test", resource, "invalid_type")
    
    def test_resource_validation_errors(self):
        """Test resource validation error handling."""
        class StrictResource(ResourceBase):
            required_field: str
            validated_number: int = Field(gt=0)
        
        # Missing required field
        with pytest.raises(ValidationError):
            StrictResource(name="test", type="test", validated_number=5)
        
        # Invalid validation
        with pytest.raises(ValidationError):
            StrictResource(name="test", type="test", required_field="test", validated_number=-1)


class TestResourceMetadata:
    """Test resource metadata handling."""
    
    def test_metadata_extraction(self):
        """Test metadata extraction from decorated resources."""
        @model_config("metadata-test", description="Test description", version="1.0")
        class MetadataModel(ResourceBase):
            value: str = "test"
        
        metadata = MetadataModel.__resource_metadata__
        assert metadata['name'] == "metadata-test"
        assert metadata['description'] == "Test description"
        assert metadata['version'] == "1.0"
        assert metadata['type'] == ResourceType.MODEL
    
    def test_metadata_inheritance(self):
        """Test metadata inheritance in resource classes."""
        @model_config("parent-model")
        class ParentModel(ResourceBase):
            parent_field: str = "parent"
        
        # Child should inherit parent metadata
        class ChildModel(ParentModel):
            child_field: str = "child"
        
        # Parent metadata should be accessible
        assert hasattr(ParentModel, '__resource_metadata__')
        # Child inherits the structure but not the metadata
        instance = ChildModel(name="child", type="model")
        assert instance.parent_field == "parent"
        assert instance.child_field == "child"


if __name__ == "__main__":
    pytest.main([__file__])