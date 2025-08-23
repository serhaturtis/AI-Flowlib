"""Tests for resource registry implementation."""
import pytest
from typing import Any, Dict, List
from unittest.mock import Mock, patch

from flowlib.resources.registry.registry import ResourceRegistry, resource_registry
from flowlib.resources.models.base import ResourceBase
from flowlib.resources.models.constants import ResourceType
# from flowlib.resources.models.model_resource import ModelResource


class MockResourceForRegistry(ResourceBase):
    """Mock resource class for registry testing."""
    value: str = "test"


class AnotherMockResource(ResourceBase):
    """Another mock resource class."""
    number: int = 42


class TestResourceRegistry:
    """Test ResourceRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh registry for testing."""
        return ResourceRegistry()
    
    @pytest.fixture
    def test_resource(self):
        """Create a test resource."""
        return MockResourceForRegistry(name="test_resource", type="test_type", value="test_value")
    
    @pytest.fixture
    def model_resource(self):
        """Create a model resource."""
        return MockResourceForRegistry(
            name="test_model",
            type="model",
            value="model_value"
        )
    
    def test_registry_initialization(self, registry):
        """Test registry initialization."""
        assert isinstance(registry._resources, dict)
        assert isinstance(registry._metadata, dict)
        assert isinstance(registry._resource_types, set)
        assert len(registry._resources) == 0
        assert len(registry._metadata) == 0
        assert len(registry._resource_types) == 0
    
    def test_register_valid_resource(self, registry, test_resource):
        """Test registering a valid resource."""
        registry.register(
            name="test_resource",
            obj=test_resource,
            resource_type="test_type",
            version="1.0",
            author="test"
        )
        
        # Verify resource is stored
        key = ("test_type", "test_resource")
        assert key in registry._resources
        assert registry._resources[key] == test_resource
        
        # Verify metadata is stored
        assert key in registry._metadata
        assert registry._metadata[key] == {"version": "1.0", "author": "test"}
        
        # Verify resource type is tracked
        assert "test_type" in registry._resource_types
    
    def test_register_invalid_resource(self, registry):
        """Test registering an invalid resource."""
        class InvalidResource:
            pass
        
        invalid = InvalidResource()
        
        with pytest.raises(TypeError) as exc_info:
            registry.register("invalid", invalid, "test_type")
        
        assert "must be a ResourceBase subclass" in str(exc_info.value)
        assert len(registry._resources) == 0
    
    def test_register_duplicate_resource(self, registry, test_resource):
        """Test registering a duplicate resource."""
        # Register first time
        registry.register("test_resource", test_resource, "test_type")
        
        # Try to register again
        duplicate = MockResourceForRegistry(name="test_resource", type="test_type", value="duplicate")
        
        with pytest.raises(ValueError) as exc_info:
            registry.register("test_resource", duplicate, "test_type")
        
        assert "already exists" in str(exc_info.value)
        assert registry._resources[("test_type", "test_resource")] == test_resource
    
    def test_register_same_name_different_type(self, registry, test_resource):
        """Test registering same name with different type."""
        # Register with first type
        registry.register("resource", test_resource, "type1")
        
        # Register same name with different type
        another = AnotherMockResource(name="resource", type="type2", number=100)
        registry.register("resource", another, "type2")
        
        # Both should exist
        assert ("type1", "resource") in registry._resources
        assert ("type2", "resource") in registry._resources
        assert len(registry._resources) == 2
    
    def test_get_existing_resource(self, registry, test_resource):
        """Test getting an existing resource."""
        registry.register("test_resource", test_resource, "test_type")
        
        retrieved = registry.get("test_resource")
        assert retrieved == test_resource
    
    def test_get_nonexistent_resource(self, registry):
        """Test getting a non-existent resource."""
        with pytest.raises(KeyError) as exc_info:
            registry.get("nonexistent")
        
        assert "not found" in str(exc_info.value)
    
    def test_get_with_type_validation(self, registry, test_resource):
        """Test getting resource with type validation."""
        registry.register("test_resource", test_resource, "test_type")
        
        # Valid type
        retrieved = registry.get("test_resource", MockResourceForRegistry)
        assert retrieved == test_resource
        
        # Invalid type
        with pytest.raises(TypeError) as exc_info:
            registry.get("test_resource", AnotherMockResource)
        
        assert "expected AnotherMockResource" in str(exc_info.value)
    
    def test_get_method(self, registry, test_resource):
        """Test resource registry get method."""
        registry.register("test_resource", test_resource, "test_type")
        
        retrieved = registry.get("test_resource")
        assert retrieved == test_resource
    
    def test_get_metadata_existing(self, registry, test_resource):
        """Test getting metadata for existing resource."""
        metadata = {"version": "1.0", "author": "test"}
        registry.register("test_resource", test_resource, "test_type", **metadata)
        
        retrieved_metadata = registry.get_metadata("test_resource")
        assert retrieved_metadata == metadata
    
    def test_get_metadata_nonexistent(self, registry):
        """Test getting metadata for non-existent resource."""
        with pytest.raises(KeyError) as exc_info:
            registry.get_metadata("nonexistent")
        
        assert "not found" in str(exc_info.value)
    
    def test_get_metadata_no_metadata(self, registry, test_resource):
        """Test getting metadata when none was provided."""
        registry.register("test_resource", test_resource, "test_type")
        
        metadata = registry.get_metadata("test_resource")
        assert metadata == {}
    
    def test_contains_existing(self, registry, test_resource):
        """Test contains for existing resource."""
        registry.register("test_resource", test_resource, "test_type")
        
        assert registry.contains("test_resource") is True
    
    def test_contains_nonexistent(self, registry):
        """Test contains for non-existent resource."""
        assert registry.contains("nonexistent") is False
    
    def test_list_all_resources(self, registry, test_resource, model_resource):
        """Test listing all resources."""
        registry.register("test_resource", test_resource, "test_type")
        registry.register("model_resource", model_resource, ResourceType.MODEL_CONFIG)
        
        all_resources = registry.list()
        assert "test_resource" in all_resources
        assert "model_resource" in all_resources
        assert len(all_resources) == 2
    
    def test_list_filtered_resources(self, registry, test_resource, model_resource):
        """Test listing resources with filter."""
        registry.register("test_resource", test_resource, "test_type")
        registry.register("model_resource", model_resource, ResourceType.MODEL_CONFIG)
        
        # Filter by resource type
        filtered = registry.list({"resource_type": "test_type"})
        assert filtered == ["test_resource"]
        
        model_filtered = registry.list({"resource_type": ResourceType.MODEL_CONFIG})
        assert model_filtered == ["model_resource"]
    
    def test_list_types(self, registry, test_resource, model_resource):
        """Test listing resource types."""
        registry.register("test_resource", test_resource, "test_type")
        registry.register("model_resource", model_resource, ResourceType.MODEL_CONFIG)
        
        types = registry.list_types()
        assert "test_type" in types
        assert ResourceType.MODEL_CONFIG in types
        assert len(types) == 2
    
    def test_get_by_type(self, registry, test_resource, model_resource):
        """Test getting resources by type."""
        registry.register("test_resource1", test_resource, "test_type")
        registry.register("test_resource2", test_resource, "test_type")
        registry.register("model_resource", model_resource, ResourceType.MODEL_CONFIG)
        
        test_resources = registry.get_by_type("test_type")
        assert len(test_resources) == 2
        assert "test_resource1" in test_resources
        assert "test_resource2" in test_resources
        
        model_resources = registry.get_by_type(ResourceType.MODEL_CONFIG)
        assert len(model_resources) == 1
        assert "model_resource" in model_resources
    
    def test_get_typed_with_specific_type(self, registry, test_resource):
        """Test getting resource with type casting and specific resource type."""
        registry.register("test_resource", test_resource, "test_type")
        
        typed_resource = registry.get("test_resource", MockResourceForRegistry)
        assert typed_resource == test_resource
        assert isinstance(typed_resource, MockResourceForRegistry)
    
    def test_get_typed_search_all_types(self, registry, test_resource):
        """Test getting resource by searching all types."""
        registry.register("test_resource", test_resource, "test_type")
        
        typed_resource = registry.get("test_resource", MockResourceForRegistry)
        assert typed_resource == test_resource
        assert isinstance(typed_resource, MockResourceForRegistry)
    
    def test_get_typed_wrong_type(self, registry, test_resource):
        """Test getting resource with wrong type."""
        registry.register("test_resource", test_resource, "test_type")
        
        with pytest.raises(TypeError) as exc_info:
            registry.get("test_resource", AnotherMockResource)
        
        assert "expected AnotherMockResource" in str(exc_info.value)
    
    def test_get_typed_nonexistent(self, registry):
        """Test getting non-existent resource with typing."""
        with pytest.raises(KeyError) as exc_info:
            registry.get("nonexistent", MockResourceForRegistry)
        
        assert "not found" in str(exc_info.value)
    
    def test_multiple_resources_same_type(self, registry):
        """Test handling multiple resources of the same type."""
        resource1 = MockResourceForRegistry(name="resource1", type="test", value="value1")
        resource2 = MockResourceForRegistry(name="resource2", type="test", value="value2")
        
        registry.register("resource1", resource1, "test_type")
        registry.register("resource2", resource2, "test_type")
        
        assert registry.get("resource1").value == "value1"
        assert registry.get("resource2").value == "value2"
        
        by_type = registry.get_by_type("test_type")
        assert len(by_type) == 2
    
    def test_registry_inheritance_compatibility(self, registry):
        """Test registry works with resource inheritance."""
        class ChildResource(MockResourceForRegistry):
            extra_field: str = "extra"
        
        child = ChildResource(name="child", type="test", value="child_value", extra_field="extra_value")
        registry.register("child", child, "child_type")
        
        # Should be retrievable as parent type
        retrieved = registry.get("child", MockResourceForRegistry)
        assert isinstance(retrieved, ChildResource)
        assert retrieved.value == "child_value"
        assert retrieved.extra_field == "extra_value"


class TestResourceRegistryIntegration:
    """Integration tests for ResourceRegistry."""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh registry for testing."""
        return ResourceRegistry()
    
    def test_default_resource_types(self, registry):
        """Test registry with default resource types."""
        model = MockResourceForRegistry(
            name="test_model",
            type="model",
            value="model_value"
        )
        
        registry.register("test_model", model, ResourceType.MODEL_CONFIG)
        
        # Should be retrievable
        retrieved = registry.get("test_model")
        assert retrieved == model
        
        # Should appear in type listing
        assert ResourceType.MODEL_CONFIG in registry.list_types()
    
    def test_registry_with_all_resource_types(self, registry):
        """Test registry with all standard resource types."""
        resources = {}
        
        for resource_type in ResourceType:
            resource = MockResourceForRegistry(
                name=f"resource_{resource_type.value}",
                type=resource_type.value,
                value=f"value_{resource_type.value}"
            )
            resources[resource_type] = resource
            registry.register(f"resource_{resource_type.value}", resource, resource_type)
        
        # All should be retrievable
        for resource_type, expected_resource in resources.items():
            retrieved = registry.get(f"resource_{resource_type.value}")
            assert retrieved == expected_resource
        
        # All types should be listed
        listed_types = registry.list_types()
        for resource_type in ResourceType:
            assert resource_type in listed_types
    
    def test_registry_metadata_integration(self, registry):
        """Test registry metadata functionality."""
        resource = MockResourceForRegistry(name="meta_resource", type="test", value="meta_value")
        
        metadata = {
            "version": "1.0.0",
            "author": "test_author",
            "description": "Test resource with metadata",
            "tags": ["test", "example"],
            "created_at": "2024-01-01"
        }
        
        registry.register("meta_resource", resource, "meta_type", **metadata)
        
        retrieved_metadata = registry.get_metadata("meta_resource")
        assert retrieved_metadata == metadata
        assert retrieved_metadata["version"] == "1.0.0"
        assert "test" in retrieved_metadata["tags"]


class TestGlobalResourceRegistry:
    """Test the global resource registry instance."""
    
    def test_global_registry_exists(self):
        """Test that global registry exists and is correct type."""
        assert resource_registry is not None
        assert isinstance(resource_registry, ResourceRegistry)
    
    def test_global_registry_isolation(self):
        """Test that global registry doesn't interfere with local ones."""
        # Create local registry
        local_registry = ResourceRegistry()
        
        # They should be different instances
        assert local_registry is not resource_registry
        
        # Register in local registry
        resource = MockResourceForRegistry(name="local_resource", type="test", value="local")
        local_registry.register("local_resource", resource, "test_type")
        
        # Should not appear in global registry
        assert not resource_registry.contains("local_resource")
        assert local_registry.contains("local_resource")
    
    @pytest.mark.integration
    def test_global_registry_persistence(self):
        """Test that global registry persists across imports."""
        # This would test that the registry maintains state
        # In a real scenario, this would involve multiple modules
        
        # Register a resource
        resource = MockResourceForRegistry(name="persistent_resource", type="test", value="persistent")
        resource_registry.register("persistent_resource", resource, "test_type")
        
        # Should be retrievable
        assert resource_registry.contains("persistent_resource")
        retrieved = resource_registry.get("persistent_resource")
        assert retrieved.value == "persistent"


class TestResourceRegistryErrorHandling:
    """Test error handling in ResourceRegistry."""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh registry for testing."""
        return ResourceRegistry()
    
    @pytest.fixture
    def test_resource(self):
        """Create a test resource."""
        return MockResourceForRegistry(name="test_resource", type="test_type", value="test_value")
    
    def test_register_none_resource(self, registry):
        """Test registering None as resource."""
        with pytest.raises(TypeError):
            registry.register("none_resource", None, "test_type")
    
    def test_register_empty_name(self, registry, test_resource):
        """Test registering with empty name."""
        # Empty string should be allowed but not None
        registry.register("", test_resource, "test_type")
        assert registry.contains("")
    
    def test_get_with_none_type(self, registry):
        """Test getting resource with None type."""
        # This should work - None is a valid resource_type value
        with pytest.raises(KeyError):  # Resource doesn't exist
            registry.get("test")
    
    def test_complex_metadata_types(self, registry, test_resource):
        """Test registry with complex metadata types."""
        complex_metadata = {
            "nested_dict": {"key": "value", "number": 42},
            "list_data": [1, 2, 3, "string"],
            "tuple_data": (1, 2, 3),
            "boolean_flag": True,
            "none_value": None
        }
        
        registry.register("complex_resource", test_resource, "test_type", **complex_metadata)
        
        retrieved_metadata = registry.get_metadata("complex_resource")
        assert retrieved_metadata["nested_dict"]["key"] == "value"
        assert retrieved_metadata["list_data"] == [1, 2, 3, "string"]
        assert retrieved_metadata["boolean_flag"] is True
        assert retrieved_metadata["none_value"] is None