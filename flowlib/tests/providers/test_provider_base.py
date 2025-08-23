"""Tests for provider base functionality."""

import pytest
from pydantic import ValidationError
from typing import Any

from flowlib.providers.core.provider_base import ProviderBase
from pydantic import BaseModel

class MockSettings(BaseModel):
    """Mock settings for provider base tests."""
    model_config = {"frozen": True}
    bucket: str = "test"


class TestProviderBase:
    """Test ProviderBase Pydantic model."""
    
    def test_provider_base_creation_valid(self):
        """Test creating ProviderBase with valid data."""
        provider = ProviderBase(
            name="test_provider",
            provider_type="llm",
            settings={"param1": "value1"}
        )
        
        assert provider.name == "test_provider"
        assert provider.provider_type == "llm"
        assert provider.settings == {"param1": "value1"}
    
    def test_provider_base_creation_minimal(self):
        """Test creating ProviderBase with minimal required fields."""
        provider = ProviderBase(
            name="minimal_provider",
            provider_type="database",
            settings=None
        )
        
        assert provider.name == "minimal_provider"
        assert provider.provider_type == "database"
        assert provider.settings is None
    
    def test_provider_base_missing_name(self):
        """Test that missing name raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ProviderBase(
                provider_type="llm",
                settings={}
            )
        
        errors = exc_info.value.errors()
        assert any(error['loc'] == ('name',) for error in errors)
        assert any(error['type'] == 'missing' for error in errors)
    
    def test_provider_base_missing_provider_type(self):
        """Test that missing provider_type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ProviderBase(
                name="test_provider",
                settings={}
            )
        
        errors = exc_info.value.errors()
        assert any(error['loc'] == ('provider_type',) for error in errors)
        assert any(error['type'] == 'missing' for error in errors)
    
    def test_provider_base_missing_settings(self):
        """Test that missing settings raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ProviderBase(
                name="test_provider",
                provider_type="llm"
            )
        
        errors = exc_info.value.errors()
        assert any(error['loc'] == ('settings',) for error in errors)
        assert any(error['type'] == 'missing' for error in errors)
    
    def test_provider_base_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            ProviderBase(
                name="test_provider",
                provider_type="llm",
                settings={},
                extra_field="not_allowed"
            )
        
        errors = exc_info.value.errors()
        assert any(error['type'] == 'extra_forbidden' for error in errors)
    
    def test_provider_base_settings_any_type(self):
        """Test that settings accepts any type."""
        # Test with dict
        provider1 = ProviderBase(
            name="dict_provider",
            provider_type="llm",
            settings={"key": "value"}
        )
        assert provider1.settings == {"key": "value"}
        
        # Test with string
        provider2 = ProviderBase(
            name="string_provider",
            provider_type="llm",
            settings="string_settings"
        )
        assert provider2.settings == "string_settings"
        
        # Test with number
        provider3 = ProviderBase(
            name="number_provider",
            provider_type="llm",
            settings=42
        )
        assert provider3.settings == 42
        
        # Test with list
        provider4 = ProviderBase(
            name="list_provider",
            provider_type="llm",
            settings=[1, 2, 3]
        )
        assert provider4.settings == [1, 2, 3]
        
        # Test with complex object
        class CustomSettings:
            def __init__(self, value):
                self.value = value
        
        custom_obj = CustomSettings("test")
        provider5 = ProviderBase(
            name="custom_provider",
            provider_type="llm",
            settings=custom_obj
        )
        assert provider5.settings == custom_obj
    
    def test_provider_base_string_fields_validation(self):
        """Test string field validation."""
        # Test empty string name
        with pytest.raises(ValidationError):
            ProviderBase(
                name="",
                provider_type="llm",
                settings={}
            )
        
        # Test empty string provider_type
        with pytest.raises(ValidationError):
            ProviderBase(
                name="test_provider",
                provider_type="",
                settings={}
            )
    
    def test_provider_base_serialization(self):
        """Test provider serialization and deserialization."""
        original = ProviderBase(
            name="serialize_provider",
            provider_type="vector_db",
            settings={"dim": 128, "metric": "cosine"}
        )
        
        # Test model_dump
        data = original.model_dump()
        expected = {
            "name": "serialize_provider",
            "provider_type": "vector_db",
            "settings": {"dim": 128, "metric": "cosine"}
        }
        assert data == expected
        
        # Test model_validate (deserialization)
        recreated = ProviderBase.model_validate(data)
        assert recreated.name == original.name
        assert recreated.provider_type == original.provider_type
        assert recreated.settings == original.settings
    
    def test_provider_base_equality(self):
        """Test provider equality comparison."""
        provider1 = ProviderBase(
            name="equal_provider",
            provider_type="cache",
            settings={"ttl": 300}
        )
        
        provider2 = ProviderBase(
            name="equal_provider",
            provider_type="cache",
            settings={"ttl": 300}
        )
        
        provider3 = ProviderBase(
            name="different_provider",
            provider_type="cache",
            settings={"ttl": 300}
        )
        
        assert provider1 == provider2
        assert provider1 != provider3
    
    def test_provider_base_hash(self):
        """Test provider hashing."""
        settings = MockSettings(bucket="test")
        
        provider1 = ProviderBase(
            name="hash_provider",
            provider_type="storage",
            settings=settings
        )
        
        provider2 = ProviderBase(
            name="hash_provider",
            provider_type="storage",
            settings=settings
        )
        
        # Same providers should have same hash
        assert hash(provider1) == hash(provider2)
        
        # Should be usable in sets
        provider_set = {provider1, provider2}
        assert len(provider_set) == 1  # Should deduplicate
    
    def test_provider_base_immutability(self):
        """Test that ProviderBase instances are immutable after creation."""
        provider = ProviderBase(
            name="immutable_provider",
            provider_type="graph_db",
            settings={"url": "bolt://localhost:7687"}
        )
        
        # Test that fields cannot be modified
        with pytest.raises(ValidationError):
            provider.name = "new_name"
        
        with pytest.raises(ValidationError):
            provider.provider_type = "new_type"
        
        with pytest.raises(ValidationError):
            provider.settings = {"new": "settings"}
    
    def test_provider_base_copy_with_changes(self):
        """Test creating copies with modifications."""
        original = ProviderBase(
            name="original_provider",
            provider_type="embedding",
            settings={"model": "text-embedding-ada-002"}
        )
        
        # Test model_copy with changes
        modified = original.model_copy(update={"name": "modified_provider"})
        
        assert modified.name == "modified_provider"
        assert modified.provider_type == "embedding"  # Unchanged
        assert modified.settings == {"model": "text-embedding-ada-002"}  # Unchanged
        
        # Original should be unchanged
        assert original.name == "original_provider"
    
    def test_provider_base_json_serialization(self):
        """Test JSON serialization and deserialization."""
        provider = ProviderBase(
            name="json_provider",
            provider_type="mq",
            settings={"host": "localhost", "port": 5672}
        )
        
        # Test to JSON
        json_str = provider.model_dump_json()
        assert isinstance(json_str, str)
        assert "json_provider" in json_str
        assert "mq" in json_str
        
        # Test from JSON
        recreated = ProviderBase.model_validate_json(json_str)
        assert recreated == provider
    
    def test_provider_base_nested_settings(self):
        """Test provider with deeply nested settings."""
        nested_settings = {
            "connection": {
                "host": "localhost",
                "port": 5432,
                "credentials": {
                    "username": "user",
                    "password": "pass"
                }
            },
            "pool": {
                "min_size": 5,
                "max_size": 20
            },
            "features": ["ssl", "compression"]
        }
        
        provider = ProviderBase(
            name="nested_provider",
            provider_type="database",
            settings=nested_settings
        )
        
        assert provider.settings["connection"]["host"] == "localhost"
        assert provider.settings["connection"]["credentials"]["username"] == "user"
        assert provider.settings["pool"]["min_size"] == 5
        assert "ssl" in provider.settings["features"]
    
    def test_provider_base_model_config(self):
        """Test that model configuration is properly set."""
        # Test that extra fields are forbidden
        assert ProviderBase.model_config["extra"] == "forbid"
        
        # Verify this is enforced
        with pytest.raises(ValidationError) as exc_info:
            ProviderBase(
                name="test",
                provider_type="test",
                settings={},
                forbidden_extra="value"
            )
        
        errors = exc_info.value.errors()
        assert any(error['type'] == 'extra_forbidden' for error in errors)


class TestProviderBaseEdgeCases:
    """Test edge cases for ProviderBase."""
    
    def test_provider_base_unicode_names(self):
        """Test provider with unicode characters in name."""
        provider = ProviderBase(
            name="测试提供者",  # Chinese characters
            provider_type="llm",
            settings={"model": "gpt-4"}
        )
        
        assert provider.name == "测试提供者"
    
    def test_provider_base_very_long_strings(self):
        """Test provider with very long strings."""
        long_name = "a" * 1000
        long_type = "b" * 500
        
        provider = ProviderBase(
            name=long_name,
            provider_type=long_type,
            settings={"key": "value"}
        )
        
        assert len(provider.name) == 1000
        assert len(provider.provider_type) == 500
    
    def test_provider_base_special_characters(self):
        """Test provider with special characters."""
        provider = ProviderBase(
            name="test-provider_123!@#",
            provider_type="custom.type/with-chars",
            settings={"url": "https://example.com:8080/path?param=value&other=data"}
        )
        
        assert provider.name == "test-provider_123!@#"
        assert provider.provider_type == "custom.type/with-chars"
    
    def test_provider_base_none_in_settings(self):
        """Test provider with None values in settings dict."""
        provider = ProviderBase(
            name="none_settings_provider",
            provider_type="test",
            settings={
                "param1": None,
                "param2": "value",
                "param3": None
            }
        )
        
        assert provider.settings["param1"] is None
        assert provider.settings["param2"] == "value"
        assert provider.settings["param3"] is None
    
    def test_provider_base_circular_reference_in_settings(self):
        """Test provider with circular reference in settings."""
        # Create a circular reference
        settings = {"ref": None}
        settings["ref"] = settings
        
        # This should work - Pydantic allows circular references in Any fields
        provider = ProviderBase(
            name="circular_provider",
            provider_type="test",
            settings=settings
        )
        
        assert provider.settings["ref"] is provider.settings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])