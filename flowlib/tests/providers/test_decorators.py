"""Comprehensive tests for provider decorators module."""

import pytest
from typing import Optional, Dict, Any
from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel, Field, ValidationError

from flowlib.providers.core.decorators import (
    provider,
    llm_provider,
    db_provider,
    vector_db_provider,
    cache_provider,
    storage_provider,
    message_queue_provider
)
from flowlib.providers.core.provider_base import ProviderBase


# Test helper classes
class MockSettings(BaseModel):
    """Mock settings class for provider testing."""
    host: str = Field(default="localhost", description="Host address")
    port: int = Field(default=8080, description="Port number")
    timeout: float = Field(default=30.0, description="Timeout in seconds")
    enabled: bool = Field(default=True, description="Whether provider is enabled")


class MockSettingsWithDefaults(BaseModel):
    """Mock settings with all defaults."""
    name: str = Field(default="default_name")
    value: int = Field(default=42)


class MockSettingsRequiredFields(BaseModel):
    """Mock settings with required fields."""
    required_field: str = Field(..., description="Required field")
    optional_field: str = Field(default="optional", description="Optional field")


class MockProvider(ProviderBase):
    """Mock provider class for testing."""
    
    def __init__(self, name: str, provider_type: str = "test", settings: MockSettings = None, **kwargs):
        super().__init__(name=name, provider_type=provider_type, settings=settings)
        self.extra_kwargs = kwargs


class MockInvalidProvider:
    """Invalid provider class that doesn't inherit from ProviderBase."""
    pass


class MockProviderDecorator:
    """Test the main provider decorator."""
    
    def test_provider_decorator_basic_registration(self):
        """Test basic provider registration."""
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @provider(name="test_provider", settings_class=MockSettings)
            class BasicMockProvider(ProviderBase[MockSettings]):
                pass
            
            # Check that registration was called
            mock_registry.register_factory.assert_called_once()
            call_args = mock_registry.register_factory.call_args
            
            assert call_args.kwargs["name"] == "test_provider"
            assert call_args.kwargs["provider_type"] == "llm"  # default
            assert call_args.kwargs["settings_class"] == MockSettings
            
            # Check class attributes
            assert BasicMockProvider.__provider_name__ == "test_provider"
            assert BasicMockProvider.__provider_type__ == "llm"
            assert BasicMockProvider.__provider_metadata__ == {}
    
    def test_provider_decorator_with_custom_type(self):
        """Test provider registration with custom type."""
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @provider(name="custom_provider", provider_type="custom", settings_class=MockSettings)
            class CustomMockProvider(ProviderBase[MockSettings]):
                pass
            
            call_args = mock_registry.register_factory.call_args
            assert call_args.kwargs["provider_type"] == "custom"
            assert CustomMockProvider.__provider_type__ == "custom"
    
    def test_provider_decorator_with_metadata(self):
        """Test provider registration with metadata."""
        metadata = {
            "description": "Test provider",
            "version": "1.0",
            "author": "Test Author"
        }
        
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @provider(name="meta_provider", settings_class=MockSettings, **metadata)
            class MetaMockProvider(ProviderBase[MockSettings]):
                pass
            
            call_args = mock_registry.register_factory.call_args
            assert call_args.kwargs["description"] == "Test provider"
            assert call_args.kwargs["version"] == "1.0"
            assert call_args.kwargs["author"] == "Test Author"
            
            assert MetaMockProvider.__provider_metadata__ == metadata
    
    def test_provider_decorator_missing_settings_class(self):
        """Test that decorator fails when settings_class is missing."""
        with pytest.raises(TypeError) as exc_info:
            @provider(name="no_settings_provider")
            class NoSettingsProvider(ProviderBase):
                pass
        
        assert "must supply a 'settings_class' argument" in str(exc_info.value)
    
    def test_provider_decorator_invalid_class_type(self):
        """Test that decorator fails for non-ProviderBase classes."""
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            with pytest.raises(TypeError) as exc_info:
                @provider(name="invalid_provider", settings_class=MockSettings)
                class InvalidProvider:
                    pass
            
            assert "must be a ProviderBase subclass" in str(exc_info.value)
    
    def test_provider_decorator_no_registry(self):
        """Test that decorator fails when registry is not initialized."""
        with patch('flowlib.providers.decorators.provider_registry', None):
            with pytest.raises(RuntimeError) as exc_info:
                @provider(name="no_registry_provider", settings_class=MockSettings)
                class NoRegistryProvider(ProviderBase[MockSettings]):
                    pass
            
            assert "Provider registry not initialized" in str(exc_info.value)


class MockProviderFactory:
    """Test the factory function created by the decorator."""
    
    def test_factory_with_runtime_settings_dict(self):
        """Test factory with runtime settings dictionary."""
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @provider(name="runtime_provider", settings_class=MockSettings)
            class RuntimeMockProvider(ProviderBase[MockSettings]):
                def __init__(self, name: str, provider_type: str = "test", settings: MockSettings = None):
                    super().__init__(name=name, provider_type=provider_type, settings=settings)
            
            # Get the factory function
            factory_call = mock_registry.register_factory.call_args
            factory_func = factory_call.kwargs["factory"]
            
            # Test factory with runtime settings
            runtime_settings = {"host": "custom.host", "port": 9090, "timeout": 60.0}
            instance = factory_func(runtime_settings_dict=runtime_settings)
            
            assert instance.name == "runtime_provider"
            assert instance.settings.host == "custom.host"
            assert instance.settings.port == 9090
            assert instance.settings.timeout == 60.0
    
    def test_factory_with_invalid_runtime_settings(self):
        """Test factory with invalid runtime settings."""
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @provider(name="invalid_runtime_provider", settings_class=MockSettingsRequiredFields)
            class InvalidRuntimeProvider(ProviderBase[MockSettingsRequiredFields]):
                pass
            
            factory_call = mock_registry.register_factory.call_args
            factory_func = factory_call.kwargs["factory"]
            
            # Test with missing required field
            with pytest.raises(ValueError) as exc_info:
                factory_func(runtime_settings_dict={"optional_field": "test"})
            
            assert "Error parsing runtime_settings" in str(exc_info.value)
    
    def test_factory_with_decorator_settings_dict(self):
        """Test factory with settings defined in decorator."""
        decorator_settings = {"host": "decorator.host", "port": 7070}
        
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @provider(
                name="decorator_provider",
                settings_class=MockSettings,
                settings=decorator_settings
            )
            class DecoratorMockProvider(ProviderBase[MockSettings]):
                def __init__(self, name: str, provider_type: str = "test", settings: MockSettings = None):
                    super().__init__(name=name, provider_type=provider_type, settings=settings)
            
            factory_call = mock_registry.register_factory.call_args
            factory_func = factory_call.kwargs["factory"]
            
            instance = factory_func()
            
            assert instance.settings.host == "decorator.host"
            assert instance.settings.port == 7070
            assert instance.settings.timeout == 30.0  # default value
    
    def test_factory_with_decorator_settings_instance(self):
        """Test factory with settings instance defined in decorator."""
        decorator_settings = MockSettings(host="instance.host", port=8888)
        
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @provider(
                name="instance_provider",
                settings_class=MockSettings,
                settings=decorator_settings
            )
            class InstanceMockProvider(ProviderBase[MockSettings]):
                def __init__(self, name: str, provider_type: str = "test", settings: MockSettings = None):
                    super().__init__(name=name, provider_type=provider_type, settings=settings)
            
            factory_call = mock_registry.register_factory.call_args
            factory_func = factory_call.kwargs["factory"]
            
            instance = factory_func()
            
            assert instance.settings.host == "instance.host"
            assert instance.settings.port == 8888
    
    def test_factory_with_default_settings(self):
        """Test factory with default settings from Pydantic model."""
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @provider(name="default_provider", settings_class=MockSettingsWithDefaults)
            class DefaultMockProvider(ProviderBase[MockSettingsWithDefaults]):
                def __init__(self, name: str, provider_type: str = "test", settings: MockSettingsWithDefaults = None):
                    super().__init__(name=name, provider_type=provider_type, settings=settings)
            
            factory_call = mock_registry.register_factory.call_args
            factory_func = factory_call.kwargs["factory"]
            
            instance = factory_func()
            
            assert instance.settings.name == "default_name"
            assert instance.settings.value == 42
    
    def test_factory_runtime_settings_priority(self):
        """Test that runtime settings take priority over decorator settings."""
        decorator_settings = {"host": "decorator.host", "port": 7070}
        runtime_settings = {"host": "runtime.host", "port": 9090}
        
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @provider(
                name="priority_provider",
                settings_class=MockSettings,
                settings=decorator_settings
            )
            class PriorityMockProvider(ProviderBase[MockSettings]):
                def __init__(self, name: str, provider_type: str = "test", settings: MockSettings = None):
                    super().__init__(name=name, provider_type=provider_type, settings=settings)
            
            factory_call = mock_registry.register_factory.call_args
            factory_func = factory_call.kwargs["factory"]
            
            instance = factory_func(runtime_settings_dict=runtime_settings)
            
            # Runtime settings should override decorator settings
            assert instance.settings.host == "runtime.host"
            assert instance.settings.port == 9090
    
    def test_factory_with_extra_metadata_kwargs(self):
        """Test that extra metadata kwargs are passed to provider."""
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @provider(
                name="kwargs_provider",
                settings_class=MockSettings,
                description="Test description",
                version="1.0"
            )
            class KwargsMockProvider(ProviderBase[MockSettings]):
                def __init__(self, name: str, provider_type: str = "test", settings: MockSettings = None, **kwargs):
                    super().__init__(name=name, provider_type=provider_type, settings=settings)
                    # Use object.__setattr__ to bypass frozen model restriction
                    object.__setattr__(self, 'extra_kwargs', kwargs)
            
            factory_call = mock_registry.register_factory.call_args
            factory_func = factory_call.kwargs["factory"]
            
            instance = factory_func()
            
            # Extra kwargs should be passed (excluding 'settings')
            assert "description" in instance.extra_kwargs
            assert instance.extra_kwargs["description"] == "Test description"
            assert instance.extra_kwargs["version"] == "1.0"
            assert "settings" not in instance.extra_kwargs  # Should be excluded
    
    def test_factory_invalid_decorator_settings_type(self):
        """Test factory with invalid decorator settings type."""
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            with pytest.raises(TypeError):
                @provider(
                    name="invalid_settings_provider",
                    settings_class=MockSettings,
                    settings="invalid_settings_type"  # Should be dict or MockSettings instance
                )
                class InvalidSettingsProvider(ProviderBase[MockSettings]):
                    pass
    
    def test_factory_no_settings_class_fallback(self):
        """Test factory behavior when no settings_class is provided."""
        # This should not be possible due to the decorator check, but test the factory logic
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            # Mock the decorator to skip the settings_class check
            def mock_decorator(cls):
                def factory(runtime_settings_dict=None):
                    # Simulate factory without settings_class
                    final_settings_arg = runtime_settings_dict or {}
                    return cls(name="no_settings", provider_type="test", settings=final_settings_arg)
                
                mock_registry.register_factory(
                    name="no_settings_provider",
                    factory=factory,
                    provider_type="test",
                    settings_class=None
                )
                return cls
            
            @mock_decorator
            class NoSettingsClassProvider(ProviderBase):
                def __init__(self, name: str, provider_type: str = "test", settings=None):
                    super().__init__(name=name, provider_type=provider_type, settings=settings)


class TestSpecializedDecorators:
    """Test specialized provider decorators."""
    
    def test_llm_provider_decorator(self):
        """Test LLM provider decorator."""
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @llm_provider(name="test_llm", settings_class=MockSettings)
            class TestLLMProvider(ProviderBase[MockSettings]):
                pass
            
            call_args = mock_registry.register_factory.call_args
            assert call_args.kwargs["name"] == "test_llm"
            assert call_args.kwargs["provider_type"] == "llm"
            assert TestLLMProvider.__provider_type__ == "llm"
    
    def test_db_provider_decorator(self):
        """Test database provider decorator."""
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @db_provider(name="test_db", settings_class=MockSettings)
            class TestDBProvider(ProviderBase[MockSettings]):
                pass
            
            call_args = mock_registry.register_factory.call_args
            assert call_args.kwargs["provider_type"] == "database"
            assert TestDBProvider.__provider_type__ == "database"
    
    def test_vector_db_provider_decorator(self):
        """Test vector database provider decorator."""
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @vector_db_provider(name="test_vector", settings_class=MockSettings)
            class TestVectorDBProvider(ProviderBase[MockSettings]):
                pass
            
            call_args = mock_registry.register_factory.call_args
            assert call_args.kwargs["provider_type"] == "vector_db"
            assert TestVectorDBProvider.__provider_type__ == "vector_db"
    
    def test_cache_provider_decorator(self):
        """Test cache provider decorator."""
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @cache_provider(name="test_cache", settings_class=MockSettings)
            class TestCacheProvider(ProviderBase[MockSettings]):
                pass
            
            call_args = mock_registry.register_factory.call_args
            assert call_args.kwargs["provider_type"] == "cache"
            assert TestCacheProvider.__provider_type__ == "cache"
    
    def test_storage_provider_decorator(self):
        """Test storage provider decorator."""
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @storage_provider(name="test_storage", settings_class=MockSettings)
            class TestStorageProvider(ProviderBase[MockSettings]):
                pass
            
            call_args = mock_registry.register_factory.call_args
            assert call_args.kwargs["provider_type"] == "storage"
            assert TestStorageProvider.__provider_type__ == "storage"
    
    def test_message_queue_provider_decorator(self):
        """Test message queue provider decorator."""
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @message_queue_provider(name="test_mq", settings_class=MockSettings)
            class TestMQProvider(ProviderBase[MockSettings]):
                pass
            
            call_args = mock_registry.register_factory.call_args
            assert call_args.kwargs["provider_type"] == "message_queue"
            assert TestMQProvider.__provider_type__ == "message_queue"
    
    def test_specialized_decorators_with_metadata(self):
        """Test specialized decorators with metadata."""
        metadata = {"description": "Test cache provider", "version": "2.0"}
        
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @cache_provider(name="meta_cache", settings_class=MockSettings, **metadata)
            class MetaCacheProvider(ProviderBase[MockSettings]):
                pass
            
            call_args = mock_registry.register_factory.call_args
            assert call_args.kwargs["description"] == "Test cache provider"
            assert call_args.kwargs["version"] == "2.0"
            assert MetaCacheProvider.__provider_metadata__ == metadata


class TestDecoratorIntegration:
    """Test integration aspects of the decorators."""
    
    def test_decorator_with_real_provider_structure(self):
        """Test decorator with realistic provider structure."""
        class DatabaseSettings(BaseModel):
            connection_string: str = Field(..., description="Database connection string")
            pool_size: int = Field(default=10, description="Connection pool size")
            timeout: float = Field(default=30.0, description="Query timeout")
            ssl_enabled: bool = Field(default=True, description="Enable SSL")
        
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @db_provider(
                name="postgres_provider",
                settings_class=DatabaseSettings,
                description="PostgreSQL database provider",
                version="1.0.0",
                author="FlowLib Team"
            )
            class PostgreSQLProvider(ProviderBase[DatabaseSettings]):
                def __init__(self, name: str, provider_type: str = "database", settings: DatabaseSettings = None, **kwargs):
                    # Don't pass extra kwargs to super() as ProviderBase doesn't accept them
                    super().__init__(name=name, provider_type=provider_type, settings=settings)
                    # Use object.__setattr__ to bypass frozen model restriction
                    object.__setattr__(self, 'connection', None)
                    object.__setattr__(self, 'extra_metadata', kwargs)
                
                async def connect(self):
                    """Connect to database."""
                    # Implementation would go here
                    pass
            
            # Verify registration
            mock_registry.register_factory.assert_called_once()
            call_args = mock_registry.register_factory.call_args
            
            assert call_args.kwargs["name"] == "postgres_provider"
            assert call_args.kwargs["provider_type"] == "database"
            assert call_args.kwargs["settings_class"] == DatabaseSettings
            assert call_args.kwargs["description"] == "PostgreSQL database provider"
            
            # Test factory function
            factory_func = call_args.kwargs["factory"]
            runtime_config = {
                "connection_string": "postgresql://localhost:5432/testdb",
                "pool_size": 20,
                "ssl_enabled": False
            }
            
            instance = factory_func(runtime_settings_dict=runtime_config)
            
            assert isinstance(instance, PostgreSQLProvider)
            assert instance.name == "postgres_provider"
            assert instance.settings.connection_string == "postgresql://localhost:5432/testdb"
            assert instance.settings.pool_size == 20
            assert instance.settings.ssl_enabled is False
            assert instance.settings.timeout == 30.0  # default value
    
    def test_multiple_providers_registration(self):
        """Test registering multiple providers."""
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @llm_provider(name="openai_provider", settings_class=MockSettings)
            class OpenAIProvider(ProviderBase[MockSettings]):
                pass
            
            @llm_provider(name="claude_provider", settings_class=MockSettings)
            class ClaudeProvider(ProviderBase):
                pass
            
            @cache_provider(name="redis_provider", settings_class=MockSettings)
            class RedisProvider(ProviderBase[MockSettings]):
                pass
            
            # Should have been called 3 times
            assert mock_registry.register_factory.call_count == 3
            
            # Check that each provider has correct attributes
            assert OpenAIProvider.__provider_name__ == "openai_provider"
            assert OpenAIProvider.__provider_type__ == "llm"
            
            assert ClaudeProvider.__provider_name__ == "claude_provider"
            assert ClaudeProvider.__provider_type__ == "llm"
            
            assert RedisProvider.__provider_name__ == "redis_provider"
            assert RedisProvider.__provider_type__ == "cache"
    
    def test_decorator_error_propagation(self):
        """Test that errors are properly propagated."""
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            # Test registry error propagation
            mock_registry.register_factory.side_effect = Exception("Registry error")
            
            with pytest.raises(Exception) as exc_info:
                @provider(name="error_provider", settings_class=MockSettings)
                class ErrorProvider(ProviderBase[MockSettings]):
                    pass
            
            assert "Registry error" in str(exc_info.value)
    
    def test_factory_edge_cases(self):
        """Test factory function edge cases."""
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @provider(name="edge_provider", settings_class=MockSettingsWithDefaults)
            class EdgeMockProvider(ProviderBase[MockSettingsWithDefaults]):
                def __init__(self, name: str, provider_type: str = "test", settings: MockSettingsWithDefaults = None):
                    super().__init__(name=name, provider_type=provider_type, settings=settings)
            
            factory_call = mock_registry.register_factory.call_args
            factory_func = factory_call.kwargs["factory"]
            
            # Test with None runtime settings
            instance1 = factory_func(runtime_settings_dict=None)
            assert instance1.settings.name == "default_name"
            
            # Test with empty dict
            instance2 = factory_func(runtime_settings_dict={})
            assert instance2.settings.name == "default_name"
            
            # Test with partial settings
            instance3 = factory_func(runtime_settings_dict={"name": "custom"})
            assert instance3.settings.name == "custom"
            assert instance3.settings.value == 42  # default
    
    def test_settings_validation_edge_cases(self):
        """Test settings validation in various scenarios."""
        class StrictSettings(BaseModel):
            required_str: str = Field(..., min_length=1)
            required_int: int = Field(..., ge=0, le=100)
        
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @provider(name="strict_provider", settings_class=StrictSettings)
            class StrictProvider(ProviderBase[StrictSettings]):
                pass
            
            factory_call = mock_registry.register_factory.call_args
            factory_func = factory_call.kwargs["factory"]
            
            # Test validation failure
            with pytest.raises(ValueError) as exc_info:
                factory_func(runtime_settings_dict={
                    "required_str": "",  # Too short
                    "required_int": 150  # Too large
                })
            
            assert "Error parsing runtime_settings" in str(exc_info.value)
    
    def test_decorator_metadata_handling(self):
        """Test comprehensive metadata handling."""
        complex_metadata = {
            "description": "Complex test provider",
            "version": "2.1.0",
            "author": "Test Team",
            "tags": ["test", "complex"],
            "config": {"feature_flag": True},
            "settings": {"host": "metadata.host"}  # This should be handled specially
        }
        
        with patch('flowlib.providers.decorators.provider_registry') as mock_registry:
            @provider(name="complex_provider", settings_class=MockSettings, **complex_metadata)
            class ComplexProvider(ProviderBase[MockSettings]):
                def __init__(self, name: str, provider_type: str = "test", settings: MockSettings = None, **kwargs):
                    super().__init__(name=name, provider_type=provider_type, settings=settings)
                    # Use object.__setattr__ to bypass frozen model restriction
                    object.__setattr__(self, 'extra_kwargs', kwargs)
            
            call_args = mock_registry.register_factory.call_args
            
            # All metadata should be registered
            assert call_args.kwargs["description"] == "Complex test provider"
            assert call_args.kwargs["version"] == "2.1.0"
            assert call_args.kwargs["tags"] == ["test", "complex"]
            
            # Test factory with metadata settings
            factory_func = call_args.kwargs["factory"]
            instance = factory_func()
            
            # Settings should come from metadata
            assert instance.settings.host == "metadata.host"
            
            # Extra kwargs should exclude 'settings'
            assert "description" in instance.extra_kwargs
            assert "settings" not in instance.extra_kwargs