"""Tests for core dependency container."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from flowlib.core.container.container import (
    DependencyContainer,
    RegistryEntry,
    get_container,
    set_global_container
)
from flowlib.core.interfaces.interfaces import Configuration


class MockProvider:
    """Mock provider for testing."""
    
    def __init__(self, name: str = "mock", initialized: bool = False):
        self.name = name
        self.initialized = initialized
        self.initialize_called = False
        self.shutdown_called = False
    
    async def initialize(self):
        """Initialize the provider."""
        self.initialize_called = True
        self.initialized = True
    
    async def shutdown(self):
        """Shutdown the provider."""
        self.shutdown_called = True
        self.initialized = False


class MockResource:
    """Mock resource for testing."""
    
    def __init__(self, name: str = "mock"):
        self.name = name


class MockFlow:
    """Mock flow for testing."""
    
    def __init__(self, name: str = "mock"):
        self.name = name


class MockConfiguration:
    """Mock configuration for testing."""
    
    def __init__(self, provider_type: str = "mock", settings: dict = None):
        self.name = "mock_config"
        self.resource_type = "config"
        self._provider_type = provider_type
        self._settings = settings or {}
    
    def get_provider_type(self) -> str:
        """Get provider type."""
        return self._provider_type
    
    def get_settings(self) -> dict:
        """Get settings."""
        return self._settings
    
    def get_data(self) -> dict:
        """Get resource data."""
        return self._settings
    
    def get_metadata(self) -> dict:
        """Get resource metadata."""
        return {"provider_type": self._provider_type}


class TestRegistryEntry:
    """Test RegistryEntry dataclass."""
    
    def test_creation_minimal(self):
        """Test creating registry entry with minimal parameters."""
        factory = lambda: MockProvider()
        entry = RegistryEntry(
            name="test",
            item_type="provider",
            factory=factory,
            metadata={}
        )
        
        assert entry.name == "test"
        assert entry.item_type == "provider"
        assert entry.factory == factory
        assert entry.metadata == {}
        assert entry.instance is None
        assert entry.initialized is False
    
    def test_creation_with_instance(self):
        """Test creating registry entry with instance."""
        factory = lambda: MockProvider()
        instance = MockProvider("test")
        
        entry = RegistryEntry(
            name="test",
            item_type="provider",
            factory=factory,
            metadata={"version": "1.0"},
            instance=instance,
            initialized=True
        )
        
        assert entry.instance is instance
        assert entry.initialized is True
        assert entry.metadata["version"] == "1.0"


class TestDependencyContainer:
    """Test DependencyContainer functionality."""
    
    def test_initialization(self):
        """Test container initialization."""
        container = DependencyContainer()
        
        assert len(container._entries) == 0
        assert len(container._aliases) == 0
        assert len(container._provider_types) == 0
        assert len(container._resource_types) == 0
        assert len(container._flow_names) == 0
        assert len(container._initialization_locks) == 0
        assert len(container._config_cache) == 0
        assert container._loader is not None
    
    def test_register_provider(self):
        """Test registering a provider."""
        container = DependencyContainer()
        factory = lambda: MockProvider("test")
        metadata = {
            "provider_type": "test", 
            "provider_class": "MockProvider",
            "settings_class": "MockProviderSettings"
        }
        
        container.register("provider", "test_provider", factory, metadata)
        
        assert ("provider", "test_provider") in container._entries
        assert "test" in container._provider_types
        
        entry = container._entries[("provider", "test_provider")]
        assert entry.name == "test_provider"
        assert entry.item_type == "provider"
        assert entry.factory == factory
        assert entry.metadata == metadata
    
    def test_register_resource(self):
        """Test registering a resource."""
        container = DependencyContainer()
        factory = lambda: MockResource("test")
        metadata = {
            "resource_type": "config", 
            "resource_class": "MockResource"
        }
        
        container.register("resource", "test_resource", factory, metadata)
        
        assert ("resource", "test_resource") in container._entries
        assert "config" in container._resource_types
        
        entry = container._entries[("resource", "test_resource")]
        assert entry.name == "test_resource"
        assert entry.item_type == "resource"
    
    def test_register_flow(self):
        """Test registering a flow."""
        container = DependencyContainer()
        factory = lambda: MockFlow("test")
        metadata = {
            "flow_class": "MockFlow",
            "flow_category": "analysis",
            "input_type": "dict",
            "output_type": "dict"
        }
        
        container.register("flow", "test_flow", factory, metadata)
        
        assert ("flow", "test_flow") in container._entries
        assert "test_flow" in container._flow_names
        
        entry = container._entries[("flow", "test_flow")]
        assert entry.name == "test_flow"
        assert entry.item_type == "flow"
    
    def test_register_alias(self):
        """Test registering an alias."""
        container = DependencyContainer()
        factory = lambda: MockProvider("test")
        metadata = {
            "provider_type": "test",
            "provider_class": "MockProvider",
            "settings_class": "MockProviderSettings"
        }
        
        container.register("provider", "test_provider", factory, metadata)
        container.register_alias("alias", "provider", "test_provider")
        
        assert "alias" in container._aliases
        assert container._aliases["alias"] == ("provider", "test_provider")
    
    def test_get_resource_direct(self):
        """Test getting resource by direct lookup."""
        container = DependencyContainer()
        resource = MockResource("test")
        factory = lambda: resource
        metadata = {
            "resource_type": "config",
            "resource_class": "MockResource"
        }
        
        container.register("resource", "test_resource", factory, metadata)
        
        result = container.get_resource("test_resource", "resource")
        
        assert result is resource
        
        # Should be cached
        entry = container._entries[("resource", "test_resource")]
        assert entry.instance is resource
    
    def test_get_resource_via_alias(self):
        """Test getting resource via alias."""
        container = DependencyContainer()
        resource = MockResource("test")
        factory = lambda: resource
        metadata = {
            "resource_type": "config",
            "resource_class": "MockResource"
        }
        
        container.register("resource", "test_resource", factory, metadata)
        container.register_alias("alias", "resource", "test_resource")
        
        result = container.get_resource("alias")
        
        assert result is resource
    
    def test_get_resource_not_found(self):
        """Test getting non-existent resource."""
        container = DependencyContainer()
        
        with pytest.raises(KeyError, match="Resource 'nonexistent' not found"):
            container.get_resource("nonexistent")
    
    def test_get_flow_direct(self):
        """Test getting flow by direct lookup."""
        container = DependencyContainer()
        flow = MockFlow("test")
        factory = lambda: flow
        metadata = {
            "flow_class": "MockFlow",
            "flow_category": "analysis",
            "input_type": "dict",
            "output_type": "dict"
        }
        
        container.register("flow", "test_flow", factory, metadata)
        
        result = container.get_flow("test_flow")
        
        assert result is flow
        
        # Should be cached
        entry = container._entries[("flow", "test_flow")]
        assert entry.instance is flow
    
    def test_get_flow_via_alias(self):
        """Test getting flow via alias."""
        container = DependencyContainer()
        flow = MockFlow("test")
        factory = lambda: flow
        metadata = {
            "flow_class": "MockFlow",
            "flow_category": "analysis",
            "input_type": "dict",
            "output_type": "dict"
        }
        
        container.register("flow", "test_flow", factory, metadata)
        container.register_alias("flow_alias", "flow", "test_flow")
        
        result = container.get_flow("flow_alias")
        
        assert result is flow
    
    def test_get_flow_not_found(self):
        """Test getting non-existent flow."""
        container = DependencyContainer()
        
        with pytest.raises(KeyError, match="Flow 'nonexistent' not found"):
            container.get_flow("nonexistent")
    
    @pytest.mark.asyncio
    async def test_get_provider_with_config(self):
        """Test getting provider with configuration."""
        container = DependencyContainer()
        
        # Mock configuration
        config = MockConfiguration("test_provider", {"host": "localhost"})
        config_factory = lambda: config
        metadata = {
            "resource_type": "config",
            "resource_class": "MockConfiguration"
        }
        container.register("config", "test_config", config_factory, metadata)
        
        # Mock dynamic loader
        provider = MockProvider("test")
        with patch.object(container._loader, 'create_provider', return_value=provider) as mock_create:
            result = await container.get_provider("test_config")
            
            assert result is provider
            assert provider.initialize_called
            mock_create.assert_called_once_with("test_provider", {"host": "localhost"})
            
            # Should be cached
            assert "test_provider:test_config" in container._config_cache
    
    @pytest.mark.asyncio
    async def test_get_provider_invalid_config(self):
        """Test getting provider with invalid configuration."""
        container = DependencyContainer()
        
        # Register non-configuration resource
        resource = MockResource("test")
        factory = lambda: resource
        metadata = {
            "resource_type": "model",
            "resource_class": "MockResource"
        }
        container.register("resource", "not_config", factory, metadata)
        
        with pytest.raises(ValueError, match="Resource 'not_config' is not a configuration"):
            await container.get_provider("not_config")
    
    @pytest.mark.asyncio
    async def test_get_provider_by_type_with_name(self):
        """Test getting provider by type with specific name."""
        container = DependencyContainer()
        provider = MockProvider("test")
        factory = lambda: provider
        
        metadata = {
            "provider_type": "test",
            "provider_class": "MockProvider",
            "settings_class": "MockProviderSettings"
        }
        container.register("provider", "test_provider", factory, metadata)
        
        result = await container.get_provider_by_type("test", "test_provider")
        
        assert result is provider
        assert provider.initialize_called
    
    def test_list_providers(self):
        """Test listing all providers."""
        container = DependencyContainer()
        
        provider_metadata = {
            "provider_type": "test",
            "provider_class": "MockProvider",
            "settings_class": "MockProviderSettings"
        }
        resource_metadata = {
            "resource_type": "config",
            "resource_class": "MockResource"
        }
        
        container.register("provider", "provider1", lambda: MockProvider(), provider_metadata)
        container.register("provider", "provider2", lambda: MockProvider(), provider_metadata)
        container.register("resource", "resource1", lambda: MockResource(), resource_metadata)
        
        providers = container.list_providers()
        
        assert len(providers) == 2
        assert "provider1" in providers
        assert "provider2" in providers
        assert "resource1" not in providers
    
    def test_list_resources(self):
        """Test listing all resources."""
        container = DependencyContainer()
        
        resource_metadata = {
            "resource_type": "config",
            "resource_class": "MockResource"
        }
        config_metadata = {
            "resource_type": "config",
            "resource_class": "MockConfiguration"
        }
        provider_metadata = {
            "provider_type": "test",
            "provider_class": "MockProvider",
            "settings_class": "MockProviderSettings"
        }
        
        container.register("resource", "resource1", lambda: MockResource(), resource_metadata)
        container.register("config", "config1", lambda: MockConfiguration(), config_metadata)
        container.register("provider", "provider1", lambda: MockProvider(), provider_metadata)
        
        resources = container.list_resources()
        
        assert len(resources) == 2
        assert "resource1" in resources
        assert "config1" in resources
        assert "provider1" not in resources
    
    def test_list_flows(self):
        """Test listing all flows."""
        container = DependencyContainer()
        
        flow_metadata = {
            "flow_class": "MockFlow",
            "flow_category": "analysis",
            "input_type": "dict",
            "output_type": "dict"
        }
        provider_metadata = {
            "provider_type": "test",
            "provider_class": "MockProvider",
            "settings_class": "MockProviderSettings"
        }
        
        container.register("flow", "flow1", lambda: MockFlow(), flow_metadata)
        container.register("flow", "flow2", lambda: MockFlow(), flow_metadata)
        container.register("provider", "provider1", lambda: MockProvider(), provider_metadata)
        
        flows = container.list_flows()
        
        assert len(flows) == 2
        assert "flow1" in flows
        assert "flow2" in flows
        assert "provider1" not in flows
    
    def test_get_provider_types(self):
        """Test getting provider types."""
        container = DependencyContainer()
        
        metadata1 = {
            "provider_type": "type1",
            "provider_class": "MockProvider",
            "settings_class": "MockProviderSettings"
        }
        metadata2 = {
            "provider_type": "type2",
            "provider_class": "MockProvider",
            "settings_class": "MockProviderSettings"
        }
        
        container.register("provider", "p1", lambda: MockProvider(), metadata1)
        container.register("provider", "p2", lambda: MockProvider(), metadata2)
        container.register("provider", "p3", lambda: MockProvider(), metadata1)
        
        types = container.get_provider_types()
        
        assert len(types) == 2
        assert "type1" in types
        assert "type2" in types
    
    def test_contains(self):
        """Test checking if item exists."""
        container = DependencyContainer()
        
        metadata = {
            "provider_type": "test",
            "provider_class": "MockProvider",
            "settings_class": "MockProviderSettings"
        }
        container.register("provider", "test", lambda: MockProvider(), metadata)
        
        assert container.contains("provider", "test")
        assert not container.contains("provider", "nonexistent")
        assert not container.contains("flow", "test")
    
    def test_get_metadata(self):
        """Test getting item metadata."""
        container = DependencyContainer()
        metadata = {
            "provider_type": "test",
            "provider_class": "MockProvider", 
            "settings_class": "MockProviderSettings"
        }
        
        container.register("provider", "test", lambda: MockProvider(), metadata)
        
        result = container.get_metadata("provider", "test")
        
        assert result == metadata
        assert result is not metadata  # Should be a copy
    
    def test_get_metadata_not_found(self):
        """Test getting metadata for non-existent item."""
        container = DependencyContainer()
        
        with pytest.raises(KeyError, match="provider 'nonexistent' not found"):
            container.get_metadata("provider", "nonexistent")
    
    @pytest.mark.asyncio
    async def test_initialize_all_providers(self):
        """Test initializing all providers."""
        container = DependencyContainer()
        
        provider1 = MockProvider("p1")
        provider2 = MockProvider("p2")
        
        provider_metadata = {
            "provider_type": "test",
            "provider_class": "MockProvider",
            "settings_class": "MockProviderSettings"
        }
        resource_metadata = {
            "resource_type": "config",
            "resource_class": "MockResource"
        }
        
        container.register("provider", "p1", lambda: provider1, provider_metadata)
        container.register("provider", "p2", lambda: provider2, provider_metadata)
        container.register("resource", "r1", lambda: MockResource(), resource_metadata)
        
        await container.initialize_all_providers()
        
        # Check providers were initialized
        entry1 = container._entries[("provider", "p1")]
        entry2 = container._entries[("provider", "p2")]
        
        assert entry1.instance is provider1
        assert entry1.initialized
        assert provider1.initialize_called
        
        assert entry2.instance is provider2
        assert entry2.initialized
        assert provider2.initialize_called
    
    @pytest.mark.asyncio
    async def test_shutdown_all_providers(self):
        """Test shutting down all providers."""
        container = DependencyContainer()
        
        provider1 = MockProvider("p1")
        provider2 = MockProvider("p2")
        
        # Set up as initialized
        entry1 = RegistryEntry("p1", "provider", lambda: provider1, {}, provider1, True)
        entry2 = RegistryEntry("p2", "provider", lambda: provider2, {}, provider2, True)
        
        container._entries[("provider", "p1")] = entry1
        container._entries[("provider", "p2")] = entry2
        resource_metadata = {
            "resource_type": "config",
            "resource_class": "MockResource"
        }
        container._entries[("resource", "r1")] = RegistryEntry("r1", "resource", lambda: MockResource(), resource_metadata)
        
        await container.shutdown_all_providers()
        
        # Check providers were shut down
        assert provider1.shutdown_called
        assert provider2.shutdown_called
        assert not entry1.initialized
        assert not entry2.initialized
    
    def test_get_stats(self):
        """Test getting container statistics."""
        container = DependencyContainer()
        
        provider_metadata1 = {
            "provider_type": "type1",
            "provider_class": "MockProvider",
            "settings_class": "MockProviderSettings"
        }
        provider_metadata2 = {
            "provider_type": "type2",
            "provider_class": "MockProvider",
            "settings_class": "MockProviderSettings"
        }
        resource_metadata = {
            "resource_type": "rtype1",
            "resource_class": "MockResource"
        }
        flow_metadata = {
            "flow_class": "MockFlow",
            "flow_category": "analysis",
            "input_type": "dict",
            "output_type": "dict"
        }
        
        container.register("provider", "p1", lambda: MockProvider(), provider_metadata1)
        container.register("provider", "p2", lambda: MockProvider(), provider_metadata2)
        container.register("resource", "r1", lambda: MockResource(), resource_metadata)
        container.register("flow", "f1", lambda: MockFlow(), flow_metadata)
        container.register_alias("alias1", "provider", "p1")
        container._config_cache["cache1"] = MockProvider()
        
        stats = container.get_stats()
        
        assert stats["total_entries"] == 4
        assert stats["type_counts"]["provider"] == 2
        assert stats["type_counts"]["resource"] == 1
        assert stats["type_counts"]["flow"] == 1
        assert stats["aliases"] == 1
        assert stats["provider_types"] == 2
        assert stats["resource_types"] == 1
        assert stats["cached_configs"] == 1


class TestGlobalContainer:
    """Test global container functions."""
    
    def test_get_container(self):
        """Test getting global container."""
        container = get_container()
        
        assert isinstance(container, DependencyContainer)
        
        # Should return same instance
        container2 = get_container()
        assert container is container2
    
    def test_set_global_container(self):
        """Test setting global container."""
        original = get_container()
        new_container = DependencyContainer()
        
        try:
            set_global_container(new_container)
            
            current = get_container()
            assert current is new_container
            assert current is not original
        finally:
            # Restore original
            set_global_container(original)