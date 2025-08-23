"""
Tests for the CleanProviderRegistry.

This module tests the core provider registry functionality including
provider registration, configuration-based access, factory patterns,
lifecycle management, and error handling.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, Optional

from flowlib.providers.core.registry import CleanProviderRegistry, provider_registry
from flowlib.providers.core.base import Provider, ProviderSettings
from flowlib.core.errors.errors import ExecutionError
from flowlib.resources.models.config_resource import ProviderConfigResource


class MockProviderSettings(ProviderSettings):
    """Mock provider settings."""
    model: str = "default_model"
    temperature: float = 0.7


class MockProvider(Provider):
    """Mock provider for testing."""
    
    def __init__(self, name: str = "test_provider", provider_type: str = "test", settings: Any = None):
        if settings is None:
            settings = MockProviderSettings()
        super().__init__(name=name, provider_type=provider_type, settings=settings)
        # Use private attributes to avoid Pydantic field conflicts
        self._configured = False
        self._shutdown_called = False

    @property
    def configured(self) -> bool:
        return self._configured
        
    @property 
    def shutdown_called(self) -> bool:
        return self._shutdown_called
    
    @property
    def current_settings(self) -> MockProviderSettings:
        """Get current configured settings."""
        return getattr(self, '_configured_settings', self.settings)
    
    async def _initialize(self):
        """Initialize the provider."""
        pass
    
    async def configure(self, settings: Dict[str, Any]):
        """Configure the provider."""
        self._configured = True
        # Create new settings instance with updated values (immutable pattern)
        current_settings = self.settings.model_dump()
        current_settings.update(settings)
        # Store as private attribute since we can't reassign frozen models
        self._configured_settings = MockProviderSettings(**current_settings)
    
    async def _shutdown(self):
        """Shutdown the provider."""
        self._shutdown_called = True


class MockProviderWithoutConfigure(Provider):
    """Mock provider without configure method for testing."""
    
    def __init__(self, name: str = "test_provider", provider_type: str = "test", settings: Any = None):
        if settings is None:
            settings = MockProviderSettings()
        super().__init__(name=name, provider_type=provider_type, settings=settings)
        # Use private attributes to avoid Pydantic field conflicts
        self._update_settings_calls = []
    
    def update_settings(self, new_settings: Dict[str, Any]):
        """Alternative configuration method."""
        self._update_settings_calls.append(new_settings)
    
    @property
    def update_settings_calls(self):
        """Get update_settings call history."""
        return self._update_settings_calls
    
    async def _initialize(self):
        """Initialize the provider."""
        pass
    
    async def _shutdown(self):
        """Shutdown the provider."""
        pass


class MockProviderConfigResource(ProviderConfigResource):
    """Mock provider config resource."""
    
    def __init__(self, provider_type: str = "test", settings: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(
            name=kwargs.get('name', 'test_config'),
            type=kwargs.get('type', 'provider_config'),
            provider_type=provider_type, 
            settings=settings or {},
            **kwargs
        )


class TestCleanProviderRegistry:
    """Test the CleanProviderRegistry class."""
    
    def setup_method(self):
        """Set up each test with a fresh registry."""
        self.registry = CleanProviderRegistry()
    
    def test_init(self):
        """Test registry initialization."""
        assert len(self.registry._providers) == 0
        assert len(self.registry._factories) == 0
        assert len(self.registry._initialized_providers) == 0
        assert len(self.registry._initialization_locks) == 0
    
    def test_register_provider_success(self):
        """Test successful provider registration."""
        provider = MockProvider("test_provider", "llm")
        
        self.registry.register_provider(provider)
        
        key = ("llm", "test_provider")
        assert key in self.registry._providers
        assert self.registry._providers[key] is provider
    
    def test_register_provider_invalid_type(self):
        """Test registering invalid provider type."""
        invalid_provider = "not a provider"
        
        with pytest.raises(TypeError, match="Provider must be a ProviderBase subclass"):
            self.registry.register_provider(invalid_provider)
    
    def test_register_provider_missing_name(self):
        """Test registering provider without name."""
        provider = MockProvider()
        # Use object.__setattr__ for frozen Pydantic models
        object.__setattr__(provider, 'name', None)
        
        with pytest.raises(ValueError, match="Provider must have a non-empty name"):
            self.registry.register_provider(provider)
    
    def test_register_provider_missing_type(self):
        """Test registering provider without type."""
        provider = MockProvider()
        # Use object.__setattr__ for frozen Pydantic models
        object.__setattr__(provider, 'provider_type', None)
        
        with pytest.raises(ValueError, match="Provider must have a non-empty provider_type"):
            self.registry.register_provider(provider)
    
    def test_register_interface_with_provider(self):
        """Test BaseRegistry register interface with provider."""
        provider = MockProvider("test_provider", "llm")
        
        self.registry.register("test_provider", provider)
        
        key = ("llm", "test_provider")
        assert key in self.registry._providers
    
    def test_register_interface_with_non_provider(self):
        """Test BaseRegistry register interface with non-provider."""
        with pytest.raises(TypeError, match="CleanProviderRegistry only accepts Provider objects"):
            self.registry.register("test", "not a provider")
    
    def test_register_factory(self):
        """Test factory registration."""
        def factory(settings=None):
            return MockProvider("factory_provider", "test")
        
        self.registry.register_factory("test", "factory_provider", factory, extra_meta="test")
        
        key = ("test", "factory_provider")
        assert key in self.registry._factories
        assert self.registry._factories[key] is factory
        assert self.registry._factory_metadata[key]["provider_type"] == "test"
        assert self.registry._factory_metadata[key]["extra_meta"] == "test"
    
    @pytest.mark.asyncio
    async def test_get_by_config_success(self):
        """Test successful config-based provider retrieval."""
        # Register a provider - the key will be (provider_type, name)
        provider = MockProvider("llamacpp", "llm")  # name="llamacpp", type="llm" 
        self.registry.register_provider(provider)
        
        # Mock the resource registry
        mock_config = MockProviderConfigResource("llamacpp", {"model": "test_model"})
        
        with patch('flowlib.providers.core.registry.resource_registry') as mock_resource_registry:
            mock_resource_registry.get.return_value = mock_config
            
            # Mock the provider category inference to return "llm" 
            with patch.object(self.registry, '_infer_provider_category_from_config', return_value="llm"):
                # The lookup will try key ("llm", "llamacpp") but provider is stored as ("llm", "llamacpp")
                # So we need to register provider with the right key pattern
                result = await self.registry.get_by_config("test_config")
                
                assert result is provider
                assert result.initialized
                # Note: Provider configuration now happens at creation time via factory(settings)
                # No need to check separate configuration step
    
    @pytest.mark.asyncio
    async def test_get_by_config_config_not_found(self):
        """Test config-based retrieval with missing config."""
        with patch('flowlib.providers.core.registry.resource_registry') as mock_resource_registry:
            mock_resource_registry.get.side_effect = KeyError("Config not found")
            
            with pytest.raises(KeyError):
                await self.registry.get_by_config("missing_config")
    
    @pytest.mark.asyncio
    async def test_get_by_config_invalid_config_type(self):
        """Test config-based retrieval with invalid config type."""
        mock_config = "not a config resource"
        
        with patch('flowlib.providers.core.registry.resource_registry') as mock_resource_registry:
            mock_resource_registry.get.return_value = mock_config
            
            with pytest.raises(TypeError, match="not a ProviderConfigResource"):
                await self.registry.get_by_config("invalid_config")
    
    @pytest.mark.asyncio
    async def test_get_by_config_missing_provider_type(self):
        """Test config-based retrieval with missing provider type."""
        mock_config = MockProviderConfigResource("", {})
        
        with patch('flowlib.providers.core.registry.resource_registry') as mock_resource_registry:
            mock_resource_registry.get.return_value = mock_config
            
            with pytest.raises(ValueError, match="does not specify a provider_type"):
                await self.registry.get_by_config("invalid_config")
    
    @pytest.mark.asyncio
    async def test_get_or_create_provider_existing(self):
        """Test getting existing provider."""
        provider = MockProvider("test_provider", "llm")
        self.registry.register_provider(provider)
        
        result = await self.registry._get_or_create_provider("llm", "test_provider")
        
        assert result is provider
        assert result.initialized
    
    @pytest.mark.asyncio
    async def test_get_or_create_provider_from_factory(self):
        """Test creating provider from factory."""
        def factory(settings=None):
            return MockProvider("factory_provider", "test")
        
        self.registry.register_factory("test", "factory_provider", factory)
        
        result = await self.registry._get_or_create_provider("test", "factory_provider")
        
        assert isinstance(result, MockProvider)
        assert result.name == "factory_provider"
        assert result.initialized
        
        # Should be cached for future calls
        key = ("test", "factory_provider")
        assert key in self.registry._initialized_providers
    
    @pytest.mark.asyncio
    async def test_get_or_create_provider_not_found(self):
        """Test provider not found error."""
        with pytest.raises(ExecutionError, match="Provider 'missing' of category 'test' not found"):
            await self.registry._get_or_create_provider("test", "missing")
    
    @pytest.mark.asyncio
    async def test_get_or_create_provider_initialization_error(self):
        """Test provider initialization error."""
        class FailingMockProvider(MockProvider):
            """Mock provider that fails initialization."""
            
            async def _initialize(self):
                raise Exception("Init failed")
        
        def failing_factory():
            return FailingMockProvider("failing_provider", "test")
        
        self.registry.register_factory("test", "failing_provider", failing_factory)
        
        with pytest.raises(ExecutionError, match="Failed to initialize provider"):
            await self.registry._get_or_create_provider("test", "failing_provider")
    
    @pytest.mark.asyncio
    async def test_get_or_create_provider_concurrent_access(self):
        """Test concurrent provider creation is properly locked."""
        call_count = 0
        
        def factory(settings=None):
            nonlocal call_count
            call_count += 1
            return MockProvider(f"concurrent_provider_{call_count}", "test")
        
        self.registry.register_factory("test", "concurrent_provider", factory)
        
        # Start multiple concurrent requests
        tasks = [
            self.registry._get_or_create_provider("test", "concurrent_provider")
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should return the same instance
        first_provider = results[0]
        for provider in results[1:]:
            assert provider is first_provider
        
        # Factory should only be called once
        assert call_count == 1
    
    # REMOVED: test_configure_provider_with_config and test_configure_provider_no_configure_method
    # Settings are now passed at provider creation time via factory(settings),
    # eliminating the need for separate configuration after creation
    
    def test_infer_provider_category_from_config(self):
        """Test provider category inference from config type."""
        from flowlib.resources.models.config_resource import LLMConfigResource, DatabaseConfigResource
        
        # Test with actual LLMConfigResource instance
        llm_config = LLMConfigResource(
            name="test_llm_config",
            type="llm_config",
            provider_type="test_llm",
            settings={}
        )
        result = self.registry._infer_provider_category_from_config("test", llm_config)
        assert result == "llm"
        
        # Test with DatabaseConfigResource instance  
        db_config = DatabaseConfigResource(
            name="test_db_config",
            type="db_config", 
            provider_type="test_db",
            settings={
                "host": "localhost",
                "port": 5432,
                "database": "test_db"
            }
        )
        result = self.registry._infer_provider_category_from_config("test", db_config)
        assert result == "database"
    
    def test_infer_provider_category_validation(self):
        """Test provider category inference fallback to type mapping."""
        config = Mock()
        # Should not match any specific config types
        
        result = self.registry._infer_provider_category_from_config("postgres", config)
        assert result == "database"
        
        # Unknown provider types should raise an error with strict contracts
        import pytest
        with pytest.raises(ValueError, match="Unknown provider type: unknown_type"):
            self.registry._infer_provider_category_from_config("unknown_type", config)
    
    def test_list_available_configs(self):
        """Test listing available configurations."""
        with patch('flowlib.providers.core.registry.resource_registry') as mock_resource_registry:
            # Mock list_by_type to return different configs for different types
            def mock_list_by_type(resource_type):
                if str(resource_type) == "llm_config":
                    return {"default-llm": Mock(), "fast-llm": Mock()}
                elif str(resource_type) == "database_config":
                    return {"test-db": Mock()}
                else:
                    return {}
            
            mock_resource_registry.get_by_type.side_effect = mock_list_by_type
            
            configs = self.registry._list_available_configs()
            
            # Should contain configs from multiple resource types
            assert len(configs) >= 3  # default-llm, fast-llm, test-db
            assert "default-llm" in configs
            assert "fast-llm" in configs
            assert "test-db" in configs
    
    def test_list_available_configs_error(self):
        """Test listing configs with registry error."""
        with patch('flowlib.providers.core.registry.resource_registry') as mock_resource_registry:
            mock_resource_registry.list_by_type.side_effect = Exception("Registry error")
            
            configs = self.registry._list_available_configs()
            assert configs == []
    
    def test_list_providers(self):
        """Test listing registered providers."""
        provider1 = MockProvider("provider1", "llm")
        provider2 = MockProvider("provider2", "database")
        
        self.registry.register_provider(provider1)
        self.registry.register_provider(provider2)
        
        def factory(settings=None):
            return MockProvider("factory_provider", "test")
        self.registry.register_factory("test", "factory_provider", factory)
        
        providers = self.registry.list_providers()
        
        assert "llm:provider1" in providers
        assert "database:provider2" in providers
        assert "test:factory_provider (factory)" in providers
    
    def test_contains(self):
        """Test provider config existence check."""
        # Mock a provider config in resource registry
        mock_config = MockProviderConfigResource("test_provider", {})
        
        with patch('flowlib.providers.core.registry.resource_registry') as mock_resource_registry:
            # Config exists
            mock_resource_registry.get.return_value = mock_config
            assert self.registry.contains("test_config")
            
            # Config doesn't exist
            mock_resource_registry.get.side_effect = KeyError("Not found")
            assert not self.registry.contains("missing_config")
    
    def test_get_interface_sync_wrapper(self):
        """Test get interface shows async-first architecture error (BaseRegistry compatibility)."""
        # The new async-first architecture doesn't support synchronous get()
        with pytest.raises(NotImplementedError, match="Synchronous get\\(\\) method not supported"):
            self.registry.get("test_config")
    
    def test_get_interface_async_context_error(self):
        """Test get interface error in async context."""
        with patch('asyncio.get_event_loop') as mock_get_loop:
            mock_loop = Mock()
            mock_loop.is_running.return_value = True
            mock_get_loop.return_value = mock_loop
            
            with pytest.raises(NotImplementedError, match="Synchronous get\\(\\) method not supported"):
                self.registry.get("test_config")
    
    def test_list_interface(self):
        """Test list interface (BaseRegistry compatibility)."""
        with patch.object(self.registry, '_list_available_configs', return_value=["config1", "config2", "llm_config"]):
            # No filter
            result = self.registry.list()
            assert result == ["config1", "config2", "llm_config"]
            
            # With filter
            result = self.registry.list({"provider_type": "llm"})
            assert result == ["llm_config"]
    
    def test_list_interface_error(self):
        """Test list interface with error."""
        with patch.object(self.registry, '_list_available_configs', side_effect=Exception("Error")):
            result = self.registry.list()
            assert result == []
    
    @pytest.mark.asyncio
    async def test_initialize_all(self):
        """Test initializing all providers."""
        provider1 = MockProvider("provider1", "llm")
        provider2 = MockProvider("provider2", "database")
        
        self.registry.register_provider(provider1)
        self.registry.register_provider(provider2)
        
        def factory(settings=None):
            return MockProvider("factory_provider", "test")
        self.registry.register_factory("test", "factory_provider", factory)
        
        await self.registry.initialize_all()
        
        # All providers should be initialized
        assert len(self.registry._initialized_providers) == 3
        assert provider1.initialized
        assert provider2.initialized
    
    @pytest.mark.asyncio
    async def test_initialize_all_with_failures(self):
        """Test initializing all providers with some failures."""
        provider1 = MockProvider("provider1", "llm")
        
        class FailingMockProvider(MockProvider):
            """Mock provider that fails initialization."""
            
            async def _initialize(self):
                raise Exception("Init failed")
        
        def failing_factory():
            return FailingMockProvider("failing_provider", "test")
        
        self.registry.register_provider(provider1)
        self.registry.register_factory("test", "failing_provider", failing_factory)
        
        # Should not raise exception, just log warnings
        await self.registry.initialize_all()
        
        # Good provider should be initialized
        assert provider1.initialized
        assert len(self.registry._initialized_providers) == 1
    
    @pytest.mark.asyncio
    async def test_shutdown_all(self):
        """Test shutting down all providers."""
        provider1 = MockProvider("provider1", "llm")
        provider2 = MockProvider("provider2", "database")
        
        self.registry.register_provider(provider1)
        self.registry.register_provider(provider2)
        
        # Initialize them first
        await self.registry._get_or_create_provider("llm", "provider1")
        await self.registry._get_or_create_provider("database", "provider2")
        
        assert len(self.registry._initialized_providers) == 2
        
        await self.registry.shutdown_all()
        
        assert provider1.shutdown_called
        assert provider2.shutdown_called
        assert len(self.registry._initialized_providers) == 0
        assert len(self.registry._initialization_locks) == 0
    
    @pytest.mark.asyncio
    async def test_shutdown_all_with_failures(self):
        """Test shutting down providers with some failures."""
        provider1 = MockProvider("provider1", "llm")
        provider2 = MockProvider("provider2", "database")
        
        # Mock the shutdown method on provider2 to raise an exception
        original_shutdown = provider2._shutdown
        async def failing_shutdown():
            raise Exception("Shutdown failed")
        provider2._shutdown = failing_shutdown
        
        self.registry.register_provider(provider1)
        self.registry.register_provider(provider2)
        
        # Initialize them first
        await self.registry._get_or_create_provider("llm", "provider1")
        await self.registry._get_or_create_provider("database", "provider2")
        
        # Should not raise exception, just log warnings
        await self.registry.shutdown_all()
        
        assert provider1.shutdown_called
        assert len(self.registry._initialized_providers) == 0


class TestGlobalRegistry:
    """Test the global provider registry instance."""
    
    def test_global_registry_exists(self):
        """Test that global registry is available."""
        assert provider_registry is not None
        assert isinstance(provider_registry, CleanProviderRegistry)
    
    def test_global_registry_singleton(self):
        """Test that importing gives same instance."""
        from flowlib.providers.core.registry import provider_registry as registry2
        assert provider_registry is registry2


@pytest.mark.asyncio
async def test_integration_provider_lifecycle():
    """Integration test for complete provider lifecycle."""
    registry = CleanProviderRegistry()
    
    # Create factory that uses settings from config
    created_provider = None
    def provider_factory(settings=None):
        nonlocal created_provider
        # Create provider with settings from config
        if settings:
            provider_settings = MockProviderSettings(**settings)
            created_provider = MockProvider("llamacpp", "llm", settings=provider_settings)
        else:
            created_provider = MockProvider("llamacpp", "llm")
        return created_provider
    
    # Register factory instead of provider directly
    registry.register_factory("llm", "llamacpp", provider_factory)
    
    # Mock configuration
    mock_config = MockProviderConfigResource("llamacpp", {
        "model": "test_model",
        "temperature": 0.8
    })
    
    with patch('flowlib.providers.core.registry.resource_registry') as mock_resource_registry:
        mock_resource_registry.get.return_value = mock_config
        
        with patch.object(registry, '_infer_provider_category_from_config', return_value="llm"):
            # Get provider by config (should create with settings and initialize)
            result = await registry.get_by_config("test_config")
            
            assert result is created_provider
            assert result.initialized
            # Check that settings were applied
            assert result.settings.model == "test_model"
            assert result.settings.temperature == 0.8
            
            # Second call should return cached instance
            result2 = await registry.get_by_config("test_config")
            assert result2 is result
            
            # Shutdown
            await registry.shutdown_all()
            assert result.shutdown_called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])