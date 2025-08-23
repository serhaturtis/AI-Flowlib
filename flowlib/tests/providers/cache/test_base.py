"""Tests for cache provider base classes."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Optional, Any
from pydantic import BaseModel

from flowlib.providers.cache.base import CacheProviderSettings, CacheProvider
from flowlib.core.errors.errors import ProviderError


class MockModel(BaseModel):
    """Mock model for structured cache operations."""
    name: str
    value: int
    active: bool = True


class TestCacheProviderSettings:
    """Test CacheProviderSettings configuration class."""
    
    def test_default_settings_values(self):
        """Test default settings values."""
        settings = CacheProviderSettings()
        
        assert settings.host is None
        assert settings.port is None
        assert settings.username is None
        assert settings.password is None
        assert settings.default_ttl == 300
        assert settings.max_size is None
        assert settings.eviction_policy == "lru"
        assert settings.use_compression is False
        assert settings.serialize_method == "json"
        assert settings.namespace == "default"
        assert settings.pool_size == 5
        assert settings.timeout == 5.0
        assert settings.retry_on_timeout is True
        assert settings.max_retries == 3
    
    def test_custom_values(self):
        """Test settings with custom values."""
        settings = CacheProviderSettings(
            host="localhost",
            port=6379,
            username="cache_user",
            password="cache_pass",
            default_ttl=600,
            max_size=1000,
            eviction_policy="lfu",
            use_compression=True,
            serialize_method="pickle",
            namespace="test",
            pool_size=10,
            timeout=10.0,
            retry_on_timeout=False,
            max_retries=5
        )
        
        assert settings.host == "localhost"
        assert settings.port == 6379
        assert settings.username == "cache_user"
        assert settings.password == "cache_pass"
        assert settings.default_ttl == 600
        assert settings.max_size == 1000
        assert settings.eviction_policy == "lfu"
        assert settings.use_compression is True
        assert settings.serialize_method == "pickle"
        assert settings.namespace == "test"
        assert settings.pool_size == 10
        assert settings.timeout == 10.0
        assert settings.retry_on_timeout is False
        assert settings.max_retries == 5


class ConcreteCacheProvider(CacheProvider):
    """Concrete implementation for testing."""
    
    def __init__(self, name: str = "test_cache", settings: Optional[CacheProviderSettings] = None):
        if settings is None:
            settings = CacheProviderSettings()
        super().__init__(name, settings)
        self._data = {}
        self._ttls = {}
    
    async def get(self, key: str) -> Optional[Any]:
        return self._data.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        self._data[key] = value
        if ttl:
            self._ttls[key] = ttl
        return True
    
    async def delete(self, key: str) -> bool:
        self._data.pop(key, None)
        self._ttls.pop(key, None)
        return True
    
    async def exists(self, key: str) -> bool:
        return key in self._data
    
    async def ttl(self, key: str) -> Optional[int]:
        return self._ttls.get(key)
    
    async def clear(self) -> bool:
        self._data.clear()
        self._ttls.clear()
        return True
    
    async def check_connection(self) -> bool:
        return self._initialized


class TestCacheProvider:
    """Test CacheProvider base class."""
    
    def test_initialization_default(self):
        """Test provider initialization with defaults."""
        provider = ConcreteCacheProvider()
        
        assert provider.name == "test_cache"
        assert provider.provider_type == "cache"
        assert provider.initialized is False
        assert provider._connection is None
        assert isinstance(provider.settings, CacheProviderSettings)
    
    def test_initialization_custom(self):
        """Test provider initialization with custom settings."""
        settings = CacheProviderSettings(
            default_ttl=1200,
            namespace="custom",
            pool_size=8
        )
        provider = ConcreteCacheProvider("custom_cache", settings)
        
        assert provider.name == "custom_cache"
        assert provider.settings.default_ttl == 1200
        assert provider.settings.namespace == "custom"
        assert provider.settings.pool_size == 8
    
    @pytest.mark.asyncio
    async def test_initialization_lifecycle(self):
        """Test provider initialization lifecycle."""
        provider = ConcreteCacheProvider()
        
        # Initially not initialized
        assert provider.initialized is False
        
        # Initialize
        await provider.initialize()
        assert provider.initialized is True
        
        # Shutdown
        await provider.shutdown()
        assert provider.initialized is False
        assert provider._connection is None
    
    @pytest.mark.asyncio
    async def test_basic_cache_operations(self):
        """Test basic cache get/set/delete operations."""
        provider = ConcreteCacheProvider()
        await provider.initialize()
        
        # Set a value
        result = await provider.set("test_key", "test_value")
        assert result is True
        
        # Get the value
        value = await provider.get("test_key")
        assert value == "test_value"
        
        # Check existence
        exists = await provider.exists("test_key")
        assert exists is True
        
        # Delete the value
        result = await provider.delete("test_key")
        assert result is True
        
        # Verify deletion
        value = await provider.get("test_key")
        assert value is None
        
        exists = await provider.exists("test_key")
        assert exists is False
    
    @pytest.mark.asyncio
    async def test_cache_operations_with_ttl(self):
        """Test cache operations with TTL."""
        provider = ConcreteCacheProvider()
        await provider.initialize()
        
        # Set value with TTL
        result = await provider.set("ttl_key", "ttl_value", ttl=300)
        assert result is True
        
        # Check TTL
        ttl_value = await provider.ttl("ttl_key")
        assert ttl_value == 300
        
        # Verify value exists
        value = await provider.get("ttl_key")
        assert value == "ttl_value"
    
    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test clearing all cache data."""
        provider = ConcreteCacheProvider()
        await provider.initialize()
        
        # Set multiple values
        await provider.set("key1", "value1")
        await provider.set("key2", "value2")
        await provider.set("key3", "value3")
        
        # Verify values exist
        assert await provider.exists("key1") is True
        assert await provider.exists("key2") is True
        assert await provider.exists("key3") is True
        
        # Clear cache
        result = await provider.clear()
        assert result is True
        
        # Verify all values are gone
        assert await provider.exists("key1") is False
        assert await provider.exists("key2") is False
        assert await provider.exists("key3") is False
    
    @pytest.mark.asyncio
    async def test_get_structured_success(self):
        """Test successful structured data retrieval."""
        provider = ConcreteCacheProvider()
        await provider.initialize()
        
        # Set structured data
        test_data = {"name": "test", "value": 42, "active": True}
        await provider.set("struct_key", test_data)
        
        # Get structured data
        result = await provider.get_structured("struct_key", MockModel)
        
        assert result is not None
        assert isinstance(result, MockModel)
        assert result.name == "test"
        assert result.value == 42
        assert result.active is True
    
    @pytest.mark.asyncio
    async def test_get_structured_not_found(self):
        """Test structured data retrieval when key doesn't exist."""
        provider = ConcreteCacheProvider()
        await provider.initialize()
        
        # Try to get non-existent structured data
        result = await provider.get_structured("missing_key", MockModel)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_structured_parse_error(self):
        """Test structured data retrieval with parse error."""
        provider = ConcreteCacheProvider()
        await provider.initialize()
        
        # Set invalid data for the model
        invalid_data = {"invalid_field": "value"}
        await provider.set("invalid_key", invalid_data)
        
        # Try to get structured data - should raise ProviderError
        with pytest.raises(ProviderError) as exc_info:
            await provider.get_structured("invalid_key", MockModel)
        
        assert "Failed to get structured value" in str(exc_info.value)
        assert exc_info.value.provider_context.provider_name == "test_cache"
    
    @pytest.mark.asyncio
    async def test_set_structured_success(self):
        """Test successful structured data storage."""
        provider = ConcreteCacheProvider()
        await provider.initialize()
        
        # Create test model
        test_model = MockModel(name="test", value=42, active=False)
        
        # Set structured data
        result = await provider.set_structured("model_key", test_model, ttl=600)
        assert result is True
        
        # Verify data was stored correctly
        stored_data = await provider.get("model_key")
        assert stored_data == {"name": "test", "value": 42, "active": False}
        
        # Verify TTL was set
        ttl_value = await provider.ttl("model_key")
        assert ttl_value == 600
    
    @pytest.mark.asyncio
    async def test_set_structured_error(self):
        """Test structured data storage with error."""
        provider = ConcreteCacheProvider()
        await provider.initialize()
        
        # Create a provider that will fail on set
        class FailingCacheProvider(ConcreteCacheProvider):
            async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
                raise Exception("Set failed")
        
        failing_provider = FailingCacheProvider()
        await failing_provider.initialize()
        
        test_model = MockModel(name="test", value=42)
        
        with pytest.raises(ProviderError) as exc_info:
            await failing_provider.set_structured("error_key", test_model)
        
        assert "Failed to set structured value" in str(exc_info.value)
        assert exc_info.value.provider_context.provider_name == "test_cache"
    
    @pytest.mark.asyncio
    async def test_check_connection(self):
        """Test connection checking."""
        provider = ConcreteCacheProvider()
        
        # Initially not connected
        assert await provider.check_connection() is False
        
        # After initialization, should be connected
        await provider.initialize()
        assert await provider.check_connection() is True
        
        # After shutdown, should be disconnected
        await provider.shutdown()
        assert await provider.check_connection() is False
    
    def test_make_namespaced_key_with_namespace(self):
        """Test key namespacing with namespace configured."""
        settings = CacheProviderSettings(namespace="test_ns")
        provider = ConcreteCacheProvider(settings=settings)
        
        namespaced_key = provider.make_namespaced_key("my_key")
        assert namespaced_key == "test_ns:my_key"
    
    def test_make_namespaced_key_without_namespace(self):
        """Test key namespacing without namespace configured."""
        settings = CacheProviderSettings(namespace="")
        provider = ConcreteCacheProvider(settings=settings)
        
        namespaced_key = provider.make_namespaced_key("my_key")
        assert namespaced_key == "my_key"
    
    def test_make_namespaced_key_default_namespace(self):
        """Test key namespacing with default namespace."""
        provider = ConcreteCacheProvider()
        
        namespaced_key = provider.make_namespaced_key("my_key")
        assert namespaced_key == "default:my_key"


class AbstractCacheProvider(CacheProvider):
    """Abstract provider implementation to test NotImplementedError methods."""
    
    def __init__(self, name: str = "abstract_cache", settings: Optional[CacheProviderSettings] = None):
        if settings is None:
            settings = CacheProviderSettings()
        super().__init__(name=name, settings=settings)


class TestAbstractMethods:
    """Test that abstract methods raise NotImplementedError."""
    
    @pytest.mark.asyncio
    async def test_abstract_get(self):
        """Test that get method raises NotImplementedError."""
        provider = AbstractCacheProvider()
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement get"):
            await provider.get("key")
    
    @pytest.mark.asyncio
    async def test_abstract_set(self):
        """Test that set method raises NotImplementedError."""
        provider = AbstractCacheProvider()
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement set"):
            await provider.set("key", "value")
    
    @pytest.mark.asyncio
    async def test_abstract_delete(self):
        """Test that delete method raises NotImplementedError."""
        provider = AbstractCacheProvider()
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement delete"):
            await provider.delete("key")
    
    @pytest.mark.asyncio
    async def test_abstract_exists(self):
        """Test that exists method raises NotImplementedError."""
        provider = AbstractCacheProvider()
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement exists"):
            await provider.exists("key")
    
    @pytest.mark.asyncio
    async def test_abstract_ttl(self):
        """Test that ttl method raises NotImplementedError."""
        provider = AbstractCacheProvider()
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement ttl"):
            await provider.ttl("key")
    
    @pytest.mark.asyncio
    async def test_abstract_clear(self):
        """Test that clear method raises NotImplementedError."""
        provider = AbstractCacheProvider()
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement clear"):
            await provider.clear()
    
    @pytest.mark.asyncio
    async def test_abstract_check_connection(self):
        """Test that check_connection method raises NotImplementedError."""
        provider = AbstractCacheProvider()
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement check_connection"):
            await provider.check_connection()