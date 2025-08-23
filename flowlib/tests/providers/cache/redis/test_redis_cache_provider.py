"""Tests for Redis cache provider."""
import pytest
import asyncio
import json
import pickle
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Any, Dict

# Test both with and without redis installed
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from flowlib.providers.cache.redis.provider import (
    RedisCacheProvider,
    RedisCacheProviderSettings,
)
from flowlib.core.errors.errors import ProviderError


class TestRedisCacheProviderSettings:
    """Test Redis cache provider settings."""
    
    def test_default_settings(self):
        """Test default Redis cache provider settings."""
        settings = RedisCacheProviderSettings()
        
        # Test Redis-specific defaults
        assert settings.db == 0
        assert settings.socket_timeout is None
        assert settings.socket_connect_timeout is None
        assert settings.socket_keepalive is False
        assert settings.socket_keepalive_options is None
        assert settings.encoding == "utf-8"
        assert settings.encoding_errors == "strict"
        assert settings.decode_responses is False
        assert settings.sentinel_kwargs == {}
        assert settings.sentinel is None
        assert settings.sentinel_master is None
        
        # Test inherited cache settings
        assert settings.host == "localhost"
        assert settings.port == 6379
        assert settings.username is None
        assert settings.password is None
        assert settings.namespace is None
        assert settings.default_ttl is None
        assert settings.serialize_method == "json"
        assert settings.pool_size == 10
        
    def test_custom_settings(self):
        """Test custom Redis cache provider settings."""
        settings = RedisCacheProviderSettings(
            host="custom-redis",
            port=6380,
            db=1,
            username="test_user",
            password="test_pass",
            namespace="test_namespace",
            default_ttl=3600,
            serialize_method="pickle",
            pool_size=20,
            socket_timeout=5.0,
            socket_connect_timeout=3.0,
            socket_keepalive=True,
            socket_keepalive_options={1: 1, 2: 3, 3: 5},
            encoding="latin-1",
            encoding_errors="ignore",
            decode_responses=True,
            sentinel=["sentinel1:26379", "sentinel2:26379"],
            sentinel_master="mymaster",
            sentinel_kwargs={"password": "sentinel_pass"}
        )
        
        assert settings.host == "custom-redis"
        assert settings.port == 6380
        assert settings.db == 1
        assert settings.username == "test_user"
        assert settings.password == "test_pass"
        assert settings.namespace == "test_namespace"
        assert settings.default_ttl == 3600
        assert settings.serialize_method == "pickle"
        assert settings.pool_size == 20
        assert settings.socket_timeout == 5.0
        assert settings.socket_connect_timeout == 3.0
        assert settings.socket_keepalive is True
        assert settings.socket_keepalive_options == {1: 1, 2: 3, 3: 5}
        assert settings.encoding == "latin-1"
        assert settings.encoding_errors == "ignore"
        assert settings.decode_responses is True
        assert settings.sentinel == ["sentinel1:26379", "sentinel2:26379"]
        assert settings.sentinel_master == "mymaster"
        assert settings.sentinel_kwargs == {"password": "sentinel_pass"}


class TestRedisCacheProvider:
    """Test Redis cache provider."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return RedisCacheProviderSettings(
            host="localhost",
            port=6379,
            db=1,
            namespace="test",
            default_ttl=300,
            serialize_method="json"
        )
    
    @pytest.fixture
    def provider(self, settings):
        """Create test provider."""
        return RedisCacheProvider(name="test_redis", settings=settings)
    
    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        mock = AsyncMock()
        mock.ping.return_value = True
        mock.get.return_value = None
        mock.set.return_value = True
        mock.delete.return_value = 1
        mock.exists.return_value = 1
        mock.ttl.return_value = 300
        mock.flushdb.return_value = True
        mock.scan.return_value = (b'0', [])
        return mock
    
    @pytest.fixture
    def mock_pool(self):
        """Create mock connection pool."""
        mock = Mock()
        mock.disconnect = Mock()
        return mock
    
    def test_init_default_settings(self):
        """Test provider initialization with default settings."""
        provider = RedisCacheProvider()
        
        assert provider.name == "redis_cache"
        assert isinstance(provider._redis_settings, RedisCacheProviderSettings)
        assert provider._pool is None
        assert provider._redis is None
    
    def test_init_custom_settings(self, settings):
        """Test provider initialization with custom settings."""
        provider = RedisCacheProvider(name="custom_redis", settings=settings)
        
        assert provider.name == "custom_redis"
        assert provider._redis_settings == settings
        assert provider._pool is None
        assert provider._redis is None
    
    @patch('flowlib.providers.cache.redis.provider.redis')
    @patch('flowlib.providers.cache.redis.provider.ConnectionPool')
    async def test_initialize_success(self, mock_pool_class, mock_redis_module, provider, mock_pool, mock_redis):
        """Test successful provider initialization."""
        # Setup mocks
        mock_pool_class.return_value = mock_pool
        mock_redis_module.Redis.return_value = mock_redis
        
        # Initialize provider
        await provider._initialize()
        
        # Verify pool creation
        mock_pool_class.assert_called_once_with(
            host="localhost",
            port=6379,
            db=1,
            username=None,
            password=None,
            socket_timeout=None,
            socket_connect_timeout=None,
            socket_keepalive=False,
            socket_keepalive_options=None,
            encoding="utf-8",
            encoding_errors="strict",
            decode_responses=False,
            max_connections=10,
        )
        
        # Verify Redis client creation
        mock_redis_module.Redis.assert_called_once_with(connection_pool=mock_pool)
        
        # Verify connection test
        mock_redis.ping.assert_called_once()
        
        assert provider._pool == mock_pool
        assert provider._redis == mock_redis
    
    @patch('flowlib.providers.cache.redis.provider.redis')
    @patch('flowlib.providers.cache.redis.provider.ConnectionPool')
    async def test_initialize_connection_fail(self, mock_pool_class, mock_redis_module, provider, mock_pool, mock_redis):
        """Test provider initialization with connection failure."""
        # Setup mocks
        mock_pool_class.return_value = mock_pool
        mock_redis_module.Redis.return_value = mock_redis
        mock_redis.ping.return_value = False
        
        # Test initialization failure
        with pytest.raises(ProviderError) as exc_info:
            await provider._initialize()
        
        assert "Failed to connect to Redis server" in str(exc_info.value)
        assert exc_info.value.provider_context.provider_name == "test_redis"
    
    @patch('flowlib.providers.cache.redis.provider.redis')
    @patch('flowlib.providers.cache.redis.provider.ConnectionPool')
    async def test_initialize_redis_error(self, mock_pool_class, mock_redis_module, provider, mock_pool):
        """Test provider initialization with Redis error."""
        from flowlib.providers.cache.redis.provider import RedisError
        
        # Setup mocks
        mock_pool_class.return_value = mock_pool
        mock_redis_module.Redis.side_effect = RedisError("Connection failed")
        
        # Test initialization failure
        with pytest.raises(ProviderError) as exc_info:
            await provider._initialize()
        
        assert "Redis initialization error: Connection failed" in str(exc_info.value)
        assert exc_info.value.provider_context.provider_name == "test_redis"
    
    async def test_shutdown(self, provider, mock_pool):
        """Test provider shutdown."""
        # Set up provider state
        provider._pool = mock_pool
        provider._redis = Mock()
        
        # Shutdown provider
        await provider._shutdown()
        
        # Verify cleanup
        mock_pool.disconnect.assert_called_once()
        assert provider._pool is None
        assert provider._redis is None
    
    async def test_check_connection_success(self, provider, mock_redis):
        """Test successful connection check."""
        provider._redis = mock_redis
        mock_redis.ping.return_value = True
        
        result = await provider.check_connection()
        assert result is True
        mock_redis.ping.assert_called_once()
    
    async def test_check_connection_failure(self, provider, mock_redis):
        """Test failed connection check."""
        provider._redis = mock_redis
        mock_redis.ping.side_effect = Exception("Connection failed")
        
        result = await provider.check_connection()
        assert result is False
    
    async def test_get_success_json(self, provider, mock_redis):
        """Test successful get operation with JSON serialization."""
        provider._redis = mock_redis
        test_data = {"key": "value", "number": 42}
        mock_redis.get.return_value = json.dumps(test_data).encode()
        
        result = await provider.get("test_key")
        
        assert result == test_data
        mock_redis.get.assert_called_once_with("test:test_key")
    
    async def test_get_success_pickle(self, mock_redis):
        """Test successful get operation with pickle serialization."""
        # Create provider with pickle serialization
        settings = RedisCacheProviderSettings(
            host="localhost",
            port=6379,
            namespace="test",
            serialize_method="pickle"
        )
        pickle_provider = RedisCacheProvider("test_redis", settings)
        pickle_provider._redis = mock_redis
        test_data = {"key": "value", "number": 42}
        mock_redis.get.return_value = pickle.dumps(test_data)
        
        result = await pickle_provider.get("test_key")
        
        assert result == test_data
        mock_redis.get.assert_called_once_with("test:test_key")
    
    async def test_get_not_found(self, provider, mock_redis):
        """Test get operation for non-existent key."""
        provider._redis = mock_redis
        mock_redis.get.return_value = None
        
        result = await provider.get("nonexistent_key")
        
        assert result is None
        mock_redis.get.assert_called_once_with("test:nonexistent_key")
    
    async def test_get_redis_error(self, provider, mock_redis):
        """Test get operation with Redis error."""
        from flowlib.providers.cache.redis.provider import RedisError
        
        provider._redis = mock_redis
        mock_redis.get.side_effect = RedisError("Get failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.get("test_key")
        
        assert "Redis get error: Get failed" in str(exc_info.value)
        assert exc_info.value.provider_context.provider_name == "test_redis"
    
    async def test_set_success_json(self, provider, mock_redis):
        """Test successful set operation with JSON serialization."""
        provider._redis = mock_redis
        test_data = {"key": "value", "number": 42}
        
        result = await provider.set("test_key", test_data, ttl=600)
        
        assert result is True
        mock_redis.set.assert_called_once_with(
            "test:test_key", 
            json.dumps(test_data), 
            ex=600
        )
    
    async def test_set_success_pickle(self, mock_redis):
        """Test successful set operation with pickle serialization."""
        # Create provider with pickle serialization
        settings = RedisCacheProviderSettings(
            host="localhost",
            port=6379,
            namespace="test",
            default_ttl=300,
            serialize_method="pickle"
        )
        pickle_provider = RedisCacheProvider("test_redis", settings)
        pickle_provider._redis = mock_redis
        test_data = {"key": "value", "number": 42}
        
        result = await pickle_provider.set("test_key", test_data)
        
        assert result is True
        mock_redis.set.assert_called_once_with(
            "test:test_key", 
            pickle.dumps(test_data), 
            ex=300  # default TTL
        )
    
    async def test_set_with_default_ttl(self, provider, mock_redis):
        """Test set operation with default TTL."""
        provider._redis = mock_redis
        test_data = "test_value"
        
        result = await provider.set("test_key", test_data)
        
        assert result is True
        mock_redis.set.assert_called_once_with(
            "test:test_key", 
            json.dumps(test_data), 
            ex=300  # default TTL from settings
        )
    
    async def test_set_redis_error(self, provider, mock_redis):
        """Test set operation with Redis error."""
        from flowlib.providers.cache.redis.provider import RedisError
        
        provider._redis = mock_redis
        mock_redis.set.side_effect = RedisError("Set failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.set("test_key", "test_value")
        
        assert "Redis set error: Set failed" in str(exc_info.value)
        assert exc_info.value.provider_context.provider_name == "test_redis"
    
    async def test_delete_success(self, provider, mock_redis):
        """Test successful delete operation."""
        provider._redis = mock_redis
        mock_redis.delete.return_value = 1
        
        result = await provider.delete("test_key")
        
        assert result is True
        mock_redis.delete.assert_called_once_with("test:test_key")
    
    async def test_delete_not_found(self, provider, mock_redis):
        """Test delete operation for non-existent key."""
        provider._redis = mock_redis
        mock_redis.delete.return_value = 0
        
        result = await provider.delete("nonexistent_key")
        
        assert result is False
        mock_redis.delete.assert_called_once_with("test:nonexistent_key")
    
    async def test_delete_redis_error(self, provider, mock_redis):
        """Test delete operation with Redis error."""
        from flowlib.providers.cache.redis.provider import RedisError
        
        provider._redis = mock_redis
        mock_redis.delete.side_effect = RedisError("Delete failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.delete("test_key")
        
        assert "Redis delete error: Delete failed" in str(exc_info.value)
        assert exc_info.value.provider_context.provider_name == "test_redis"
    
    async def test_exists_true(self, provider, mock_redis):
        """Test exists operation for existing key."""
        provider._redis = mock_redis
        mock_redis.exists.return_value = 1
        
        result = await provider.exists("test_key")
        
        assert result is True
        mock_redis.exists.assert_called_once_with("test:test_key")
    
    async def test_exists_false(self, provider, mock_redis):
        """Test exists operation for non-existent key."""
        provider._redis = mock_redis
        mock_redis.exists.return_value = 0
        
        result = await provider.exists("nonexistent_key")
        
        assert result is False
        mock_redis.exists.assert_called_once_with("test:nonexistent_key")
    
    async def test_exists_redis_error(self, provider, mock_redis):
        """Test exists operation with Redis error."""
        from flowlib.providers.cache.redis.provider import RedisError
        
        provider._redis = mock_redis
        mock_redis.exists.side_effect = RedisError("Exists failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.exists("test_key")
        
        assert "Redis exists error: Exists failed" in str(exc_info.value)
        assert exc_info.value.provider_context.provider_name == "test_redis"
    
    async def test_ttl_with_value(self, provider, mock_redis):
        """Test TTL operation for key with TTL."""
        provider._redis = mock_redis
        mock_redis.ttl.return_value = 300
        
        result = await provider.ttl("test_key")
        
        assert result == 300
        mock_redis.ttl.assert_called_once_with("test:test_key")
    
    async def test_ttl_no_expiration(self, provider, mock_redis):
        """Test TTL operation for key without expiration."""
        provider._redis = mock_redis
        mock_redis.ttl.return_value = -1
        
        result = await provider.ttl("test_key")
        
        assert result == -1
        mock_redis.ttl.assert_called_once_with("test:test_key")
    
    async def test_ttl_key_not_exists(self, provider, mock_redis):
        """Test TTL operation for non-existent key."""
        provider._redis = mock_redis
        mock_redis.ttl.return_value = -2
        
        result = await provider.ttl("nonexistent_key")
        
        assert result is None
        mock_redis.ttl.assert_called_once_with("test:nonexistent_key")
    
    async def test_ttl_redis_error(self, provider, mock_redis):
        """Test TTL operation with Redis error."""
        from flowlib.providers.cache.redis.provider import RedisError
        
        provider._redis = mock_redis
        mock_redis.ttl.side_effect = RedisError("TTL failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.ttl("test_key")
        
        assert "Redis TTL error: TTL failed" in str(exc_info.value)
        assert exc_info.value.provider_context.provider_name == "test_redis"
    
    async def test_clear_with_namespace(self, provider, mock_redis):
        """Test clear operation with namespace."""
        provider._redis = mock_redis
        
        # Mock scan results
        mock_redis.scan.side_effect = [
            (b'10', [b'test:key1', b'test:key2']),
            (b'0', [b'test:key3'])
        ]
        mock_redis.delete.return_value = 2
        
        result = await provider.clear()
        
        assert result is True
        # Should call scan with pattern
        assert mock_redis.scan.call_count == 2
        mock_redis.scan.assert_any_call(cursor=b'0', match="test:*", count=100)
        mock_redis.scan.assert_any_call(cursor=b'10', match="test:*", count=100)
        
        # Should delete keys in batches
        assert mock_redis.delete.call_count == 2
        mock_redis.delete.assert_any_call(b'test:key1', b'test:key2')
        mock_redis.delete.assert_any_call(b'test:key3')
    
    async def test_clear_without_namespace(self, mock_redis):
        """Test clear operation without namespace."""
        # Create provider without namespace
        settings = RedisCacheProviderSettings(
            host="localhost",
            port=6379,
            namespace=None
        )
        no_namespace_provider = RedisCacheProvider("test_redis", settings)
        no_namespace_provider._redis = mock_redis
        
        result = await no_namespace_provider.clear()
        
        assert result is True
        mock_redis.flushdb.assert_called_once()
        mock_redis.scan.assert_not_called()
        mock_redis.delete.assert_not_called()
    
    async def test_clear_redis_error(self, provider, mock_redis):
        """Test clear operation with Redis error."""
        from flowlib.providers.cache.redis.provider import RedisError
        
        provider._redis = mock_redis
        mock_redis.scan.side_effect = RedisError("Scan failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.clear()
        
        assert "Redis clear error: Scan failed" in str(exc_info.value)
        assert exc_info.value.provider_context.provider_name == "test_redis"
    
    async def test_make_namespaced_key(self, provider):
        """Test key namespacing."""
        # With namespace
        key = provider.make_namespaced_key("test_key")
        assert key == "test:test_key"
        
        # Without namespace
        settings = RedisCacheProviderSettings(
            host="localhost",
            port=6379,
            namespace=None
        )
        no_namespace_provider = RedisCacheProvider("test_redis", settings)
        key = no_namespace_provider.make_namespaced_key("test_key")
        assert key == "test_key"
    
    def test_provider_registration(self):
        """Test that provider is properly registered."""
        # This would be tested in integration tests
        # Here we just verify the decorator is applied
        assert hasattr(RedisCacheProvider, '__provider_type__')
        assert hasattr(RedisCacheProvider, '__provider_name__')


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis package not available")
@pytest.mark.integration
class TestRedisCacheProviderIntegration:
    """Integration tests for Redis cache provider.
    
    These tests require a running Redis instance.
    """
    
    @pytest.fixture
    def settings(self, redis_settings):
        """Create integration test settings from global config."""
        return redis_settings
    
    @pytest.fixture
    async def provider(self, settings):
        """Create and initialize provider for integration tests."""
        provider = RedisCacheProvider(name="integration_redis", settings=settings)
        
        try:
            await provider._initialize()
            # Clear any existing test data
            await provider.clear()
            yield provider
        finally:
            # Cleanup
            try:
                await provider.clear()
                await provider._shutdown()
            except:
                pass
    
    async def test_full_cache_cycle(self, provider):
        """Test complete cache operations cycle."""
        # Test data
        test_data = {
            "string": "hello",
            "number": 42,
            "list": [1, 2, 3],
            "nested": {"key": "value"}
        }
        
        # Key should not exist initially
        assert await provider.exists("test_key") is False
        # Removed redundant context.get() test - strict validation
        
        # Set value
        assert await provider.set("test_key", test_data) is True
        
        # Key should now exist
        assert await provider.exists("test_key") is True
        
        # Get value
        retrieved = await provider.get("test_key")
        assert retrieved == test_data
        
        # Check TTL
        ttl = await provider.ttl("test_key")
        assert ttl is not None and ttl > 0
        
        # Delete value
        assert await provider.delete("test_key") is True
        
        # Key should no longer exist
        assert await provider.exists("test_key") is False
        # Removed redundant context.get() test - strict validation
    
    async def test_ttl_expiration(self, provider):
        """Test TTL expiration."""
        # Set value with short TTL
        await provider.set("ttl_test", "value", ttl=1)
        
        # Should exist immediately
        assert await provider.exists("ttl_test") is True
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should no longer exist
        assert await provider.exists("ttl_test") is False
    
    async def test_clear_namespace(self, provider):
        """Test clearing with namespace."""
        # Set multiple values
        await provider.set("key1", "value1")
        await provider.set("key2", "value2")
        await provider.set("key3", "value3")
        
        # All should exist
        assert await provider.exists("key1") is True
        assert await provider.exists("key2") is True
        assert await provider.exists("key3") is True
        
        # Clear cache
        await provider.clear()
        
        # None should exist
        assert await provider.exists("key1") is False
        assert await provider.exists("key2") is False
        assert await provider.exists("key3") is False