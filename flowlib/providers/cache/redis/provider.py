"""Redis provider for the Cache provider API.

This module implements a Redis-based cache provider using the async Redis client.
"""

import json
import logging
import pickle
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

try:
    import redis
except ImportError:
    pass

from pydantic import Field

from flowlib.core.errors.errors import ErrorContext, ProviderError
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.cache.base import CacheProvider, CacheProviderSettings
from flowlib.providers.core.decorators import provider

# Removed ProviderType import - using config-driven provider access

logger = logging.getLogger(__name__)

# Type annotation fallback handled by import statements

if TYPE_CHECKING:
    import redis.asyncio as redis_client
    from redis.asyncio import Redis as RedisClient
    from redis.asyncio.connection import ConnectionPool as AsyncConnectionPool
    from redis.exceptions import RedisError as RedisException
    REDIS_AVAILABLE = True
else:
    try:
        import redis.asyncio as redis_client
        from redis.asyncio import Redis as RedisClient
        from redis.asyncio.connection import ConnectionPool as AsyncConnectionPool
        from redis.exceptions import RedisError as RedisException
        REDIS_AVAILABLE = True
    except ImportError:
        logger.warning("redis package not found. Install with 'pip install redis'")
        # Define placeholders for type checking
        class _RedisClientModule:
            pass

        class AsyncConnectionPool:  # type: ignore
            pass

        class RedisError(Exception):  # type: ignore
            pass

        class RedisClient:  # type: ignore
            pass

        redis_client = _RedisClientModule()  # type: ignore
        REDIS_AVAILABLE = False


class RedisCacheProviderSettings(CacheProviderSettings):
    """Redis cache provider settings - direct inheritance, only Redis-specific fields.
    
    Redis requires:
    1. Connection details (host, port, db)
    2. Authentication (username, password)
    3. Connection pooling configuration
    4. Redis-specific features (sentinel, encoding)
    """

    # Connection settings
    host: str = Field(default="localhost", description="Redis server host (e.g., 'localhost', 'redis.example.com')")
    port: int = Field(default=6379, description="Redis server port (default: 6379)")
    db: int = Field(default=0, description="Redis database number (0-15, default: 0)")
    username: Optional[str] = Field(default=None, description="Redis username for ACL auth (Redis 6.0+)")
    password: Optional[str] = Field(default=None, description="Redis password (e.g., 'mySecurePassword123')")

    # Cache behavior settings - inherit from base CacheProviderSettings
    serialize_method: str = Field(default="json", description="Serialization method: json or pickle")

    # Connection pool settings
    pool_size: int = Field(default=10, description="Maximum connection pool size")
    socket_timeout: Optional[float] = Field(default=None, description="Socket timeout in seconds")
    socket_connect_timeout: Optional[float] = Field(default=None, description="Socket connection timeout in seconds")
    socket_keepalive: bool = Field(default=False, description="Whether to use socket keepalive")
    socket_keepalive_options: Optional[Dict[int, Union[int, bytes]]] = Field(default=None, description="Socket keepalive options")

    # Redis encoding settings
    encoding: str = Field(default="utf-8", description="Redis response encoding")
    encoding_errors: str = Field(default="strict", description="How to handle encoding errors")
    decode_responses: bool = Field(default=False, description="Whether to decode responses to strings")

    # Redis Sentinel settings (for high availability)
    sentinel_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional options for Redis Sentinel")
    sentinel: Optional[List[str]] = Field(default=None, description="List of Redis Sentinel nodes")
    sentinel_master: Optional[str] = Field(default=None, description="Name of the Redis Sentinel master")



@provider(provider_type="cache", name="redis-cache", settings_class=RedisCacheProviderSettings)
class RedisCacheProvider(CacheProvider[RedisCacheProviderSettings]):
    """Redis implementation of the CacheProvider.
    
    This provider uses Redis for caching, supporting all standard
    cache operations with TTL support, atomic operations, and
    distributed locking mechanisms.
    """

    def __init__(self, name: str = "redis-cache", settings: Optional[RedisCacheProviderSettings] = None):
        """Initialize Redis cache provider.
        
        Args:
            name: Unique provider name
            settings: Optional Redis-specific provider settings
        """
        # Use Redis-specific settings or create default
        redis_settings = settings or RedisCacheProviderSettings()
        super().__init__(name=name, settings=redis_settings)
        self._redis_settings = redis_settings
        self._pool: Optional[AsyncConnectionPool] = None
        self._redis: Optional[RedisClient] = None

    async def _initialize(self) -> None:
        """Initialize the Redis connection pool.
        
        Raises:
            ProviderError: If Redis connection cannot be established
        """
        try:
            # Create a connection pool - use settings with their validated defaults
            self._pool = AsyncConnectionPool(
                host=self._redis_settings.host,
                port=self._redis_settings.port,
                db=self._redis_settings.db,
                username=self._redis_settings.username,
                password=self._redis_settings.password,
                socket_timeout=self._redis_settings.socket_timeout,
                socket_connect_timeout=self._redis_settings.socket_connect_timeout,
                socket_keepalive=self._redis_settings.socket_keepalive,
                socket_keepalive_options=self._redis_settings.socket_keepalive_options,
                encoding=self._redis_settings.encoding,
                encoding_errors=self._redis_settings.encoding_errors,
                decode_responses=self._redis_settings.decode_responses,
                max_connections=self._redis_settings.pool_size,
            )

            # Create Redis client
            self._redis = redis_client.Redis(connection_pool=self._pool)

            # Test connection
            if not await self.check_connection():
                raise ProviderError(
                    message="Failed to connect to Redis server",
                    context=ErrorContext.create(
                        flow_name="redis_cache_provider",
                        error_type="ConnectionError",
                        error_location="_initialize",
                        component=self.name,
                        operation="connection_test"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="cache",
                        operation="connection_test",
                        retry_count=0
                    )
                )

            # Mark as initialized
            logger.info(f"Redis cache provider '{self.name}' initialized successfully")

        except RedisException as e:
            # Wrap Redis errors
            raise ProviderError(
                message=f"Redis initialization error: {str(e)}",
                context=ErrorContext.create(
                    flow_name="redis_cache_provider",
                    error_type="RedisError",
                    error_location="_initialize",
                    component=self.name,
                    operation="redis_initialization"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="cache",
                    operation="redis_initialization",
                    retry_count=0
                ),
                cause=e
            )
        except Exception as e:
            # Re-raise other errors
            raise ProviderError(
                message=f"Failed to initialize Redis provider: {str(e)}",
                context=ErrorContext.create(
                    flow_name="redis_cache_provider",
                    error_type="InitializationError",
                    error_location="_initialize",
                    component=self.name,
                    operation="provider_initialization"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="cache",
                    operation="provider_initialization",
                    retry_count=0
                ),
                cause=e
            )

    async def _shutdown(self) -> None:
        """Close Redis connections and release resources."""
        # Close pool
        if self._pool:
            await self._pool.disconnect()
            self._pool = None

        # Clear client
        self._redis = None

        # Mark as not initialized
        logger.info(f"Redis cache provider '{self.name}' shut down")

    async def check_connection(self) -> bool:
        """Check if Redis connection is active.
        
        Returns:
            True if connection is active, False otherwise
        """
        if self._redis is None:
            return False

        try:
            # Simple PING command
            result = await self._redis.ping()
            return bool(result)
        except Exception:
            return False

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from Redis.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
            
        Raises:
            ProviderError: If retrieval fails
        """
        if self._redis is None:
            raise ProviderError(
                message="Redis client not initialized",
                context=ErrorContext.create(
                    flow_name="redis_provider",
                    error_type="InitializationError",
                    error_location=f"{self.__class__.__name__}",
                    component=self.name,
                    operation="cache_access"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="cache",
                    operation="cache_access",
                    retry_count=0
                )
            )

        try:
            # Create namespaced key
            ns_key = self.make_namespaced_key(key)

            # Get value from Redis
            value = await self._redis.get(ns_key)

            # Return None if not found
            if value is None:
                return None

            # Deserialize value based on serialization method
            if self._redis_settings.serialize_method == "json":
                if isinstance(value, bytes):
                    value = value.decode(self._redis_settings.encoding)
                return json.loads(value)
            elif self._redis_settings.serialize_method == "pickle":
                return pickle.loads(value)
            else:
                return value

        except RedisException as e:
            # Wrap Redis errors
            raise ProviderError(
                message=f"Redis get error: {str(e)}",
                context=ErrorContext.create(
                    flow_name="redis_cache_provider",
                    error_type="RedisError",
                    error_location="get",
                    component=self.name,
                    operation=f"key_operation_{key}"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="cache",
                    operation=f"get_{key}",
                    retry_count=0
                ),
                cause=e
            )
        except Exception as e:
            # Re-raise other errors
            raise ProviderError(
                message=f"Failed to get value from Redis: {str(e)}",
                context=ErrorContext.create(
                    flow_name="redis_cache_provider",
                    error_type="GetError",
                    error_location="get",
                    component=self.name,
                    operation=f"key_operation_{key}"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="cache",
                    operation=f"get_{key}",
                    retry_count=0
                ),
                cause=e
            )

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in Redis.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for default)
            
        Returns:
            True if value was cached successfully
            
        Raises:
            ProviderError: If caching fails
        """
        if self._redis is None:
            raise ProviderError(
                message="Redis client not initialized",
                context=ErrorContext.create(
                    flow_name="redis_provider",
                    error_type="InitializationError",
                    error_location=f"{self.__class__.__name__}",
                    component=self.name,
                    operation="cache_access"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="cache",
                    operation="cache_access",
                    retry_count=0
                )
            )

        try:
            # Create namespaced key
            ns_key = self.make_namespaced_key(key)

            # Set TTL to default if not specified
            if ttl is None:
                ttl = self._redis_settings.default_ttl

            # Serialize value based on serialization method
            serialized: Union[str, bytes]
            if self._redis_settings.serialize_method == "json":
                serialized = json.dumps(value)
            elif self._redis_settings.serialize_method == "pickle":
                serialized = pickle.dumps(value)
            else:
                serialized = value

            # Set value in Redis with TTL
            result = await self._redis.set(ns_key, serialized, ex=ttl)
            return result is True

        except RedisException as e:
            # Wrap Redis errors
            raise ProviderError(
                message=f"Redis set error: {str(e)}",
                context=ErrorContext.create(
                    flow_name="redis_cache_provider",
                    error_type="RedisError",
                    error_location="set",
                    component=self.name,
                    operation=f"key_operation_{key}"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="cache",
                    operation=f"set_{key}",
                    retry_count=0
                ),
                cause=e
            )
        except Exception as e:
            # Re-raise other errors
            raise ProviderError(
                message=f"Failed to set value in Redis: {str(e)}",
                context=ErrorContext.create(
                    flow_name="redis_cache_provider",
                    error_type="SetError",
                    error_location="set",
                    component=self.name,
                    operation=f"key_operation_{key}"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="cache",
                    operation=f"set_{key}",
                    retry_count=0
                ),
                cause=e
            )

    async def delete(self, key: str) -> bool:
        """Delete a value from Redis.
        
        Args:
            key: Cache key
            
        Returns:
            True if value was deleted
            
        Raises:
            ProviderError: If deletion fails
        """
        if self._redis is None:
            raise ProviderError(
                message="Redis client not initialized",
                context=ErrorContext.create(
                    flow_name="redis_provider",
                    error_type="InitializationError",
                    error_location=f"{self.__class__.__name__}",
                    component=self.name,
                    operation="cache_access"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="cache",
                    operation="cache_access",
                    retry_count=0
                )
            )

        try:
            # Create namespaced key
            ns_key = self.make_namespaced_key(key)

            # Delete value from Redis
            result = await self._redis.delete(ns_key)
            return bool(result > 0)

        except RedisException as e:
            # Wrap Redis errors
            raise ProviderError(
                message=f"Redis delete error: {str(e)}",
                context=ErrorContext.create(
                    flow_name="redis_cache_provider",
                    error_type="RedisError",
                    error_location="delete",
                    component=self.name,
                    operation=f"key_operation_{key}"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="cache",
                    operation=f"delete_{key}",
                    retry_count=0
                ),
                cause=e
            )
        except Exception as e:
            # Re-raise other errors
            raise ProviderError(
                message=f"Failed to delete value from Redis: {str(e)}",
                context=ErrorContext.create(
                    flow_name="redis_cache_provider",
                    error_type="DeleteError",
                    error_location="delete",
                    component=self.name,
                    operation=f"key_operation_{key}"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="cache",
                    operation=f"delete_{key}",
                    retry_count=0
                ),
                cause=e
            )

    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
            
        Raises:
            ProviderError: If check fails
        """
        if self._redis is None:
            raise ProviderError(
                message="Redis client not initialized",
                context=ErrorContext.create(
                    flow_name="redis_provider",
                    error_type="InitializationError",
                    error_location=f"{self.__class__.__name__}",
                    component=self.name,
                    operation="cache_access"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="cache",
                    operation="cache_access",
                    retry_count=0
                )
            )

        try:
            # Create namespaced key
            ns_key = self.make_namespaced_key(key)

            # Check if key exists in Redis
            result = await self._redis.exists(ns_key)
            return bool(result > 0)

        except RedisException as e:
            # Wrap Redis errors
            raise ProviderError(
                message=f"Redis exists error: {str(e)}",
                context=ErrorContext.create(
                    flow_name="redis_cache_provider",
                    error_type="RedisError",
                    error_location="exists",
                    component=self.name,
                    operation=f"key_operation_{key}"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="cache",
                    operation=f"exists_{key}",
                    retry_count=0
                ),
                cause=e
            )
        except Exception as e:
            # Re-raise other errors
            raise ProviderError(
                message=f"Failed to check key existence in Redis: {str(e)}",
                context=ErrorContext.create(
                    flow_name="redis_cache_provider",
                    error_type="ExistsError",
                    error_location="exists",
                    component=self.name,
                    operation=f"key_operation_{key}"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="cache",
                    operation=f"exists_{key}",
                    retry_count=0
                ),
                cause=e
            )

    async def ttl(self, key: str) -> Optional[int]:
        """Get the remaining TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Remaining TTL in seconds, or None if key doesn't exist
            
        Raises:
            ProviderError: If TTL check fails
        """
        try:
            # Create namespaced key
            ns_key = self.make_namespaced_key(key)

            # Get TTL from Redis
            if self._redis is None:
                raise ProviderError(
                    message="Redis client not initialized",
                    context=ErrorContext.create(
                        flow_name="redis_cache_provider",
                        error_type="RedisError",
                        error_location="get_ttl",
                        component=self.name,
                        operation="get_ttl"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="cache",
                        operation="get_ttl",
                        retry_count=0
                    )
                )
            ttl = cast(int, await self._redis.ttl(ns_key))

            # Redis returns -2 if key doesn't exist, -1 if no TTL set
            if ttl == -2:
                return None
            elif ttl == -1:
                return -1  # No expiration
            else:
                return ttl

        except RedisException as e:
            # Wrap Redis errors
            raise ProviderError(
                message=f"Redis TTL error: {str(e)}",
                context=ErrorContext.create(
                    flow_name="redis_cache_provider",
                    error_type="RedisError",
                    error_location="ttl",
                    component=self.name,
                    operation=f"key_operation_{key}"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="cache",
                    operation=f"ttl_{key}",
                    retry_count=0
                ),
                cause=e
            )
        except Exception as e:
            # Re-raise other errors
            raise ProviderError(
                message=f"Failed to get TTL from Redis: {str(e)}",
                context=ErrorContext.create(
                    flow_name="redis_cache_provider",
                    error_type="TtlError",
                    error_location="ttl",
                    component=self.name,
                    operation=f"key_operation_{key}"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="cache",
                    operation=f"ttl_{key}",
                    retry_count=0
                ),
                cause=e
            )

    async def clear(self) -> bool:
        """Clear all values from the cache with the current namespace.
        
        Returns:
            True if cache was cleared successfully
            
        Raises:
            ProviderError: If clearing fails
        """
        try:
            if self._redis is None:
                raise ProviderError(
                    message="Redis client not initialized",
                    context=ErrorContext.create(
                        flow_name="redis_cache_provider",
                        error_type="RedisError",
                        error_location="clear",
                        component=self.name,
                        operation="clear"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="cache",
                        operation="clear",
                        retry_count=0
                    )
                )

            # If a namespace is set, only clear keys in that namespace
            if self._redis_settings.namespace:
                pattern = f"{self._redis_settings.namespace}:*"
                # Get all keys matching the pattern
                cursor: int = 0
                deleted_count = 0

                while True:
                    cursor, keys = await self._redis.scan(cursor=cursor, match=pattern, count=100)

                    if keys:
                        # Delete keys in batches
                        result = await self._redis.delete(*keys)
                        deleted_count += result

                    # Exit when no more keys (cursor is 0)
                    if cursor == 0:
                        break

                return True
            else:
                # Clear all keys (dangerous!)
                await self._redis.flushdb()
                return True

        except RedisException as e:
            # Wrap Redis errors
            raise ProviderError(
                message=f"Redis clear error: {str(e)}",
                context=ErrorContext.create(
                    flow_name="redis_cache_provider",
                    error_type="RedisError",
                    error_location="clear",
                    component=self.name,
                    operation=f"namespace_{self._redis_settings.namespace}"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="cache",
                    operation="clear",
                    retry_count=0
                ),
                cause=e
            )
        except Exception as e:
            # Re-raise other errors
            raise ProviderError(
                message=f"Failed to clear cache in Redis: {str(e)}",
                context=ErrorContext.create(
                    flow_name="redis_cache_provider",
                    error_type="ClearError",
                    error_location="clear",
                    component=self.name,
                    operation=f"namespace_{self._redis_settings.namespace}"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="cache",
                    operation="clear",
                    retry_count=0
                ),
                cause=e
            )
