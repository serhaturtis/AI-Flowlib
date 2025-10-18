"""In-memory implementation of the CacheProvider.

This module provides a concrete implementation of the CacheProvider
using an in-memory dictionary as the backend for caching.
"""

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from pydantic import Field

from flowlib.core.errors.errors import ErrorContext, ProviderError
from flowlib.core.errors.models import ProviderErrorContext

# Removed ProviderType import - using config-driven provider access
from flowlib.providers.cache.base import CacheProvider, CacheProviderSettings
from flowlib.providers.core.decorators import provider

logger = logging.getLogger(__name__)


class InMemoryCacheProviderSettings(CacheProviderSettings):
    """In-memory cache provider settings - direct inheritance, only in-memory specific fields.
    
    In-memory cache requires:
    1. Memory management (max size)
    2. Cleanup scheduling
    3. TTL configuration
    4. Namespace for key isolation
    
    No host/port/connection needed - purely in-memory.
    """

    # In-memory specific settings
    max_size: int = Field(default=1000, description="Maximum number of items to store in memory")
    cleanup_interval: int = Field(default=300, description="Cleanup interval in seconds (5 minutes)")
    default_ttl: int = Field(default=3600, description="Default TTL in seconds (1 hour)")
    # namespace inherited from CacheProviderSettings with default="default"



@provider(provider_type="cache", name="memory-cache", settings_class=InMemoryCacheProviderSettings)
class MemoryCacheProvider(CacheProvider[InMemoryCacheProviderSettings]):
    """In-memory implementation of the CacheProvider.
    
    This provider implements a simple in-memory cache for storing
    key-value pairs without persistence.
    """

    def __init__(self, name: str = "memory-cache", settings: Optional[InMemoryCacheProviderSettings] = None):
        """Initialize in-memory cache provider.
        
        Args:
            name: Provider instance name
            settings: Optional provider settings
        """
        super().__init__(name=name, settings=settings or InMemoryCacheProviderSettings())
        self._memory_settings = settings or InMemoryCacheProviderSettings()
        self._lock = asyncio.Lock()
        self._cache: OrderedDict[str, Any] = OrderedDict()  # For LRU eviction
        self._expiry: Dict[str, Optional[float]] = {}  # Maps keys to expiry times
        self._cleanup_task: Optional[asyncio.Task[None]] = None

    async def initialize(self) -> None:
        """Initialize the in-memory cache.
        
        This method sets up the cleanup task for expired keys.
        """
        # Mark as initialized first in parent
        await super().initialize()

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info(f"In-memory cache provider '{self.name}' initialized successfully")

    async def shutdown(self) -> None:
        """Release resources and stop cleanup task."""
        # Cancel cleanup task if it exists and is running
        if self._cleanup_task is not None:
            if not self._cleanup_task.done():
                self._cleanup_task.cancel()

            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                # Expected when cancelling
                pass
            except Exception as e:
                logger.warning(f"Error during cleanup task shutdown: {e}")
            finally:
                self._cleanup_task = None

        # Clear cache and expiry
        try:
            async with self._lock:
                self._cache.clear()
                self._expiry.clear()
        except Exception as e:
            logger.warning(f"Error during cache cleanup: {e}")

        # Mark as not initialized
        await super().shutdown()
        logger.info(f"In-memory cache provider '{self.name}' shut down")

    async def check_connection(self) -> bool:
        """Check if in-memory cache is active.
        
        Returns:
            True if active, False otherwise
        """
        # In-memory cache is always active if initialized
        return self.initialized

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired keys."""
        try:
            while True:
                # Wait for cleanup interval
                await asyncio.sleep(self._memory_settings.cleanup_interval)

                # Cleanup expired keys
                await self._cleanup_expired()

        except asyncio.CancelledError:
            # Task was cancelled, clean up one last time if still initialized
            if self.initialized:
                try:
                    await self._cleanup_expired()
                except Exception:
                    # Ignore errors during cancellation cleanup
                    pass
            raise

    async def _cleanup_expired(self) -> None:
        """Remove expired keys from cache."""
        now = time.time()
        to_delete = []

        # Find expired keys
        async with self._lock:
            for key, expiry_time in self._expiry.items():
                if expiry_time is not None and expiry_time <= now:
                    to_delete.append(key)

            # Delete expired keys
            for key in to_delete:
                if key in self._cache:
                    del self._cache[key]
                if key in self._expiry:
                    del self._expiry[key]

        if to_delete:
            logger.debug(f"Cleaned up {len(to_delete)} expired keys from in-memory cache")

    async def _enforce_max_size(self) -> None:
        """Enforce maximum cache size by evicting items."""
        if not self._memory_settings.max_size:
            return

        # Check if we're over the limit
        while len(self._cache) > self._memory_settings.max_size:
            # Use LRU eviction - remove the first item (least recently used)
            # OrderedDict maintains insertion order, and move_to_end puts recently used items at the end
            evicted_key, _ = self._cache.popitem(last=False)
            # Also remove from expiry dict
            if evicted_key in self._expiry:
                del self._expiry[evicted_key]

    async def _initialize(self) -> None:
        """Initialize in-memory cache provider.
        
        This is a simple implementation that just sets up the in-memory structures.
        """
        # Initialize cache data structures (OrderedDict for LRU)
        self._cache = OrderedDict()
        self._expiry = {}

        # Start cleanup task if needed
        if hasattr(self, '_cleanup_task') and self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from in-memory cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
            
        Raises:
            ProviderError: If retrieval fails
        """
        try:
            # Create namespaced key
            ns_key = self.make_namespaced_key(key)

            async with self._lock:
                # Check if key exists
                if ns_key not in self._cache:
                    return None

                # Check if key is expired
                if ns_key in self._expiry:
                    expiry_time = self._expiry[ns_key]
                    if expiry_time is not None and expiry_time <= time.time():
                        # Key is expired, remove it
                        del self._cache[ns_key]
                        del self._expiry[ns_key]
                        return None

                # Move to end for LRU
                value = self._cache[ns_key]
                self._cache.move_to_end(ns_key)

                return value

        except Exception as e:
            # Wrap errors
            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="cache",
                operation=f"get_{key}",
                retry_count=0
            )

            raise ProviderError(
                message=f"Failed to get value from in-memory cache: {str(e)}",
                context=ErrorContext.create(
                    flow_name="cache",
                    error_type="cache_error",
                    error_location="provider",
                    component="cache",
                    operation=f"key_operation_{key}"
                ),
                provider_context=provider_context,
                cause=e
            )

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in in-memory cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for default)
            
        Returns:
            True if value was cached successfully
            
        Raises:
            ProviderError: If caching fails
        """
        try:
            # Create namespaced key
            ns_key = self.make_namespaced_key(key)

            # Set TTL to default if not specified
            if ttl is None:
                ttl = self._memory_settings.default_ttl

            # Calculate expiry time
            expiry_time = time.time() + ttl if ttl > 0 else None

            async with self._lock:
                # Store value and expiry
                self._cache[ns_key] = value
                self._expiry[ns_key] = expiry_time

                # Move to end for LRU
                self._cache.move_to_end(ns_key)

                # Enforce max size
                await self._enforce_max_size()

            return True

        except Exception as e:
            # Wrap errors
            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="cache",
                operation=f"set_{key}",
                retry_count=0
            )

            raise ProviderError(
                message=f"Failed to set value in in-memory cache: {str(e)}",
                context=ErrorContext.create(
                    flow_name="cache",
                    error_type="cache_error",
                    error_location="provider",
                    component="cache",
                    operation=f"key_operation_{key}"
                ),
                provider_context=provider_context,
                cause=e
            )

    async def delete(self, key: str) -> bool:
        """Delete a value from in-memory cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if value was deleted
            
        Raises:
            ProviderError: If deletion fails
        """
        try:
            # Create namespaced key
            ns_key = self.make_namespaced_key(key)

            async with self._lock:
                # Delete value and expiry
                was_present = ns_key in self._cache

                if ns_key in self._cache:
                    del self._cache[ns_key]

                if ns_key in self._expiry:
                    del self._expiry[ns_key]

            return was_present

        except Exception as e:
            # Wrap errors
            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="cache",
                operation=f"delete_{key}",
                retry_count=0
            )

            raise ProviderError(
                message=f"Failed to delete value from in-memory cache: {str(e)}",
                context=ErrorContext.create(
                    flow_name="cache",
                    error_type="cache_error",
                    error_location="provider",
                    component="cache",
                    operation=f"key_operation_{key}"
                ),
                provider_context=provider_context,
                cause=e
            )

    async def exists(self, key: str) -> bool:
        """Check if a key exists in in-memory cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists and is not expired, False otherwise
            
        Raises:
            ProviderError: If check fails
        """
        try:
            # Create namespaced key
            ns_key = self.make_namespaced_key(key)

            async with self._lock:
                # Check if key exists
                if ns_key not in self._cache:
                    return False

                # Check if key is expired
                if ns_key in self._expiry:
                    expiry_time = self._expiry[ns_key]
                    if expiry_time is not None and expiry_time <= time.time():
                        # Key is expired, remove it
                        del self._cache[ns_key]
                        del self._expiry[ns_key]
                        return False

                return True

        except Exception as e:
            # Wrap errors
            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="cache",
                operation=f"exists_{key}",
                retry_count=0
            )

            raise ProviderError(
                message=f"Failed to check key existence in in-memory cache: {str(e)}",
                context=ErrorContext.create(
                    flow_name="cache",
                    error_type="cache_error",
                    error_location="provider",
                    component="cache",
                    operation=f"key_operation_{key}"
                ),
                provider_context=provider_context,
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

            async with self._lock:
                # Check if key exists
                if ns_key not in self._cache:
                    return None

                # Check if key has expiry
                if ns_key not in self._expiry:
                    return -1  # No expiration

                expiry_time = self._expiry[ns_key]
                if expiry_time is None:
                    return -1  # No expiration

                # Calculate remaining TTL
                remaining = expiry_time - time.time()

                # If expired, remove it and return None
                if remaining <= 0:
                    del self._cache[ns_key]
                    del self._expiry[ns_key]
                    return None

                return int(remaining)

        except Exception as e:
            # Wrap errors
            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="cache",
                operation=f"ttl_{key}",
                retry_count=0
            )

            raise ProviderError(
                message=f"Failed to get TTL from in-memory cache: {str(e)}",
                context=ErrorContext.create(
                    flow_name="cache",
                    error_type="cache_error",
                    error_location="provider",
                    component="cache",
                    operation=f"key_operation_{key}"
                ),
                provider_context=provider_context,
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
            # If a namespace is set, only clear keys in that namespace
            if self._memory_settings.namespace:
                prefix = f"{self._memory_settings.namespace}:"

                async with self._lock:
                    # Find keys to delete
                    keys_to_delete = [key for key in self._cache.keys()
                                     if key.startswith(prefix)]

                    # Delete keys
                    for key in keys_to_delete:
                        del self._cache[key]
                        if key in self._expiry:
                            del self._expiry[key]

                return True
            else:
                # Clear all keys
                async with self._lock:
                    self._cache.clear()
                    self._expiry.clear()

                return True

        except Exception as e:
            # Wrap errors
            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="cache",
                operation="clear_cache",
                retry_count=0
            )

            raise ProviderError(
                message=f"Failed to clear in-memory cache: {str(e)}",
                context=ErrorContext.create(
                    flow_name="cache",
                    error_type="cache_error",
                    error_location="provider",
                    component="cache",
                    operation="clear_cache"
                ),
                provider_context=provider_context,
                cause=e
            )
