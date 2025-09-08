"""
Modern Working Memory Component.

Provides a simple, TTL-based key-value store for short-term memory using
the modernized agent framework patterns with config-driven providers.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta, timezone

from pydantic import Field
from flowlib.core.models import StrictBaseModel

from ...core.errors import MemoryError, ErrorContext
from .models import (
    MemoryStoreRequest,
    MemoryRetrieveRequest,
    MemorySearchRequest,
    MemoryContext
)
from .models import MemoryItem, MemorySearchResult

logger = logging.getLogger(__name__)


class WorkingMemoryConfig(StrictBaseModel):
    """Configuration for working memory."""
    
    default_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description="Default TTL for memory items in seconds"
    )
    max_items: int = Field(
        default=10000,
        ge=100,
        description="Maximum number of items to store"
    )
    cleanup_interval_seconds: int = Field(
        default=300,
        ge=60,
        description="Background cleanup interval in seconds"
    )
    max_memory_mb: int = Field(
        default=100,
        ge=10,
        description="Maximum memory usage in MB"
    )


class WorkingMemory:
    """Modern working memory implementation with TTL and resource management."""
    
    def __init__(self, config: Optional[WorkingMemoryConfig] = None):
        """Initialize working memory with instance-specific storage."""
        self._config = config or WorkingMemoryConfig()
        
        # Instance-specific storage (fixes global state bug)
        self._store: dict[str, dict[str, MemoryItem]] = {}
        self._ttl_map: dict[str, datetime] = {}
        self._contexts: Set[str] = set()
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Resource tracking
        self._item_count = 0
        self._memory_usage_bytes = 0
        
        self._initialized = False
        logger.info(f"Initialized WorkingMemory with config: {self._config}")
    
    @property
    def initialized(self) -> bool:
        """Check if memory is initialized."""
        return self._initialized
    
    async def initialize(self) -> None:
        """Initialize working memory and start cleanup task."""
        if self._initialized:
            return
            
        logger.info("Initializing WorkingMemory...")
        
        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self._initialized = True
        logger.info("WorkingMemory initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown working memory and cleanup resources."""
        if not self._initialized:
            return
            
        logger.info("Shutting down WorkingMemory...")
        
        self._shutdown = True
        
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clear all data
        self._store.clear()
        self._ttl_map.clear()
        self._contexts.clear()
        
        self._initialized = False
        logger.info("WorkingMemory shutdown completed")
    
    async def create_context(self, context_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a memory context."""
        if not self._initialized:
            raise MemoryError("WorkingMemory not initialized")
            
        logger.debug(f"Creating context: {context_name}")
        
        if context_name not in self._store:
            self._store[context_name] = {}
            self._contexts.add(context_name)
            
        logger.debug(f"Context '{context_name}' created/verified")
        return context_name
    
    async def store(self, request: MemoryStoreRequest) -> str:
        """Store a memory item with TTL."""
        if not self._initialized:
            raise MemoryError("WorkingMemory not initialized")
            
        # Cleanup expired items before storing
        await self._cleanup_expired()
        
        # Check resource limits
        if self._item_count >= self._config.max_items:
            raise MemoryError(
                f"Memory limit reached: {self._item_count}/{self._config.max_items} items",
                operation="store",
                context=request.context
            )
        
        # Ensure context exists
        if request.context not in self._store:
            await self.create_context(request.context)
        
        try:
            # Create TTL entry
            if request.ttl:
                ttl_delta = timedelta(seconds=request.ttl)
            else:
                ttl_delta = timedelta(seconds=self._config.default_ttl_seconds)
            
            expiry = datetime.now(timezone.utc) + ttl_delta
            key = f"{request.context}:{request.key}"
            
            # Create a MemoryItem to store
            memory_item = MemoryItem(
                key=request.key,
                value=request.value,
                context=request.context,
                metadata=request.metadata or {}
            )
            
            # Store the item
            self._store[request.context][request.key] = memory_item
            self._ttl_map[key] = expiry
            
            # Update resource tracking
            self._item_count += 1
            self._memory_usage_bytes += len(str(memory_item))
            
            logger.debug(f"Stored item '{request.key}' in context '{request.context}' with TTL {ttl_delta}")
            return request.key
            
        except Exception as e:
            raise MemoryError(
                f"Failed to store item '{request.key}': {str(e)}",
                operation="store",
                context=request.context,
                key=request.key,
                cause=e
            ) from e
    
    async def retrieve(self, request: MemoryRetrieveRequest) -> Optional[MemoryItem]:
        """Retrieve a memory item by key."""
        if not self._initialized:
            raise MemoryError("WorkingMemory not initialized")
            
        # Cleanup expired items
        await self._cleanup_expired()
        
        try:
            if request.context in self._store:
                context_store = self._store[request.context]
                item = context_store.get(request.key) if request.key in context_store else None
            else:
                context_store = {}
                item = None
            
            if item:
                logger.debug(f"Retrieved item '{request.key}' from context '{request.context}'")
            else:
                logger.debug(f"Item '{request.key}' not found in context '{request.context}'")
                
            return item
            
        except Exception as e:
            raise MemoryError(
                f"Failed to retrieve item '{request.key}': {str(e)}",
                operation="retrieve",
                context=request.context,
                key=request.key,
                cause=e
            ) from e
    
    async def search(self, request: MemorySearchRequest) -> List[MemorySearchResult]:
        """Search memory items by query string."""
        if not self._initialized:
            raise MemoryError("WorkingMemory not initialized")
            
        # Cleanup expired items
        await self._cleanup_expired()
        
        try:
            results = []
            context_store = self._store[request.context] if request.context in self._store else {}
            
            # Simple text-based search
            query_lower = request.query.lower()
            
            for key, item in context_store.items():
                # Check if query matches key or content
                matches = False
                
                if query_lower in key.lower():
                    matches = True
                elif hasattr(item, 'value') and query_lower in str(item.value).lower():
                    matches = True
                elif hasattr(item, 'metadata'):
                    metadata_str = str(item.metadata).lower()
                    if query_lower in metadata_str:
                        matches = True
                
                if matches:
                    results.append(MemorySearchResult(
                        item=item,
                        score=1.0,  # Simple binary matching
                        metadata={"match_type": "text_search"}
                    ))
            
            # Apply limit
            if request.limit and len(results) > request.limit:
                results = results[:request.limit]
            
            logger.debug(f"Found {len(results)} results for query '{request.query}' in context '{request.context}'")
            return results
            
        except Exception as e:
            raise MemoryError(
                f"Failed to search for '{request.query}': {str(e)}",
                operation="search",
                context=request.context,
                query=request.query,
                cause=e
            ) from e
    
    async def retrieve_relevant(
        self, 
        query: str, 
        context: Optional[str] = None, 
        limit: int = 10
    ) -> List[str]:
        """Retrieve relevant memories based on query."""
        if not self._initialized:
            raise MemoryError("WorkingMemory not initialized")
        if context is None:
            raise ValueError("Context cannot be None - explicit context is required")
            
        search_request = MemorySearchRequest(
            query=query,
            context=context,
            limit=limit
        )
        
        search_results = await self.search(search_request)
        
        # Convert to string list format
        return [
            f"{result.item.key}: {result.item.value if hasattr(result.item, 'value') else str(result.item)}"
            for result in search_results
        ]
    
    async def wipe_context(self, context: str) -> None:
        """Remove all items from a specific context."""
        if not self._initialized:
            raise MemoryError("WorkingMemory not initialized")
            
        try:
            if context in self._store:
                # Remove TTL entries for this context
                keys_to_remove = [key for key in self._ttl_map.keys() if key.startswith(f"{context}:")]
                for key in keys_to_remove:
                    del self._ttl_map[key]
                
                # Update resource tracking
                items_removed = len(self._store[context])
                self._item_count -= items_removed
                
                # Remove context store
                del self._store[context]
                self._contexts.discard(context)
                
                logger.info(f"Wiped context '{context}', removed {items_removed} items")
            else:
                logger.debug(f"Context '{context}' not found for wiping")
                
        except Exception as e:
            raise MemoryError(
                f"Failed to wipe context '{context}': {str(e)}",
                operation="wipe",
                context=context,
                cause=e
            ) from e
    
    async def _cleanup_expired(self) -> None:
        """Remove expired items from the store."""
        if not self._initialized:
            return
            
        now = datetime.now(timezone.utc)
        expired_keys = []
        
        # Find expired keys
        for key, expiry in self._ttl_map.items():
            if now > expiry:
                expired_keys.append(key)
        
        # Remove expired items
        for key in expired_keys:
            try:
                context, item_key = key.split(":", 1)
                if context in self._store and item_key in self._store[context]:
                    del self._store[context][item_key]
                    self._item_count -= 1
                del self._ttl_map[key]
            except Exception as e:
                logger.warning(f"Error removing expired key '{key}': {e}")
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired items")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        logger.info("Starting WorkingMemory cleanup loop")
        
        while not self._shutdown:
            try:
                await asyncio.sleep(self._config.cleanup_interval_seconds)
                
                if not self._shutdown:
                    await self._cleanup_expired()
                    
            except asyncio.CancelledError:
                logger.info("WorkingMemory cleanup loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in WorkingMemory cleanup loop: {e}")
                # Continue loop despite errors
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            "initialized": self._initialized,
            "item_count": self._item_count,
            "memory_usage_bytes": self._memory_usage_bytes,
            "memory_usage_mb": self._memory_usage_bytes / (1024 * 1024),
            "context_count": len(self._contexts),
            "contexts": list(self._contexts),
            "config": self._config.model_dump()
        }