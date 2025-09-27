"""Performance optimizations and caching for agent components."""

import asyncio
import time
import logging
from typing import Any, Dict, Optional, List, Callable, TypeVar, Tuple, Generic, cast, Awaitable
import types
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from functools import wraps
import hashlib
import json

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Input item type
R = TypeVar('R')  # Result type


class CacheStats(BaseModel):
    """Strict Pydantic model for cache statistics."""
    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True, strict=True)
    
    size: int = Field(..., ge=0, description="Current cache size")
    max_size: int = Field(..., gt=0, description="Maximum cache size")
    hits: int = Field(..., ge=0, description="Cache hits count")
    misses: int = Field(..., ge=0, description="Cache misses count")
    hit_rate: float = Field(..., ge=0.0, le=1.0, description="Cache hit rate")
    expired_entries: int = Field(..., ge=0, description="Number of expired entries")


class PerformanceStats(BaseModel):
    """Strict Pydantic model for performance statistics."""
    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True, strict=True)
    
    total_operations: int = Field(..., ge=0, description="Total operations performed")
    successful_operations: int = Field(..., ge=0, description="Successful operations count")
    failed_operations: int = Field(..., ge=0, description="Failed operations count")
    average_execution_time: float = Field(..., ge=0.0, description="Average execution time in seconds")
    cache_stats: CacheStats = Field(..., description="Cache performance statistics")


class FlowMetadata(BaseModel):
    """Strict Pydantic model for flow metadata."""
    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True, strict=True)
    
    name: str = Field(..., min_length=1, description="Flow name")
    description: Optional[str] = Field(default=None, description="Flow description")
    input_model: Optional[str] = Field(default=None, description="Input model class name")
    output_model: Optional[str] = Field(default=None, description="Output model class name")
    is_infrastructure: bool = Field(default=False, description="Whether flow is infrastructure-level")
    tags: List[str] = Field(default_factory=list, description="Flow tags")


class OperationStats(BaseModel):
    """Strict Pydantic model for operation statistics."""
    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True, strict=True)
    
    operation_name: str = Field(..., min_length=1, description="Operation name")
    call_count: int = Field(..., ge=0, description="Number of times called")
    total_time: float = Field(..., ge=0.0, description="Total execution time")
    average_time: float = Field(..., ge=0.0, description="Average execution time")
    min_time: float = Field(..., ge=0.0, description="Minimum execution time")
    max_time: float = Field(..., ge=0.0, description="Maximum execution time")
    success_count: int = Field(..., ge=0, description="Successful executions")
    error_count: int = Field(..., ge=0, description="Failed executions")


class AllStatsData(BaseModel):
    """Strict Pydantic model for all statistics."""
    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True, strict=True)
    
    operations: Dict[str, OperationStats] = Field(default_factory=dict, description="Per-operation statistics")
    cache_stats: CacheStats = Field(..., description="Cache statistics")
    total_operations: int = Field(..., ge=0, description="Total operations across all types")
    uptime_seconds: float = Field(..., ge=0.0, description="System uptime in seconds")


@dataclass
class CacheEntry:
    """Entry in the cache with metadata."""
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self) -> None:
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


class LRUCache:
    """LRU Cache with TTL support and analytics."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self._cache:
            self._misses += 1
            return None
        
        entry = self._cache[key]
        
        # Check if expired
        if entry.is_expired():
            del self._cache[key]
            self._misses += 1
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.touch()
        self._hits += 1
        
        return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache."""
        now = time.time()
        ttl = ttl or self.default_ttl
        
        if key in self._cache:
            # Update existing entry
            entry = self._cache[key]
            entry.value = value
            entry.created_at = now
            entry.last_accessed = now
            entry.ttl = ttl
            self._cache.move_to_end(key)
        else:
            # Add new entry
            if self.max_size == 0:
                # Can't store anything in zero-sized cache
                return
            if len(self._cache) >= self.max_size:
                # Evict least recently used
                self._cache.popitem(last=False)
                self._evictions += 1
            
            self._cache[key] = CacheEntry(
                value=value,
                created_at=now,
                last_accessed=now,
                access_count=1,
                ttl=ttl
            )
    
    def invalidate(self, key: str) -> bool:
        """Remove key from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        return len(expired_keys)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return CacheStats(
            size=len(self._cache),
            max_size=self.max_size,
            hits=self._hits,
            misses=self._misses,
            hit_rate=hit_rate,
            expired_entries=self._evictions
        )


class MemoryAnalytics:
    """Track memory access patterns and performance."""

    def __init__(self) -> None:
        self.access_patterns: Dict[str, int] = defaultdict(int)
        self.retrieval_times: List[float] = []
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.start_time = time.time()
    
    def record_access(
        self,
        operation: str,
        key: str,
        duration: float,
        success: bool = True,
        error_type: str = ""
    ) -> None:
        """Record a memory access operation."""
        self.access_patterns[key] += 1
        self.retrieval_times.append(duration)
        self.operation_counts[operation] += 1
        
        if not success and error_type:
            self.error_counts[error_type] += 1
    
    def get_hot_keys(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently accessed keys."""
        return sorted(
            self.access_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
    
    def get_performance_stats(self) -> PerformanceStats:
        """Get performance statistics."""
        if not self.retrieval_times:
            raise ValueError("Cannot generate performance stats - no operations recorded")
        
        times = self.retrieval_times
        times.sort()
        
        total_operations = len(times)
        successful_operations = total_operations - sum(self.error_counts.values())
        failed_operations = sum(self.error_counts.values())
        avg_duration = sum(times) / len(times)
        
        # Get cache stats if available
        cache_stats = CacheStats(
            size=0,
            max_size=100,  # Default values since this class doesn't have cache
            hits=0,
            misses=0,
            hit_rate=0.0,
            expired_entries=0
        )
        
        return PerformanceStats(
            total_operations=total_operations,
            successful_operations=successful_operations,
            failed_operations=failed_operations,
            average_execution_time=avg_duration,
            cache_stats=cache_stats
        )


class FlowMetadataCache:
    """Specialized cache for flow metadata."""
    
    def __init__(self, max_size: int = 500):
        self.cache = LRUCache(max_size=max_size, default_ttl=3600)  # 1 hour TTL
        self._metadata_computer: Optional[Callable[..., Dict[str, Any]]] = None
    
    def set_metadata_computer(self, computer: Callable[..., Dict[str, Any]]) -> None:
        """Set the function to compute metadata for flows."""
        self._metadata_computer = computer
    
    def get_flow_metadata(self, flow_name: str, flow_obj: object = None) -> Optional[FlowMetadata]:
        """Get flow metadata with caching."""
        cached_metadata = self.cache.get(flow_name)

        # Ensure cached value is correct type
        metadata: Optional[FlowMetadata] = None
        if cached_metadata is not None and isinstance(cached_metadata, FlowMetadata):
            metadata = cached_metadata

        if metadata is None and flow_obj and self._metadata_computer:
            # Compute and cache metadata
            raw_metadata = self._metadata_computer(flow_obj)
            if raw_metadata:
                # Convert raw metadata dict to FlowMetadata model - fail-fast approach
                metadata = FlowMetadata(
                    name=flow_name,
                    description=raw_metadata["description"] if "description" in raw_metadata else None,
                    input_model=raw_metadata["input_model"] if "input_model" in raw_metadata else None,
                    output_model=raw_metadata["output_model"] if "output_model" in raw_metadata else None,
                    is_infrastructure=raw_metadata["is_infrastructure"] if "is_infrastructure" in raw_metadata else False,
                    tags=raw_metadata["tags"] if "tags" in raw_metadata else []
                )
                self.cache.put(flow_name, metadata)
                logger.debug(f"Computed and cached metadata for flow: {flow_name}")
        
        return metadata
    
    def invalidate_flow(self, flow_name: str) -> None:
        """Invalidate cached metadata for a flow."""
        self.cache.invalidate(flow_name)


class BatchProcessor(Generic[T, R]):
    """Batch processor for efficient bulk operations."""

    def __init__(self, batch_size: int = 50, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self._pending_items: List[Tuple[T, asyncio.Future[R], Callable[[List[T]], Awaitable[List[R]]]]] = []
        self._processor_task: Optional[asyncio.Task[None]] = None
        self._last_flush = time.time()

    async def add_item(self, item: T, processor: Callable[[List[T]], Awaitable[List[R]]]) -> R:
        """Add item to batch and return result when processed."""
        future: asyncio.Future[R] = asyncio.Future()
        self._pending_items.append((item, future, processor))
        
        # Start processor if not running
        if self._processor_task is None or self._processor_task.done():
            self._processor_task = asyncio.create_task(self._process_batches())
        
        return await future
    
    async def _process_batches(self) -> None:
        """Process items in batches."""
        while True:
            # Wait for items or timeout
            time.time()
            while (
                len(self._pending_items) < self.batch_size and
                time.time() - self._last_flush < self.max_wait_time
            ):
                await asyncio.sleep(0.01)  # Small sleep to avoid busy waiting
            
            if not self._pending_items:
                break
            
            # Process current batch
            batch = self._pending_items[:self.batch_size]
            self._pending_items = self._pending_items[self.batch_size:]
            
            await self._process_batch(batch)
            self._last_flush = time.time()
    
    async def _process_batch(self, batch: List[Tuple[T, asyncio.Future[R], Callable[[List[T]], Awaitable[List[R]]]]]) -> None:
        """Process a single batch."""
        try:
            # Group by processor function
            processors = defaultdict(list)
            for item, future, processor in batch:
                processors[processor].append((item, future))
            
            # Process each group
            for processor, items in processors.items():
                try:
                    # Extract items and futures
                    batch_items = [item for item, _ in items]
                    futures = [future for _, future in items]
                    
                    # Process batch
                    results = await processor(batch_items)
                    
                    # Set results
                    for future, result in zip(futures, results):
                        if not future.done():
                            future.set_result(result)
                            
                except Exception as e:
                    # Set error for all futures in this batch
                    for _, future in items:
                        if not future.done():
                            future.set_exception(e)
                            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")


class PerformanceMonitor:
    """Monitor performance of agent operations."""
    
    def __init__(self) -> None:
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.memory_analytics = MemoryAnalytics()
        self.start_time = time.time()
        
    def time_operation(self, operation_name: str) -> 'OperationTimer':
        """Context manager for timing operations."""
        return OperationTimer(self, operation_name)
    
    def record_operation(self, operation_name: str, duration: float, success: bool = True) -> None:
        """Record an operation's performance."""
        self.operation_times[operation_name].append(duration)
        self.operation_counts[operation_name] += 1
        
        if not success:
            self.error_counts[operation_name] += 1
    
    def get_operation_stats(self, operation_name: str) -> OperationStats:
        """Get statistics for a specific operation."""
        if operation_name not in self.operation_times:
            raise ValueError(f"No timing data recorded for operation: {operation_name}")
        
        times = self.operation_times[operation_name]
        if not times:
            raise ValueError(f"No timing data available for operation: {operation_name}")
        
        sorted(times)
        count = self.operation_counts[operation_name]
        errors = self.error_counts[operation_name]
        
        return OperationStats(
            operation_name=operation_name,
            call_count=count,
            total_time=sum(times),
            average_time=sum(times) / len(times),
            min_time=min(times),
            max_time=max(times),
            success_count=count - errors,
            error_count=errors
        )
    
    def get_all_stats(self) -> AllStatsData:
        """Get statistics for all operations."""
        operations_stats = {
            operation: self.get_operation_stats(operation)
            for operation in self.operation_counts.keys()
        }
        
        # Calculate overall totals
        total_operations = sum(stats.call_count for stats in operations_stats.values())
        
        # Create default cache stats since this class doesn't have cache
        cache_stats = CacheStats(
            size=0,
            max_size=100,
            hits=0,
            misses=0,
            hit_rate=0.0,
            expired_entries=0
        )
        
        return AllStatsData(
            operations=operations_stats,
            cache_stats=cache_stats,
            total_operations=total_operations,
            uptime_seconds=time.time() - self.start_time
        )


class OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time: Optional[float] = None
        self.success = True
    
    def __enter__(self) -> 'OperationTimer':
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[types.TracebackType]) -> None:
        if self.start_time:
            duration = time.time() - self.start_time
            success = exc_type is None
            self.monitor.record_operation(self.operation_name, duration, success)
    
    async def __aenter__(self) -> 'OperationTimer':
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[types.TracebackType]) -> None:
        if self.start_time:
            duration = time.time() - self.start_time
            success = exc_type is None
            self.monitor.record_operation(self.operation_name, duration, success)


def cached(
    cache: LRUCache,
    key_func: Optional[Callable[..., str]] = None,
    ttl: Optional[float] = None
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for caching function results."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_data = {
                    "func": func.__name__,
                    "args": str(args),
                    "kwargs": str(sorted(kwargs.items()))
                }
                cache_key = hashlib.md5(
                    json.dumps(key_data, sort_keys=True).encode()
                ).hexdigest()
            
            # Try cache first
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cast(T, cached_result)
            
            # Compute and cache result
            result = await func(*args, **kwargs)
            cache.put(cache_key, result, ttl=ttl)
            return result
        
        return wrapper
    return decorator


class ParallelExecutor:
    """Execute operations in parallel with concurrency control."""
    
    def __init__(self, max_concurrency: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.active_tasks = 0
    
    async def execute_parallel(
        self,
        operations: List[Callable[..., Any]],
        *args: Any,
        **kwargs: Any
    ) -> List[Any]:
        """Execute operations in parallel."""
        async def bounded_operation(op: Callable[..., Any]) -> Any:
            async with self.semaphore:
                self.active_tasks += 1
                try:
                    return await op(*args, **kwargs)
                finally:
                    self.active_tasks -= 1
        
        tasks = [bounded_operation(op) for op in operations]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_stats(self) -> Dict[str, int]:
        """Get executor statistics."""
        return {
            "active_tasks": self.active_tasks,
            "available_slots": self.semaphore._value
        }