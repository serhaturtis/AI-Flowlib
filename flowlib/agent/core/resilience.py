"""Error handling, retries, and resilience patterns for agent components."""

import asyncio
import time
import logging
from typing import Callable, TypeVar, Optional, Tuple, Type, Any, Dict, List, Awaitable
from functools import wraps
from enum import Enum

from flowlib.agent.core.errors import AgentError, ComponentError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class RetryableOperation:
    """Decorator for operations that should be retried on failure."""
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        max_backoff: float = 60.0,
        retryable_errors: Tuple[Type[Exception], ...] = (Exception,),
        non_retryable_errors: Tuple[Type[Exception], ...] = (ValueError, TypeError)
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.retryable_errors = retryable_errors
        self.non_retryable_errors = non_retryable_errors
    
    def __call__(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error = None

            for attempt in range(self.max_retries + 1):  # +1 for initial attempt
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"{func.__name__} succeeded after {attempt} retries")
                    return result
                    
                except self.non_retryable_errors as e:
                    logger.error(f"{func.__name__} failed with non-retryable error: {e}")
                    raise
                    
                except self.retryable_errors as e:
                    last_error = e
                    
                    if attempt < self.max_retries:
                        wait_time = min(
                            self.backoff_factor ** attempt,
                            self.max_backoff
                        )
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{self.max_retries + 1}), "
                            f"retrying in {wait_time:.2f}s: {e}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"{func.__name__} failed after {self.max_retries} retries: {e}")
            
            if last_error is not None:
                raise last_error
            else:
                raise ComponentError(
                    f"Operation {func.__name__} failed without specific error",
                    component_name="RetryableOperation",
                    operation=func.__name__
                )
        
        return wrapper


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        success_threshold: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        
    async def call(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Execute function through circuit breaker."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitOpenError(
                    f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}"
                )
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure(e)
            raise
    
    async def _on_success(self) -> None:
        """Handle successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker CLOSED after successful recovery")
        else:
            self.failure_count = 0
    
    async def _on_failure(self, error: Exception) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker OPEN again after failure in HALF_OPEN state")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.reset_timeout
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state for monitoring."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "time_since_last_failure": (
                time.time() - self.last_failure_time 
                if self.last_failure_time else None
            )
        }


class CircuitOpenError(AgentError):
    """Raised when circuit breaker is open."""
    pass


class TimeoutOperation:
    """Decorator for operations with timeout."""
    
    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
    
    def __call__(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.error(f"{func.__name__} timed out after {self.timeout_seconds}s")
                raise ComponentError(
                    f"Operation timed out: {func.__name__}",
                    component_name=getattr(args[0], 'name', 'unknown') if args else 'unknown',
                    operation=func.__name__
                )
        
        return wrapper


class RateLimiter:
    """Rate limiter for operations."""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: List[float] = []
    
    async def acquire(self) -> None:
        """Acquire permission to make a call."""
        now = time.time()
        
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        if len(self.calls) >= self.max_calls:
            # Calculate how long to wait
            oldest_call = min(self.calls)
            wait_time = self.time_window - (now - oldest_call)
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                return await self.acquire()  # Recursive call after waiting
        
        self.calls.append(now)


class BulkheadIsolation:
    """Bulkhead pattern for resource isolation."""
    
    def __init__(self, pool_size: int):
        if pool_size <= 0:
            raise ValueError("pool_size must be greater than 0")
        self.semaphore = asyncio.Semaphore(pool_size)
        self.active_operations = 0
        self.queued_operations = 0
    
    async def execute(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Execute function with resource isolation."""
        self.queued_operations += 1
        try:
            async with self.semaphore:
                self.queued_operations -= 1
                self.active_operations += 1
                try:
                    return await func(*args, **kwargs)
                finally:
                    self.active_operations -= 1
        except Exception:
            # Only decrement if we haven't acquired the semaphore yet
            if self.queued_operations > 0:
                self.queued_operations -= 1
            raise
    
    def get_stats(self) -> Dict[str, int]:
        """Get current bulkhead statistics."""
        return {
            "active_operations": self.active_operations,
            "queued_operations": self.queued_operations,
            "available_permits": self.semaphore._value
        }


class ErrorContext:
    """Context manager for enhanced error handling."""
    
    def __init__(
        self,
        operation: str,
        component_name: str = "unknown",
        reraise_as: Optional[Type[Exception]] = None
    ):
        self.operation = operation
        self.component_name = component_name
        self.reraise_as = reraise_as
        self.start_time: Optional[float] = None
    
    async def __aenter__(self) -> 'ErrorContext':
        self.start_time = time.time()
        logger.debug(f"Starting {self.operation} in {self.component_name}")
        return self
    
    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Any) -> None:
        duration = time.time() - self.start_time if self.start_time else 0
        
        if exc_type is None:
            logger.debug(f"Completed {self.operation} in {duration:.3f}s")
        else:
            logger.error(
                f"Failed {self.operation} after {duration:.3f}s: {exc_val}",
                exc_info=bool(exc_type)
            )
            
            if self.reraise_as:
                # Reraise as specified exception type
                if issubclass(self.reraise_as, ComponentError):
                    raise self.reraise_as(
                        message=f"{self.operation} failed",
                        component_name=self.component_name,
                        operation=self.operation,
                        cause=exc_val if isinstance(exc_val, Exception) else None
                    )
                else:
                    raise self.reraise_as(str(exc_val)) from exc_val


class ResilienceManager:
    """Manages resilience patterns for agent components."""
    
    def __init__(self) -> None:
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.bulkheads: Dict[str, BulkheadIsolation] = {}
    
    def get_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                reset_timeout=reset_timeout
            )
        return self.circuit_breakers[name]
    
    def get_rate_limiter(
        self,
        name: str,
        max_calls: int = 100,
        time_window: float = 60.0
    ) -> RateLimiter:
        """Get or create a rate limiter."""
        if name not in self.rate_limiters:
            self.rate_limiters[name] = RateLimiter(
                max_calls=max_calls,
                time_window=time_window
            )
        return self.rate_limiters[name]
    
    def get_bulkhead(
        self,
        name: str,
        pool_size: int = 10
    ) -> BulkheadIsolation:
        """Get or create a bulkhead."""
        if name not in self.bulkheads:
            self.bulkheads[name] = BulkheadIsolation(pool_size=pool_size)
        return self.bulkheads[name]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all resilience components."""
        return {
            "circuit_breakers": {
                name: cb.get_state() 
                for name, cb in self.circuit_breakers.items()
            },
            "bulkheads": {
                name: bh.get_stats()
                for name, bh in self.bulkheads.items()
            }
        }


# Convenience decorators for common patterns
def retryable(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    retryable_errors: Tuple[Type[Exception], ...] = (Exception,)
) -> RetryableOperation:
    """Convenience decorator for retryable operations."""
    return RetryableOperation(
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        retryable_errors=retryable_errors
    )


def with_timeout(timeout_seconds: float) -> TimeoutOperation:
    """Convenience decorator for timeout operations."""
    return TimeoutOperation(timeout_seconds)


def with_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    reset_timeout: float = 60.0
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Convenience decorator for circuit breaker protection."""
    manager = ResilienceManager()
    circuit_breaker = manager.get_circuit_breaker(
        name, failure_threshold, reset_timeout
    )
    
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await circuit_breaker.call(func, *args, **kwargs)
        return wrapper
    
    return decorator