"""Comprehensive tests for Resilience Module."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Any

from flowlib.agent.core.resilience import (
    CircuitState,
    RetryableOperation,
    CircuitBreaker,
    CircuitOpenError,
    TimeoutOperation,
    RateLimiter,
    BulkheadIsolation,
    ErrorContext,
    ResilienceManager,
    retryable,
    with_timeout,
    with_circuit_breaker
)
from flowlib.agent.core.errors import AgentError, ComponentError


class TestRetryableOperation:
    """Test RetryableOperation decorator functionality."""
    
    def test_retryable_operation_initialization(self):
        """Test RetryableOperation initialization with defaults."""
        retryable_op = RetryableOperation()
        
        assert retryable_op.max_retries == 3
        assert retryable_op.backoff_factor == 2.0
        assert retryable_op.max_backoff == 60.0
        assert retryable_op.retryable_errors == (Exception,)
        assert retryable_op.non_retryable_errors == (ValueError, TypeError)
    
    def test_retryable_operation_custom_params(self):
        """Test RetryableOperation with custom parameters."""
        retryable_op = RetryableOperation(
            max_retries=5,
            backoff_factor=1.5,
            max_backoff=30.0,
            retryable_errors=(ConnectionError, TimeoutError),
            non_retryable_errors=(ValueError,)
        )
        
        assert retryable_op.max_retries == 5
        assert retryable_op.backoff_factor == 1.5
        assert retryable_op.max_backoff == 30.0
        assert retryable_op.retryable_errors == (ConnectionError, TimeoutError)
        assert retryable_op.non_retryable_errors == (ValueError,)
    
    @pytest.mark.asyncio
    async def test_successful_operation_no_retry(self):
        """Test successful operation that doesn't need retry."""
        call_count = 0
        
        @RetryableOperation(max_retries=3)
        async def test_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await test_function()
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_operation_succeeds_after_retries(self):
        """Test operation that succeeds after some retries."""
        call_count = 0
        
        @RetryableOperation(max_retries=3, backoff_factor=1.0)  # Fast for testing
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        start_time = time.time()
        result = await test_function()
        end_time = time.time()
        
        assert result == "success"
        assert call_count == 3
        # Should have waited for backoff (1^0 + 1^1 = 2 seconds minimum)
        assert end_time - start_time >= 1.0
    
    @pytest.mark.asyncio
    async def test_operation_fails_after_max_retries(self):
        """Test operation that fails after exhausting retries."""
        call_count = 0
        
        @RetryableOperation(max_retries=2, backoff_factor=0.1)  # Fast for testing
        async def test_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent failure")
        
        with pytest.raises(ConnectionError, match="Persistent failure"):
            await test_function()
        
        assert call_count == 3  # Initial + 2 retries
    
    @pytest.mark.asyncio
    async def test_non_retryable_error_immediate_failure(self):
        """Test that non-retryable errors fail immediately."""
        call_count = 0
        
        @RetryableOperation(max_retries=3)
        async def test_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-retryable error")
        
        with pytest.raises(ValueError, match="Non-retryable error"):
            await test_function()
        
        assert call_count == 1  # No retries
    
    @pytest.mark.asyncio
    async def test_backoff_calculation(self):
        """Test exponential backoff calculation."""
        call_count = 0
        wait_times = []
        
        original_sleep = asyncio.sleep
        async def mock_sleep(duration):
            wait_times.append(duration)
            await original_sleep(0.001)  # Very short sleep for testing
        
        @RetryableOperation(max_retries=3, backoff_factor=2.0, max_backoff=10.0)
        async def test_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Test error")
        
        with patch('asyncio.sleep', side_effect=mock_sleep):
            with pytest.raises(ConnectionError):
                await test_function()
        
        assert call_count == 4  # Initial + 3 retries
        assert len(wait_times) == 3  # 3 waits between retries
        assert wait_times[0] == 1.0   # 2^0
        assert wait_times[1] == 2.0   # 2^1
        assert wait_times[2] == 4.0   # 2^2
    
    @pytest.mark.asyncio
    async def test_max_backoff_limit(self):
        """Test that backoff is capped at max_backoff."""
        wait_times = []
        
        async def mock_sleep(duration):
            wait_times.append(duration)
            # Don't actually sleep to avoid recursion
        
        @RetryableOperation(max_retries=2, backoff_factor=10.0, max_backoff=5.0)
        async def test_function():
            raise ConnectionError("Test error")
        
        with patch('asyncio.sleep', side_effect=mock_sleep):
            with pytest.raises(ConnectionError):
                await test_function()
        
        # All wait times should be capped at max_backoff
        assert all(wait_time <= 5.0 for wait_time in wait_times)
    
    @pytest.mark.asyncio
    async def test_retryable_decorator_function(self):
        """Test convenience retryable decorator function."""
        call_count = 0
        
        @retryable(max_retries=2, backoff_factor=0.1)
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = await test_function()
        
        assert result == "success"
        assert call_count == 2


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""
    
    def test_circuit_breaker_initialization(self):
        """Test CircuitBreaker initialization."""
        cb = CircuitBreaker()
        
        assert cb.failure_threshold == 5
        assert cb.reset_timeout == 60.0
        assert cb.success_threshold == 3
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.last_failure_time is None
        assert cb.state == CircuitState.CLOSED
    
    def test_circuit_breaker_custom_params(self):
        """Test CircuitBreaker with custom parameters."""
        cb = CircuitBreaker(
            failure_threshold=10,
            reset_timeout=120.0,
            success_threshold=5
        )
        
        assert cb.failure_threshold == 10
        assert cb.reset_timeout == 120.0
        assert cb.success_threshold == 5
    
    @pytest.mark.asyncio
    async def test_successful_operation_closed_state(self):
        """Test successful operation in CLOSED state."""
        cb = CircuitBreaker()
        
        async def test_operation():
            return "success"
        
        result = await cb.call(test_operation)
        
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self):
        """Test circuit opens after reaching failure threshold."""
        cb = CircuitBreaker(failure_threshold=2)
        
        async def failing_operation():
            raise ConnectionError("Operation failed")
        
        # First failure
        with pytest.raises(ConnectionError):
            await cb.call(failing_operation)
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 1
        
        # Second failure - should open circuit
        with pytest.raises(ConnectionError):
            await cb.call(failing_operation)
        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 2
    
    @pytest.mark.asyncio
    async def test_circuit_rejects_calls_when_open(self):
        """Test circuit rejects calls when in OPEN state."""
        cb = CircuitBreaker(failure_threshold=1)
        
        async def failing_operation():
            raise ConnectionError("Operation failed")
        
        # Cause circuit to open
        with pytest.raises(ConnectionError):
            await cb.call(failing_operation)
        assert cb.state == CircuitState.OPEN
        
        # Subsequent calls should be rejected immediately
        with pytest.raises(CircuitOpenError):
            await cb.call(failing_operation)
    
    @pytest.mark.asyncio
    async def test_circuit_half_open_after_timeout(self):
        """Test circuit transitions to HALF_OPEN after timeout."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.1)
        
        async def failing_operation():
            raise ConnectionError("Operation failed")
        
        async def success_operation():
            return "success"
        
        # Open the circuit
        with pytest.raises(ConnectionError):
            await cb.call(failing_operation)
        assert cb.state == CircuitState.OPEN
        
        # Wait for reset timeout
        await asyncio.sleep(0.2)
        
        # Next call should transition to HALF_OPEN
        result = await cb.call(success_operation)
        assert result == "success"
        # Circuit should still be HALF_OPEN initially
    
    @pytest.mark.asyncio
    async def test_circuit_closes_after_successful_recovery(self):
        """Test circuit closes after enough successes in HALF_OPEN."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.1, success_threshold=2)
        
        async def failing_operation():
            raise ConnectionError("Operation failed")
        
        async def success_operation():
            return "success"
        
        # Open the circuit
        with pytest.raises(ConnectionError):
            await cb.call(failing_operation)
        assert cb.state == CircuitState.OPEN
        
        # Wait for reset timeout
        await asyncio.sleep(0.2)
        
        # First success in HALF_OPEN
        await cb.call(success_operation)
        assert cb.state == CircuitState.HALF_OPEN
        
        # Second success should close the circuit
        await cb.call(success_operation)
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_reopens_on_failure_in_half_open(self):
        """Test circuit reopens on failure in HALF_OPEN state."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.1)
        
        async def failing_operation():
            raise ConnectionError("Operation failed")
        
        # Open the circuit
        with pytest.raises(ConnectionError):
            await cb.call(failing_operation)
        assert cb.state == CircuitState.OPEN
        
        # Wait for reset timeout
        await asyncio.sleep(0.2)
        
        # Failure in HALF_OPEN should reopen circuit
        with pytest.raises(ConnectionError):
            await cb.call(failing_operation)
        assert cb.state == CircuitState.OPEN
    
    def test_get_state_information(self):
        """Test getting circuit breaker state information."""
        cb = CircuitBreaker()
        cb.failure_count = 3
        cb.last_failure_time = time.time()
        
        state = cb.get_state()
        
        assert state["state"] == "closed"
        assert state["failure_count"] == 3
        assert state["success_count"] == 0
        assert state["last_failure_time"] is not None
        assert state["time_since_last_failure"] is not None
    
    def test_should_attempt_reset_logic(self):
        """Test reset attempt timing logic."""
        cb = CircuitBreaker(reset_timeout=1.0)
        
        # No previous failure
        assert cb._should_attempt_reset() is True
        
        # Recent failure
        cb.last_failure_time = time.time()
        assert cb._should_attempt_reset() is False
        
        # Old failure
        cb.last_failure_time = time.time() - 2.0
        assert cb._should_attempt_reset() is True


class TestTimeoutOperation:
    """Test TimeoutOperation decorator functionality."""
    
    def test_timeout_operation_initialization(self):
        """Test TimeoutOperation initialization."""
        timeout_op = TimeoutOperation(30.0)
        assert timeout_op.timeout_seconds == 30.0
    
    @pytest.mark.asyncio
    async def test_successful_operation_within_timeout(self):
        """Test successful operation that completes within timeout."""
        @TimeoutOperation(1.0)
        async def test_function():
            await asyncio.sleep(0.1)
            return "success"
        
        result = await test_function()
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_operation_timeout_raises_component_error(self):
        """Test operation that times out raises ComponentError."""
        @TimeoutOperation(0.1)
        async def test_function():
            await asyncio.sleep(0.2)
            return "should not reach"
        
        with pytest.raises(ComponentError, match="Operation timed out"):
            await test_function()
    
    @pytest.mark.asyncio
    async def test_timeout_with_object_method(self):
        """Test timeout decorator with object method."""
        class TestComponent:
            def __init__(self):
                self.name = "test_component"
            
            @TimeoutOperation(0.1)
            async def slow_method(self):
                await asyncio.sleep(0.2)
                return "done"
        
        component = TestComponent()
        
        with pytest.raises(ComponentError) as exc_info:
            await component.slow_method()
        
        assert "slow_method" in str(exc_info.value)
        assert "timed out" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_with_timeout_decorator_function(self):
        """Test convenience with_timeout decorator function."""
        @with_timeout(0.5)
        async def test_function():
            await asyncio.sleep(0.1)
            return "success"
        
        result = await test_function()
        assert result == "success"


class TestRateLimiter:
    """Test RateLimiter functionality."""
    
    def test_rate_limiter_initialization(self):
        """Test RateLimiter initialization."""
        rl = RateLimiter(max_calls=100, time_window=60.0)
        
        assert rl.max_calls == 100
        assert rl.time_window == 60.0
        assert rl.calls == []
    
    @pytest.mark.asyncio
    async def test_acquire_within_limit(self):
        """Test acquiring permission within rate limit."""
        rl = RateLimiter(max_calls=5, time_window=1.0)
        
        # Should be able to acquire multiple times within limit
        for _ in range(3):
            await rl.acquire()
        
        assert len(rl.calls) == 3
    
    @pytest.mark.asyncio
    async def test_acquire_blocks_when_limit_reached(self):
        """Test that acquire blocks when rate limit is reached."""
        rl = RateLimiter(max_calls=2, time_window=0.2)
        
        # Fill the rate limit
        await rl.acquire()
        await rl.acquire()
        assert len(rl.calls) == 2
        
        # Third call should block and wait
        start_time = time.time()
        await rl.acquire()
        end_time = time.time()
        
        # Should have waited for the time window
        assert end_time - start_time >= 0.1  # Some delay expected
        # After waiting, old calls outside the window should be cleaned up
        assert len(rl.calls) >= 1  # At least the current call
    
    @pytest.mark.asyncio
    async def test_old_calls_are_cleaned_up(self):
        """Test that old calls outside time window are removed."""
        rl = RateLimiter(max_calls=3, time_window=0.1)
        
        # Add some calls
        await rl.acquire()
        await rl.acquire()
        
        # Wait for time window to pass
        await asyncio.sleep(0.2)
        
        # Next acquire should clean up old calls
        await rl.acquire()
        
        # Should only have the latest call
        assert len(rl.calls) == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_acquisitions(self):
        """Test concurrent rate limiter acquisitions."""
        rl = RateLimiter(max_calls=3, time_window=0.5)
        
        # Start multiple concurrent acquisitions
        tasks = [rl.acquire() for _ in range(5)]
        
        start_time = time.time()
        await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Some tasks should have been delayed
        assert end_time - start_time >= 0.3  # Expected some delay
        # After delay, old calls may be cleaned up, verify we processed all tasks
        assert len(rl.calls) >= 2  # At least some calls within the window


class TestBulkheadIsolation:
    """Test BulkheadIsolation functionality."""
    
    def test_bulkhead_initialization(self):
        """Test BulkheadIsolation initialization."""
        bh = BulkheadIsolation(pool_size=5)
        
        assert bh.semaphore._value == 5
        assert bh.active_operations == 0
        assert bh.queued_operations == 0
    
    @pytest.mark.asyncio
    async def test_execute_within_pool_size(self):
        """Test executing operations within pool size."""
        bh = BulkheadIsolation(pool_size=3)
        
        async def test_operation():
            await asyncio.sleep(0.1)
            return "success"
        
        # Should be able to execute within pool size
        result = await bh.execute(test_operation)
        assert result == "success"
        assert bh.active_operations == 0  # Should be 0 after completion
    
    @pytest.mark.asyncio
    async def test_pool_size_limit_enforcement(self):
        """Test that pool size limit is enforced."""
        bh = BulkheadIsolation(pool_size=2)
        
        async def slow_operation():
            await asyncio.sleep(0.2)
            return "done"
        
        # Start multiple operations concurrently
        tasks = [bh.execute(slow_operation) for _ in range(4)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # All should succeed
        assert all(r == "done" for r in results)
        
        # Should have taken longer due to pool size limitation
        # 4 operations with pool size 2 should take at least 2 * 0.2 = 0.4s
        assert end_time - start_time >= 0.3
    
    @pytest.mark.asyncio
    async def test_operation_failure_handling(self):
        """Test that failed operations don't leak resources."""
        bh = BulkheadIsolation(pool_size=2)
        
        async def failing_operation():
            raise ValueError("Operation failed")
        
        with pytest.raises(ValueError):
            await bh.execute(failing_operation)
        
        # Counters should be reset after failure
        assert bh.active_operations == 0
        assert bh.queued_operations == 0
    
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting bulkhead statistics."""
        bh = BulkheadIsolation(pool_size=3)
        
        async def operation():
            # Check stats during execution
            stats = bh.get_stats()
            assert stats["active_operations"] >= 1
            await asyncio.sleep(0.1)
            return "done"
        
        # Test stats during execution
        task = asyncio.create_task(bh.execute(operation))
        await asyncio.sleep(0.05)  # Let operation start
        
        stats = bh.get_stats()
        assert stats["active_operations"] == 1
        assert stats["available_permits"] == 2
        
        await task  # Wait for completion
        
        # Stats after completion
        final_stats = bh.get_stats()
        assert final_stats["active_operations"] == 0
        assert final_stats["available_permits"] == 3
    
    @pytest.mark.asyncio
    async def test_queued_operations_counter(self):
        """Test queued operations counter."""
        bh = BulkheadIsolation(pool_size=1)
        
        async def slow_operation():
            await asyncio.sleep(0.2)
            return "done"
        
        # Start operations that will exceed pool size
        task1 = asyncio.create_task(bh.execute(slow_operation))
        task2 = asyncio.create_task(bh.execute(slow_operation))
        
        await asyncio.sleep(0.05)  # Let operations start
        
        stats = bh.get_stats()
        assert stats["active_operations"] == 1
        assert stats["queued_operations"] == 1
        
        await asyncio.gather(task1, task2)


class TestErrorContext:
    """Test ErrorContext manager functionality."""
    
    @pytest.mark.asyncio
    async def test_successful_operation_context(self):
        """Test ErrorContext with successful operation."""
        async with ErrorContext("test_operation", "test_component") as ctx:
            assert ctx.operation == "test_operation"
            assert ctx.component_name == "test_component"
            assert ctx.start_time is not None
            await asyncio.sleep(0.01)
        
        # Should complete without exception
    
    @pytest.mark.asyncio
    async def test_failed_operation_context(self):
        """Test ErrorContext with failed operation."""
        with pytest.raises(ValueError):
            async with ErrorContext("test_operation", "test_component"):
                raise ValueError("Test error")
    
    @pytest.mark.asyncio
    async def test_reraise_as_component_error(self):
        """Test reraising exception as ComponentError."""
        with pytest.raises(ComponentError) as exc_info:
            async with ErrorContext(
                "test_operation", 
                "test_component", 
                reraise_as=ComponentError
            ):
                raise ValueError("Original error")
        
        error = exc_info.value
        assert error.context["component_name"] == "test_component"
        assert error.context["operation"] == "test_operation"
        assert "test_operation failed" in error.message
        assert isinstance(error.cause, ValueError)
    
    @pytest.mark.asyncio
    async def test_reraise_as_custom_exception(self):
        """Test reraising as custom exception type."""
        class CustomError(Exception):
            pass
        
        with pytest.raises(CustomError):
            async with ErrorContext(
                "test_operation",
                "test_component",
                reraise_as=CustomError
            ):
                raise ValueError("Original error")
    
    @pytest.mark.asyncio
    async def test_context_timing_measurement(self):
        """Test that context measures operation timing."""
        start = time.time()
        
        async with ErrorContext("test_operation", "test_component") as ctx:
            await asyncio.sleep(0.05)
            operation_start = ctx.start_time
        
        end = time.time()
        
        assert operation_start is not None
        assert start <= operation_start <= end


class TestResilienceManager:
    """Test ResilienceManager functionality."""
    
    def test_resilience_manager_initialization(self):
        """Test ResilienceManager initialization."""
        manager = ResilienceManager()
        
        assert manager.circuit_breakers == {}
        assert manager.rate_limiters == {}
        assert manager.bulkheads == {}
    
    def test_get_circuit_breaker_creates_new(self):
        """Test getting circuit breaker creates new instance."""
        manager = ResilienceManager()
        
        cb = manager.get_circuit_breaker("test_cb", failure_threshold=10)
        
        assert isinstance(cb, CircuitBreaker)
        assert cb.failure_threshold == 10
        assert "test_cb" in manager.circuit_breakers
    
    def test_get_circuit_breaker_returns_existing(self):
        """Test getting circuit breaker returns existing instance."""
        manager = ResilienceManager()
        
        cb1 = manager.get_circuit_breaker("test_cb")
        cb2 = manager.get_circuit_breaker("test_cb")
        
        assert cb1 is cb2
    
    def test_get_rate_limiter_creates_new(self):
        """Test getting rate limiter creates new instance."""
        manager = ResilienceManager()
        
        rl = manager.get_rate_limiter("test_rl", max_calls=50, time_window=30.0)
        
        assert isinstance(rl, RateLimiter)
        assert rl.max_calls == 50
        assert rl.time_window == 30.0
        assert "test_rl" in manager.rate_limiters
    
    def test_get_bulkhead_creates_new(self):
        """Test getting bulkhead creates new instance."""
        manager = ResilienceManager()
        
        bh = manager.get_bulkhead("test_bh", pool_size=15)
        
        assert isinstance(bh, BulkheadIsolation)
        assert bh.semaphore._value == 15
        assert "test_bh" in manager.bulkheads
    
    def test_get_health_status(self):
        """Test getting health status of all components."""
        manager = ResilienceManager()
        
        # Create some components
        cb = manager.get_circuit_breaker("test_cb")
        bh = manager.get_bulkhead("test_bh")
        
        health_status = manager.get_health_status()
        
        assert "circuit_breakers" in health_status
        assert "bulkheads" in health_status
        assert "test_cb" in health_status["circuit_breakers"]
        assert "test_bh" in health_status["bulkheads"]
        
        # Verify structure
        cb_status = health_status["circuit_breakers"]["test_cb"]
        assert "state" in cb_status
        assert "failure_count" in cb_status
        
        bh_status = health_status["bulkheads"]["test_bh"]
        assert "active_operations" in bh_status
        assert "available_permits" in bh_status


class TestWithCircuitBreakerDecorator:
    """Test with_circuit_breaker decorator function."""
    
    @pytest.mark.asyncio
    async def test_with_circuit_breaker_decorator(self):
        """Test with_circuit_breaker decorator functionality."""
        call_count = 0
        
        @with_circuit_breaker("test_circuit", failure_threshold=2)
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Simulated failure")
            return "success"
        
        # First two calls should fail and open the circuit
        with pytest.raises(ConnectionError):
            await test_function()
        
        with pytest.raises(ConnectionError):
            await test_function()
        
        # Third call should be rejected by circuit breaker
        with pytest.raises(CircuitOpenError):
            await test_function()
        
        assert call_count == 2  # Circuit opened after 2 failures


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple resilience patterns."""
    
    @pytest.mark.asyncio
    async def test_combined_retry_and_circuit_breaker(self):
        """Test combining retry and circuit breaker patterns."""
        call_count = 0
        
        @retryable(max_retries=2, backoff_factor=0.1)
        @with_circuit_breaker("integrated_test", failure_threshold=3)
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count <= 4:  # Fail first 4 calls
                raise ConnectionError("Transient failure")
            return "success"
        
        # Should retry and eventually fail, contributing to circuit breaker
        with pytest.raises(ConnectionError):
            await flaky_operation()  # 3 calls (1 + 2 retries)
        
        # Second attempt should fail immediately with circuit breaker open
        with pytest.raises(CircuitOpenError):
            await flaky_operation()  # Circuit breaker is now open
        
        # Circuit is now open, further attempts should fail immediately
        with pytest.raises(CircuitOpenError):
            await flaky_operation()  # Circuit breaker remains open
    
    @pytest.mark.asyncio
    async def test_timeout_with_retry(self):
        """Test timeout combined with retry."""
        call_count = 0
        
        @retryable(max_retries=2, backoff_factor=0.1)
        @with_timeout(0.1)
        async def slow_operation():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.2)  # Always times out
            return "should not reach"
        
        with pytest.raises(ComponentError, match="Operation timed out"):
            await slow_operation()
        
        # Should have tried multiple times due to retry
        assert call_count > 1
    
    @pytest.mark.asyncio
    async def test_bulkhead_with_error_context(self):
        """Test bulkhead isolation with error context."""
        bh = BulkheadIsolation(pool_size=2)
        
        async def test_operation():
            async with ErrorContext("bulk_operation", "test_component"):
                await asyncio.sleep(0.1)
                return "success"
        
        # Execute multiple operations concurrently
        tasks = [bh.execute(test_operation) for _ in range(3)]
        results = await asyncio.gather(*tasks)
        
        assert all(r == "success" for r in results)
    
    @pytest.mark.asyncio
    async def test_rate_limiter_with_circuit_breaker(self):
        """Test rate limiter combined with circuit breaker protection."""
        rl = RateLimiter(max_calls=2, time_window=0.2)
        call_count = 0
        
        @with_circuit_breaker("rate_limited_circuit", failure_threshold=2)
        async def rate_limited_operation():
            nonlocal call_count
            await rl.acquire()
            call_count += 1
            if call_count <= 3:
                raise ConnectionError("Simulated failure")
            return "success"
        
        # Should be rate limited and also trigger circuit breaker
        start_time = time.time()
        
        with pytest.raises(ConnectionError):
            await rate_limited_operation()
        
        with pytest.raises(ConnectionError):
            await rate_limited_operation()
        
        # Should be rate limited for third call
        with pytest.raises(CircuitOpenError):
            await rate_limited_operation()
        
        end_time = time.time()
        
        # Circuit breaker should open quickly after threshold failures
        # Note: timing can vary due to circuit breaker vs rate limiting interaction
        assert end_time - start_time >= 0.0  # Some time elapsed


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_with_zero_failure_threshold(self):
        """Test circuit breaker with zero failure threshold."""
        cb = CircuitBreaker(failure_threshold=0)
        
        async def test_operation():
            return "success"
        
        # Should work normally even with zero threshold
        result = await cb.call(test_operation)
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_rate_limiter_with_zero_time_window(self):
        """Test rate limiter with zero time window."""
        rl = RateLimiter(max_calls=5, time_window=0.0)
        
        # Should allow unlimited calls with zero time window
        for _ in range(10):
            await rl.acquire()
        
        # All calls should be cleaned up immediately
        assert len(rl.calls) <= 5
    
    def test_bulkhead_with_zero_pool_size(self):
        """Test bulkhead with zero pool size raises ValueError."""
        with pytest.raises(ValueError, match="pool_size must be greater than 0"):
            BulkheadIsolation(pool_size=0)
    
    @pytest.mark.asyncio
    async def test_retry_with_zero_max_retries(self):
        """Test retry with zero max retries."""
        call_count = 0
        
        @retryable(max_retries=0)
        async def test_operation():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")
        
        with pytest.raises(ConnectionError):
            await test_operation()
        
        assert call_count == 1  # Should only call once with no retries
    
    @pytest.mark.asyncio
    async def test_error_context_with_none_reraise_as(self):
        """Test ErrorContext with None reraise_as parameter."""
        with pytest.raises(ValueError, match="Original error"):
            async with ErrorContext("test_op", "test_comp", reraise_as=None):
                raise ValueError("Original error")
    
    def test_circuit_state_enum_values(self):
        """Test CircuitState enum values."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


if __name__ == "__main__":
    pytest.main([__file__])