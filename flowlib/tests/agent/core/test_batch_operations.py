"""Comprehensive tests for Batch Operations module."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any, Tuple

from flowlib.agent.core.batch_operations import (
    BatchStatus,
    BatchResult,
    BatchFailedItem,
    MemoryStoreRequest,
    MemoryRetrieveRequest,
    FlowExecutionRequest,
    BatchMemoryOperations,
    BatchFlowOperations,
    BatchAnalyticsOperations,
    BatchOperationManager
)
from flowlib.agent.core.errors import ComponentError
from flowlib.agent.core.performance import PerformanceMonitor


class MockMemoryManager:
    """Mock memory manager for testing."""
    
    def __init__(self, should_fail: bool = False, fail_keys: List[str] = None):
        self.storage = {}
        self.should_fail = should_fail
        self.fail_keys = fail_keys or []
        self.store_calls = []
        self.retrieve_calls = []
        self.search_calls = []
    
    async def store(self, key: str, value: Any):
        """Mock store operation."""
        self.store_calls.append((key, value))
        if self.should_fail or key in self.fail_keys:
            raise Exception(f"Failed to store {key}")
        self.storage[key] = value
    
    async def store_with_metadata(self, key: str, value: Any, metadata: Dict[str, Any]):
        """Mock store with metadata operation."""
        self.store_calls.append((key, value, metadata))
        if self.should_fail or key in self.fail_keys:
            raise Exception(f"Failed to store {key}")
        self.storage[key] = {"value": value, "metadata": metadata}
    
    async def retrieve(self, key: str):
        """Mock retrieve operation."""
        self.retrieve_calls.append(key)
        if self.should_fail or key in self.fail_keys:
            raise Exception(f"Failed to retrieve {key}")
        return self.storage.get(key)
    
    async def search(self, query: str):
        """Mock search operation."""
        self.search_calls.append(query)
        if self.should_fail or query in self.fail_keys:
            raise Exception(f"Failed to search {query}")
        return f"search_result_for_{query}"
    
    async def search_all_memories(self, query: str, memory_types: List[str] = None):
        """Mock advanced search operation."""
        self.search_calls.append((query, memory_types))
        if self.should_fail or query in self.fail_keys:
            raise Exception(f"Failed to search {query}")
        return f"advanced_search_result_for_{query}"


class MockFlowManager:
    """Mock flow manager for testing."""
    
    def __init__(self, should_fail: bool = False, fail_flows: List[str] = None):
        self.flows = {}
        self.should_fail = should_fail
        self.fail_flows = fail_flows or []
        self.execution_calls = []
    
    def get_flow(self, flow_name: str):
        """Get a flow by name."""
        if flow_name in self.fail_flows:
            return None
        # Check if we have a specific flow registered first
        if flow_name in self.flows:
            return self.flows[flow_name]
        return MockFlow(flow_name, self.should_fail)
    
    def add_flow(self, name: str, flow):
        """Add a flow to the manager."""
        self.flows[name] = flow


class MockFlow:
    """Mock flow for testing."""
    
    def __init__(self, name: str, should_fail: bool = False):
        self.name = name
        self.should_fail = should_fail
        self.execution_calls = []
    
    async def run_pipeline(self, input_data: Any):
        """Mock pipeline execution."""
        self.execution_calls.append(input_data)
        if self.should_fail:
            raise Exception(f"Flow {self.name} failed")
        return f"result_for_{self.name}_with_{input_data}"


class TestBatchStatus:
    """Test BatchStatus enum."""
    
    def test_batch_status_values(self):
        """Test BatchStatus enum values."""
        assert BatchStatus.PENDING.value == "pending"
        assert BatchStatus.PROCESSING.value == "processing"
        assert BatchStatus.COMPLETED.value == "completed"
        assert BatchStatus.FAILED.value == "failed"
        assert BatchStatus.PARTIAL_SUCCESS.value == "partial_success"


class TestBatchResult:
    """Test BatchResult functionality."""
    
    def test_batch_result_creation(self):
        """Test creating BatchResult."""
        result = BatchResult(
            status=BatchStatus.COMPLETED,
            successful_items=["item1", "item2"],
            failed_items=[],
            total_count=2,
            success_count=2,
            failure_count=0,
            processing_time=1.5
        )
        
        assert result.status == BatchStatus.COMPLETED
        assert result.successful_items == ["item1", "item2"]
        assert result.failed_items == []
        assert result.total_count == 2
        assert result.success_count == 2
        assert result.failure_count == 0
        assert result.processing_time == 1.5
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        # 100% success
        result = BatchResult(
            status=BatchStatus.COMPLETED,
            successful_items=[],
            failed_items=[],
            total_count=10,
            success_count=10,
            failure_count=0,
            processing_time=1.0
        )
        assert result.success_rate == 100.0
        
        # 50% success - create new instance
        result_50 = BatchResult(
            status=BatchStatus.PARTIAL_SUCCESS,
            successful_items=[],
            failed_items=[],
            total_count=10,
            success_count=5,
            failure_count=5,
            processing_time=1.0
        )
        assert result_50.success_rate == 50.0
        
        # 0% success - create new instance
        result_0 = BatchResult(
            status=BatchStatus.FAILED,
            successful_items=[],
            failed_items=[],
            total_count=10,
            success_count=0,
            failure_count=10,
            processing_time=1.0
        )
        assert result_0.success_rate == 0.0
        
        # Edge case: zero total - create new instance
        result_zero = BatchResult(
            status=BatchStatus.COMPLETED,
            successful_items=[],
            failed_items=[],
            total_count=0,
            success_count=0,
            failure_count=0,
            processing_time=1.0
        )
        assert result_zero.success_rate == 0.0


class TestDataClasses:
    """Test data classes."""
    
    def test_memory_store_request(self):
        """Test MemoryStoreRequest creation."""
        request = MemoryStoreRequest.create(
            key="test_key",
            value="test_value",
            metadata={"type": "test"},
            ttl=3600
        )
        
        assert request.key == "test_key"
        assert request.value == "test_value"
        assert request.metadata.data == {"type": "test"}
        assert request.ttl_seconds == 3600
    
    def test_memory_retrieve_request(self):
        """Test MemoryRetrieveRequest creation."""
        request = MemoryRetrieveRequest.create(key="test_key")
        
        assert request.key == "test_key"
    
    def test_flow_execution_request(self):
        """Test FlowExecutionRequest creation."""
        request = FlowExecutionRequest.create(
            flow_name="test_flow",
            input_data={"param": "value"},
            context={"session": "123"}
        )
        
        assert request.flow_name == "test_flow"
        assert request.input_data == {"param": "value"}
        assert request.context.data == {"session": "123"}


class TestBatchMemoryOperations:
    """Test BatchMemoryOperations functionality."""
    
    def test_initialization(self):
        """Test BatchMemoryOperations initialization."""
        memory_manager = MockMemoryManager()
        monitor = PerformanceMonitor()
        
        batch_ops = BatchMemoryOperations(memory_manager, monitor)
        
        assert batch_ops.memory_manager == memory_manager
        assert batch_ops.performance_monitor == monitor
    
    def test_initialization_with_default_monitor(self):
        """Test initialization with default performance monitor."""
        memory_manager = MockMemoryManager()
        
        batch_ops = BatchMemoryOperations(memory_manager)
        
        assert batch_ops.memory_manager == memory_manager
        assert isinstance(batch_ops.performance_monitor, PerformanceMonitor)
    
    @pytest.mark.asyncio
    async def test_batch_store_success(self):
        """Test successful batch store operation."""
        memory_manager = MockMemoryManager()
        batch_ops = BatchMemoryOperations(memory_manager)
        
        requests = [
            MemoryStoreRequest.create("key1", "value1"),
            MemoryStoreRequest.create("key2", "value2", {"type": "test"}),
            MemoryStoreRequest.create("key3", "value3", ttl=3600)
        ]
        
        result = await batch_ops.batch_store(requests, chunk_size=2)
        
        assert result.status == BatchStatus.COMPLETED
        assert result.success_count == 3
        assert result.failure_count == 0
        assert result.total_count == 3
        assert result.success_rate == 100.0
        assert result.processing_time > 0
        assert len(result.successful_items) == 3
        assert len(result.failed_items) == 0
        
        # Verify storage calls
        assert len(memory_manager.store_calls) == 3
    
    @pytest.mark.asyncio
    async def test_batch_store_partial_failure(self):
        """Test batch store with partial failures."""
        memory_manager = MockMemoryManager(fail_keys=["key2"])
        batch_ops = BatchMemoryOperations(memory_manager)
        
        requests = [
            MemoryStoreRequest.create("key1", "value1"),
            MemoryStoreRequest.create("key2", "value2"),  # This will fail
            MemoryStoreRequest.create("key3", "value3")
        ]
        
        result = await batch_ops.batch_store(requests)
        
        assert result.status == BatchStatus.PARTIAL_SUCCESS
        assert result.success_count == 2
        assert result.failure_count == 1
        assert result.total_count == 3
        assert abs(result.success_rate - (2/3 * 100)) < 0.01  # ~66.67%
        assert len(result.successful_items) == 2
        assert len(result.failed_items) == 1
        
        # Check failed item (using BatchFailedItem model)
        failed_item = result.failed_items[0]
        assert isinstance(failed_item, BatchFailedItem)
        assert failed_item.item.key == "key2"
        assert "Failed to store key2" in failed_item.error_message
    
    @pytest.mark.asyncio
    async def test_batch_store_complete_failure(self):
        """Test batch store with complete failure."""
        memory_manager = MockMemoryManager(should_fail=True)
        batch_ops = BatchMemoryOperations(memory_manager)
        
        requests = [
            MemoryStoreRequest.create("key1", "value1"),
            MemoryStoreRequest.create("key2", "value2")
        ]
        
        result = await batch_ops.batch_store(requests)
        
        assert result.status == BatchStatus.FAILED
        assert result.success_count == 0
        assert result.failure_count == 2
        assert result.total_count == 2
        assert result.success_rate == 0.0
    
    @pytest.mark.asyncio
    async def test_batch_retrieve_success(self):
        """Test successful batch retrieve operation."""
        memory_manager = MockMemoryManager()
        # Pre-populate storage
        memory_manager.storage = {"key1": "value1", "key2": "value2"}
        
        batch_ops = BatchMemoryOperations(memory_manager)
        
        requests = [
            MemoryRetrieveRequest.create("key1"),
            MemoryRetrieveRequest.create("key2"),
            MemoryRetrieveRequest.create("key3")
        ]
        
        result = await batch_ops.batch_retrieve(requests)
        
        assert result.status == BatchStatus.COMPLETED
        assert result.success_count == 3
        assert result.failure_count == 0
        assert result.total_count == 3
        
        # Check retrieved values
        successful_items = result.successful_items
        assert len(successful_items) == 3
        
        # Find specific items (order might vary due to async processing)
        key_values = {key: value for key, value in successful_items}
        assert key_values["key1"] == "value1"
        assert key_values["key2"] == "value2"
        assert key_values["key3"] is None  # Missing key returns None (no fallback)
    
    @pytest.mark.asyncio
    async def test_batch_retrieve_with_failures(self):
        """Test batch retrieve with some failures."""
        memory_manager = MockMemoryManager(fail_keys=["key2"])
        memory_manager.storage = {"key1": "value1", "key3": "value3"}
        
        batch_ops = BatchMemoryOperations(memory_manager)
        
        requests = [
            MemoryRetrieveRequest.create("key1"),
            MemoryRetrieveRequest.create("key2"),  # This will fail
            MemoryRetrieveRequest.create("key3")
        ]
        
        result = await batch_ops.batch_retrieve(requests)
        
        assert result.status == BatchStatus.PARTIAL_SUCCESS
        assert result.success_count == 2
        assert result.failure_count == 1
        assert len(result.failed_items) == 1
    
    @pytest.mark.asyncio
    async def test_batch_search_success(self):
        """Test successful batch search operation."""
        memory_manager = MockMemoryManager()
        batch_ops = BatchMemoryOperations(memory_manager)
        
        queries = ["query1", "query2", "query3"]
        
        result = await batch_ops.batch_search(queries, max_concurrent=2)
        
        assert result.status == BatchStatus.COMPLETED
        assert result.success_count == 3
        assert result.failure_count == 0
        assert len(result.successful_items) == 3
        
        # Verify search calls
        assert len(memory_manager.search_calls) == 3
    
    @pytest.mark.asyncio
    async def test_batch_search_with_advanced_search(self):
        """Test batch search with advanced search method."""
        memory_manager = MockMemoryManager()
        memory_manager.search_all_memories = AsyncMock(
            side_effect=lambda q, mt: f"advanced_search_result_for_{q}"
        )
        
        batch_ops = BatchMemoryOperations(memory_manager)
        
        queries = ["query1", "query2"]
        memory_types = ["working", "vector"]
        
        result = await batch_ops.batch_search(queries, memory_types)
        
        assert result.status == BatchStatus.COMPLETED
        assert result.success_count == 2
        
        # Verify advanced search was called
        assert memory_manager.search_all_memories.call_count == 2
    
    @pytest.mark.asyncio
    async def test_batch_search_with_failures(self):
        """Test batch search with some failures."""
        memory_manager = MockMemoryManager(fail_keys=["query2"])
        batch_ops = BatchMemoryOperations(memory_manager)
        
        queries = ["query1", "query2", "query3"]
        
        result = await batch_ops.batch_search(queries)
        
        assert result.status == BatchStatus.PARTIAL_SUCCESS
        assert result.success_count == 2
        assert result.failure_count == 1
    
    @pytest.mark.asyncio
    async def test_store_with_strict_contract_validation(self):
        """Test store operation with strict contract validation (no fallbacks)."""
        # Create a memory manager without store_with_metadata
        class BasicMemoryManager:
            def __init__(self):
                self.storage = {}
                self.store_calls = []
            
            async def store(self, key: str, value: Any):
                """Mock store operation."""
                self.store_calls.append((key, value))
                self.storage[key] = value
        
        memory_manager = BasicMemoryManager()
        batch_ops = BatchMemoryOperations(memory_manager)
        
        requests = [MemoryStoreRequest.create("key1", "value1", {"type": "test"})]
        
        result = await batch_ops.batch_store(requests)
        
        # Should fail due to strict contract validation (no fallbacks)
        assert result.status == BatchStatus.FAILED
        assert result.success_count == 0
        assert result.failure_count == 1
        
        # Verify the error message indicates missing required method
        failed_item = result.failed_items[0]
        assert "missing required 'store_with_metadata' method" in failed_item.error_message
        assert failed_item.error_type == "ComponentError"
    
    @pytest.mark.asyncio
    async def test_chunking_behavior(self):
        """Test that operations are properly chunked."""
        memory_manager = MockMemoryManager()
        batch_ops = BatchMemoryOperations(memory_manager)
        
        # Create more requests than chunk size
        requests = [MemoryStoreRequest.create(f"key{i}", f"value{i}") for i in range(10)]
        
        result = await batch_ops.batch_store(requests, chunk_size=3)
        
        assert result.status == BatchStatus.COMPLETED
        assert result.success_count == 10
        assert len(memory_manager.store_calls) == 10


class TestBatchFlowOperations:
    """Test BatchFlowOperations functionality."""
    
    def test_initialization(self):
        """Test BatchFlowOperations initialization."""
        flow_manager = MockFlowManager()
        monitor = PerformanceMonitor()
        
        batch_ops = BatchFlowOperations(flow_manager, monitor)
        
        assert batch_ops.flow_manager == flow_manager
        assert batch_ops.performance_monitor == monitor
    
    @pytest.mark.asyncio
    async def test_batch_execute_flows_success(self):
        """Test successful batch flow execution."""
        flow_manager = MockFlowManager()
        batch_ops = BatchFlowOperations(flow_manager)
        
        requests = [
            FlowExecutionRequest.create("flow1", "input1"),
            FlowExecutionRequest.create("flow2", "input2"),
            FlowExecutionRequest.create("flow3", "input3")
        ]
        
        result = await batch_ops.batch_execute_flows(requests, max_concurrent=2)
        
        assert result.status == BatchStatus.COMPLETED
        assert result.success_count == 3
        assert result.failure_count == 0
        assert len(result.successful_items) == 3
        
        # Verify all flows were executed
        for req, res in result.successful_items:
            assert req.flow_name in ["flow1", "flow2", "flow3"]
            assert "result_for_" in res
    
    @pytest.mark.asyncio
    async def test_batch_execute_flows_with_failures(self):
        """Test batch flow execution with some failures."""
        flow_manager = MockFlowManager(fail_flows=["flow2"])
        batch_ops = BatchFlowOperations(flow_manager)
        
        requests = [
            FlowExecutionRequest.create("flow1", "input1"),
            FlowExecutionRequest.create("flow2", "input2"),  # This will fail
            FlowExecutionRequest.create("flow3", "input3")
        ]
        
        result = await batch_ops.batch_execute_flows(requests)
        
        assert result.status == BatchStatus.PARTIAL_SUCCESS
        assert result.success_count == 2
        assert result.failure_count == 1
        
        # Check failed item (using BatchFailedItem model)
        failed_item = result.failed_items[0]
        assert isinstance(failed_item, BatchFailedItem)
        assert failed_item.item.flow_name == "flow2"
        assert failed_item.error_type == "ComponentError"
    
    @pytest.mark.asyncio
    async def test_batch_execute_flows_timeout(self):
        """Test batch flow execution with timeout."""
        flow_manager = MockFlowManager()
        batch_ops = BatchFlowOperations(flow_manager)
        
        # Mock a slow flow
        async def slow_flow(input_data):
            await asyncio.sleep(0.5)
            return "slow_result"
        
        flow_manager.flows["slow_flow"] = Mock()
        flow_manager.flows["slow_flow"].run_pipeline = slow_flow
        
        requests = [FlowExecutionRequest.create("slow_flow", "input1")]
        
        result = await batch_ops.batch_execute_flows(
            requests, 
            timeout_per_flow=0.1  # Very short timeout
        )
        
        assert result.status == BatchStatus.FAILED
        assert result.failure_count == 1
        
        # Should be timeout error
        failed_item = result.failed_items[0]
        assert isinstance(failed_item, BatchFailedItem)
        assert failed_item.error_type == "ComponentError"
        assert "timed out" in failed_item.error_message
    
    @pytest.mark.asyncio
    async def test_batch_execute_same_flow(self):
        """Test executing same flow with different inputs."""
        flow_manager = MockFlowManager()
        batch_ops = BatchFlowOperations(flow_manager)
        
        input_list = ["input1", "input2", "input3"]
        
        result = await batch_ops.batch_execute_same_flow(
            "test_flow", 
            input_list, 
            max_concurrent=2
        )
        
        assert result.status == BatchStatus.COMPLETED
        assert result.success_count == 3
        assert len(result.successful_items) == 3
    
    @pytest.mark.asyncio
    async def test_execute_single_flow_strict_contract_validation(self):
        """Test single flow execution with strict contract validation (no fallbacks)."""
        # Create a flow manager without get_flow method
        class BasicFlowManager:
            def __init__(self):
                self.test_flow = MockFlow("test_flow")
        
        flow_manager = BasicFlowManager()
        batch_ops = BatchFlowOperations(flow_manager)
        
        requests = [FlowExecutionRequest.create("test_flow", "input1")]
        
        result = await batch_ops.batch_execute_flows(requests)
        
        # Should fail due to strict contract validation (no fallbacks)
        assert result.status == BatchStatus.FAILED
        assert result.success_count == 0
        assert result.failure_count == 1
        
        # Verify the error indicates flow execution failure (due to missing contract)
        failed_item = result.failed_items[0]
        assert "Flow execution failed: test_flow" in failed_item.error_message
        assert failed_item.error_type == "ComponentError"
    
    @pytest.mark.asyncio
    async def test_execute_single_flow_not_found(self):
        """Test single flow execution with flow not found."""
        # Create a flow manager without the needed flow
        class EmptyFlowManager:
            pass
        
        flow_manager = EmptyFlowManager()
        batch_ops = BatchFlowOperations(flow_manager)
        
        requests = [FlowExecutionRequest.create("nonexistent_flow", "input1")]
        
        result = await batch_ops.batch_execute_flows(requests)
        
        assert result.status == BatchStatus.FAILED
        assert result.failure_count == 1
        
        failed_item = result.failed_items[0]
        assert isinstance(failed_item, BatchFailedItem)
        assert "Flow execution failed" in failed_item.error_message or "Flow not found" in failed_item.error_message
    
    @pytest.mark.asyncio
    async def test_execute_single_flow_not_callable(self):
        """Test single flow execution with non-callable flow."""
        flow_manager = MockFlowManager()
        flow_manager.flows["invalid_flow"] = "not_callable"
        
        batch_ops = BatchFlowOperations(flow_manager)
        
        requests = [FlowExecutionRequest.create("invalid_flow", "input1")]
        
        result = await batch_ops.batch_execute_flows(requests)
        
        assert result.status == BatchStatus.FAILED
        assert result.failure_count == 1
        
        failed_item = result.failed_items[0]
        assert isinstance(failed_item, BatchFailedItem)
        assert "Flow execution failed" in failed_item.error_message or "not executable" in failed_item.error_message


class TestBatchAnalyticsOperations:
    """Test BatchAnalyticsOperations functionality."""
    
    def test_initialization(self):
        """Test BatchAnalyticsOperations initialization."""
        monitor = PerformanceMonitor()
        batch_ops = BatchAnalyticsOperations(monitor)
        
        assert batch_ops.performance_monitor == monitor
    
    @pytest.mark.asyncio
    async def test_batch_analyze_memory_patterns(self):
        """Test batch memory pattern analysis."""
        monitor = PerformanceMonitor()
        batch_ops = BatchAnalyticsOperations(monitor)
        
        memory_keys = ["key1", "key2", "key3"]
        
        async def analysis_func1(keys):
            return {"pattern": "sequential", "keys_analyzed": len(keys)}
        
        async def analysis_func2(keys):
            return {"pattern": "random", "total_keys": len(keys)}
        
        analysis_functions = [analysis_func1, analysis_func2]
        
        results = await batch_ops.batch_analyze_memory_patterns(
            memory_keys, analysis_functions
        )
        
        assert "analysis_func1" in results
        assert "analysis_func2" in results
        assert results["analysis_func1"]["keys_analyzed"] == 3
        assert results["analysis_func2"]["total_keys"] == 3
    
    @pytest.mark.asyncio
    async def test_batch_analyze_memory_patterns_with_failure(self):
        """Test batch memory pattern analysis with failures."""
        monitor = PerformanceMonitor()
        batch_ops = BatchAnalyticsOperations(monitor)
        
        memory_keys = ["key1", "key2"]
        
        async def working_func(keys):
            return {"result": "success"}
        
        async def failing_func(keys):
            raise Exception("Analysis failed")
        
        analysis_functions = [working_func, failing_func]
        
        results = await batch_ops.batch_analyze_memory_patterns(
            memory_keys, analysis_functions
        )
        
        assert "working_func" in results
        assert "failing_func" in results
        assert results["working_func"]["result"] == "success"
        assert "error" in results["failing_func"]
    
    @pytest.mark.asyncio
    async def test_batch_generate_reports(self):
        """Test batch report generation."""
        monitor = PerformanceMonitor()
        # Add some performance data
        monitor.record_operation("test_op", 1.0, success=True)
        
        batch_ops = BatchAnalyticsOperations(monitor)
        
        report_configs = [
            {"name": "performance_report", "type": "performance"},
            {"name": "memory_report", "type": "memory"},
            {"name": "custom_report", "type": "unknown"}
        ]
        
        reports = await batch_ops.batch_generate_reports(report_configs)
        
        assert "performance_report" in reports
        assert "memory_report" in reports
        assert "custom_report" in reports
        
        # Performance report should have operation stats
        assert "test_op" in reports["performance_report"]
        
        # Memory report should have performance stats
        assert isinstance(reports["memory_report"], dict)
        
        # Unknown type should have error
        assert "error" in reports["custom_report"]
    
    @pytest.mark.asyncio
    async def test_generate_single_report_types(self):
        """Test different report type generation."""
        monitor = PerformanceMonitor()
        batch_ops = BatchAnalyticsOperations(monitor)
        
        # Test performance report
        config = {"type": "performance"}
        report = await batch_ops._generate_single_report(config)
        assert isinstance(report, dict)
        
        # Test memory report
        config = {"type": "memory"}
        report = await batch_ops._generate_single_report(config)
        assert isinstance(report, dict)
        
        # Test unknown report type
        config = {"type": "unknown"}
        report = await batch_ops._generate_single_report(config)
        assert "error" in report


class TestBatchOperationManager:
    """Test BatchOperationManager functionality."""
    
    def test_initialization_with_all_managers(self):
        """Test BatchOperationManager initialization with all managers."""
        memory_manager = MockMemoryManager()
        flow_manager = MockFlowManager()
        monitor = PerformanceMonitor()
        
        batch_manager = BatchOperationManager(
            memory_manager=memory_manager,
            flow_manager=flow_manager,
            performance_monitor=monitor
        )
        
        assert batch_manager.performance_monitor == monitor
        assert batch_manager.memory_ops is not None
        assert batch_manager.flow_ops is not None
        assert batch_manager.analytics_ops is not None
    
    def test_initialization_with_partial_managers(self):
        """Test initialization with only some managers."""
        memory_manager = MockMemoryManager()
        
        batch_manager = BatchOperationManager(memory_manager=memory_manager)
        
        assert batch_manager.memory_ops is not None
        assert batch_manager.flow_ops is None
        assert batch_manager.analytics_ops is not None
        assert isinstance(batch_manager.performance_monitor, PerformanceMonitor)
    
    def test_initialization_with_no_managers(self):
        """Test initialization with no managers."""
        batch_manager = BatchOperationManager()
        
        assert batch_manager.memory_ops is None
        assert batch_manager.flow_ops is None
        assert batch_manager.analytics_ops is not None
    
    def test_get_operations_methods(self):
        """Test getting specific operation types."""
        memory_manager = MockMemoryManager()
        flow_manager = MockFlowManager()
        
        batch_manager = BatchOperationManager(
            memory_manager=memory_manager,
            flow_manager=flow_manager
        )
        
        memory_ops = batch_manager.get_memory_operations()
        flow_ops = batch_manager.get_flow_operations()
        analytics_ops = batch_manager.get_analytics_operations()
        
        assert isinstance(memory_ops, BatchMemoryOperations)
        assert isinstance(flow_ops, BatchFlowOperations)
        assert isinstance(analytics_ops, BatchAnalyticsOperations)
    
    def test_get_operations_when_none(self):
        """Test getting operations when managers are None."""
        batch_manager = BatchOperationManager()
        
        assert batch_manager.get_memory_operations() is None
        assert batch_manager.get_flow_operations() is None
        assert batch_manager.get_analytics_operations() is not None
    
    def test_get_performance_stats(self):
        """Test getting performance statistics."""
        monitor = PerformanceMonitor()
        monitor.record_operation("test_op", 1.0, success=True)
        
        batch_manager = BatchOperationManager(performance_monitor=monitor)
        
        stats = batch_manager.get_performance_stats()
        
        assert "test_op" in stats
        assert isinstance(stats, dict)


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple operations."""
    
    @pytest.mark.asyncio
    async def test_memory_and_flow_batch_operations(self):
        """Test combining memory and flow batch operations."""
        memory_manager = MockMemoryManager()
        flow_manager = MockFlowManager()
        
        batch_manager = BatchOperationManager(
            memory_manager=memory_manager,
            flow_manager=flow_manager
        )
        
        # First, store some data
        store_requests = [
            MemoryStoreRequest.create("flow_input_1", "data1"),
            MemoryStoreRequest.create("flow_input_2", "data2")
        ]
        
        memory_result = await batch_manager.get_memory_operations().batch_store(store_requests)
        assert memory_result.status == BatchStatus.COMPLETED
        
        # Then, execute flows using stored data
        flow_requests = [
            FlowExecutionRequest.create("process_flow", "flow_input_1"),
            FlowExecutionRequest.create("process_flow", "flow_input_2")
        ]
        
        flow_result = await batch_manager.get_flow_operations().batch_execute_flows(flow_requests)
        assert flow_result.status == BatchStatus.COMPLETED
        
        # Verify performance monitoring tracked both operations
        stats = batch_manager.get_performance_stats()
        assert "batch_memory_store" in stats
        assert "batch_flow_execution" in stats
    
    @pytest.mark.asyncio
    async def test_full_workflow_with_analytics(self):
        """Test complete workflow including analytics."""
        memory_manager = MockMemoryManager()
        flow_manager = MockFlowManager()
        
        batch_manager = BatchOperationManager(
            memory_manager=memory_manager,
            flow_manager=flow_manager
        )
        
        # Execute some operations
        store_requests = [MemoryStoreRequest.create(f"key{i}", f"value{i}") for i in range(5)]
        await batch_manager.get_memory_operations().batch_store(store_requests)
        
        flow_requests = [FlowExecutionRequest.create("test_flow", f"input{i}") for i in range(3)]
        await batch_manager.get_flow_operations().batch_execute_flows(flow_requests)
        
        # Generate analytics reports
        report_configs = [
            {"name": "performance_summary", "type": "performance"},
            {"name": "memory_analysis", "type": "memory"}
        ]
        
        reports = await batch_manager.get_analytics_operations().batch_generate_reports(report_configs)
        
        assert "performance_summary" in reports
        assert "memory_analysis" in reports
        assert "batch_memory_store" in reports["performance_summary"]
        assert "batch_flow_execution" in reports["performance_summary"]


class TestConcurrencyAndPerformance:
    """Test concurrency control and performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_concurrent_memory_operations(self):
        """Test concurrent memory operations don't interfere."""
        memory_manager = MockMemoryManager()
        batch_ops = BatchMemoryOperations(memory_manager)
        
        # Run multiple batch operations concurrently
        store_requests1 = [MemoryStoreRequest.create(f"set1_key{i}", f"value{i}") for i in range(5)]
        store_requests2 = [MemoryStoreRequest.create(f"set2_key{i}", f"value{i}") for i in range(5)]
        
        results = await asyncio.gather(
            batch_ops.batch_store(store_requests1),
            batch_ops.batch_store(store_requests2)
        )
        
        assert all(result.status == BatchStatus.COMPLETED for result in results)
        assert sum(result.success_count for result in results) == 10
        
        # Verify all items were stored
        assert len(memory_manager.store_calls) == 10
    
    @pytest.mark.asyncio
    async def test_flow_concurrency_control(self):
        """Test flow execution concurrency control."""
        flow_manager = MockFlowManager()
        batch_ops = BatchFlowOperations(flow_manager)
        
        # Create requests that would exceed concurrency if not controlled
        requests = [FlowExecutionRequest.create("test_flow", f"input{i}") for i in range(10)]
        
        start_time = time.time()
        result = await batch_ops.batch_execute_flows(requests, max_concurrent=3)
        end_time = time.time()
        
        assert result.status == BatchStatus.COMPLETED
        assert result.success_count == 10
        
        # With concurrency control, should take some minimum time
        # (This is a rough check, actual timing may vary)
        assert end_time - start_time > 0
    
    @pytest.mark.asyncio
    async def test_search_concurrency_control(self):
        """Test search operation concurrency control."""
        memory_manager = MockMemoryManager()
        batch_ops = BatchMemoryOperations(memory_manager)
        
        queries = [f"query{i}" for i in range(20)]
        
        result = await batch_ops.batch_search(queries, max_concurrent=5)
        
        assert result.status == BatchStatus.COMPLETED
        assert result.success_count == 20
        assert len(memory_manager.search_calls) == 20


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_memory_manager_none_handling(self):
        """Test handling when memory manager is None."""
        batch_ops = BatchMemoryOperations(None)
        
        requests = [MemoryStoreRequest.create("key1", "value1")]
        
        result = await batch_ops.batch_store(requests)
        
        # Should handle None gracefully and return failed result
        assert result.status == BatchStatus.FAILED
        assert result.failure_count == 1
        assert result.success_count == 0
    
    @pytest.mark.asyncio
    async def test_flow_manager_none_handling(self):
        """Test handling when flow manager is None."""
        batch_ops = BatchFlowOperations(None)
        
        requests = [FlowExecutionRequest.create("test_flow", "input")]
        
        result = await batch_ops.batch_execute_flows(requests)
        
        # Should handle None gracefully and return failed result
        assert result.status == BatchStatus.FAILED
        assert result.failure_count == 1
        assert result.success_count == 0
    
    @pytest.mark.asyncio
    async def test_mixed_success_failure_status_determination(self):
        """Test status determination with mixed results."""
        memory_manager = MockMemoryManager(fail_keys=["key2", "key4"])
        batch_ops = BatchMemoryOperations(memory_manager)
        
        requests = [MemoryStoreRequest.create(f"key{i}", f"value{i}") for i in range(1, 6)]
        
        result = await batch_ops.batch_store(requests)
        
        assert result.status == BatchStatus.PARTIAL_SUCCESS
        assert result.success_count == 3
        assert result.failure_count == 2
        assert result.success_rate == 60.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.asyncio
    async def test_empty_request_lists(self):
        """Test operations with empty request lists."""
        memory_manager = MockMemoryManager()
        batch_ops = BatchMemoryOperations(memory_manager)
        
        # Empty store requests
        result = await batch_ops.batch_store([])
        assert result.status == BatchStatus.COMPLETED
        assert result.total_count == 0
        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.success_rate == 0.0
        
        # Empty retrieve requests
        result = await batch_ops.batch_retrieve([])
        assert result.status == BatchStatus.COMPLETED
        assert result.total_count == 0
        
        # Empty search queries
        result = await batch_ops.batch_search([])
        assert result.status == BatchStatus.COMPLETED
        assert result.total_count == 0
    
    @pytest.mark.asyncio
    async def test_single_item_operations(self):
        """Test operations with single items."""
        memory_manager = MockMemoryManager()
        batch_ops = BatchMemoryOperations(memory_manager)
        
        requests = [MemoryStoreRequest.create("single_key", "single_value")]
        
        result = await batch_ops.batch_store(requests)
        
        assert result.status == BatchStatus.COMPLETED
        assert result.success_count == 1
        assert result.total_count == 1
        assert result.success_rate == 100.0
    
    @pytest.mark.asyncio
    async def test_very_large_batch_operations(self):
        """Test operations with large batches."""
        memory_manager = MockMemoryManager()
        batch_ops = BatchMemoryOperations(memory_manager)
        
        # Create a large number of requests
        requests = [MemoryStoreRequest.create(f"key{i}", f"value{i}") for i in range(1000)]
        
        result = await batch_ops.batch_store(requests, chunk_size=100)
        
        assert result.status == BatchStatus.COMPLETED
        assert result.success_count == 1000
        assert result.total_count == 1000
        assert len(memory_manager.store_calls) == 1000
    
    def test_batch_result_with_zero_total_count(self):
        """Test BatchResult with zero total count."""
        result = BatchResult(
            status=BatchStatus.COMPLETED,
            successful_items=[],
            failed_items=[],
            total_count=0,
            success_count=0,
            failure_count=0,
            processing_time=0.0
        )
        
        assert result.success_rate == 0.0


if __name__ == "__main__":
    pytest.main([__file__])