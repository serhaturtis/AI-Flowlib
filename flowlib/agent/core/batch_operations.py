"""Batch operations for efficient bulk processing in agent components."""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, TypeVar, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
from pydantic import BaseModel, Field, ConfigDict

from flowlib.agent.core.errors import AgentError, ComponentError
from flowlib.agent.core.performance import PerformanceMonitor

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class BatchStatus(Enum):
    """Status of batch operations."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL_SUCCESS = "partial_success"


class BatchFailedItem(BaseModel):
    """Represents a failed item in a batch operation."""
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    
    item: Any = Field(..., description="The original item that failed")
    error_message: str = Field(..., description="Error message describing the failure")
    error_type: str = Field(..., description="Type of the error that occurred")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class BatchResult(BaseModel):
    """Result of a batch operation."""
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    
    status: BatchStatus
    successful_items: List[Any] = Field(default_factory=list)
    failed_items: List[BatchFailedItem] = Field(default_factory=list)
    total_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    processing_time: float = 0.0
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    error_summary: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        return (self.success_count / self.total_count * 100) if self.total_count > 0 else 0


class StrictMetadata(BaseModel):
    """Strict metadata model with no fallbacks."""
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    
    data: Dict[str, Any] = Field(default_factory=dict, description="Metadata content")
    
    @classmethod
    def create(cls, metadata: Optional[Dict[str, Any]] = None) -> 'StrictMetadata':
        """Create metadata from optional dict."""
        return cls(data=metadata if metadata is not None else {})


class StrictContext(BaseModel):
    """Strict context model with no fallbacks.""" 
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    
    data: Dict[str, Any] = Field(default_factory=dict, description="Context data")
    
    @classmethod
    def create(cls, context: Optional[Dict[str, Any]] = None) -> 'StrictContext':
        """Create context from optional dict."""
        return cls(data=context if context is not None else {})


class MemoryStoreRequest(BaseModel):
    """Request to store memory item with strict contracts."""
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    
    key: str = Field(description="Memory key")
    value: Any = Field(description="Value to store")
    metadata: StrictMetadata = Field(default_factory=StrictMetadata, description="Validated metadata")
    ttl_seconds: Optional[float] = Field(default=None, description="TTL in seconds if applicable")
    
    @classmethod
    def create(cls, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None, ttl: Optional[float] = None) -> 'MemoryStoreRequest':
        """Create store request with validated inputs."""
        return cls(
            key=key,
            value=value, 
            metadata=StrictMetadata.create(metadata),
            ttl_seconds=ttl
        )


class MemoryRetrieveRequest(BaseModel):
    """Request to retrieve memory item with strict contracts."""
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    
    key: str = Field(description="Memory key") 
    # No default fallback - callers must handle None explicitly
    
    @classmethod
    def create(cls, key: str) -> 'MemoryRetrieveRequest':
        """Create retrieve request."""
        return cls(key=key)


class FlowExecutionRequest(BaseModel):
    """Request to execute a flow with strict contracts."""
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    
    flow_name: str = Field(description="Flow name to execute")
    input_data: Any = Field(description="Input data for flow")
    context: StrictContext = Field(default_factory=StrictContext, description="Execution context")
    timeout_seconds: Optional[float] = Field(default=None, description="Timeout in seconds") 
    priority: str = Field(default="medium", description="Execution priority")
    metadata: StrictMetadata = Field(default_factory=StrictMetadata, description="Request metadata")
    
    @classmethod
    def create(
        cls,
        flow_name: str,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None,
        timeout_ms: Optional[float] = None,
        priority: str = "medium",
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'FlowExecutionRequest':
        """Create flow execution request with validated inputs."""
        timeout_seconds = timeout_ms / 1000.0 if timeout_ms is not None else None
        return cls(
            flow_name=flow_name,
            input_data=input_data,
            context=StrictContext.create(context),
            timeout_seconds=timeout_seconds,
            priority=priority,
            metadata=StrictMetadata.create(metadata)
        )


class BatchMemoryOperations:
    """Batch operations for memory subsystem."""
    
    def __init__(self, memory_manager, performance_monitor: Optional[PerformanceMonitor] = None):
        self.memory_manager = memory_manager
        # Create default PerformanceMonitor if not provided (sensible default, not a fallback)
        self.performance_monitor = performance_monitor if performance_monitor is not None else PerformanceMonitor()
    
    async def batch_store(
        self,
        requests: List[MemoryStoreRequest],
        chunk_size: int = 50
    ) -> BatchResult:
        """Store multiple memory items in batches."""
        start_time = asyncio.get_event_loop().time()
        
        successful_items = []
        failed_items = []
        
        # Process in chunks to avoid overwhelming the system
        for i in range(0, len(requests), chunk_size):
            chunk = requests[i:i + chunk_size]
            chunk_results = await self._process_store_chunk(chunk)
            
            successful_items.extend(chunk_results[0])
            failed_items.extend(chunk_results[1])
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        # Record performance metrics
        self.performance_monitor.record_operation(
            "batch_memory_store",
            processing_time,
            success=len(failed_items) == 0
        )
        
        # Convert failed items tuples to BatchFailedItem models
        failed_items_models = []
        for item, error in failed_items:
            failed_items_models.append(BatchFailedItem(
                item=item,
                error_message=str(error),
                error_type=type(error).__name__
            ))
        
        return BatchResult(
            status=self._determine_batch_status(len(successful_items), len(failed_items)),
            successful_items=successful_items,
            failed_items=failed_items_models,
            total_count=len(requests),
            success_count=len(successful_items),
            failure_count=len(failed_items),
            processing_time=processing_time,
            completed_at=datetime.now().isoformat()
        )
    
    async def batch_retrieve(
        self,
        requests: List[MemoryRetrieveRequest],
        chunk_size: int = 100
    ) -> BatchResult:
        """Retrieve multiple memory items in batches."""
        start_time = asyncio.get_event_loop().time()
        
        successful_items = []
        failed_items = []
        
        # Process in chunks
        for i in range(0, len(requests), chunk_size):
            chunk = requests[i:i + chunk_size]
            chunk_results = await self._process_retrieve_chunk(chunk)
            
            successful_items.extend(chunk_results[0])
            failed_items.extend(chunk_results[1])
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        # Record performance metrics
        self.performance_monitor.record_operation(
            "batch_memory_retrieve",
            processing_time,
            success=len(failed_items) == 0
        )
        
        # Convert failed items tuples to BatchFailedItem models
        failed_items_models = []
        for item, error in failed_items:
            failed_items_models.append(BatchFailedItem(
                item=item,
                error_message=str(error),
                error_type=type(error).__name__
            ))
        
        return BatchResult(
            status=self._determine_batch_status(len(successful_items), len(failed_items)),
            successful_items=successful_items,
            failed_items=failed_items_models,
            total_count=len(requests),
            success_count=len(successful_items),
            failure_count=len(failed_items),
            processing_time=processing_time,
            completed_at=datetime.now().isoformat()
        )
    
    async def batch_search(
        self,
        queries: List[str],
        memory_types: Optional[List[str]] = None,
        max_concurrent: int = 10
    ) -> BatchResult:
        """Search multiple queries across memory types."""
        start_time = asyncio.get_event_loop().time()
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def search_with_semaphore(query: str):
            async with semaphore:
                try:
                    # Strict contract - memory manager must have search_all_memories method
                    if not hasattr(self.memory_manager, 'search_all_memories'):
                        raise ComponentError(
                            f"Memory manager {type(self.memory_manager).__name__} missing required 'search_all_memories' method",
                            "BatchMemoryOperations"
                        )
                    result = await self.memory_manager.search_all_memories(query, memory_types)
                    return (query, result)
                except Exception as e:
                    raise ComponentError(f"Search failed for query: {query}", "BatchMemoryOperations") from e
        
        # Execute all searches concurrently
        tasks = [search_with_semaphore(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_items = []
        failed_items = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_items.append((queries[i], result))
            else:
                successful_items.append(result)
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        self.performance_monitor.record_operation(
            "batch_memory_search",
            processing_time,
            success=len(failed_items) == 0
        )
        
        # Convert failed items tuples to BatchFailedItem models
        failed_items_models = []
        for item, error in failed_items:
            failed_items_models.append(BatchFailedItem(
                item=item,
                error_message=str(error),
                error_type=type(error).__name__
            ))
        
        return BatchResult(
            status=self._determine_batch_status(len(successful_items), len(failed_items)),
            successful_items=successful_items,
            failed_items=failed_items_models,
            total_count=len(queries),
            success_count=len(successful_items),
            failure_count=len(failed_items),
            processing_time=processing_time
        )
    
    async def _process_store_chunk(
        self,
        chunk: List[MemoryStoreRequest]
    ) -> Tuple[List[str], List[Tuple[MemoryStoreRequest, Exception]]]:
        """Process a chunk of store requests."""
        successful = []
        failed = []
        
        # Create tasks for parallel execution
        tasks = []
        for request in chunk:
            task = self._store_single_item(request)
            tasks.append((request, task))
        
        # Execute all tasks
        for request, task in tasks:
            try:
                result = await task
                successful.append(result)
            except Exception as e:
                failed.append((request, e))
                logger.warning(f"Failed to store {request.key}: {e}")
        
        return successful, failed
    
    async def _process_retrieve_chunk(
        self,
        chunk: List[MemoryRetrieveRequest]
    ) -> Tuple[List[Tuple[str, Any]], List[Tuple[MemoryRetrieveRequest, Exception]]]:
        """Process a chunk of retrieve requests."""
        successful = []
        failed = []
        
        # Create tasks for parallel execution
        tasks = []
        for request in chunk:
            task = self._retrieve_single_item(request)
            tasks.append((request, task))
        
        # Execute all tasks
        for request, task in tasks:
            try:
                result = await task
                successful.append((request.key, result))
            except Exception as e:
                failed.append((request, e))
                logger.warning(f"Failed to retrieve {request.key}: {e}")
        
        return successful, failed
    
    async def _store_single_item(self, request: MemoryStoreRequest) -> str:
        """Store a single memory item with strict contract enforcement."""
        # Enforce strict contract - memory manager must have store_with_metadata method
        if not hasattr(self.memory_manager, 'store_with_metadata'):
            raise ComponentError(
                f"Memory manager {type(self.memory_manager).__name__} missing required 'store_with_metadata' method",
                "BatchMemoryOperations"
            )
        
        await self.memory_manager.store_with_metadata(
            request.key,
            request.value,
            request.metadata.data
        )
        return request.key
    
    async def _retrieve_single_item(self, request: MemoryRetrieveRequest) -> Any:
        """Retrieve a single memory item - no fallbacks."""
        result = await self.memory_manager.retrieve(request.key)
        # No default fallback - callers must handle None explicitly
        return result
    
    def _determine_batch_status(self, success_count: int, failure_count: int) -> BatchStatus:
        """Determine overall batch status."""
        if failure_count == 0:
            return BatchStatus.COMPLETED
        elif success_count == 0:
            return BatchStatus.FAILED
        else:
            return BatchStatus.PARTIAL_SUCCESS


class BatchFlowOperations:
    """Batch operations for flow execution."""
    
    def __init__(self, flow_manager, performance_monitor: Optional[PerformanceMonitor] = None):
        self.flow_manager = flow_manager
        # Create default PerformanceMonitor if not provided (sensible default, not a fallback)
        self.performance_monitor = performance_monitor if performance_monitor is not None else PerformanceMonitor()
    
    async def batch_execute_flows(
        self,
        requests: List[FlowExecutionRequest],
        max_concurrent: int = 5,
        timeout_per_flow: float = 300.0
    ) -> BatchResult:
        """Execute multiple flows in parallel with concurrency control."""
        start_time = asyncio.get_event_loop().time()
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(request: FlowExecutionRequest):
            async with semaphore:
                try:
                    # Execute with timeout
                    result = await asyncio.wait_for(
                        self._execute_single_flow(request),
                        timeout=timeout_per_flow
                    )
                    return (request, result)
                except asyncio.TimeoutError:
                    raise ComponentError(
                        f"Flow execution timed out: {request.flow_name}", "BatchFlowOperations"
                    )
                except Exception as e:
                    raise ComponentError(
                        f"Flow execution failed: {request.flow_name}", "BatchFlowOperations"
                    ) from e
        
        # Execute all flows concurrently
        tasks = [execute_with_semaphore(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_items = []
        failed_items = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_items.append((requests[i], result))
            else:
                successful_items.append(result)
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        self.performance_monitor.record_operation(
            "batch_flow_execution",
            processing_time,
            success=len(failed_items) == 0
        )
        
        # Convert failed items tuples to BatchFailedItem models
        failed_items_models = []
        for item, error in failed_items:
            failed_items_models.append(BatchFailedItem(
                item=item,
                error_message=str(error),
                error_type=type(error).__name__
            ))
        
        return BatchResult(
            status=self._determine_batch_status(len(successful_items), len(failed_items)),
            successful_items=successful_items,
            failed_items=failed_items_models,
            total_count=len(requests),
            success_count=len(successful_items),
            failure_count=len(failed_items),
            processing_time=processing_time
        )
    
    async def batch_execute_same_flow(
        self,
        flow_name: str,
        input_list: List[Any],
        max_concurrent: int = 5
    ) -> BatchResult:
        """Execute the same flow with different inputs."""
        requests = [
            FlowExecutionRequest(flow_name=flow_name, input_data=input_data)
            for input_data in input_list
        ]
        
        return await self.batch_execute_flows(requests, max_concurrent)
    
    async def _execute_single_flow(self, request: FlowExecutionRequest) -> Any:
        """Execute a single flow with strict contract enforcement."""
        # Enforce strict contract - flow manager must have get_flow method
        if not hasattr(self.flow_manager, 'get_flow'):
            raise ComponentError(
                f"Flow manager {type(self.flow_manager).__name__} missing required 'get_flow' method",
                "BatchFlowOperations"
            )
        
        flow = self.flow_manager.get_flow(request.flow_name)
        if flow is None:
            raise ComponentError(f"Flow not found: {request.flow_name}", "BatchFlowOperations")
        
        # Enforce strict execution contract - flow must have run_pipeline method
        if not hasattr(flow, 'run_pipeline'):
            raise ComponentError(
                f"Flow {request.flow_name} missing required 'run_pipeline' method. "
                f"All flows must implement the run_pipeline interface.",
                "BatchFlowOperations"
            )
        
        return await flow.run_pipeline(request.input_data)
    
    def _determine_batch_status(self, success_count: int, failure_count: int) -> BatchStatus:
        """Determine overall batch status."""
        if failure_count == 0:
            return BatchStatus.COMPLETED
        elif success_count == 0:
            return BatchStatus.FAILED
        else:
            return BatchStatus.PARTIAL_SUCCESS


class BatchAnalyticsOperations:
    """Batch operations for analytics and reporting."""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        self.performance_monitor = performance_monitor
    
    async def batch_analyze_memory_patterns(
        self,
        memory_keys: List[str],
        analysis_functions: List[Callable]
    ) -> Dict[str, Any]:
        """Analyze multiple memory access patterns."""
        results = {}
        
        for analysis_func in analysis_functions:
            func_name = analysis_func.__name__
            
            try:
                # Run analysis on all keys
                analysis_result = await analysis_func(memory_keys)
                results[func_name] = analysis_result
                
            except Exception as e:
                logger.error(f"Analysis function {func_name} failed: {e}")
                # Store error in results instead of propagating
                results[func_name] = {"error": str(e)}
        
        return results
    
    async def batch_generate_reports(
        self,
        report_configs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate multiple reports concurrently."""
        reports = {}
        
        tasks = []
        for config in report_configs:
            if "name" not in config:
                raise ValueError(f"Report config missing required 'name' field: {config}")
            report_name = config["name"]
            task = self._generate_single_report(config)
            tasks.append((report_name, task))
        
        # Execute all report generation concurrently
        for report_name, task in tasks:
            try:
                report_data = await task
                reports[report_name] = report_data
            except Exception as e:
                logger.error(f"Report generation failed for {report_name}: {e}")
                # Store error in reports instead of propagating
                reports[report_name] = {"error": str(e)}
        
        return reports
    
    async def _generate_single_report(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single report based on configuration."""
        try:
            if "type" not in config:
                raise ValueError(f"Report config missing required 'type' field: {config}")
            report_type = config["type"]
            
            if report_type == "performance":
                return self.performance_monitor.get_all_stats()
            elif report_type == "memory":
                return self.performance_monitor.memory_analytics.get_performance_stats()
            else:
                raise ComponentError(f"Unknown report type: {report_type}. Supported types: performance, memory", "BatchReportOperations")
        except Exception as e:
            return {"error": str(e)}


class BatchOperationManager:
    """Central manager for all batch operations."""
    
    def __init__(
        self,
        memory_manager=None,
        flow_manager=None,
        performance_monitor: Optional[PerformanceMonitor] = None
    ):
        # Create default PerformanceMonitor if not provided (sensible default, not a fallback)
        self.performance_monitor = performance_monitor if performance_monitor is not None else PerformanceMonitor()
        
        self.memory_ops = BatchMemoryOperations(
            memory_manager, self.performance_monitor
        ) if memory_manager else None
        
        self.flow_ops = BatchFlowOperations(
            flow_manager, self.performance_monitor
        ) if flow_manager else None
        
        self.analytics_ops = BatchAnalyticsOperations(self.performance_monitor)
    
    def get_memory_operations(self) -> Optional[BatchMemoryOperations]:
        """Get memory batch operations."""
        return self.memory_ops
    
    def get_flow_operations(self) -> Optional[BatchFlowOperations]:
        """Get flow batch operations."""
        return self.flow_ops
    
    def get_analytics_operations(self) -> BatchAnalyticsOperations:
        """Get analytics batch operations."""
        return self.analytics_ops
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics."""
        return self.performance_monitor.get_all_stats()