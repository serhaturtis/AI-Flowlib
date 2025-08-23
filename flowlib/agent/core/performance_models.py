"""Performance monitoring models with structured Pydantic contracts."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class OperationStatus(str, Enum):
    """Status of an operation."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class PerformanceMetric(BaseModel):
    """Individual performance metric."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    operation_name: str = Field(..., description="Name of the operation")
    duration_ms: float = Field(..., description="Duration in milliseconds")
    status: OperationStatus = Field(..., description="Operation status")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When the operation occurred")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional operation metadata")
    error_type: Optional[str] = Field(None, description="Type of error if operation failed")
    error_message: Optional[str] = Field(None, description="Error message if operation failed")


class OperationStats(BaseModel):
    """Statistics for a specific operation type."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    operation_name: str = Field(..., description="Name of the operation")
    total_count: int = Field(0, description="Total number of operations")
    success_count: int = Field(0, description="Number of successful operations")
    error_count: int = Field(0, description="Number of failed operations")
    success_rate: float = Field(0.0, description="Success rate as percentage (0-100)")
    avg_duration_ms: float = Field(0.0, description="Average duration in milliseconds")
    min_duration_ms: float = Field(0.0, description="Minimum duration in milliseconds")
    max_duration_ms: float = Field(0.0, description="Maximum duration in milliseconds")
    median_duration_ms: float = Field(0.0, description="Median duration in milliseconds")
    p95_duration_ms: float = Field(0.0, description="95th percentile duration in milliseconds")
    p99_duration_ms: float = Field(0.0, description="99th percentile duration in milliseconds")
    last_execution: Optional[str] = Field(None, description="Timestamp of last execution")
    error_types: Dict[str, int] = Field(default_factory=dict, description="Count of different error types")


class CacheStats(BaseModel):
    """Statistics for cache performance."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    cache_name: str = Field(..., description="Name of the cache")
    total_size: int = Field(0, description="Total number of entries in cache")
    max_size: int = Field(0, description="Maximum cache size")
    hit_count: int = Field(0, description="Number of cache hits")
    miss_count: int = Field(0, description="Number of cache misses")
    hit_rate: float = Field(0.0, description="Cache hit rate as percentage (0-100)")
    eviction_count: int = Field(0, description="Number of evictions")
    oldest_entry_age_ms: float = Field(0.0, description="Age of oldest entry in milliseconds")
    memory_usage_bytes: Optional[int] = Field(None, description="Memory usage in bytes")
    last_cleanup: Optional[str] = Field(None, description="Timestamp of last cleanup")


class MemoryAccessPattern(BaseModel):
    """Memory access pattern statistics."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    key: str = Field(..., description="Memory key that was accessed")
    access_count: int = Field(0, description="Number of times accessed")
    last_accessed: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Last access timestamp")
    avg_retrieval_time_ms: float = Field(0.0, description="Average retrieval time in milliseconds")
    access_frequency: float = Field(0.0, description="Accesses per minute")
    is_hot_key: bool = Field(False, description="Whether this is considered a hot key")


class BatchOperationStats(BaseModel):
    """Statistics for batch operations."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    operation_type: str = Field(..., description="Type of batch operation")
    batch_size: int = Field(0, description="Size of the batch")
    total_items: int = Field(0, description="Total items processed")
    successful_items: int = Field(0, description="Number of successful items")
    failed_items: int = Field(0, description="Number of failed items")
    success_rate: float = Field(0.0, description="Success rate as percentage (0-100)")
    total_duration_ms: float = Field(0.0, description="Total processing time in milliseconds")
    avg_item_duration_ms: float = Field(0.0, description="Average time per item in milliseconds")
    throughput_items_per_second: float = Field(0.0, description="Items processed per second")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When the batch was processed")
    error_details: Dict[str, Any] = Field(default_factory=dict, description="Details about any errors")


class ResourceUsageStats(BaseModel):
    """Resource usage statistics."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    cpu_percent: Optional[float] = Field(None, description="CPU usage percentage")
    memory_mb: Optional[float] = Field(None, description="Memory usage in MB")
    disk_io_bytes: Optional[int] = Field(None, description="Disk I/O in bytes")
    network_io_bytes: Optional[int] = Field(None, description="Network I/O in bytes")
    active_tasks: int = Field(0, description="Number of active tasks")
    thread_count: Optional[int] = Field(None, description="Number of active threads")
    open_files: Optional[int] = Field(None, description="Number of open file descriptors")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When the stats were collected")


class PerformanceReport(BaseModel):
    """Comprehensive performance report."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    report_name: str = Field(..., description="Name of the report")
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When the report was generated")
    time_period_start: str = Field(..., description="Start of the reporting period")
    time_period_end: str = Field(..., description="End of the reporting period")
    operation_stats: List[OperationStats] = Field(default_factory=list, description="Statistics for each operation type")
    cache_stats: List[CacheStats] = Field(default_factory=list, description="Statistics for each cache")
    memory_patterns: List[MemoryAccessPattern] = Field(default_factory=list, description="Memory access patterns")
    batch_operations: List[BatchOperationStats] = Field(default_factory=list, description="Batch operation statistics")
    resource_usage: Optional[ResourceUsageStats] = Field(None, description="Current resource usage")
    summary: Dict[str, Any] = Field(default_factory=dict, description="High-level summary metrics")
    recommendations: List[str] = Field(default_factory=list, description="Performance improvement recommendations")


class AlertThreshold(BaseModel):
    """Threshold for performance alerts."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    metric_name: str = Field(..., description="Name of the metric to monitor")
    threshold_value: float = Field(..., description="Threshold value that triggers alert")
    comparison_operator: str = Field(..., description="Comparison operator (>, <, >=, <=, ==)")
    severity: str = Field("warning", description="Alert severity level")
    enabled: bool = Field(True, description="Whether this alert is enabled")
    description: str = Field("", description="Description of what this alert monitors")


class PerformanceAlert(BaseModel):
    """Performance alert triggered by threshold breach."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    alert_id: str = Field(..., description="Unique identifier for the alert")
    metric_name: str = Field(..., description="Name of the metric that triggered alert")
    current_value: float = Field(..., description="Current value of the metric")
    threshold_value: float = Field(..., description="Threshold that was breached")
    severity: str = Field(..., description="Alert severity level")
    message: str = Field(..., description="Alert message")
    triggered_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When the alert was triggered")
    resolved_at: Optional[str] = Field(None, description="When the alert was resolved")
    acknowledged: bool = Field(False, description="Whether the alert has been acknowledged")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional alert metadata")


class SystemHealthCheck(BaseModel):
    """Result of system health check."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    healthy: bool = Field(..., description="Overall system health status")
    checks: Dict[str, bool] = Field(default_factory=dict, description="Individual health check results")
    response_times: Dict[str, float] = Field(default_factory=dict, description="Response times for each check")
    error_messages: Dict[str, str] = Field(default_factory=dict, description="Error messages for failed checks")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When the health check was performed")
    uptime_seconds: Optional[float] = Field(None, description="System uptime in seconds")
    version_info: Dict[str, str] = Field(default_factory=dict, description="Version information")