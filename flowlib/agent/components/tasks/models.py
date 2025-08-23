"""Task execution models with structured Pydantic contracts."""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import Field, ConfigDict
from flowlib.core.models import StrictBaseModel
from enum import Enum


class TaskStatus(str, Enum):
    """Status of a task execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(str, Enum):
    """Priority levels for tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskDependency(StrictBaseModel):
    """Dependency relationship between tasks."""
    
    task_id: str = Field(..., description="ID of the dependent task")
    dependency_type: str = Field("hard", description="Type of dependency (hard, soft)")
    required: bool = Field(True, description="Whether this dependency is required")


class TaskResource(StrictBaseModel):
    """Resource requirement for task execution."""
    
    resource_type: str = Field(..., description="Type of resource (cpu, memory, gpu, etc.)")
    amount: float = Field(..., description="Amount of resource required")
    unit: str = Field(..., description="Unit of measurement")
    exclusive: bool = Field(False, description="Whether resource needs exclusive access")


class TaskMetadata(StrictBaseModel):
    """Metadata for task execution."""
    
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    labels: Dict[str, str] = Field(default_factory=dict, description="Key-value labels")
    owner: Optional[str] = Field(None, description="Task owner or creator")
    group: Optional[str] = Field(None, description="Task group for organization")
    environment: Optional[str] = Field(None, description="Execution environment")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")


class TaskInput(StrictBaseModel):
    """Input data for task execution."""
    
    data: Union[Dict[str, Any], StrictBaseModel, Any] = Field(..., description="Input data for the task")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    validation_schema: Optional[str] = Field(None, description="Schema for input validation")


class TaskOutput(StrictBaseModel):
    """Output data from task execution."""
    
    data: Union[Dict[str, Any], StrictBaseModel, Any] = Field(..., description="Output data from the task")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Output metadata")
    artifacts: List[str] = Field(default_factory=list, description="Paths to generated artifacts")
    validation_schema: Optional[str] = Field(None, description="Schema for output validation")


class TaskError(StrictBaseModel):
    """Error information for failed tasks."""
    
    error_type: str = Field(..., description="Type of error")
    error_message: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    stack_trace: Optional[str] = Field(None, description="Full stack trace")
    recovery_suggestions: List[str] = Field(default_factory=list, description="Suggestions for recovery")
    retryable: bool = Field(False, description="Whether the error is retryable")


class TaskProgress(StrictBaseModel):
    """Progress information for running tasks."""
    
    percentage: float = Field(0.0, ge=0.0, le=100.0, description="Completion percentage")
    current_step: str = Field("", description="Current execution step")
    total_steps: Optional[int] = Field(None, description="Total number of steps")
    completed_steps: int = Field(0, description="Number of completed steps")
    estimated_remaining_ms: Optional[float] = Field(None, description="Estimated remaining time in milliseconds")
    status_message: str = Field("", description="Current status message")
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Last progress update")


class TaskExecutionRequest(StrictBaseModel):
    """Request to execute a task."""
    
    task_id: str = Field(..., description="Unique task identifier")
    task_name: str = Field(..., description="Name of the task to execute")
    task_type: str = Field(..., description="Type of task")
    priority: TaskPriority = Field(TaskPriority.MEDIUM, description="Task priority")
    input_data: TaskInput = Field(..., description="Input data for the task")
    dependencies: List[TaskDependency] = Field(default_factory=list, description="Task dependencies")
    resources: List[TaskResource] = Field(default_factory=list, description="Required resources")
    timeout_ms: Optional[float] = Field(None, description="Timeout in milliseconds")
    retry_count: int = Field(0, description="Number of retry attempts")
    max_retries: int = Field(3, description="Maximum number of retries")
    metadata: TaskMetadata = Field(default_factory=TaskMetadata, description="Task metadata")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Creation timestamp")


class TaskExecutionResult(StrictBaseModel):
    """Result of task execution."""
    
    task_id: str = Field(..., description="Unique task identifier")
    task_name: str = Field(..., description="Name of the executed task")
    status: TaskStatus = Field(..., description="Execution status")
    output: Optional[TaskOutput] = Field(None, description="Task output data")
    error: Optional[TaskError] = Field(None, description="Error information if failed")
    started_at: str = Field(..., description="When execution started")
    completed_at: Optional[str] = Field(None, description="When execution completed")
    duration_ms: Optional[float] = Field(None, description="Execution duration in milliseconds")
    retry_count: int = Field(0, description="Number of retries attempted")
    resource_usage: Dict[str, float] = Field(default_factory=dict, description="Resource usage during execution")
    progress: Optional[TaskProgress] = Field(None, description="Final progress information")
    metadata: TaskMetadata = Field(default_factory=TaskMetadata, description="Task metadata")
    execution_log: List[str] = Field(default_factory=list, description="Execution log entries")


class TaskBatchRequest(StrictBaseModel):
    """Request to execute multiple tasks."""
    
    batch_id: str = Field(..., description="Unique batch identifier")
    tasks: List[TaskExecutionRequest] = Field(..., description="Tasks to execute")
    max_concurrent: int = Field(5, description="Maximum concurrent executions")
    fail_fast: bool = Field(False, description="Whether to stop on first failure")
    timeout_ms: Optional[float] = Field(None, description="Overall batch timeout")
    metadata: TaskMetadata = Field(default_factory=TaskMetadata, description="Batch metadata")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Creation timestamp")


class TaskBatchResult(StrictBaseModel):
    """Result of batch task execution."""
    
    batch_id: str = Field(..., description="Unique batch identifier")
    total_tasks: int = Field(0, description="Total number of tasks in batch")
    completed_tasks: int = Field(0, description="Number of completed tasks")
    failed_tasks: int = Field(0, description="Number of failed tasks")
    cancelled_tasks: int = Field(0, description="Number of cancelled tasks")
    success_rate: float = Field(0.0, description="Success rate as percentage")
    results: List[TaskExecutionResult] = Field(default_factory=list, description="Individual task results")
    started_at: str = Field(..., description="When batch execution started")
    completed_at: Optional[str] = Field(None, description="When batch execution completed")
    total_duration_ms: Optional[float] = Field(None, description="Total batch duration in milliseconds")
    metadata: TaskMetadata = Field(default_factory=TaskMetadata, description="Batch metadata")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Batch execution summary")


class TaskQueue(StrictBaseModel):
    """Task queue status and statistics."""
    
    queue_name: str = Field(..., description="Name of the task queue")
    pending_tasks: int = Field(0, description="Number of pending tasks")
    running_tasks: int = Field(0, description="Number of running tasks")
    completed_tasks: int = Field(0, description="Number of completed tasks")
    failed_tasks: int = Field(0, description="Number of failed tasks")
    max_size: Optional[int] = Field(None, description="Maximum queue size")
    priority_distribution: Dict[TaskPriority, int] = Field(default_factory=dict, description="Tasks by priority")
    avg_wait_time_ms: Optional[float] = Field(None, description="Average wait time in milliseconds")
    avg_execution_time_ms: Optional[float] = Field(None, description="Average execution time in milliseconds")
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Last update timestamp")


class TaskSchedule(StrictBaseModel):
    """Scheduled task configuration."""
    
    schedule_id: str = Field(..., description="Unique schedule identifier")
    task_template: TaskExecutionRequest = Field(..., description="Template for scheduled tasks")
    schedule_expression: str = Field(..., description="Cron or schedule expression")
    enabled: bool = Field(True, description="Whether the schedule is enabled")
    next_run: Optional[str] = Field(None, description="Next scheduled run time")
    last_run: Optional[str] = Field(None, description="Last run time")
    run_count: int = Field(0, description="Number of times executed")
    success_count: int = Field(0, description="Number of successful runs")
    failure_count: int = Field(0, description="Number of failed runs")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Creation timestamp")
    metadata: TaskMetadata = Field(default_factory=TaskMetadata, description="Schedule metadata")


class WorkerStatus(StrictBaseModel):
    """Status of a task worker."""
    
    worker_id: str = Field(..., description="Unique worker identifier")
    worker_name: str = Field(..., description="Worker name")
    status: str = Field(..., description="Worker status (idle, busy, error, offline)")
    current_task: Optional[str] = Field(None, description="ID of currently executing task")
    total_tasks_completed: int = Field(0, description="Total tasks completed by this worker")
    total_tasks_failed: int = Field(0, description="Total tasks failed by this worker")
    avg_task_duration_ms: Optional[float] = Field(None, description="Average task duration in milliseconds")
    cpu_usage: Optional[float] = Field(None, description="Current CPU usage percentage")
    memory_usage: Optional[float] = Field(None, description="Current memory usage in MB")
    last_heartbeat: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Last heartbeat timestamp")
    started_at: str = Field(..., description="When worker was started")
    capabilities: List[str] = Field(default_factory=list, description="Worker capabilities")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Worker metadata")