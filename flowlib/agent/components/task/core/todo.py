"""Core TODO domain models.

This module contains the fundamental TODO models with no dependencies
on other task components.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated, Any
from uuid import uuid4

from pydantic import BeforeValidator, Field

from flowlib.core.models import MutableStrictBaseModel, StrictBaseModel


class TodoStatus(Enum):
    """Status of a TODO item."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    BLOCKED = "blocked"


class TodoPriority(Enum):
    """Priority levels for TODO items."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


def validate_todo_status(value: TodoStatus | str | object) -> TodoStatus:
    """Convert string to TodoStatus enum."""
    if isinstance(value, TodoStatus):
        return value
    if isinstance(value, str):
        try:
            return TodoStatus(value.lower())
        except ValueError:
            return TodoStatus.PENDING
    return TodoStatus.PENDING


def validate_todo_priority(value: TodoPriority | str | object) -> TodoPriority:
    """Convert string to TodoPriority enum."""
    if isinstance(value, TodoPriority):
        return value
    if isinstance(value, str):
        try:
            return TodoPriority(value.lower())
        except ValueError:
            return TodoPriority.MEDIUM
    return TodoPriority.MEDIUM


class TodoStatusSummary(StrictBaseModel):
    """Status summary for TODO items."""

    total: int = Field(0, ge=0, description="Total number of TODO items")
    pending: int = Field(0, ge=0, description="Number of pending items")
    in_progress: int = Field(0, ge=0, description="Number of in-progress items")
    completed: int = Field(0, ge=0, description="Number of completed items")
    cancelled: int = Field(0, ge=0, description="Number of cancelled items")
    failed: int = Field(0, ge=0, description="Number of failed items")
    blocked: int = Field(0, ge=0, description="Number of blocked items")


class TodoItem(StrictBaseModel):
    """Individual TODO item with complete lifecycle management.

    This is the core TODO model without execution-specific dependencies.
    """

    # Core identification
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier")
    content: str = Field(..., description="Description of the task")

    # Status and priority with validation
    status: Annotated[TodoStatus, BeforeValidator(validate_todo_status)] = Field(
        default=TodoStatus.PENDING, description="Current status"
    )
    priority: Annotated[TodoPriority, BeforeValidator(validate_todo_priority)] = Field(
        default=TodoPriority.MEDIUM, description="Task priority"
    )

    # Timestamps - system-managed fields
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime | None = Field(default=None, description="Last update timestamp")
    started_at: datetime | None = Field(default=None, description="When task started")
    completed_at: datetime | None = Field(default=None, description="When task completed")

    # Duration tracking
    estimated_duration: timedelta | None = Field(
        default=None, description="Estimated completion time"
    )
    actual_duration: timedelta | None = Field(default=None, description="Actual completion time")

    # Dependencies and relationships
    depends_on: list[str] = Field(default_factory=list, description="TODO IDs this depends on")
    blocks: list[str] = Field(default_factory=list, description="TODO IDs this blocks")
    parent_id: str | None = Field(
        default=None, description="Parent TODO ID if this is a subtask"
    )
    children: list[str] = Field(default_factory=list, description="Child TODO IDs")

    # Execution details - stored as Any to avoid circular dependencies
    # The execution module will handle the typed versions
    assigned_tool: str | None = Field(default=None, description="Tool assigned for execution")
    execution_context: Any | None = Field(
        default=None, description="Structured context for tool execution"
    )
    result: Any | None = Field(default=None, description="Structured execution result")
    error_message: str | None = Field(default=None, description="Error message if failed")
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")

    # Metadata
    tags: list[str] = Field(default_factory=list, description="Classification tags")
    notes: list[str] = Field(default_factory=list, description="Additional notes")

    def mark_in_progress(self) -> TodoItem:
        """Mark TODO as in progress."""
        now = datetime.now()
        return self.model_copy(
            update={"status": TodoStatus.IN_PROGRESS, "started_at": now, "updated_at": now}
        )

    def mark_completed(self, result: Any | None = None) -> TodoItem:
        """Mark TODO as completed."""
        now = datetime.now()
        update_data = {"status": TodoStatus.COMPLETED, "completed_at": now, "updated_at": now}

        if result is not None:
            update_data["result"] = result

        if self.started_at:
            update_data["actual_duration"] = now - self.started_at

        return self.model_copy(update=update_data)

    def mark_failed(self, error: str) -> TodoItem:
        """Mark TODO as failed."""
        return self.model_copy(
            update={
                "status": TodoStatus.FAILED,
                "error_message": error,
                "updated_at": datetime.now(),
            }
        )

    def mark_blocked(self, reason: str | None = None) -> TodoItem:
        """Mark TODO as blocked."""
        update_data = {"status": TodoStatus.BLOCKED, "updated_at": datetime.now()}

        if reason:
            update_data["error_message"] = reason

        return self.model_copy(update=update_data)

    def increment_retry(self) -> TodoItem:
        """Increment retry count."""
        return self.model_copy(
            update={"retry_count": self.retry_count + 1, "updated_at": datetime.now()}
        )

    def can_retry(self) -> bool:
        """Check if TODO can be retried."""
        return self.retry_count < self.max_retries

    def is_ready_to_execute(self, completed_todos: list[str]) -> bool:
        """Check if TODO is ready for execution based on dependencies."""
        if self.status != TodoStatus.PENDING:
            return False

        # Check if all dependencies are completed
        return all(dep_id in completed_todos for dep_id in self.depends_on)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Use mode='json' to properly serialize enums and nested models
        data = self.model_dump(mode="json")

        # Convert datetime objects to ISO format
        for field in ["created_at", "updated_at", "started_at", "completed_at"]:
            if data.get(field):
                data[field] = (
                    data[field].isoformat() if isinstance(data[field], datetime) else data[field]
                )

        # Convert timedelta objects from ISO format back to seconds for compatibility
        for field in ["estimated_duration", "actual_duration"]:
            if data.get(field) and isinstance(data[field], str) and data[field].startswith("PT"):
                # Convert ISO 8601 duration back to seconds
                duration_str = data[field]
                seconds = 0.0

                if "H" in duration_str:
                    hours = int(duration_str.split("H")[0].split("T")[1])
                    seconds += hours * 3600
                    duration_str = (
                        duration_str.split("H")[1] if "H" in duration_str else duration_str
                    )

                if "M" in duration_str:
                    minutes = int(duration_str.split("M")[0].split("T")[-1])
                    seconds += minutes * 60
                    duration_str = (
                        duration_str.split("M")[1] if "M" in duration_str else duration_str
                    )

                if "S" in duration_str:
                    secs = float(duration_str.split("S")[0].split("T")[-1])
                    seconds += secs

                data[field] = seconds

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TodoItem:
        """Create TodoItem from dictionary."""
        # Convert ISO datetime strings back to datetime objects
        for field in ["created_at", "updated_at", "started_at", "completed_at"]:
            if data.get(field) and isinstance(data[field], str):
                try:
                    data[field] = datetime.fromisoformat(data[field])
                except ValueError:
                    data[field] = None

        # Convert seconds back to timedelta objects
        for field in ["estimated_duration", "actual_duration"]:
            if data.get(field) and isinstance(data[field], (int, float)):
                data[field] = timedelta(seconds=data[field])

        # Execution context and result are left as-is (Any type)
        # The execution module will handle proper typing

        return cls(**data)


class TodoList(MutableStrictBaseModel):
    """Collection of TODO items with operations."""

    items: list[TodoItem] = Field(default_factory=list, description="TODO items in this list")
    name: str = Field(default="default", description="List name")
    description: str | None = Field(default=None, description="List description")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    def add_todo(self, todo: TodoItem) -> None:
        """Add a TODO item to the list."""
        self.items.append(todo)

    def remove_todo(self, todo_id: str) -> bool:
        """Remove a TODO item by ID."""
        for i, item in enumerate(self.items):
            if item.id == todo_id:
                self.items.pop(i)
                return True
        return False

    def get_todo(self, todo_id: str) -> TodoItem | None:
        """Get TODO by ID."""
        for item in self.items:
            if item.id == todo_id:
                return item
        return None

    def update_todo(self, todo_id: str, updated_todo: TodoItem) -> bool:
        """Update a TODO item."""
        for i, item in enumerate(self.items):
            if item.id == todo_id:
                self.items[i] = updated_todo
                return True
        return False

    def get_by_status(self, status: TodoStatus) -> list[TodoItem]:
        """Get all TODOs with specified status."""
        return [item for item in self.items if item.status == status]

    def get_ready_to_execute(self) -> list[TodoItem]:
        """Get TODOs ready for execution (no unmet dependencies)."""
        completed_ids = [item.id for item in self.items if item.status == TodoStatus.COMPLETED]
        return [item for item in self.items if item.is_ready_to_execute(completed_ids)]

    def get_status_summary(self) -> TodoStatusSummary:
        """Get status summary for all TODOs."""
        status_counts = {}
        for status in TodoStatus:
            status_counts[status.value] = 0

        for item in self.items:
            status_counts[item.status.value] += 1

        return TodoStatusSummary(
            total=len(self.items),
            pending=status_counts.get("pending", 0),
            in_progress=status_counts.get("in_progress", 0),
            completed=status_counts.get("completed", 0),
            cancelled=status_counts.get("cancelled", 0),
            failed=status_counts.get("failed", 0),
            blocked=status_counts.get("blocked", 0),
        )

    def clear_completed(self) -> int:
        """Remove all completed TODOs and return count removed."""
        initial_count = len(self.items)
        self.items = [item for item in self.items if item.status != TodoStatus.COMPLETED]
        return initial_count - len(self.items)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TodoList:
        """Create TodoList from dictionary."""
        items = [TodoItem.from_dict(item_data) for item_data in data.get("items", [])]
        created_at = (
            datetime.fromisoformat(data["created_at"])
            if isinstance(data.get("created_at"), str)
            else datetime.now()
        )

        return cls(
            name=data.get("name", "default"),
            description=data.get("description"),
            created_at=created_at,
            items=items,
        )
