"""TODO management system for agents."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from uuid import uuid4
from pydantic import Field
from flowlib.core.models import StrictBaseModel

from ...core.base import AgentComponent
from ...models.state import AgentState


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
    """Individual TODO item."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(..., description="Description of the task")
    status: TodoStatus = Field(default=TodoStatus.PENDING)
    priority: TodoPriority = Field(default=TodoPriority.MEDIUM)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None
    actual_duration: Optional[timedelta] = None
    
    # Dependencies and relationships
    depends_on: List[str] = Field(default_factory=list, description="TODO IDs this depends on")
    blocks: List[str] = Field(default_factory=list, description="TODO IDs this blocks")
    parent_id: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    
    # Execution details
    assigned_tool: Optional[str] = None
    execution_context: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    
    def mark_in_progress(self) -> 'TodoItem':
        """Mark TODO as in progress.
        
        Returns:
            New TodoItem instance with updated status
        """
        now = datetime.now()
        return self.model_copy(update={
            "status": TodoStatus.IN_PROGRESS,
            "started_at": now,
            "updated_at": now
        })
    
    def mark_completed(self, result: Optional[Dict[str, Any]] = None) -> 'TodoItem':
        """Mark TODO as completed.
        
        Returns:
            New TodoItem instance with updated status
        """
        now = datetime.now()
        updates = {
            "status": TodoStatus.COMPLETED,
            "completed_at": now,
            "updated_at": now
        }
        if result:
            updates["result"] = result
        
        if self.started_at:
            updates["actual_duration"] = now - self.started_at
            
        return self.model_copy(update=updates)
    
    def mark_failed(self, error: str) -> 'TodoItem':
        """Mark TODO as failed.
        
        Returns:
            New TodoItem instance with updated status
        """
        return self.model_copy(update={
            "status": TodoStatus.FAILED,
            "error_message": error,
            "updated_at": datetime.now()
        })
    
    def mark_blocked(self, reason: str) -> 'TodoItem':
        """Mark TODO as blocked.
        
        Returns:
            New TodoItem instance with updated status
        """
        return self.model_copy(update={
            "status": TodoStatus.BLOCKED,
            "error_message": reason,
            "updated_at": datetime.now()
        })
    
    def add_note(self, note: str) -> 'TodoItem':
        """Add a note to the TODO.
        
        Returns:
            New TodoItem instance with added note
        """
        now = datetime.now()
        new_notes = self.notes + [f"[{now.isoformat()}] {note}"]
        return self.model_copy(update={
            "notes": new_notes,
            "updated_at": now
        })
    
    def can_execute(self, all_todos: Dict[str, 'TodoItem']) -> bool:
        """Check if this TODO can be executed (dependencies are met)."""
        if self.status != TodoStatus.PENDING:
            return False
        
        # Check dependencies
        for dep_id in self.depends_on:
            if dep_id in all_todos:
                dep_todo = all_todos[dep_id]
                if dep_todo.status != TodoStatus.COMPLETED:
                    return False
        
        return True


class TodoList(StrictBaseModel):
    """Collection of TODO items with management capabilities."""
    items: Dict[str, TodoItem] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    
    def add_todo(self, content: str, priority: TodoPriority = TodoPriority.MEDIUM, **kwargs) -> str:
        """Add a new TODO item."""
        todo = TodoItem(
            content=content,
            priority=priority,
            **kwargs
        )
        self.items[todo.id] = todo
        self.updated_at = datetime.now()
        return todo.id
    
    def get_todo(self, todo_id: str) -> Optional[TodoItem]:
        """Get a TODO by ID."""
        if todo_id not in self.items:
            return None
        return self.items[todo_id]
    
    def update_todo(self, todo_id: str, **updates) -> bool:
        """Update a TODO item."""
        if todo_id not in self.items:
            return False
        
        todo = self.items[todo_id]
        
        # Add updated_at to the updates
        updates["updated_at"] = datetime.now()
        
        # Create a new TodoItem with the updates
        updated_todo = todo.model_copy(update=updates)
        
        # Replace the old todo with the updated one
        self.items[todo_id] = updated_todo
        self.updated_at = datetime.now()
        return True
    
    def delete_todo(self, todo_id: str) -> bool:
        """Delete a TODO item."""
        if todo_id not in self.items:
            return False
        
        # Remove dependencies
        todo = self.items[todo_id]
        for dep_id in todo.depends_on:
            if dep_id in self.items:
                dep_todo = self.items[dep_id]
                new_blocks = [bid for bid in dep_todo.blocks if bid != todo_id]
                self.items[dep_id] = dep_todo.model_copy(update={"blocks": new_blocks})
        
        for blocked_id in todo.blocks:
            if blocked_id in self.items:
                blocked_todo = self.items[blocked_id]
                new_depends_on = [did for did in blocked_todo.depends_on if did != todo_id]
                self.items[blocked_id] = blocked_todo.model_copy(update={"depends_on": new_depends_on})
        
        del self.items[todo_id]
        self.updated_at = datetime.now()
        return True
    
    def get_executable_todos(self) -> List[TodoItem]:
        """Get TODOs that can be executed (dependencies met, status pending)."""
        return [
            todo for todo in self.items.values()
            if todo.can_execute(self.items)
        ]
    
    def get_next_todo(self) -> Optional[TodoItem]:
        """Get the next TODO to execute based on priority and dependencies."""
        executable = self.get_executable_todos()
        if not executable:
            return None
        
        # Sort by priority (urgent -> high -> medium -> low) then by creation time
        priority_order = {
            TodoPriority.URGENT: 0,
            TodoPriority.HIGH: 1,
            TodoPriority.MEDIUM: 2,
            TodoPriority.LOW: 3
        }
        
        return min(
            executable,
            key=lambda t: (priority_order[t.priority], t.created_at)
        )
    
    def get_todos_by_status(self, status: TodoStatus) -> List[TodoItem]:
        """Get TODOs by status."""
        return [todo for todo in self.items.values() if todo.status == status]
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get progress summary."""
        total = len(self.items)
        if total == 0:
            return {"total": 0, "progress": 0}
        
        completed = len(self.get_todos_by_status(TodoStatus.COMPLETED))
        failed = len(self.get_todos_by_status(TodoStatus.FAILED))
        in_progress = len(self.get_todos_by_status(TodoStatus.IN_PROGRESS))
        pending = len(self.get_todos_by_status(TodoStatus.PENDING))
        blocked = len(self.get_todos_by_status(TodoStatus.BLOCKED))
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "pending": pending,
            "blocked": blocked,
            "progress": completed / total if total > 0 else 0
        }


class TodoManager(AgentComponent):
    """Manages TODO lists for agent tasks."""
    
    def __init__(self, name: str = "TodoManager", activity_stream=None):
        super().__init__(name)
        self.current_list: Optional[TodoList] = None
        self.saved_lists: Dict[str, TodoList] = {}
        self.execution_queue = asyncio.Queue()
        self.executor_task: Optional[asyncio.Task] = None
        self.auto_execute = False
        self._activity_stream = activity_stream
    
    async def initialize(self):
        """Initialize the TODO manager."""
        await super().initialize()
        self.current_list = TodoList()
    
    async def shutdown(self):
        """Shutdown the TODO manager."""
        if self.executor_task:
            self.executor_task.cancel()
            try:
                await self.executor_task
            except asyncio.CancelledError:
                pass
        await super().shutdown()
    
    def create_list(self, context: Optional[Dict[str, Any]] = None) -> TodoList:
        """Create a new TODO list."""
        todo_list = TodoList(context=context or {})
        self.current_list = todo_list
        return todo_list
    
    def save_list(self, name: str, todo_list: Optional[TodoList] = None) -> bool:
        """Save a TODO list with a name."""
        list_to_save = todo_list or self.current_list
        if not list_to_save:
            return False
        
        self.saved_lists[name] = list_to_save
        return True
    
    def load_list(self, name: str) -> bool:
        """Load a saved TODO list."""
        if name not in self.saved_lists:
            return False
        
        self.current_list = self.saved_lists[name]
        return True
    
    def add_todo(self, content: str, **kwargs) -> Optional[str]:
        """Add a TODO to the current list."""
        if not self.current_list:
            self.current_list = TodoList()
        
        todo_id = self.current_list.add_todo(content, **kwargs)
        
        # Stream TODO creation
        if self._activity_stream and todo_id:
            priority = kwargs['priority'] if 'priority' in kwargs else TodoPriority.MEDIUM
            priority_str = priority.value if isinstance(priority, TodoPriority) else str(priority)
            self._activity_stream.todo_create(content, priority_str)
        
        return todo_id
    
    def update_todo(self, todo_id: str, **updates) -> bool:
        """Update a TODO in the current list."""
        if not self.current_list:
            return False
        
        return self.current_list.update_todo(todo_id, **updates)
    
    def mark_todo_completed(self, todo_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """Mark a TODO as completed."""
        if not self.current_list:
            return False
        
        todo = self.current_list.get_todo(todo_id)
        if not todo:
            return False
        
        completed_todo = todo.mark_completed(result)
        self.current_list.items[todo_id] = completed_todo
        
        # Stream TODO completion
        if self._activity_stream:
            self._activity_stream.todo_update(todo_id, "COMPLETED", content=todo.content)
        
        return True
    
    def mark_todo_failed(self, todo_id: str, error: str) -> bool:
        """Mark a TODO as failed."""
        if not self.current_list:
            return False
        
        todo = self.current_list.get_todo(todo_id)
        if not todo:
            return False
        
        failed_todo = todo.mark_failed(error)
        self.current_list.items[todo_id] = failed_todo
        
        # Stream TODO failure
        if self._activity_stream:
            self._activity_stream.todo_update(todo_id, "FAILED", content=todo.content, error=error)
        
        return True
    
    def get_current_list(self) -> Optional[TodoList]:
        """Get the current TODO list."""
        return self.current_list
    
    def get_next_todo(self) -> Optional[TodoItem]:
        """Get the next TODO to execute."""
        if not self.current_list:
            return None
        
        return self.current_list.get_next_todo()
    
    def get_todo_status_summary(self) -> TodoStatusSummary:
        """Get summary of TODO statuses."""
        if not self.current_list:
            return TodoStatusSummary()
        
        # Count items by status
        counts = {
            "total": 0,
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "cancelled": 0,
            "failed": 0,
            "blocked": 0
        }
        
        for todo in self.current_list.items.values():
            counts["total"] += 1
            status_key = todo.status.value
            if status_key in counts:
                counts[status_key] += 1
        
        return TodoStatusSummary(**counts)
    
    def stream_todo_status(self):
        """Stream current TODO status if activity stream is available."""
        if self._activity_stream:
            status = self.get_todo_status_summary()
            self._activity_stream.todo_status(
                total=status.total,
                completed=status.completed,
                in_progress=status.in_progress
            )
    
    async def execute_todo(self, todo_id: str, executor_callback) -> bool:
        """Execute a specific TODO using the provided callback."""
        if not self.current_list:
            return False
        
        todo = self.current_list.get_todo(todo_id)
        if not todo:
            return False
        
        if not todo.can_execute(self.current_list.items):
            return False
        
        in_progress_todo = todo.mark_in_progress()
        self.current_list.items[todo_id] = in_progress_todo
        
        try:
            result = await executor_callback(in_progress_todo)
            completed_todo = in_progress_todo.mark_completed(result)
            self.current_list.items[todo_id] = completed_todo
            return True
        except Exception as e:
            failed_todo = in_progress_todo.mark_failed(str(e))
            self.current_list.items[todo_id] = failed_todo
            return False
    
    async def start_auto_execution(self, executor_callback, delay: float = 1.0):
        """Start automatic execution of TODOs."""
        self.auto_execute = True
        self.executor_task = asyncio.create_task(
            self._auto_executor(executor_callback, delay)
        )
    
    def stop_auto_execution(self):
        """Stop automatic execution."""
        self.auto_execute = False
        if self.executor_task:
            self.executor_task.cancel()
    
    async def _auto_executor(self, executor_callback, delay: float):
        """Auto-executor loop."""
        while self.auto_execute:
            try:
                next_todo = self.get_next_todo()
                if next_todo:
                    await self.execute_todo(next_todo.id, executor_callback)
                else:
                    await asyncio.sleep(delay)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                print(f"Error in auto-executor: {e}")
                await asyncio.sleep(delay)
    
    def export_list(self, format: str = "json") -> str:
        """Export current list to string format."""
        if not self.current_list:
            return ""
        
        if format == "json":
            return self.current_list.model_dump_json(indent=2)
        elif format == "markdown":
            return self._export_markdown()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_markdown(self) -> str:
        """Export current list as markdown."""
        if not self.current_list:
            return ""
        
        lines = ["# TODO List", ""]
        
        # Progress summary
        summary = self.current_list.get_progress_summary()
        progress_pct = int(summary["progress"] * 100)
        lines.extend([
            f"**Progress:** {summary['completed']}/{summary['total']} ({progress_pct}%)",
            ""
        ])
        
        # Group by status
        for status in TodoStatus:
            todos = self.current_list.get_todos_by_status(status)
            if not todos:
                continue
            
            lines.append(f"## {status.value.title().replace('_', ' ')}")
            lines.append("")
            
            for todo in sorted(todos, key=lambda t: t.created_at):
                priority_emoji = {
                    TodoPriority.URGENT: "🚨",
                    TodoPriority.HIGH: "⚠️",
                    TodoPriority.MEDIUM: "📋",
                    TodoPriority.LOW: "📝"
                }
                
                emoji = priority_emoji[todo.priority] if todo.priority in priority_emoji else "📋"
                lines.append(f"- {emoji} {todo.content}")
                
                if todo.error_message:
                    lines.append(f"  ❌ Error: {todo.error_message}")
                
                if todo.notes:
                    lines.append(f"  📝 Notes: {todo.notes[-1]}")  # Show latest note
                
                lines.append("")
        
        return "\n".join(lines)