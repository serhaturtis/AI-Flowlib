"""TODO management system for agents.

This module provides the TodoManager component that handles persistent
TODO list management, execution tracking, and agent activity integration.
"""

import asyncio
from datetime import timedelta
from typing import List, Dict, Optional, Protocol, Callable, Any
from asyncio import Queue, Task

from flowlib.agent.core.base import AgentComponent
from ..models import (
    TodoItem, TodoList, TodoPriority, TodoStatus, TodoStatusSummary,
    TodoExecutionContext, TodoExecutionResult
)


class ActivityStreamProtocol(Protocol):
    """Protocol for activity stream interface."""

    def todo_create(self, content: str, priority: str) -> None:
        """Log TODO creation."""
        ...

    def todo_update(self, todo_id: str, status: str, content: str, error: Optional[str] = None) -> None:
        """Log TODO update."""
        ...

    def todo_status(self, summary: Dict[str, int]) -> None:
        """Log TODO status summary."""
        ...


class TodoManager(AgentComponent):
    """Manages TODO lists for agent tasks.
    
    Provides persistent TODO management with execution tracking,
    dependency resolution, and activity stream integration.
    """
    
    def __init__(self, name: str = "TodoManager", activity_stream: Optional[ActivityStreamProtocol] = None) -> None:
        super().__init__(name)
        self.current_list: Optional[TodoList] = None
        self.saved_lists: Dict[str, TodoList] = {}
        self.execution_queue: Queue[TodoItem] = asyncio.Queue()
        self.executor_task: Optional[Task[None]] = None
        self.auto_execute = False
        self._activity_stream = activity_stream
    
    async def _initialize_impl(self) -> None:
        """Initialize the TODO manager."""
        self.current_list = TodoList()
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the TODO manager."""
        if self.executor_task:
            self.executor_task.cancel()
            try:
                await self.executor_task
            except asyncio.CancelledError:
                pass
    
    def create_list(self, name: str = "default", description: Optional[str] = None) -> TodoList:
        """Create a new TODO list."""
        todo_list = TodoList(name=name, description=description)
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
    
    def add_todo(
        self, 
        content: str, 
        priority: TodoPriority = TodoPriority.MEDIUM,
        assigned_tool: Optional[str] = None,
        execution_context: Optional[TodoExecutionContext] = None,
        depends_on: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
        estimated_duration: Optional[timedelta] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[str]:
        """Add a TODO to the current list.
        
        Args:
            content: Task description
            priority: Task priority level
            assigned_tool: Tool to use for execution
            execution_context: Structured context for execution
            depends_on: List of TODO IDs this depends on
            parent_id: Parent TODO ID if this is a subtask
            estimated_duration: Estimated completion time
            tags: Classification tags
            
        Returns:
            TODO ID if successful, None otherwise
        """
        if not self.current_list:
            self.current_list = TodoList()
        
        todo = TodoItem(
            content=content,
            priority=priority,
            assigned_tool=assigned_tool,
            execution_context=execution_context,
            depends_on=depends_on or [],
            parent_id=parent_id,
            estimated_duration=estimated_duration,
            tags=tags or []
        )
        
        self.current_list.add_todo(todo)
        
        # Stream TODO creation
        if self._activity_stream:
            priority_str = priority.value if isinstance(priority, TodoPriority) else str(priority)
            self._activity_stream.todo_create(content, priority_str)
        
        return todo.id
    
    def get_todo(self, todo_id: str) -> Optional[TodoItem]:
        """Get a TODO by ID from the current list."""
        if not self.current_list:
            return None
        
        return self.current_list.get_todo(todo_id)
    
    def update_todo(self, todo_id: str, updated_todo: TodoItem) -> bool:
        """Update a TODO in the current list."""
        if not self.current_list:
            return False
        
        return self.current_list.update_todo(todo_id, updated_todo)
    
    def mark_todo_completed(self, todo_id: str, result: Optional[TodoExecutionResult] = None) -> bool:
        """Mark a TODO as completed."""
        if not self.current_list:
            return False
        
        todo = self.current_list.get_todo(todo_id)
        if not todo:
            return False
        
        completed_todo = todo.mark_completed(result)
        self.current_list.update_todo(todo_id, completed_todo)
        
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
        self.current_list.update_todo(todo_id, failed_todo)
        
        # Stream TODO failure
        if self._activity_stream:
            self._activity_stream.todo_update(todo_id, "FAILED", content=todo.content, error=error)
        
        return True
    
    def mark_todo_in_progress(self, todo_id: str) -> bool:
        """Mark a TODO as in progress."""
        if not self.current_list:
            return False
        
        todo = self.current_list.get_todo(todo_id)
        if not todo:
            return False
        
        in_progress_todo = todo.mark_in_progress()
        self.current_list.update_todo(todo_id, in_progress_todo)
        
        # Stream TODO status change
        if self._activity_stream:
            self._activity_stream.todo_update(todo_id, "IN_PROGRESS", content=todo.content)
        
        return True
    
    def get_current_list(self) -> Optional[TodoList]:
        """Get the current TODO list."""
        return self.current_list
    
    def get_next_todo(self) -> Optional[TodoItem]:
        """Get the next TODO ready for execution."""
        if not self.current_list:
            return None
        
        ready_todos = self.current_list.get_ready_to_execute()
        return ready_todos[0] if ready_todos else None
    
    def get_todo_status_summary(self) -> TodoStatusSummary:
        """Get summary of TODO statuses."""
        if not self.current_list:
            return TodoStatusSummary(
                total=0,
                pending=0,
                in_progress=0,
                completed=0,
                cancelled=0,
                failed=0,
                blocked=0
            )
        
        return self.current_list.get_status_summary()
    
    def stream_todo_status(self) -> None:
        """Stream current TODO status if activity stream is available."""
        if self._activity_stream:
            status = self.get_todo_status_summary()
            self._activity_stream.todo_status({
                "total": status.total,
                "completed": status.completed,
                "in_progress": status.in_progress
            })
    
    async def execute_todo(self, todo_id: str, executor_callback: Callable[[TodoItem], Any]) -> bool:
        """Execute a specific TODO using the provided callback.
        
        Args:
            todo_id: ID of the TODO to execute
            executor_callback: Async function to execute the TODO
            
        Returns:
            True if execution succeeded, False otherwise
        """
        if not self.current_list:
            return False
        
        todo = self.current_list.get_todo(todo_id)
        if not todo:
            return False
        
        # Check if TODO is ready to execute
        completed_todos = [
            item.id for item in self.current_list.items 
            if item.status == TodoStatus.COMPLETED
        ]
        if not todo.is_ready_to_execute(completed_todos):
            return False
        
        # Mark as in progress
        in_progress_todo = todo.mark_in_progress()
        self.current_list.update_todo(todo_id, in_progress_todo)
        
        try:
            result = await executor_callback(in_progress_todo)
            completed_todo = in_progress_todo.mark_completed(result)
            self.current_list.update_todo(todo_id, completed_todo)
            return True
        except Exception as e:
            failed_todo = in_progress_todo.mark_failed(str(e))
            self.current_list.update_todo(todo_id, failed_todo)
            return False
    
    async def start_auto_execution(self, executor_callback: Callable[[TodoItem], Any], delay: float = 1.0) -> None:
        """Start automatic execution of TODOs.
        
        Args:
            executor_callback: Async function to execute TODOs
            delay: Delay between execution attempts
        """
        self.auto_execute = True
        self.executor_task = asyncio.create_task(
            self._auto_executor(executor_callback, delay)
        )
    
    def stop_auto_execution(self) -> None:
        """Stop automatic execution."""
        self.auto_execute = False
        if self.executor_task:
            self.executor_task.cancel()
    
    async def _auto_executor(self, executor_callback: Callable[[TodoItem], Any], delay: float) -> None:
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
        """Export current list to string format.
        
        Args:
            format: Export format ("json" or "markdown")
            
        Returns:
            Formatted string representation
        """
        if not self.current_list:
            return ""
        
        if format == "json":
            return self.current_list.to_json()
        elif format == "markdown":
            return self._export_markdown()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_markdown(self) -> str:
        """Export current list as markdown."""
        if not self.current_list or not self.current_list.items:
            return "# TODO List\n\nNo items in the current list."
        
        lines = [f"# TODO List: {self.current_list.name}\n"]
        
        if self.current_list.description:
            lines.append(f"{self.current_list.description}\n")
        
        # Group by status
        status_groups = {}
        for status in TodoStatus:
            status_groups[status] = self.current_list.get_by_status(status)
        
        for status, todos in status_groups.items():
            if not todos:
                continue
            
            status_name = status.value.replace('_', ' ').title()
            lines.append(f"## {status_name}\n")
            
            for todo in sorted(todos, key=lambda t: t.created_at):
                priority_emoji = {
                    TodoPriority.URGENT: "🚨",
                    TodoPriority.HIGH: "⚠️",
                    TodoPriority.MEDIUM: "📋",
                    TodoPriority.LOW: "📝"
                }
                
                emoji = priority_emoji.get(todo.priority, "📋")
                lines.append(f"- {emoji} {todo.content}")
                
                if todo.error_message:
                    lines.append(f"  ❌ Error: {todo.error_message}")
                
                if todo.notes:
                    lines.append(f"  📝 Notes: {todo.notes[-1]}")  # Show latest note
                
                lines.append("")
        
        return "\n".join(lines)