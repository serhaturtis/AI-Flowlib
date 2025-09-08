"""Tests for agent planning todo system."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from flowlib.agent.components.task_decomposition.todo import (
    TodoStatus,
    TodoPriority,
    TodoItem,
    TodoList
)


class TestTodoStatus:
    """Test TodoStatus enum."""
    
    def test_todo_status_values(self):
        """Test TodoStatus enum values."""
        assert TodoStatus.PENDING.value == "pending"
        assert TodoStatus.IN_PROGRESS.value == "in_progress"
        assert TodoStatus.COMPLETED.value == "completed"
        assert TodoStatus.CANCELLED.value == "cancelled"
        assert TodoStatus.FAILED.value == "failed"
        assert TodoStatus.BLOCKED.value == "blocked"
    
    def test_todo_status_membership(self):
        """Test TodoStatus membership."""
        all_statuses = [
            TodoStatus.PENDING,
            TodoStatus.IN_PROGRESS,
            TodoStatus.COMPLETED,
            TodoStatus.CANCELLED,
            TodoStatus.FAILED,
            TodoStatus.BLOCKED
        ]
        
        assert len(set(all_statuses)) == 6  # All unique
        
        for status in all_statuses:
            assert isinstance(status, TodoStatus)


class TestTodoPriority:
    """Test TodoPriority enum."""
    
    def test_todo_priority_values(self):
        """Test TodoPriority enum values."""
        assert TodoPriority.LOW.value == "low"
        assert TodoPriority.MEDIUM.value == "medium"
        assert TodoPriority.HIGH.value == "high"
        assert TodoPriority.URGENT.value == "urgent"
    
    def test_todo_priority_membership(self):
        """Test TodoPriority membership."""
        all_priorities = [
            TodoPriority.LOW,
            TodoPriority.MEDIUM,
            TodoPriority.HIGH,
            TodoPriority.URGENT
        ]
        
        assert len(set(all_priorities)) == 4  # All unique
        
        for priority in all_priorities:
            assert isinstance(priority, TodoPriority)


class TestTodoItem:
    """Test TodoItem model."""
    
    def test_todo_item_creation_minimal(self):
        """Test creating TodoItem with minimal required fields."""
        todo = TodoItem(content="Test task")
        
        assert todo.content == "Test task"
        assert todo.status == TodoStatus.PENDING
        assert todo.priority == TodoPriority.MEDIUM
        assert todo.id is not None
        assert isinstance(todo.created_at, datetime)
        assert todo.updated_at is None
        assert todo.started_at is None
        assert todo.completed_at is None
    
    def test_todo_item_creation_with_options(self):
        """Test creating TodoItem with optional fields."""
        custom_id = str(uuid4())
        todo = TodoItem(
            id=custom_id,
            content="Complex task",
            status=TodoStatus.IN_PROGRESS,
            priority=TodoPriority.HIGH,
            assigned_tool="test_tool",
            tags=["important", "urgent"],
            notes=["Initial note"]
        )
        
        assert todo.id == custom_id
        assert todo.content == "Complex task"
        assert todo.status == TodoStatus.IN_PROGRESS
        assert todo.priority == TodoPriority.HIGH
        assert todo.assigned_tool == "test_tool"
        assert todo.tags == ["important", "urgent"]
        assert todo.notes == ["Initial note"]
    
    # Removed redundant default values test - handled by Pydantic
        """Test TodoItem default values."""
        todo = TodoItem(content="Test")
        
        assert todo.depends_on == []
        assert todo.blocks == []
        assert todo.parent_id is None
        assert todo.children == []
        assert todo.execution_context == {}
        assert todo.result is None
        assert todo.error_message is None
        assert todo.retry_count == 0
        assert todo.max_retries == 3
        assert todo.tags == []
        assert todo.notes == []
    
    def test_todo_item_mark_in_progress(self):
        """Test marking TodoItem as in progress."""
        todo = TodoItem(content="Test task")
        
        # Initially pending
        assert todo.status == TodoStatus.PENDING
        assert todo.started_at is None
        assert todo.updated_at is None
        
        # Mark in progress
        updated_todo = todo.mark_in_progress()
        
        assert updated_todo.status == TodoStatus.IN_PROGRESS
        assert updated_todo.started_at is not None
        assert updated_todo.updated_at is not None
        assert isinstance(updated_todo.started_at, datetime)
        assert isinstance(updated_todo.updated_at, datetime)
    
    def test_todo_item_mark_completed(self):
        """Test marking TodoItem as completed."""
        todo = TodoItem(content="Test task")
        in_progress_todo = todo.mark_in_progress()  # Start it first
        
        test_result = {"output": "success", "value": 42}
        completed_todo = in_progress_todo.mark_completed(result=test_result)
        
        assert completed_todo.status == TodoStatus.COMPLETED
        assert completed_todo.completed_at is not None
        assert completed_todo.updated_at is not None
        assert completed_todo.result == test_result
        assert completed_todo.actual_duration is not None
        assert isinstance(completed_todo.actual_duration, timedelta)
    
    def test_todo_item_mark_completed_without_result(self):
        """Test marking TodoItem as completed without result."""
        todo = TodoItem(content="Test task")
        in_progress_todo = todo.mark_in_progress()
        
        completed_todo = in_progress_todo.mark_completed()
        
        assert completed_todo.status == TodoStatus.COMPLETED
        assert completed_todo.completed_at is not None
        assert completed_todo.result is None
    
    def test_todo_item_mark_failed(self):
        """Test marking TodoItem as failed."""
        todo = TodoItem(content="Test task")
        
        error_message = "Task failed due to network error"
        failed_todo = todo.mark_failed(error_message)
        
        assert failed_todo.status == TodoStatus.FAILED
        assert failed_todo.error_message == error_message
        assert failed_todo.updated_at is not None
    
    def test_todo_item_mark_blocked(self):
        """Test marking TodoItem as blocked."""
        todo = TodoItem(content="Test task")
        
        block_reason = "Waiting for external dependency"
        blocked_todo = todo.mark_blocked(block_reason)
        
        assert blocked_todo.status == TodoStatus.BLOCKED
        assert blocked_todo.error_message == block_reason
        assert blocked_todo.updated_at is not None
    
    def test_todo_item_add_note(self):
        """Test adding notes to TodoItem."""
        todo = TodoItem(content="Test task")
        
        assert len(todo.notes) == 0
        
        todo_with_first_note = todo.add_note("First note")
        todo_with_both_notes = todo_with_first_note.add_note("Second note")
        
        assert len(todo_with_both_notes.notes) == 2
        assert "First note" in todo_with_both_notes.notes[0]
        assert "Second note" in todo_with_both_notes.notes[1]
        
        # Notes should include timestamps
        for note in todo_with_both_notes.notes:
            assert "[" in note and "]" in note  # Timestamp format
        
        assert todo_with_both_notes.updated_at is not None
    
    def test_todo_item_can_execute_no_dependencies(self):
        """Test can_execute with no dependencies."""
        todo = TodoItem(content="Independent task")
        
        assert todo.can_execute({})
        assert todo.can_execute({"other_id": TodoItem(content="Other")})
    
    def test_todo_item_can_execute_with_completed_dependencies(self):
        """Test can_execute with completed dependencies."""
        dep_todo = TodoItem(content="Dependency")
        completed_dep_todo = dep_todo.mark_completed()
        
        todo = TodoItem(content="Dependent task", depends_on=[dep_todo.id])
        all_todos = {dep_todo.id: completed_dep_todo, todo.id: todo}
        
        assert todo.can_execute(all_todos)
    
    def test_todo_item_can_execute_with_pending_dependencies(self):
        """Test can_execute with pending dependencies."""
        dep_todo = TodoItem(content="Dependency")  # Still pending
        
        todo = TodoItem(content="Dependent task", depends_on=[dep_todo.id])
        all_todos = {dep_todo.id: dep_todo, todo.id: todo}
        
        assert not todo.can_execute(all_todos)
    
    def test_todo_item_can_execute_non_pending_status(self):
        """Test can_execute with non-pending status."""
        todo = TodoItem(content="Test task")
        in_progress_todo = todo.mark_in_progress()
        
        assert not in_progress_todo.can_execute({})
        
        completed_todo = in_progress_todo.mark_completed()
        assert not completed_todo.can_execute({})
    
    def test_todo_item_can_execute_missing_dependency(self):
        """Test can_execute with missing dependency."""
        todo = TodoItem(content="Dependent task", depends_on=["missing_id"])
        
        # If dependency doesn't exist, it should be able to execute
        assert todo.can_execute({})


class TestTodoList:
    """Test TodoList model."""
    
    def test_todo_list_creation(self):
        """Test creating empty TodoList."""
        todo_list = TodoList()
        
        assert len(todo_list.items) == 0
        assert isinstance(todo_list.created_at, datetime)
        assert todo_list.updated_at is None
        assert todo_list.context == {}
    
    def test_todo_list_add_todo(self):
        """Test adding TODO to list."""
        todo_list = TodoList()
        
        todo_id = todo_list.add_todo("Test task")
        
        assert len(todo_list.items) == 1
        assert todo_id in todo_list.items
        assert todo_list.items[todo_id].content == "Test task"
        assert todo_list.updated_at is not None
    
    def test_todo_list_add_todo_with_priority(self):
        """Test adding TODO with specific priority."""
        todo_list = TodoList()
        
        todo_id = todo_list.add_todo("Urgent task", priority=TodoPriority.URGENT)
        
        assert todo_list.items[todo_id].priority == TodoPriority.URGENT
    
    def test_todo_list_add_todo_with_kwargs(self):
        """Test adding TODO with additional kwargs."""
        todo_list = TodoList()
        
        todo_id = todo_list.add_todo(
            "Tagged task",
            tags=["important"],
            assigned_tool="test_tool"
        )
        
        todo = todo_list.items[todo_id]
        assert todo.tags == ["important"]
        assert todo.assigned_tool == "test_tool"
    
    def test_todo_list_get_todo(self):
        """Test getting TODO by ID."""
        todo_list = TodoList()
        todo_id = todo_list.add_todo("Test task")
        
        todo = todo_list.get_todo(todo_id)
        assert todo is not None
        assert todo.content == "Test task"
        
        # Test getting non-existent TODO
        assert todo_list.get_todo("nonexistent") is None
    
    def test_todo_list_update_todo(self):
        """Test updating TODO."""
        todo_list = TodoList()
        todo_id = todo_list.add_todo("Original task")
        
        success = todo_list.update_todo(
            todo_id,
            content="Updated task",
            priority=TodoPriority.HIGH
        )
        
        assert success
        todo = todo_list.get_todo(todo_id)
        assert todo.content == "Updated task"
        assert todo.priority == TodoPriority.HIGH
        assert todo.updated_at is not None
    
    def test_todo_list_update_nonexistent_todo(self):
        """Test updating non-existent TODO."""
        todo_list = TodoList()
        
        success = todo_list.update_todo("nonexistent", content="New content")
        
        assert not success
    
    def test_todo_list_delete_todo(self):
        """Test deleting TODO."""
        todo_list = TodoList()
        todo_id = todo_list.add_todo("Task to delete")
        
        assert len(todo_list.items) == 1
        
        success = todo_list.delete_todo(todo_id)
        
        assert success
        assert len(todo_list.items) == 0
        assert todo_list.updated_at is not None
    
    def test_todo_list_delete_todo_with_dependencies(self):
        """Test deleting TODO with dependencies."""
        todo_list = TodoList()
        
        # Create dependency chain: A -> B -> C
        todo_a_id = todo_list.add_todo("Task A")
        todo_b_id = todo_list.add_todo("Task B", depends_on=[todo_a_id])
        todo_c_id = todo_list.add_todo("Task C", depends_on=[todo_b_id])
        
        # Set up blocking relationships
        todo_a = todo_list.items[todo_a_id]
        todo_b = todo_list.items[todo_b_id]
        todo_list.items[todo_a_id] = todo_a.model_copy(update={"blocks": [todo_b_id]})
        todo_list.items[todo_b_id] = todo_b.model_copy(update={"blocks": [todo_c_id]})
        
        # Delete B
        success = todo_list.delete_todo(todo_b_id)
        
        assert success
        assert len(todo_list.items) == 2
        
        # A should no longer block anything
        assert todo_list.items[todo_a_id].blocks == []
        
        # C should no longer depend on anything
        assert todo_list.items[todo_c_id].depends_on == []
    
    def test_todo_list_delete_nonexistent_todo(self):
        """Test deleting non-existent TODO."""
        todo_list = TodoList()
        
        success = todo_list.delete_todo("nonexistent")
        
        assert not success
    
    def test_todo_list_get_executable_todos(self):
        """Test getting executable TODOs."""
        todo_list = TodoList()
        
        # Add independent TODOs
        todo1_id = todo_list.add_todo("Independent 1")
        todo2_id = todo_list.add_todo("Independent 2")
        
        # Add dependent TODO
        todo3_id = todo_list.add_todo("Dependent", depends_on=[todo1_id])
        
        # Mark one as in progress
        todo2 = todo_list.items[todo2_id]
        todo_list.items[todo2_id] = todo2.mark_in_progress()
        
        executable = todo_list.get_executable_todos()
        
        # Should only get independent pending TODO
        assert len(executable) == 1
        assert executable[0].id == todo1_id
    
    def test_todo_list_get_next_todo_by_priority(self):
        """Test getting next TODO based on priority."""
        todo_list = TodoList()
        
        # Add TODOs with different priorities
        low_id = todo_list.add_todo("Low priority", priority=TodoPriority.LOW)
        medium_id = todo_list.add_todo("Medium priority", priority=TodoPriority.MEDIUM)
        high_id = todo_list.add_todo("High priority", priority=TodoPriority.HIGH)
        urgent_id = todo_list.add_todo("Urgent priority", priority=TodoPriority.URGENT)
        
        next_todo = todo_list.get_next_todo()
        
        # Should get urgent priority TODO
        assert next_todo is not None
        assert next_todo.id == urgent_id
    
    def test_todo_list_get_next_todo_by_creation_time(self):
        """Test getting next TODO based on creation time when priorities are equal."""
        todo_list = TodoList()
        
        # Add TODOs with same priority
        first_id = todo_list.add_todo("First task", priority=TodoPriority.HIGH)
        second_id = todo_list.add_todo("Second task", priority=TodoPriority.HIGH)
        
        next_todo = todo_list.get_next_todo()
        
        # Should get first created TODO
        assert next_todo is not None
        assert next_todo.id == first_id
    
    def test_todo_list_get_next_todo_no_executable(self):
        """Test getting next TODO when none are executable."""
        todo_list = TodoList()
        
        # Add TODO and mark as in progress
        todo_id = todo_list.add_todo("In progress task")
        todo = todo_list.items[todo_id]
        todo_list.items[todo_id] = todo.mark_in_progress()
        
        next_todo = todo_list.get_next_todo()
        
        assert next_todo is None
    
    def test_todo_list_get_todos_by_status(self):
        """Test getting TODOs by status."""
        todo_list = TodoList()
        
        # Add TODOs with different statuses
        pending_id = todo_list.add_todo("Pending task")
        in_progress_id = todo_list.add_todo("In progress task")
        completed_id = todo_list.add_todo("Completed task")
        
        in_progress_todo = todo_list.items[in_progress_id]
        completed_todo = todo_list.items[completed_id]
        todo_list.items[in_progress_id] = in_progress_todo.mark_in_progress()
        todo_list.items[completed_id] = completed_todo.mark_completed()
        
        pending_todos = todo_list.get_todos_by_status(TodoStatus.PENDING)
        in_progress_todos = todo_list.get_todos_by_status(TodoStatus.IN_PROGRESS)
        completed_todos = todo_list.get_todos_by_status(TodoStatus.COMPLETED)
        
        assert len(pending_todos) == 1
        assert pending_todos[0].id == pending_id
        
        assert len(in_progress_todos) == 1
        assert in_progress_todos[0].id == in_progress_id
        
        assert len(completed_todos) == 1
        assert completed_todos[0].id == completed_id


class TestTodoIntegration:
    """Test integration scenarios with TODO system."""
    
    def test_complex_dependency_chain(self):
        """Test complex dependency chain execution."""
        todo_list = TodoList()
        
        # Create dependency chain: A -> B -> C, D (independent)
        a_id = todo_list.add_todo("Task A", priority=TodoPriority.HIGH)
        b_id = todo_list.add_todo("Task B", depends_on=[a_id], priority=TodoPriority.HIGH)
        c_id = todo_list.add_todo("Task C", depends_on=[b_id], priority=TodoPriority.HIGH)
        d_id = todo_list.add_todo("Task D", priority=TodoPriority.LOW)
        
        # Initially only A and D should be executable
        executable = todo_list.get_executable_todos()
        executable_ids = {t.id for t in executable}
        assert executable_ids == {a_id, d_id}
        
        # Next should be A (higher priority)
        next_todo = todo_list.get_next_todo()
        assert next_todo.id == a_id
        
        # Complete A
        a_todo = todo_list.items[a_id]
        todo_list.items[a_id] = a_todo.mark_completed()
        
        # Now B and D should be executable
        executable = todo_list.get_executable_todos()
        executable_ids = {t.id for t in executable}
        assert executable_ids == {b_id, d_id}
        
        # Complete B
        b_todo = todo_list.items[b_id]
        todo_list.items[b_id] = b_todo.mark_completed()
        
        # Now C and D should be executable
        executable = todo_list.get_executable_todos()
        executable_ids = {t.id for t in executable}
        assert executable_ids == {c_id, d_id}
    
    def test_todo_lifecycle(self):
        """Test complete TODO lifecycle."""
        todo_list = TodoList()
        
        # Create TODO
        todo_id = todo_list.add_todo(
            "Complete lifecycle task",
            priority=TodoPriority.HIGH,
            tags=["lifecycle", "test"]
        )
        
        todo = todo_list.get_todo(todo_id)
        assert todo.status == TodoStatus.PENDING
        
        # Add note
        todo_with_note = todo.add_note("Starting work on this task")
        assert len(todo_with_note.notes) == 1
        
        # Mark in progress
        in_progress_todo = todo_with_note.mark_in_progress()
        assert in_progress_todo.status == TodoStatus.IN_PROGRESS
        assert in_progress_todo.started_at is not None
        
        # Add another note
        todo_with_notes = in_progress_todo.add_note("Making good progress")
        assert len(todo_with_notes.notes) == 2
        
        # Complete with result
        result = {"status": "success", "output": "Task completed successfully"}
        completed_todo = todo_with_notes.mark_completed(result=result)
        
        assert completed_todo.status == TodoStatus.COMPLETED
        assert completed_todo.completed_at is not None
        assert completed_todo.result == result
        assert completed_todo.actual_duration is not None
        
        # Update the todo in the list for the final check
        todo_list.items[todo_id] = completed_todo
        
        # Verify it's not executable anymore
        executable = todo_list.get_executable_todos()
        assert todo not in executable
    
    def test_todo_failure_and_retry(self):
        """Test TODO failure scenario."""
        todo_list = TodoList()
        
        todo_id = todo_list.add_todo("Task that might fail")
        todo = todo_list.get_todo(todo_id)
        
        # Start and fail
        in_progress_todo = todo.mark_in_progress()
        failed_todo = in_progress_todo.mark_failed("Network connection error")
        
        assert failed_todo.status == TodoStatus.FAILED
        assert failed_todo.error_message == "Network connection error"
        assert failed_todo.retry_count == 0  # Not incremented automatically
        
        # Manual retry (reset status)
        retried_todo = failed_todo.model_copy(update={
            "retry_count": failed_todo.retry_count + 1,
            "status": TodoStatus.PENDING,
            "error_message": None
        })
        todo_list.items[todo_id] = retried_todo
        
        assert retried_todo.retry_count == 1
        assert retried_todo.can_execute({})
    
    def test_todo_serialization(self):
        """Test TODO serialization and deserialization."""
        todo_list = TodoList()
        
        todo_id = todo_list.add_todo(
            "Serialization test",
            priority=TodoPriority.URGENT,
            tags=["serialize", "test"],
            assigned_tool="test_tool"
        )
        
        # Serialize
        data = todo_list.model_dump()
        
        # Deserialize
        restored_list = TodoList(**data)
        
        assert len(restored_list.items) == 1
        restored_todo = list(restored_list.items.values())[0]
        
        assert restored_todo.content == "Serialization test"
        assert restored_todo.priority == TodoPriority.URGENT
        assert restored_todo.tags == ["serialize", "test"]
        assert restored_todo.assigned_tool == "test_tool"