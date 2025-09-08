"""Tests for Todo Generation models."""

import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError

from flowlib.agent.components.task_decomposition.todo_generation.models import (
    TodoGenerationInput,
    TodoGenerationOutput
)
from flowlib.agent.components.task_decomposition.models import Plan, PlanStep
from flowlib.agent.components.task_decomposition.todo import TodoItem, TodoPriority, TodoStatus


class TestTodoGenerationInput:
    """Test the TodoGenerationInput model."""

    def test_valid_input_minimal(self):
        """Test creating valid input with minimal fields."""
        plan = Plan(
            task_description="Test task",
            steps=[
                PlanStep(
                    flow_name="test-flow",
                    step_intent="Test step",
                    rationale="For testing"
                )
            ]
        )
        
        input_data = TodoGenerationInput(
            plan=plan,
            task_description="Test task"
        )
        
        assert input_data.plan == plan
        assert input_data.task_description == "Test task"
        assert input_data.context == {}
        assert input_data.default_priority == TodoPriority.MEDIUM

    def test_valid_input_full(self):
        """Test creating valid input with all fields."""
        plan = Plan(
            task_description="Complex task",
            steps=[
                PlanStep(
                    flow_name="setup",
                    step_intent="Initialize",
                    rationale="Setup needed"
                ),
                PlanStep(
                    flow_name="execute",
                    step_intent="Run main task",
                    rationale="Core functionality"
                )
            ]
        )
        
        input_data = TodoGenerationInput(
            plan=plan,
            task_description="Complex task",
            context={"framework": "React", "deadline": "2024-01-01"},
            default_priority=TodoPriority.HIGH
        )
        
        assert input_data.plan == plan
        assert input_data.task_description == "Complex task"
        assert input_data.context == {"framework": "React", "deadline": "2024-01-01"}
        assert input_data.default_priority == TodoPriority.HIGH

    def test_missing_required_fields(self):
        """Test that required fields must be provided."""
        # Missing plan
        with pytest.raises(ValidationError) as exc_info:
            TodoGenerationInput(task_description="Test")
        
        errors = exc_info.value.errors()
        assert any(error['loc'] == ('plan',) for error in errors)
        
        # Missing task_description
        plan = Plan(task_description="Test", steps=[])
        with pytest.raises(ValidationError) as exc_info:
            TodoGenerationInput(plan=plan)
        
        errors = exc_info.value.errors()
        assert any(error['loc'] == ('task_description',) for error in errors)

    def test_empty_context(self):
        """Test that empty context defaults to empty dict."""
        plan = Plan(task_description="Test", steps=[])
        
        input_data = TodoGenerationInput(
            plan=plan,
            task_description="Test"
        )
        
        assert input_data.context == {}

    def test_default_priority_validation(self):
        """Test that default priority accepts valid enum values."""
        plan = Plan(task_description="Test", steps=[])
        
        for priority in TodoPriority:
            input_data = TodoGenerationInput(
                plan=plan,
                task_description="Test",
                default_priority=priority
            )
            assert input_data.default_priority == priority


class TestTodoGenerationOutput:
    """Test the TodoGenerationOutput model."""

    def test_valid_output_minimal(self):
        """Test creating valid output with minimal fields."""
        todos = [
            TodoItem(
                content="Test todo",
                priority=TodoPriority.MEDIUM,
                status=TodoStatus.PENDING
            )
        ]
        
        output = TodoGenerationOutput(
            todos=todos,
            execution_strategy="sequential",
            reasoning="Simple task requires sequential execution"
        )
        
        assert output.todos == todos
        assert output.execution_strategy == "sequential"
        assert output.reasoning == "Simple task requires sequential execution"
        assert output.estimated_duration is None
        assert output.dependency_map == {}

    def test_valid_output_full(self):
        """Test creating valid output with all fields."""
        todos = [
            TodoItem(
                id="todo-1",
                content="First todo",
                priority=TodoPriority.HIGH,
                status=TodoStatus.PENDING
            ),
            TodoItem(
                id="todo-2",
                content="Second todo",
                priority=TodoPriority.MEDIUM,
                status=TodoStatus.PENDING,
                depends_on=["todo-1"]
            )
        ]
        
        output = TodoGenerationOutput(
            todos=todos,
            execution_strategy="hybrid with parallel opportunities",
            estimated_duration="2-3 hours",
            dependency_map={"todo-2": ["todo-1"]},
            reasoning="Complex task benefits from hybrid approach"
        )
        
        assert len(output.todos) == 2
        assert output.execution_strategy == "hybrid with parallel opportunities"
        assert output.estimated_duration == "2-3 hours"
        assert output.dependency_map == {"todo-2": ["todo-1"]}
        assert output.reasoning == "Complex task benefits from hybrid approach"

    def test_missing_required_fields(self):
        """Test that required fields must be provided."""
        # Missing todos
        with pytest.raises(ValidationError) as exc_info:
            TodoGenerationOutput(
                execution_strategy="sequential",
                reasoning="Test"
            )
        
        errors = exc_info.value.errors()
        assert any(error['loc'] == ('todos',) for error in errors)
        
        # Missing execution_strategy
        with pytest.raises(ValidationError) as exc_info:
            TodoGenerationOutput(
                todos=[],
                reasoning="Test"
            )
        
        errors = exc_info.value.errors()
        assert any(error['loc'] == ('execution_strategy',) for error in errors)
        
        # Missing reasoning
        with pytest.raises(ValidationError) as exc_info:
            TodoGenerationOutput(
                todos=[],
                execution_strategy="sequential"
            )
        
        errors = exc_info.value.errors()
        assert any(error['loc'] == ('reasoning',) for error in errors)

    def test_empty_todos_list(self):
        """Test that empty todos list is valid."""
        output = TodoGenerationOutput(
            todos=[],
            execution_strategy="none",
            reasoning="No todos needed"
        )
        
        assert output.todos == []

    def test_get_user_display_no_todos(self):
        """Test user display when no todos are generated."""
        output = TodoGenerationOutput(
            todos=[],
            execution_strategy="none",
            reasoning="No todos needed"
        )
        
        display = output.get_user_display()
        assert display == "📝 No TODOs generated for this task."

    def test_get_user_display_single_todo(self):
        """Test user display with single todo."""
        todos = [
            TodoItem(
                content="Single task",
                priority=TodoPriority.HIGH,
                status=TodoStatus.PENDING
            )
        ]
        
        output = TodoGenerationOutput(
            todos=todos,
            execution_strategy="immediate",
            reasoning="Simple single task"
        )
        
        display = output.get_user_display()
        assert "Generated 1 TODO" in display  # Not "TODOs"
        assert "**Execution Strategy:**" in display
        assert "immediate" in display
        assert "**Reasoning:**" in display
        assert "Simple single task" in display

    def test_get_user_display_multiple_todos(self):
        """Test user display with multiple todos."""
        todos = [
            TodoItem(
                id="todo-1",
                content="First task",
                priority=TodoPriority.HIGH,
                status=TodoStatus.PENDING
            ),
            TodoItem(
                id="todo-2", 
                content="Second task",
                priority=TodoPriority.MEDIUM,
                status=TodoStatus.PENDING,
                depends_on=["todo-1"]
            ),
            TodoItem(
                id="todo-3",
                content="Third task",
                priority=TodoPriority.LOW,
                status=TodoStatus.PENDING
            )
        ]
        
        output = TodoGenerationOutput(
            todos=todos,
            execution_strategy="hybrid approach",
            estimated_duration="4-5 hours",
            dependency_map={"todo-2": ["todo-1"]},
            reasoning="Multi-step task with dependencies"
        )
        
        display = output.get_user_display()
        assert "Generated 3 TODOs" in display
        assert "with 1 dependencies" in display  # todo-2 depends on todo-1
        assert "Priorities: 1 high, 1 medium, 1 low" in display
        assert "Estimated duration: 4-5 hours" in display
        assert "**Execution Strategy:**" in display
        assert "hybrid approach" in display
        assert "**Reasoning:**" in display
        assert "Multi-step task with dependencies" in display

    def test_get_user_display_mixed_priorities(self):
        """Test user display with mixed priority breakdown."""
        todos = [
            TodoItem(content="Task 1", priority=TodoPriority.URGENT, status=TodoStatus.PENDING),
            TodoItem(content="Task 2", priority=TodoPriority.URGENT, status=TodoStatus.PENDING),
            TodoItem(content="Task 3", priority=TodoPriority.HIGH, status=TodoStatus.PENDING),
            TodoItem(content="Task 4", priority=TodoPriority.MEDIUM, status=TodoStatus.PENDING),
            TodoItem(content="Task 5", priority=TodoPriority.LOW, status=TodoStatus.PENDING),
            TodoItem(content="Task 6", priority=TodoPriority.LOW, status=TodoStatus.PENDING),
        ]
        
        output = TodoGenerationOutput(
            todos=todos,
            execution_strategy="prioritized",
            reasoning="Mixed priority tasks"
        )
        
        display = output.get_user_display()
        assert "Generated 6 TODOs" in display
        assert "2 urgent, 1 high, 1 medium, 2 low" in display

    def test_get_user_display_with_dependencies(self):
        """Test user display counting dependencies correctly."""
        todos = [
            TodoItem(
                id="todo-1",
                content="Independent task",
                priority=TodoPriority.HIGH,
                status=TodoStatus.PENDING
            ),
            TodoItem(
                id="todo-2",
                content="Dependent task 1",
                priority=TodoPriority.MEDIUM,
                status=TodoStatus.PENDING,
                depends_on=["todo-1"]
            ),
            TodoItem(
                id="todo-3",
                content="Dependent task 2", 
                priority=TodoPriority.MEDIUM,
                status=TodoStatus.PENDING,
                depends_on=["todo-1", "todo-2"]
            )
        ]
        
        output = TodoGenerationOutput(
            todos=todos,
            execution_strategy="sequential",
            reasoning="Cascading dependencies"
        )
        
        display = output.get_user_display()
        assert "Generated 3 TODOs" in display
        assert "with 2 dependencies" in display  # todo-2 and todo-3 have dependencies

    def test_get_user_display_single_priority(self):
        """Test user display when all todos have same priority."""
        todos = [
            TodoItem(content="Task 1", priority=TodoPriority.MEDIUM, status=TodoStatus.PENDING),
            TodoItem(content="Task 2", priority=TodoPriority.MEDIUM, status=TodoStatus.PENDING),
            TodoItem(content="Task 3", priority=TodoPriority.MEDIUM, status=TodoStatus.PENDING),
        ]
        
        output = TodoGenerationOutput(
            todos=todos,
            execution_strategy="parallel",
            reasoning="Same priority tasks"
        )
        
        display = output.get_user_display()
        assert "Generated 3 TODOs" in display
        # Should not show priority breakdown when all are the same
        assert "Priorities:" not in display

    def test_get_user_display_no_estimated_duration(self):
        """Test user display without estimated duration."""
        todos = [
            TodoItem(content="Task", priority=TodoPriority.MEDIUM, status=TodoStatus.PENDING)
        ]
        
        output = TodoGenerationOutput(
            todos=todos,
            execution_strategy="immediate",
            reasoning="Quick task"
        )
        
        display = output.get_user_display()
        assert "Estimated duration:" not in display

    def test_get_user_display_no_reasoning(self):
        """Test user display without reasoning."""
        todos = [
            TodoItem(content="Task", priority=TodoPriority.MEDIUM, status=TodoStatus.PENDING)
        ]
        
        output = TodoGenerationOutput(
            todos=todos,
            execution_strategy="immediate",
            reasoning=""
        )
        
        display = output.get_user_display()
        assert "**Reasoning:**" not in display

    def test_dependency_map_validation(self):
        """Test that dependency_map accepts various formats."""
        todos = [TodoItem(content="Test", priority=TodoPriority.MEDIUM, status=TodoStatus.PENDING)]
        
        # Empty dependency map
        output = TodoGenerationOutput(
            todos=todos,
            execution_strategy="sequential",
            reasoning="Test"
        )
        assert output.dependency_map == {}
        
        # Complex dependency map
        dependency_map = {
            "todo-1": ["todo-2", "todo-3"],
            "todo-4": ["todo-1"],
            "todo-5": []
        }
        
        output = TodoGenerationOutput(
            todos=todos,
            execution_strategy="sequential",
            reasoning="Test",
            dependency_map=dependency_map
        )
        assert output.dependency_map == dependency_map