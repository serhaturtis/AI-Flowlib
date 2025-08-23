"""
Tests for TaskExecutionHandler functionality.

These tests cover the task execution handler's ability to coordinate planning,
flow execution, and reflection for complex tasks.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from pydantic import BaseModel, Field

from flowlib.agent.components.tasks.handler import TaskExecutionHandler
from flowlib.agent.models.state import AgentState
from flowlib.agent.components.planning import AgentPlanner
from flowlib.agent.components.reflection.base import AgentReflection
from flowlib.flows.models.results import FlowResult
from flowlib.core.context.context import Context


class MockInputModel(BaseModel):
    """Mock input model for testing flow execution."""
    task: str = Field(..., description="Task to execute")
    context: str = Field(default="", description="Additional context")


class MockPlanResult(BaseModel):
    """Mock plan result for testing."""
    selected_flow: str = Field(..., description="Selected flow name")
    reasoning: str = Field(..., description="Reasoning for selection")


class TestTaskExecutionHandlerInitialization:
    """Test task execution handler initialization."""

    def test_handler_creation(self):
        """Test creating task handler with required components."""
        planner = MagicMock(spec=AgentPlanner)
        reflection = MagicMock(spec=AgentReflection)
        
        handler = TaskExecutionHandler(
            planner=planner,
            reflection=reflection,
            memory_context="test_context"
        )
        
        assert handler.planner is planner
        assert handler.reflection is reflection
        assert handler.memory_context == "test_context"

    def test_handler_creation_with_defaults(self):
        """Test creating task handler with default memory context."""
        planner = MagicMock(spec=AgentPlanner)
        reflection = MagicMock(spec=AgentReflection)
        
        handler = TaskExecutionHandler(
            planner=planner,
            reflection=reflection
        )
        
        assert handler.memory_context == "agent"


class TestTaskExecutionHandlerExecution:
    """Test task execution handler execution flow."""

    @pytest.fixture
    def mock_planner(self):
        """Create mock planner for testing."""
        planner = AsyncMock(spec=AgentPlanner)
        return planner

    @pytest.fixture
    def mock_reflection(self):
        """Create mock reflection for testing."""
        reflection = AsyncMock(spec=AgentReflection)
        return reflection

    @pytest.fixture
    def mock_state(self):
        """Create mock agent state for testing."""
        state = AgentState(
            task_description="Test task execution",
            task_id="test_task_123"
        )
        state.add_user_message("Test user message")
        state.add_system_message("Test system response")
        return state

    @pytest.fixture
    def handler(self, mock_planner, mock_reflection):
        """Create task handler for testing."""
        return TaskExecutionHandler(
            planner=mock_planner,
            reflection=mock_reflection,
            memory_context="test_context"
        )

    @pytest.mark.asyncio
    async def test_execute_task_success(self, handler, mock_state, mock_planner, mock_reflection):
        """Test successful task execution flow."""
        # Mock plan result
        plan_result = MockPlanResult(
            selected_flow="test_flow",
            reasoning="Test flow is appropriate for this task"
        )
        mock_planner.plan.return_value = plan_result
        
        # Mock flow inputs
        flow_inputs = {"task": "test task", "context": "test context"}
        mock_planner.generate_inputs.return_value = flow_inputs
        
        # Mock flow
        mock_flow = MagicMock()
        mock_flow.name = "test_flow"
        mock_flow.get_pipeline_input_model.return_value = MockInputModel
        mock_flow.execute = AsyncMock(return_value=FlowResult(
            status="SUCCESS",
            data={"result": "Task completed successfully"}
        ))
        
        # Mock reflection result
        mock_reflection.reflect.return_value = {
            "success": True,
            "insights": "Task was executed well"
        }
        
        with patch('flowlib.agent.components.tasks.handler.flow_registry') as mock_registry:
            mock_registry.get_flow.return_value = mock_flow
            
            result = await handler.execute_task(mock_state)
        
        # Verify planner was called correctly
        mock_planner.plan.assert_called_once_with(context=mock_state)
        mock_planner.generate_inputs.assert_called_once_with(
            state=mock_state,
            flow_name="test_flow",
            planning_result=plan_result.model_dump(),
            memory_context_id=mock_state.task_id
        )
        
        # Verify flow execution
        mock_registry.get_flow.assert_called_once_with("test_flow")
        mock_flow.execute.assert_called_once()
        
        # Verify reflection
        mock_reflection.reflect.assert_called_once_with(
            state=mock_state,
            flow_name="test_flow",
            flow_inputs=flow_inputs,
            flow_result=mock_flow.execute.return_value,
            memory_context=mock_state.task_id
        )
        
        # Verify result
        assert result.status == "SUCCESS"
        assert "test_flow" in result.message
        assert result.flow_result is not None
        assert result.reflection is not None

    @pytest.mark.asyncio
    async def test_execute_task_no_flow_selected(self, handler, mock_state, mock_planner, mock_reflection):
        """Test task execution when no flow is selected."""
        # Mock plan result with no flow
        plan_result = MockPlanResult(
            selected_flow="none",
            reasoning="No appropriate flow found"
        )
        mock_planner.plan.return_value = plan_result
        
        result = await handler.execute_task(mock_state)
        
        assert result.status == "NO_FLOW_SELECTED"
        assert "No appropriate flow selected" in result.message
        
        # Verify planner was called but no further execution occurred
        mock_planner.plan.assert_called_once_with(context=mock_state)
        mock_planner.generate_inputs.assert_not_called()
        mock_reflection.reflect.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_task_flow_not_found(self, handler, mock_state, mock_planner, mock_reflection):
        """Test task execution when selected flow is not found in registry."""
        # Mock plan result
        plan_result = MockPlanResult(
            selected_flow="nonexistent_flow",
            reasoning="Selected non-existent flow"
        )
        mock_planner.plan.return_value = plan_result
        
        # Mock flow inputs
        flow_inputs = {"task": "test task"}
        mock_planner.generate_inputs.return_value = flow_inputs
        
        with patch('flowlib.agent.components.tasks.handler.flow_registry') as mock_registry:
            mock_registry.get_flow.return_value = None
            
            result = await handler.execute_task(mock_state)
        
        assert result.status == "FLOW_NOT_FOUND"
        assert "nonexistent_flow" in result.message
        
        # Verify planner was called but reflection was not
        mock_planner.plan.assert_called_once()
        mock_planner.generate_inputs.assert_called_once()
        mock_reflection.reflect.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_task_input_parsing_error(self, handler, mock_state, mock_planner, mock_reflection):
        """Test task execution when input parsing fails."""
        # Mock plan result
        plan_result = MockPlanResult(
            selected_flow="test_flow",
            reasoning="Test flow selected"
        )
        mock_planner.plan.return_value = plan_result
        
        # Mock invalid flow inputs (missing required field)
        flow_inputs = {"context": "test context"}  # Missing required 'task' field
        mock_planner.generate_inputs.return_value = flow_inputs
        
        # Mock flow
        mock_flow = MagicMock()
        mock_flow.name = "test_flow"
        mock_flow.get_pipeline_input_model.return_value = MockInputModel
        
        with patch('flowlib.agent.components.tasks.handler.flow_registry') as mock_registry:
            mock_registry.get_flow.return_value = mock_flow
            
            result = await handler.execute_task(mock_state)
        
        assert result.status == "INPUT_PARSING_ERROR"
        assert "Failed to parse inputs" in result.message
        
        # Verify flow execution and reflection were not called
        mock_flow.execute.assert_not_called()
        mock_reflection.reflect.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_task_invalid_input_model(self, handler, mock_state, mock_planner, mock_reflection):
        """Test task execution when flow has invalid input model."""
        # Mock plan result
        plan_result = MockPlanResult(
            selected_flow="test_flow",
            reasoning="Test flow selected"
        )
        mock_planner.plan.return_value = plan_result
        
        # Mock flow inputs
        flow_inputs = {"task": "test task"}
        mock_planner.generate_inputs.return_value = flow_inputs
        
        # Mock flow with invalid input model
        mock_flow = MagicMock()
        mock_flow.name = "test_flow"
        mock_flow.get_pipeline_input_model.return_value = None  # Invalid model
        
        with patch('flowlib.agent.components.tasks.handler.flow_registry') as mock_registry:
            mock_registry.get_flow.return_value = mock_flow
            
            result = await handler.execute_task(mock_state)
        
        assert result.status == "INPUT_PARSING_ERROR"
        assert "Failed to parse inputs" in result.message

    @pytest.mark.asyncio
    async def test_execute_task_with_pydantic_flow_inputs(self, handler, mock_state, mock_planner, mock_reflection):
        """Test task execution when planner returns Pydantic model as flow inputs."""
        # Mock plan result
        plan_result = MockPlanResult(
            selected_flow="test_flow",
            reasoning="Test flow selected"
        )
        mock_planner.plan.return_value = plan_result
        
        # Mock flow inputs as Pydantic model
        flow_inputs = MockInputModel(task="test task", context="test context")
        mock_planner.generate_inputs.return_value = flow_inputs
        
        # Mock flow
        mock_flow = MagicMock()
        mock_flow.name = "test_flow"
        mock_flow.get_pipeline_input_model.return_value = MockInputModel
        mock_flow.execute = AsyncMock(return_value=FlowResult(
            status="SUCCESS",
            data={"result": "Task completed"}
        ))
        
        # Mock reflection result
        mock_reflection.reflect.return_value = {"success": True}
        
        with patch('flowlib.agent.components.tasks.handler.flow_registry') as mock_registry:
            mock_registry.get_flow.return_value = mock_flow
            
            result = await handler.execute_task(mock_state)
        
        assert result.status == "SUCCESS"
        
        # Verify reflection was called with dict form of inputs
        mock_reflection.reflect.assert_called_once()
        call_args = mock_reflection.reflect.call_args
        assert isinstance(call_args.kwargs["flow_inputs"], dict)

    @pytest.mark.asyncio
    async def test_execute_task_context_creation(self, handler, mock_state, mock_planner, mock_reflection):
        """Test that Context object is created correctly for flow execution."""
        # Mock plan result
        plan_result = MockPlanResult(
            selected_flow="test_flow",
            reasoning="Test flow selected"
        )
        mock_planner.plan.return_value = plan_result
        
        # Mock flow inputs
        flow_inputs = {"task": "test task", "context": "test context"}
        mock_planner.generate_inputs.return_value = flow_inputs
        
        # Mock flow
        mock_flow = MagicMock()
        mock_flow.name = "test_flow"
        mock_flow.get_pipeline_input_model.return_value = MockInputModel
        mock_flow.execute = AsyncMock(return_value=FlowResult(status="SUCCESS", data={}))
        
        # Mock reflection
        mock_reflection.reflect.return_value = {}
        
        with patch('flowlib.agent.components.tasks.handler.flow_registry') as mock_registry:
            mock_registry.get_flow.return_value = mock_flow
            
            await handler.execute_task(mock_state)
        
        # Verify flow.execute was called with Context containing parsed input
        mock_flow.execute.assert_called_once()
        call_args = mock_flow.execute.call_args
        context_arg = call_args[0][0]
        
        assert isinstance(context_arg, Context)
        assert isinstance(context_arg.data, dict)
        assert context_arg.data["task"] == "test task"
        assert context_arg.data["context"] == "test context"


class TestTaskExecutionHandlerErrorHandling:
    """Test error handling in task execution handler."""

    @pytest.fixture
    def handler(self):
        """Create task handler for error testing."""
        planner = AsyncMock(spec=AgentPlanner)
        reflection = AsyncMock(spec=AgentReflection)
        return TaskExecutionHandler(planner=planner, reflection=reflection)

    @pytest.fixture
    def mock_state(self):
        """Create mock state for error testing."""
        state = AgentState(
            task_description="Error handling test",
            task_id="error_test"
        )
        state.add_user_message("Error test")
        return state

    @pytest.mark.asyncio
    async def test_execute_task_planner_exception(self, handler, mock_state):
        """Test task execution when planner raises exception."""
        handler.planner.plan.side_effect = Exception("Planner error")
        
        with pytest.raises(Exception, match="Planner error"):
            await handler.execute_task(mock_state)

    @pytest.mark.asyncio
    async def test_execute_task_flow_execution_exception(self, handler, mock_state):
        """Test task execution when flow execution raises exception."""
        # Mock successful planning
        plan_result = MockPlanResult(
            selected_flow="test_flow",
            reasoning="Test flow selected"
        )
        handler.planner.plan.return_value = plan_result
        handler.planner.generate_inputs.return_value = {"task": "test"}
        
        # Mock flow that raises exception
        mock_flow = MagicMock()
        mock_flow.name = "test_flow"
        mock_flow.get_pipeline_input_model.return_value = MockInputModel
        mock_flow.execute = AsyncMock(side_effect=Exception("Flow execution error"))
        
        with patch('flowlib.agent.components.tasks.handler.flow_registry') as mock_registry:
            mock_registry.get_flow.return_value = mock_flow
            
            with pytest.raises(Exception, match="Flow execution error"):
                await handler.execute_task(mock_state)

    @pytest.mark.asyncio
    async def test_execute_task_reflection_exception(self, handler, mock_state):
        """Test task execution when reflection raises exception."""
        # Mock successful planning and flow execution
        plan_result = MockPlanResult(
            selected_flow="test_flow",
            reasoning="Test flow selected"
        )
        handler.planner.plan.return_value = plan_result
        handler.planner.generate_inputs.return_value = {"task": "test"}
        
        mock_flow = MagicMock()
        mock_flow.name = "test_flow"
        mock_flow.get_pipeline_input_model.return_value = MockInputModel
        mock_flow.execute = AsyncMock(return_value=FlowResult(status="SUCCESS", data={}))
        
        # Mock reflection that raises exception
        handler.reflection.reflect.side_effect = Exception("Reflection error")
        
        with patch('flowlib.agent.components.tasks.handler.flow_registry') as mock_registry:
            mock_registry.get_flow.return_value = mock_flow
            
            with pytest.raises(Exception, match="Reflection error"):
                await handler.execute_task(mock_state)