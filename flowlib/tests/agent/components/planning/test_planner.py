"""
Tests for AgentPlanner functionality.

These tests cover the unified planner's ability to generate plans and TODOs,
handle complex tasks, and integrate with the agent's planning system.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from flowlib.agent.components.planning.planner import AgentPlanner
from flowlib.agent.components.planning.models import PlanningResult, Plan, PlanStep
from flowlib.agent.components.planning.todo import TodoItem, TodoPriority, TodoStatus
from flowlib.agent.models.state import AgentState
from flowlib.agent.models.config import AgentConfig, PlannerConfig


class TestAgentPlannerInitialization:
    """Test unified planner initialization."""

    def test_planner_creation(self):
        """Test creating unified planner."""
        config = PlannerConfig()
        planner = AgentPlanner(config=config)
        
        assert planner is not None
        assert planner.todo_generation_flow is not None
        assert planner.complexity_threshold == 3

    def test_planner_creation_with_activity_stream(self):
        """Test creating planner with activity stream."""
        config = PlannerConfig()
        mock_activity_stream = MagicMock()
        
        planner = AgentPlanner(
            config=config,
            name="test_planner",
            activity_stream=mock_activity_stream
        )
        
        assert planner._activity_stream == mock_activity_stream

    @pytest.mark.asyncio
    async def test_planner_initialization(self):
        """Test planner initialization."""
        config = PlannerConfig()
        planner = AgentPlanner(config=config)
        
        # Test that the planner starts uninitialized
        assert not planner.initialized
        
        # Mock the provider registry
        with patch('flowlib.providers.core.registry.provider_registry') as mock_registry:
            mock_llm = AsyncMock()
            mock_registry.get_by_config = AsyncMock(return_value=mock_llm)
            
            # Mock templates directly
            planner._planning_template = "test_planning_template"
            planner._input_generation_template = "test_input_template"
            
            # Initialize the planner
            await planner.initialize()
        
        # Test that the planner is now initialized
        assert planner.initialized


class TestAgentPlannerBasicPlanning:
    """Test basic planning functionality."""

    @pytest.fixture
    def planner(self):
        """Create initialized planner."""
        config = PlannerConfig()
        planner = AgentPlanner(config=config)
        planner._initialized = True
        return planner

    @pytest.fixture
    def mock_state(self):
        """Create mock agent state."""
        state = AgentState(task_description="Complete a complex task")
        state.context = {"user_input": "Create a web application"}
        return state

    @pytest.fixture
    def mock_flows(self):
        """Create mock available flows."""
        return {
            "conversation": {"description": "Handle conversation"},
            "shell_command": {"description": "Execute shell commands"},
            "classification": {"description": "Classify messages"}
        }

    @pytest.mark.asyncio
    async def test_simple_planning_no_todos(self, planner, mock_state, mock_flows):
        """Test simple planning that doesn't generate TODOs."""
        # Mock a simple plan with few steps
        simple_plan = Plan(
            task_description="Simple task",
            steps=[
                PlanStep(
                    step_intent="respond", 
                    rationale="Provide response to user", 
                    flow_name="conversation"
                )
            ]
        )
        
        # Mock all the necessary registries and providers
        mock_llm_provider = AsyncMock()
        mock_llm_provider.generate_structured = AsyncMock(return_value=simple_plan)
        
        with patch('flowlib.flows.registry.flow_registry') as mock_flow_registry, \
             patch('flowlib.agent.components.planning.planner.flow_registry') as mock_agent_flow_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            
            # Setup flow registry mocks
            mock_flow_instances = {
                "conversation": MagicMock(),
                "shell_command": MagicMock(),
                "classification": MagicMock()
            }
            mock_flow_registry.get_agent_selectable_flows.return_value = mock_flow_instances
            mock_agent_flow_registry.get_agent_selectable_flows.return_value = mock_flow_instances
            
            # Setup flow metadata with mock
            mock_metadata = MagicMock()
            mock_metadata.description = "Handle conversation"
            mock_flow_registry.get_flow_metadata.return_value = mock_metadata
            mock_agent_flow_registry.get_flow_metadata.return_value = mock_metadata
            
            # Setup provider registry
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm_provider)
            
            # Setup resource registry
            mock_template = MagicMock()
            mock_resource_registry.get.return_value = mock_template
            mock_resource_registry.contains.return_value = True
            
            # Set planner's LLM provider
            planner.llm_provider = mock_llm_provider
            planner._planning_template = mock_template
            
            result, todos = await planner.plan_with_todos(
                state=mock_state,
                available_flows=mock_flows,
                memory_context="test_context"
            )
            
            assert isinstance(result, PlanningResult)
            assert todos is None  # No TODOs for simple tasks

    @pytest.mark.asyncio
    async def test_complex_planning_with_todos(self, planner, mock_state, mock_flows):
        """Test complex planning that generates TODOs."""
        # Mock a complex plan with many steps
        complex_plan = Plan(
            task_description="Complex web application",
            steps=[
                PlanStep(step_intent="setup", rationale="Setup project structure", flow_name="shell_command"),
                PlanStep(step_intent="create", rationale="Create frontend components", flow_name="shell_command"),
                PlanStep(step_intent="create", rationale="Create backend API", flow_name="shell_command"),
                PlanStep(step_intent="test", rationale="Run comprehensive tests", flow_name="shell_command"),
                PlanStep(step_intent="deploy", rationale="Deploy application to production", flow_name="shell_command")
            ]
        )
        
        mock_todos = [
            TodoItem(
                content="Setup project structure",
                priority=TodoPriority.HIGH,
                status=TodoStatus.PENDING
            ),
            TodoItem(
                content="Create frontend components",
                priority=TodoPriority.MEDIUM,
                status=TodoStatus.PENDING
            )
        ]
        
        # Mock all the necessary registries and providers
        mock_llm_provider = AsyncMock()
        mock_llm_provider.generate_structured = AsyncMock(return_value=complex_plan)
        
        with patch('flowlib.flows.registry.flow_registry') as mock_flow_registry, \
             patch('flowlib.agent.components.planning.planner.flow_registry') as mock_agent_flow_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry, \
             patch.object(planner, '_generate_todos_from_plan') as mock_todo_gen, \
             patch.object(planner, '_is_complex_task') as mock_complex_check:
            
            # Setup flow registry mocks
            mock_flow_instances = {
                "conversation": MagicMock(),
                "shell_command": MagicMock(),
                "classification": MagicMock()
            }
            mock_flow_registry.get_agent_selectable_flows.return_value = mock_flow_instances
            mock_agent_flow_registry.get_agent_selectable_flows.return_value = mock_flow_instances
            
            # Setup flow metadata with mock
            mock_metadata = MagicMock()
            mock_metadata.description = "Execute shell commands"
            mock_flow_registry.get_flow_metadata.return_value = mock_metadata
            mock_agent_flow_registry.get_flow_metadata.return_value = mock_metadata
            
            # Setup provider registry
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm_provider)
            
            # Setup resource registry
            mock_template = MagicMock()
            mock_resource_registry.get.return_value = mock_template
            mock_resource_registry.contains.return_value = True
            
            # Set planner's LLM provider
            planner.llm_provider = mock_llm_provider
            planner._planning_template = mock_template
            
            # Setup complex task detection and todo generation
            mock_todo_gen.return_value = mock_todos
            mock_complex_check.return_value = True
            
            result, todos = await planner.plan_with_todos(
                state=mock_state,
                available_flows=mock_flows,
                memory_context="test_context"
            )
            
            assert isinstance(result, PlanningResult)
            assert todos is not None
            assert len(todos) == 2
            assert todos[0].content == "Setup project structure"

    @pytest.mark.asyncio
    async def test_todo_generation_disabled(self, planner, mock_state, mock_flows):
        """Test planning with TODO generation disabled."""
        complex_plan = Plan(
            task_description="Complex task",
            steps=[
                PlanStep(step_intent="step1", rationale="First step in process", flow_name="conversation"),
                PlanStep(step_intent="step2", rationale="Second step in process", flow_name="conversation"),
                PlanStep(step_intent="step3", rationale="Third step in process", flow_name="conversation"),
                PlanStep(step_intent="step4", rationale="Fourth step in process", flow_name="conversation")
            ]
        )
        
        # Mock all the necessary registries and providers
        mock_llm_provider = AsyncMock()
        mock_llm_provider.generate_structured = AsyncMock(return_value=complex_plan)
        
        with patch('flowlib.flows.registry.flow_registry') as mock_flow_registry, \
             patch('flowlib.agent.components.planning.planner.flow_registry') as mock_agent_flow_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            
            # Setup flow registry mocks
            mock_flow_instances = {
                "conversation": MagicMock(),
                "shell_command": MagicMock(),
                "classification": MagicMock()
            }
            mock_flow_registry.get_agent_selectable_flows.return_value = mock_flow_instances
            mock_agent_flow_registry.get_agent_selectable_flows.return_value = mock_flow_instances
            
            # Setup flow metadata with mock
            mock_metadata = MagicMock()
            mock_metadata.description = "Handle conversation"
            mock_flow_registry.get_flow_metadata.return_value = mock_metadata
            mock_agent_flow_registry.get_flow_metadata.return_value = mock_metadata
            
            # Setup provider registry
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm_provider)
            
            # Setup resource registry
            mock_template = MagicMock()
            mock_resource_registry.get.return_value = mock_template
            mock_resource_registry.contains.return_value = True
            
            # Set planner's LLM provider
            planner.llm_provider = mock_llm_provider
            planner._planning_template = mock_template
            
            result, todos = await planner.plan_with_todos(
                state=mock_state,
                available_flows=mock_flows,
                memory_context="test_context",
                auto_generate_todos=False
            )
            
            assert isinstance(result, PlanningResult)
            assert todos is None  # TODOs disabled


class TestAgentPlannerTodoGeneration:
    """Test TODO generation functionality."""

    @pytest.fixture
    def planner(self):
        """Create initialized planner."""
        config = PlannerConfig()
        planner = AgentPlanner(config=config)
        planner._initialized = True
        return planner

    @pytest.mark.asyncio
    async def test_generate_todos_from_plan_success(self, planner):
        """Test successful TODO generation from plan."""
        plan = Plan(
            task_description="Build web application",
            steps=[
                PlanStep(step_intent="setup", rationale="Setup development environment", flow_name="shell_command"),
                PlanStep(step_intent="frontend", rationale="Create frontend", flow_name="shell_command"),
                PlanStep(step_intent="backend", rationale="Create backend", flow_name="shell_command")
            ]
        )
        
        state = AgentState(task_description="Build a web application")
        
        mock_todos = [
            TodoItem(content="Setup development environment", priority=TodoPriority.HIGH),
            TodoItem(content="Create frontend components", priority=TodoPriority.MEDIUM),
            TodoItem(content="Implement backend API", priority=TodoPriority.MEDIUM)
        ]
        
        # Mock the todo generation flow
        with patch.object(planner.todo_generation_flow, 'run_pipeline') as mock_run_pipeline:
            mock_output = MagicMock()
            mock_output.todos = mock_todos
            mock_run_pipeline.return_value = mock_output
            
            todos = await planner._generate_todos_from_plan(plan, state)
            
            assert len(todos) == 3
            assert todos[0].content == "Setup development environment"
            assert todos[0].priority == TodoPriority.HIGH

    @pytest.mark.asyncio
    async def test_generate_todos_from_plan_failure(self, planner):
        """Test TODO generation failure handling."""
        plan = Plan(
            task_description="Test task",
            steps=[PlanStep(step_intent="test", rationale="Test step", flow_name="conversation")]
        )
        
        state = AgentState(task_description="Test task")
        
        # Mock todo generation flow to raise exception
        with patch.object(planner.todo_generation_flow, 'execute') as mock_execute:
            mock_execute.side_effect = Exception("TODO generation failed")
            
            with pytest.raises(Exception):
                await planner._generate_todos_from_plan(plan, state)

    def test_is_complex_task_true(self, planner):
        """Test complex task detection - positive case."""
        task_description = "Create a comprehensive web application with user authentication, database integration, and real-time features"
        
        plan = Plan(
            task_description="Complex web app",
            steps=[PlanStep(step_intent="step", rationale="Step", flow_name="conv") for _ in range(5)]
        )
        
        result = planner._is_complex_task(task_description, plan)
        assert result is True

    def test_is_complex_task_false(self, planner):
        """Test complex task detection - negative case."""
        task_description = "Say hello"
        
        plan = Plan(
            task_description="Simple greeting",
            steps=[PlanStep(step_intent="greet", rationale="Say hello", flow_name="conversation")]
        )
        
        result = planner._is_complex_task(task_description, plan)
        assert result is False


class TestAgentPlannerActivityStreaming:
    """Test activity streaming functionality."""

    @pytest.fixture
    def planner_with_stream(self):
        """Create planner with activity stream."""
        config = PlannerConfig()
        mock_activity_stream = MagicMock()
        planner = AgentPlanner(config=config, activity_stream=mock_activity_stream)
        planner._initialized = True
        return planner, mock_activity_stream

    @pytest.mark.asyncio
    async def test_activity_streaming_todo_generation(self, planner_with_stream):
        """Test activity streaming during TODO generation."""
        planner, mock_stream = planner_with_stream
        
        complex_plan = Plan(
            task_description="Complex task",
            steps=[
                PlanStep(step_intent="step", rationale="Step", flow_name="conv") for _ in range(4)
            ]
        )
        
        mock_todos = [
            TodoItem(content="Task 1", priority=TodoPriority.HIGH),
            TodoItem(content="Task 2", priority=TodoPriority.MEDIUM)
        ]
        
        state = AgentState(task_description="Complex task")
        
        # Mock all the necessary registries and providers
        mock_llm_provider = AsyncMock()
        mock_llm_provider.generate_structured = AsyncMock(return_value=complex_plan)
        
        with patch('flowlib.flows.registry.flow_registry') as mock_flow_registry, \
             patch('flowlib.agent.components.planning.planner.flow_registry') as mock_agent_flow_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry, \
             patch.object(planner, '_generate_todos_from_plan') as mock_todo_gen, \
             patch.object(planner, '_is_complex_task') as mock_complex_check:
            
            # Setup flow registry mocks
            mock_flow_instances = {
                "conv": MagicMock(),
                "conversation": MagicMock(),
                "shell_command": MagicMock()
            }
            mock_flow_registry.get_agent_selectable_flows.return_value = mock_flow_instances
            mock_agent_flow_registry.get_agent_selectable_flows.return_value = mock_flow_instances
            
            # Setup flow metadata with mock
            mock_metadata = MagicMock()
            mock_metadata.description = "Handle conversation"
            mock_flow_registry.get_flow_metadata.return_value = mock_metadata
            mock_agent_flow_registry.get_flow_metadata.return_value = mock_metadata
            
            # Setup provider registry
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm_provider)
            
            # Setup resource registry
            mock_template = MagicMock()
            mock_resource_registry.get.return_value = mock_template
            mock_resource_registry.contains.return_value = True
            
            # Set planner's LLM provider
            planner.llm_provider = mock_llm_provider
            planner._planning_template = mock_template
            
            # Setup complex task detection and todo generation
            mock_todo_gen.return_value = mock_todos
            mock_complex_check.return_value = True
            
            await planner.plan_with_todos(
                state=state,
                available_flows={"conv": {}},
                memory_context="test"
            )
            
            # Verify activity stream calls
            mock_stream.planning.assert_called()
            mock_stream.todo_create.assert_called()

    @pytest.mark.asyncio
    async def test_activity_streaming_todo_generation_error(self, planner_with_stream):
        """Test activity streaming when TODO generation fails."""
        planner, mock_stream = planner_with_stream
        
        complex_plan = Plan(
            task_description="Complex task",
            steps=[
                PlanStep(step_intent="step", rationale="Step", flow_name="conv") for _ in range(4)
            ]
        )
        
        state = AgentState(task_description="Complex task")
        
        # Mock all the necessary registries and providers
        mock_llm_provider = AsyncMock()
        mock_llm_provider.generate_structured = AsyncMock(return_value=complex_plan)
        
        with patch('flowlib.flows.registry.flow_registry') as mock_flow_registry, \
             patch('flowlib.agent.components.planning.planner.flow_registry') as mock_agent_flow_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry, \
             patch.object(planner, '_generate_todos_from_plan') as mock_todo_gen, \
             patch.object(planner, '_is_complex_task') as mock_complex_check:
            
            # Setup flow registry mocks
            mock_flow_instances = {
                "conv": MagicMock(),
                "conversation": MagicMock(),
                "shell_command": MagicMock()
            }
            mock_flow_registry.get_agent_selectable_flows.return_value = mock_flow_instances
            mock_agent_flow_registry.get_agent_selectable_flows.return_value = mock_flow_instances
            
            # Setup flow metadata with mock
            mock_metadata = MagicMock()
            mock_metadata.description = "Handle conversation"
            mock_flow_registry.get_flow_metadata.return_value = mock_metadata
            mock_agent_flow_registry.get_flow_metadata.return_value = mock_metadata
            
            # Setup provider registry
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm_provider)
            
            # Setup resource registry
            mock_template = MagicMock()
            mock_resource_registry.get.return_value = mock_template
            mock_resource_registry.contains.return_value = True
            
            # Set planner's LLM provider
            planner.llm_provider = mock_llm_provider
            planner._planning_template = mock_template
            
            # Setup complex task detection and todo generation with error
            mock_todo_gen.side_effect = Exception("TODO gen failed")
            mock_complex_check.return_value = True
            
            result, todos = await planner.plan_with_todos(
                state=state,
                available_flows={"conv": {}},
                memory_context="test"
            )
            
            # Should handle error gracefully
            assert todos is None
            mock_stream.error.assert_called()


class TestAgentPlannerIntegration:
    """Test integration with other components."""

    @pytest.mark.asyncio
    async def test_plan_conversion(self):
        """Test conversion from Plan to PlanningResult."""
        config = PlannerConfig()
        planner = AgentPlanner(config=config)
        
        plan = Plan(
            task_description="Test goal",
            steps=[
                PlanStep(step_intent="action1", rationale="First action", flow_name="flow1"),
                PlanStep(step_intent="action2", rationale="Second action", flow_name="flow2")
            ]
        )
        
        result = planner._convert_plan_to_planning_result(plan)
        
        assert isinstance(result, PlanningResult)
        assert result.selected_flow == "flow1"  # Should select first step
        assert result.reasoning is not None
        assert result.reasoning.explanation == "First action"

    def test_complexity_threshold_configuration(self):
        """Test complexity threshold configuration."""
        config = PlannerConfig()
        planner = AgentPlanner(config=config)
        
        # Default threshold
        assert planner.complexity_threshold == 3
        
        # Should be configurable
        planner.complexity_threshold = 5
        assert planner.complexity_threshold == 5

    @pytest.mark.asyncio
    async def test_error_handling_not_initialized(self):
        """Test error handling when planner is not initialized."""
        config = PlannerConfig()
        planner = AgentPlanner(config=config)
        # Don't initialize
        
        state = AgentState(task_description="Test task")
        
        with pytest.raises(Exception):  # Should raise not initialized error
            await planner.plan_with_todos(
                state=state,
                available_flows={},
                memory_context="test"
            )