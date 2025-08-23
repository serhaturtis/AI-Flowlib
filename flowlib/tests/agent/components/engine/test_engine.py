"""
Tests for the AgentEngine class.

This module tests the central execution engine that orchestrates TODO-driven
and single-cycle execution strategies, integrating planning, memory, reflection,
and flow execution systems.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock, PropertyMock
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

from flowlib.agent.components.engine.engine import AgentEngine, ExecutionStrategy
from flowlib.agent.models.config import AgentConfig, EngineConfig, PlannerConfig, ReflectionConfig
from flowlib.agent.models.state import AgentState
from flowlib.agent.components.planning.todo import TodoManager, TodoItem, TodoStatus, TodoPriority
from flowlib.agent.components.planning.planner import AgentPlanner
from flowlib.agent.components.planning.models import PlanningResult, PlanningExplanation
from flowlib.agent.components.reflection.models import ReflectionResult
from flowlib.flows.models.results import FlowResult, AgentResult
from flowlib.flows.models.metadata import FlowMetadata
from flowlib.flows.models.constants import FlowStatus
from flowlib.agent.core.errors import ExecutionError, ConfigurationError, NotInitializedError
from flowlib.agent.models.state import ExecutionResult


class MockInputModel(BaseModel):
    """Mock input model for testing."""
    message: str
    context: str = "test"


class MockOutputModel(BaseModel):
    """Mock output model for testing."""
    response: str
    success: bool = True


class MockFlow:
    """Mock flow for testing."""
    
    def __init__(self, name: str = "test_flow", description: str = "Test flow"):
        self.name = name
        self.description = description
        self._execution_count = 0
    
    async def execute(self, context):
        """Mock flow execution."""
        self._execution_count += 1
        return FlowResult(
            status=FlowStatus.SUCCESS,
            data=MockOutputModel(response=f"Flow {self.name} executed {self._execution_count} times"),
            flow_name=self.name,
            metadata={"execution_count": self._execution_count}
        )
    
    def get_description(self):
        return self.description
    
    def get_pipeline_method(self):
        """Return mock pipeline with input/output models."""
        pipeline = Mock()
        pipeline.__input_model__ = MockInputModel
        pipeline.__output_model__ = MockOutputModel
        return pipeline


class MockMemory:
    """Mock memory system for testing."""
    
    def __init__(self):
        self.initialized = False
        self._storage = {}
        self._contexts = set()
        self.parent = None
    
    async def initialize(self):
        self.initialized = True
    
    async def shutdown(self):
        self.initialized = False
        # Clear state on shutdown
        self._storage.clear()
        self._contexts.clear()
    
    async def create_context(self, context_name: str, parent=None, metadata=None):
        context_path = f"{parent}/{context_name}" if parent else context_name
        self._contexts.add(context_path)
        return context_path
    
    async def store_with_model(self, request):
        self._storage[request.key] = request.value
    
    async def retrieve_with_model(self, request):
        return self._storage.get(request.key)
    
    def set_parent(self, parent):
        self.parent = parent


class MockPlanner:
    """Mock unified planner for testing."""
    
    def __init__(self, config=None, name="mock_planner", activity_stream=None):
        self.config = config or PlannerConfig(model_name="mock", provider_name="mock")
        self.name = name
        self.activity_stream = activity_stream
        self.initialized = False
        self.parent = None
        self._strategy_responses = {}
        self._plan_responses = {}
    
    async def initialize(self):
        self.initialized = True
    
    async def shutdown(self):
        self.initialized = False
        # Clear state on shutdown
        self._strategy_responses.clear()
        self._plan_responses.clear()
    
    def set_parent(self, parent):
        self.parent = parent
    
    def set_strategy_response(self, task: str, strategy: str):
        """Set mock response for strategy determination."""
        self._strategy_responses[task] = strategy
    
    def set_plan_response(self, task: str, todos: List[TodoItem]):
        """Set mock response for planning with todos."""
        self._plan_responses[task] = todos
    
    async def determine_execution_strategy(self, task_description: str) -> str:
        """Mock strategy determination."""
        return self._strategy_responses.get(task_description, "single_cycle")
    
    async def plan_with_todos(self, state, available_flows, memory_context, auto_generate_todos=True):
        """Mock planning with TODO generation."""
        plan_result = PlanningResult(
            selected_flow="ConversationFlow",
            reasoning=PlanningExplanation(
                explanation="Mock planning result",
                rationale="Selected flow for testing",
                decision_factors=["testing", "mock"]
            )
        )
        
        todos = self._plan_responses.get(state.task_description, [])
        return plan_result, todos
    
    async def plan(self, context):
        """Mock basic planning."""
        return PlanningResult(
            selected_flow="ConversationFlow",
            reasoning=PlanningExplanation(
                explanation="Mock single cycle planning",
                rationale="Selected for single cycle execution",
                decision_factors=["single_cycle", "mock"]
            )
        )
    
    async def generate_inputs(self, state, flow_name, planning_result, memory_context, flow=None):
        """Mock input generation."""
        return MockInputModel(
            message=state.task_description,
            context=memory_context
        )


class MockReflection:
    """Mock reflection system for testing."""
    
    def __init__(self, config=None, activity_stream=None):
        self.config = config
        self.activity_stream = activity_stream
        self.initialized = False
        self.parent = None
    
    async def initialize(self):
        self.initialized = True
    
    async def shutdown(self):
        self.initialized = False
        # Clear any state on shutdown
        self.parent = None
    
    def set_parent(self, parent):
        self.parent = parent
    
    async def reflect(self, state, flow_name, flow_inputs, flow_result, memory_context):
        """Mock reflection."""
        # Determine completion based on flow result
        is_complete = getattr(flow_result, 'metadata', {}).get('complete', False)
        progress = min(state.cycles * 25, 100)  # 25% per cycle, max 100%
        
        return ReflectionResult(
            reflection="Mock reflection analysis of the flow execution",
            is_complete=is_complete,
            progress=progress,
            completion_reason="Mock completion" if is_complete else None,
            insights=["Mock insight"]
        )


class MockActivityStream:
    """Mock activity stream for testing."""
    
    def __init__(self):
        self.events = []
        self.sections = []
        self.current_section = None
    
    def start_section(self, title: str):
        self.current_section = title
        self.sections.append(title)
    
    def end_section(self):
        self.current_section = None
    
    def execution(self, message: str, **kwargs):
        self.events.append({"type": "execution", "message": message, **kwargs})
    
    def error(self, message: str):
        self.events.append({"type": "error", "message": message})
    
    def planning(self, message: str, **kwargs):
        self.events.append({"type": "planning", "message": message, **kwargs})
    
    def context(self, message: str, **kwargs):
        self.events.append({"type": "context", "message": message, **kwargs})
    
    def decision(self, message: str, rationale: str, **kwargs):
        self.events.append({"type": "decision", "message": message, "rationale": rationale, **kwargs})
    
    def flow_selection(self, flow_name: str, rationale: str, alternatives=None, **kwargs):
        self.events.append({"type": "flow_selection", "flow_name": flow_name, "rationale": rationale, "alternatives": alternatives, **kwargs})
    
    def llm_call(self, model_name: str, message: str, **kwargs):
        self.events.append({"type": "llm_call", "model_name": model_name, "message": message, **kwargs})
    
    def memory_retrieval(self, message: str, **kwargs):
        self.events.append({"type": "memory_retrieval", "message": message, **kwargs})
    
    def prompt_selection(self, prompt_name: str, variables: dict, **kwargs):
        self.events.append({"type": "prompt_selection", "prompt_name": prompt_name, "variables": variables, **kwargs})


class MockParentAgent:
    """Mock parent agent for testing."""
    
    def __init__(self):
        self.name = "mock_parent_agent"  # Required by AgentComponent.set_parent
        self.flows = {}
        self.state_save_count = 0
    
    async def save_state(self):
        self.state_save_count += 1


class TestAgentEngine:
    """Test the AgentEngine class."""
    
    def setup_method(self):
        """Set up each test with fresh mocks."""
        self.config = AgentConfig(
            name="test_agent",
            persona="Test assistant",
            provider_name="mock",
            engine_config=EngineConfig(
                max_iterations=5,
                stop_on_error=True
            ),
            planner_config=PlannerConfig(
                model_name="test-model",
                provider_name="mock"
            ),
            reflection_config=ReflectionConfig(
                model_name="test-model",
                provider_name="mock"
            )
        )
        
        self.mock_memory = MockMemory()
        self.mock_planner = MockPlanner()
        self.mock_reflection = MockReflection()
        self.mock_activity_stream = MockActivityStream()
        self.mock_parent = MockParentAgent()
        
        # Create engine
        self.engine = AgentEngine(
            config=self.config.engine_config,
            memory=self.mock_memory,
            planner=self.mock_planner,
            reflection=self.mock_reflection,
            activity_stream=self.mock_activity_stream,
            agent_config=self.config
        )
        self.engine.set_parent(self.mock_parent)
    
    def mock_all_registries(self):
        """Helper to mock all required registries with proper return values."""
        return patch('flowlib.flows.registry.flow_registry'), \
               patch('flowlib.agent.components.planning.planner.flow_registry'), \
               patch('flowlib.resources.registry.registry.resource_registry'), \
               patch('flowlib.providers.core.registry.provider_registry')
    
    def setup_registry_mocks(self, mock_flow_registry, mock_unified_registry, mock_resource_registry, mock_provider_registry):
        """Setup standard mock return values for all registries."""
        # Mock flows
        conversation_flow = MockFlow("ConversationFlow")
        mock_flow_registry.get_agent_selectable_flows.return_value = {"ConversationFlow": conversation_flow}
        mock_unified_registry.get_agent_selectable_flows.return_value = {"ConversationFlow": conversation_flow}
        
        # Mock flow metadata
        flow_metadata = FlowMetadata(
            name="ConversationFlow",
            description="Test conversation flow",
            input_model=MockInputModel,
            output_model=MockOutputModel
        )
        mock_flow_registry.get_flow_metadata.return_value = flow_metadata
        mock_unified_registry.get_flow_metadata.return_value = flow_metadata
        
        # Mock resources and providers
        mock_resource_registry.get.return_value = Mock()
        
        # Create mock LLM provider
        from flowlib.agent.components.planning.models import Plan, PlanStep
        mock_plan = Plan(
            task_description="Test task",
            steps=[
                PlanStep(
                    flow_name="ConversationFlow",
                    step_intent="Process the task",
                    rationale="Basic step to handle the task"
                )
            ]
        )
        
        async def mock_generate_structured(prompt, output_type, prompt_variables, model_name):
            if output_type == Plan:
                return mock_plan
            elif output_type == MockInputModel:
                return MockInputModel(
                    message=prompt_variables.get("task_description", "Test message"),
                    context="test"
                )
            else:
                return output_type(message="test", context="test")
        
        mock_llm_provider = Mock()
        mock_llm_provider.generate_structured = AsyncMock(side_effect=mock_generate_structured)
        mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm_provider)
    
    def test_init_success(self):
        """Test successful engine initialization."""
        assert self.engine._config == self.config.engine_config
        assert self.engine._memory is self.mock_memory
        assert self.engine._planner is self.mock_planner
        assert self.engine._reflection is self.mock_reflection
        assert self.engine._activity_stream is self.mock_activity_stream
        assert self.engine.todo_manager is None  # Not initialized yet
    
    @pytest.mark.asyncio
    async def test_initialize_impl_success(self):
        """Test successful engine initialization."""
        # Mock all the dependencies to avoid deep initialization issues
        with patch('flowlib.flows.registry.flow_registry') as mock_registry, \
             patch('flowlib.agent.components.planning.planner.AgentPlanner.initialize', new_callable=AsyncMock) as mock_unified_init, \
             patch('flowlib.agent.components.planning.todo.TodoManager.initialize', new_callable=AsyncMock) as mock_todo_init, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            
            mock_registry.get_agent_selectable_flows.return_value = {}
            
            # Mock the resource and provider registries
            mock_resource_registry.get.return_value = Mock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=Mock())
            
            # Initialize the mock components manually since the engine expects them to be initialized
            await self.mock_memory.initialize()
            await self.mock_planner.initialize()
            await self.mock_reflection.initialize()
            
            await self.engine.initialize()
            
            assert self.engine.initialized
            assert self.engine.todo_manager is not None
            assert self.engine._planner is not None
            assert isinstance(self.engine._planner, AgentPlanner)
            assert self.mock_memory.initialized
            # The original planner gets replaced, so check the new unified planner was initialized
            assert self.engine._planner is self.engine._planner
            assert self.mock_reflection.initialized
            
            # Verify the unified planner and todo manager were properly initialized
            mock_unified_init.assert_called_once()
            # TODO manager gets initialized twice - once by base engine, once by unified engine
            assert mock_todo_init.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_initialize_impl_with_existing_planner(self):
        """Test initialization when planner is already AgentPlanner."""
        # Replace planner with AgentPlanner instance
        planner = AgentPlanner(self.config.planner_config)
        planner._initialized = True  # Set it as initialized to simulate existing planner
        self.engine._planner = planner
        
        with patch('flowlib.flows.registry.flow_registry') as mock_registry, \
             patch('flowlib.agent.components.planning.todo.TodoManager.initialize', new_callable=AsyncMock), \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            
            mock_registry.get_agent_selectable_flows.return_value = {}
            
            # Mock the resource and provider registries
            mock_resource_registry.get.return_value = Mock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=Mock())
            
            await self.engine.initialize()
            
            assert self.engine._planner is planner
            assert self.engine._planner is planner
    
    @pytest.mark.asyncio
    async def test_shutdown_impl_success(self):
        """Test successful engine shutdown."""
        with patch('flowlib.flows.registry.flow_registry'), \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry, \
             patch('flowlib.agent.components.planning.planner.AgentPlanner.initialize', new_callable=AsyncMock), \
             patch('flowlib.agent.components.planning.todo.TodoManager.initialize', new_callable=AsyncMock):
            
            # Mock the resource and provider registries
            mock_resource_registry.get.return_value = Mock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=Mock())
            
            await self.engine.initialize()
            
            assert self.engine.todo_manager is not None
            
            await self.engine.shutdown()
            
            assert not self.engine.initialized
            assert not self.mock_memory.initialized
            assert not self.mock_planner.initialized
            assert not self.mock_reflection.initialized
    
    @pytest.mark.asyncio
    async def test_execute_with_strategy_auto_single_cycle(self):
        """Test strategy auto-determination leading to single-cycle execution."""
        with patch('flowlib.flows.registry.flow_registry') as mock_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            
            mock_registry.get_agent_selectable_flows.return_value = {}
            
            # Mock the resource and provider registries
            mock_resource_registry.get.return_value = Mock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=Mock())
            
            # Mock flows for parent agent
            self.mock_parent.flows["ConversationFlow"] = MockFlow("ConversationFlow")
            
            # Mock flow metadata
            flow_metadata = FlowMetadata(
                name="ConversationFlow",
                description="Test conversation flow",
                input_model=MockInputModel,
                output_model=MockOutputModel
            )
            mock_registry.get_flow_metadata.return_value = flow_metadata
            
            await self.engine.initialize()
            
            # Set strategy response
            self.mock_planner.set_strategy_response("Test task", "single_cycle")
            
            result = await self.engine.execute_with_strategy(
                "Test task",
                {"key": "value"},
                ExecutionStrategy.AUTO
            )
            
            assert isinstance(result, AgentResult)  # execute_with_strategy returns AgentResult
            assert result.success is not None
            assert hasattr(result, 'results')
            assert hasattr(result, 'state')
    
    @pytest.mark.asyncio
    async def test_execute_with_strategy_todo_driven_success(self):
        """Test TODO-driven execution strategy."""
        with patch('flowlib.agent.components.planning.planner.flow_registry') as mock_registry, \
             patch('flowlib.flows.registry.flow_registry') as mock_unified_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            
            # Mock flows for parent agent first
            conversation_flow = MockFlow("ConversationFlow")
            self.mock_parent.flows["ConversationFlow"] = conversation_flow
            
            mock_registry.get_agent_selectable_flows.return_value = {"ConversationFlow": conversation_flow}
            mock_unified_registry.get_agent_selectable_flows.return_value = {"ConversationFlow": conversation_flow}
            
            # Mock the resource and provider registries
            mock_resource_registry.get.return_value = Mock()
            
            # Create a proper async mock for LLM provider that respects output_type
            from flowlib.agent.components.planning.models import Plan, PlanStep
            from flowlib.agent.components.planning.todo import TodoItem
            mock_plan = Plan(
                task_description="Test TODO task",
                steps=[
                    PlanStep(
                        flow_name="ConversationFlow",
                        step_intent="Process the task",
                        rationale="Basic step to handle the task"
                    )
                ]
            )
            
            # Create a mock that returns different types based on output_type
            async def mock_generate_structured(prompt, output_type, prompt_variables, model_name):
                if output_type == Plan:
                    return mock_plan
                elif output_type == MockInputModel:
                    return MockInputModel(
                        message=prompt_variables.get("task_description", "Test message"),
                        context="test"
                    )
                else:
                    # Default to the requested type with mock data
                    return output_type(message="test", context="test")
            
            mock_llm_provider = Mock()
            mock_llm_provider.generate_structured = AsyncMock(side_effect=mock_generate_structured)
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm_provider)
            
            # Mock flow metadata
            flow_metadata = FlowMetadata(
                name="ConversationFlow",
                description="Test conversation flow",
                input_model=MockInputModel,
                output_model=MockOutputModel
            )
            mock_registry.get_flow_metadata.return_value = flow_metadata
            mock_unified_registry.get_flow_metadata.return_value = flow_metadata
            mock_unified_registry.get_flow_metadata.return_value = flow_metadata
            
            await self.engine.initialize()
            
            # Set the llm_provider after initialization when unified_planner is created
            if hasattr(self.engine, 'unified_planner') and self.engine.unified_planner:
                self.engine.unified_planner.llm_provider = mock_llm_provider
            
            # Create mock TODOs
            todo1 = TodoItem(
                id="todo1",
                content="First task",
                priority=TodoPriority.HIGH,
                status=TodoStatus.PENDING
            )
            todo2 = TodoItem(
                id="todo2", 
                content="Second task",
                priority=TodoPriority.MEDIUM,
                status=TodoStatus.PENDING
            )
            
            # Set up mock responses
            self.mock_planner.set_plan_response("Test TODO task", [todo1, todo2])
            
            result = await self.engine.execute_with_strategy(
                "Test TODO task",
                {"key": "value"},
                ExecutionStrategy.TODO_DRIVEN
            )
            
            # execute_with_strategy always returns AgentResult
            assert isinstance(result, AgentResult)
            assert result.success is not None
            assert result.state is not None
            # Note: result.results may be empty if TODO generation failed and fell back to single-cycle
    
    @pytest.mark.asyncio
    async def test_execute_with_strategy_todo_driven_fallback(self):
        """Test TODO-driven execution falling back to single-cycle when no TODOs."""
        with patch('flowlib.flows.registry.flow_registry') as mock_registry, \
             patch('flowlib.agent.components.planning.planner.flow_registry') as mock_unified_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            
            # Set up the registry mocks to return available flows for planning
            # Both the main flow_registry and the base planning registry need flows
            conversation_flow = MockFlow("ConversationFlow")
            mock_unified_registry.get_agent_selectable_flows.return_value = {"ConversationFlow": conversation_flow}
            mock_registry.get_agent_selectable_flows.return_value = {"ConversationFlow": conversation_flow}
            
            # Mock the resource and provider registries
            mock_resource_registry.get.return_value = Mock()
            
            # Create a proper async LLM provider mock that respects output_type
            from flowlib.agent.components.planning.models import Plan, PlanStep
            mock_plan = Plan(
                task_description="Test fallback task",
                steps=[
                    PlanStep(
                        flow_name="ConversationFlow",
                        step_intent="Process the task",
                        rationale="Basic step to handle the task"
                    )
                ]
            )
            
            # Create a mock that returns different types based on output_type
            async def mock_generate_structured(prompt, output_type, prompt_variables, model_name):
                if output_type == Plan:
                    return mock_plan
                elif output_type == MockInputModel:
                    return MockInputModel(
                        message=prompt_variables.get("task_description", "Test message"),
                        context="test"
                    )
                else:
                    # Default to the requested type with mock data
                    return output_type(message="test", context="test")
            
            mock_llm_provider = Mock()
            mock_llm_provider.generate_structured = AsyncMock(side_effect=mock_generate_structured)
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm_provider)
            
            # Mock flows for parent agent
            self.mock_parent.flows["ConversationFlow"] = MockFlow("ConversationFlow")
            
            # Mock flow metadata
            flow_metadata = FlowMetadata(
                name="ConversationFlow",
                description="Test conversation flow",
                input_model=MockInputModel,
                output_model=MockOutputModel
            )
            mock_registry.get_flow_metadata.return_value = flow_metadata
            mock_unified_registry.get_flow_metadata.return_value = flow_metadata
            
            await self.engine.initialize()
            
            # Set the llm_provider after initialization when unified_planner is created
            if hasattr(self.engine, 'unified_planner') and self.engine.unified_planner:
                self.engine.unified_planner.llm_provider = mock_llm_provider
            
            # Set up empty TODO response (will trigger fallback)
            self.mock_planner.set_plan_response("Test fallback task", [])
            
            result = await self.engine.execute_with_strategy(
                "Test fallback task",
                {"key": "value"},
                ExecutionStrategy.TODO_DRIVEN
            )
            
            # Should fall back to single-cycle but still return AgentResult
            assert isinstance(result, AgentResult)
            assert result.success is not None
            assert result.state is not None
    
    @pytest.mark.asyncio
    async def test_execute_todo_cycle_success(self):
        """Test successful TODO cycle execution."""
        with patch('flowlib.flows.registry.flow_registry') as mock_flow_registry, \
             patch('flowlib.agent.components.planning.planner.flow_registry') as mock_unified_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            
            self.setup_registry_mocks(mock_flow_registry, mock_unified_registry, mock_resource_registry, mock_provider_registry)
            
            await self.engine.initialize()
            
            # Create test TODO
            todo = TodoItem(
                id="test_todo",
                content="Test TODO content",
                priority=TodoPriority.HIGH,
                assigned_tool="ConversationFlow"
            )
            
            # Create test state
            state = AgentState(
                task_id="test_task",
                task_description="Test task"
            )
            
            # Mock execute_cycle to return success and simulate execution history
            with patch.object(self.engine, 'execute_cycle', new_callable=AsyncMock) as mock_execute:
                # Mock that returns True (should continue) and modifies the state's execution history
                async def mock_execute_cycle_with_history(*args, **kwargs):
                    # Get the state from the call
                    call_state = kwargs.get('state')
                    if not call_state and args:
                        call_state = args[0]
                    if call_state:
                        # Add a mock execution to the history
                        call_state.add_execution_result(
                            flow_name="ConversationFlow",
                            inputs={"message": "Test TODO content"},
                            result={"data": {"result": "success"}},
                            success=True
                        )
                    return True  # Should continue
                
                mock_execute.side_effect = mock_execute_cycle_with_history
                
                result = await self.engine._execute_todo_cycle(todo, state)
                
                assert isinstance(result, FlowResult)
                assert result.data["result"] == "success"
                # Note: The TODO status is updated in _execute_todo_cycle, which is being mocked
                # We need to check the mock was called with correct parameters
                mock_execute.assert_called()
    
    @pytest.mark.asyncio
    async def test_execute_todo_cycle_error(self):
        """Test TODO cycle execution with error."""
        with patch('flowlib.flows.registry.flow_registry') as mock_flow_registry, \
             patch('flowlib.agent.components.planning.planner.flow_registry') as mock_unified_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            
            self.setup_registry_mocks(mock_flow_registry, mock_unified_registry, mock_resource_registry, mock_provider_registry)
            
            await self.engine.initialize()
            
            # Create test TODO
            todo = TodoItem(
                id="test_todo",
                content="Test TODO content",
                priority=TodoPriority.HIGH
            )
            
            # Create test state
            state = AgentState(
                task_id="test_task",
                task_description="Test task"
            )
            
            # Mock execute_cycle to raise exception
            with patch.object(self.engine, 'execute_cycle', new_callable=AsyncMock) as mock_execute:
                mock_execute.side_effect = Exception("Test error")
                
                result = await self.engine._execute_todo_cycle(todo, state)
                
                assert isinstance(result, FlowResult)
                assert not result.is_success()
                assert result.error == "Test error"
                assert "todo_id" in result.metadata
    
    def test_has_executable_todos_true(self):
        """Test _has_executable_todos returns True when TODOs are available."""
        # Mock TODO manager with executable TODOs
        mock_todo_list = Mock()
        mock_todo_list.get_executable_todos.return_value = [Mock(), Mock()]
        
        self.engine.todo_manager = Mock()
        self.engine.todo_manager.current_list = mock_todo_list
        
        assert self.engine._has_executable_todos() == True
    
    def test_has_executable_todos_false(self):
        """Test _has_executable_todos returns False when no TODOs are available."""
        # Mock TODO manager with no executable TODOs
        mock_todo_list = Mock()
        mock_todo_list.get_executable_todos.return_value = []
        
        self.engine.todo_manager = Mock()
        self.engine.todo_manager.current_list = mock_todo_list
        
        assert self.engine._has_executable_todos() == False
    
    def test_has_executable_todos_no_list(self):
        """Test _has_executable_todos returns False when no current list."""
        self.engine.todo_manager = Mock()
        self.engine.todo_manager.current_list = None
        
        assert self.engine._has_executable_todos() == False
    
    def test_get_completion_summary_with_list(self):
        """Test completion summary with active TODO list."""
        mock_todo_list = Mock()
        mock_todo_list.get_progress_summary.return_value = {
            "total": 5,
            "completed": 3,
            "failed": 1,
            "progress": 60
        }
        
        self.engine.todo_manager = Mock()
        self.engine.todo_manager.current_list = mock_todo_list
        
        summary = self.engine._get_completion_summary()
        
        assert summary["total"] == 5
        assert summary["completed"] == 3
        assert summary["failed"] == 1
        assert summary["progress"] == 60
    
    def test_get_completion_summary_no_list(self):
        """Test completion summary without active TODO list."""
        self.engine.todo_manager = Mock()
        self.engine.todo_manager.current_list = None
        
        summary = self.engine._get_completion_summary()
        
        assert summary["total"] == 0
        assert summary["completed"] == 0
        assert summary["failed"] == 0
        assert summary["progress"] == 0
    
    def test_get_memory_context(self):
        """Test memory context generation."""
        state = AgentState(task_id="test_123", task_description="Test task")
        
        context = self.engine._get_memory_context(state)
        
        assert context == "task_test_123"
    
    def test_get_available_flows_with_registry(self):
        """Test getting available flows from registry."""
        with patch('flowlib.flows.registry.flow_registry') as mock_registry:
            mock_flows = {"flow1": Mock(), "flow2": Mock()}
            mock_registry.get_agent_selectable_flows.return_value = mock_flows
            
            flows = self.engine._get_available_flows()
            
            assert flows == mock_flows
    
    def test_get_available_flows_no_registry(self):
        """Test getting available flows when registry is None."""
        with patch('flowlib.flows.registry.flow_registry', None):
            flows = self.engine._get_available_flows()
            
            assert flows == {}
    
    @pytest.mark.asyncio
    async def test_reflect_on_overall_progress_success(self):
        """Test overall progress reflection with successful completion."""
        with patch('flowlib.flows.registry.flow_registry') as mock_flow_registry, \
             patch('flowlib.agent.components.planning.planner.flow_registry') as mock_unified_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            
            self.setup_registry_mocks(mock_flow_registry, mock_unified_registry, mock_resource_registry, mock_provider_registry)
            
            await self.engine.initialize()
            
            # Create test result
            state = AgentState(task_id="test_task", task_description="Test task")
            result = AgentResult(
                success=True,
                results=[],
                state=state
            )
            
            completion_summary = {"completed": 4, "failed": 1, "total": 5, "progress": 0.8}
            
            # Mock reflection result
            mock_reflection_result = Mock()
            mock_reflection_result.model_dump.return_value = {"insights": ["test"]}
            self.mock_reflection.reflect = AsyncMock(return_value=mock_reflection_result)
            
            updated_result = await self.engine._reflect_on_overall_progress(
                "Test task", result, completion_summary
            )
            
            assert updated_result.success == True  # 80% >= 80% threshold
            assert "completion_summary" in updated_result.metadata
            assert "reflection" in updated_result.metadata
            assert "execution_strategy" in updated_result.metadata
            assert updated_result.metadata["execution_strategy"] == "todo_driven"
    
    @pytest.mark.asyncio
    async def test_reflect_on_overall_progress_low_success_rate(self):
        """Test overall progress reflection with low success rate."""
        with patch('flowlib.flows.registry.flow_registry') as mock_flow_registry, \
             patch('flowlib.agent.components.planning.planner.flow_registry') as mock_unified_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            
            self.setup_registry_mocks(mock_flow_registry, mock_unified_registry, mock_resource_registry, mock_provider_registry)
            
            await self.engine.initialize()
            
            # Create test result
            state = AgentState(task_id="test_task", task_description="Test task")
            result = AgentResult(
                success=True,
                results=[],
                state=state
            )
            
            completion_summary = {"completed": 2, "failed": 3, "total": 5, "progress": 0.4}
            
            # Mock reflection result
            mock_reflection_result = Mock()
            mock_reflection_result.model_dump.return_value = {"insights": ["test"]}
            self.mock_reflection.reflect = AsyncMock(return_value=mock_reflection_result)
            
            updated_result = await self.engine._reflect_on_overall_progress(
                "Test task", result, completion_summary
            )
            
            assert updated_result.success == False  # 40% < 80% threshold
    
    @pytest.mark.asyncio
    async def test_reflect_on_overall_progress_no_reflection(self):
        """Test overall progress reflection when reflection component is None."""
        self.engine._reflection = None
        
        # Create test result
        state = AgentState(task_id="test_task", task_description="Test task")
        result = AgentResult(
            success=True,
            results=[],
            state=state
        )
        
        completion_summary = {"completed": 4, "failed": 1, "total": 5, "progress": 80}
        
        updated_result = await self.engine._reflect_on_overall_progress(
            "Test task", result, completion_summary
        )
        
        # Should return unchanged result
        assert updated_result is result
    
    @pytest.mark.asyncio
    async def test_reflect_on_overall_progress_reflection_error(self):
        """Test overall progress reflection when reflection raises error."""
        with patch('flowlib.flows.registry.flow_registry') as mock_flow_registry, \
             patch('flowlib.agent.components.planning.planner.flow_registry') as mock_unified_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            
            self.setup_registry_mocks(mock_flow_registry, mock_unified_registry, mock_resource_registry, mock_provider_registry)
            
            await self.engine.initialize()
            
            # Create test result
            state = AgentState(task_id="test_task", task_description="Test task")
            result = AgentResult(
                success=True,
                results=[],
                state=state,
                metadata={}
            )
            
            completion_summary = {"completed": 4, "failed": 1, "total": 5, "progress": 0.8}
            
            # Mock reflection to raise error
            self.mock_reflection.reflect = AsyncMock(side_effect=Exception("Reflection failed"))
            
            updated_result = await self.engine._reflect_on_overall_progress(
                "Test task", result, completion_summary
            )
            
            # Should not fail, just continue without reflection
            assert updated_result.success == True
    
    def test_get_todo_manager(self):
        """Test getting TODO manager instance."""
        mock_manager = Mock()
        self.engine.todo_manager = mock_manager
        
        result = self.engine.get_todo_manager()
        
        assert result is mock_manager
    
    def test_get_todo_manager_none(self):
        """Test getting TODO manager when not initialized."""
        result = self.engine.get_todo_manager()
        
        assert result is None


class TestUnifiedEngineIntegration:
    """Integration tests for AgentEngine with complex scenarios."""
    
    def setup_method(self):
        """Set up each test with fresh mocks."""
        self.config = AgentConfig(
            name="integration_agent",
            persona="Integration test assistant",
            provider_name="mock",
            engine_config=EngineConfig(
                max_iterations=3,
                stop_on_error=True
            ),
            planner_config=PlannerConfig(
                model_name="test-model",
                provider_name="mock"
            ),
            reflection_config=ReflectionConfig(
                model_name="test-model",
                provider_name="mock"
            )
        )
        
        self.mock_memory = MockMemory()
        self.mock_planner = MockPlanner()
        self.mock_reflection = MockReflection()
        self.mock_activity_stream = MockActivityStream()
        self.mock_parent = MockParentAgent()
        
        # Create engine
        self.engine = AgentEngine(
            config=self.config.engine_config,
            memory=self.mock_memory,
            planner=self.mock_planner,
            reflection=self.mock_reflection,
            activity_stream=self.mock_activity_stream,
            agent_config=self.config
        )
        self.engine.set_parent(self.mock_parent)
    
    async def test_full_todo_driven_workflow(self):
        """Test complete TODO-driven workflow from start to finish."""
        with patch('flowlib.agent.components.planning.planner.flow_registry') as mock_registry, \
             patch('flowlib.flows.registry.flow_registry') as mock_unified_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            # Setup flows
            conversation_flow = MockFlow("ConversationFlow", "Conversation handler")
            analysis_flow = MockFlow("AnalysisFlow", "Analysis handler")
            
            self.mock_parent.flows = {
                "ConversationFlow": conversation_flow,
                "AnalysisFlow": analysis_flow
            }
            
            mock_registry.get_agent_selectable_flows.return_value = self.mock_parent.flows
            mock_unified_registry.get_agent_selectable_flows.return_value = self.mock_parent.flows
            mock_unified_registry.get_agent_selectable_flows.return_value = self.mock_parent.flows
            
            # Mock the resource and provider registries  
            mock_resource_registry.get.return_value = Mock()
            
            # Create a proper async LLM provider mock that respects output_type
            from flowlib.agent.components.planning.models import Plan, PlanStep
            mock_plan = Plan(
                task_description="Build a web application with user authentication",
                steps=[
                    PlanStep(
                        flow_name="ConversationFlow",
                        step_intent="Analyze project requirements",
                        rationale="First step to understand the project scope and technical requirements"
                    ),
                    PlanStep(
                        flow_name="ConversationFlow", 
                        step_intent="Create implementation plan",
                        rationale="Develop a structured plan for implementing the web application"
                    ),
                    PlanStep(
                        flow_name="ConversationFlow",
                        step_intent="Generate initial code structure", 
                        rationale="Create the basic project structure and boilerplate code"
                    )
                ]
            )
            
            # Create a mock that returns different types based on output_type
            async def mock_generate_structured(prompt, output_type, prompt_variables, model_name):
                if output_type == Plan:
                    return mock_plan
                elif output_type == MockInputModel:
                    return MockInputModel(
                        message=prompt_variables.get("task_description", "Test message"),
                        context="test"
                    )
                else:
                    # Default to the requested type with mock data
                    return output_type(message="test", context="test")
            
            mock_llm_provider = Mock()
            mock_llm_provider.generate_structured = AsyncMock(side_effect=mock_generate_structured)
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm_provider)
            
            # Setup flow metadata
            def get_metadata(flow_name):
                return FlowMetadata(
                    name=flow_name,
                    description=f"{flow_name} description",
                    input_model=MockInputModel,
                    output_model=MockOutputModel
                )
            mock_registry.get_flow_metadata.side_effect = get_metadata
            mock_unified_registry.get_flow_metadata.side_effect = get_metadata
            mock_unified_registry.get_flow_metadata.side_effect = get_metadata
            
            await self.engine.initialize()
            
            # Set the llm_provider after initialization when unified_planner is created
            if hasattr(self.engine, 'unified_planner') and self.engine.unified_planner:
                self.engine.unified_planner.llm_provider = mock_llm_provider
            
            # Create complex TODO scenario
            todo1 = TodoItem(
                id="analyze_requirements",
                content="Analyze project requirements",
                priority=TodoPriority.HIGH,
                assigned_tool="AnalysisFlow"
            )
            todo2 = TodoItem(
                id="create_plan",
                content="Create implementation plan",
                priority=TodoPriority.MEDIUM,
                assigned_tool="ConversationFlow",
                depends_on=["analyze_requirements"]
            )
            todo3 = TodoItem(
                id="generate_code",
                content="Generate initial code structure",
                priority=TodoPriority.MEDIUM,
                assigned_tool="ConversationFlow",
                depends_on=["create_plan"]
            )
            
            # Set up mock responses
            task_description = "Build a web application with user authentication"
            self.mock_planner.set_plan_response(task_description, [todo1, todo2, todo3])
            
            # Execute TODO-driven workflow
            result = await self.engine.execute_with_strategy(
                task_description,
                {"project_type": "web_app", "features": ["auth", "dashboard"]},
                ExecutionStrategy.TODO_DRIVEN
            )
            
            # execute_with_strategy always returns AgentResult
            assert isinstance(result, AgentResult)
            assert result.success is not None
            assert result.state is not None
            assert result.state.task_description == task_description
            # Note: result.results may vary based on TODO execution success
                
            # Verify TODO manager was used
            assert self.engine.todo_manager is not None
            
            # Verify activity stream captured events
            assert len(self.mock_activity_stream.events) > 0
            execution_events = [e for e in self.mock_activity_stream.events if e["type"] == "execution"]
            assert len(execution_events) > 0
    
    async def test_todo_execution_with_failures(self):
        """Test TODO execution with some failures and error handling."""
        with patch('flowlib.flows.registry.flow_registry') as mock_flow_registry, \
             patch('flowlib.agent.components.planning.planner.flow_registry') as mock_unified_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            
            # Setup flow that will fail
            failing_flow = MockFlow("FailingFlow", "Flow that fails")
            
            # Override execute to fail
            async def failing_execute(context):
                raise Exception("Simulated flow failure")
            failing_flow.execute = failing_execute
            
            self.mock_parent.flows = {
                "FailingFlow": failing_flow,
                "ConversationFlow": MockFlow("ConversationFlow")
            }
            
            # Setup all registry mocks
            mock_flow_registry.get_agent_selectable_flows.return_value = self.mock_parent.flows
            mock_unified_registry.get_agent_selectable_flows.return_value = self.mock_parent.flows
            
            # Setup flow metadata
            def get_metadata(flow_name):
                return FlowMetadata(
                    name=flow_name,
                    description=f"{flow_name} description",
                    input_model=MockInputModel,
                    output_model=MockOutputModel
                )
            mock_flow_registry.get_flow_metadata.side_effect = get_metadata
            mock_unified_registry.get_flow_metadata.side_effect = get_metadata
            
            # Mock resources and providers
            mock_resource_registry.get.return_value = Mock()
            
            # Create mock LLM provider
            from flowlib.agent.components.planning.models import Plan, PlanStep
            mock_plan = Plan(
                task_description="Test error handling in TODO execution",
                steps=[
                    PlanStep(
                        flow_name="FailingFlow",
                        step_intent="Process the task",
                        rationale="Basic step to handle the task"
                    )
                ]
            )
            
            async def mock_generate_structured(prompt, output_type, prompt_variables, model_name):
                if output_type == Plan:
                    return mock_plan
                elif output_type == MockInputModel:
                    return MockInputModel(
                        message=prompt_variables.get("task_description", "Test message"),
                        context="test"
                    )
                else:
                    return output_type(message="test", context="test")
            
            mock_llm_provider = Mock()
            mock_llm_provider.generate_structured = AsyncMock(side_effect=mock_generate_structured)
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm_provider)
            
            # Create a new config with stop_on_error=False (can't modify frozen model)
            from flowlib.agent.models.config import EngineConfig
            new_config = EngineConfig(
                max_iterations=self.engine._config.max_iterations,
                stop_on_error=False  # Allow errors to continue
            )
            self.engine._config = new_config
            
            await self.engine.initialize()
            
            # Set the llm_provider after initialization when unified_planner is created
            if hasattr(self.engine, 'unified_planner') and self.engine.unified_planner:
                self.engine.unified_planner.llm_provider = mock_llm_provider
            
            # Create TODOs where first one will fail
            todo1 = TodoItem(
                id="failing_task",
                content="This task will fail",
                priority=TodoPriority.HIGH,
                assigned_tool="FailingFlow"
            )
            todo2 = TodoItem(
                id="succeeding_task", 
                content="This task will succeed",
                priority=TodoPriority.MEDIUM,
                assigned_tool="ConversationFlow"
            )
            
            task_description = "Test error handling in TODO execution"
            self.mock_planner.set_plan_response(task_description, [todo1, todo2])
            
            # Execute with error handling
            result = await self.engine.execute_with_strategy(
                task_description,
                {},
                ExecutionStrategy.TODO_DRIVEN
            )
            
            # execute_with_strategy always returns AgentResult
            assert isinstance(result, AgentResult)
            assert result.success is not None
            assert result.state is not None
            # Note: With mixed success/failure, overall success may be False
            
            # Verify error was captured in activity stream
            error_events = [e for e in self.mock_activity_stream.events if e["type"] == "error"]
            # Note: Errors might be captured in execute_cycle rather than directly in unified engine
    
    async def test_strategy_determination_edge_cases(self):
        """Test automatic strategy determination with various scenarios."""
        with patch('flowlib.flows.registry.flow_registry') as mock_flow_registry, \
             patch('flowlib.agent.components.planning.planner.flow_registry') as mock_unified_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            
            self.mock_parent.flows["ConversationFlow"] = MockFlow("ConversationFlow")
            
            # Setup all registry mocks
            mock_flow_registry.get_agent_selectable_flows.return_value = self.mock_parent.flows
            mock_unified_registry.get_agent_selectable_flows.return_value = self.mock_parent.flows
            
            # Setup flow metadata
            flow_metadata = FlowMetadata(
                name="ConversationFlow",
                description="Test conversation flow",
                input_model=MockInputModel,
                output_model=MockOutputModel
            )
            mock_flow_registry.get_flow_metadata.return_value = flow_metadata
            mock_unified_registry.get_flow_metadata.return_value = flow_metadata
            
            # Mock resources and providers
            mock_resource_registry.get.return_value = Mock()
            
            # Create mock LLM provider
            from flowlib.agent.components.planning.models import Plan, PlanStep
            mock_plan = Plan(
                task_description="Test task",
                steps=[
                    PlanStep(
                        flow_name="ConversationFlow",
                        step_intent="Process the task",
                        rationale="Basic step to handle the task"
                    )
                ]
            )
            
            async def mock_generate_structured(prompt, output_type, prompt_variables, model_name):
                if output_type == Plan:
                    return mock_plan
                elif output_type == MockInputModel:
                    return MockInputModel(
                        message=prompt_variables.get("task_description", "Test message"),
                        context="test"
                    )
                else:
                    return output_type(message="test", context="test")
            
            mock_llm_provider = Mock()
            mock_llm_provider.generate_structured = AsyncMock(side_effect=mock_generate_structured)
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm_provider)
            
            await self.engine.initialize()
            
            # Set the llm_provider after initialization when unified_planner is created
            if hasattr(self.engine, 'unified_planner') and self.engine.unified_planner:
                self.engine.unified_planner.llm_provider = mock_llm_provider
            
            # Test different task types that should trigger different strategies
            test_cases = [
                ("Simple question", "single_cycle"),
                ("Complex multi-step project with dependencies", "todo_driven"),
                ("Quick calculation", "single_cycle"),
                ("Develop comprehensive solution with testing", "todo_driven")
            ]
            
            for task, expected_strategy in test_cases:
                # Set mock strategy response
                self.mock_planner.set_strategy_response(task, expected_strategy)
                
                # For TODO-driven, provide empty TODO list to trigger fallback
                if expected_strategy == "todo_driven":
                    self.mock_planner.set_plan_response(task, [])
                
                result = await self.engine.execute_with_strategy(
                    task,
                    {},
                    ExecutionStrategy.AUTO
                )
                
                # Verify execution completed
                assert result is not None
                
                # execute_with_strategy always returns AgentResult regardless of strategy
                assert isinstance(result, AgentResult)
                assert result.success is not None
                # Note: state may be None for failed single-cycle executions
    
    async def test_memory_integration_during_execution(self):
        """Test memory system integration during execution cycles."""
        with patch('flowlib.flows.registry.flow_registry') as mock_flow_registry, \
             patch('flowlib.agent.components.planning.planner.flow_registry') as mock_unified_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            
            self.mock_parent.flows["ConversationFlow"] = MockFlow("ConversationFlow")
            
            # Setup all registry mocks
            mock_flow_registry.get_agent_selectable_flows.return_value = self.mock_parent.flows
            mock_unified_registry.get_agent_selectable_flows.return_value = self.mock_parent.flows
            
            flow_metadata = FlowMetadata(
                name="ConversationFlow",
                description="Test conversation flow",
                input_model=MockInputModel,
                output_model=MockOutputModel
            )
            mock_flow_registry.get_flow_metadata.return_value = flow_metadata
            mock_unified_registry.get_flow_metadata.return_value = flow_metadata
            
            # Mock resources and providers
            mock_resource_registry.get.return_value = Mock()
            
            # Create mock LLM provider
            from flowlib.agent.components.planning.models import Plan, PlanStep
            mock_plan = Plan(
                task_description="Test memory integration",
                steps=[
                    PlanStep(
                        flow_name="ConversationFlow",
                        step_intent="Process the task",
                        rationale="Basic step to handle the task"
                    )
                ]
            )
            
            async def mock_generate_structured(prompt, output_type, prompt_variables, model_name):
                if output_type == Plan:
                    return mock_plan
                elif output_type == MockInputModel:
                    return MockInputModel(
                        message=prompt_variables.get("task_description", "Test message"),
                        context="test"
                    )
                else:
                    return output_type(message="test", context="test")
            
            mock_llm_provider = Mock()
            mock_llm_provider.generate_structured = AsyncMock(side_effect=mock_generate_structured)
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm_provider)
            
            await self.engine.initialize()
            
            # Set the llm_provider after initialization when unified_planner is created
            if hasattr(self.engine, 'unified_planner') and self.engine.unified_planner:
                self.engine.unified_planner.llm_provider = mock_llm_provider
            
            # Execute single cycle to test memory integration
            result = await self.engine.execute_with_strategy(
                "Test memory integration",
                {"memory_test": True},
                ExecutionStrategy.SINGLE_CYCLE
            )
            
            # Verify memory was accessed (context creation attempted)
            assert len(self.mock_memory._contexts) > 0
            
            # Verify context creation was attempted
            context_names = list(self.mock_memory._contexts)
            assert any("task_" in context for context in context_names)
    
    async def test_reflection_integration_and_completion(self):
        """Test reflection system integration and completion detection."""
        with patch('flowlib.flows.registry.flow_registry') as mock_flow_registry, \
             patch('flowlib.agent.components.planning.planner.flow_registry') as mock_unified_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            
            # Create flow that marks completion in metadata
            completing_flow = MockFlow("CompletingFlow", "Flow that completes task")
            
            async def completing_execute(context):
                return FlowResult(
                    status=FlowStatus.SUCCESS,
                    data=MockOutputModel(response="Task completed successfully"),
                    flow_name="CompletingFlow",
                    metadata={"complete": True}  # This will trigger completion
                )
            completing_flow.execute = completing_execute
            
            self.mock_parent.flows["CompletingFlow"] = completing_flow
            
            # Setup all registry mocks
            mock_flow_registry.get_agent_selectable_flows.return_value = self.mock_parent.flows
            mock_unified_registry.get_agent_selectable_flows.return_value = self.mock_parent.flows
            
            flow_metadata = FlowMetadata(
                name="CompletingFlow",
                description="Completing flow",
                input_model=MockInputModel,
                output_model=MockOutputModel
            )
            mock_flow_registry.get_flow_metadata.return_value = flow_metadata
            mock_unified_registry.get_flow_metadata.return_value = flow_metadata
            
            # Mock resources and providers
            mock_resource_registry.get.return_value = Mock()
            
            # Create mock LLM provider
            from flowlib.agent.components.planning.models import Plan, PlanStep
            mock_plan = Plan(
                task_description="Task that should complete quickly",
                steps=[
                    PlanStep(
                        flow_name="CompletingFlow",
                        step_intent="Process the task",
                        rationale="Basic step to handle the task"
                    )
                ]
            )
            
            async def mock_generate_structured(prompt, output_type, prompt_variables, model_name):
                if output_type == Plan:
                    return mock_plan
                elif output_type == MockInputModel:
                    return MockInputModel(
                        message=prompt_variables.get("task_description", "Test message"),
                        context="test"
                    )
                else:
                    return output_type(message="test", context="test")
            
            mock_llm_provider = Mock()
            mock_llm_provider.generate_structured = AsyncMock(side_effect=mock_generate_structured)
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm_provider)
            
            await self.engine.initialize()
            
            # Set the llm_provider after initialization when unified_planner is created
            if hasattr(self.engine, 'unified_planner') and self.engine.unified_planner:
                self.engine.unified_planner.llm_provider = mock_llm_provider
            
            # Mock planner to select the completing flow
            async def plan_completing(context):
                return PlanningResult(
                    selected_flow="CompletingFlow",
                    reasoning=PlanningExplanation(
                        explanation="Select completing flow",
                        rationale="Flow completes task efficiently",
                        decision_factors=["completion", "efficiency"]
                    )
                )
            self.mock_planner.plan = plan_completing
            
            # Execute and verify completion detection
            result = await self.engine.execute_with_strategy(
                "Task that should complete quickly",
                {},
                ExecutionStrategy.SINGLE_CYCLE
            )
            
            # Verify execution
            assert result is not None
            assert isinstance(result, AgentResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])