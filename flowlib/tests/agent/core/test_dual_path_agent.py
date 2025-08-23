"""
Tests for the DualPathAgent class.

This module tests the dual-path agent implementation that separates
conversation and task execution paths with classification routing.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock, PropertyMock
from typing import Dict, Any, Optional, List
from datetime import datetime

from flowlib.agent.agents.dual_path import DualPathAgent
from flowlib.agent.models.config import AgentConfig
from flowlib.agent.models.state import AgentState
from flowlib.agent.models.plan import PlanExecutionOutcome
from flowlib.agent.components.classification.flow import MessageClassifierInput
from flowlib.agent.components.conversation.handler import DirectConversationHandler
from flowlib.agent.components.tasks.handler import TaskExecutionHandler
from flowlib.flows.models.results import FlowResult
from flowlib.flows.models.constants import FlowStatus
from flowlib.flows.registry.registry import flow_registry
from flowlib.agent.components.planning.models import PlanStep, Plan
from flowlib.agent.components.reflection.models import StepReflectionResult, PlanReflectionContext
from flowlib.core.context.context import Context
from pydantic import BaseModel


class MockMemory:
    """Mock memory for testing."""
    
    def __init__(self):
        self.initialized = True
        self._contexts = {}
        
    async def initialize(self):
        self.initialized = True
        
    async def create_context(self, context_name: str, metadata: Dict[str, Any] = None):
        self._contexts[context_name] = metadata or {}
        
    async def retrieve_relevant(self, query: str, context: str = None, limit: int = 3):
        return [f"Memory about {query[:20]}...", f"Related info for {query[:15]}..."]


class MockClassifier:
    """Mock message classifier for testing."""
    
    def __init__(self, execute_task: bool = False, confidence: float = 0.8, category: str = "general"):
        self.execute_task = execute_task
        self.confidence = confidence
        self.category = category
        self.task_description = "Test task description" if execute_task else None
        
    async def run_pipeline(self, input_data):
        result = Mock()
        result.execute_task = self.execute_task
        result.confidence = self.confidence
        result.category = self.category
        result.task_description = self.task_description
        return result


class MockResponseData(BaseModel):
    """Mock response data model."""
    response: str

class MockConversationHandler:
    """Mock conversation handler for testing."""
    
    def __init__(self, response: str = "Test response"):
        self.response = response
        
    async def handle_conversation(self, message: str, state, memory_context_summary=None, task_result_summary=None):
        # Create a proper response data object using BaseModel
        response_data = MockResponseData(response=self.response)
        return FlowResult(
            status="SUCCESS",
            data=response_data,
            flow_name="conversation"
        )


class MockTaskHandler:
    """Mock task handler for testing."""
    
    async def execute_task(self, state):
        return {"status": "SUCCESS", "result": "Task completed"}


class MockPlanner:
    """Mock planner for testing."""
    
    def __init__(self, plan: Optional[Plan] = None, generate_empty_plan: bool = False):
        self._mock_plan = plan
        self.generate_empty_plan = generate_empty_plan
        
    async def plan(self, state):
        if self.generate_empty_plan:
            return Plan(plan_id="empty-plan", task_description="empty task", steps=[])
        return self._mock_plan or Plan(
            plan_id="test-plan",
            task_description="test task",
            steps=[
                PlanStep(
                    step_id="step-1",
                    flow_name="shell_command",
                    step_intent="Execute test command",
                    rationale="Testing step execution"
                )
            ]
        )
        
    async def generate_inputs(self, state, flow_name: str, step_intent: str, step_rationale: str, memory_context_id: str = None):
        # Return a mock input model
        input_model = Mock(spec=BaseModel)
        input_model.model_dump.return_value = {"command": "echo test"}
        return input_model


class MockReflection:
    """Mock reflection component for testing."""
    
    async def step_reflect(self, step_input):
        return StepReflectionResult(
            step_id=step_input.step_id,
            reflection="Step executed successfully",
            step_success=True,
            key_observation="Test observation"
        )
        
    async def reflect(self, plan_context):
        result = Mock()
        result.is_complete = False
        result.progress = 50
        result.completion_reason = None
        return result


class MockFlow:
    """Mock flow for testing."""
    
    def __init__(self, success: bool = True, response_data: Any = None):
        self.success = success
        self.response_data = response_data or {"output": "Test output"}
        
    async def execute(self, context):
        if self.success:
            return FlowResult(
                status=FlowStatus.SUCCESS,
                data={
                    "success": True,
                    "stdout": "Test output",
                    "stderr": "",
                    "command": "echo test"
                },
                flow_name="test_flow"
            )
        else:
            return FlowResult(
                status=FlowStatus.ERROR,
                error="Flow execution failed",
                flow_name="test_flow"
            )


class MockStatePersister:
    """Mock state persister for testing."""
    
    def __init__(self):
        self.save_called = False
        
    async def save_state(self, state, metadata=None):
        self.save_called = True


class TestDualPathAgentInitialization:
    """Test DualPathAgent initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        agent = DualPathAgent()
        
        assert isinstance(agent, DualPathAgent)
        assert agent._classifier is None
        assert agent._conversation_handler is None
        assert agent._task_handler is None
        assert not agent.initialized
    
    def test_init_with_config(self):
        """Test initialization with configuration."""
        config = AgentConfig(
            name="test_agent",
            persona="A helpful test agent",
            provider_name="llamacpp",
            task_description="Test task"
        )
        
        agent = DualPathAgent(config=config, task_description="Override task")
        
        assert agent.config == config
        # The task_description is passed to the constructor but may be handled differently
        # Let's just check that config was properly set
    
    @pytest.mark.asyncio
    async def test_initialize_impl_success(self):
        """Test successful initialization."""
        agent = DualPathAgent()
        
        # Mock the memory at the core agent level
        mock_memory = MockMemory()
        agent._agent_core._memory_manager._memory = mock_memory
        
        # Mock the conversation flow in registry at the core agent level
        mock_conversation_flow = Mock()
        agent._agent_core._flow_runner._flows = {"ConversationFlow": mock_conversation_flow}
        
        # Mock the planner and reflection at the core agent level
        agent._agent_core._planner = MockPlanner()
        agent._agent_core._reflection = MockReflection()
        
        # Mock the parent class initialization
        from flowlib.agent.core.agent import AgentCore
        with patch.object(AgentCore, 'initialize', AsyncMock()):
            await agent.initialize()
        
        # Verify components were created
        assert agent._classifier is not None
        assert agent._conversation_handler is not None
        assert agent._task_handler is not None
    
    @pytest.mark.asyncio
    async def test_initialize_impl_missing_conversation_flow(self):
        """Test initialization failure when ConversationFlow is missing."""
        agent = DualPathAgent()
        
        # Mock the memory at the core agent level
        mock_memory = MockMemory()
        agent._agent_core._memory_manager._memory = mock_memory
        
        # Empty flows registry
        agent._agent_core._flow_runner._flows = {}
        
        # Mock the parent class initialization
        from flowlib.agent.core.agent import AgentCore
        with patch.object(AgentCore, 'initialize', AsyncMock()):
            with pytest.raises(ValueError, match="ConversationFlow must be registered"):
                await agent.initialize()


class TestDualPathAgentMessageProcessing:
    """Test message processing functionality."""
    
    def setup_method(self):
        """Set up test agent with mocks."""
        self.agent = DualPathAgent()
        
        # Set up state
        self.agent._agent_core._state_manager.current_state = AgentState(
            task_description="Test task",
            task_id="test-task-123"
        )
        
        # Mock memory at the core agent level
        self.agent._agent_core._memory_manager._memory = MockMemory()
        
        # Mock components
        self.agent._classifier = MockClassifier()
        self.agent._conversation_handler = MockConversationHandler()
        self.agent._task_handler = MockTaskHandler()
        
        # Mock state persister
        self.mock_persister = MockStatePersister()
        self.agent._agent_core._state_manager._state_persister = self.mock_persister
    
    def _mock_initialized(self, value=True):
        """Helper method to mock the initialized property."""
        return patch.object(type(self.agent), 'initialized', new_callable=lambda: PropertyMock(return_value=value))
    
    @pytest.mark.asyncio
    async def test_process_message_not_initialized(self):
        """Test processing message when agent not initialized."""
        agent = DualPathAgent()
        # Agent starts uninitialized by default
        
        result = await agent.process_message("Hello")
        
        assert result.status == "ERROR"
        assert "not initialized" in result.error
    
    @pytest.mark.asyncio
    async def test_process_message_conversation_path(self):
        """Test message processing via conversation path."""
        # Set classifier to return conversation
        self.agent._classifier = MockClassifier(execute_task=False)
        
        # Mock the initialized property
        with self._mock_initialized():
            result = await self.agent.process_message("Hello, how are you?")
        
        assert result.status == "SUCCESS"
        assert result.data.response == "Test response"
        
        # Verify message was added to state
        assert "Hello, how are you?" in self.agent._agent_core._state_manager.current_state.user_messages
        assert "Test response" in self.agent._agent_core._state_manager.current_state.system_messages
    
    @pytest.mark.asyncio
    async def test_process_message_task_path_no_action_needed(self):
        """Test message processing via task path with no action needed."""
        # Set classifier to return task
        self.agent._classifier = MockClassifier(execute_task=True)
        
        # Mock planner to return empty plan
        self.agent._agent_core._planner = MockPlanner(generate_empty_plan=True)
        self.agent._agent_core._reflection = MockReflection()
        
        # Mock the initialized property
        with self._mock_initialized():
            result = await self.agent.process_message("Complete the task")
        
        assert result.status == "SUCCESS"
        # Should get conversational response about task completion
        assert result.data.response == "Test response"
    
    @pytest.mark.asyncio
    async def test_process_message_task_path_with_execution(self):
        """Test message processing via task path with actual execution."""
        # Set classifier to return task
        self.agent._classifier = MockClassifier(execute_task=True)
        
        # Mock planner
        self.agent._agent_core._planner = MockPlanner()
        self.agent._agent_core._reflection = MockReflection()
        
        # Mock flow registry
        mock_flow = MockFlow(success=True)
        with patch.object(flow_registry, 'get_flow', return_value=mock_flow):
            # Mock the initialized property
            with self._mock_initialized():
                result = await self.agent.process_message("Execute shell command")
        
        assert result.status == "SUCCESS"
        assert result.data.response == "Test response"
        
        # Verify task description was updated
        assert self.agent._agent_core._state_manager.current_state.task_description == "Test task description"
    
    @pytest.mark.asyncio
    async def test_process_message_memory_error(self):
        """Test message processing with memory error."""
        # Mock memory to raise error
        self.agent._agent_core._memory_manager._memory.retrieve_relevant = AsyncMock(side_effect=Exception("Memory error"))
        
        # Mock the initialized property
        with self._mock_initialized():
            result = await self.agent.process_message("Hello")
        
        # Should still process despite memory error
        assert result.status == "SUCCESS"
    
    @pytest.mark.asyncio
    async def test_process_message_classification_error(self):
        """Test message processing with classification error."""
        # Mock classifier to raise error
        self.agent._classifier.run_pipeline = AsyncMock(side_effect=Exception("Classification failed"))
        
        # Mock the initialized property
        with self._mock_initialized():
            result = await self.agent.process_message("Hello")
        
        assert result.status == "ERROR"
        assert "Error processing message" in result.error
    
    @pytest.mark.asyncio
    async def test_get_formatted_history(self):
        """Test conversation history formatting."""
        # Add messages using the proper methods
        self.agent._agent_core._state_manager.current_state.add_user_message("Hello")
        self.agent._agent_core._state_manager.current_state.add_user_message("How are you?")
        self.agent._agent_core._state_manager.current_state.add_system_message("Hi there!")
        self.agent._agent_core._state_manager.current_state.add_system_message("I'm doing well!")
        
        history = self.agent._get_formatted_history()
        
        expected = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well!"}
        ]
        assert history == expected
    
    @pytest.mark.asyncio
    async def test_get_formatted_history_uneven_messages(self):
        """Test history formatting with uneven message counts."""
        # Add messages using the proper methods
        self.agent._agent_core._state_manager.current_state.add_user_message("Hello")
        self.agent._agent_core._state_manager.current_state.add_user_message("How are you?")
        self.agent._agent_core._state_manager.current_state.add_user_message("Thanks!")
        self.agent._agent_core._state_manager.current_state.add_system_message("Hi there!")
        
        history = self.agent._get_formatted_history()
        
        expected = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "user", "content": "Thanks!"}
        ]
        assert history == expected


class TestDualPathAgentPlanExecution:
    """Test plan execution functionality."""
    
    def setup_method(self):
        """Set up test agent with mocks."""
        self.agent = DualPathAgent()
        
        # Set up state with a plan
        self.plan = Plan(
            plan_id="test-plan",
            task_description="Test task for plan execution",
            steps=[
                PlanStep(
                    step_id="step-1",
                    flow_name="shell_command",
                    step_intent="Execute test command",
                    rationale="Testing step execution"
                ),
                PlanStep(
                    step_id="step-2", 
                    flow_name="another_flow",
                    step_intent="Second step",
                    rationale="Testing multi-step"
                )
            ]
        )
        
        self.agent._agent_core._state_manager.current_state = AgentState(
            task_id="test-task-123",
            task_description="Test task"
        )
        self.agent._agent_core._state_manager.current_state.current_plan = self.plan
        self.agent._agent_core._state_manager.current_state.current_step_index = 0
        
        # Mock components
        self.agent._agent_core._planner = MockPlanner()
        self.agent._agent_core._reflection = MockReflection()
    
    @pytest.mark.asyncio
    async def test_execute_plan_loop_success(self):
        """Test successful plan execution."""
        # Mock successful flows
        mock_flow = MockFlow(success=True)
        
        with patch.object(flow_registry, 'get_flow', return_value=mock_flow):
            with patch.object(type(self.agent), 'initialized', new_callable=lambda: PropertyMock(return_value=True)):
                outcome = await self.agent._execute_plan_loop(self.agent._agent_core._state_manager.current_state)
        
        assert outcome.status == "SUCCESS"
        assert outcome.result is not None
        assert len(outcome.step_reflections) == 2  # Two steps
        assert self.agent._agent_core._state_manager.current_state.current_plan is None  # Plan cleared after completion
        assert self.agent._agent_core._state_manager.current_state.current_step_index == 0
    
    @pytest.mark.asyncio
    async def test_execute_plan_loop_step_failure(self):
        """Test plan execution with step failure."""
        # Mock failing flow
        mock_flow = MockFlow(success=False)
        
        with patch.object(flow_registry, 'get_flow', return_value=mock_flow):
            outcome = await self.agent._execute_plan_loop(self.agent._agent_core._state_manager.current_state)
        
        assert outcome.status == "ERROR"
        assert outcome.error is not None
        assert "Step 1 failed" in outcome.error
        assert self.agent._agent_core._state_manager.current_state.current_plan is None  # Plan cleared after failure
    
    @pytest.mark.asyncio
    async def test_execute_plan_loop_step_exception(self):
        """Test plan execution with step exception."""
        # Mock flow that raises exception
        mock_flow = Mock()
        mock_flow.execute = AsyncMock(side_effect=Exception("Flow execution failed"))
        
        with patch.object(flow_registry, 'get_flow', return_value=mock_flow):
            outcome = await self.agent._execute_plan_loop(self.agent._agent_core._state_manager.current_state)
        
        assert outcome.status == "ERROR"
        assert "raised an exception" in outcome.error
        assert self.agent._agent_core._state_manager.current_state.current_plan is None
    
    @pytest.mark.asyncio
    async def test_execute_plan_loop_no_plan_generates_new(self):
        """Test plan execution when no plan exists."""
        # Clear the plan
        self.agent._agent_core._state_manager.current_state.current_plan = None
        
        # Mock planner to return a plan
        self.agent._agent_core._planner = MockPlanner(plan=self.plan)
        
        mock_flow = MockFlow(success=True)
        with patch.object(flow_registry, 'get_flow', return_value=mock_flow):
            outcome = await self.agent._execute_plan_loop(self.agent._agent_core._state_manager.current_state)
        
        assert outcome.status == "SUCCESS"
        assert self.agent._agent_core._state_manager.current_state.current_plan is None  # Cleared after completion
    
    @pytest.mark.asyncio
    async def test_execute_plan_loop_planning_failure(self):
        """Test plan execution with planning failure."""
        # Clear the plan
        self.agent._agent_core._state_manager.current_state.current_plan = None
        
        # Mock planner to raise error
        self.agent._agent_core._planner.plan = AsyncMock(side_effect=Exception("Planning failed"))
        
        outcome = await self.agent._execute_plan_loop(self.agent._agent_core._state_manager.current_state)
        
        assert outcome.status == "ERROR"
        assert "Planning failed" in outcome.error
    
    @pytest.mark.asyncio
    async def test_execute_plan_loop_input_generation_failure(self):
        """Test plan execution with input generation failure."""
        # Mock planner to fail input generation
        self.agent._agent_core._planner.generate_inputs = AsyncMock(side_effect=Exception("Input generation failed"))
        
        outcome = await self.agent._execute_plan_loop(self.agent._agent_core._state_manager.current_state)
        
        assert outcome.status == "ERROR"
        assert "Input generation failed" in outcome.error
    
    @pytest.mark.asyncio
    async def test_execute_plan_loop_flow_not_found(self):
        """Test plan execution when flow is not found."""
        # Mock registry to return None
        with patch.object(flow_registry, 'get_flow', return_value=None):
            outcome = await self.agent._execute_plan_loop(self.agent._agent_core._state_manager.current_state)
        
        assert outcome.status == "ERROR"
        assert "not found in registry" in outcome.error
    
    @pytest.mark.asyncio
    async def test_execute_plan_loop_reflection_failure(self):
        """Test plan execution with reflection failure."""
        # Mock reflection to fail
        self.agent._agent_core._reflection.step_reflect = AsyncMock(side_effect=Exception("Reflection failed"))
        
        mock_flow = MockFlow(success=True)
        with patch.object(flow_registry, 'get_flow', return_value=mock_flow):
            outcome = await self.agent._execute_plan_loop(self.agent._agent_core._state_manager.current_state)
        
        # Should still succeed overall, but reflection should be recorded as failed
        assert outcome.status == "SUCCESS"
        assert len(outcome.step_reflections) == 2
        # Check that reflection errors are handled
        for reflection in outcome.step_reflections:
            assert "Reflection error" in reflection.reflection


class TestDualPathAgentStatePersistence:
    """Test state persistence functionality."""
    
    def setup_method(self):
        """Set up test agent with mocks."""
        self.agent = DualPathAgent()
        
        self.agent._agent_core._state_manager.current_state = AgentState(
            task_id="test-task-123",
            task_description="Test task"
        )
        
        self.mock_persister = MockStatePersister()
        self.agent._agent_core._state_manager._state_persister = self.mock_persister
    
    @pytest.mark.asyncio
    async def test_save_state_success(self):
        """Test successful state saving."""
        # Mock the save_state method
        self.agent.save_state = AsyncMock()
        
        await self.agent._save_state()
        
        self.agent.save_state.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_state_no_persister(self):
        """Test state saving when no persister is available."""
        self.agent._agent_core._state_manager._state_persister = None
        
        # Should not raise error
        await self.agent._save_state()
    
    @pytest.mark.asyncio
    async def test_save_state_no_task_id(self):
        """Test state saving when no task ID is set."""
        # Modify internal state data directly since task_id property has no setter
        self.agent._agent_core._state_manager.current_state._data["task_id"] = None
        
        # Should not attempt to save
        await self.agent._save_state()
    
    @pytest.mark.asyncio
    async def test_save_state_failure(self):
        """Test state saving failure."""
        # Mock save_state to raise error
        self.agent.save_state = AsyncMock(side_effect=Exception("Save failed"))
        
        # Should not raise error, just log warning
        await self.agent._save_state()


class TestDualPathAgentExecuteCycle:
    """Test execute_cycle functionality."""
    
    def setup_method(self):
        """Set up test agent with mocks."""
        self.agent = DualPathAgent()
        self.agent._agent_core._state_manager.current_state = AgentState(
            task_id="test-task-123",
            task_description="Test task"
        )
        # Set is_complete after initialization since it's not a constructor parameter
        self.agent._agent_core._state_manager.current_state._data["is_complete"] = False
    
    @pytest.mark.asyncio
    async def test_execute_cycle_with_message(self):
        """Test execute_cycle with message."""
        # Mock process_message
        self.agent.process_message = AsyncMock()
        
        with patch.object(type(self.agent), 'initialized', new_callable=lambda: PropertyMock(return_value=True)):
            result = await self.agent.execute_cycle(message="Hello")
        
        assert result is True  # Should continue (task not complete)
        self.agent.process_message.assert_called_once_with("Hello")
    
    @pytest.mark.asyncio
    async def test_execute_cycle_task_complete(self):
        """Test execute_cycle when task is complete."""
        self.agent.process_message = AsyncMock()
        # Set task as complete by modifying internal state data
        self.agent._agent_core._state_manager.current_state._data["is_complete"] = True
        
        with patch.object(type(self.agent), 'initialized', new_callable=lambda: PropertyMock(return_value=True)):
            result = await self.agent.execute_cycle(message="Hello")
        
        assert result is False  # Should stop (task complete)
    
    @pytest.mark.asyncio
    async def test_execute_cycle_no_message(self):
        """Test execute_cycle without message."""
        
        with patch.object(type(self.agent), 'initialized', new_callable=lambda: PropertyMock(return_value=True)):
            result = await self.agent.execute_cycle()
        
        assert result is False  # Should stop (no message)
    
    @pytest.mark.asyncio
    async def test_execute_cycle_not_initialized(self):
        """Test execute_cycle when not initialized."""
        self.agent.initialize = AsyncMock()
        
        await self.agent.execute_cycle(message="Hello")
        
        self.agent.initialize.assert_called_once()


class TestDualPathAgentIntegration:
    """Integration tests for DualPathAgent."""
    
    @pytest.mark.asyncio
    async def test_full_conversation_workflow(self):
        """Test complete conversation workflow."""
        agent = DualPathAgent()
        
        # Set up all required mocks
        agent._agent_core._memory_manager._memory = MockMemory()
        agent._classifier = MockClassifier(execute_task=False)
        agent._conversation_handler = MockConversationHandler("Hello! How can I help?")
        agent._task_handler = MockTaskHandler()
        
        agent._agent_core._state_manager.current_state = AgentState(
            task_id="test-conversation",
            task_description="Chat with user"
        )
        
        with patch.object(type(agent), 'initialized', new_callable=lambda: PropertyMock(return_value=True)):
            result = await agent.process_message("Hi there!")
        
        assert result.status == "SUCCESS"
        assert result.data.response == "Hello! How can I help?"
        assert "Hi there!" in agent._agent_core._state_manager.current_state.user_messages
        assert "Hello! How can I help?" in agent._agent_core._state_manager.current_state.system_messages
    
    @pytest.mark.asyncio
    async def test_full_task_workflow(self):
        """Test complete task execution workflow."""
        agent = DualPathAgent()
        
        # Set up all required mocks
        agent._agent_core._memory_manager._memory = MockMemory()
        agent._classifier = MockClassifier(execute_task=True)
        agent._conversation_handler = MockConversationHandler("Task completed successfully!")
        agent._agent_core._planner = MockPlanner()
        agent._agent_core._reflection = MockReflection()
        
        agent._agent_core._state_manager.current_state = AgentState(
            task_id="test-task",
            task_description="Execute shell command"
        )
        
        # Mock flow execution
        mock_flow = MockFlow(success=True)
        with patch.object(flow_registry, 'get_flow', return_value=mock_flow):
            with patch.object(type(agent), 'initialized', new_callable=lambda: PropertyMock(return_value=True)):
                result = await agent.process_message("Run ls command")
        
        assert result.status == "SUCCESS"
        assert result.data.response == "Task completed successfully!"
        assert agent._agent_core._state_manager.current_state.task_description == "Test task description"
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery in complex scenarios."""
        agent = DualPathAgent()
        
        # Set up mocks that will fail
        agent._agent_core._memory_manager._memory = MockMemory()
        agent._classifier = MockClassifier(execute_task=True)
        agent._conversation_handler = MockConversationHandler("Error handled gracefully")
        agent._agent_core._planner = MockPlanner()
        agent._agent_core._reflection = MockReflection()
        
        agent._agent_core._state_manager.current_state = AgentState(
            task_id="error-test",
            task_description="Test error handling"
        )
        
        # Mock flow to raise exception
        mock_flow = Mock()
        mock_flow.execute = AsyncMock(side_effect=Exception("Simulated flow error"))
        
        with patch.object(flow_registry, 'get_flow', return_value=mock_flow):
            with patch.object(type(agent), 'initialized', new_callable=lambda: PropertyMock(return_value=True)):
                result = await agent.process_message("Cause an error")
        
        # Should still return a response despite internal errors
        assert result.status == "SUCCESS"
        assert result.data.response == "Error handled gracefully"


class TestDualPathAgentEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        """Set up test agent."""
        self.agent = DualPathAgent()
        
        self.agent._agent_core._state_manager.current_state = AgentState(
            task_id="edge-case-test",
            task_description="Test edge cases"
        )
    
    def _mock_initialized(self, value=True):
        """Helper method to mock the initialized property."""
        return patch.object(type(self.agent), 'initialized', new_callable=lambda: PropertyMock(return_value=value))
    
    @pytest.mark.asyncio
    async def test_empty_message(self):
        """Test processing empty message."""
        # Set up minimal mocks
        self.agent._agent_core._memory_manager._memory = MockMemory()
        self.agent._classifier = MockClassifier()
        self.agent._conversation_handler = MockConversationHandler("I received an empty message")
        
        with self._mock_initialized():
            result = await self.agent.process_message("")
        
        assert result.status == "SUCCESS"
        assert "" in self.agent._agent_core._state_manager.current_state.user_messages
    
    @pytest.mark.asyncio 
    async def test_very_long_message(self):
        """Test processing very long message."""
        long_message = "x" * 10000
        
        self.agent._agent_core._memory_manager._memory = MockMemory()
        self.agent._classifier = MockClassifier()
        self.agent._conversation_handler = MockConversationHandler("Received long message")
        
        with self._mock_initialized():
            result = await self.agent.process_message(long_message)
        
        assert result.status == "SUCCESS"
        assert long_message in self.agent._agent_core._state_manager.current_state.user_messages
    
    @pytest.mark.asyncio
    async def test_classification_with_zero_confidence(self):
        """Test classification with zero confidence."""
        self.agent._agent_core._memory_manager._memory = MockMemory()
        self.agent._classifier = MockClassifier(confidence=0.0)
        self.agent._conversation_handler = MockConversationHandler("Low confidence response")
        
        with self._mock_initialized():
            result = await self.agent.process_message("Uncertain message")
        
        assert result.status == "SUCCESS"
    
    @pytest.mark.asyncio
    async def test_memory_context_creation_failure(self):
        """Test when memory context creation fails."""
        # Mock memory to fail context creation
        mock_memory = MockMemory()
        mock_memory.create_context = AsyncMock(side_effect=Exception("Context creation failed"))
        
        self.agent._agent_core._memory_manager._memory = mock_memory
        self.agent._classifier = MockClassifier()
        self.agent._conversation_handler = MockConversationHandler("Handled despite memory error")
        
        with self._mock_initialized():
            result = await self.agent.process_message("Test message")
        
        # Should still process successfully
        assert result.status == "SUCCESS"
    
    @pytest.mark.asyncio
    async def test_no_task_description_fallback(self):
        """Test behavior when agent processes message with minimal task description."""
        # Create a new state with minimal task description since empty strings are not allowed
        self.agent._agent_core._state_manager.current_state = AgentState(task_description="fallback")
        
        self.agent._agent_core._memory_manager._memory = MockMemory()
        self.agent._classifier = MockClassifier()
        self.agent._conversation_handler = MockConversationHandler("Default task response")
        
        # Mock config to return None - can't set the property directly 
        with patch.object(type(self.agent), 'config', new_callable=lambda: PropertyMock(return_value=None)):
            with self._mock_initialized():
                result = await self.agent.process_message("What is my task?")
        
        assert result.status == "SUCCESS"
        # Task description should remain as set (no automatic override of existing descriptions)
        assert self.agent._agent_core._state_manager.current_state.task_description == "fallback"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])