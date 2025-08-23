"""Tests for Agent Reflection System implementation."""

import pytest
import pytest_asyncio
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime
from pydantic import BaseModel

from flowlib.agent.components.reflection.base import AgentReflection
from flowlib.agent.components.reflection.models import (
    ReflectionResult, 
    ReflectionInput, 
    StepReflectionResult, 
    StepReflectionInput,
    PlanReflectionContext
)
from flowlib.agent.components.reflection.interfaces import ReflectionInterface
from flowlib.agent.components.reflection.prompts import DefaultReflectionPrompt, TaskCompletionReflectionPrompt
from flowlib.agent.components.reflection.prompts.step_reflection import DefaultStepReflectionPrompt
from flowlib.agent.models.config import ReflectionConfig
from flowlib.agent.models.state import AgentState
from flowlib.agent.core.errors import ReflectionError, NotInitializedError
from flowlib.flows.models.results import FlowResult, FlowStatus
from flowlib.providers.llm.base import LLMProvider


class TestReflectionModels:
    """Test reflection model classes."""
    
    def test_reflection_result_creation(self):
        """Test ReflectionResult creation."""
        result = ReflectionResult(
            reflection="Test reflection analysis",
            progress=75,
            is_complete=False,
            completion_reason=None,
            insights=["Insight 1", "Insight 2"]
        )
        
        assert result.reflection == "Test reflection analysis"
        assert result.progress == 75
        assert result.is_complete is False
        assert result.completion_reason is None
        assert result.insights == ["Insight 1", "Insight 2"]
    
    def test_reflection_result_defaults(self):
        """Test ReflectionResult with default values."""
        result = ReflectionResult(reflection="Test reflection")
        
        assert result.reflection == "Test reflection"
        assert result.progress == 0
        assert result.is_complete is False
        assert result.completion_reason is None
        assert result.insights is None
    
    def test_step_reflection_result_creation(self):
        """Test StepReflectionResult creation."""
        result = StepReflectionResult(
            step_id="step_123",
            reflection="Step completed successfully",
            step_success=True,
            key_observation="Important data obtained"
        )
        
        assert result.step_id == "step_123"
        assert result.reflection == "Step completed successfully"
        assert result.step_success is True
        assert result.key_observation == "Important data obtained"
    
    def test_reflection_input_creation(self):
        """Test ReflectionInput creation."""
        flow_result = FlowResult(
            status=FlowStatus.SUCCESS,
            data={"result": "test"},
            timestamp=datetime.now()
        )
        
        reflection_input = ReflectionInput(
            task_description="Test task",
            flow_name="test_flow",
            flow_status="SUCCESS",
            flow_result=flow_result,
            state_summary="Test state",
            execution_history_text="Test history",
            planning_rationale="Test rationale",
            cycle=1,
            progress=50
        )
        
        assert reflection_input.task_description == "Test task"
        assert reflection_input.flow_name == "test_flow"
        assert reflection_input.flow_status == "SUCCESS"
        assert reflection_input.flow_result == flow_result
        assert reflection_input.cycle == 1
        assert reflection_input.progress == 50
    
    def test_step_reflection_input_creation(self):
        """Test StepReflectionInput creation."""
        class TestInputs(BaseModel):
            param1: str = "value1"
            param2: int = 42
        
        flow_result = FlowResult(
            status=FlowStatus.SUCCESS,
            data={"result": "test"},
            timestamp=datetime.now()
        )
        
        step_input = StepReflectionInput(
            task_description="Test task",
            step_id="step_123",
            step_intent="Test intent",
            step_rationale="Test rationale",
            flow_name="test_flow",
            flow_inputs=TestInputs(),
            flow_result=flow_result,
            current_progress=25
        )
        
        assert step_input.task_description == "Test task"
        assert step_input.step_id == "step_123"
        assert step_input.step_intent == "Test intent"
        assert step_input.step_rationale == "Test rationale"
        assert step_input.flow_name == "test_flow"
        assert isinstance(step_input.flow_inputs, TestInputs)
        assert step_input.current_progress == 25


class TestAgentReflectionInitialization:
    """Test AgentReflection initialization and lifecycle."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider."""
        provider = Mock(spec=LLMProvider)
        provider.generate_structured = AsyncMock()
        return provider
    
    @pytest.fixture
    def reflection_config(self):
        """Create reflection configuration."""
        return ReflectionConfig(
            model_name="test-model",
            provider_name="test_provider"
        )
    
    @pytest.fixture
    def mock_activity_stream(self):
        """Create mock activity stream."""
        stream = Mock()
        stream.reflection = Mock()
        stream.error = Mock()
        return stream
    
    def test_initialization_with_config(self, reflection_config, mock_llm_provider):
        """Test reflection initialization with config."""
        reflection = AgentReflection(
            config=reflection_config,
            llm_provider=mock_llm_provider,
            name="test_reflection"
        )
        
        assert reflection._name == "test_reflection"
        assert reflection._config == reflection_config
        assert reflection._llm_provider == mock_llm_provider
        assert not reflection._initialized
    
    def test_initialization_with_defaults(self):
        """Test reflection initialization with defaults."""
        reflection = AgentReflection()
        
        assert reflection._name == "reflection"
        assert isinstance(reflection._config, ReflectionConfig)
        assert reflection._config.model_name == "agent-model-large"
        assert reflection._llm_provider is None
    
    def test_initialization_with_activity_stream(self, mock_activity_stream):
        """Test reflection initialization with activity stream."""
        reflection = AgentReflection(activity_stream=mock_activity_stream)
        
        assert reflection._activity_stream == mock_activity_stream
    
    def test_implements_reflection_interface(self):
        """Test that AgentReflection implements ReflectionInterface."""
        reflection = AgentReflection()
        assert isinstance(reflection, ReflectionInterface)
    
    @pytest.mark.asyncio
    async def test_initialization_success(self, reflection_config, mock_llm_provider):
        """Test successful initialization."""
        with patch.object(AgentReflection, '_load_reflection_template', new_callable=AsyncMock) as mock_load_template, \
             patch.object(AgentReflection, '_load_step_reflection_template', new_callable=AsyncMock) as mock_load_step_template:
            
            mock_load_template.return_value = Mock()
            mock_load_step_template.return_value = Mock()
            
            reflection = AgentReflection(
                config=reflection_config,
                llm_provider=mock_llm_provider
            )
            
            await reflection.initialize()
            
            assert reflection._initialized
            mock_load_template.assert_called_once()
            mock_load_step_template.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialization_without_provider(self, reflection_config):
        """Test initialization without LLM provider."""
        with patch('flowlib.agent.components.reflection.base.provider_registry') as mock_registry:
            mock_provider = Mock(spec=LLMProvider)
            mock_registry.get_by_config = AsyncMock(return_value=mock_provider)
            
            with patch.object(AgentReflection, '_load_reflection_template', new_callable=AsyncMock) as mock_load_template, \
                 patch.object(AgentReflection, '_load_step_reflection_template', new_callable=AsyncMock) as mock_load_step_template:
                
                mock_load_template.return_value = Mock()
                mock_load_step_template.return_value = Mock()
                
                reflection = AgentReflection(config=reflection_config)
                await reflection.initialize()
                
                assert reflection._llm_provider == mock_provider
                assert reflection._initialized
    
    @pytest.mark.asyncio
    async def test_initialization_provider_failure(self, reflection_config):
        """Test initialization failure when provider creation fails."""
        with patch('flowlib.agent.components.reflection.base.provider_registry') as mock_registry:
            mock_registry.get_by_config = AsyncMock(side_effect=Exception("Provider creation failed"))
            
            reflection = AgentReflection(config=reflection_config)
            
            with pytest.raises(ReflectionError, match="Failed to create model provider"):
                await reflection.initialize()
    
    @pytest.mark.asyncio
    async def test_initialization_template_failure(self, reflection_config, mock_llm_provider):
        """Test initialization failure when template loading fails."""
        with patch.object(AgentReflection, '_load_reflection_template', new_callable=AsyncMock) as mock_load_template:
            mock_load_template.side_effect = Exception("Template loading failed")
            
            reflection = AgentReflection(
                config=reflection_config,
                llm_provider=mock_llm_provider
            )
            
            with pytest.raises(ReflectionError, match="Failed to initialize reflection component"):
                await reflection.initialize()
    
    @pytest.mark.asyncio
    async def test_shutdown(self, reflection_config, mock_llm_provider):
        """Test reflection shutdown."""
        reflection = AgentReflection(
            config=reflection_config,
            llm_provider=mock_llm_provider
        )
        
        with patch.object(AgentReflection, '_load_reflection_template', new_callable=AsyncMock), \
             patch.object(AgentReflection, '_load_step_reflection_template', new_callable=AsyncMock):
            
            await reflection.initialize()
            assert reflection._initialized
            
            await reflection.shutdown()
            assert not reflection._initialized


class TestAgentReflectionTemplateLoading:
    """Test template loading functionality."""
    
    @pytest.fixture
    def reflection(self):
        """Create reflection instance."""
        return AgentReflection()
    
    @pytest.mark.asyncio
    async def test_load_reflection_template_from_registry(self, reflection):
        """Test loading reflection template from registry."""
        mock_template = Mock()
        
        with patch('flowlib.agent.components.reflection.base.resource_registry') as mock_registry:
            mock_registry.contains.return_value = True
            mock_registry.get.return_value = mock_template
            
            template = await reflection._load_reflection_template()
            
            assert template == mock_template
            mock_registry.contains.assert_called_once()
            mock_registry.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_reflection_template_fallback(self, reflection):
        """Test fallback to default reflection template."""
        with patch('flowlib.agent.components.reflection.base.resource_registry') as mock_registry:
            mock_registry.contains.return_value = False
            
            template = await reflection._load_reflection_template()
            
            assert isinstance(template, DefaultReflectionPrompt)
    
    @pytest.mark.asyncio
    async def test_load_reflection_template_registry_error(self, reflection):
        """Test handling of registry errors during template loading."""
        with patch('flowlib.agent.components.reflection.base.resource_registry') as mock_registry:
            mock_registry.contains.side_effect = Exception("Registry error")
            
            template = await reflection._load_reflection_template()
            
            assert isinstance(template, DefaultReflectionPrompt)
    
    @pytest.mark.asyncio
    async def test_load_step_reflection_template_from_registry(self, reflection):
        """Test loading step reflection template from registry."""
        mock_template = Mock()
        
        with patch('flowlib.agent.components.reflection.base.resource_registry') as mock_registry:
            mock_registry.contains.return_value = True
            mock_registry.get.return_value = mock_template
            
            template = await reflection._load_step_reflection_template()
            
            assert template == mock_template
    
    @pytest.mark.asyncio
    async def test_load_step_reflection_template_fallback(self, reflection):
        """Test fallback to default step reflection template."""
        with patch('flowlib.agent.components.reflection.base.resource_registry') as mock_registry:
            mock_registry.contains.return_value = False
            
            template = await reflection._load_step_reflection_template()
            
            assert isinstance(template, DefaultStepReflectionPrompt)


class TestAgentReflectionMainReflection:
    """Test main reflection functionality."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider."""
        provider = Mock(spec=LLMProvider)
        provider.generate_structured = AsyncMock()
        return provider
    
    @pytest.fixture
    def mock_template(self):
        """Create mock reflection template."""
        return Mock()
    
    @pytest.fixture
    def agent_state(self):
        """Create mock agent state."""
        state = Mock(spec=AgentState)
        state.task_description = "Test task description"
        state.progress = 50
        state.cycles = 3
        state.execution_history = []
        return state
    
    @pytest.fixture
    def flow_result(self):
        """Create mock flow result."""
        return FlowResult(
            status=FlowStatus.SUCCESS,
            data={"result": "test result"},
            timestamp=datetime.now(),
            duration=1.5
        )
    
    @pytest.fixture
    def flow_inputs(self):
        """Create mock flow inputs."""
        class TestInputs(BaseModel):
            param1: str = "value1"
            param2: int = 42
        
        return TestInputs()
    
    @pytest_asyncio.fixture
    async def initialized_reflection(self, mock_llm_provider, mock_template):
        """Create initialized reflection instance."""
        reflection = AgentReflection(llm_provider=mock_llm_provider)
        reflection._reflection_template = mock_template
        reflection._step_reflection_template = mock_template
        reflection._initialized = True
        return reflection
    
    @pytest.mark.asyncio
    async def test_reflect_success(self, initialized_reflection, agent_state, flow_result, flow_inputs, mock_llm_provider):
        """Test successful reflection."""
        expected_result = ReflectionResult(
            reflection="Test reflection analysis",
            progress=75,
            is_complete=False,
            insights=["Key insight"]
        )
        
        mock_llm_provider.generate_structured.return_value = expected_result
        
        result = await initialized_reflection.reflect(
            state=agent_state,
            flow_name="test_flow",
            flow_inputs=flow_inputs,
            flow_result=flow_result
        )
        
        assert result == expected_result
        assert result.progress == 75  # Progress validation
        mock_llm_provider.generate_structured.assert_called_once()
        
        # Verify template variables
        call_args = mock_llm_provider.generate_structured.call_args
        template_vars = call_args[1]["prompt_variables"]
        assert template_vars["task_description"] == "Test task description"
        assert template_vars["current_progress"] == 50
        assert "plan_status" in template_vars
    
    @pytest.mark.asyncio
    async def test_reflect_with_activity_stream(self, mock_llm_provider, mock_template, agent_state, flow_result, flow_inputs):
        """Test reflection with activity stream."""
        mock_activity_stream = Mock()
        mock_activity_stream.reflection = Mock()
        
        reflection = AgentReflection(
            llm_provider=mock_llm_provider,
            activity_stream=mock_activity_stream
        )
        reflection._reflection_template = mock_template
        reflection._step_reflection_template = mock_template
        reflection._initialized = True
        
        expected_result = ReflectionResult(
            reflection="Test reflection",
            progress=80,
            is_complete=True,
            completion_reason="Task completed"
        )
        
        mock_llm_provider.generate_structured.return_value = expected_result
        
        result = await reflection.reflect(
            state=agent_state,
            flow_name="test_flow",
            flow_inputs=flow_inputs,
            flow_result=flow_result
        )
        
        assert result == expected_result
        assert mock_activity_stream.reflection.call_count == 2  # Start and end
    
    @pytest.mark.asyncio
    async def test_reflect_progress_validation(self, initialized_reflection, agent_state, flow_result, flow_inputs, mock_llm_provider):
        """Test progress validation in reflection."""
        # Test progress clamping
        invalid_result = ReflectionResult(
            reflection="Test reflection",
            progress=150,  # Invalid - over 100
            is_complete=False
        )
        
        mock_llm_provider.generate_structured.return_value = invalid_result
        
        result = await initialized_reflection.reflect(
            state=agent_state,
            flow_name="test_flow",
            flow_inputs=flow_inputs,
            flow_result=flow_result
        )
        
        assert result.progress == 100  # Should be clamped
    
    @pytest.mark.asyncio
    async def test_reflect_with_flow_error(self, initialized_reflection, agent_state, flow_inputs, mock_llm_provider):
        """Test reflection with flow error."""
        flow_result = FlowResult(
            status=FlowStatus.ERROR,
            data={},
            timestamp=datetime.now(),
            error="Test error message"
        )
        
        expected_result = ReflectionResult(
            reflection="Error analysis",
            progress=30,
            is_complete=False
        )
        
        mock_llm_provider.generate_structured.return_value = expected_result
        
        result = await initialized_reflection.reflect(
            state=agent_state,
            flow_name="test_flow",
            flow_inputs=flow_inputs,
            flow_result=flow_result
        )
        
        assert result == expected_result
        
        # Verify error information is passed to template
        call_args = mock_llm_provider.generate_structured.call_args
        template_vars = call_args[1]["prompt_variables"]
        assert "FAILED" in template_vars["plan_status"]
        assert "Test error message" in template_vars["plan_error"]
    
    @pytest.mark.asyncio
    async def test_reflect_not_initialized(self, mock_llm_provider):
        """Test reflection failure when not initialized."""
        reflection = AgentReflection(llm_provider=mock_llm_provider)
        
        with pytest.raises(NotInitializedError):
            await reflection.reflect(
                state=Mock(),
                flow_name="test_flow",
                flow_inputs=Mock(),
                flow_result=Mock()
            )
    
    @pytest.mark.asyncio
    async def test_reflect_no_template(self, mock_llm_provider):
        """Test reflection failure when template not loaded."""
        reflection = AgentReflection(llm_provider=mock_llm_provider)
        reflection._initialized = True
        reflection._reflection_template = None
        
        with pytest.raises(NotInitializedError, match="Reflection template not loaded"):
            await reflection.reflect(
                state=Mock(),
                flow_name="test_flow",
                flow_inputs=Mock(),
                flow_result=Mock()
            )
    
    @pytest.mark.asyncio
    async def test_reflect_llm_error(self, initialized_reflection, agent_state, flow_result, flow_inputs, mock_llm_provider):
        """Test reflection error handling when LLM fails."""
        mock_llm_provider.generate_structured.side_effect = Exception("LLM error")
        
        with pytest.raises(ReflectionError, match="Reflection failed"):
            await initialized_reflection.reflect(
                state=agent_state,
                flow_name="test_flow",
                flow_inputs=flow_inputs,
                flow_result=flow_result
            )


class TestAgentReflectionStepReflection:
    """Test step reflection functionality."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider."""
        provider = Mock(spec=LLMProvider)
        provider.generate_structured = AsyncMock()
        return provider
    
    @pytest_asyncio.fixture
    async def initialized_reflection(self, mock_llm_provider):
        """Create initialized reflection instance."""
        reflection = AgentReflection(llm_provider=mock_llm_provider)
        reflection._reflection_template = Mock()
        reflection._step_reflection_template = Mock()
        reflection._initialized = True
        return reflection
    
    @pytest.fixture
    def step_input(self):
        """Create step reflection input."""
        class TestInputs(BaseModel):
            param1: str = "value1"
            param2: int = 42
        
        flow_result = FlowResult(
            status=FlowStatus.SUCCESS,
            data={"result": "step result"},
            timestamp=datetime.now()
        )
        
        return StepReflectionInput(
            task_description="Test task",
            step_id="step_123",
            step_intent="Test step intent",
            step_rationale="Test step rationale",
            flow_name="test_flow",
            flow_inputs=TestInputs(),
            flow_result=flow_result,
            current_progress=25
        )
    
    @pytest.mark.asyncio
    async def test_step_reflect_success(self, initialized_reflection, step_input, mock_llm_provider):
        """Test successful step reflection."""
        expected_result = StepReflectionResult(
            step_id="step_123",
            reflection="Step completed successfully",
            step_success=True,
            key_observation="Important data obtained"
        )
        
        mock_llm_provider.generate_structured.return_value = expected_result
        
        result = await initialized_reflection.step_reflect(step_input)
        
        assert result == expected_result
        mock_llm_provider.generate_structured.assert_called_once()
        
        # Verify template variables
        call_args = mock_llm_provider.generate_structured.call_args
        template_vars = call_args[1]["prompt_variables"]
        assert template_vars["task_description"] == "Test task"
        assert template_vars["step_id"] == "step_123"
        assert template_vars["step_intent"] == "Test step intent"
        assert template_vars["flow_name"] == "test_flow"
    
    @pytest.mark.asyncio
    async def test_step_reflect_missing_step_id(self, initialized_reflection, step_input, mock_llm_provider):
        """Test step reflection when LLM doesn't return step_id."""
        result_without_id = StepReflectionResult(
            step_id="",  # Missing step_id
            reflection="Step analysis",
            step_success=True
        )
        
        mock_llm_provider.generate_structured.return_value = result_without_id
        
        result = await initialized_reflection.step_reflect(step_input)
        
        assert result.step_id == "step_123"  # Should be corrected
        assert result.reflection == "Step analysis"
    
    @pytest.mark.asyncio
    async def test_step_reflect_not_initialized(self, mock_llm_provider, step_input):
        """Test step reflection failure when not initialized."""
        reflection = AgentReflection(llm_provider=mock_llm_provider)
        
        with pytest.raises(NotInitializedError):
            await reflection.step_reflect(step_input)
    
    @pytest.mark.asyncio
    async def test_step_reflect_no_template(self, mock_llm_provider, step_input):
        """Test step reflection failure when template not loaded."""
        reflection = AgentReflection(llm_provider=mock_llm_provider)
        reflection._initialized = True
        reflection._step_reflection_template = None
        
        with pytest.raises(NotInitializedError, match="Step reflection template not loaded"):
            await reflection.step_reflect(step_input)
    
    @pytest.mark.asyncio
    async def test_step_reflect_error_handling(self, initialized_reflection, step_input, mock_llm_provider):
        """Test step reflection error handling."""
        mock_llm_provider.generate_structured.side_effect = Exception("Step reflection failed")
        
        result = await initialized_reflection.step_reflect(step_input)
        
        # Should return a default failure result
        assert result.step_id == "step_123"
        assert "Step reflection failed" in result.reflection
        assert result.step_success is False
        assert result.key_observation == "Reflection process encountered an error."


class TestAgentReflectionFormatting:
    """Test formatting helper methods."""
    
    @pytest.fixture
    def reflection(self):
        """Create reflection instance."""
        return AgentReflection()
    
    def test_format_execution_history_empty(self, reflection):
        """Test formatting empty execution history."""
        result = reflection._format_execution_history([])
        
        assert "No execution history available." in result
    
    def test_format_execution_history_with_entries(self, reflection):
        """Test formatting execution history with entries."""
        mock_entry1 = Mock()
        mock_entry1.flow_name = "flow1"
        mock_entry1.result = "Result 1"
        
        mock_entry2 = Mock()
        mock_entry2.flow_name = "flow2"
        mock_entry2.result = "Result 2"
        
        history = [mock_entry1, mock_entry2]
        result = reflection._format_execution_history(history)
        
        assert "Recent Execution History:" in result
        assert "Flow: flow1" in result
        assert "Flow: flow2" in result
        assert "Result: Result 1" in result
        assert "Result: Result 2" in result
    
    def test_format_execution_history_long_result(self, reflection):
        """Test formatting execution history with long result."""
        mock_entry = Mock()
        mock_entry.flow_name = "test_flow"
        mock_entry.result = "A" * 150  # Long result
        
        result = reflection._format_execution_history([mock_entry])
        
        assert "Flow: test_flow" in result
        assert "..." in result  # Should be truncated
    
    def test_format_execution_history_without_flow_name(self, reflection):
        """Test formatting execution history without flow_name attribute."""
        mock_entry = "Simple string entry"
        
        result = reflection._format_execution_history([mock_entry])
        
        assert "Recent Execution History:" in result
        assert "Simple string entry" in result
    
    def test_format_step_reflections_empty(self, reflection):
        """Test formatting empty step reflections."""
        result = reflection._format_step_reflections([])
        
        assert "No step reflections were recorded" in result
    
    def test_format_step_reflections_with_data(self, reflection):
        """Test formatting step reflections with data."""
        step_reflection1 = StepReflectionResult(
            step_id="step_1",
            reflection="First step reflection",
            step_success=True,
            key_observation="Key observation 1"
        )
        
        step_reflection2 = StepReflectionResult(
            step_id="step_2", 
            reflection="Second step reflection",
            step_success=False
        )
        
        result = reflection._format_step_reflections([step_reflection1, step_reflection2])
        
        assert "Summary of Plan Step Reflections:" in result
        assert "Step 1 (ID: step_1):" in result
        assert "Success: True" in result
        assert "First step reflection" in result
        assert "Key Observation: Key observation 1" in result
        assert "Step 2 (ID: step_2):" in result
        assert "Success: False" in result
    
    def test_format_flow_result_basic(self, reflection):
        """Test formatting basic flow result."""
        flow_result = FlowResult(
            status=FlowStatus.SUCCESS,
            data={"key": "value"},
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            duration=2.5
        )
        
        result = reflection._format_flow_result(flow_result)
        
        assert "Status: SUCCESS" in result
        assert "Duration: 2.500 seconds" in result
        assert "Data:" in result
        assert "key: value" in result
    
    def test_format_flow_result_with_error(self, reflection):
        """Test formatting flow result with error."""
        flow_result = FlowResult(
            status=FlowStatus.ERROR,
            data={},
            timestamp=datetime.now(),
            error="Test error message"
        )
        
        result = reflection._format_flow_result(flow_result)
        
        assert "Status: ERROR" in result
        assert "Error: Test error message" in result
    
    def test_format_flow_result_with_pydantic_data(self, reflection):
        """Test formatting flow result with Pydantic model data."""
        class TestData(BaseModel):
            field1: str = "value1"
            field2: int = 42
        
        flow_result = FlowResult(
            status=FlowStatus.SUCCESS,
            data=TestData(),
            timestamp=datetime.now()
        )
        
        result = reflection._format_flow_result(flow_result)
        
        assert "field1: value1" in result
        assert "field2: 42" in result
    
    def test_format_flow_inputs_basic(self, reflection):
        """Test formatting basic flow inputs."""
        class TestInputs(BaseModel):
            param1: str = "value1"
            param2: int = 42
            param3: List[str] = ["item1", "item2"]
        
        inputs = TestInputs()
        result = reflection._format_flow_inputs(inputs)
        
        assert "Model type: TestInputs" in result
        assert "param1: value1" in result
        assert "param2: 42" in result
        assert "param3: [item1, item2]" in result
    
    def test_format_flow_inputs_with_dict(self, reflection):
        """Test formatting flow inputs with dictionary field."""
        class TestInputs(BaseModel):
            config: Dict[str, Any] = {"key1": "value1", "key2": "value2"}
        
        inputs = TestInputs()
        result = reflection._format_flow_inputs(inputs)
        
        assert "config: {key1: value1, key2: value2}" in result
    
    def test_format_flow_inputs_with_long_list(self, reflection):
        """Test formatting flow inputs with long list."""
        class TestInputs(BaseModel):
            items: List[str] = [f"item{i}" for i in range(10)]
        
        inputs = TestInputs()
        result = reflection._format_flow_inputs(inputs)
        
        assert "items: [List with 10 items]" in result
    
    def test_format_flow_inputs_skip_private_fields(self, reflection):
        """Test formatting flow inputs skips private fields."""
        class TestInputs(BaseModel):
            public_field: str = "public"
            _private_field: str = "private"
        
        inputs = TestInputs()
        result = reflection._format_flow_inputs(inputs)
        
        assert "public_field: public" in result
        assert "_private_field" not in result


class TestAgentReflectionPrompts:
    """Test reflection prompt classes."""
    
    def test_default_reflection_prompt(self):
        """Test DefaultReflectionPrompt class."""
        prompt = DefaultReflectionPrompt(name="test_prompt", type="prompt_config")
        
        assert hasattr(prompt, 'template')
        assert hasattr(prompt, 'config')
        assert isinstance(prompt.template, str)
        assert isinstance(prompt.config, dict)
        assert "task_description" in prompt.template
        assert "plan_status" in prompt.template
        assert "execution_history_text" in prompt.template
    
    def test_task_completion_reflection_prompt(self):
        """Test TaskCompletionReflectionPrompt class."""
        prompt = TaskCompletionReflectionPrompt(name="test_completion_prompt", type="prompt_config")
        
        assert hasattr(prompt, 'template')
        assert hasattr(prompt, 'config')
        assert isinstance(prompt.template, str)
        assert isinstance(prompt.config, dict)
        assert "task_description" in prompt.template
        assert "flow_result" in prompt.template
    
    def test_default_step_reflection_prompt(self):
        """Test DefaultStepReflectionPrompt class."""
        prompt = DefaultStepReflectionPrompt(name="test_step_prompt", type="prompt_config")
        
        assert hasattr(prompt, 'template')
        assert hasattr(prompt, 'output_model')
        assert prompt.output_model == StepReflectionResult
        assert "step_id" in prompt.template
        assert "step_intent" in prompt.template
        assert "flow_result_formatted" in prompt.template
    
    def test_step_reflection_prompt_format(self):
        """Test step reflection prompt template structure."""
        prompt = DefaultStepReflectionPrompt(name="test_step_prompt", type="prompt_config")
        
        # Test that the template contains expected placeholder variables
        assert hasattr(prompt, 'template')
        assert isinstance(prompt.template, str)
        assert "{{task_description}}" in prompt.template
        assert "{{step_id}}" in prompt.template
        assert "{{step_intent}}" in prompt.template
        assert "{{flow_name}}" in prompt.template
    
    def test_step_reflection_prompt_format_missing_key(self):
        """Test step reflection prompt formatting with missing key."""
        prompt = DefaultStepReflectionPrompt(name="test_step_prompt", type="prompt_config")
        
        kwargs = {
            "task_description": "Test task",
            # Missing step_id and other required keys
        }
        
        with pytest.raises(ValueError, match="Missing required key"):
            prompt.format(**kwargs)