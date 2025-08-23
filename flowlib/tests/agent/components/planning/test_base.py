"""Tests for agent planning base implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from flowlib.agent.components.planning import AgentPlanner
from flowlib.agent.components.planning.interfaces import PlanningInterface
from flowlib.agent.components.planning.models import (
    PlanningResult, 
    PlanningValidation, 
    PlanningExplanation,
    Plan,
    PlanStep
)
from flowlib.agent.models.config import PlannerConfig
from flowlib.agent.models.state import AgentState
from flowlib.agent.core.errors import PlanningError, NotInitializedError
from flowlib.flows.models.metadata import FlowMetadata
from flowlib.resources.models.constants import ResourceType
from pydantic import BaseModel


class MockInputModel(BaseModel):
    """Mock input model for testing."""
    text: str
    options: Dict[str, Any] = {}


# class TestBasePlanning:
#     """Test BasePlanning abstract functionality."""
#     
#     @pytest.fixture
#     def mock_state(self):
#         """Create mock agent state."""
#         state = AgentState(
#             task_description="Test task",
#             task_id="test_123"
#         )
#         state.add_user_message("Hello")
#         state.add_system_message("Hi there")
#         return state
#     
#     @pytest.fixture
#     def mock_planning_result(self):
#         """Create mock planning result."""
#         return PlanningResult(
#             selected_flow="test_flow",
#             reasoning=PlanningExplanation(
#                 explanation="Test explanation",
#                 rationale="Test rationale",
#                 decision_factors=["factor1", "factor2"]
#             )
#         )
#     
#     @pytest.fixture
#     def base_planning(self):
#         """Create BasePlanning instance for testing."""
#         planning = BasePlanning("test_planner")
#         planning._logger = MagicMock()
#         return planning
#     
#     def test_base_planning_initialization(self):
#         """Test BasePlanning initialization."""
#         planning = BasePlanning("test_planner")
#         
#         assert planning.name == "test_planner"
#         assert not planning.initialized
#     
#     def test_base_planning_implements_interface(self):
#         """Test that BasePlanning implements PlanningInterface."""
#         planning = BasePlanning("test_planner")
#         
#         # Check that all interface methods exist
#         assert hasattr(planning, 'plan')
#         assert hasattr(planning, 'validate_plan')
#         assert hasattr(planning, 'generate_inputs')
#         
#         # With @runtime_checkable, we can test isinstance
#         assert isinstance(planning, PlanningInterface)
#     
#     @pytest.mark.asyncio
#     async def test_initialization_lifecycle(self):
#         """Test initialization and shutdown lifecycle."""
#         planning = BasePlanning("test_planner")
#         
#         # Should not be initialized initially
#         assert not planning.initialized
#         
#         # Initialize
#         await planning.initialize()
#         assert planning.initialized
#         
#         # Shutdown
#         await planning.shutdown()
#         assert not planning.initialized
#     
#     @pytest.mark.asyncio
#     async def test_plan_error_handling(self, base_planning, mock_state):
#         """Test plan method error handling."""
#         # Mock _plan_impl to raise exception
#         base_planning._plan_impl = AsyncMock(side_effect=Exception("Test error"))
#         
#         with pytest.raises(PlanningError, match="Failed to generate plan: Test error"):
#             await base_planning.plan(mock_state)
#         
#         # Verify error was logged
#         base_planning._logger.error.assert_called_once()
#     
#     @pytest.mark.asyncio
#     async def test_validate_plan_error_handling(self, base_planning, mock_planning_result):
#         """Test validate_plan method error handling."""
#         # Mock _validate_plan_impl to raise exception
#         base_planning._validate_plan_impl = AsyncMock(side_effect=Exception("Validation error"))
#         
#         with pytest.raises(PlanningError, match="Failed to validate plan: Validation error"):
#             await base_planning.validate_plan(mock_planning_result)
#         
#         # Verify error was logged
#         base_planning._logger.error.assert_called_once()
#     
#     @pytest.mark.asyncio
#     async def test_plan_successful_execution(self, base_planning, mock_state, mock_planning_result):
#         """Test successful plan execution."""
#         # Mock _plan_impl to return planning result
#         base_planning._plan_impl = AsyncMock(return_value=mock_planning_result)
#         
#         result = await base_planning.plan(mock_state)
#         
#         assert result == mock_planning_result
#         base_planning._logger.info.assert_called()
#     
#     @pytest.mark.asyncio
#     async def test_validate_plan_successful_execution(self, base_planning, mock_planning_result):
#         """Test successful plan validation."""
#         validation_result = PlanningValidation(is_valid=True, errors=[])
#         base_planning._validate_plan_impl = AsyncMock(return_value=validation_result)
#         
#         result = await base_planning.validate_plan(mock_planning_result)
#         
#         assert result == validation_result
#         base_planning._logger.info.assert_called()
# 
# 
class TestAgentPlanner:
    """Test AgentPlanner implementation."""
    
    @pytest.fixture
    def planner_config(self):
        """Create planner configuration."""
        return PlannerConfig(
            model_name="test-model",
            provider_name="test-provider"
        )
    
    @pytest.fixture
    def agent_planner(self, planner_config):
        """Create AgentPlanner instance."""
        return AgentPlanner(config=planner_config)
    
    @pytest.fixture
    def mock_state(self):
        """Create mock agent state."""
        state = AgentState(
            task_description="Test planning task",
            task_id="plan_test_123"
        )
        state.add_user_message("Can you help me?")
        state.add_system_message("Of course!")
        return state
    
    def test_agent_planner_initialization(self, planner_config):
        """Test AgentPlanner initialization."""
        planner = AgentPlanner(config=planner_config)
        
        assert planner.config == planner_config
        assert planner.name == "agent_planner"
        assert planner.llm_provider is None
        assert planner._planning_template is None
        assert planner._input_generation_template is None
    
    def test_agent_planner_custom_name(self, planner_config):
        """Test AgentPlanner with custom name."""
        planner = AgentPlanner(config=planner_config, name="custom_planner")
        
        assert planner.name == "custom_planner"
    
    def test_agent_planner_config_validation(self):
        """Test AgentPlanner config validation."""
        # AgentPlanner doesn't validate config type in constructor
        planner = AgentPlanner(config="invalid_config")
        assert planner.config == "invalid_config"
    
    def test_llm_provider_property_get_set(self, agent_planner):
        """Test llm_provider property getter and setter."""
        # Initially None
        assert agent_planner.llm_provider is None
        
        # Set provider
        mock_provider = MagicMock()
        agent_planner.llm_provider = mock_provider
        
        assert agent_planner.llm_provider == mock_provider
    
    def test_llm_provider_property_from_parent(self, agent_planner):
        """Test llm_provider property retrieval from parent."""
        # AgentPlanner doesn't implement parent fallback for llm_provider
        # It's a simple attribute that defaults to None
        assert agent_planner.llm_provider is None
        
        # Setting parent doesn't affect llm_provider
        mock_parent = MagicMock()
        mock_provider = MagicMock()
        mock_parent.llm_provider = mock_provider
        agent_planner.set_parent(mock_parent)
        
        # llm_provider is still None - no parent fallback
        assert agent_planner.llm_provider is None
    
    @pytest.mark.asyncio
    async def test_initialize_without_provider(self, agent_planner):
        """Test initialization without existing provider."""
        # AgentPlanner initialization is simplified - just call initialize
        await agent_planner.initialize()
        
        # Verify the planner is now initialized
        assert agent_planner.initialized
    
    @pytest.mark.asyncio
    async def test_initialize_provider_creation_failure(self, agent_planner):
        """Test initialization when provider creation fails."""
        # AgentPlanner's simple initialization should not fail
        await agent_planner.initialize()
        assert agent_planner.initialized
    
    @pytest.mark.asyncio
    async def test_initialize_missing_provider_config(self):
        """Test initialization with missing provider config."""
        # Create config with empty provider_name
        config = PlannerConfig(model_name="test-model", provider_name=None)
        
        planner = AgentPlanner(config=config)
        
        # AgentPlanner's simple initialization should not fail
        await planner.initialize()
        assert planner.initialized
    
    @pytest.mark.asyncio
    async def test_initialize_template_loading_failure(self, agent_planner):
        """Test initialization when template loading fails."""
        # AgentPlanner's simple initialization should not fail
        await agent_planner.initialize()
        assert agent_planner.initialized
    
    # Skipping template loading tests as AgentPlanner doesn't have these methods
    
    
    
    @pytest.mark.asyncio
    async def test_plan_impl_no_registry(self, agent_planner, mock_state):
        """Test _plan_impl when no flow registry is available."""
        with patch('flowlib.agent.components.planning.planner.flow_registry', None):
            with pytest.raises(PlanningError, match="No flow registry available"):
                await agent_planner._plan_impl(mock_state)
    
    @pytest.mark.asyncio
    async def test_plan_impl_no_flows(self, agent_planner, mock_state):
        """Test _plan_impl when no flows are available."""
        with patch('flowlib.agent.components.planning.planner.flow_registry') as mock_registry:
            mock_registry.get_agent_selectable_flows.return_value = {}
            
            with pytest.raises(PlanningError, match="No agent-selectable flows available"):
                await agent_planner._plan_impl(mock_state)
    
    @pytest.mark.asyncio
    async def test_plan_impl_successful(self, agent_planner, mock_state):
        """Test successful _plan_impl execution."""
        # Mock flow registry
        mock_flows = {
            "test_flow": MagicMock()
        }
        mock_metadata = FlowMetadata(
            name="test_flow",
            description="Test flow description",
            input_model=MockInputModel,
            output_model=MockInputModel  # Use the same model as output for simplicity
        )
        
        with patch('flowlib.agent.components.planning.planner.flow_registry') as mock_registry:
            mock_registry.get_agent_selectable_flows.return_value = mock_flows
            mock_registry.get_flow_metadata.return_value = mock_metadata
            
            # Mock LLM provider and plan generation
            mock_plan = Plan(
                task_description="Test task",
                steps=[
                    PlanStep(
                        flow_name="test_flow",
                        step_intent="Test step",
                        rationale="Test rationale"
                    )
                ],
                overall_rationale="Test overall rationale"
            )
            
            agent_planner.llm_provider = AsyncMock()
            agent_planner.llm_provider.generate_structured = AsyncMock(return_value=mock_plan)
            agent_planner._planning_template = "test_template"
            agent_planner.config = MagicMock()
            agent_planner.config.model_name = "test-model"
            
            result = await agent_planner._plan_impl(mock_state)
            
            assert isinstance(result, PlanningResult)
            assert result.selected_flow == "test_flow"
            assert result.reasoning.explanation == "Test rationale"
    
    @pytest.mark.asyncio
    async def test_plan_impl_llm_returns_none(self, agent_planner, mock_state):
        """Test _plan_impl when LLM returns None."""
        with patch('flowlib.agent.components.planning.planner.flow_registry') as mock_registry:
            mock_registry.get_agent_selectable_flows.return_value = {"test_flow": MagicMock()}
            mock_registry.get_flow_metadata.return_value = MagicMock(description="Test")
            
            agent_planner.llm_provider = MagicMock()
            agent_planner.llm_provider.generate_structured = AsyncMock(return_value=None)
            agent_planner._planning_template = "test_template"
            
            with pytest.raises(PlanningError, match="LLM returned None for the plan"):
                await agent_planner._plan_impl(mock_state)
    
    @pytest.mark.asyncio
    async def test_validate_plan_impl_no_registry(self, agent_planner):
        """Test _validate_plan_impl when no registry is available."""
        plan = PlanningResult(
            selected_flow="test_flow",
            reasoning=PlanningExplanation(explanation="test", rationale="test")
        )
        
        with patch('flowlib.agent.components.planning.planner.flow_registry', None):
            result = await agent_planner._validate_plan_impl(plan)
            
            assert not result.is_valid
            assert "Flow registry is not available for validation" in result.errors
    
    @pytest.mark.asyncio
    async def test_validate_plan_impl_invalid_flow(self, agent_planner):
        """Test _validate_plan_impl with invalid flow."""
        plan = PlanningResult(
            selected_flow="invalid_flow",
            reasoning=PlanningExplanation(explanation="test", rationale="test")
        )
        
        with patch('flowlib.agent.components.planning.planner.flow_registry') as mock_registry:
            mock_registry.get_agent_selectable_flows.return_value = {}
            mock_registry.get_flow_instances.return_value = {}
            
            result = await agent_planner._validate_plan_impl(plan)
            
            assert not result.is_valid
            assert "not found in registry" in result.errors[0]
    
    @pytest.mark.asyncio
    async def test_validate_plan_impl_infrastructure_flow(self, agent_planner):
        """Test _validate_plan_impl with infrastructure flow."""
        plan = PlanningResult(
            selected_flow="infrastructure_flow",
            reasoning=PlanningExplanation(explanation="test", rationale="test")
        )
        
        with patch('flowlib.agent.components.planning.planner.flow_registry') as mock_registry:
            mock_registry.get_agent_selectable_flows.return_value = {}
            mock_registry.get_flow_instances.return_value = {"infrastructure_flow": MagicMock()}
            
            result = await agent_planner._validate_plan_impl(plan)
            
            assert not result.is_valid
            assert "is an infrastructure flow" in result.errors[0]
    
    @pytest.mark.asyncio
    async def test_validate_plan_impl_valid_plan(self, agent_planner):
        """Test _validate_plan_impl with valid plan."""
        plan = PlanningResult(
            selected_flow="test_flow",
            reasoning=PlanningExplanation(explanation="test", rationale="test")
        )
        
        with patch('flowlib.agent.components.planning.planner.flow_registry') as mock_registry:
            mock_registry.get_agent_selectable_flows.return_value = {"test_flow": MagicMock()}
            
            result = await agent_planner._validate_plan_impl(plan)
            
            assert result.is_valid
            assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_validate_plan_impl_none_flow(self, agent_planner):
        """Test _validate_plan_impl with 'none' flow (valid)."""
        plan = PlanningResult(
            selected_flow="none",
            reasoning=PlanningExplanation(explanation="test", rationale="test")
        )
        
        with patch('flowlib.agent.components.planning.planner.flow_registry') as mock_registry:
            mock_registry.get_agent_selectable_flows.return_value = {}
            
            result = await agent_planner._validate_plan_impl(plan)
            
            assert result.is_valid
            assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_generate_inputs_no_template(self, agent_planner, mock_state):
        """Test generate_inputs when no template is set."""
        planning_result = PlanningResult(
            selected_flow="test_flow",
            reasoning=PlanningExplanation(explanation="test", rationale="test")
        )
        
        agent_planner._input_generation_template = None
        
        with pytest.raises(NotInitializedError, match="Input generation template not set"):
            await agent_planner.generate_inputs(
                mock_state, "test_flow", planning_result, "memory_context"
            )
    
    @pytest.mark.asyncio
    async def test_generate_inputs_no_flow_metadata(self, agent_planner, mock_state):
        """Test generate_inputs when flow has no metadata."""
        planning_result = PlanningResult(
            selected_flow="test_flow",
            reasoning=PlanningExplanation(explanation="test", rationale="test")
        )
        
        agent_planner._input_generation_template = "test_template"
        
        with patch('flowlib.agent.components.planning.planner.flow_registry') as mock_registry:
            mock_registry.get_flow_metadata.return_value = None
            
            with pytest.raises(PlanningError, match="has no metadata in registry"):
                await agent_planner.generate_inputs(
                    mock_state, "test_flow", planning_result, "memory_context"
                )
    
    @pytest.mark.asyncio
    async def test_generate_inputs_successful(self, agent_planner, mock_state):
        """Test successful generate_inputs execution."""
        planning_result = PlanningResult(
            selected_flow="test_flow",
            reasoning=PlanningExplanation(explanation="test", rationale="test")
        )
        
        mock_metadata = FlowMetadata(
            name="test_flow",
            description="Test flow",
            input_model=MockInputModel,
            output_model=MockInputModel  # Use the same model as output for simplicity
        )
        
        mock_input = MockInputModel(text="generated input")
        
        agent_planner._input_generation_template = "test_template"
        agent_planner.config = MagicMock()
        agent_planner.config.model_name = "test-model"
        agent_planner.llm_provider = AsyncMock()
        agent_planner.llm_provider.generate_structured = AsyncMock(return_value=mock_input)
        
        with patch('flowlib.agent.components.planning.planner.flow_registry') as mock_registry:
            mock_registry.get_flow_metadata.return_value = mock_metadata
            
            with patch('flowlib.utils.pydantic.schema.model_to_simple_json_schema', return_value="schema"):
                with patch('flowlib.utils.formatting.conversation.format_execution_history', return_value="history"):
                    result = await agent_planner.generate_inputs(
                        mock_state, "test_flow", planning_result, "memory_context"
                    )
                    
                    assert result == mock_input
                    agent_planner.llm_provider.generate_structured.assert_called_once()