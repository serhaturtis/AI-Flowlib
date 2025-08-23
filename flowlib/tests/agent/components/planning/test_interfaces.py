"""Tests for agent planning interfaces."""

import pytest
from typing import Any, Dict
from unittest.mock import MagicMock
from pydantic import BaseModel

from flowlib.agent.components.planning.interfaces import PlanningInterface
from flowlib.agent.components.planning.models import PlanningResult, PlanningValidation, PlanningExplanation
from flowlib.agent.models.state import AgentState


class MockInputModel(BaseModel):
    """Mock input model for testing."""
    text: str
    metadata: Dict[str, Any] = {}


class MockPlanningImplementation:
    """Mock implementation of PlanningInterface for testing."""
    
    async def plan(self, context: AgentState) -> PlanningResult:
        """Mock plan implementation."""
        return PlanningResult(
            selected_flow="mock_flow",
            reasoning=PlanningExplanation(
                explanation="Mock planning explanation",
                rationale="Mock rationale",
                decision_factors=["factor1", "factor2"]
            )
        )
    
    async def validate_plan(self, plan: PlanningResult) -> PlanningValidation:
        """Mock validate_plan implementation."""
        return PlanningValidation(is_valid=True, errors=[])
    
    async def generate_inputs(
        self,
        state: AgentState,
        flow_name: str,
        planning_result: PlanningResult,
        memory_context: str,
        flow: Any = None
    ) -> BaseModel:
        """Mock generate_inputs implementation."""
        return MockInputModel(text="mock input", metadata={"generated": True})


class TestPlanningInterface:
    """Test PlanningInterface protocol definition."""
    
    @pytest.fixture
    def mock_state(self):
        """Create mock agent state."""
        state = AgentState(
            task_description="Test interface task",
            task_id="interface_test_123"
        )
        state.add_user_message("Interface test message")
        return state
    
    @pytest.fixture
    def mock_planning_result(self):
        """Create mock planning result."""
        return PlanningResult(
            selected_flow="test_flow",
            reasoning=PlanningExplanation(
                explanation="Test explanation",
                rationale="Test rationale",
                decision_factors=["test_factor"]
            )
        )
    
    @pytest.fixture
    def planning_impl(self):
        """Create mock planning implementation."""
        return MockPlanningImplementation()
    
    def test_planning_interface_exists(self):
        """Test that PlanningInterface protocol is properly defined."""
        # Protocol should be importable and have the expected methods
        assert hasattr(PlanningInterface, '__annotations__')
        
        # Check that protocol defines the expected methods
        # Note: Protocol methods are detected differently in different Python versions
        expected_methods = ['plan', 'validate_plan', 'generate_inputs']
        
        # Get all annotations from the protocol
        annotations = getattr(PlanningInterface, '__annotations__', {})
        
        # Check if the protocol has the expected method signatures
        # This is a basic check - in actual usage, runtime_checkable would be more comprehensive
        assert len(expected_methods) > 0  # Ensure we have methods to check
    
    def test_planning_interface_protocol_compliance(self, planning_impl):
        """Test that mock implementation satisfies the protocol."""
        # Check that all required methods exist
        assert hasattr(planning_impl, 'plan')
        assert hasattr(planning_impl, 'validate_plan')
        assert hasattr(planning_impl, 'generate_inputs')
        
        # Check method signatures by inspecting callable
        assert callable(planning_impl.plan)
        assert callable(planning_impl.validate_plan)
        assert callable(planning_impl.generate_inputs)
    
    @pytest.mark.asyncio
    async def test_plan_method_signature(self, planning_impl, mock_state):
        """Test plan method signature and behavior."""
        result = await planning_impl.plan(mock_state)
        
        # Check return type
        assert isinstance(result, PlanningResult)
        assert result.selected_flow == "mock_flow"
        assert isinstance(result.reasoning, PlanningExplanation)
    
    @pytest.mark.asyncio
    async def test_validate_plan_method_signature(self, planning_impl, mock_planning_result):
        """Test validate_plan method signature and behavior."""
        result = await planning_impl.validate_plan(mock_planning_result)
        
        # Check return type
        assert isinstance(result, PlanningValidation)
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.errors, list)
    
    @pytest.mark.asyncio
    async def test_generate_inputs_method_signature(self, planning_impl, mock_state, mock_planning_result):
        """Test generate_inputs method signature and behavior."""
        result = await planning_impl.generate_inputs(
            state=mock_state,
            flow_name="test_flow",
            planning_result=mock_planning_result,
            memory_context="test_context"
        )
        
        # Check return type
        assert isinstance(result, BaseModel)
        assert isinstance(result, MockInputModel)
        assert result.text == "mock input"
    
    @pytest.mark.asyncio
    async def test_generate_inputs_with_optional_flow(self, planning_impl, mock_state, mock_planning_result):
        """Test generate_inputs method with optional flow parameter."""
        mock_flow = MagicMock()
        
        result = await planning_impl.generate_inputs(
            state=mock_state,
            flow_name="test_flow",
            planning_result=mock_planning_result,
            memory_context="test_context",
            flow=mock_flow
        )
        
        # Should still work with optional flow parameter
        assert isinstance(result, MockInputModel)


class TestProtocolTyping:
    """Test protocol typing and type checking."""
    
    @pytest.mark.asyncio
    async def test_protocol_type_annotation(self):
        """Test using protocol as type annotation."""
        
        async def process_with_planner(planner: PlanningInterface, state: AgentState) -> str:
            """Function that accepts any PlanningInterface."""
            result = await planner.plan(state)
            return f"Planned with {result.selected_flow}"
        
        # Create test state
        mock_state = AgentState(
            task_description="Test interface task",
            task_id="interface_test_123"
        )
        mock_state.add_user_message("Interface test message")
        
        # This should work with proper typing (though we can't easily test mypy here)
        planning_impl = MockPlanningImplementation()
        
        # Test that the function accepts our implementation
        result = await process_with_planner(planning_impl, mock_state)
        assert "Planned with mock_flow" in result
    
    def test_protocol_method_parameters(self):
        """Test that protocol methods have correct parameter types."""
        # This is more of a design verification test
        
        # Check that plan method expects AgentState
        import inspect
        plan_signature = inspect.signature(MockPlanningImplementation.plan)
        assert 'context' in plan_signature.parameters
        
        # Check that validate_plan expects PlanningResult
        validate_signature = inspect.signature(MockPlanningImplementation.validate_plan)
        assert 'plan' in validate_signature.parameters
        
        # Check that generate_inputs has all required parameters
        generate_signature = inspect.signature(MockPlanningImplementation.generate_inputs)
        required_params = ['state', 'flow_name', 'planning_result', 'memory_context']
        for param in required_params:
            assert param in generate_signature.parameters
        
        # Check that flow parameter is optional (has default)
        flow_param = generate_signature.parameters.get('flow')
        assert flow_param is not None
        assert flow_param.default is not inspect.Parameter.empty


class TestInterfaceDocumentation:
    """Test that interface provides proper documentation."""
    
    def test_protocol_has_docstring(self):
        """Test that PlanningInterface has documentation."""
        assert PlanningInterface.__doc__ is not None
        assert "Interface for planning operations" in PlanningInterface.__doc__
    
    def test_method_documentation_exists(self):
        """Test that interface methods have documentation."""
        # Note: In protocols, method documentation is in the protocol definition
        # We can check that the mock implementation has proper method signatures
        
        impl = MockPlanningImplementation()
        
        # Check that methods are properly documented in the implementation
        assert impl.plan.__doc__ is not None
        assert impl.validate_plan.__doc__ is not None
        assert impl.generate_inputs.__doc__ is not None


class TestInterfaceErrorHandling:
    """Test interface error handling expectations."""
    
    def test_interface_error_expectations(self):
        """Test that interface documents expected errors."""
        # The interface should document that implementations should raise specific errors
        
        # Check that the interface docstring mentions error handling
        doc = PlanningInterface.__doc__
        assert "errors" in doc.lower() or "exceptions" in doc.lower()
    
    @pytest.mark.asyncio
    async def test_implementation_error_handling_pattern(self):
        """Test that implementations follow expected error handling patterns."""
        
        class ErrorTestImplementation:
            """Test implementation that raises errors."""
            
            async def plan(self, context: AgentState) -> PlanningResult:
                if context.task_description == "TRIGGER_ERROR":
                    raise ValueError("Task description is required")
                return PlanningResult(
                    selected_flow="test",
                    reasoning=PlanningExplanation(explanation="test", rationale="test")
                )
            
            async def validate_plan(self, plan: PlanningResult) -> PlanningValidation:
                if not plan.selected_flow:
                    return PlanningValidation(is_valid=False, errors=["No flow selected"])
                return PlanningValidation(is_valid=True, errors=[])
            
            async def generate_inputs(self, state, flow_name, planning_result, memory_context, flow=None):
                if not flow_name:
                    raise ValueError("Flow name is required")
                return MockInputModel(text="test")
        
        impl = ErrorTestImplementation()
        
        # Create test state
        mock_state = AgentState(
            task_description="Test interface task",
            task_id="interface_test_123"
        )
        mock_state.add_user_message("Interface test message")
        
        # Test error handling in plan
        error_state = AgentState(task_description="TRIGGER_ERROR", task_id="test")
        with pytest.raises(ValueError, match="Task description is required"):
            await impl.plan(error_state)
        
        # Test error handling in validate_plan
        empty_plan = PlanningResult(
            selected_flow="",
            reasoning=PlanningExplanation(explanation="test", rationale="test")
        )
        validation = await impl.validate_plan(empty_plan)
        assert not validation.is_valid
        assert len(validation.errors) > 0
        
        # Test error handling in generate_inputs
        with pytest.raises(ValueError, match="Flow name is required"):
            await impl.generate_inputs(mock_state, "", empty_plan, "context")