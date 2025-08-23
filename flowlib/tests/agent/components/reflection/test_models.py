"""Tests for reflection models."""

import pytest
from pydantic import ValidationError, BaseModel
from unittest.mock import Mock

from flowlib.agent.components.reflection.models import (
    ReflectionResult,
    StepReflectionResult,
    ReflectionInput,
    StepReflectionInput,
    PlanReflectionContext
)
from flowlib.flows.models.results import FlowResult


class TestReflectionResult:
    """Test ReflectionResult model."""
    
    def test_minimal_reflection_result(self):
        """Test creating ReflectionResult with minimal fields."""
        result = ReflectionResult(reflection="Test reflection")
        
        assert result.reflection == "Test reflection"
        assert result.progress == 0
        assert result.is_complete is False
        assert result.completion_reason is None
        assert result.insights is None
    
    def test_full_reflection_result(self):
        """Test creating ReflectionResult with all fields."""
        result = ReflectionResult(
            reflection="Detailed analysis",
            progress=75,
            is_complete=True,
            completion_reason="All objectives achieved",
            insights=["Insight 1", "Insight 2", "Insight 3"]
        )
        
        assert result.reflection == "Detailed analysis"
        assert result.progress == 75
        assert result.is_complete is True
        assert result.completion_reason == "All objectives achieved"
        assert len(result.insights) == 3
        assert "Insight 2" in result.insights
    
    def test_reflection_result_validation(self):
        """Test ReflectionResult validation."""
        # Missing required field
        with pytest.raises(ValidationError) as exc_info:
            ReflectionResult()
        assert "reflection" in str(exc_info.value)
        
        # Invalid progress value (should be int)
        with pytest.raises(ValidationError):
            ReflectionResult(reflection="Test", progress="not_an_int")
    
    def test_reflection_result_json_serialization(self):
        """Test JSON serialization of ReflectionResult."""
        result = ReflectionResult(
            reflection="Test",
            progress=50,
            insights=["Test insight"]
        )
        
        json_data = result.model_dump_json()
        assert "reflection" in json_data
        assert "50" in json_data
        assert "Test insight" in json_data


class TestStepReflectionResult:
    """Test StepReflectionResult model."""
    
    def test_step_reflection_result(self):
        """Test creating StepReflectionResult."""
        result = StepReflectionResult(
            step_id="step_1",
            reflection="Step completed successfully",
            step_success=True,
            key_observation="Data processed correctly"
        )
        
        assert result.step_id == "step_1"
        assert result.reflection == "Step completed successfully"
        assert result.step_success is True
        assert result.key_observation == "Data processed correctly"
    
    def test_step_reflection_minimal(self):
        """Test minimal StepReflectionResult."""
        result = StepReflectionResult(
            step_id="step_2",
            reflection="Failed",
            step_success=False
        )
        
        assert result.step_id == "step_2"
        assert result.step_success is False
        assert result.key_observation is None
    
    def test_step_reflection_validation(self):
        """Test StepReflectionResult validation."""
        # Missing required fields
        with pytest.raises(ValidationError):
            StepReflectionResult(step_id="test")  # Missing reflection and step_success


class TestReflectionInput:
    """Test ReflectionInput model."""
    
    def test_reflection_input_minimal(self):
        """Test creating ReflectionInput with required fields."""
        mock_flow_result = Mock(spec=FlowResult)
        
        input_data = ReflectionInput(
            task_description="Test task",
            flow_name="test_flow",
            flow_status="success",
            flow_result=mock_flow_result,
            state_summary="Current state",
            execution_history_text="History",
            planning_rationale="Rationale",
            cycle=1
        )
        
        assert input_data.task_description == "Test task"
        assert input_data.flow_name == "test_flow"
        assert input_data.flow_status == "success"
        assert input_data.flow_result == mock_flow_result
        assert input_data.progress == 0  # Default value
        assert input_data.memory_context is None
        assert input_data.flow_inputs is None
    
    def test_reflection_input_full(self):
        """Test creating ReflectionInput with all fields."""
        mock_flow_result = Mock(spec=FlowResult)
        mock_flow_inputs = Mock(spec=BaseModel)
        
        input_data = ReflectionInput(
            task_description="Complex task",
            flow_name="complex_flow",
            flow_status="error",
            flow_result=mock_flow_result,
            flow_inputs=mock_flow_inputs,
            state_summary="Error state",
            execution_history_text="Long history",
            planning_rationale="Complex rationale",
            cycle=5,
            progress=45,
            memory_context="task_context"
        )
        
        assert input_data.cycle == 5
        assert input_data.progress == 45
        assert input_data.memory_context == "task_context"
        assert input_data.flow_inputs == mock_flow_inputs
    
    def test_reflection_input_validation(self):
        """Test ReflectionInput validation."""
        # Test missing required fields
        with pytest.raises(ValidationError) as exc_info:
            ReflectionInput(
                task_description="Test",
                flow_name="test"
                # Missing other required fields
            )
        assert "flow_status" in str(exc_info.value)


class TestStepReflectionInput:
    """Test StepReflectionInput model."""
    
    def test_step_reflection_input(self):
        """Test creating StepReflectionInput."""
        mock_flow_result = Mock(spec=FlowResult)
        mock_flow_inputs = Mock(spec=BaseModel)
        
        input_data = StepReflectionInput(
            task_description="Main task",
            step_id="step_1",
            step_intent="Process data",
            step_rationale="Need to transform input",
            flow_name="data_processor",
            flow_inputs=mock_flow_inputs,
            flow_result=mock_flow_result,
            current_progress=25
        )
        
        assert input_data.task_description == "Main task"
        assert input_data.step_id == "step_1"
        assert input_data.step_intent == "Process data"
        assert input_data.step_rationale == "Need to transform input"
        assert input_data.flow_name == "data_processor"
        assert input_data.current_progress == 25
    
    def test_step_reflection_input_validation(self):
        """Test StepReflectionInput validation."""
        # Missing required fields
        with pytest.raises(ValidationError) as exc_info:
            StepReflectionInput(
                task_description="Test",
                step_id="step_1"
                # Missing other required fields
            )
        
        error_str = str(exc_info.value)
        assert "step_intent" in error_str
        assert "flow_name" in error_str


class TestPlanReflectionContext:
    """Test PlanReflectionContext model."""
    
    def test_plan_reflection_context_minimal(self):
        """Test creating PlanReflectionContext with minimal fields."""
        step_reflections = [
            StepReflectionResult(
                step_id="step_1",
                reflection="Step 1 done",
                step_success=True
            )
        ]
        
        context = PlanReflectionContext(
            task_description="Complete task",
            plan_status="SUCCESS",
            step_reflections=step_reflections,
            state_summary="Final state",
            execution_history_text="Full history",
            current_progress=100
        )
        
        assert context.task_description == "Complete task"
        assert context.plan_status == "SUCCESS"
        assert context.plan_error is None
        assert len(context.step_reflections) == 1
        assert context.current_progress == 100
    
    def test_plan_reflection_context_with_error(self):
        """Test PlanReflectionContext with error."""
        step_reflections = [
            StepReflectionResult(
                step_id="step_1",
                reflection="Step 1 done",
                step_success=True
            ),
            StepReflectionResult(
                step_id="step_2",
                reflection="Step 2 failed",
                step_success=False,
                key_observation="Missing resource"
            )
        ]
        
        context = PlanReflectionContext(
            task_description="Failed task",
            plan_status="ERROR",
            plan_error="Step 2 failed due to missing resource",
            step_reflections=step_reflections,
            state_summary="Error state",
            execution_history_text="History with error",
            current_progress=50
        )
        
        assert context.plan_status == "ERROR"
        assert context.plan_error == "Step 2 failed due to missing resource"
        assert len(context.step_reflections) == 2
        assert not context.step_reflections[1].step_success
    
    def test_plan_reflection_context_validation(self):
        """Test PlanReflectionContext validation."""
        # Empty step_reflections should be valid
        context = PlanReflectionContext(
            task_description="Test",
            plan_status="SUCCESS",
            step_reflections=[],
            state_summary="State",
            execution_history_text="History",
            current_progress=0
        )
        assert len(context.step_reflections) == 0
        
        # Invalid step reflection type
        with pytest.raises(ValidationError):
            PlanReflectionContext(
                task_description="Test",
                plan_status="SUCCESS",
                step_reflections=["not_a_step_reflection"],
                state_summary="State",
                execution_history_text="History",
                current_progress=0
            )