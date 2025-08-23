"""Tests for agent planning models."""

import pytest
import uuid
from typing import Dict, Any, List
from pydantic import ValidationError

from flowlib.agent.components.planning.models import (
    PlanningExplanation,
    PlanningResult,
    PlanningValidation,
    PlanStep,
    Plan
)


class TestPlanningExplanation:
    """Test PlanningExplanation model."""
    
    def test_planning_explanation_creation(self):
        """Test creating PlanningExplanation with required fields."""
        explanation = PlanningExplanation(
            explanation="This is a test explanation",
            rationale="Test rationale",
            decision_factors=["factor1", "factor2", "factor3"]
        )
        
        assert explanation.explanation == "This is a test explanation"
        assert explanation.rationale == "Test rationale"
        assert explanation.decision_factors == ["factor1", "factor2", "factor3"]
    
    def test_planning_explanation_minimal(self):
        """Test creating PlanningExplanation with minimal required fields."""
        explanation = PlanningExplanation(explanation="Minimal explanation")
        
        assert explanation.explanation == "Minimal explanation"
        assert explanation.rationale is None
        assert explanation.decision_factors == []
    
    def test_planning_explanation_optional_fields(self):
        """Test PlanningExplanation with optional fields."""
        explanation = PlanningExplanation(
            explanation="Test explanation",
            decision_factors=["single_factor"]
        )
        
        assert explanation.explanation == "Test explanation"
        assert explanation.rationale is None
        assert explanation.decision_factors == ["single_factor"]
    
    def test_planning_explanation_validation(self):
        """Test PlanningExplanation validation."""
        # Test missing required field
        with pytest.raises(ValidationError):
            PlanningExplanation()
        
        # Test invalid type for decision_factors
        with pytest.raises(ValidationError):
            PlanningExplanation(
                explanation="Test",
                decision_factors="not_a_list"
            )
    
    def test_planning_explanation_serialization(self):
        """Test PlanningExplanation serialization."""
        explanation = PlanningExplanation(
            explanation="Serialization test",
            rationale="Test rationale",
            decision_factors=["factor1", "factor2"]
        )
        
        data = explanation.model_dump()
        
        assert data["explanation"] == "Serialization test"
        assert data["rationale"] == "Test rationale"
        assert data["decision_factors"] == ["factor1", "factor2"]
    
    def test_planning_explanation_deserialization(self):
        """Test PlanningExplanation deserialization."""
        data = {
            "explanation": "Deserialization test",
            "rationale": "Test rationale",
            "decision_factors": ["factor1", "factor2"]
        }
        
        explanation = PlanningExplanation(**data)
        
        assert explanation.explanation == "Deserialization test"
        assert explanation.rationale == "Test rationale"
        assert explanation.decision_factors == ["factor1", "factor2"]


class TestPlanningResult:
    """Test PlanningResult model."""
    
    @pytest.fixture
    def sample_reasoning(self):
        """Create sample reasoning for testing."""
        return PlanningExplanation(
            explanation="Test reasoning",
            rationale="Test rationale",
            decision_factors=["factor1", "factor2"]
        )
    
    def test_planning_result_creation(self, sample_reasoning):
        """Test creating PlanningResult with required fields."""
        result = PlanningResult(
            selected_flow="test_flow",
            reasoning=sample_reasoning
        )
        
        assert result.selected_flow == "test_flow"
        assert result.reasoning == sample_reasoning
    
    def test_planning_result_validation_missing_fields(self):
        """Test PlanningResult validation with missing fields."""
        # Missing selected_flow
        with pytest.raises(ValidationError):
            PlanningResult(reasoning=PlanningExplanation(explanation="test"))
        
        # Missing reasoning
        with pytest.raises(ValidationError):
            PlanningResult(selected_flow="test_flow")
    
    def test_planning_result_with_none_flow(self, sample_reasoning):
        """Test PlanningResult with 'none' flow (valid case)."""
        result = PlanningResult(
            selected_flow="none",
            reasoning=sample_reasoning
        )
        
        assert result.selected_flow == "none"
        assert result.reasoning == sample_reasoning
    
    def test_planning_result_serialization(self, sample_reasoning):
        """Test PlanningResult serialization."""
        result = PlanningResult(
            selected_flow="test_flow",
            reasoning=sample_reasoning
        )
        
        data = result.model_dump()
        
        assert data["selected_flow"] == "test_flow"
        assert "reasoning" in data
        assert data["reasoning"]["explanation"] == "Test reasoning"
    
    def test_planning_result_deserialization(self):
        """Test PlanningResult deserialization."""
        data = {
            "selected_flow": "deserialization_flow",
            "reasoning": {
                "explanation": "Deserialization test",
                "rationale": "Test rationale",
                "decision_factors": ["factor1"]
            }
        }
        
        result = PlanningResult(**data)
        
        assert result.selected_flow == "deserialization_flow"
        assert isinstance(result.reasoning, PlanningExplanation)
        assert result.reasoning.explanation == "Deserialization test"


class TestPlanningValidation:
    """Test PlanningValidation model."""
    
    def test_planning_validation_valid(self):
        """Test creating valid PlanningValidation."""
        validation = PlanningValidation(is_valid=True, errors=[])
        
        assert validation.is_valid is True
        assert validation.errors == []
    
    def test_planning_validation_invalid(self):
        """Test creating invalid PlanningValidation."""
        errors = ["Error 1", "Error 2", "Error 3"]
        validation = PlanningValidation(is_valid=False, errors=errors)
        
        assert validation.is_valid is False
        assert validation.errors == errors
    
    def test_planning_validation_minimal(self):
        """Test creating PlanningValidation with minimal fields."""
        validation = PlanningValidation(is_valid=True)
        
        assert validation.is_valid is True
        assert validation.errors == []
    
    def test_planning_validation_validation_errors(self):
        """Test PlanningValidation validation."""
        # Missing required field
        with pytest.raises(ValidationError):
            PlanningValidation()
        
        # Invalid type for errors
        with pytest.raises(ValidationError):
            PlanningValidation(is_valid=True, errors="not_a_list")
    
    def test_planning_validation_serialization(self):
        """Test PlanningValidation serialization."""
        validation = PlanningValidation(
            is_valid=False,
            errors=["Validation error 1", "Validation error 2"]
        )
        
        data = validation.model_dump()
        
        assert data["is_valid"] is False
        assert data["errors"] == ["Validation error 1", "Validation error 2"]


class TestPlanStep:
    """Test PlanStep model."""
    
    def test_plan_step_creation(self):
        """Test creating PlanStep with required fields."""
        step = PlanStep(
            flow_name="test_flow",
            step_intent="Test step intent",
            rationale="Test step rationale"
        )
        
        assert step.flow_name == "test_flow"
        assert step.step_intent == "Test step intent"
        assert step.rationale == "Test step rationale"
        assert step.expected_outcome is None
        
        # Check that step_id is auto-generated
        assert step.step_id is not None
        assert isinstance(step.step_id, str)
        # Should be a valid UUID string
        uuid.UUID(step.step_id)  # Will raise if not valid UUID
    
    def test_plan_step_with_optional_fields(self):
        """Test creating PlanStep with optional fields."""
        step = PlanStep(
            flow_name="test_flow",
            step_intent="Test intent",
            rationale="Test rationale",
            expected_outcome="Expected test outcome"
        )
        
        assert step.expected_outcome == "Expected test outcome"
    
    def test_plan_step_custom_id(self):
        """Test creating PlanStep with custom step_id."""
        custom_id = "custom-step-id-123"
        step = PlanStep(
            step_id=custom_id,
            flow_name="test_flow",
            step_intent="Test intent",
            rationale="Test rationale"
        )
        
        assert step.step_id == custom_id
    
    def test_plan_step_validation(self):
        """Test PlanStep validation."""
        # Missing required fields
        with pytest.raises(ValidationError):
            PlanStep()
        
        with pytest.raises(ValidationError):
            PlanStep(flow_name="test_flow")
        
        with pytest.raises(ValidationError):
            PlanStep(
                flow_name="test_flow",
                step_intent="Test intent"
                # Missing rationale
            )
    
    def test_plan_step_extra_fields_forbidden(self):
        """Test that PlanStep forbids extra fields."""
        with pytest.raises(ValidationError):
            PlanStep(
                flow_name="test_flow",
                step_intent="Test intent",
                rationale="Test rationale",
                extra_field="not_allowed"  # Should raise ValidationError
            )
    
    def test_plan_step_serialization(self):
        """Test PlanStep serialization."""
        step = PlanStep(
            flow_name="serialization_flow",
            step_intent="Serialization test",
            rationale="Test serialization",
            expected_outcome="Serialized data"
        )
        
        data = step.model_dump()
        
        assert data["flow_name"] == "serialization_flow"
        assert data["step_intent"] == "Serialization test"
        assert data["rationale"] == "Test serialization"
        assert data["expected_outcome"] == "Serialized data"
        assert "step_id" in data


class TestPlan:
    """Test Plan model."""
    
    @pytest.fixture
    def sample_steps(self):
        """Create sample plan steps for testing."""
        return [
            PlanStep(
                flow_name="step1_flow",
                step_intent="First step",
                rationale="First step rationale"
            ),
            PlanStep(
                flow_name="step2_flow",
                step_intent="Second step",
                rationale="Second step rationale"
            )
        ]
    
    def test_plan_creation(self, sample_steps):
        """Test creating Plan with required fields."""
        plan = Plan(
            task_description="Test task description",
            steps=sample_steps
        )
        
        assert plan.task_description == "Test task description"
        assert plan.steps == sample_steps
        assert plan.overall_rationale is None
        
        # Check that plan_id is auto-generated
        assert plan.plan_id is not None
        assert isinstance(plan.plan_id, str)
        uuid.UUID(plan.plan_id)  # Will raise if not valid UUID
    
    def test_plan_with_optional_fields(self, sample_steps):
        """Test creating Plan with optional fields."""
        plan = Plan(
            task_description="Test task",
            steps=sample_steps,
            overall_rationale="Overall test rationale"
        )
        
        assert plan.overall_rationale == "Overall test rationale"
    
    def test_plan_custom_id(self, sample_steps):
        """Test creating Plan with custom plan_id."""
        custom_id = "custom-plan-id-456"
        plan = Plan(
            plan_id=custom_id,
            task_description="Test task",
            steps=sample_steps
        )
        
        assert plan.plan_id == custom_id
    
    def test_plan_empty_steps(self):
        """Test creating Plan with empty steps list."""
        plan = Plan(
            task_description="Task with no steps",
            steps=[]
        )
        
        assert plan.steps == []
    
    def test_plan_validation(self):
        """Test Plan validation."""
        # Missing required fields
        with pytest.raises(ValidationError):
            Plan()
        
        with pytest.raises(ValidationError):
            Plan(task_description="Test task")
        
        with pytest.raises(ValidationError):
            Plan(steps=[])
    
    def test_plan_extra_fields_forbidden(self, sample_steps):
        """Test that Plan forbids extra fields."""
        with pytest.raises(ValidationError):
            Plan(
                task_description="Test task",
                steps=sample_steps,
                extra_field="not_allowed"  # Should raise ValidationError
            )
    
    def test_plan_serialization(self, sample_steps):
        """Test Plan serialization."""
        plan = Plan(
            task_description="Serialization test task",
            steps=sample_steps,
            overall_rationale="Test serialization rationale"
        )
        
        data = plan.model_dump()
        
        assert data["task_description"] == "Serialization test task"
        assert data["overall_rationale"] == "Test serialization rationale"
        assert "plan_id" in data
        assert "steps" in data
        assert len(data["steps"]) == 2
        
        # Check steps serialization
        for i, step_data in enumerate(data["steps"]):
            assert step_data["flow_name"] == sample_steps[i].flow_name
            assert step_data["step_intent"] == sample_steps[i].step_intent
    
    def test_plan_deserialization(self):
        """Test Plan deserialization."""
        data = {
            "plan_id": "deserialization-plan-id",
            "task_description": "Deserialization test",
            "steps": [
                {
                    "step_id": "step-1",
                    "flow_name": "flow1",
                    "step_intent": "Intent 1",
                    "rationale": "Rationale 1"
                },
                {
                    "step_id": "step-2",
                    "flow_name": "flow2",
                    "step_intent": "Intent 2",
                    "rationale": "Rationale 2"
                }
            ],
            "overall_rationale": "Overall deserialization test"
        }
        
        plan = Plan(**data)
        
        assert plan.plan_id == "deserialization-plan-id"
        assert plan.task_description == "Deserialization test"
        assert plan.overall_rationale == "Overall deserialization test"
        assert len(plan.steps) == 2
        
        # Check steps deserialization
        assert isinstance(plan.steps[0], PlanStep)
        assert plan.steps[0].step_id == "step-1"
        assert plan.steps[0].flow_name == "flow1"
        
        assert isinstance(plan.steps[1], PlanStep)
        assert plan.steps[1].step_id == "step-2"
        assert plan.steps[1].flow_name == "flow2"


class TestModelIntegration:
    """Test integration between different planning models."""
    
    def test_planning_result_with_plan_step_data(self):
        """Test creating PlanningResult that could be derived from Plan."""
        # Simulate converting a Plan to PlanningResult
        plan_step = PlanStep(
            flow_name="integration_flow",
            step_intent="Integration test",
            rationale="Test integration between models"
        )
        
        # Create PlanningResult using data from PlanStep
        result = PlanningResult(
            selected_flow=plan_step.flow_name,
            reasoning=PlanningExplanation(
                explanation=plan_step.step_intent,
                rationale=plan_step.rationale,
                decision_factors=["integration_test"]
            )
        )
        
        assert result.selected_flow == "integration_flow"
        assert result.reasoning.explanation == "Integration test"
        assert result.reasoning.rationale == "Test integration between models"
    
    def test_validation_with_planning_result(self):
        """Test validating a PlanningResult."""
        result = PlanningResult(
            selected_flow="valid_flow",
            reasoning=PlanningExplanation(
                explanation="Valid flow selected",
                rationale="Flow exists and is appropriate"
            )
        )
        
        # Simulate validation
        validation = PlanningValidation(is_valid=True, errors=[])
        
        assert validation.is_valid
        assert len(validation.errors) == 0
    
    def test_complex_plan_to_planning_result_conversion(self):
        """Test converting complex Plan to PlanningResult."""
        complex_plan = Plan(
            task_description="Complex multi-step task",
            steps=[
                PlanStep(
                    flow_name="analysis_flow",
                    step_intent="Analyze the problem",
                    rationale="Need to understand the requirements first"
                ),
                PlanStep(
                    flow_name="solution_flow",
                    step_intent="Generate solution",
                    rationale="Based on analysis, create solution"
                ),
                PlanStep(
                    flow_name="validation_flow",
                    step_intent="Validate solution",
                    rationale="Ensure solution meets requirements"
                )
            ],
            overall_rationale="Systematic approach to problem solving"
        )
        
        # Convert to PlanningResult (taking first step as primary)
        first_step = complex_plan.steps[0]
        result = PlanningResult(
            selected_flow=first_step.flow_name,
            reasoning=PlanningExplanation(
                explanation=first_step.step_intent,
                rationale=complex_plan.overall_rationale or first_step.rationale,
                decision_factors=[step.step_intent for step in complex_plan.steps]
            )
        )
        
        assert result.selected_flow == "analysis_flow"
        assert result.reasoning.explanation == "Analyze the problem"
        assert result.reasoning.rationale == "Systematic approach to problem solving"
        assert len(result.reasoning.decision_factors) == 3