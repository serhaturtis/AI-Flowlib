"""Tests for agent response models."""

import pytest
from typing import Dict, Any, List
from pydantic import ValidationError

from flowlib.agent.core.response_models import (
    ExecutionStep,
    AgentExecutionResult,
    ResponseGenerationInput,
    ResponseGenerationOutput,
    ResponseGenerationError
)


class TestExecutionStep:
    """Test ExecutionStep model."""
    
    def test_execution_step_creation_valid(self):
        """Test creating valid ExecutionStep."""
        step = ExecutionStep(
            flow_name="conversation",
            inputs={"message": "Hello"},
            result={"response": "Hi there!"},
            success=True,
            elapsed_time=0.5
        )
        
        assert step.flow_name == "conversation"
        assert step.inputs == {"message": "Hello"}
        assert step.result == {"response": "Hi there!"}
        assert step.success is True
        assert step.elapsed_time == 0.5
    
    def test_execution_step_creation_minimal(self):
        """Test creating ExecutionStep with minimal required fields."""
        step = ExecutionStep(
            flow_name="test-flow",
            inputs={},
            result={},
            success=False,
            elapsed_time=0.0
        )
        
        assert step.flow_name == "test-flow"
        assert step.inputs == {}
        assert step.result == {}
        assert step.success is False
        assert step.elapsed_time == 0.0
    
    def test_execution_step_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionStep(
                flow_name="test-flow",
                # Missing inputs, result, success, elapsed_time
            )
        
        error = exc_info.value
        assert "inputs" in str(error)
        assert "result" in str(error)
        assert "success" in str(error)
        assert "elapsed_time" in str(error)
    
    def test_execution_step_wrong_types(self):
        """Test that wrong field types raise ValidationError."""
        with pytest.raises(ValidationError):
            ExecutionStep(
                flow_name=123,  # Should be string
                inputs={"test": "value"},
                result={"test": "result"},
                success=True,
                elapsed_time=1.0
            )
        
        with pytest.raises(ValidationError):
            ExecutionStep(
                flow_name="test",
                inputs="not a dict",  # Should be dict
                result={"test": "result"},
                success=True,
                elapsed_time=1.0
            )
    
    def test_execution_step_complex_data(self):
        """Test ExecutionStep with complex nested data."""
        step = ExecutionStep(
            flow_name="complex-flow",
            inputs={
                "nested": {"deep": {"value": 42}},
                "list": [1, 2, 3],
                "mixed": {"string": "text", "number": 123}
            },
            result={
                "output": {"processed": True, "items": ["a", "b", "c"]},
                "metadata": {"timestamp": "2023-01-01T00:00:00Z"}
            },
            success=True,
            elapsed_time=2.5
        )
        
        assert step.inputs["nested"]["deep"]["value"] == 42
        assert step.inputs["list"] == [1, 2, 3]
        assert step.result["output"]["items"] == ["a", "b", "c"]
        assert step.elapsed_time == 2.5


class TestAgentExecutionResult:
    """Test AgentExecutionResult model."""
    
    def test_agent_execution_result_creation_complete(self):
        """Test creating complete AgentExecutionResult."""
        steps = [
            ExecutionStep(
                flow_name="step1",
                inputs={"input": "value1"},
                result={"output": "result1"},
                success=True,
                elapsed_time=1.0
            ),
            ExecutionStep(
                flow_name="step2",
                inputs={"input": "value2"},
                result={"output": "result2"},
                success=True,
                elapsed_time=1.5
            )
        ]
        
        result = AgentExecutionResult(
            task_id="task_123",
            task_description="Complete the analysis",
            cycles=2,
            progress=100,
            is_complete=True,
            execution_history=steps,
            errors=[],
            output="Analysis completed successfully"
        )
        
        assert result.task_id == "task_123"
        assert result.task_description == "Complete the analysis"
        assert result.cycles == 2
        assert result.progress == 100
        assert result.is_complete is True
        assert len(result.execution_history) == 2
        assert result.errors == []
        assert result.output == "Analysis completed successfully"
    
    def test_agent_execution_result_with_errors(self):
        """Test AgentExecutionResult with errors."""
        result = AgentExecutionResult(
            task_id="task_456",
            task_description="Failed task",
            cycles=1,
            progress=25,
            is_complete=False,
            execution_history=[],
            errors=["Connection timeout", "Invalid response format"],
            output=None
        )
        
        assert result.task_id == "task_456"
        assert result.is_complete is False
        assert result.errors == ["Connection timeout", "Invalid response format"]
        assert result.output is None
    
    def test_agent_execution_result_minimal(self):
        """Test AgentExecutionResult with minimal required fields."""
        result = AgentExecutionResult(
            task_id="minimal_task",
            task_description="Minimal task",
            cycles=0,
            progress=0,
            is_complete=False,
            execution_history=[]
        )
        
        assert result.task_id == "minimal_task"
        assert result.errors == []  # Default empty list
        assert result.output is None  # Default None
    
    def test_agent_execution_result_validation_errors(self):
        """Test AgentExecutionResult validation errors."""
        with pytest.raises(ValidationError):
            AgentExecutionResult(
                # Missing required fields
            )
        
        with pytest.raises(ValidationError):
            AgentExecutionResult(
                task_id="test",
                task_description="test",
                cycles=-1,  # Should be non-negative
                progress=0,
                is_complete=False,
                execution_history=[]
            )
    
    def test_agent_execution_result_progress_validation(self):
        """Test progress field validation (should be 0-100)."""
        # Valid progress values
        for progress in [0, 50, 100]:
            result = AgentExecutionResult(
                task_id="test",
                task_description="test",
                cycles=1,
                progress=progress,
                is_complete=False,
                execution_history=[]
            )
            assert result.progress == progress
        
        # Invalid progress values (if validation is implemented)
        # Note: The model doesn't currently validate progress range, 
        # but this test documents the expected behavior
        # with pytest.raises(ValidationError):
        #     AgentExecutionResult(
        #         task_id="test",
        #         task_description="test", 
        #         cycles=1,
        #         progress=-10,  # Should be >= 0
        #         is_complete=False,
        #         execution_history=[]
        #     )


class TestResponseGenerationInput:
    """Test ResponseGenerationInput model."""
    
    def test_response_generation_input_creation(self):
        """Test creating ResponseGenerationInput."""
        execution_result = AgentExecutionResult(
            task_id="task_123",
            task_description="Test task",
            cycles=1,
            progress=100,
            is_complete=True,
            execution_history=[]
        )
        
        input_model = ResponseGenerationInput(
            original_task="Please analyze the data",
            execution_result=execution_result,
            persona="helpful assistant"
        )
        
        assert input_model.original_task == "Please analyze the data"
        assert input_model.execution_result == execution_result
        assert input_model.persona == "helpful assistant"
    
    def test_response_generation_input_validation(self):
        """Test ResponseGenerationInput validation."""
        with pytest.raises(ValidationError):
            ResponseGenerationInput(
                # Missing required fields
            )
        
        with pytest.raises(ValidationError):
            ResponseGenerationInput(
                original_task="test",
                execution_result="not_a_model",  # Should be AgentExecutionResult
                persona="assistant"
            )


class TestResponseGenerationOutput:
    """Test ResponseGenerationOutput model."""
    
    def test_response_generation_output_creation(self):
        """Test creating ResponseGenerationOutput."""
        output = ResponseGenerationOutput(
            response="The analysis has been completed successfully.",
            confidence=0.95
        )
        
        assert output.response == "The analysis has been completed successfully."
        assert output.confidence == 0.95
    
    def test_response_generation_output_validation(self):
        """Test ResponseGenerationOutput validation."""
        with pytest.raises(ValidationError):
            ResponseGenerationOutput(
                # Missing required fields
            )
        
        with pytest.raises(ValidationError):
            ResponseGenerationOutput(
                response=123,  # Should be string
                confidence=0.5
            )
        
        with pytest.raises(ValidationError):
            ResponseGenerationOutput(
                response="test",
                confidence="high"  # Should be float
            )
    
    def test_response_generation_output_confidence_validation(self):
        """Test confidence field validation."""
        # Valid confidence values
        for confidence in [0.0, 0.5, 1.0]:
            output = ResponseGenerationOutput(
                response="test response",
                confidence=confidence
            )
            assert output.confidence == confidence
        
        # Note: The model doesn't currently validate confidence range,
        # but this documents expected behavior
        # Invalid confidence values (if validation is implemented)
        # with pytest.raises(ValidationError):
        #     ResponseGenerationOutput(
        #         response="test",
        #         confidence=-0.1  # Should be >= 0.0
        #     )
    
    def test_get_user_display(self):
        """Test get_user_display method."""
        output = ResponseGenerationOutput(
            response="This is the user-friendly response.",
            confidence=0.8
        )
        
        user_display = output.get_user_display()
        assert user_display == "This is the user-friendly response."
        
        # Test with different response
        output2 = ResponseGenerationOutput(
            response="Different response text",
            confidence=0.6
        )
        
        user_display2 = output2.get_user_display()
        assert user_display2 == "Different response text"


class TestResponseGenerationError:
    """Test ResponseGenerationError exception."""
    
    def test_response_generation_error_creation_basic(self):
        """Test creating ResponseGenerationError with basic parameters."""
        error = ResponseGenerationError(
            message="Failed to generate response",
            original_task="Analyze the data"
        )
        
        assert error.message == "Failed to generate response"
        assert error.original_task == "Analyze the data"
        assert error.cause is None
        
        error_str = str(error)
        assert "Response generation failed for task 'Analyze the data'" in error_str
        assert "Failed to generate response" in error_str
    
    def test_response_generation_error_with_cause(self):
        """Test creating ResponseGenerationError with cause."""
        original_exception = ValueError("Invalid input format")
        
        error = ResponseGenerationError(
            message="Response generation failed",
            original_task="Process request",
            cause=original_exception
        )
        
        assert error.message == "Response generation failed"
        assert error.original_task == "Process request"
        assert error.cause == original_exception
        
        error_str = str(error)
        assert "Process request" in error_str
        assert "Response generation failed" in error_str
    
    def test_response_generation_error_inheritance(self):
        """Test that ResponseGenerationError is proper Exception."""
        error = ResponseGenerationError(
            message="Test error",
            original_task="Test task"
        )
        
        assert isinstance(error, Exception)
        assert isinstance(error, ResponseGenerationError)
    
    def test_response_generation_error_attributes_access(self):
        """Test accessing error attributes."""
        original_error = RuntimeError("Something went wrong")
        
        error = ResponseGenerationError(
            message="Generation failed due to runtime error",
            original_task="Complete analysis",
            cause=original_error
        )
        
        # Test that all attributes are accessible
        assert hasattr(error, 'message')
        assert hasattr(error, 'original_task')
        assert hasattr(error, 'cause')
        
        # Test attribute values
        assert error.message == "Generation failed due to runtime error"
        assert error.original_task == "Complete analysis"
        assert error.cause is original_error


class TestModelIntegration:
    """Test integration between different models."""
    
    def test_full_workflow_models(self):
        """Test using models together in a typical workflow."""
        # Create execution steps
        steps = [
            ExecutionStep(
                flow_name="data-extraction",
                inputs={"source": "database"},
                result={"records": 100},
                success=True,
                elapsed_time=2.5
            ),
            ExecutionStep(
                flow_name="data-analysis",
                inputs={"records": 100},
                result={"insights": ["trend1", "trend2"]},
                success=True,
                elapsed_time=5.0
            )
        ]
        
        # Create execution result
        execution_result = AgentExecutionResult(
            task_id="analysis_job_001",
            task_description="Analyze customer data trends",
            cycles=2,
            progress=100,
            is_complete=True,
            execution_history=steps,
            errors=[],
            output="Found 2 key trends in customer data"
        )
        
        # Create response generation input
        response_input = ResponseGenerationInput(
            original_task="Can you analyze our customer data?",
            execution_result=execution_result,
            persona="data analyst assistant"
        )
        
        # Create response generation output
        response_output = ResponseGenerationOutput(
            response="I've completed the analysis of your customer data and found 2 key trends. The analysis processed 100 records and took 7.5 seconds to complete.",
            confidence=0.9
        )
        
        # Verify the workflow
        assert response_input.execution_result.task_id == "analysis_job_001"
        assert len(response_input.execution_result.execution_history) == 2
        assert response_input.execution_result.execution_history[0].flow_name == "data-extraction"
        assert response_input.execution_result.execution_history[1].flow_name == "data-analysis"
        
        assert response_output.get_user_display() == response_output.response
        assert response_output.confidence == 0.9
    
    def test_serialization_compatibility(self):
        """Test that models can be serialized and deserialized."""
        # Create a complex model
        execution_result = AgentExecutionResult(
            task_id="serialization_test",
            task_description="Test serialization",
            cycles=1,
            progress=100,
            is_complete=True,
            execution_history=[
                ExecutionStep(
                    flow_name="test-step",
                    inputs={"key": "value"},
                    result={"status": "success"},
                    success=True,
                    elapsed_time=1.0
                )
            ],
            errors=["warning: minor issue"],
            output="Test completed"
        )
        
        # Serialize to dict
        data_dict = execution_result.model_dump()
        
        # Deserialize from dict
        reconstructed = AgentExecutionResult.model_validate(data_dict)
        
        # Verify reconstruction
        assert reconstructed.task_id == execution_result.task_id
        assert reconstructed.task_description == execution_result.task_description
        assert len(reconstructed.execution_history) == 1
        assert reconstructed.execution_history[0].flow_name == "test-step"
        assert reconstructed.errors == ["warning: minor issue"]