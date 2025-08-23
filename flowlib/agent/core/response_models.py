"""
Pydantic models for agent response generation.

This module defines strict contracts for generating conversational responses
from agent execution results.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class ExecutionStep(BaseModel):
    """Model for a single execution step in agent history."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    flow_name: str = Field(..., description="Name of the executed flow")
    inputs: Dict[str, Any] = Field(..., description="Inputs provided to the flow")
    result: Dict[str, Any] = Field(..., description="Result from flow execution")
    success: bool = Field(..., description="Whether the flow executed successfully")
    elapsed_time: float = Field(..., description="Execution time in seconds")


class AgentExecutionResult(BaseModel):
    """Model for complete agent execution result."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    task_id: str = Field(..., description="Unique identifier for the task")
    task_description: str = Field(..., description="Original task description")
    cycles: int = Field(..., description="Number of execution cycles completed")
    progress: int = Field(..., description="Task progress percentage (0-100)")
    is_complete: bool = Field(..., description="Whether the task is complete")
    execution_history: List[ExecutionStep] = Field(..., description="List of executed steps")
    errors: List[str] = Field(default_factory=list, description="List of errors encountered")
    output: Optional[str] = Field(None, description="Primary output from execution")
    
    @field_validator('cycles')
    @classmethod
    def validate_cycles(cls, v):
        if v < 0:
            raise ValueError('cycles must be non-negative')
        return v
    
    @field_validator('progress')
    @classmethod
    def validate_progress(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('progress must be between 0 and 100')
        return v


class ResponseGenerationInput(BaseModel):
    """Input model for generating conversational responses."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    original_task: str = Field(..., description="The original user task/request")
    execution_result: AgentExecutionResult = Field(..., description="Complete execution result")
    persona: str = Field(..., description="Agent persona for response generation")
    
    
class ResponseGenerationOutput(BaseModel):
    """Output model for conversational response generation."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    response: str = Field(..., description="Generated conversational response")
    confidence: float = Field(..., description="Confidence in response appropriateness (0.0-1.0)")
    
    def get_user_display(self) -> str:
        """Get user-friendly display text."""
        return self.response


class ResponseGenerationError(Exception):
    """Exception raised when response generation fails."""
    
    def __init__(self, message: str, original_task: str, cause: Optional[Exception] = None):
        self.message = message
        self.original_task = original_task
        self.cause = cause
        super().__init__(f"Response generation failed for task '{original_task}': {message}")