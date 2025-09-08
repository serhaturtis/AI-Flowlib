"""Models for task debriefing component."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import Field, ConfigDict

from flowlib.core.models import StrictBaseModel
from ..execution.models import ToolResult
from ..models import TodoItem


class DebriefingDecision(str, Enum):
    """Possible debriefing decisions."""
    PRESENT_SUCCESS = "present_success"
    RETRY_WITH_CORRECTION = "retry_with_correction"
    PRESENT_FAILURE = "present_failure"


class IntentAnalysisResult(StrictBaseModel):
    """Result of intent analysis."""
    
    intent_fulfilled: bool = Field(..., description="Whether user's intent was fulfilled")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in analysis")
    user_intent_summary: str = Field(..., description="What the user actually wanted")
    execution_summary: str = Field(..., description="What was actually accomplished")
    gap_analysis: Optional[str] = Field(default=None, description="Why intent wasn't fulfilled")
    is_correctable: bool = Field(..., description="Whether the issue can be corrected with retry")
    correction_suggestion: Optional[str] = Field(default=None, description="How to correct the issue")


class DebriefingInput(StrictBaseModel):
    """Input for task debriefing analysis."""
    
    original_user_message: str = Field(..., description="User's original request")
    generated_task_description: str = Field(..., description="Task generated from user message")
    execution_results: List[ToolResult] = Field(..., description="Results from tool execution")
    todos_executed: List[TodoItem] = Field(..., description="TODOs that were executed")
    cycle_number: int = Field(..., description="Current execution cycle")
    agent_persona: str = Field(..., description="Agent's persona")
    working_directory: str = Field(..., description="Working directory")
    max_cycles: int = Field(default=3, description="Maximum allowed cycles")


class DebriefingOutput(StrictBaseModel):
    """Output from task debriefing."""
    
    decision: DebriefingDecision = Field(..., description="What action to take")
    user_response: Optional[str] = Field(default=None, description="Response to present to user")
    corrective_task: Optional[str] = Field(default=None, description="New task for retry")
    reasoning: str = Field(..., description="Why this decision was made")
    should_continue_cycle: bool = Field(..., description="Whether to continue execution cycles")
    intent_analysis: IntentAnalysisResult = Field(..., description="Analysis of intent fulfillment")


# Flow Input Models

class IntentAnalysisInput(StrictBaseModel):
    """Input for intent analysis flow."""
    
    original_user_message: str = Field(..., description="User's original request")
    generated_task: str = Field(..., description="Generated task description")
    execution_results: str = Field(..., description="Formatted execution results")
    todos_executed: List[TodoItem] = Field(..., description="TODOs that were executed")
    agent_persona: str = Field(..., description="Agent's persona")
    working_directory: str = Field(..., description="Working directory")


class SuccessPresentationInput(StrictBaseModel):
    """Input for success presentation flow."""
    
    original_user_message: str = Field(..., description="User's original request")
    execution_results: str = Field(..., description="Formatted execution results")
    intent_analysis: IntentAnalysisResult = Field(..., description="Analysis of intent fulfillment")
    agent_persona: str = Field(..., description="Agent's persona")


class CorrectiveTaskInput(StrictBaseModel):
    """Input for corrective task generation flow."""
    
    original_user_message: str = Field(..., description="User's original request")
    failed_task: str = Field(..., description="Task that failed to fulfill intent")
    execution_results: str = Field(..., description="Formatted execution results")
    intent_analysis: IntentAnalysisResult = Field(..., description="Analysis of why it failed")
    cycle_number: int = Field(..., description="Current cycle number")
    working_directory: str = Field(..., description="Working directory")


class FailureExplanationInput(StrictBaseModel):
    """Input for failure explanation flow."""
    
    original_user_message: str = Field(..., description="User's original request")
    execution_results: str = Field(..., description="Formatted execution results")
    intent_analysis: IntentAnalysisResult = Field(..., description="Analysis of failure")
    cycles_attempted: int = Field(..., description="Number of cycles attempted")
    agent_persona: str = Field(..., description="Agent's persona")


# Flow Output Models

class IntentAnalysisOutput(StrictBaseModel):
    """Output from intent analysis flow."""
    
    intent_analysis: IntentAnalysisResult = Field(..., description="Analysis result")


class SuccessPresentationOutput(StrictBaseModel):
    """Output from success presentation flow."""
    
    presentation_response: str = Field(..., description="User-friendly success response")


class CorrectiveTaskOutput(StrictBaseModel):
    """Output from corrective task generation flow."""
    
    corrected_task: str = Field(..., description="Improved task for retry")


class FailureExplanationOutput(StrictBaseModel):
    """Output from failure explanation flow."""
    
    failure_explanation: str = Field(..., description="Helpful failure explanation for user")