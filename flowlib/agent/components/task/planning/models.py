"""Models for structured task planning.

This module provides models for the Plan-and-Execute architecture where a single
LLM call generates a complete structured plan with multiple steps.
"""

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from flowlib.core.models import MutableStrictBaseModel
from flowlib.resources.models.base import StrictBaseModel

if TYPE_CHECKING:
    pass

# LLM-facing models


class LLMPlanStep(StrictBaseModel):
    """A single step in the execution plan."""

    tool_name: str = Field(..., description="Name of the tool to use for this step")
    step_description: str = Field(..., description="What this step accomplishes")
    parameters: dict = Field(default_factory=dict, description="Parameters to pass to the tool")
    depends_on_step: int | None = Field(
        default=None, description="Index of step this depends on (0-based)"
    )


class LLMStructuredPlan(StrictBaseModel):
    """Complete structured plan for task execution."""

    message_type: Literal["conversation", "single_tool", "multi_step"] = Field(
        ...,
        description="Type of message: conversation (greeting/question), single_tool (one tool handles everything), multi_step (multiple tools needed)",
    )
    reasoning: str = Field(..., description="Why this plan was chosen and what it will accomplish")
    steps: list[LLMPlanStep] = Field(
        ...,
        min_length=1,
        description="Ordered list of steps to execute - MUST contain at least 1 step",
    )
    expected_outcome: str = Field(..., description="What the user should expect as the result")


# Full models


class PlanStep(MutableStrictBaseModel):
    """A single step in the execution plan with metadata.

    Mutable to allow updating executed and result fields during execution.
    """

    step_id: str = Field(..., description="Unique identifier for this step")
    tool_name: str = Field(..., description="Name of the tool to use")
    step_description: str = Field(..., description="What this step accomplishes")
    parameters: dict = Field(default_factory=dict, description="Parameters for the tool")
    depends_on_step: int | None = Field(default=None, description="Index of dependent step")
    executed: bool = Field(default=False, description="Whether this step has been executed")
    result: str | None = Field(default=None, description="Result of execution")


class StructuredPlan(StrictBaseModel):
    """Complete structured plan with execution metadata."""

    message_type: Literal["conversation", "single_tool", "multi_step"]
    reasoning: str
    steps: list[PlanStep]
    expected_outcome: str

    # Metadata fields
    plan_id: str = Field(default="", description="Unique identifier for this plan")
    created_at: float = Field(default=0.0, description="Timestamp of plan creation")
    execution_started: bool = Field(default=False, description="Whether execution has started")
    execution_complete: bool = Field(default=False, description="Whether all steps completed")


class PlanningInput(StrictBaseModel):
    """Input for structured planning."""

    user_message: str = Field(..., description="The user's message to plan for")
    conversation_history: list[dict] = Field(
        default_factory=list, description="Recent conversation context"
    )
    available_tools: list[str] = Field(..., description="List of available tool names")
    agent_role: str = Field(default="assistant", description="Role of the agent")
    working_directory: str = Field(default=".", description="Current working directory")

    # FIX: Add domain state for context-aware planning
    domain_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific state (e.g., current song, workspace context, session state)",
    )
    shared_variables: dict[str, str] = Field(
        default_factory=dict, description="Shared string variables from execution context"
    )

    # Context validation result (if validation is enabled)
    validation_result: Any | None = Field(
        default=None, description="Result from context validation component (if enabled)"
    )


class PlanningOutput(StrictBaseModel):
    """Output from structured planning."""

    plan: StructuredPlan = Field(..., description="The generated execution plan")
    success: bool = Field(..., description="Whether planning succeeded")
    processing_time_ms: float = Field(default=0.0, description="Time taken to generate plan")
    llm_calls_made: int = Field(default=1, description="Number of LLM calls")
