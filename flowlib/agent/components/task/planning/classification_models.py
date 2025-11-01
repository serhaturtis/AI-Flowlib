"""Models for classification-based task planning.

This module provides models for the classification-based planning architecture where:
1. First, classify the request type (conversation/single_tool/multi_step)
2. Then route to specialized generator with type-specific schema
3. Prevents contradictions through mutually exclusive schemas
"""

from typing import Literal

from pydantic import Field, field_validator

from flowlib.resources.models.base import StrictBaseModel

# Classification Models


class TaskClassification(StrictBaseModel):
    """Classification of user request type - first stage output."""

    task_type: Literal["conversation", "single_tool", "multi_step"] = Field(
        ...,
        description=(
            "Type of task:\n"
            "- conversation: Pure conversation (greeting, question, clarification) - NO tools needed\n"
            "- single_tool: Task can be handled by EXACTLY ONE tool call\n"
            "- multi_step: Complex task requiring MULTIPLE tools in sequence"
        ),
    )
    reasoning: str = Field(
        ..., description="Why this classification was chosen based on the user's request"
    )


# Specialized Plan Models - Mutually Exclusive Schemas


class ConversationPlan(StrictBaseModel):
    """Plan for pure conversation - NO tools involved."""

    message_type: Literal["conversation"] = Field(
        default="conversation", description="Always 'conversation' for this plan type"
    )
    reasoning: str = Field(..., description="Why a conversational response is appropriate")
    response_guidance: str = Field(
        ..., description="Guidance on what the conversational response should cover"
    )
    expected_outcome: str = Field(
        ..., description="What the user should expect from the conversation"
    )


class SingleToolStep(StrictBaseModel):
    """The single tool step for single-tool plans."""

    tool_name: str = Field(..., description="Name of the tool to use")
    step_description: str = Field(..., description="What this tool will accomplish")
    parameters: dict = Field(default_factory=dict, description="Parameters to pass to the tool")


class SingleToolPlan(StrictBaseModel):
    """Plan for tasks requiring EXACTLY ONE tool call."""

    message_type: Literal["single_tool"] = Field(
        default="single_tool", description="Always 'single_tool' for this plan type"
    )
    reasoning: str = Field(..., description="Why a single tool is sufficient for this task")
    step: SingleToolStep = Field(..., description="The single tool step to execute")
    expected_outcome: str = Field(..., description="What the user should expect as the result")


class MultiStepPlanStep(StrictBaseModel):
    """A single step in a multi-step plan."""

    tool_name: str = Field(..., description="Name of the tool to use for this step")
    step_description: str = Field(..., description="What this step accomplishes")
    parameters: dict = Field(default_factory=dict, description="Parameters to pass to the tool")
    depends_on_step: int | None = Field(
        default=None, description="Index of step this depends on (0-based)"
    )


class MultiStepPlan(StrictBaseModel):
    """Plan for complex tasks requiring MULTIPLE tool calls."""

    message_type: Literal["multi_step"] = Field(
        default="multi_step", description="Always 'multi_step' for this plan type"
    )
    reasoning: str = Field(
        ..., description="Why multiple steps are needed and how they work together"
    )
    steps: list[MultiStepPlanStep] = Field(
        ...,
        min_length=2,
        description="Ordered list of steps to execute - MUST contain at least 2 steps",
    )
    expected_outcome: str = Field(
        ..., description="What the user should expect as the final result"
    )

    @field_validator("steps")
    @classmethod
    def validate_step_count(cls, v: list[MultiStepPlanStep]) -> list[MultiStepPlanStep]:
        """Ensure multi-step plans have at least 2 steps."""
        if len(v) < 2:
            raise ValueError(
                f"Multi-step plan must have at least 2 steps, got {len(v)}. "
                "Use single_tool classification for 1-step tasks."
            )
        return v
