"""Models for task thinking component.

This module defines the input/output models for the TaskThinking component
that provides strategic analysis and reasoning before task decomposition.
"""

from typing import List, Optional
from pydantic import Field

from flowlib.core.models import StrictBaseModel
from ..generation.models import GeneratedTask
from ..models import RequestContext


class TaskComplexityLevel(StrictBaseModel):
    """Analysis of task complexity."""

    level: str = Field(..., description="Complexity level: simple, moderate, complex, expert")
    reasoning: str = Field(..., description="Why this complexity level was assigned")
    estimated_steps: int = Field(..., ge=1, description="Estimated number of execution steps required")
    estimated_duration_minutes: Optional[int] = Field(default=None, ge=1, description="Estimated completion time in minutes")


class ToolRequirement(StrictBaseModel):
    """Analysis of tool requirements for the task."""

    tool_name: str = Field(..., description="Name of the required tool")
    necessity: str = Field(..., description="Necessity level: essential, helpful, optional")
    usage_purpose: str = Field(..., description="How this tool will be used in the task")
    alternatives: List[str] = Field(default_factory=list, description="Alternative tools that could be used")


class TaskChallenge(StrictBaseModel):
    """Potential challenge or risk in task execution."""

    challenge_type: str = Field(..., description="Type of challenge: technical, permission, dependency, resource")
    description: str = Field(..., description="Detailed description of the challenge")
    likelihood: str = Field(..., description="Likelihood: low, medium, high")
    mitigation_strategy: str = Field(..., description="Strategy to address or mitigate this challenge")


class TaskApproach(StrictBaseModel):
    """Strategic approach for task execution."""

    primary_strategy: str = Field(..., description="Main approach to execute the task")
    execution_order: str = Field(..., description="Recommended order of operations")
    parallelization_opportunities: List[str] = Field(
        default_factory=list,
        description="Parts of the task that could be executed in parallel"
    )
    critical_dependencies: List[str] = Field(
        default_factory=list,
        description="Critical dependencies that must be resolved first"
    )


class TaskThought(StrictBaseModel):
    """Complete strategic analysis of the task."""

    # Core analysis
    complexity: TaskComplexityLevel = Field(..., description="Task complexity analysis")
    tool_requirements: List[ToolRequirement] = Field(
        default_factory=list,
        description="Analysis of required and optional tools"
    )
    potential_challenges: List[TaskChallenge] = Field(
        default_factory=list,
        description="Identified challenges and mitigation strategies"
    )
    approach: TaskApproach = Field(..., description="Strategic approach for execution")

    # Context awareness
    role_considerations: str = Field(..., description="How agent role affects task execution")
    available_tools_analysis: str = Field(..., description="Analysis of available tools vs requirements")

    # Strategic insights
    success_probability: float = Field(..., ge=0.0, le=1.0, description="Estimated probability of successful completion")
    key_success_factors: List[str] = Field(
        default_factory=list,
        description="Critical factors that will determine success"
    )
    optimization_opportunities: List[str] = Field(
        default_factory=list,
        description="Ways to optimize task execution"
    )


class TaskThinkingInput(StrictBaseModel):
    """Input for task thinking component."""

    generated_task: GeneratedTask = Field(..., description="Task from generation component")
    context: RequestContext = Field(..., description="Request context with agent role and environment")
    available_tools: List[str] = Field(..., description="Tools available to this agent role")


class TaskThinkingOutput(StrictBaseModel):
    """Output from task thinking component."""

    thinking_result: TaskThought = Field(..., description="Complete strategic analysis of the task")
    enhanced_task_description: str = Field(..., description="Task description enriched with strategic insights")

    # Processing metadata
    success: bool = Field(..., description="Whether thinking was successful")
    processing_time_ms: float = Field(..., description="Time taken for strategic analysis")
    llm_calls_made: int = Field(default=1, description="Number of LLM calls made")