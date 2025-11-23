"""Models for context validation."""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from flowlib.agent.models.conversation import ConversationMessage
from flowlib.core.models import MutableStrictBaseModel, StrictBaseModel

# LLM-facing models - Two-step approach

# Step 1: Classification (simple boolean)


class LLMContextClassification(StrictBaseModel):
    """Simple classification: Does the request have sufficient context?"""

    has_sufficient_context: bool = Field(
        ...,
        description="Whether there is enough information to proceed with planning and execution",
    )

    confidence: str = Field(..., description="Confidence in assessment: low, medium, high")

    reasoning: str = Field(
        ...,
        description="Brief explanation of why context is sufficient or insufficient",
    )


# Step 2a: Generation for PROCEED case


class LLMProceedGeneration(StrictBaseModel):
    """Detailed reasoning when context is sufficient."""

    detailed_reasoning: str = Field(
        ...,
        description="Detailed analysis of what information is available and why it's sufficient",
    )


# Step 2b: Generation for CLARIFY case


class LLMClarifyGeneration(StrictBaseModel):
    """Detailed clarification info when context is insufficient."""

    missing_information: list[str] = Field(
        ..., description="Specific information gaps that prevent proceeding"
    )

    clarification_questions: list[str] = Field(
        ...,
        description="Specific questions to ask user to gather missing information",
    )

    detailed_reasoning: str = Field(
        ...,
        description="Detailed explanation of what's missing and why it's needed",
    )


# Full models


class ValidationResult(StrictBaseModel):
    """Complete validation result with metadata."""

    has_sufficient_context: bool
    confidence: float = Field(..., description="Confidence as decimal 0.0-1.0")
    reasoning: str
    next_action: Literal["proceed", "clarify"]
    missing_information: list[str]
    clarification_questions: list[str]

    # Enriched context for planning (used when user responds to clarification)
    enriched_task_context: str | None = Field(
        default=None,
        description="Enriched task context combining original request with clarification responses",
    )

    # Metadata fields
    validation_time_ms: float = Field(default=0.0, description="Time taken to validate")
    llm_calls_made: int = Field(default=1, description="Number of LLM calls")


class ValidationInput(StrictBaseModel):
    """Input for context validation."""

    user_message: str = Field(..., description="The user's current message")
    conversation_history: list[ConversationMessage] = Field(
        default_factory=list, description="Recent conversation for context"
    )
    domain_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Current domain-specific state (workspace, session data, etc.)",
    )


class ValidationOutput(StrictBaseModel):
    """Output from context validation."""

    result: ValidationResult = Field(..., description="The validation result")
    success: bool = Field(..., description="Whether validation succeeded")
    processing_time_ms: float = Field(default=0.0, description="Time taken to validate")


# Clarification tracking


class PendingClarification(MutableStrictBaseModel):
    """Tracks pending clarification request.

    When validator identifies insufficient context and generates clarification questions,
    this state is stored to recognize when the user responds.
    """

    original_message: str = Field(
        ..., description="Original user message that triggered clarification"
    )
    questions_asked: list[str] = Field(
        ..., description="Clarification questions that were sent to user"
    )
    missing_information: list[str] = Field(..., description="What information was missing")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When clarification was requested"
    )


class EnrichedContext(StrictBaseModel):
    """Context enriched with clarification responses.

    After user responds to clarification questions, their responses are merged
    with the original request to create enriched context for planning.
    """

    original_request: str = Field(..., description="Original user request")
    clarification_responses: dict[str, str] = Field(
        ..., description="Mapping of questions to user responses or AGENT_CHOICE for delegations"
    )
    combined_context: str = Field(..., description="Formatted combined context for planning")


# Clarification response parsing


class LLMClarificationResponseParsing(StrictBaseModel):
    """LLM-facing model for parsing user responses to clarification questions."""

    response_type: Literal["delegation", "specific_answers", "mixed", "insufficient"] = Field(
        ...,
        description=(
            "delegation: user asks agent to make decisions | "
            "specific_answers: user provides explicit answers | "
            "mixed: combination of both | "
            "insufficient: response is vague/unhelpful/unclear"
        ),
    )

    confidence: str = Field(..., description="Confidence in parsing: low, medium, high")

    reasoning: str = Field(..., description="Explanation of how the response was interpreted")

    delegation_items: list[str] = Field(
        default_factory=list,
        description="Items user delegated to agent (empty if no delegation or insufficient)",
    )

    provided_information: dict[str, str] = Field(
        default_factory=dict,
        description="Specific information provided by user as key-value pairs (empty if delegation or insufficient)",
    )

    enrichment_note: str = Field(
        ...,
        description="Summary of how to enrich the original request with this response (or explanation of why insufficient)",
    )

    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="Follow-up questions to ask if response is insufficient (empty otherwise)",
    )


class ClarificationResponseParsingResult(StrictBaseModel):
    """Complete result from parsing clarification response."""

    response_type: Literal["delegation", "specific_answers", "mixed", "insufficient"]
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence as decimal")
    reasoning: str
    delegation_items: list[str]
    provided_information: dict[str, str]
    enrichment_note: str
    follow_up_questions: list[str] = Field(
        default_factory=list, description="Follow-up questions if insufficient"
    )

    # Metadata
    parsing_time_ms: float = Field(default=0.0, description="Time taken to parse")
    llm_calls_made: int = Field(default=1, description="Number of LLM calls")


class ClarificationResponseParsingInput(StrictBaseModel):
    """Input for parsing clarification response."""

    user_response: str = Field(..., description="User's response to clarification questions")
    questions_asked: list[str] = Field(..., description="Questions that were asked")
    missing_information: list[str] = Field(..., description="Information that was missing")
    original_request: str = Field(..., description="Original user request")


class ClarificationResponseParsingOutput(StrictBaseModel):
    """Output from clarification response parsing."""

    result: ClarificationResponseParsingResult = Field(..., description="Parsing result")
    success: bool = Field(..., description="Whether parsing succeeded")
    processing_time_ms: float = Field(default=0.0, description="Total processing time")
