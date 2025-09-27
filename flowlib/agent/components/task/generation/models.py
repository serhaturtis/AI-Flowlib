"""Models for task generation component.

This module defines the input/output models for the TaskGenerator component
that classifies user messages and enriches them with context.
"""

from typing import List, Optional
from pydantic import Field

from flowlib.core.models import StrictBaseModel
from flowlib.agent.models.conversation import ConversationMessage


class RelevantHistoryItem(StrictBaseModel):
    """Simplified history item that LLM can generate.
    
    This is a simple model for LLM output, not the full ConversationMessage.
    """
    role: str = Field(description="Role of the speaker (user/assistant/system)")
    content: str = Field(description="Message content")
    relevance_reason: Optional[str] = Field(
        default=None, 
        description="Why this message is relevant to the current task"
    )


class TaskGenerationInput(StrictBaseModel):
    """Input for task generation component."""
    
    # Core message
    user_message: str = Field(description="The user's message to process")
    
    # Context
    conversation_history: List[ConversationMessage] = Field(
        default_factory=list, 
        description="Previous conversation messages for context"
    )
    agent_persona: str = Field(description="Agent's persona/personality")
    
    # Execution context
    working_directory: str = Field(description="Current working directory")


# Remove ExecutionStrategy - TaskDecomposer handles this


class GeneratedTask(StrictBaseModel):
    """Task generated from user message with context."""
    
    # Task for the agent to execute
    task_description: str = Field(description="What the agent should do (the actual task)")
    original_user_message: str = Field(description="The original user message that prompted this task")
    
    # Context used for task generation
    context_summary: str = Field(description="Summary of conversation context used in generation")
    relevant_history: List[RelevantHistoryItem] = Field(
        default_factory=list,
        description="Relevant conversation history that influenced task generation"
    )


class TaskGenerationOutput(StrictBaseModel):
    """Output from task generation component."""
    
    generated_task: GeneratedTask = Field(description="The task generated from user message")
    success: bool = Field(description="Whether generation was successful")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    # Processing metadata
    processing_time_ms: float = Field(description="Time taken to enrich task")
    llm_calls_made: int = Field(default=1, description="Number of LLM calls made")