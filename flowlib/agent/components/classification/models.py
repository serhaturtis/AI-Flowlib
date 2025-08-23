from typing import Dict, Any, List, Optional
from pydantic import Field, ConfigDict
from flowlib.core.models import StrictBaseModel

class ConversationMessage(StrictBaseModel):
    """A single message in conversation history."""
    # Inherits strict configuration from StrictBaseModel
    
    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")

class MessageClassification(StrictBaseModel):
    """Classification result for a user message"""
    # Inherits strict configuration from StrictBaseModel
    
    execute_task: bool = Field(..., description="True if there is need to execute a task, False if conversation")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    category: str = Field(..., description="Message category (greeting, question, instruction, etc.)")
    task_description: Optional[str] = Field(None, description="Task description for execution (when execute_task=True)")

class MessageClassifierInput(StrictBaseModel):
    """Input for message classification"""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    message: str = Field(..., description="The user message to classify")
    conversation_history: List[ConversationMessage] = Field(
        default_factory=list, 
        description="Recent conversation history"
    )
    memory_context_summary: Optional[str] = Field(None, description="Summary of relevant memory context") 