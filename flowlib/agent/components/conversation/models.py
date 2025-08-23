from typing import Any, Dict, Optional, List
from pydantic import Field, ConfigDict
from flowlib.core.models import StrictBaseModel

class ConversationInput(StrictBaseModel):
    """Input model for conversation flow."""
    # Inherits strict configuration from StrictBaseModel
    
    message: str = Field(..., description="The user message to respond to")
    persona: str = Field(default="A helpful AI agent.", description="The agent's persona for the conversation")
    language: Optional[str] = Field("English", description="Language to use for the response")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default_factory=list, description="List of conversation history entries (e.g., {'role': 'user'/'assistant', 'content': 'message'}) ")
    memory_context_summary: Optional[str] = Field(None, description="Summary of relevant memory context")
    task_result_summary: Optional[str] = Field(None, description="Summary of the result from a previously executed task")

class ConversationOutput(StrictBaseModel):
    """Output model for conversation flow."""
    # Inherits strict configuration from StrictBaseModel
    
    response: str = Field(..., description="The LLM's response to the user message")
    sentiment: Optional[str] = Field(None, description="Optional sentiment analysis of the response")
    
    def get_user_display(self) -> str:
        """Get user-friendly display text for conversation results."""
        return self.response

class ConversationResponse(StrictBaseModel):
    """Structured model for conversation responses."""
    
    response: str = Field(..., description="The response to the user's message")

class ConversationExecuteInput(StrictBaseModel):
    """Complete input model for the execute method including metadata fields."""
    
    input_data: Optional[ConversationInput] = None
    message: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None
    rationale: Optional[str] = Field(None, description="Reasoning for executing this flow (for logging only)")
    flow_context: Optional[Dict[str, Any]] = Field(None, description="Flow execution context") 