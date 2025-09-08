"""Models for conversation tool."""

from typing import Optional
from pydantic import Field
from flowlib.core.models import StrictBaseModel
from ...models import ToolResult, ToolStatus


class ConversationResult(ToolResult):
    """Result from conversation tool execution."""
    
    response: Optional[str] = Field(default=None, description="Generated response message")
    context_used: Optional[str] = Field(default=None, description="Context that was considered")
    
    def get_display_content(self) -> str:
        """Get user-friendly display text."""
        if self.status == ToolStatus.SUCCESS and self.response:
            return self.response
        elif self.status == ToolStatus.ERROR:
            return f"Conversation error: {self.message}"
        else:
            return self.message or "No response generated"