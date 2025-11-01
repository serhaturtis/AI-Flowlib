"""Models for conversation tool."""


from pydantic import Field

from ...models import ToolParameters, ToolResult, ToolStatus


class ConversationParameters(ToolParameters):
    """Parameters for conversation tool."""

    message: str = Field(..., description="The exact text to present to the user")


class ConversationResult(ToolResult):
    """Result from conversation tool execution."""

    response: str | None = Field(default=None, description="Generated response message")
    context_used: str | None = Field(default=None, description="Context that was considered")

    def get_display_content(self) -> str:
        """Get user-friendly display text."""
        if self.status == ToolStatus.SUCCESS and self.response:
            return self.response
        elif self.status == ToolStatus.ERROR:
            return f"Conversation error: {self.message}"
        else:
            return self.message or "No response generated"
