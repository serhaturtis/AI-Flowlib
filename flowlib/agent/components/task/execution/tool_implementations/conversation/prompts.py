"""Prompts for conversation tool."""

from pydantic import Field

from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase


@prompt("conversation_response_generation")
class ConversationResponseGenerationPrompt(ResourceBase):
    """Prompt for generating conversation responses."""

    template: str = Field(default="""{{persona}}

History:
{{conversation_history}}

User: {{message}}

Respond to what user said. Use only actual history. Be concise.""")
