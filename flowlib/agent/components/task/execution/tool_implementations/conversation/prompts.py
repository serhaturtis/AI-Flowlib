"""Prompts for conversation tool."""

from pydantic import Field

from flowlib.resources.decorators.decorators import prompt


@prompt("conversation_response_generation")
class ConversationResponseGenerationPrompt:
    """Prompt for generating conversation responses."""

    template: str = Field(
        default="""{{persona}}

History:
{{conversation_history}}

User: {{message}}

Respond to what user said. Use only actual history. Be concise."""
    )
