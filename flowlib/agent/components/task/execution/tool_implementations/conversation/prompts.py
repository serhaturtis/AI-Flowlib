"""Prompts for conversation tool."""

from pydantic import Field

from flowlib.resources.decorators.decorators import prompt


@prompt("conversation_response_generation")
class ConversationResponseGenerationPrompt:
    """Prompt for generating conversation responses."""

    template: str = Field(
        default="""Conversation History:
{{conversation_history}}

User: {{message}}

Respond to what the user said. Base your response only on the conversation shown above. Be concise and natural."""
    )
