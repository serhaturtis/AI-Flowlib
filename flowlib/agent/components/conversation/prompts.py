"""
Prompts for conversation flow.

This module defines the prompts used by the conversation flow according to
flowlib conventions.
"""

from typing import ClassVar
from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase


@prompt(name="conversation-prompt")
class ConversationPrompt(ResourceBase):
    """Prompt for generating conversational responses."""
    
    template: ClassVar[str] = """{{persona}}

Conversation History:
{{conversation_history}}

User message: {{user_message}}

Context: {{context}}

Give response to user's input with this persona. Use the conversation history to provide context-aware responses and remember what has been discussed previously.

You must respond with a JSON object in the following format:
{
  "response": "your conversational response here",
  "sentiment": "positive|negative|neutral (optional)"
}

Ensure your response is valid JSON and contains a natural, helpful conversational response."""