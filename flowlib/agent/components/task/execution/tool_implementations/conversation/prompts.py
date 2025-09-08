"""Prompts for conversation tool."""

from typing import ClassVar
from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase


@prompt("conversation_response_generation")
class ConversationResponseGenerationPrompt(ResourceBase):
    """Prompt for generating conversation responses."""
    
    template: ClassVar[str] = """You are {{persona}}.

Respond to this message: {{message}}

Provide a helpful, clear response that directly addresses the user's message or question.
Be concise but informative while staying in character."""