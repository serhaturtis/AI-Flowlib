"""Tool selection prompts for agent tool calling.

This module provides prompt templates for LLM-driven tool selection
following flowlib's prompt resource pattern.
"""

from flowlib.resources.models.base import ResourceBase
from flowlib.resources.decorators.decorators import prompt
from typing import Dict, Any


@prompt("intent-classification")
class IntentClassificationPrompt(ResourceBase):
    """Prompt for classifying user intent before tool selection."""
    
    template: str = """Classify this user message as either CONVERSATION or ACTION.

USER MESSAGE: {{user_message}}

CORE PRINCIPLE:
- ACTION: The user wants me to DO something - any kind of call to action, task execution, or operational request
- CONVERSATION: The user wants me to SAY something - seeking information, discussion, explanation, or social interaction

CLASSIFICATION APPROACH:
Ask yourself: "Is the user requesting that I perform an operation or task?"
- If YES → ACTION (regardless of how it's phrased)  
- If NO → CONVERSATION (they want information, discussion, or social interaction)

The key distinction is DOING vs TALKING:
- Requesting execution of any task = ACTION
- Requesting information, opinions, or discussion = CONVERSATION

Analyze the user's fundamental intent: Do they want me to PERFORM AN OPERATION or PROVIDE INFORMATION/DISCUSSION?

USER MESSAGE: "{{user_message}}"

Classification:"""
    
    config: Dict[str, Any] = {
        "max_tokens": 200,
        "temperature": 0.1,
        "top_p": 0.9
    }


@prompt("tool-selection")
class ToolSelectionPrompt(ResourceBase):
    """Prompt for LLM tool selection with structured output."""
    
    template: str = """Analyze the user request and select appropriate tools if needed.

USER REQUEST: {{task_description}}

AVAILABLE TOOLS:
{{available_tools}}

Use tools for file operations, commands, or system tasks. For conversations, explanations, or questions, use no tools."""
    
    config: Dict[str, Any] = {
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.9
    }