"""
Prompts for message classification.

This module provides prompt templates for message classification.
"""

from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase
from typing import ClassVar

@prompt("message_classifier_prompt")
class MessageClassifierPrompt(ResourceBase):
    """Prompt for classifying user messages"""
    
    template: ClassVar[str] = """You are a message classifier for an agent system.

Your task is to classify this user message based on whether it can be handled through simple conversation or requires executing a complex task.

User message: {{message}}

Conversation history:
{{conversation_history}}

Classification rules:
- CONVERSATION: Greetings, simple questions answerable from general knowledge or conversation history, social responses, acknowledgements, confirmations. The key is that you can respond fully and accurately *without* needing external tools, real-time data, or complex computation.
- TASK: Any request asking the agent to *perform an action* or *do something*. This includes, but is not limited to: instructions to run commands (e.g., `df -h`), manage files, perform calculations, generate content, analyze information, answer complex questions needing external/up-to-date information (e.g., current prices, weather), or fulfill multi-step requests. If the user asks you to *do* something beyond simple conversation, it's a TASK.

## Memory Context Summary
{{memory_context_summary}}

Analyze the message carefully based on complexity, intent, and available memory context. Does fulfilling the request require capabilities beyond simple conversational recall and generation, or accessing information not present in memory?

Output a classification with:
1. execute_task: Set to `true` if the user is asking the agent to *perform any action* or *do something* as defined in the TASK category above. Set to `false` only for simple conversational exchanges (greetings, acknowledgements, questions answerable from history/general knowledge).
2. confidence: How confident you are in the classification (0.0-1.0).
3. category: Specific category (e.g., greeting, question, instruction, research_query, planning_request).
4. task_description: If execute_task=true, provide a clear, concise task description reformulating the user's request for the agent. Leave empty or null for conversation messages.

"""