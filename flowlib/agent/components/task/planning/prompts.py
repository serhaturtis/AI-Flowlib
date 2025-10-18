"""Prompts for structured task planning."""

from pydantic import Field

from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase


@prompt(name="structured-planning-prompt")
class StructuredPlanningPrompt(ResourceBase):
    """Prompt for generating structured execution plans in a single call.

    This follows the Plan-and-Execute pattern optimized for local LLMs:
    - One planning call instead of multiple phases
    - Structured output with clear plan steps
    - Minimal cognitive load per decision
    """

    template: str = Field(default="""Request: {{user_message}}

Tools:
{{available_tools}}

Working directory: {{working_directory}}
Conversation: {{conversation_history}}
State: {{domain_state}}

Planning process:
1. Read tool descriptions carefully
2. If ANY tool says "COMPLETE WORKFLOW", "ONE SINGLE STEP", "STANDALONE", or "DO NOT combine" - stop and use ONLY that tool
3. If user names a specific tool, use that exact tool
4. For greetings/questions, use conversation tool
5. Otherwise, create multi-step plan with specific values and dependencies

Create single-step plan for complete workflow tools. Multi-step only when no single tool exists.""")
