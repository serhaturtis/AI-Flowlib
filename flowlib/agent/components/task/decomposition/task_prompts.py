"""Prompts for task decomposition flow.

This module provides the prompt templates for the task decomposition system
that breaks down user requests into actionable TODO items.
"""

from typing import ClassVar
from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase


@prompt(name="task-decomposition-prompt")
class TaskDecompositionPrompt(ResourceBase):
    """Prompt for decomposing user requests into structured TODO items with tool assignments."""
    
    template: ClassVar[str] = """Analyze and decompose this task: {{task_description}}

Available tools:
{{available_tools}}

Additional context:
{{context}}

Strategic insights from thinking phase:
{{thinking_insights}}

DECOMPOSITION GUIDELINES:
- Use the strategic insights above to guide decomposition decisions
- Select tools ONLY from the available tools list above
- Match tool capabilities to task requirements based on strategic analysis
- **CRITICAL: Simple greetings/conversations → create ONLY 1 TODO with appropriate conversation tool**
- **DO NOT break conversations into multiple TODOs**
- Complex tasks → break into logical, executable steps with dependencies following the strategic approach
- Consider file operations, system commands, and interactive responses as separate task categories
- Prioritize tasks based on dependencies, logical flow, and strategic execution order
- Account for identified challenges and use recommended mitigation strategies

IMPORTANT: Every TODO MUST have an assigned_tool field set to one of the available tools above!

EXECUTION STRATEGY:
- Determine optimal execution order considering dependencies
- Estimate realistic completion timeframes
- Provide clear reasoning for decomposition decisions

Focus on creating the minimum necessary TODOs that fully accomplish the task.

CONVERSATION RULE: If the task involves responding to user messages, greetings, questions, or general conversation → CREATE ONLY ONE TODO with conversation tool. Do not decompose conversations into multiple steps.
"""