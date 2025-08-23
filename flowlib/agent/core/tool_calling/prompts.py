"""Tool selection prompts for agent tool calling.

This module provides prompt templates for LLM-driven tool selection
following flowlib's prompt resource pattern.
"""

from flowlib.resources.models.base import ResourceBase
from flowlib.resources.decorators.decorators import prompt
from flowlib.core.models import StrictBaseModel
from typing import Dict, Any


@prompt("tool-selection")
class ToolSelectionPrompt(ResourceBase):
    """Prompt for LLM tool selection with structured output."""
    
    template: str = """You are an autonomous agent that can execute tools to complete tasks.

TASK: {task_description}

AVAILABLE TOOLS:
{available_tools}

CONTEXT:
- Working Directory: {working_directory}
- Additional Context: {context}

INSTRUCTIONS:
1. Analyze the task and determine which tools are needed
2. Select the most appropriate tools from the available list
3. Generate proper parameters for each selected tool
4. Provide reasoning for each tool selection
5. Return a valid JSON array of tool calls

OUTPUT FORMAT (must be valid JSON):
[
    {{
        "tool_name": "tool_name_here",
        "parameters": {{
            "param1": "value1",
            "param2": "value2"
        }},
        "reasoning": "Why this tool is needed for the task"
    }}
]

CONSTRAINTS:
- Only use tools from the available list
- Ensure all required parameters are provided
- Use appropriate parameter types and values
- Maximum {max_tools} tools per task
- If no tools are needed, return an empty array []

Generate the tool calls now:"""
    
    config: Dict[str, Any] = {
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.9
    }