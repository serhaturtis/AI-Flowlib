"""Prompt definitions for shell command generation flow."""

from typing import ClassVar
from flowlib.resources.models.base import ResourceBase
from flowlib.resources.decorators.decorators import prompt


@prompt("shell_command_generation")
class ShellCommandGenerationPrompt(ResourceBase):
    """Prompt to generate a shell command based on intent and available tools."""
    
    template: ClassVar[str] = """
You are an expert shell command generator.
Your task is to generate a SINGLE, executable shell command to achieve the given intent, using ONLY the provided available commands.

**Intent:** {{intent}}
**Target Resource:** {{target_resource}}
**Parameters:** {{parameters}}
**Desired Output:** {{output_description}}

**Available Commands:**
{{available_commands_list}}

**Guidelines:**
- Construct a single command line that achieves the intent exactly
- Use ONLY commands from the 'Available Commands' list
- Extract actual content from the intent and parameters - DO NOT use placeholders or empty strings
- For file creation: use echo "ACTUAL_CONTENT" > filename with the real content from the intent
- Be mindful of quoting and escaping special characters
- If the intent cannot be achieved with available tools, output: echo "Error: Cannot achieve intent with available tools."

Return your response as a JSON object with this exact structure:
{
    "command": "the shell command to execute",
    "reasoning": "brief explanation of why this command achieves the intent"
}
"""
    
    config: ClassVar[dict] = {
        "max_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.95
    }