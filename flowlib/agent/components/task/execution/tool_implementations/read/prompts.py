"""
Prompt templates for the Read tool parameter generation.

This module provides prompts for converting natural language task descriptions
into structured ReadParameters.
"""

from typing import ClassVar
from flowlib.resources.models.base import ResourceBase
from flowlib.resources.decorators.decorators import prompt


@prompt("read_tool_parameter_generation")
class ReadToolParameterGenerationPrompt(ResourceBase):
    """Prompt for generating ReadParameters from task description."""
    
    template: ClassVar[str] = """Extract read parameters from this task: {{task_content}}

Working directory: {{working_directory}}

Guidelines:
- If the task mentions line ranges like "lines 10-20", set start_line=10 and line_count=11
- If it says "first 5 lines", set start_line=1 and line_count=5
- If it says "last 10 lines" and you don't know the file size, read entire file
- File paths can be absolute or relative to the working directory
- Default encoding is utf-8

Extract the file path and any specific line reading requirements from the task description."""