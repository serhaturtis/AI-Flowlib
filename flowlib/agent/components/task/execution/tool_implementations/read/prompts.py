"""
Prompt templates for the Read tool parameter generation.

This module provides prompts for converting natural language task descriptions
into structured ReadParameters.
"""

from pydantic import Field

from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase

@prompt("read_tool_parameter_generation")
class ReadToolParameterGenerationPrompt(ResourceBase):
    """Prompt for generating ReadParameters from task description."""

    template: str = Field(default="""Extract read parameters from this context.

# Full Context

{{full_context}}

# Working Directory

{{working_directory}}

**IMPORTANT**: The context above contains the ORIGINAL USER REQUEST and CONVERSATION HISTORY.
Extract parameters from the ORIGINAL USER REQUEST when available, using conversation context for additional clarity.

# Guidelines

- If the task mentions line ranges like "lines 10-20", set start_line=10 and line_count=11
- If it says "first 5 lines", set start_line=1 and line_count=5
- If it says "last 10 lines" and you don't know the file size, read entire file
- File paths can be absolute or relative to the working directory
- Default encoding is utf-8

Extract the file path and any specific line reading requirements from the task description.""")
