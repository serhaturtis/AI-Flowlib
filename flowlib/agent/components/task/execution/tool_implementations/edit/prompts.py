"""Prompts for edit tool."""

from typing import ClassVar
from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase


@prompt("edit_tool_parameter_generation")
class EditToolParameterGenerationPrompt(ResourceBase):
    """Prompt for generating edit tool parameters."""
    
    template: ClassVar[str] = """Extract edit parameters from this task: {{task_content}}

Working directory: {{working_directory}}

Extract the file path and edit operations from the task description. Be precise with text matching - it must match exactly including whitespace and formatting."""