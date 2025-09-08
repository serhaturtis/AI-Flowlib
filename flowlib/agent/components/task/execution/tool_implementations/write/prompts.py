"""Prompts for write tool."""

from typing import ClassVar
from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase


@prompt("write_tool_parameter_generation")
class WriteToolParameterGenerationPrompt(ResourceBase):
    """Prompt for generating write tool parameters."""
    
    template: ClassVar[str] = """Extract write parameters from this task: {{task_content}}

Working directory: {{working_directory}}

Extract the file path and content to write from the task description."""