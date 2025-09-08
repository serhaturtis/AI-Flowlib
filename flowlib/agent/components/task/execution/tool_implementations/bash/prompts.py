"""Prompts for bash tool."""

from typing import ClassVar
from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase


@prompt("bash_tool_parameter_generation")
class BashToolParameterGenerationPrompt(ResourceBase):
    """Prompt for generating bash tool parameters."""
    
    template: ClassVar[str] = """Extract bash parameters from this task: {{task_content}}

Working directory: {{working_directory}}

Extract the command from the task description. Be careful to preserve the exact command syntax including quotes, pipes, and redirections."""