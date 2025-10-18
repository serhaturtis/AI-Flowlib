"""Prompts for bash tool."""

from pydantic import Field

from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase

@prompt("bash_tool_parameter_generation")
class BashToolParameterGenerationPrompt(ResourceBase):
    """Prompt for generating bash tool parameters."""

    template: str = Field(default="""Extract bash parameters from this context.

# Full Context

{{full_context}}

# Working Directory

{{working_directory}}

**IMPORTANT**: The context above contains the ORIGINAL USER REQUEST and CONVERSATION HISTORY.
Extract parameters from the ORIGINAL USER REQUEST when available, using conversation context for additional clarity.

# Guidelines

Extract the command from the task description. Be careful to preserve the exact command syntax including quotes, pipes, and redirections.""")
