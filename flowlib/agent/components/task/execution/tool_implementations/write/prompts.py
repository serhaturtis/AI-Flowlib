"""Prompts for write tool."""

from pydantic import Field

from flowlib.resources.decorators.decorators import prompt


@prompt("write_tool_parameter_generation")
class WriteToolParameterGenerationPrompt:
    """Prompt for generating write tool parameters."""

    template: str = Field(
        default="""Extract write parameters from this context.

# Full Context

{{full_context}}

# Working Directory

{{working_directory}}

**IMPORTANT**: The context above contains the ORIGINAL USER REQUEST and CONVERSATION HISTORY.
Extract parameters from the ORIGINAL USER REQUEST when available, using conversation context for additional clarity.

# Guidelines

Extract the file path and content to write from the task description."""
    )
