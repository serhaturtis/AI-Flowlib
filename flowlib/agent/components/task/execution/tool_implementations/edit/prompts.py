"""Prompts for edit tool."""

from pydantic import Field

from flowlib.resources.decorators.decorators import prompt


@prompt("edit_tool_parameter_generation")
class EditToolParameterGenerationPrompt:
    """Prompt for generating edit tool parameters."""

    template: str = Field(
        default="""Extract edit parameters from this context.

# Full Context

{{full_context}}

# Working Directory

{{working_directory}}

**IMPORTANT**: The context above contains the ORIGINAL USER REQUEST and CONVERSATION HISTORY.
Extract parameters from the ORIGINAL USER REQUEST when available, using conversation context for additional clarity.

# Guidelines

Extract the file path and edit operations from the task description. Be precise with text matching - it must match exactly including whitespace and formatting."""
    )
