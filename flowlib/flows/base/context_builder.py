"""Context building flow for tool parameter generation.

This flow provides reusable context building functionality that parameter
generation flows can use to build comprehensive context from conversation
history, original user messages, and domain state.
"""

from typing import Any

from pydantic import Field

from flowlib.core.models import StrictBaseModel
from flowlib.flows.decorators.decorators import flow, pipeline


class ContextBuildingInput(StrictBaseModel):
    """Input for context building flow."""

    task_content: str = Field(..., description="Task description to extract parameters from")
    working_directory: str = Field(default=".", description="Working directory context")

    # Conversation context for parameter extraction
    conversation_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recent conversation history for context-aware parameter generation",
    )
    original_user_message: str | None = Field(
        default=None, description="Original user message that started this task"
    )

    # Domain-specific state (optional, for domain-specific tools)
    domain_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific state (e.g., current song, workspace context)",
    )


class ContextBuildingOutput(StrictBaseModel):
    """Output from context building flow."""

    full_context: str = Field(
        ..., description="Built context string with priority-ordered information"
    )
    prompt_variables: dict[str, Any] = Field(
        ..., description="Standard prompt variables including full_context and working_directory"
    )


@flow(
    name="context-building", description="Build comprehensive context for tool parameter generation"
)  # type: ignore[arg-type]
class ContextBuildingFlow:
    """Flow for building comprehensive context from conversation and domain state.

    This flow builds context in priority order:
    1. Original user message (highest priority - what user actually wants)
    2. Recent conversation (last 3 messages - provides context)
    3. Domain state (if applicable - current workspace/session state)
    4. Current task description (decomposed task from planning)

    Usage:
        # Get the flow from registry
        context_flow = flow_registry.get("context-building")

        # Build context
        result = await context_flow.run_pipeline(
            ContextBuildingInput(
                task_content=task_content,
                working_directory=working_dir,
                conversation_history=conversation,
                original_user_message=original_message,
                domain_state=domain_state
            )
        )

        # Use in parameter generation
        parameters = await llm.generate_structured(
            prompt=prompt_template,
            output_type=ParameterType,
            prompt_variables=result.prompt_variables
        )
    """

    @pipeline(input_model=ContextBuildingInput, output_model=ContextBuildingOutput)
    async def run_pipeline(self, request: ContextBuildingInput) -> ContextBuildingOutput:
        """Build comprehensive context with priority order."""

        context_parts = []

        # 1. Original user message (highest priority)
        if request.original_user_message:
            context_parts.append(f"ORIGINAL USER REQUEST: {request.original_user_message}")

        # 2. Recent conversation (last 3 messages for context)
        if request.conversation_history:
            recent_conv = request.conversation_history[-3:]
            conv_text = "\n".join(
                [f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in recent_conv]
            )
            context_parts.append(f"RECENT CONVERSATION:\n{conv_text}")

        # 3. Domain state (if applicable)
        if request.domain_state:
            state_text = self._format_domain_state(request.domain_state)
            context_parts.append(f"DOMAIN STATE:\n{state_text}")

        # 4. Current task description
        context_parts.append(f"CURRENT TASK: {request.task_content}")

        full_context = "\n\n".join(context_parts) if context_parts else request.task_content

        # Build standard prompt variables
        prompt_variables = {
            "full_context": full_context,
            "working_directory": request.working_directory,
        }

        return ContextBuildingOutput(full_context=full_context, prompt_variables=prompt_variables)

    def _format_domain_state(self, domain_state: dict[str, Any]) -> str:
        """Format domain state for inclusion in context."""
        if not domain_state:
            return "No domain state available"

        formatted_parts = []
        for key, value in domain_state.items():
            if isinstance(value, dict):
                # Format nested dicts nicely
                formatted_parts.append(f"{key}:")
                for sub_key, sub_value in value.items():
                    formatted_parts.append(f"  {sub_key}: {sub_value}")
            elif isinstance(value, list) and value:
                # Format lists compactly
                formatted_parts.append(f"{key}: {len(value)} items")
            else:
                formatted_parts.append(f"{key}: {value}")

        return "\n".join(formatted_parts)


def create_context_aware_prompt_template(tool_guidelines: str) -> str:
    """Generate a standard context-aware prompt template.

    This helper creates a consistent prompt structure that all tools should use
    for parameter generation. The prompt includes:
    - Full context section (original request, conversation, domain state, current task)
    - Working directory
    - Important instruction about context prioritization
    - Tool-specific guidelines

    Args:
        tool_guidelines: Tool-specific extraction guidelines (e.g., how to parse file paths)

    Returns:
        Complete prompt template string with placeholders

    Example:
        template = create_context_aware_prompt_template(
            "- Extract file path from the request\\n"
            "- File paths can be absolute or relative"
        )
    """
    return f"""Extract parameters from this context.

# Full Context

{{{{full_context}}}}

# Working Directory

{{{{working_directory}}}}

**IMPORTANT**: The context above contains the ORIGINAL USER REQUEST and CONVERSATION HISTORY.
Extract parameters from the ORIGINAL USER REQUEST when available, using conversation context for additional clarity.

# Guidelines

{tool_guidelines}"""
