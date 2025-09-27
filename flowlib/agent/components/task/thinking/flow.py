"""Task thinking flow implementation.

This flow provides strategic analysis and reasoning about tasks before decomposition,
enabling more intelligent and efficient task execution.
"""

from typing import List, cast, Dict, Any
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.llm.base import LLMProvider, PromptTemplate
from flowlib.resources.registry.registry import resource_registry
from flowlib.agent.core.context.models import ConversationMessage
from .models import TaskThinkingInput, TaskThinkingOutput, TaskThought
from ..models import RequestContext


@flow(  # type: ignore[arg-type]
    name="task-thinking",
    description="Analyze tasks strategically and create comprehensive execution plans",
    is_infrastructure=False
)
class TaskThinkingFlow:
    """Provides strategic analysis and reasoning about tasks before decomposition."""

    @pipeline(input_model=TaskThinkingInput, output_model=TaskThinkingOutput)
    async def run_pipeline(self, input_data: TaskThinkingInput) -> TaskThinkingOutput:
        """Analyze task strategically and create execution plan.

        Args:
            input_data: Contains generated task, context, and available tools

        Returns:
            TaskThinkingOutput with strategic analysis and enhanced task description
        """
        # Get LLM provider using config-driven approach
        llm = cast(LLMProvider, await provider_registry.get_by_config("default-llm"))

        # Get prompt from registry
        prompt_instance = resource_registry.get("task-thinking-prompt")

        # Format available tools for analysis
        tools_text = self._format_available_tools(input_data.available_tools)

        # Format conversation context
        context_text = self._format_context(input_data.context)

        # Prepare prompt variables
        prompt_vars: Dict[str, Any] = {
            "enhanced_task_description": input_data.generated_task.task_description,
            "agent_role": input_data.context.agent_role or "general_purpose",
            "available_tools": tools_text,
            "working_directory": input_data.context.working_directory,
            "agent_persona": input_data.context.agent_persona or "AI Assistant",
            "conversation_context": context_text
        }

        # Generate strategic analysis using LLM
        thinking_result = await llm.generate_structured(
            prompt=cast(PromptTemplate, prompt_instance),
            output_type=TaskThought,
            model_name="default-model",
            prompt_variables=prompt_vars
        )

        # Enhance task description with strategic insights
        enhanced_description = self._enhance_task_description(
            original_task=input_data.generated_task.task_description,
            thinking_result=thinking_result
        )

        return TaskThinkingOutput(
            thinking_result=thinking_result,
            enhanced_task_description=enhanced_description,
            success=True,
            processing_time_ms=0.0,  # Will be set by component
            llm_calls_made=1
        )

    def _format_available_tools(self, tools: List[str]) -> str:
        """Format available tools list for the prompt."""
        if not tools:
            return "No tools available"

        # Import here to avoid circular imports
        from flowlib.agent.components.task.execution.registry import tool_registry

        tools_list = []
        for tool_name in tools:
            try:
                metadata = tool_registry.get_metadata(tool_name)
                if metadata:
                    tools_list.append(f"- {tool_name}: {metadata.description}")
                else:
                    tools_list.append(f"- {tool_name}: Available tool")
            except KeyError:
                tools_list.append(f"- {tool_name}: Available tool")

        return "\n".join(tools_list)

    def _format_context(self, context: RequestContext) -> str:
        """Format RequestContext for the prompt."""
        if not context:
            return "No additional context provided"

        context_lines = []

        if context.agent_name:
            context_lines.append(f"Agent Name: {context.agent_name}")

        if context.previous_messages:
            conversation_history = self._format_conversation_history(context.previous_messages)
            context_lines.append(f"Recent Conversation:\n{conversation_history}")

        if context.memory_context:
            context_lines.append(f"Memory Context: {context.memory_context}")

        return "\n".join(context_lines) if context_lines else "No additional context provided"

    def _format_conversation_history(self, messages: List[ConversationMessage]) -> str:
        """Format conversation history for context."""
        if not messages or len(messages) == 0:
            return "No previous conversation"

        formatted_messages = []
        for msg in messages[-3:]:  # Last 3 messages for thinking context
            # msg is ConversationMessage with role and content attributes
            role = msg.role
            content = msg.content
            formatted_messages.append(f"{role}: {content}")

        return "\n".join(formatted_messages)

    def _enhance_task_description(self, original_task: str, thinking_result: TaskThought) -> str:
        """Enhance task description with strategic insights."""
        enhancements = []

        # Add complexity insight
        complexity_level = thinking_result.complexity.level
        estimated_steps = thinking_result.complexity.estimated_steps
        enhancements.append(f"[{complexity_level.title()} Task - ~{estimated_steps} steps]")

        # Add primary strategy
        strategy = thinking_result.approach.primary_strategy
        enhancements.append(f"[Strategy: {strategy}]")

        # Add critical dependencies if any
        if thinking_result.approach.critical_dependencies:
            deps = ", ".join(thinking_result.approach.critical_dependencies[:2])  # First 2
            enhancements.append(f"[Dependencies: {deps}]")

        # Combine original task with enhancements
        enhancement_prefix = " ".join(enhancements)
        return f"{enhancement_prefix} {original_task}"