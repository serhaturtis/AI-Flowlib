"""Conversation tool implementation."""

from typing import cast

from flowlib.agent.components.task.models import TodoItem

from ...decorators import tool
from ...models import (
    PresentationMode,
    ToolExecutionContext,
    ToolPresentationInterface,
    ToolPresentationPreference,
    ToolResult,
    ToolStatus,
)
from .models import ConversationResult


@tool(name="conversation", tool_category="generic", description="Handle conversational interactions and generate responses")
class ConversationTool(ToolPresentationInterface):
    """Tool for handling conversational interactions.

    This tool generates user-friendly conversational responses and declares
    that its output should be used directly without additional presentation layer processing.
    """

    def get_name(self) -> str:
        """Get tool name."""
        return "conversation"

    def get_description(self) -> str:
        """Get tool description."""
        return "Handle conversational interactions and generate responses"

    def get_presentation_preference(self) -> ToolPresentationPreference:
        """Declare that conversation tool output should be used directly.

        Conversation tools generate user-friendly responses that don't need
        additional LLM processing for presentation.

        Returns:
            ToolPresentationPreference with DIRECT mode
        """
        return ToolPresentationPreference(
            mode=PresentationMode.DIRECT,
            reason="Conversation tool generates user-friendly responses that should be presented directly without additional LLM processing"
        )

    async def execute(
        self,
        todo: TodoItem,
        context: ToolExecutionContext  # Execution context
    ) -> ToolResult:
        """Execute conversation interaction."""

        # Generate response using flow (includes parameter generation)
        try:
            result = await self._generate_conversation(todo, context)

            return ConversationResult(
                status=ToolStatus.SUCCESS,
                message="Response generated successfully",
                response=result.response,
                context_used=None  # No parameters used anymore
            )

        except Exception as e:
            return ConversationResult(
                status=ToolStatus.ERROR,
                message=f"Failed to generate conversation: {str(e)}"
            )

    async def _generate_conversation(self, todo: TodoItem, context: ToolExecutionContext) -> ConversationResult:
        """Generate conversation using flow."""
        from flowlib.flows.registry.registry import flow_registry

        from .flow import ConversationFlow, ConversationInput

        # Get the conversation flow class
        flow_obj = flow_registry.get("conversation-generation")
        if flow_obj is None:
            raise RuntimeError("Conversation generation flow not found in registry")
        flow_instance = cast(ConversationFlow, flow_obj)


        # Use original user message if available, otherwise fall back to todo content
        task_content = context.original_user_message or todo.content

        # Create flow input with agent persona and conversation history
        flow_input = ConversationInput(
            task_content=task_content,
            working_directory=context.working_directory,
            agent_persona=context.agent_persona,
            conversation_history=context.conversation_history
        )

        # Execute flow to generate conversation
        result = await flow_instance.run_pipeline(flow_input)
        return cast(ConversationResult, result)
