"""Flow for conversation tool direct response generation."""

from typing import Any, cast

from pydantic import Field

from flowlib.agent.models.conversation import ConversationMessage
from flowlib.core.models import StrictBaseModel
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.llm.base import LLMProvider, PromptTemplate
from flowlib.resources.registry.registry import resource_registry
from flowlib.config.required_resources import RequiredAlias


class ConversationInput(StrictBaseModel):
    """Input for conversation flow."""

    task_content: str = Field(..., description="User message to respond to")
    working_directory: str = Field(..., description="Working directory context")
    conversation_history: list[ConversationMessage] = Field(
        default_factory=list, description="Recent conversation history"
    )


class ConversationOutput(StrictBaseModel):
    """Output from conversation flow."""

    response: str = Field(..., description="Generated conversation response")


class ConversationResponseModel(StrictBaseModel):
    """Model for structured response generation."""

    response: str = Field(..., description="Generated conversation response")


@flow(name="conversation-generation", description="Generate direct response to user message")  # type: ignore[arg-type]
class ConversationFlow:
    """Flow for generating conversation response directly from user input."""

    @pipeline(input_model=ConversationInput, output_model=ConversationOutput)
    async def run_pipeline(self, request: ConversationInput) -> ConversationOutput:
        """Generate conversation response directly from user message."""

        # Get LLM provider
        llm = cast(LLMProvider, await provider_registry.get_by_config(RequiredAlias.DEFAULT_LLM.value))

        # Generate response directly using the user's message
        response_prompt = resource_registry.get("conversation_response_generation")

        # Format conversation history
        history_text = self._format_conversation_history(request.conversation_history)

        response_variables: dict[str, Any] = {
            "message": request.task_content,
            "conversation_history": history_text,
        }

        response_obj = await llm.generate_structured(
            prompt=cast(PromptTemplate, response_prompt),
            output_type=ConversationResponseModel,
            model_name=RequiredAlias.DEFAULT_MODEL.value,
            prompt_variables=response_variables,
        )

        return ConversationOutput(response=response_obj.response.strip())

    def _format_conversation_history(self, history: list[ConversationMessage]) -> str:
        """Format conversation history for prompt."""
        if not history:
            return "No previous conversation"

        formatted = []
        for msg in history[-10:]:  # Last 10 messages
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            content = msg.content
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)
