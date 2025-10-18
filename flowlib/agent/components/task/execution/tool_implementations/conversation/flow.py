"""Flow for conversation tool direct response generation."""

from typing import Any, Dict, cast

from pydantic import Field

from flowlib.core.models import StrictBaseModel
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.llm.base import LLMProvider, PromptTemplate
from flowlib.resources.registry.registry import resource_registry

from .prompts import ConversationResponseGenerationPrompt


class ConversationInput(StrictBaseModel):
    """Input for conversation flow."""

    task_content: str = Field(..., description="User message to respond to")
    working_directory: str = Field(..., description="Working directory context")
    agent_persona: str = Field(..., description="Agent's persona/personality")
    conversation_history: list[dict] = Field(default_factory=list, description="Recent conversation history")


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
        llm = cast(LLMProvider, await provider_registry.get_by_config("default-llm"))

        # Generate response directly using the user's message
        response_prompt = resource_registry.get("conversation_response_generation", ConversationResponseGenerationPrompt)

        # Format conversation history
        history_text = self._format_conversation_history(request.conversation_history)

        response_variables: Dict[str, Any] = {
            "message": request.task_content,
            "persona": request.agent_persona,
            "conversation_history": history_text
        }

        response_obj = await llm.generate_structured(
            prompt=cast(PromptTemplate, response_prompt),
            output_type=ConversationResponseModel,
            model_name="default-model",
            prompt_variables=response_variables
        )

        return ConversationOutput(
            response=response_obj.response.strip()
        )

    def _format_conversation_history(self, history: list[dict]) -> str:
        """Format conversation history for prompt."""
        if not history:
            return "No previous conversation"

        formatted = []
        for msg in history[-10:]:  # Last 10 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)
