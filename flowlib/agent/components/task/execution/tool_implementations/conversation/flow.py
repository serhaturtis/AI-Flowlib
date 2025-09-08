"""Flow for conversation tool direct response generation."""

from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.core.models import StrictBaseModel
from flowlib.providers.core.registry import provider_registry
from flowlib.resources.registry.registry import resource_registry
from pydantic import Field
from .prompts import ConversationResponseGenerationPrompt


class ConversationInput(StrictBaseModel):
    """Input for conversation flow."""
    
    task_content: str = Field(..., description="User message to respond to")
    working_directory: str = Field(..., description="Working directory context")
    agent_persona: str = Field(..., description="Agent's persona/personality")


class ConversationOutput(StrictBaseModel):
    """Output from conversation flow."""
    
    response: str = Field(..., description="Generated conversation response")


class ConversationResponseModel(StrictBaseModel):
    """Model for structured response generation."""
    
    response: str = Field(..., description="Generated conversation response")


@flow(name="conversation-generation", description="Generate direct response to user message")
class ConversationFlow:
    """Flow for generating conversation response directly from user input."""
    
    @pipeline(input_model=ConversationInput, output_model=ConversationOutput)
    async def run_pipeline(self, request: ConversationInput) -> ConversationOutput:
        """Generate conversation response directly from user message."""
        
        # Get LLM provider
        llm = await provider_registry.get_by_config("default-llm")
        
        # Generate response directly using the user's message
        response_prompt = resource_registry.get("conversation_response_generation", ConversationResponseGenerationPrompt)
        
        response_variables = {
            "message": request.task_content,  # Use user input directly
            "persona": request.agent_persona  # Use agent's persona
        }
        
        response_obj = await llm.generate_structured(
            prompt=response_prompt,
            output_type=ConversationResponseModel,
            model_name="default-model",
            prompt_variables=response_variables
        )
        
        return ConversationOutput(
            response=response_obj.response.strip()
        )