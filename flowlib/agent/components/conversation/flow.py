"""
Simple conversation flow for handling basic user interactions.

This flow provides conversational responses for greetings, simple questions,
and general interactions that don't require complex task execution.
"""

from typing import Optional
from pydantic import Field

from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.providers.core.registry import provider_registry
# Removed ProviderType import - using config-driven provider access
from flowlib.resources.registry.registry import resource_registry
from flowlib.resources.models.constants import ResourceType
from .models import ConversationInput, ConversationOutput


@flow(
    name="conversation",
    description="Handle simple conversational interactions and provide friendly responses",
    is_infrastructure=False
)
class ConversationFlow:
    """Flow for handling basic conversational interactions."""
    
    @pipeline(input_model=ConversationInput, output_model=ConversationOutput)
    async def run_pipeline(self, input_data: ConversationInput) -> ConversationOutput:
        """Generate a conversational response to user input.
        
        Args:
            input_data: Conversation input with user message and context
            
        Returns:
            Conversational response
        """
        user_message = input_data.message
        context = input_data.memory_context_summary or "No previous context available."
        
        # Format conversation history
        conversation_history_text = "No previous conversation history."
        if input_data.conversation_history:
            history_lines = []
            for msg in input_data.conversation_history:
                role = msg.get('role', 'unknown') if isinstance(msg, dict) else getattr(msg, 'role', 'unknown')
                content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')
                history_lines.append(f"{role.title()}: {content}")
            conversation_history_text = "\n".join(history_lines)
        
        # Get LLM provider using config-driven approach
        llm = await provider_registry.get_by_config("default-llm")
        
        # Get prompt from registry
        prompt_instance = resource_registry.get("conversation-prompt")
        
        # Prepare variables
        prompt_vars = {
            "persona": input_data.persona,
            "user_message": user_message,
            "context": context,
            "conversation_history": conversation_history_text
        }
        
        # Generate response using config-driven approach
        result = await llm.generate_structured(
            prompt=prompt_instance,
            output_type=ConversationOutput,
            model_name="default-model",
            prompt_variables=prompt_vars
        )
        
        return result