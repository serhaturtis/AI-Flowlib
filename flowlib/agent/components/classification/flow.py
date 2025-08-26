"""
Message Classification System for the Agent Architecture.

This module provides a classifier flow that determines if a user message
requires simple conversation or complex task execution.
"""

from typing import Dict, Any, List, Optional
from pydantic import Field

from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.resources.registry.registry import resource_registry
from flowlib.resources.models.constants import ResourceType
from flowlib.providers.core.registry import provider_registry
# Removed ProviderType import - using config-driven provider access
from flowlib.agent.components.classification.models import MessageClassification, MessageClassifierInput, ConversationMessage


@flow(name="message-classifier-flow", description="Classify user messages into conversation or task", is_infrastructure=True)
class MessageClassifierFlow:
    """Flow that determines if a message requires simple conversation or complex task execution"""
    
    @pipeline(input_model=MessageClassifierInput, output_model=MessageClassification)
    async def run_pipeline(self, input_data: MessageClassifierInput) -> MessageClassification:
        """Classify a user message into conversation or task categories.
        
        Args:
            input_data: Input containing message and conversation history
            
        Returns:
            Classification result with confidence score and category
        """
        # Get the message and conversation history
        message = input_data.message
        conversation_history = input_data.conversation_history
        memory_summary = input_data.memory_context_summary or "No specific memory context provided."
        
        try:
            # Format conversation history as text
            history_text = self._format_conversation_history(conversation_history)
            
            # Create prompt variables
            prompt_vars = {
                "message": message,
                "conversation_history": history_text,
                "memory_context_summary": memory_summary
            }
            
            # Get classification prompt from registry
            classification_prompt = resource_registry.get("message_classifier_prompt")
            if not classification_prompt:
                raise RuntimeError("Could not find message_classifier_prompt in resource registry")
            
            # Get LLM provider using model-driven approach
            llm = await provider_registry.get_by_config("default-llm")
            
            # Generate classification using LLM
            result = await llm.generate_structured(
                prompt=classification_prompt,
                output_type=MessageClassification,
                model_name="default-model",
                prompt_variables=prompt_vars,
            )
            
            # Ensure confidence is between 0 and 1 and set task description if needed
            clamped_confidence = max(0.0, min(1.0, result.confidence))
            task_description = result.task_description
            
            # Ensure task is set for non-conversation messages if missing
            if result.execute_task and not task_description:
                task_description = f"Assist the user with their request: {message}"
            
            # Create new instance with clamped values since model is frozen
            return MessageClassification(
                execute_task=result.execute_task,
                confidence=clamped_confidence,
                category=result.category,
                task_description=task_description
            )
            
        except Exception as e:
            # Let the error bubble up instead of masking it
            raise
    
    def _format_conversation_history(self, history: List[ConversationMessage]) -> str:
        """Format conversation history as text.
        
        Args:
            history: List of conversation messages
            
        Returns:
            Formatted history text
        """
        if not history:
            return "No conversation history available."
            
        formatted = []
        for item in history:
            role = item.role
            content = item.content
            formatted.append(f"{role.capitalize()}: {content}")
            
        return "\n".join(formatted) 