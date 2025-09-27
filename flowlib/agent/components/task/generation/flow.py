"""Task generation flow implementation.

This flow uses an LLM to classify user messages and enrich them with context
for optimal task processing.
"""

from typing import cast
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.llm.base import LLMProvider, PromptTemplate
from flowlib.resources.registry.registry import resource_registry
from flowlib.agent.models.conversation import ConversationMessage
from .models import TaskGenerationInput, TaskGenerationOutput, GeneratedTask


@flow(  # type: ignore[arg-type]
    name="task-generation",
    description="Classify user messages and generate enriched task definitions with context",
    is_infrastructure=False
)
class TaskGenerationFlow:
    """Classifies and enriches user messages for optimal processing."""
    
    @pipeline(input_model=TaskGenerationInput, output_model=TaskGenerationOutput)
    async def run_pipeline(self, input_data: TaskGenerationInput) -> TaskGenerationOutput:
        """Generate enriched task definition from user message.
        
        Args:
            input_data: Contains user message and context
            
        Returns:
            TaskGenerationOutput with classified and enriched task definition
        """
        # Get LLM provider using config-driven approach
        llm = cast(LLMProvider, await provider_registry.get_by_config("default-llm"))
        
        # Get prompt from registry
        prompt_instance = resource_registry.get("task-generation-prompt")
        
        # Format conversation history for prompt
        history_text = self._format_conversation_history(input_data.conversation_history)
        
        # Prepare prompt variables
        prompt_vars = {
            "user_message": input_data.user_message,
            "conversation_history": history_text,
            "agent_persona": input_data.agent_persona,
            "working_directory": input_data.working_directory
        }
        
        # Generate task from user message using LLM
        result = await llm.generate_structured(
            prompt=cast(PromptTemplate, prompt_instance),
            output_type=GeneratedTask,
            model_name="default-model",
            prompt_variables=cast(dict[str, object], prompt_vars)
        )
        
        return TaskGenerationOutput(
            generated_task=result,
            success=True,
            processing_time_ms=0.0,  # Will be set by component
            llm_calls_made=1
        )
    
    def _format_conversation_history(self, history: list[ConversationMessage]) -> str:
        """Format conversation history for the prompt."""
        if not history:
            return "No previous conversation"
        
        formatted_messages = []
        for msg in history[-5:]:  # Last 5 messages for context
            formatted_messages.append(f"{msg.role}: {msg.content}")
        
        return "\n".join(formatted_messages)