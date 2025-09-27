"""Task decomposition flow implementation.

This module implements the task decomposition system that analyzes user requests
and breaks them down into structured, actionable TODO items with dependencies.
"""

import uuid
from typing import Optional, cast, Any
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.llm.base import LLMProvider, PromptTemplate
from flowlib.resources.registry.registry import resource_registry
from ..models import RequestContext
from .task_models import TaskDecompositionInput, TaskDecompositionOutput


@flow(  # type: ignore[arg-type]
    name="task-decomposition",
    description="Decompose user requests into structured TODO items with dependencies and tool assignments",
    is_infrastructure=False
)
class TaskDecompositionFlow:
    """Decomposes user requests into actionable TODO items with intelligent tool assignment."""
    
    @pipeline(input_model=TaskDecompositionInput, output_model=TaskDecompositionOutput)
    async def run_pipeline(self, input_data: TaskDecompositionInput) -> TaskDecompositionOutput:
        """Decompose task into TODOs with tool assignments.
        
        Args:
            input_data: Contains task description and context
            
        Returns:
            Decomposed TODO items with dependencies and execution strategy
        """
        # Get LLM provider using config-driven approach
        llm = cast(LLMProvider, await provider_registry.get_by_config("default-llm"))
        
        # Get prompt from registry
        prompt_instance = resource_registry.get("task-decomposition-prompt")
        
        # Get available tools filtered by agent role
        agent_role = input_data.context.agent_role if input_data.context else None
        tools_text = await self._get_available_tools_text(agent_role)
        
        # Format context properly
        context_text = self._format_context(input_data.context)
        
        # Format conversation history for the prompt (actual content, not just count)
        execution_history_text = self._format_conversation_history(input_data.context.previous_messages if input_data.context else None)
        
        # Prepare variables
        prompt_vars = {
            "task_description": input_data.task_description,
            "context": context_text,
            "available_tools": tools_text,
            "execution_history_text": execution_history_text,
            "relevant_memories_text": input_data.relevant_memories,
            "thinking_insights": input_data.thinking_insights
        }
        
        # Generate task decomposition using LLM with proper model
        result = await llm.generate_structured(
            prompt=cast(PromptTemplate, prompt_instance),
            output_type=TaskDecompositionOutput,
            model_name="default-model",
            prompt_variables=cast(dict[str, object], prompt_vars)
        )

        # Type validation following flowlib's no-fallbacks principle
        if not isinstance(result, TaskDecompositionOutput):
            raise ValueError(f"Expected TaskDecompositionOutput from LLM, got {type(result)}")

        # Validate and ensure all TODOs have proper IDs and timestamps
        for todo in result.todos:
            if not todo.id:
                todo.id = str(uuid.uuid4())
        
        # DEBUG: Print generated TODOs
        print(f"🔍 DEBUG: Generated {len(result.todos)} TODOs:")
        for i, todo in enumerate(result.todos, 1):
            print(f"  {i}. ID: {todo.id}")
            print(f"     Content: {todo.content}")
            print(f"     Assigned Tool: {todo.assigned_tool}")
            print(f"     Priority: {todo.priority}")
            if todo.depends_on:
                print(f"     Depends On: {todo.depends_on}")
            print()
        
        return result
    
    async def _get_available_tools_text(self, agent_role: Optional[str] = None) -> str:
        """Get available tools filtered by agent role and format for prompt."""
        from flowlib.agent.components.task.execution.tool_role_manager import tool_role_manager
        from flowlib.agent.components.task.execution.registry import tool_registry

        # Get tools allowed for this agent role using role-based filtering
        allowed_tools = tool_role_manager.get_allowed_tools(agent_role)

        # Get metadata for allowed tools
        tools_list = []
        for tool_name in allowed_tools:
            metadata = tool_registry.get_metadata(tool_name)
            if metadata:
                tools_list.append(f"- {tool_name}: {metadata.description}")
            else:
                tools_list.append(f"- {tool_name}: Available tool")

        return "\n".join(tools_list) if tools_list else "No tools available for this agent role"
    
    def _format_context(self, context: Optional[RequestContext]) -> str:
        """Format RequestContext for the prompt."""
        if not context:
            return "No additional context provided"
        
        context_lines = []
        
        if context.agent_persona:
            context_lines.append(f"Agent Persona: {context.agent_persona}")
        
        if context.working_directory and context.working_directory != ".":
            context_lines.append(f"Working Directory: {context.working_directory}")
        
        if context.previous_messages:
            # Use actual conversation content instead of useless count
            conversation_history = self._format_conversation_history(context.previous_messages)
            context_lines.append(f"Previous Conversation:\n{conversation_history}")
        
        if context.agent_name:
            context_lines.append(f"Agent Name: {context.agent_name}")
        
        return "\n".join(context_lines) if context_lines else "No additional context provided"
    
    def _format_conversation_history(self, messages: Optional[list[Any]]) -> str:
        """Format conversation history for the prompt with actual message content."""
        if not messages or len(messages) == 0:
            return "No previous conversation"
        
        formatted_messages = []
        for msg in messages[-5:]:  # Last 5 messages for context
            # Handle both ConversationMessage objects and dict formats
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                formatted_messages.append(f"{msg.role}: {msg.content}")
            elif isinstance(msg, dict):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                formatted_messages.append(f"{role}: {content}")
            else:
                formatted_messages.append(f"message: {str(msg)}")
        
        return "\n".join(formatted_messages)

