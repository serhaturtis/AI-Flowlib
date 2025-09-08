"""Task decomposition flow implementation.

This module implements the task decomposition system that analyzes user requests
and breaks them down into structured, actionable TODO items with dependencies.
"""

import uuid
from typing import Dict, List, Any
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.providers.core.registry import provider_registry
from flowlib.resources.registry.registry import resource_registry
from .task_models import TaskDecompositionInput, TaskDecompositionOutput
from ..models import TodoItem, TodoPriority, TodoStatus


@flow(
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
        llm = await provider_registry.get_by_config("default-llm")
        
        # Get prompt from registry
        prompt_instance = resource_registry.get("task-decomposition-prompt")
        
        # Get available tools from tool registry
        tools_text = await self._get_available_tools_text()
        
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
            "relevant_memories_text": "No memories retrieved"  # TODO: Add memory retrieval if needed
        }
        
        # Generate task decomposition using LLM with proper model
        result = await llm.generate_structured(
            prompt=prompt_instance,
            output_type=TaskDecompositionOutput,
            model_name="default-model",
            prompt_variables=prompt_vars
        )
        
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
    
    async def _get_available_tools_text(self) -> str:
        """Get available tools and format for prompt."""
        from flowlib.agent.components.task.execution.registry import tool_registry
        
        # Get all registered tools with metadata
        tools_list = []
        for tool_name in tool_registry.list_tools():
            metadata = tool_registry.get_metadata(tool_name)
            if metadata:
                tools_list.append(f"- {tool_name}: {metadata.description}")
            else:
                tools_list.append(f"- {tool_name}: Available tool")
        
        return "\n".join(tools_list) if tools_list else "No tools available"
    
    def _format_context(self, context) -> str:
        """Format RequestContext for the prompt."""
        if not context:
            return "No additional context provided"
        
        context_lines = []
        
        if context.agent_persona:
            context_lines.append(f"Agent Persona: {context.agent_persona}")
        
        if context.working_directory and context.working_directory != ".":
            context_lines.append(f"Working Directory: {context.working_directory}")
        
        if context.previous_messages:
            if len(context.previous_messages) > 0:
                context_lines.append(f"Previous Messages: {len(context.previous_messages)} messages in conversation history")
            else:
                context_lines.append("Previous Messages: No previous conversation")
        
        if context.agent_name:
            context_lines.append(f"Agent Name: {context.agent_name}")
        
        return "\n".join(context_lines) if context_lines else "No additional context provided"
    
    def _format_conversation_history(self, messages) -> str:
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