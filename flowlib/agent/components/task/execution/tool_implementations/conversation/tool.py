"""Conversation tool implementation."""

from typing import cast
from ...models import ToolResult, ToolExecutionContext, ToolStatus
from ...decorators import tool
from .models import ConversationResult
from flowlib.agent.components.task.models import TodoItem


@tool(name="conversation", tool_category="generic", description="Handle conversational interactions and generate responses")
class ConversationTool:
    """Tool for handling conversational interactions."""
    
    def get_name(self) -> str:
        """Get tool name."""
        return "conversation"
    
    def get_description(self) -> str:
        """Get tool description."""
        return "Handle conversational interactions and generate responses"
    
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
        from .flow import ConversationInput, ConversationFlow

        # Get the conversation flow class
        flow_obj = flow_registry.get("conversation-generation")
        if flow_obj is None:
            raise RuntimeError("Conversation generation flow not found in registry")
        flow_instance = cast(ConversationFlow, flow_obj)
        
        
        # Extract task content from todo
        task_content = todo.content
        
        # Create flow input with agent persona
        flow_input = ConversationInput(
            task_content=task_content,
            working_directory=context.working_directory,
            agent_persona=context.agent_persona
        )
        
        # Execute flow to generate conversation
        result = await flow_instance.run_pipeline(flow_input)
        return cast(ConversationResult, result)