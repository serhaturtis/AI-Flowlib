"""Conversation tool implementation."""

from typing import Any
from ...models import ToolResult, ToolExecutionContext, ToolStatus
from ...decorators import tool
from .models import ConversationResult


@tool(name="conversation", category="communication", description="Handle conversational interactions and generate responses")
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
        todo: Any,  # TodoItem with task description
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
    
    async def _generate_conversation(self, todo: Any, context: ToolExecutionContext):
        """Generate conversation using flow."""
        from flowlib.flows.registry.registry import flow_registry
        from .flow import ConversationInput
        
        # Get the conversation flow class
        flow_instance = flow_registry.get("conversation-generation")
        
        
        # Extract task content from todo
        task_content = todo.content
        
        # Create flow input with agent persona
        flow_input = ConversationInput(
            task_content=task_content,
            working_directory=context.working_directory,
            agent_persona=context.agent_persona
        )
        
        # Execute flow to generate conversation
        return await flow_instance.run_pipeline(flow_input)