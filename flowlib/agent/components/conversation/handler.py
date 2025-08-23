"""
Direct conversation handler for the Agent Architecture.

This module provides a handler for direct conversations that bypasses
planning and reflection for simple interactions.
"""

from typing import Dict, Any, Optional, List

from flowlib.flows.models.results import FlowResult
from flowlib.agent.models.state import AgentState
from flowlib.core.context.context import Context
from .flow import ConversationFlow, ConversationInput, ConversationOutput

class DirectConversationHandler:
    """Handles direct conversation without planning or reflection"""
    
    def __init__(self, conversation_flow: ConversationFlow):
        """Initialize the direct conversation handler.
        
        Args:
            conversation_flow: The conversation flow to use for handling messages
        """
        self.conversation_flow = conversation_flow
        
    async def handle_conversation(self, 
                               message: str, 
                               state: AgentState,
                               memory_context_summary: Optional[str] = None,
                               task_result_summary: Optional[str] = None) -> FlowResult:
        """Process a conversation message directly without planning or reflection.
        
        Args:
            message: The user message to respond to
            state: The agent state containing history and other context
            memory_context_summary: Optional summary of relevant memories
            task_result_summary: Optional summary of a task result to present
            
        Returns:
            Flow result containing the response
        """
        # Extract history from state
        history = []
        # Ensure we don't go out of bounds if lists have different lengths
        max_len = min(len(state.user_messages), len(state.system_messages) + 1) 
        for i in range(max_len):
            if i < len(state.user_messages):
                 history.append({"role": "user", "content": state.user_messages[i]})
            if i < len(state.system_messages):
                 history.append({"role": "assistant", "content": state.system_messages[i]})

        # Create conversation input model with all context
        input_model = ConversationInput(
            message=message,
            conversation_history=history, # Pass the structured history
            memory_context_summary=memory_context_summary,
            task_result_summary=task_result_summary
            # language and persona can use defaults or be sourced from state/config if needed
        )
        
        # Execute conversation flow directly, passing the instantiated model
        # The @pipeline decorator should handle this input correctly.
        result = await self.conversation_flow.process_conversation(input_model)
        
        # Wrap the direct result in a FlowResult for consistency
        # (Assuming process_conversation returns ConversationOutput)
        if isinstance(result, ConversationOutput):
             return FlowResult(
                 status="SUCCESS",
                 data=result
             )
        elif isinstance(result, FlowResult):
             # If it somehow already returned FlowResult, pass it through
             return result
        else:
             # Handle unexpected return type
             return FlowResult(
                 status="ERROR",
                 error="Conversation flow returned unexpected type.",
                 data={"result_type": str(type(result))}
             ) 