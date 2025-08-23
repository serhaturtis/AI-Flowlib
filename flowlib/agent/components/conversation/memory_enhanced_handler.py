"""Memory-enhanced conversation handler with automatic storage and retrieval."""

import logging
from typing import Dict, Any, Optional, List

from flowlib.flows.models.results import FlowResult
from flowlib.flows.models.constants import FlowStatus
from ...models.state import AgentState
from ..memory.agent_memory import AgentMemory
from .flow import ConversationFlow, ConversationInput, ConversationOutput
from .handler import DirectConversationHandler

logger = logging.getLogger(__name__)


class MemoryEnhancedConversationHandler(DirectConversationHandler):
    """Conversation handler with automatic memory storage and retrieval."""
    
    def __init__(
        self, 
        conversation_flow: ConversationFlow,
        memory: AgentMemory,
        enhanced_retrieval: bool = True
    ):
        """Initialize memory-enhanced conversation handler.
        
        Args:
            conversation_flow: The conversation flow to use
            memory: The comprehensive memory system
            enhanced_retrieval: Whether to use enhanced memory retrieval
        """
        super().__init__(conversation_flow)
        self.memory = memory
        self.enhanced_retrieval = enhanced_retrieval
        
        logger.info("Memory-enhanced conversation handler initialized")
    
    async def handle_conversation(
        self, 
        message: str, 
        state: AgentState,
        memory_context_summary: Optional[str] = None,
        task_result_summary: Optional[str] = None
    ) -> FlowResult:
        """Process conversation with automatic memory integration."""
        
        # Enhanced memory retrieval if enabled
        if self.enhanced_retrieval:
            enhanced_memory_context = await self._get_enhanced_memory_context(
                message, state, memory_context_summary
            )
        else:
            enhanced_memory_context = memory_context_summary
        
        # Extract conversation history
        history = self._extract_conversation_history(state)
        
        # Create enhanced conversation input
        input_model = ConversationInput(
            message=message,
            conversation_history=history,
            memory_context_summary=enhanced_memory_context,
            task_result_summary=task_result_summary
        )
        
        # Execute conversation flow
        result = await self.conversation_flow.process_conversation(input_model)
        
        # Process result for memory storage
        flow_result = self._wrap_result(result)
        
        # Automatic memory storage (background processing)
        if flow_result.status == "SUCCESS":
            await self._trigger_automatic_memory_storage(
                message, 
                flow_result.data.response if hasattr(flow_result.data, 'response') else str(flow_result.data),
                state
            )
        
        return flow_result
    
    async def _get_enhanced_memory_context(
        self,
        message: str,
        state: AgentState,
        existing_context: Optional[str] = None
    ) -> str:
        """Get enhanced memory context using automatic memory manager."""
        
        try:
            # Import here to avoid circular imports
            from ..memory.models import MemorySearchRequest
            
            # Use comprehensive memory for direct search
            search_request = MemorySearchRequest(
                query=message,
                context=state.task_id or "default",
                limit=6,
                search_type="hybrid"
            )
            search_results = await self.memory.search(search_request)
            
            # Process search results into categorized format
            relevant_memories = {
                "facts": [],
                "preferences": [], 
                "conversations": [],
                "summary": "Relevant information found in memory."
            }
            
            for result in search_results:
                if hasattr(result, 'item') and result.item:
                    # Categorize based on item type or metadata
                    item = result.item
                    if item.item_type == "conversation":
                        relevant_memories["conversations"].append(item)
                    elif item.item_type == "fact":
                        relevant_memories["facts"].append(item)
                    elif item.item_type == "preference":
                        relevant_memories["preferences"].append(item)
            
            # Format enhanced context
            context_parts = []
            
            if existing_context and existing_context != "No relevant memories found.":
                context_parts.append(f"Basic Context:\n{existing_context}\n")
            
            # Add categorized memories
            if "facts" in relevant_memories and relevant_memories["facts"]:
                facts = [item.value for item in relevant_memories["facts"][:3]]
                context_parts.append(f"Relevant Facts:\n" + "\n".join(f"- {fact}" for fact in facts))
            
            if "preferences" in relevant_memories and relevant_memories["preferences"]:
                prefs = [item.value for item in relevant_memories["preferences"][:2]]
                context_parts.append(f"User Preferences:\n" + "\n".join(f"- {pref}" for pref in prefs))
            
            if "conversations" in relevant_memories and relevant_memories["conversations"]:
                conv_summaries = []
                for conv in relevant_memories["conversations"][:2]:
                    if isinstance(conv.value, dict):
                        user_msg = conv.value["user_message"] if "user_message" in conv.value else ""
                        assistant_msg = conv.value["assistant_response"] if "assistant_response" in conv.value else ""
                        conv_summaries.append(f"Previous: User said '{user_msg[:100]}...' and I responded '{assistant_msg[:100]}...'")
                if conv_summaries:
                    context_parts.append(f"Recent Relevant Conversations:\n" + "\n".join(conv_summaries))
            
            if "summary" in relevant_memories and relevant_memories["summary"]:
                context_parts.append(f"Memory Summary: {relevant_memories['summary']}")
            
            if context_parts:
                enhanced_context = "\n\n".join(context_parts)
                logger.debug(f"Enhanced memory context with {len(context_parts)} sections")
                return enhanced_context
            else:
                return existing_context or "No relevant memories found."
                
        except Exception as e:
            logger.error(f"Error getting enhanced memory context: {e}")
            return existing_context or "Error retrieving enhanced memory context."
    
    def _extract_conversation_history(self, state: AgentState) -> List[Dict[str, str]]:
        """Extract structured conversation history from agent state."""
        history = []
        
        # Ensure we don't go out of bounds
        max_len = min(len(state.user_messages), len(state.system_messages) + 1)
        
        for i in range(max_len):
            if i < len(state.user_messages):
                history.append({"role": "user", "content": state.user_messages[i]})
            if i < len(state.system_messages):
                history.append({"role": "assistant", "content": state.system_messages[i]})
        
        return history
    
    def _wrap_result(self, result) -> FlowResult:
        """Wrap conversation flow result in FlowResult."""
        if isinstance(result, ConversationOutput):
            return FlowResult(
                status=FlowStatus.SUCCESS,
                data=result
            )
        elif isinstance(result, FlowResult):
            return result
        else:
            return FlowResult(
                status=FlowStatus.ERROR,
                error="Conversation flow returned unexpected type.",
                data={"result_type": str(type(result))}
            )
    
    async def _trigger_automatic_memory_storage(
        self,
        user_message: str,
        assistant_response: str,
        state: AgentState
    ):
        """Trigger automatic memory storage for the conversation."""
        
        try:
            # Import here to avoid circular imports
            from ..memory.models import MemoryStoreRequest
            from datetime import datetime
            
            # Create memory store request for the conversation
            conversation_data = {
                "user_message": user_message,
                "assistant_response": assistant_response,
                "timestamp": datetime.now().isoformat(),
                "session_id": getattr(state, 'session_id', None)
            }
            
            store_request = MemoryStoreRequest(
                key=f"conversation_{datetime.now().timestamp()}",
                value=f"User: {user_message}\nAssistant: {assistant_response}",
                context=state.task_id or "default",
                metadata={
                    "conversation_data": conversation_data,
                    "item_type": "conversation"
                },
                importance=0.5
            )
            
            # Store to comprehensive memory
            item_id = await self.memory.store(store_request)
            logger.debug(f"Stored conversation to memory with ID: {item_id}")
            
        except Exception as e:
            logger.error(f"Error storing conversation to memory: {e}")
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return self.memory.get_stats()


class MemoryAwareFlow:
    """Mixin for flows that want memory integration."""
    
    def __init__(self, memory: Optional[AgentMemory] = None):
        self.memory = memory
    
    async def get_relevant_memories(
        self, 
        query: str, 
        context: str,
        limit: int = 3
    ) -> Dict[str, Any]:
        """Get relevant memories for this flow execution."""
        
        if not self.memory:
            return {"memories": [], "summary": "No memory available"}
        
        try:
            # Import here to avoid circular imports
            from ..memory.models import MemorySearchRequest
            
            search_request = MemorySearchRequest(
                query=query,
                context=context,
                limit=limit,
                search_type="hybrid"
            )
            search_results = await self.memory.search(search_request)
            
            # Convert search results to memory format
            memories = []
            for result in search_results:
                if hasattr(result, 'item') and result.item:
                    memories.append(result.item)
            
            return {
                "memories": memories,
                "summary": f"Found {len(memories)} relevant memories",
                "categories": {
                    "total": len(memories)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting relevant memories for flow: {e}")
            return {"memories": [], "summary": f"Error: {e}", "categories": {}}
    
    async def store_flow_result(
        self,
        flow_name: str,
        inputs: Any,
        outputs: Any,
        context: str,
        importance: float = 0.5
    ):
        """Store flow execution result in memory."""
        
        if not self.memory:
            return
        
        try:
            # Import here to avoid circular imports
            from ..memory.models import MemoryStoreRequest
            from datetime import datetime
            
            # Store flow execution result
            store_request = MemoryStoreRequest(
                key=f"flow_{flow_name}_{datetime.now().timestamp()}",
                value=f"Flow {flow_name} executed with inputs: {str(inputs)[:200]} -> Result: {str(outputs)[:200]}",
                context=context,
                metadata={
                    "flow_name": flow_name,
                    "inputs": str(inputs)[:500],
                    "outputs": str(outputs)[:500],
                    "timestamp": datetime.now().isoformat(),
                    "item_type": "flow_execution"
                },
                importance=importance
            )
            
            await self.memory.store(store_request)
            
        except Exception as e:
            logger.error(f"Error storing flow result in memory: {e}")


# Factory function for creating memory-enhanced handlers
async def create_memory_enhanced_handler(
    conversation_flow: ConversationFlow,
    memory: AgentMemory,
    llm_provider=None,  # No longer needed, kept for compatibility
    auto_storage_threshold: float = 0.4  # No longer used, kept for compatibility
) -> MemoryEnhancedConversationHandler:
    """Factory function to create a memory-enhanced conversation handler."""
    
    # Create enhanced handler with direct memory integration
    handler = MemoryEnhancedConversationHandler(
        conversation_flow=conversation_flow,
        memory=memory,
        enhanced_retrieval=True
    )
    
    logger.info("Created memory-enhanced conversation handler with direct memory integration")
    return handler