"""Tests for memory-enhanced conversation handler."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, Optional, List

from flowlib.agent.components.conversation.memory_enhanced_handler import (
    MemoryEnhancedConversationHandler,
    MemoryAwareFlow,
    create_memory_enhanced_handler
)
from flowlib.agent.components.conversation.flow import ConversationFlow, ConversationInput, ConversationOutput
from flowlib.agent.components.conversation.handler import DirectConversationHandler
from flowlib.agent.components.memory.agent_memory import AgentMemory
from flowlib.agent.models.state import AgentState
from flowlib.flows.models.results import FlowResult
from flowlib.flows.models.constants import FlowStatus


# Mock classes for testing
class MockConversationFlow:
    """Mock conversation flow for testing."""
    
    def __init__(self):
        self.process_calls = []
    
    async def process_conversation(self, input_data: ConversationInput) -> ConversationOutput:
        """Mock process conversation."""
        self.process_calls.append(input_data)
        return ConversationOutput(
            response=f"Response to: {input_data.message}",
            sentiment="positive"
        )


class MockAgentMemory:
    """Mock comprehensive memory for testing."""
    
    def __init__(self):
        self.stored_memories = []
    
    async def store_memory(self, memory_item):
        """Mock store memory."""
        self.stored_memories.append(memory_item)
    
    async def search(self, search_request):
        """Mock search method."""
        # Return mock search results
        class MockSearchResult:
            def __init__(self, item_type, value):
                self.item = MockMemoryItem(item_type, value)
        
        class MockMemoryItem:
            def __init__(self, item_type, value):
                self.item_type = item_type
                self.value = value
        
        return [
            MockSearchResult("fact", "User prefers Python programming"),
            MockSearchResult("preference", "Prefers detailed explanations"),
            MockSearchResult("conversation", {"user_message": "How do I learn Python?", "assistant_response": "Start with basics"})
        ]
    
    async def store(self, store_request):
        """Mock store method.""" 
        return "mock_item_id"
    
    def get_stats(self):
        """Mock get stats method."""
        return {"total_memories": 100, "processing_queue": 5}


class MockAutomaticMemoryManager:
    """Mock automatic memory manager for testing."""
    
    def __init__(self, **kwargs):
        self.retrieve_calls = []
        self.process_calls = []
    
    async def retrieve_relevant_for_context(self, query, context, limit=10, **kwargs):
        """Mock retrieve relevant for context."""
        self.retrieve_calls.append({
            "query": query,
            "context": context,
            "limit": limit,
            **kwargs
        })
        return {
            "memories": [
                {"type": "fact", "content": "User prefers Python programming"},
                {"type": "conversation", "content": "How do I learn Python?"}
            ],
            "summary": "User is a software developer learning Python and ML",
            "categories": {
                "facts": 3,
                "preferences": 2,
                "conversations": 2
            }
        }
    
    async def process_conversation_turn(self, user_message, assistant_response, context, session_id, **kwargs):
        """Mock process conversation turn."""
        self.process_calls.append({
            "user_message": user_message,
            "assistant_response": assistant_response,
            "context": context,
            "session_id": session_id,
            **kwargs
        })
    
    async def start_background_processing(self):
        """Mock start background processing."""
        pass
    
    async def get_statistics(self):
        """Mock get statistics."""
        return {"total_memories": 100, "processing_queue": 5}




class MockAgentState:
    """Mock agent state for testing."""
    
    def __init__(
        self,
        task_id: str = "test_task",
        session_id: str = "test_session",
        user_messages: List[str] = None,
        system_messages: List[str] = None
    ):
        self.task_id = task_id
        self.session_id = session_id
        self.user_messages = user_messages if user_messages is not None else ["Hello", "How are you?"]
        self.system_messages = system_messages if system_messages is not None else ["Hi there!", "I'm doing well, thanks!"]


class TestMemoryEnhancedConversationHandler:
    """Test MemoryEnhancedConversationHandler class."""
    
    def test_memory_enhanced_handler_initialization(self):
        """Test handler initialization."""
        conversation_flow = MockConversationFlow()
        memory = MockAgentMemory()
        
        handler = MemoryEnhancedConversationHandler(
            conversation_flow=conversation_flow,
            memory=memory,
            enhanced_retrieval=True
        )
        
        assert handler.conversation_flow == conversation_flow
        assert handler.memory == memory
        assert handler.enhanced_retrieval is True
        
        # Should inherit from DirectConversationHandler
        assert isinstance(handler, DirectConversationHandler)
    
    def test_memory_enhanced_handler_initialization_default_enhanced_retrieval(self):
        """Test handler initialization with default enhanced_retrieval."""
        conversation_flow = MockConversationFlow()
        memory = MockAgentMemory()
        
        handler = MemoryEnhancedConversationHandler(
            conversation_flow=conversation_flow,
            memory=memory,
            enhanced_retrieval=True
        )
        
        assert handler.enhanced_retrieval is True  # Default value
    
    @pytest.mark.asyncio
    async def test_handle_conversation_with_enhanced_retrieval(self):
        """Test conversation handling with enhanced memory retrieval."""
        conversation_flow = MockConversationFlow()
        memory = MockAgentMemory()
        
        handler = MemoryEnhancedConversationHandler(
            conversation_flow=conversation_flow,
            memory=memory,
            enhanced_retrieval=True
        )
        
        state = MockAgentState()
        
        result = await handler.handle_conversation(
            message="Tell me about Python",
            state=state,
            memory_context_summary="User asked about programming",
            task_result_summary="Previous task completed"
        )
        
        # Verify result
        assert isinstance(result, FlowResult)
        assert result.status == FlowStatus.SUCCESS
        assert isinstance(result.data, ConversationOutput)
        assert "Response to: Tell me about Python" in result.data.response
        
        # Verify conversation flow was called
        assert len(conversation_flow.process_calls) == 1
        process_call = conversation_flow.process_calls[0]
        assert process_call.message == "Tell me about Python"
        assert process_call.task_result_summary == "Previous task completed"
        
        # Memory storage is now handled by the direct memory integration
    
    @pytest.mark.asyncio
    async def test_handle_conversation_without_enhanced_retrieval(self):
        """Test conversation handling without enhanced memory retrieval."""
        conversation_flow = MockConversationFlow()
        memory = MockAgentMemory()
        
        handler = MemoryEnhancedConversationHandler(
            conversation_flow=conversation_flow,
            memory=memory,
            enhanced_retrieval=False
        )
        
        state = MockAgentState()
        
        result = await handler.handle_conversation(
            message="Hello there",
            state=state,
            memory_context_summary="Basic context"
        )
        
        # Verify conversation flow was called with basic context
        assert len(conversation_flow.process_calls) == 1
        process_call = conversation_flow.process_calls[0]
        assert process_call.memory_context_summary == "Basic context"
    
    @pytest.mark.asyncio
    async def test_get_enhanced_memory_context_success(self):
        """Test enhanced memory context retrieval."""
        conversation_flow = MockConversationFlow()
        memory = MockAgentMemory()
        auto_memory_manager = MockAutomaticMemoryManager()
        
        handler = MemoryEnhancedConversationHandler(
            conversation_flow=conversation_flow,
            memory=memory,
            enhanced_retrieval=True
        )
        
        state = MockAgentState()
        
        enhanced_context = await handler._get_enhanced_memory_context(
            message="What's the best way to learn ML?",
            state=state,
            existing_context="User is interested in AI"
        )
        
        # Verify enhanced context contains all sections
        assert "Basic Context:" in enhanced_context
        assert "User is interested in AI" in enhanced_context
        assert "Relevant Facts:" in enhanced_context
        assert "User prefers Python programming" in enhanced_context
        assert "User Preferences:" in enhanced_context
        assert "Prefers detailed explanations" in enhanced_context
        assert "Recent Relevant Conversations:" in enhanced_context
        assert "How do I learn Python?" in enhanced_context
        assert "Memory Summary:" in enhanced_context
        assert "Relevant information found in memory." in enhanced_context
    
    @pytest.mark.asyncio
    async def test_get_enhanced_memory_context_no_existing_context(self):
        """Test enhanced memory context without existing context."""
        conversation_flow = MockConversationFlow()
        memory = MockAgentMemory()
        auto_memory_manager = MockAutomaticMemoryManager()
        
        handler = MemoryEnhancedConversationHandler(
            conversation_flow=conversation_flow,
            memory=memory,
            enhanced_retrieval=True
        )
        
        state = MockAgentState()
        
        enhanced_context = await handler._get_enhanced_memory_context(
            message="Test query",
            state=state,
            existing_context=None
        )
        
        # Should not include Basic Context section
        assert "Basic Context:" not in enhanced_context
        assert "Relevant Facts:" in enhanced_context
        assert "User Preferences:" in enhanced_context
    
    @pytest.mark.asyncio
    async def test_get_enhanced_memory_context_error_handling(self):
        """Test enhanced memory context error handling."""
        conversation_flow = MockConversationFlow()
        memory = MockAgentMemory()
        auto_memory_manager = MockAutomaticMemoryManager()
        
        # Make memory.search fail
        memory.search = AsyncMock(side_effect=Exception("Memory retrieval failed"))
        
        handler = MemoryEnhancedConversationHandler(
            conversation_flow=conversation_flow,
            memory=memory,
            enhanced_retrieval=True
        )
        
        state = MockAgentState()
        
        enhanced_context = await handler._get_enhanced_memory_context(
            message="Test query",
            state=state,
            existing_context="Fallback context"
        )
        
        # Should return fallback context
        assert enhanced_context == "Fallback context"
    
    def test_extract_conversation_history(self):
        """Test conversation history extraction."""
        conversation_flow = MockConversationFlow()
        memory = MockAgentMemory()
        auto_memory_manager = MockAutomaticMemoryManager()
        
        handler = MemoryEnhancedConversationHandler(
            conversation_flow=conversation_flow,
            memory=memory,
            enhanced_retrieval=True
        )
        
        state = MockAgentState(
            user_messages=["Hello", "How are you?", "Tell me about AI"],
            system_messages=["Hi there!", "I'm doing well, thanks!"]
        )
        
        history = handler._extract_conversation_history(state)
        
        # Should extract conversation in order
        expected_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thanks!"},
            {"role": "user", "content": "Tell me about AI"}
        ]
        
        assert history == expected_history
    
    def test_extract_conversation_history_empty(self):
        """Test conversation history extraction with empty state."""
        conversation_flow = MockConversationFlow()
        memory = MockAgentMemory()
        auto_memory_manager = MockAutomaticMemoryManager()
        
        handler = MemoryEnhancedConversationHandler(
            conversation_flow=conversation_flow,
            memory=memory,
            enhanced_retrieval=True
        )
        
        state = MockAgentState(user_messages=[], system_messages=[])
        
        history = handler._extract_conversation_history(state)
        
        assert history == []
    
    def test_wrap_result_conversation_output(self):
        """Test wrapping ConversationOutput in FlowResult."""
        conversation_flow = MockConversationFlow()
        memory = MockAgentMemory()
        auto_memory_manager = MockAutomaticMemoryManager()
        
        handler = MemoryEnhancedConversationHandler(
            conversation_flow=conversation_flow,
            memory=memory,
            enhanced_retrieval=True
        )
        
        conversation_output = ConversationOutput(
            response="Test response",
            sentiment="positive"
        )
        
        flow_result = handler._wrap_result(conversation_output)
        
        assert isinstance(flow_result, FlowResult)
        assert flow_result.status == FlowStatus.SUCCESS
        assert flow_result.data == conversation_output
    
    def test_wrap_result_flow_result(self):
        """Test wrapping existing FlowResult."""
        conversation_flow = MockConversationFlow()
        memory = MockAgentMemory()
        auto_memory_manager = MockAutomaticMemoryManager()
        
        handler = MemoryEnhancedConversationHandler(
            conversation_flow=conversation_flow,
            memory=memory,
            enhanced_retrieval=True
        )
        
        existing_result = FlowResult(status=FlowStatus.ERROR, error="Test error")
        
        flow_result = handler._wrap_result(existing_result)
        
        assert flow_result is existing_result
    
    def test_wrap_result_unexpected_type(self):
        """Test wrapping unexpected result type."""
        conversation_flow = MockConversationFlow()
        memory = MockAgentMemory()
        auto_memory_manager = MockAutomaticMemoryManager()
        
        handler = MemoryEnhancedConversationHandler(
            conversation_flow=conversation_flow,
            memory=memory,
            enhanced_retrieval=True
        )
        
        unexpected_result = "String result"
        
        flow_result = handler._wrap_result(unexpected_result)
        
        assert isinstance(flow_result, FlowResult)
        assert flow_result.status == FlowStatus.ERROR
        assert "unexpected type" in flow_result.error
        assert flow_result.data["result_type"] == "<class 'str'>"
    
    @pytest.mark.asyncio
    async def test_trigger_automatic_memory_storage_success(self):
        """Test successful automatic memory storage."""
        conversation_flow = MockConversationFlow()
        memory = MockAgentMemory()
        memory.store = AsyncMock(return_value="stored_id")
        
        handler = MemoryEnhancedConversationHandler(
            conversation_flow=conversation_flow,
            memory=memory,
            enhanced_retrieval=True
        )
        
        state = MockAgentState()
        
        await handler._trigger_automatic_memory_storage(
            user_message="What is Python?",
            assistant_response="Python is a programming language...",
            state=state
        )
        
        # Verify memory storage was triggered
        memory.store.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_trigger_automatic_memory_storage_error(self):
        """Test automatic memory storage error handling."""
        conversation_flow = MockConversationFlow()
        memory = MockAgentMemory()
        auto_memory_manager = MockAutomaticMemoryManager()
        
        # Make process_conversation_turn fail
        auto_memory_manager.process_conversation_turn = AsyncMock(
            side_effect=Exception("Storage failed")
        )
        
        handler = MemoryEnhancedConversationHandler(
            conversation_flow=conversation_flow,
            memory=memory,
            enhanced_retrieval=True
        )
        
        state = MockAgentState()
        
        # Should not raise exception
        await handler._trigger_automatic_memory_storage(
            user_message="Test",
            assistant_response="Response",
            state=state
        )
    
    @pytest.mark.asyncio
    async def test_get_memory_statistics(self):
        """Test getting memory statistics."""
        conversation_flow = MockConversationFlow()
        memory = MockAgentMemory()
        auto_memory_manager = MockAutomaticMemoryManager()
        
        handler = MemoryEnhancedConversationHandler(
            conversation_flow=conversation_flow,
            memory=memory,
            enhanced_retrieval=True
        )
        
        stats = await handler.get_memory_statistics()
        
        assert stats == {"total_memories": 100, "processing_queue": 5}


class TestMemoryAwareFlow:
    """Test MemoryAwareFlow mixin."""
    
    def test_memory_aware_flow_initialization(self):
        """Test MemoryAwareFlow initialization."""
        memory = MockAgentMemory()
        
        flow = MemoryAwareFlow(memory=memory)
        
        assert flow.memory == memory
    
    def test_memory_aware_flow_initialization_no_manager(self):
        """Test MemoryAwareFlow initialization without memory manager."""
        flow = MemoryAwareFlow()
        
        assert flow.memory is None
    
    @pytest.mark.asyncio
    async def test_get_relevant_memories_success(self):
        """Test successful memory retrieval."""
        memory = MockAgentMemory()
        flow = MemoryAwareFlow(memory=memory)
        
        # Mock the search method to return mock results
        mock_results = []
        memory.search = AsyncMock(return_value=mock_results)
        
        memories = await flow.get_relevant_memories(
            query="Python programming",
            context="learning",
            limit=5
        )
        
        # Verify returned structure
        assert "memories" in memories
        assert "summary" in memories
        assert "categories" in memories
    
    @pytest.mark.asyncio
    async def test_get_relevant_memories_no_manager(self):
        """Test memory retrieval without memory manager."""
        flow = MemoryAwareFlow()
        
        memories = await flow.get_relevant_memories(
            query="Test query",
            context="test"
        )
        
        assert memories == {
            "memories": [],
            "summary": "No memory available"
        }
    
    @pytest.mark.asyncio
    async def test_get_relevant_memories_error(self):
        """Test memory retrieval error handling."""
        memory = MockAgentMemory()
        memory.search = AsyncMock(side_effect=Exception("Retrieval failed"))
        
        flow = MemoryAwareFlow(memory=memory)
        
        memories = await flow.get_relevant_memories(
            query="Test",
            context="test"
        )
        
        assert memories["memories"] == []
        assert "Error: Retrieval failed" in memories["summary"]
        assert memories["categories"] == {}
    
    @pytest.mark.asyncio
    async def test_store_flow_result_success(self):
        """Test successful flow result storage."""
        memory = MockAgentMemory()
        memory.store = AsyncMock(return_value="stored_id")
        
        flow = MemoryAwareFlow(memory=memory)
        
        await flow.store_flow_result(
            flow_name="test_flow",
            inputs={"param": "value"},
            outputs={"result": "success"},
            context="test_context",
            importance=0.8
        )
        
        # Verify memory.store was called
        memory.store.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_flow_result_no_manager(self):
        """Test flow result storage without memory manager."""
        flow = MemoryAwareFlow()
        
        # Should not raise exception
        await flow.store_flow_result(
            flow_name="test_flow",
            inputs={},
            outputs={},
            context="test"
        )
    
    @pytest.mark.asyncio
    async def test_store_flow_result_error(self):
        """Test flow result storage error handling."""
        memory = MockAgentMemory()
        memory.store = AsyncMock(side_effect=Exception("Storage failed"))
        
        flow = MemoryAwareFlow(memory=memory)
        
        # Should not raise exception
        await flow.store_flow_result(
            flow_name="test_flow",
            inputs={},
            outputs={},
            context="test"
        )


class TestCreateMemoryEnhancedHandler:
    """Test factory function for creating memory-enhanced handlers."""
    
    @pytest.mark.asyncio
    async def test_create_memory_enhanced_handler_success(self):
        """Test successful creation of memory-enhanced handler."""
        conversation_flow = MockConversationFlow()
        memory = MockAgentMemory()
        llm_provider = Mock()
        
        handler = await create_memory_enhanced_handler(
            conversation_flow=conversation_flow,
            memory=memory,
            llm_provider=llm_provider,
            auto_storage_threshold=0.5
        )
        
        # Verify handler was created correctly
        assert isinstance(handler, MemoryEnhancedConversationHandler)
        assert handler.conversation_flow == conversation_flow
        assert handler.memory == memory
        assert handler.enhanced_retrieval is True
    
    @pytest.mark.asyncio
    async def test_create_memory_enhanced_handler_default_threshold(self):
        """Test creation with default auto storage threshold."""
        conversation_flow = MockConversationFlow()
        memory = MockAgentMemory()
        llm_provider = Mock()
        
        handler = await create_memory_enhanced_handler(
            conversation_flow=conversation_flow,
            memory=memory,
            llm_provider=llm_provider
        )
        
        # Verify handler was created
        assert isinstance(handler, MemoryEnhancedConversationHandler)
        assert handler.enhanced_retrieval is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])