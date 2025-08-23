"""
Tests for DirectConversationHandler functionality.

These tests cover the conversation handler's ability to process direct
conversations without planning or reflection.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import List, Dict

from flowlib.agent.components.conversation.handler import DirectConversationHandler
from flowlib.agent.components.conversation.flow import ConversationFlow
from flowlib.agent.components.conversation.models import ConversationInput, ConversationOutput
from flowlib.agent.models.state import AgentState
from flowlib.flows.models.results import FlowResult


class TestDirectConversationHandlerInitialization:
    """Test direct conversation handler initialization."""

    def test_handler_creation(self):
        """Test creating conversation handler with conversation flow."""
        mock_flow = MagicMock(spec=ConversationFlow)
        
        handler = DirectConversationHandler(conversation_flow=mock_flow)
        
        assert handler.conversation_flow is mock_flow

    def test_handler_creation_requires_flow(self):
        """Test that conversation handler requires a conversation flow."""
        with pytest.raises(TypeError):
            DirectConversationHandler()


class TestDirectConversationHandlerExecution:
    """Test direct conversation handler execution."""

    @pytest.fixture
    def mock_conversation_flow(self):
        """Create mock conversation flow for testing."""
        flow = AsyncMock()
        # Manually add the process_conversation method since the spec doesn't include it
        flow.process_conversation = AsyncMock()
        return flow

    @pytest.fixture
    def handler(self, mock_conversation_flow):
        """Create conversation handler for testing."""
        return DirectConversationHandler(conversation_flow=mock_conversation_flow)

    @pytest.fixture
    def basic_state(self):
        """Create basic agent state for testing."""
        state = AgentState(
            task_description="Basic conversation",
            task_id="conv_test_123"
        )
        state.add_user_message("Hello")
        state.add_system_message("Hi there!")
        state.add_user_message("How are you?")
        state.add_system_message("I'm doing well, thanks!")
        return state

    @pytest.fixture
    def empty_state(self):
        """Create empty agent state for testing."""
        return AgentState(
            task_description="Empty conversation",
            task_id="empty_test"
        )

    @pytest.mark.asyncio
    async def test_handle_conversation_basic(self, handler, basic_state, mock_conversation_flow):
        """Test basic conversation handling."""
        # Mock conversation output
        expected_output = ConversationOutput(
            response="Hello! How can I help you today?",
            sentiment="positive"
        )
        mock_conversation_flow.process_conversation.return_value = expected_output
        
        result = await handler.handle_conversation(
            message="Hi there!",
            state=basic_state
        )
        
        # Verify flow was called with correct input
        mock_conversation_flow.process_conversation.assert_called_once()
        call_args = mock_conversation_flow.process_conversation.call_args
        input_model = call_args[0][0]
        
        assert isinstance(input_model, ConversationInput)
        assert input_model.message == "Hi there!"
        assert len(input_model.conversation_history) > 0
        assert input_model.memory_context_summary is None
        assert input_model.task_result_summary is None
        
        # Verify result
        assert isinstance(result, FlowResult)
        assert result.status == "SUCCESS"
        assert result.data == expected_output

    @pytest.mark.asyncio
    async def test_handle_conversation_with_context(self, handler, basic_state, mock_conversation_flow):
        """Test conversation handling with memory context and task result."""
        # Mock conversation output
        expected_output = ConversationOutput(
            response="Based on our previous conversation and the task results...",
            sentiment="informative"
        )
        mock_conversation_flow.process_conversation.return_value = expected_output
        
        result = await handler.handle_conversation(
            message="What did we discuss earlier?",
            state=basic_state,
            memory_context_summary="Previous discussion about AI capabilities",
            task_result_summary="Successfully completed data analysis task"
        )
        
        # Verify flow was called with context
        call_args = mock_conversation_flow.process_conversation.call_args
        input_model = call_args[0][0]
        
        assert input_model.message == "What did we discuss earlier?"
        assert input_model.memory_context_summary == "Previous discussion about AI capabilities"
        assert input_model.task_result_summary == "Successfully completed data analysis task"
        
        assert result.status == "SUCCESS"
        assert result.data == expected_output

    @pytest.mark.asyncio
    async def test_handle_conversation_empty_state(self, handler, empty_state, mock_conversation_flow):
        """Test conversation handling with empty state."""
        expected_output = ConversationOutput(
            response="Hello! This appears to be our first interaction.",
            sentiment="welcoming"
        )
        mock_conversation_flow.process_conversation.return_value = expected_output
        
        result = await handler.handle_conversation(
            message="Hello",
            state=empty_state
        )
        
        # Verify flow was called with empty history
        call_args = mock_conversation_flow.process_conversation.call_args
        input_model = call_args[0][0]
        
        assert input_model.message == "Hello"
        assert len(input_model.conversation_history) == 0
        
        assert result.status == "SUCCESS"

    @pytest.mark.asyncio
    async def test_handle_conversation_history_construction(self, handler, mock_conversation_flow):
        """Test proper construction of conversation history."""
        # Create state with unbalanced history (more user messages than system)
        state = AgentState(
            task_description="History test",
            task_id="history_test"
        )
        state.add_user_message("First user message")
        state.add_system_message("First system response")
        state.add_user_message("Second user message")
        state.add_system_message("Second system response")
        state.add_user_message("Third user message")
        
        mock_conversation_flow.process_conversation.return_value = ConversationOutput(
            response="Response based on history"
        )
        
        await handler.handle_conversation("New message", state)
        
        # Verify history construction
        call_args = mock_conversation_flow.process_conversation.call_args
        input_model = call_args[0][0]
        history = input_model.conversation_history
        
        # Should have structured history with proper alternating roles
        expected_entries = min(len(state.user_messages), len(state.system_messages) + 1)
        assert len(history) <= expected_entries * 2  # Max possible entries
        
        # Verify role structure
        for entry in history:
            assert "role" in entry
            assert "content" in entry
            assert entry["role"] in ["user", "assistant"]

    @pytest.mark.asyncio
    async def test_handle_conversation_flow_returns_flow_result(self, handler, basic_state, mock_conversation_flow):
        """Test handling when conversation flow returns FlowResult directly."""
        # Mock flow returning FlowResult instead of ConversationOutput
        expected_result = FlowResult(
            status="SUCCESS",
            data=ConversationOutput(response="Direct FlowResult response"),
            metadata={"message": "Flow executed successfully"}
        )
        mock_conversation_flow.process_conversation.return_value = expected_result
        
        result = await handler.handle_conversation("Test message", basic_state)
        
        # Should pass through the FlowResult unchanged
        assert result is expected_result
        assert result.status == "SUCCESS"
        assert result.metadata["message"] == "Flow executed successfully"

    @pytest.mark.asyncio
    async def test_handle_conversation_unexpected_return_type(self, handler, basic_state, mock_conversation_flow):
        """Test handling when conversation flow returns unexpected type."""
        # Mock flow returning unexpected type
        mock_conversation_flow.process_conversation.return_value = "Unexpected string response"
        
        result = await handler.handle_conversation("Test message", basic_state)
        
        # Should wrap in error FlowResult
        assert result.status == "ERROR"
        assert "unexpected type" in result.error
        assert result.data["result_type"] == "<class 'str'>"

    @pytest.mark.asyncio
    async def test_handle_conversation_with_long_history(self, handler, mock_conversation_flow):
        """Test conversation handling with extensive history."""
        # Create state with long conversation history
        state = AgentState(
            task_description="Long history test",
            task_id="long_history_test"
        )
        for i in range(10):
            state.add_user_message(f"User message {i}")
            if i < 9:  # Only add 9 system messages
                state.add_system_message(f"System response {i}")
        
        mock_conversation_flow.process_conversation.return_value = ConversationOutput(
            response="Response considering long history"
        )
        
        await handler.handle_conversation("New message", state)
        
        # Verify history was constructed without errors
        call_args = mock_conversation_flow.process_conversation.call_args
        input_model = call_args[0][0]
        
        assert input_model.message == "New message"
        assert len(input_model.conversation_history) > 0
        # Verify no index out of bounds errors occurred

    @pytest.mark.asyncio
    async def test_handle_conversation_with_special_characters(self, handler, basic_state, mock_conversation_flow):
        """Test conversation handling with special characters and unicode."""
        mock_conversation_flow.process_conversation.return_value = ConversationOutput(
            response="Response with émojis 🤖 and spëcial chars"
        )
        
        result = await handler.handle_conversation(
            message="Message with émojis 😊 and spëcial characters: áéíóú",
            state=basic_state,
            memory_context_summary="Context with spëcial chars: ñçß",
            task_result_summary="Task with Unicode: 数据分析完成"
        )
        
        # Verify special characters are handled properly
        call_args = mock_conversation_flow.process_conversation.call_args
        input_model = call_args[0][0]
        
        assert "😊" in input_model.message
        assert "spëcial" in input_model.message
        assert "ñçß" in input_model.memory_context_summary
        assert "数据分析完成" in input_model.task_result_summary
        
        assert result.status == "SUCCESS"


class TestDirectConversationHandlerErrorHandling:
    """Test error handling in direct conversation handler."""

    @pytest.fixture
    def handler(self):
        """Create conversation handler for error testing."""
        mock_flow = AsyncMock()
        mock_flow.process_conversation = AsyncMock()
        return DirectConversationHandler(conversation_flow=mock_flow)

    @pytest.fixture
    def mock_state(self):
        """Create mock state for error testing."""
        state = AgentState(
            task_description="Error handling test",
            task_id="error_test"
        )
        state.add_user_message("Error test message")
        state.add_system_message("Error test response")
        return state

    @pytest.mark.asyncio
    async def test_handle_conversation_flow_exception(self, handler, mock_state):
        """Test conversation handling when flow raises exception."""
        handler.conversation_flow.process_conversation.side_effect = Exception("Flow processing error")
        
        with pytest.raises(Exception, match="Flow processing error"):
            await handler.handle_conversation("Test message", mock_state)

    @pytest.mark.asyncio
    async def test_handle_conversation_flow_async_exception(self, handler, mock_state):
        """Test conversation handling when flow raises async exception."""
        async def failing_process(input_model):
            raise ValueError("Async processing error")
        
        handler.conversation_flow.process_conversation = failing_process
        
        with pytest.raises(ValueError, match="Async processing error"):
            await handler.handle_conversation("Test message", mock_state)

    @pytest.mark.asyncio
    async def test_handle_conversation_none_return(self, handler, mock_state):
        """Test conversation handling when flow returns None."""
        handler.conversation_flow.process_conversation.return_value = None
        
        result = await handler.handle_conversation("Test message", mock_state)
        
        # Should handle None as unexpected type
        assert result.status == "ERROR"
        assert "unexpected type" in result.error
        assert result.data["result_type"] == "<class 'NoneType'>"


class TestDirectConversationHandlerIntegration:
    """Integration tests for direct conversation handler."""

    @pytest.mark.asyncio
    async def test_conversation_handler_with_real_models(self):
        """Test conversation handler with real Pydantic models."""
        # Create real conversation flow mock that returns proper models
        mock_flow = AsyncMock()
        
        async def mock_process(input_model: ConversationInput) -> ConversationOutput:
            return ConversationOutput(
                response=f"You said: {input_model.message}",
                sentiment="neutral"
            )
        
        mock_flow.process_conversation = mock_process
        
        handler = DirectConversationHandler(conversation_flow=mock_flow)
        
        state = AgentState(
            task_description="Integration test",
            task_id="integration_test"
        )
        state.add_user_message("Previous message")
        state.add_system_message("Previous response")
        
        result = await handler.handle_conversation(
            message="Hello integration test",
            state=state,
            memory_context_summary="Test context",
            task_result_summary="Test task result"
        )
        
        assert result.status == "SUCCESS"
        assert isinstance(result.data, ConversationOutput)
        assert "Hello integration test" in result.data.response
        assert result.data.sentiment == "neutral"

    @pytest.mark.asyncio
    async def test_conversation_handler_model_validation(self):
        """Test that conversation handler properly validates input models."""
        mock_flow = AsyncMock()
        
        # Mock process that validates the input model structure
        async def mock_process(input_model: ConversationInput) -> ConversationOutput:
            # Verify all expected fields are present and properly typed
            assert isinstance(input_model.message, str)
            assert isinstance(input_model.conversation_history, list)
            assert input_model.memory_context_summary is None or isinstance(input_model.memory_context_summary, str)
            assert input_model.task_result_summary is None or isinstance(input_model.task_result_summary, str)
            
            # Verify default values
            assert input_model.persona == "A helpful AI agent."
            assert input_model.language == "English"
            
            return ConversationOutput(response="Validation passed")
        
        mock_flow.process_conversation = mock_process
        
        handler = DirectConversationHandler(conversation_flow=mock_flow)
        
        state = AgentState(
            task_description="Model validation test",
            task_id="validation_test"
        )
        
        result = await handler.handle_conversation("Validation test", state)
        
        assert result.status == "SUCCESS"
        assert result.data.response == "Validation passed"