"""Tests for conversation models."""

import pytest
from pydantic import ValidationError
from flowlib.agent.components.conversation.models import (
    ConversationInput,
    ConversationOutput,
    ConversationResponse,
    ConversationExecuteInput
)


class TestConversationInput:
    """Test ConversationInput model."""
    
    def test_minimal_input(self):
        """Test creating input with minimal required fields."""
        input_data = ConversationInput(message="Hello, how are you?")
        assert input_data.message == "Hello, how are you?"
        assert input_data.persona == "A helpful AI agent."
        assert input_data.language == "English"
        assert input_data.conversation_history == []
        assert input_data.memory_context_summary is None
        assert input_data.task_result_summary is None
    
    def test_full_input(self):
        """Test creating input with all fields."""
        history = [
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "Hello! How can I help you?"}
        ]
        input_data = ConversationInput(
            message="What's the weather like?",
            persona="A friendly weather assistant.",
            language="Spanish",
            conversation_history=history,
            memory_context_summary="Previous weather inquiries",
            task_result_summary="Weather API call results"
        )
        assert input_data.message == "What's the weather like?"
        assert input_data.persona == "A friendly weather assistant."
        assert input_data.language == "Spanish"
        assert len(input_data.conversation_history) == 2
        assert input_data.memory_context_summary == "Previous weather inquiries"
        assert input_data.task_result_summary == "Weather API call results"
    
    def test_missing_required_field(self):
        """Test that missing message field raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ConversationInput()
        assert "message" in str(exc_info.value)
    
    def test_empty_message(self):
        """Test that empty message is allowed."""
        input_data = ConversationInput(message="")
        assert input_data.message == ""


class TestConversationOutput:
    """Test ConversationOutput model."""
    
    def test_minimal_output(self):
        """Test creating output with minimal required fields."""
        output = ConversationOutput(response="I'm doing well, thank you!")
        assert output.response == "I'm doing well, thank you!"
        assert output.sentiment is None
    
    def test_full_output(self):
        """Test creating output with all fields."""
        output = ConversationOutput(
            response="I'm doing great! How about you?",
            sentiment="positive"
        )
        assert output.response == "I'm doing great! How about you?"
        assert output.sentiment == "positive"
    
    def test_get_user_display(self):
        """Test get_user_display method returns response."""
        output = ConversationOutput(
            response="This is my response",
            sentiment="neutral"
        )
        assert output.get_user_display() == "This is my response"
    
    def test_missing_required_field(self):
        """Test that missing response field raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ConversationOutput()
        assert "response" in str(exc_info.value)


class TestConversationResponse:
    """Test ConversationResponse model."""
    
    def test_create_response(self):
        """Test creating conversation response."""
        response = ConversationResponse(response="Hello there!")
        assert response.response == "Hello there!"
    
    def test_missing_response_field(self):
        """Test that missing response field raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ConversationResponse()
        assert "response" in str(exc_info.value)


class TestConversationExecuteInput:
    """Test ConversationExecuteInput model."""
    
    def test_empty_execute_input(self):
        """Test creating execute input with no fields."""
        exec_input = ConversationExecuteInput()
        assert exec_input.input_data is None
        assert exec_input.message is None
        assert exec_input.inputs is None
        assert exec_input.rationale is None
        assert exec_input.flow_context is None
    
    def test_with_input_data(self):
        """Test creating execute input with ConversationInput."""
        conv_input = ConversationInput(message="Test message")
        exec_input = ConversationExecuteInput(input_data=conv_input)
        assert exec_input.input_data.message == "Test message"
    
    def test_with_message(self):
        """Test creating execute input with simple message."""
        exec_input = ConversationExecuteInput(message="Direct message")
        assert exec_input.message == "Direct message"
    
    def test_with_all_fields(self):
        """Test creating execute input with all fields."""
        conv_input = ConversationInput(message="Test")
        exec_input = ConversationExecuteInput(
            input_data=conv_input,
            message="Alternative message",
            inputs={"key": "value"},
            rationale="Testing all fields",
            flow_context={"context": "data"}
        )
        assert exec_input.input_data.message == "Test"
        assert exec_input.message == "Alternative message"
        assert exec_input.inputs == {"key": "value"}
        assert exec_input.rationale == "Testing all fields"
        assert exec_input.flow_context == {"context": "data"}
    
    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden with strict validation."""
        with pytest.raises(ValidationError) as exc_info:
            ConversationExecuteInput(
                message="Test",
                extra_field="This should not be allowed",
                another_extra=123
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)