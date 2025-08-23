"""Tests for agent classification flow."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List
from pydantic import ValidationError

from flowlib.agent.components.classification.flow import MessageClassifierFlow
from flowlib.agent.components.classification.models import MessageClassification, MessageClassifierInput, ConversationMessage
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.resources.models.constants import ResourceType


class TestMessageClassifierFlow:
    """Test MessageClassifierFlow class."""
    
    def test_message_classifier_flow_decoration(self):
        """Test that MessageClassifierFlow is properly decorated."""
        # Check flow decorator attributes
        assert hasattr(MessageClassifierFlow, '__flow_name__')
        assert hasattr(MessageClassifierFlow, '__flow_description__')
        assert hasattr(MessageClassifierFlow, '__is_infrastructure__')
        
        assert MessageClassifierFlow.__flow_name__ == "message-classifier-flow"
        assert MessageClassifierFlow.__flow_description__ == "Classify user messages into conversation or task"
        assert MessageClassifierFlow.__is_infrastructure__ is True
    
    def test_message_classifier_flow_pipeline_decoration(self):
        """Test that run_pipeline method is properly decorated."""
        # Check that run_pipeline has pipeline decoration
        assert hasattr(MessageClassifierFlow.run_pipeline, '__pipeline_input_model__')
        assert hasattr(MessageClassifierFlow.run_pipeline, '__pipeline_output_model__')
        
        assert MessageClassifierFlow.run_pipeline.__pipeline_input_model__ == MessageClassifierInput
        assert MessageClassifierFlow.run_pipeline.__pipeline_output_model__ == MessageClassification
    
    def test_message_classifier_flow_instantiation(self):
        """Test creating MessageClassifierFlow instance."""
        flow = MessageClassifierFlow()
        
        assert isinstance(flow, MessageClassifierFlow)
        assert hasattr(flow, 'run_pipeline')
        assert hasattr(flow, '_format_conversation_history')
    
    @pytest.mark.asyncio
    async def test_run_pipeline_simple_greeting(self):
        """Test run_pipeline with simple greeting message."""
        flow = MessageClassifierFlow()
        
        input_data = MessageClassifierInput(
            message="Hello! How are you today?"
        )
        
        # Mock the dependencies
        with patch('flowlib.agent.components.classification.flow.resource_registry') as mock_resource_registry, \
             patch('flowlib.agent.components.classification.flow.provider_registry') as mock_provider_registry:
            
            # Mock prompt resource
            mock_prompt = MagicMock()
            mock_resource_registry.get.return_value = mock_prompt
            
            # Mock LLM provider
            mock_llm = AsyncMock()
            mock_classification = MessageClassification(
                execute_task=False,
                confidence=0.95,
                category="greeting"
            )
            mock_llm.generate_structured.return_value = mock_classification
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            
            result = await flow.run_pipeline(input_data)
            
            assert isinstance(result, MessageClassification)
            assert result.execute_task is False
            assert result.confidence == 0.95  # Mocked confidence value
            assert result.category == "greeting"  # Mocked category value
            
            # Verify mocks were called correctly
            mock_resource_registry.get.assert_called_once_with("message_classifier_prompt")
            mock_provider_registry.get_by_config.assert_called_once_with("default-llm")
            mock_llm.generate_structured.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_pipeline_task_request(self):
        """Test run_pipeline with task request message."""
        flow = MessageClassifierFlow()
        
        input_data = MessageClassifierInput(
            message="Can you create a Python script to analyze this data?",
            conversation_history=[
                ConversationMessage(role="user", content="I have some sales data to analyze"),
                ConversationMessage(role="assistant", content="I can help you with data analysis")
            ],
            memory_context_summary="User needs help with Python data analysis"
        )
        
        with patch('flowlib.agent.components.classification.flow.resource_registry') as mock_resource_registry, \
             patch('flowlib.agent.components.classification.flow.provider_registry') as mock_provider_registry:
            
            mock_prompt = MagicMock()
            mock_resource_registry.get.return_value = mock_prompt
            
            mock_llm = AsyncMock()
            mock_classification = MessageClassification(
                execute_task=True,
                confidence=0.92,
                category="code_generation",
                task_description="Create a Python script to analyze sales data"
            )
            mock_llm.generate_structured.return_value = mock_classification
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            
            result = await flow.run_pipeline(input_data)
            
            assert isinstance(result, MessageClassification)
            assert result.execute_task is True
            assert result.confidence == 0.92
            assert result.category == "code_generation"
            assert result.task_description == "Create a Python script to analyze sales data"
    
    @pytest.mark.asyncio
    async def test_run_pipeline_confidence_clamping(self):
        """Test that confidence values are clamped to 0-1 range."""
        flow = MessageClassifierFlow()
        
        input_data = MessageClassifierInput(message="Test message")
        
        with patch('flowlib.agent.components.classification.flow.resource_registry') as mock_resource_registry, \
             patch('flowlib.agent.components.classification.flow.provider_registry') as mock_provider_registry:
            
            mock_resource_registry.get.return_value = MagicMock()
            
            mock_llm = AsyncMock()
            # Mock LLM returning confidence outside 0-1 range
            mock_classification = MessageClassification(
                execute_task=False,
                confidence=1.5,  # Over 1.0
                category="test"
            )
            mock_llm.generate_structured.return_value = mock_classification
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            
            result = await flow.run_pipeline(input_data)
            
            # Confidence should be clamped to 1.0
            assert result.confidence == 1.0
    
    @pytest.mark.asyncio
    async def test_run_pipeline_confidence_clamping_negative(self):
        """Test that negative confidence values are clamped to 0."""
        flow = MessageClassifierFlow()
        
        input_data = MessageClassifierInput(message="Test message")
        
        with patch('flowlib.agent.components.classification.flow.resource_registry') as mock_resource_registry, \
             patch('flowlib.agent.components.classification.flow.provider_registry') as mock_provider_registry:
            
            mock_resource_registry.get.return_value = MagicMock()
            
            mock_llm = AsyncMock()
            mock_classification = MessageClassification(
                execute_task=False,
                confidence=-0.2,  # Negative
                category="test"
            )
            mock_llm.generate_structured.return_value = mock_classification
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            
            result = await flow.run_pipeline(input_data)
            
            # Confidence should be clamped to 0.0
            assert result.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_run_pipeline_task_description_fallback(self):
        """Test that task_description is generated when missing for tasks."""
        flow = MessageClassifierFlow()
        
        input_data = MessageClassifierInput(message="Create a backup of my files")
        
        with patch('flowlib.agent.components.classification.flow.resource_registry') as mock_resource_registry, \
             patch('flowlib.agent.components.classification.flow.provider_registry') as mock_provider_registry:
            
            mock_resource_registry.get.return_value = MagicMock()
            
            mock_llm = AsyncMock()
            # Mock classification with execute_task=True but no task_description
            mock_classification = MessageClassification(
                execute_task=True,
                confidence=0.85,
                category="file_operation",
                task_description=None  # Missing task description
            )
            mock_llm.generate_structured.return_value = mock_classification
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            
            result = await flow.run_pipeline(input_data)
            
            # Should generate fallback task description
            assert result.task_description is not None
            assert "Create a backup of my files" in result.task_description
            assert result.task_description.startswith("Assist the user with their request:")
    
    @pytest.mark.asyncio
    async def test_run_pipeline_error_handling(self):
        """Test error handling in run_pipeline."""
        flow = MessageClassifierFlow()
        
        input_data = MessageClassifierInput(message="Test message")
        
        with patch('flowlib.agent.components.classification.flow.resource_registry') as mock_resource_registry:
            # Mock resource registry to raise an exception
            mock_resource_registry.get.side_effect = Exception("Registry error")
            
            result = await flow.run_pipeline(input_data)
            
            # Should return error fallback classification
            assert isinstance(result, MessageClassification)
            assert result.execute_task is False
            assert result.confidence == 1.0
            assert result.category == "error_fallback"
            assert result.task_description is None
    
    @pytest.mark.asyncio
    async def test_run_pipeline_missing_prompt_resource(self):
        """Test handling when prompt resource is not found."""
        flow = MessageClassifierFlow()
        
        input_data = MessageClassifierInput(message="Test message")
        
        with patch('flowlib.agent.components.classification.flow.resource_registry') as mock_resource_registry:
            # Mock resource registry returning None
            mock_resource_registry.get.return_value = None
            
            result = await flow.run_pipeline(input_data)
            
            # Should return error fallback classification
            assert isinstance(result, MessageClassification)
            assert result.execute_task is False
            assert result.category == "error_fallback"
    
    @pytest.mark.asyncio
    async def test_run_pipeline_llm_provider_error(self):
        """Test handling when LLM provider fails."""
        flow = MessageClassifierFlow()
        
        input_data = MessageClassifierInput(message="Test message")
        
        with patch('flowlib.agent.components.classification.flow.resource_registry') as mock_resource_registry, \
             patch('flowlib.agent.components.classification.flow.provider_registry') as mock_provider_registry:
            
            mock_resource_registry.get.return_value = MagicMock()
            
            # Mock provider registry to raise an exception
            mock_provider_registry.get_by_config = AsyncMock(side_effect=Exception("Provider error"))
            
            result = await flow.run_pipeline(input_data)
            
            # Should return error fallback classification
            assert isinstance(result, MessageClassification)
            assert result.execute_task is False
            assert result.category == "error_fallback"


class TestFormatConversationHistory:
    """Test _format_conversation_history method."""
    
    def test_format_conversation_history_empty(self):
        """Test formatting empty conversation history."""
        flow = MessageClassifierFlow()
        
        result = flow._format_conversation_history([])
        
        assert result == "No conversation history available."
    
    def test_format_conversation_history_single_message(self):
        """Test formatting single message in history."""
        flow = MessageClassifierFlow()
        
        history = [
            ConversationMessage(role="user", content="Hello there!")
        ]
        
        result = flow._format_conversation_history(history)
        
        assert result == "User: Hello there!"
    
    def test_format_conversation_history_multiple_messages(self):
        """Test formatting multiple messages in history."""
        flow = MessageClassifierFlow()
        
        history = [
            ConversationMessage(role="user", content="Can you help me?"),
            ConversationMessage(role="assistant", content="Of course! What do you need help with?"),
            ConversationMessage(role="user", content="I need to analyze some data")
        ]
        
        result = flow._format_conversation_history(history)
        
        expected = ("User: Can you help me?\n"
                   "Assistant: Of course! What do you need help with?\n"
                   "User: I need to analyze some data")
        
        assert result == expected
    
    def test_format_conversation_history_empty_content(self):
        """Test formatting history with empty content."""
        flow = MessageClassifierFlow()
        
        history = [
            ConversationMessage(role="user", content=""),
            ConversationMessage(role="assistant", content="I didn't receive any message")
        ]
        
        result = flow._format_conversation_history(history)
        
        expected = ("User: \n"
                   "Assistant: I didn't receive any message")
        
        assert result == expected
    
    def test_format_conversation_history_role_capitalization(self):
        """Test that roles are properly capitalized."""
        flow = MessageClassifierFlow()
        
        history = [
            ConversationMessage(role="user", content="lowercase user"),
            ConversationMessage(role="ASSISTANT", content="uppercase assistant"),
            ConversationMessage(role="System", content="mixed case system")
        ]
        
        result = flow._format_conversation_history(history)
        
        expected = ("User: lowercase user\n"
                   "Assistant: uppercase assistant\n"
                   "System: mixed case system")
        
        assert result == expected


class TestFlowIntegration:
    """Test integration aspects of the classifier flow."""
    
    @pytest.mark.asyncio
    async def test_prompt_variables_passed_correctly(self):
        """Test that prompt variables are passed correctly to LLM."""
        flow = MessageClassifierFlow()
        
        input_data = MessageClassifierInput(
            message="Test message for analysis",
            conversation_history=[
                {"role": "user", "content": "Previous message"}
            ],
            memory_context_summary="Test memory context"
        )
        
        with patch('flowlib.agent.components.classification.flow.resource_registry') as mock_resource_registry, \
             patch('flowlib.agent.components.classification.flow.provider_registry') as mock_provider_registry:
            
            mock_prompt = MagicMock()
            mock_resource_registry.get.return_value = mock_prompt
            
            mock_llm = AsyncMock()
            mock_classification = MessageClassification(
                execute_task=False,
                confidence=0.8,
                category="test"
            )
            mock_llm.generate_structured.return_value = mock_classification
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            
            await flow.run_pipeline(input_data)
            
            # Check that generate_structured was called with correct parameters
            call_args = mock_llm.generate_structured.call_args
            assert call_args[1]['prompt'] == mock_prompt
            assert call_args[1]['output_type'] == MessageClassification
            
            prompt_vars = call_args[1]['prompt_variables']
            assert prompt_vars['message'] == "Test message for analysis"
            assert prompt_vars['conversation_history'] == "User: Previous message"
            assert prompt_vars['memory_context_summary'] == "Test memory context"
    
    @pytest.mark.asyncio
    async def test_default_memory_context_summary(self):
        """Test default memory context summary when not provided."""
        flow = MessageClassifierFlow()
        
        input_data = MessageClassifierInput(
            message="Test message",
            memory_context_summary=None
        )
        
        with patch('flowlib.agent.components.classification.flow.resource_registry') as mock_resource_registry, \
             patch('flowlib.agent.components.classification.flow.provider_registry') as mock_provider_registry:
            
            mock_resource_registry.get.return_value = MagicMock()
            
            mock_llm = AsyncMock()
            mock_classification = MessageClassification(
                execute_task=False,
                confidence=0.8,
                category="test"
            )
            mock_llm.generate_structured.return_value = mock_classification
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            
            await flow.run_pipeline(input_data)
            
            call_args = mock_llm.generate_structured.call_args
            prompt_vars = call_args[1]['prompt_variables']
            assert prompt_vars['memory_context_summary'] == "No specific memory context provided."
    
    @pytest.mark.asyncio
    async def test_flow_returns_correct_type(self):
        """Test that flow always returns MessageClassification type."""
        flow = MessageClassifierFlow()
        
        input_data = MessageClassifierInput(message="Test")
        
        with patch('flowlib.agent.components.classification.flow.resource_registry') as mock_resource_registry, \
             patch('flowlib.agent.components.classification.flow.provider_registry') as mock_provider_registry:
            
            mock_resource_registry.get.return_value = MagicMock()
            
            mock_llm = AsyncMock()
            mock_classification = MessageClassification(
                execute_task=True,
                confidence=0.7,
                category="test"
            )
            mock_llm.generate_structured.return_value = mock_classification
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            
            result = await flow.run_pipeline(input_data)
            
            assert isinstance(result, MessageClassification)
    
    def test_flow_class_attributes(self):
        """Test that flow class has expected attributes."""
        # Test class-level attributes from decorators
        assert MessageClassifierFlow.__flow_name__ == "message-classifier-flow"
        assert MessageClassifierFlow.__flow_description__ == "Classify user messages into conversation or task"
        assert MessageClassifierFlow.__is_infrastructure__ is True
    
    def test_pipeline_method_attributes(self):
        """Test that pipeline method has expected attributes."""
        # Test method-level attributes from pipeline decorator
        run_pipeline = MessageClassifierFlow.run_pipeline
        assert hasattr(run_pipeline, '__pipeline_input_model__')
        assert hasattr(run_pipeline, '__pipeline_output_model__')
        assert run_pipeline.__pipeline_input_model__ == MessageClassifierInput
        assert run_pipeline.__pipeline_output_model__ == MessageClassification


class TestErrorScenarios:
    """Test various error scenarios in the classification flow."""
    
    @pytest.mark.asyncio
    async def test_malformed_conversation_history(self):
        """Test handling of malformed conversation history."""
        flow = MessageClassifierFlow()
        
        # Malformed conversation history should raise ValidationError
        with pytest.raises(ValidationError):
            input_data = MessageClassifierInput(
                message="Test message",
                conversation_history=[
                    {"invalid": "format"},  # Missing role/content
                    {"role": "user"},       # Missing content
                    {"content": "orphaned"} # Missing role
                ]
            )
    
    @pytest.mark.asyncio
    async def test_extremely_long_message(self):
        """Test handling of extremely long messages."""
        flow = MessageClassifierFlow()
        
        # Create a very long message
        long_message = "This is a test message. " * 1000
        
        input_data = MessageClassifierInput(message=long_message)
        
        with patch('flowlib.agent.components.classification.flow.resource_registry') as mock_resource_registry, \
             patch('flowlib.agent.components.classification.flow.provider_registry') as mock_provider_registry:
            
            mock_resource_registry.get.return_value = MagicMock()
            
            mock_llm = AsyncMock()
            mock_classification = MessageClassification(
                execute_task=False,
                confidence=0.6,
                category="long_message"
            )
            mock_llm.generate_structured.return_value = mock_classification
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            
            result = await flow.run_pipeline(input_data)
            assert isinstance(result, MessageClassification)
    
    @pytest.mark.asyncio
    async def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        flow = MessageClassifierFlow()
        
        input_data = MessageClassifierInput(
            message="Hello! 你好 🚀 Can you help with データ analysis? 日本語",
            conversation_history=[
                {"role": "user", "content": "Ça va bien? 🎉"},
                {"role": "assistant", "content": "Oui! Comment puis-je vous aider? 😊"}
            ]
        )
        
        with patch('flowlib.agent.components.classification.flow.resource_registry') as mock_resource_registry, \
             patch('flowlib.agent.components.classification.flow.provider_registry') as mock_provider_registry:
            
            mock_resource_registry.get.return_value = MagicMock()
            
            mock_llm = AsyncMock()
            mock_classification = MessageClassification(
                execute_task=True,
                confidence=0.8,
                category="multilingual_request"
            )
            mock_llm.generate_structured.return_value = mock_classification
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            
            result = await flow.run_pipeline(input_data)
            assert isinstance(result, MessageClassification)
            
            # Verify that unicode characters are preserved in the prompt variables
            call_args = mock_llm.generate_structured.call_args
            prompt_vars = call_args[1]['prompt_variables']
            assert "你好" in prompt_vars['message']
            assert "🚀" in prompt_vars['message']
            assert "🎉" in prompt_vars['conversation_history']