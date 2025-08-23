"""Tests for conversation flow functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Optional

from flowlib.agent.components.conversation.flow import ConversationFlow
from flowlib.agent.components.conversation.models import ConversationInput, ConversationOutput
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.resources.models.constants import ResourceType


class TestConversationFlow:
    """Test ConversationFlow class."""
    
    def test_conversation_flow_decoration(self):
        """Test that ConversationFlow is properly decorated."""
        # Check flow decorator attributes
        assert hasattr(ConversationFlow, '__flow_name__')
        assert hasattr(ConversationFlow, '__flow_description__')
        assert hasattr(ConversationFlow, '__is_infrastructure__')
        
        assert ConversationFlow.__flow_name__ == "conversation"
        assert "Handle simple conversational interactions" in ConversationFlow.__flow_description__
        assert ConversationFlow.__is_infrastructure__ is False
    
    def test_conversation_flow_pipeline_decoration(self):
        """Test that run_pipeline method is properly decorated."""
        # Check that run_pipeline has pipeline decoration
        assert hasattr(ConversationFlow.run_pipeline, '__pipeline_input_model__')
        assert hasattr(ConversationFlow.run_pipeline, '__pipeline_output_model__')
        
        assert ConversationFlow.run_pipeline.__pipeline_input_model__ == ConversationInput
        assert ConversationFlow.run_pipeline.__pipeline_output_model__ == ConversationOutput
    
    def test_conversation_flow_instantiation(self):
        """Test creating ConversationFlow instance."""
        flow = ConversationFlow()
        
        assert isinstance(flow, ConversationFlow)
        assert hasattr(flow, 'run_pipeline')


class TestConversationFlowExecution:
    """Test ConversationFlow execution scenarios."""
    
    @pytest.mark.asyncio
    async def test_run_pipeline_successful_conversation(self):
        """Test successful conversation flow execution."""
        flow = ConversationFlow()
        
        input_data = ConversationInput(
            message="Hello, how are you?",
            persona="friendly assistant",
            memory_context_summary="User is greeting me for the first time"
        )
        
        expected_output = ConversationOutput(
            response="Hello! I'm doing well, thank you for asking. How can I help you today?",
            sentiment="positive"
        )
        
        with patch('flowlib.agent.components.conversation.flow.provider_registry') as mock_provider_registry, \
             patch('flowlib.agent.components.conversation.flow.resource_registry') as mock_resource_registry:
            
            # Mock LLM provider
            mock_llm = AsyncMock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            mock_llm.generate_structured.return_value = expected_output
            
            # Mock prompt resource
            mock_prompt = Mock()
            mock_resource_registry.get.return_value = mock_prompt
            
            result = await flow.run_pipeline(input_data)
            
            # Verify result
            assert isinstance(result, ConversationOutput)
            assert result.response == expected_output.response
            assert result.sentiment == expected_output.sentiment
            
            # Verify calls
            mock_provider_registry.get_by_config.assert_called_once_with("default-llm")
            mock_resource_registry.get.assert_called_once_with(
                "conversation-prompt"
            )
            mock_llm.generate_structured.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_pipeline_with_no_memory_context(self):
        """Test conversation flow with no memory context."""
        flow = ConversationFlow()
        
        input_data = ConversationInput(
            message="What's the weather like?",
            persona="helpful bot",
            memory_context_summary=None  # No memory context
        )
        
        expected_output = ConversationOutput(
            response="I'm sorry, I don't have access to current weather information.",
            sentiment="neutral"
        )
        
        with patch('flowlib.agent.components.conversation.flow.provider_registry') as mock_provider_registry, \
             patch('flowlib.agent.components.conversation.flow.resource_registry') as mock_resource_registry:
            
            mock_llm = AsyncMock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            mock_llm.generate_structured.return_value = expected_output
            
            mock_prompt = Mock()
            mock_resource_registry.get.return_value = mock_prompt
            
            result = await flow.run_pipeline(input_data)
            
            # Check that prompt variables were prepared correctly
            call_args = mock_llm.generate_structured.call_args
            prompt_vars = call_args[1]['prompt_variables']
            
            assert prompt_vars['user_message'] == "What's the weather like?"
            assert prompt_vars['persona'] == "helpful bot"
            assert prompt_vars['context'] == "No previous context available."
    
    @pytest.mark.asyncio
    async def test_run_pipeline_with_empty_memory_context(self):
        """Test conversation flow with empty memory context."""
        flow = ConversationFlow()
        
        input_data = ConversationInput(
            message="Tell me a joke",
            persona="funny assistant",
            memory_context_summary=""  # Empty string
        )
        
        expected_output = ConversationOutput(
            response="Why don't scientists trust atoms? Because they make up everything!",
            sentiment="positive"
        )
        
        with patch('flowlib.agent.components.conversation.flow.provider_registry') as mock_provider_registry, \
             patch('flowlib.agent.components.conversation.flow.resource_registry') as mock_resource_registry:
            
            mock_llm = AsyncMock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            mock_llm.generate_structured.return_value = expected_output
            
            mock_prompt = Mock()
            mock_resource_registry.get.return_value = mock_prompt
            
            result = await flow.run_pipeline(input_data)
            
            # Check that empty context was handled properly
            call_args = mock_llm.generate_structured.call_args
            prompt_vars = call_args[1]['prompt_variables']
            
            assert prompt_vars['context'] == "No previous context available."
    
    @pytest.mark.asyncio
    async def test_run_pipeline_with_rich_memory_context(self):
        """Test conversation flow with rich memory context."""
        flow = ConversationFlow()
        
        input_data = ConversationInput(
            message="Continue our discussion about Python",
            persona="technical expert",
            memory_context_summary="User asked about Python programming. We discussed variables, functions, and they showed interest in object-oriented programming concepts."
        )
        
        expected_output = ConversationOutput(
            response="Great! Let's dive into object-oriented programming in Python. The key concepts are classes, objects, inheritance, and encapsulation...",
            sentiment="positive"
        )
        
        with patch('flowlib.agent.components.conversation.flow.provider_registry') as mock_provider_registry, \
             patch('flowlib.agent.components.conversation.flow.resource_registry') as mock_resource_registry:
            
            mock_llm = AsyncMock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            mock_llm.generate_structured.return_value = expected_output
            
            mock_prompt = Mock()
            mock_resource_registry.get.return_value = mock_prompt
            
            result = await flow.run_pipeline(input_data)
            
            # Check that rich context was passed through
            call_args = mock_llm.generate_structured.call_args
            prompt_vars = call_args[1]['prompt_variables']
            
            assert "Python programming" in prompt_vars['context']
            assert "object-oriented programming" in prompt_vars['context']
            assert prompt_vars['persona'] == "technical expert"
    
    @pytest.mark.asyncio
    async def test_run_pipeline_prompt_variables_structure(self):
        """Test that prompt variables are structured correctly."""
        flow = ConversationFlow()
        
        input_data = ConversationInput(
            message="How do I learn programming?",
            persona="mentor",
            memory_context_summary="User is a beginner interested in coding"
        )
        
        with patch('flowlib.agent.components.conversation.flow.provider_registry') as mock_provider_registry, \
             patch('flowlib.agent.components.conversation.flow.resource_registry') as mock_resource_registry:
            
            mock_llm = AsyncMock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            mock_llm.generate_structured.return_value = ConversationOutput(
                response="Test response",
                sentiment="positive"
            )
            
            mock_prompt = Mock()
            mock_resource_registry.get.return_value = mock_prompt
            
            await flow.run_pipeline(input_data)
            
            # Verify LLM was called with correct structure
            call_args = mock_llm.generate_structured.call_args
            assert call_args[1]['prompt'] == mock_prompt
            assert call_args[1]['output_type'] == ConversationOutput
            
            prompt_vars = call_args[1]['prompt_variables']
            assert 'persona' in prompt_vars
            assert 'user_message' in prompt_vars
            assert 'context' in prompt_vars
            
            assert prompt_vars['persona'] == "mentor"
            assert prompt_vars['user_message'] == "How do I learn programming?"
            assert prompt_vars['context'] == "User is a beginner interested in coding"


class TestConversationFlowErrorHandling:
    """Test ConversationFlow error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_run_pipeline_provider_registry_failure(self):
        """Test handling when provider registry fails."""
        flow = ConversationFlow()
        
        input_data = ConversationInput(
            message="Hello",
            persona="assistant"
        )
        
        with patch('flowlib.agent.components.conversation.flow.provider_registry') as mock_provider_registry:
            # Mock provider registry to raise exception
            mock_provider_registry.get_by_config.side_effect = Exception("Provider not found")
            
            with pytest.raises(Exception, match="Provider not found"):
                await flow.run_pipeline(input_data)
    
    @pytest.mark.asyncio
    async def test_run_pipeline_resource_registry_failure(self):
        """Test handling when resource registry fails."""
        flow = ConversationFlow()
        
        input_data = ConversationInput(
            message="Hello",
            persona="assistant"
        )
        
        with patch('flowlib.agent.components.conversation.flow.provider_registry') as mock_provider_registry, \
             patch('flowlib.agent.components.conversation.flow.resource_registry') as mock_resource_registry:
            
            # Mock LLM provider
            mock_llm = AsyncMock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            
            # Mock resource registry to raise exception
            mock_resource_registry.get.side_effect = Exception("Prompt not found")
            
            with pytest.raises(Exception, match="Prompt not found"):
                await flow.run_pipeline(input_data)
    
    @pytest.mark.asyncio
    async def test_run_pipeline_llm_generation_failure(self):
        """Test handling when LLM generation fails."""
        flow = ConversationFlow()
        
        input_data = ConversationInput(
            message="Hello",
            persona="assistant"
        )
        
        with patch('flowlib.agent.components.conversation.flow.provider_registry') as mock_provider_registry, \
             patch('flowlib.agent.components.conversation.flow.resource_registry') as mock_resource_registry:
            
            # Mock LLM provider that fails
            mock_llm = AsyncMock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            mock_llm.generate_structured.side_effect = Exception("LLM generation failed")
            
            # Mock prompt resource
            mock_prompt = Mock()
            mock_resource_registry.get.return_value = mock_prompt
            
            with pytest.raises(Exception, match="LLM generation failed"):
                await flow.run_pipeline(input_data)


class TestConversationFlowIntegration:
    """Test ConversationFlow integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_run_pipeline_different_personas(self):
        """Test conversation flow with different personas."""
        flow = ConversationFlow()
        
        personas = [
            "friendly assistant",
            "technical expert", 
            "creative writer",
            "formal customer service"
        ]
        
        for persona in personas:
            input_data = ConversationInput(
                message="How can you help me?",
                persona=persona,
                memory_context_summary="New user asking about capabilities"
            )
            
            expected_output = ConversationOutput(
                response=f"As a {persona}, I can help you with various tasks.",
                sentiment="positive"
            )
            
            with patch('flowlib.agent.components.conversation.flow.provider_registry') as mock_provider_registry, \
                 patch('flowlib.agent.components.conversation.flow.resource_registry') as mock_resource_registry:
                
                mock_llm = AsyncMock()
                mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
                mock_llm.generate_structured.return_value = expected_output
                
                mock_prompt = Mock()
                mock_resource_registry.get.return_value = mock_prompt
                
                result = await flow.run_pipeline(input_data)
                
                # Verify persona was passed correctly
                call_args = mock_llm.generate_structured.call_args
                prompt_vars = call_args[1]['prompt_variables']
                assert prompt_vars['persona'] == persona
    
    @pytest.mark.asyncio
    async def test_run_pipeline_conversation_types(self):
        """Test conversation flow with different message types."""
        flow = ConversationFlow()
        
        message_types = [
            ("Hello!", "greeting"),
            ("What's 2+2?", "question"),
            ("Thank you for your help", "gratitude"),
            ("I'm having trouble with this", "problem"),
            ("Goodbye!", "farewell")
        ]
        
        for message, message_type in message_types:
            input_data = ConversationInput(
                message=message,
                persona="helpful assistant"
            )
            
            expected_output = ConversationOutput(
                response=f"Response to {message_type}",
                sentiment="neutral"
            )
            
            with patch('flowlib.agent.components.conversation.flow.provider_registry') as mock_provider_registry, \
                 patch('flowlib.agent.components.conversation.flow.resource_registry') as mock_resource_registry:
                
                mock_llm = AsyncMock()
                mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
                mock_llm.generate_structured.return_value = expected_output
                
                mock_prompt = Mock()
                mock_resource_registry.get.return_value = mock_prompt
                
                result = await flow.run_pipeline(input_data)
                
                # Verify message was passed correctly
                call_args = mock_llm.generate_structured.call_args
                prompt_vars = call_args[1]['prompt_variables']
                assert prompt_vars['user_message'] == message
    
    def test_conversation_flow_class_attributes(self):
        """Test that flow class has expected attributes."""
        assert ConversationFlow.__flow_name__ == "conversation"
        assert "Handle simple conversational interactions" in ConversationFlow.__flow_description__
        assert ConversationFlow.__is_infrastructure__ is False
    
    def test_pipeline_method_attributes(self):
        """Test that pipeline method has expected attributes."""
        run_pipeline = ConversationFlow.run_pipeline
        assert hasattr(run_pipeline, '__pipeline_input_model__')
        assert hasattr(run_pipeline, '__pipeline_output_model__')
        assert run_pipeline.__pipeline_input_model__ == ConversationInput
        assert run_pipeline.__pipeline_output_model__ == ConversationOutput


if __name__ == "__main__":
    pytest.main([__file__, "-v"])