"""Tests for conversation prompts."""

import pytest
from flowlib.agent.components.conversation.prompts import ConversationPrompt
from flowlib.resources.models.base import ResourceBase


class TestConversationPrompt:
    """Test ConversationPrompt resource."""
    
    def test_prompt_is_resource(self):
        """Test that ConversationPrompt is a ResourceBase."""
        assert issubclass(ConversationPrompt, ResourceBase)
    
    def test_prompt_template_exists(self):
        """Test that the prompt has a template."""
        assert hasattr(ConversationPrompt, 'template')
        assert isinstance(ConversationPrompt.template, str)
        assert len(ConversationPrompt.template) > 0
    
    def test_prompt_template_placeholders(self):
        """Test that template contains expected placeholders."""
        template = ConversationPrompt.template
        # Check for expected placeholders
        assert "{{persona}}" in template
        assert "{{user_message}}" in template
        assert "{{context}}" in template
    
    def test_prompt_template_json_format(self):
        """Test that template specifies JSON format."""
        template = ConversationPrompt.template
        # Check that it mentions JSON format
        assert "JSON" in template
        assert '"response"' in template
        assert '"sentiment"' in template
    
    def test_prompt_decorator_metadata(self):
        """Test that prompt decorator adds metadata."""
        # The @prompt decorator should add metadata
        assert hasattr(ConversationPrompt, '__resource_name__')
        assert hasattr(ConversationPrompt, '__resource_type__')
        assert hasattr(ConversationPrompt, '__resource_metadata__')
        assert ConversationPrompt.__resource_name__ == 'conversation-prompt'
        assert ConversationPrompt.__resource_type__ == 'prompt_config'
    
    def test_prompt_instance_creation(self):
        """Test creating an instance of ConversationPrompt."""
        # ConversationPrompt is a ResourceBase subclass that requires name and type
        prompt_instance = ConversationPrompt(
            name="conversation-prompt",
            type="prompt_config"
        )
        assert isinstance(prompt_instance, ConversationPrompt)
        assert isinstance(prompt_instance, ResourceBase)
    
    def test_template_format_structure(self):
        """Test the structure of the template."""
        template = ConversationPrompt.template
        # Check for key sections
        assert "User message:" in template
        assert "Context:" in template
        assert "Give response to user's input with this persona" in template
        assert "Ensure your response is valid JSON" in template
    
    def test_sentiment_options(self):
        """Test that template includes sentiment options."""
        template = ConversationPrompt.template
        assert "positive|negative|neutral" in template
        assert "(optional)" in template  # Sentiment is optional