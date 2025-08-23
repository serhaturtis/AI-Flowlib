"""Tests for agent classification prompts."""

import pytest
from typing import ClassVar

from flowlib.agent.components.classification.prompts import MessageClassifierPrompt
from flowlib.resources.models.base import ResourceBase
from flowlib.resources.models.constants import ResourceType


class TestMessageClassifierPrompt:
    """Test MessageClassifierPrompt class."""
    
    def test_message_classifier_prompt_creation(self):
        """Test creating MessageClassifierPrompt instance."""
        prompt = MessageClassifierPrompt(name="test_classifier", type=ResourceType.PROMPT_CONFIG)
        
        assert prompt.name == "test_classifier"
        assert prompt.type == ResourceType.PROMPT_CONFIG
        assert isinstance(prompt, ResourceBase)
    
    def test_message_classifier_prompt_template_exists(self):
        """Test that template exists and is properly defined."""
        assert hasattr(MessageClassifierPrompt, 'template')
        assert isinstance(MessageClassifierPrompt.template, str)
        assert len(MessageClassifierPrompt.template.strip()) > 0
    
    def test_message_classifier_prompt_template_content(self):
        """Test that template contains expected content."""
        template = MessageClassifierPrompt.template
        
        # Check for expected variables
        expected_variables = [
            '{{message}}',
            '{{conversation_history}}',
            '{{memory_context_summary}}'
        ]
        
        for var in expected_variables:
            assert var in template, f"Template missing variable: {var}"
    
    def test_message_classifier_prompt_classification_rules(self):
        """Test that template contains classification rules."""
        template = MessageClassifierPrompt.template
        
        # Check for rule sections
        rule_keywords = [
            'CONVERSATION:',
            'TASK:',
            'Classification rules',
            'execute_task',
            'confidence',
            'category',
            'task_description'
        ]
        
        for keyword in rule_keywords:
            assert keyword in template, f"Template missing keyword: {keyword}"
    
    def test_message_classifier_prompt_conversation_examples(self):
        """Test that template includes conversation category guidance."""
        template = MessageClassifierPrompt.template
        
        conversation_indicators = [
            'Greetings',
            'simple questions',
            'social responses',
            'acknowledgements',
            'confirmations'
        ]
        
        for indicator in conversation_indicators:
            assert indicator in template, f"Template missing conversation indicator: {indicator}"
    
    def test_message_classifier_prompt_task_examples(self):
        """Test that template includes task category guidance."""
        template = MessageClassifierPrompt.template
        
        task_indicators = [
            'perform an action',
            'run commands',
            'manage files',
            'perform calculations',
            'generate content',
            'analyze information'
        ]
        
        for indicator in task_indicators:
            assert indicator in template, f"Template missing task indicator: {indicator}"
    
    def test_message_classifier_prompt_output_format(self):
        """Test that template specifies output format."""
        template = MessageClassifierPrompt.template
        
        output_requirements = [
            'execute_task:',
            'confidence:',
            'category:',
            'task_description:'
        ]
        
        for requirement in output_requirements:
            assert requirement in template, f"Template missing output requirement: {requirement}"
    
    def test_message_classifier_prompt_memory_context(self):
        """Test that template includes memory context section."""
        template = MessageClassifierPrompt.template
        
        assert "Memory Context Summary" in template
        assert "{{memory_context_summary}}" in template
    
    def test_message_classifier_prompt_analysis_guidance(self):
        """Test that template provides analysis guidance."""
        template = MessageClassifierPrompt.template
        
        analysis_keywords = [
            'Analyze the message carefully',
            'complexity',
            'intent',
            'available memory context'
        ]
        
        for keyword in analysis_keywords:
            assert keyword in template, f"Template missing analysis keyword: {keyword}"
    
    def test_message_classifier_prompt_boolean_values(self):
        """Test that template specifies correct boolean values."""
        template = MessageClassifierPrompt.template
        
        # Should specify true/false values explicitly
        assert 'true' in template.lower()
        assert 'false' in template.lower()
    
    def test_message_classifier_prompt_confidence_range(self):
        """Test that template specifies confidence range."""
        template = MessageClassifierPrompt.template
        
        assert '0.0-1.0' in template or '(0.0-1.0)' in template


class TestPromptDecorators:
    """Test that prompts are properly decorated."""
    
    def test_message_classifier_prompt_decorated(self):
        """Test that MessageClassifierPrompt is properly decorated."""
        # Check that the prompt decorator was applied
        assert hasattr(MessageClassifierPrompt, '__resource_name__')
        assert hasattr(MessageClassifierPrompt, '__resource_type__')
        
        assert MessageClassifierPrompt.__resource_name__ == "message_classifier_prompt"
        assert MessageClassifierPrompt.__resource_type__ == ResourceType.PROMPT_CONFIG


class TestPromptIntegration:
    """Test integration aspects of the prompt."""
    
    def test_prompt_inherits_from_resource_base(self):
        """Test that prompt class inherits from ResourceBase."""
        assert issubclass(MessageClassifierPrompt, ResourceBase)
    
    def test_prompt_template_format(self):
        """Test that template follows expected format conventions."""
        template = MessageClassifierPrompt.template
        
        # Should not be empty
        assert len(template.strip()) > 0
        
        # Should contain variables in correct format
        assert '{{' in template and '}}' in template
        
        # Should not contain obvious syntax errors
        assert template.count('{{') == template.count('}}')
    
    def test_prompt_variable_consistency(self):
        """Test that template variables are consistent with expected usage."""
        template = MessageClassifierPrompt.template
        
        # Extract variables from template
        import re
        variables = re.findall(r'\{\{(\w+)\}\}', template)
        
        # Should have expected variables
        expected_vars = {'message', 'conversation_history', 'memory_context_summary'}
        found_vars = set(variables)
        
        assert expected_vars.issubset(found_vars), f"Missing variables: {expected_vars - found_vars}"
    
    def test_prompt_classification_logic(self):
        """Test that template contains clear classification logic."""
        template = MessageClassifierPrompt.template
        
        # Should distinguish between conversation and task clearly
        assert 'CONVERSATION' in template
        assert 'TASK' in template
        
        # Should provide clear criteria
        assert 'without' in template.lower()  # "without needing external tools"
        assert 'perform' in template.lower()  # "perform an action"
    
    def test_prompt_contextual_analysis(self):
        """Test that template encourages contextual analysis."""
        template = MessageClassifierPrompt.template
        
        contextual_keywords = [
            'history',
            'context',
            'memory',
            'available',
            'present'
        ]
        
        for keyword in contextual_keywords:
            assert keyword in template.lower(), f"Template should mention {keyword} for contextual analysis"


class TestPromptInstantiation:
    """Test that prompts can be properly instantiated."""
    
    def test_create_prompt_instance(self):
        """Test creating instance of prompt class."""
        # Should be able to create instance
        instance = MessageClassifierPrompt(name="message_classifier_prompt", type=ResourceType.PROMPT_CONFIG)
        
        assert instance.name == "message_classifier_prompt"
        assert instance.type == ResourceType.PROMPT_CONFIG
        assert hasattr(instance, 'template')
    
    def test_prompt_instance_has_template_access(self):
        """Test that prompt instance can access template."""
        prompt = MessageClassifierPrompt(name="test", type=ResourceType.PROMPT_CONFIG)
        
        # Should be able to access template from instance
        assert hasattr(prompt, 'template')
        assert prompt.template == MessageClassifierPrompt.template
    
    def test_prompt_template_immutability(self):
        """Test that template is a class variable and not modified per instance."""
        prompt1 = MessageClassifierPrompt(name="test1", type=ResourceType.PROMPT_CONFIG)
        prompt2 = MessageClassifierPrompt(name="test2", type=ResourceType.PROMPT_CONFIG)
        
        # Both instances should have the same template
        assert prompt1.template == prompt2.template
        assert prompt1.template is prompt2.template


class TestPromptContent:
    """Test specific content aspects of the prompt."""
    
    def test_prompt_task_identification_criteria(self):
        """Test that prompt provides clear task identification criteria."""
        template = MessageClassifierPrompt.template
        
        # Should specify what constitutes a task
        task_criteria = [
            'action',
            'do something',
            'perform',
            'execute',
            'run',
            'commands'
        ]
        
        for criterion in task_criteria:
            assert criterion in template.lower(), f"Template should include task criterion: {criterion}"
    
    def test_prompt_conversation_identification_criteria(self):
        """Test that prompt provides clear conversation identification criteria."""
        template = MessageClassifierPrompt.template
        
        # Should specify what constitutes conversation
        conversation_criteria = [
            'greeting',
            'simple',
            'knowledge',
            'social',
            'acknowledgement'
        ]
        
        for criterion in conversation_criteria:
            assert criterion in template.lower(), f"Template should include conversation criterion: {criterion}"
    
    def test_prompt_edge_case_guidance(self):
        """Test that prompt provides guidance for edge cases."""
        template = MessageClassifierPrompt.template
        
        # Should provide guidance for complex scenarios
        edge_case_keywords = [
            'complex',
            'multi-step',
            'external',
            'real-time',
            'up-to-date'
        ]
        
        found_keywords = []
        for keyword in edge_case_keywords:
            if keyword in template.lower():
                found_keywords.append(keyword)
        
        # Should have at least some edge case guidance
        assert len(found_keywords) > 0, "Template should include edge case guidance"
    
    def test_prompt_output_structure(self):
        """Test that prompt specifies clear output structure."""
        template = MessageClassifierPrompt.template
        
        # Should specify all required output fields
        output_fields = [
            'execute_task',
            'confidence',
            'category', 
            'task_description'
        ]
        
        for field in output_fields:
            assert field in template, f"Template should specify output field: {field}"
    
    def test_prompt_conditional_logic(self):
        """Test that prompt includes conditional logic for task_description."""
        template = MessageClassifierPrompt.template
        
        # Should specify when to include task_description
        conditional_keywords = [
            'If execute_task',
            'when',
            'Leave empty',
            'null for conversation'
        ]
        
        found_conditionals = []
        for keyword in conditional_keywords:
            if keyword in template:
                found_conditionals.append(keyword)
        
        # Should have some conditional logic
        assert len(found_conditionals) > 0, "Template should include conditional logic for task_description"