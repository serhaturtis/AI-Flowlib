"""Tests for agent planning prompts."""

import pytest
from typing import ClassVar

from flowlib.agent.components.planning.prompts import (
    DefaultPlanningPrompt,
    ConversationalPlanningPrompt,
    DefaultInputGenerationPrompt,
    ConversationalInputGenerationPrompt
)
from flowlib.resources.models.base import ResourceBase
from flowlib.resources.models.constants import ResourceType


class TestDefaultPlanningPrompt:
    """Test DefaultPlanningPrompt class."""
    
    def test_default_planning_prompt_creation(self):
        """Test creating DefaultPlanningPrompt instance."""
        prompt = DefaultPlanningPrompt(name="test_planning", type=ResourceType.PROMPT_CONFIG)
        
        assert prompt.name == "test_planning"
        assert prompt.type == ResourceType.PROMPT_CONFIG
        assert isinstance(prompt, ResourceBase)
    
    def test_default_planning_prompt_template_exists(self):
        """Test that template exists and is properly defined."""
        assert hasattr(DefaultPlanningPrompt, 'template')
        assert isinstance(DefaultPlanningPrompt.template, str)
        assert len(DefaultPlanningPrompt.template.strip()) > 0
    
    def test_default_planning_prompt_template_content(self):
        """Test that template contains expected content."""
        template = DefaultPlanningPrompt.template
        
        # Check for expected variables
        expected_variables = [
            '{{task_description}}',
            '{{available_flows_text}}',
            '{{cycle}}',
            '{{execution_history_text}}',
            '{{memory_context_summary}}'
        ]
        
        for var in expected_variables:
            assert var in template, f"Template missing variable: {var}"
    
    def test_default_planning_prompt_config_exists(self):
        """Test that config exists and has expected structure."""
        assert hasattr(DefaultPlanningPrompt, 'config')
        assert isinstance(DefaultPlanningPrompt.config, dict)
        
        config = DefaultPlanningPrompt.config
        
        # Check for expected config keys
        expected_keys = ['max_tokens', 'temperature', 'top_p', 'top_k']
        for key in expected_keys:
            assert key in config, f"Config missing key: {key}"
    
    def test_default_planning_prompt_config_values(self):
        """Test that config has reasonable values."""
        config = DefaultPlanningPrompt.config
        
        # Test that values are within reasonable ranges
        assert config['max_tokens'] > 0
        assert 0.0 <= config['temperature'] <= 2.0
        assert 0.0 <= config['top_p'] <= 1.0
        assert config['top_k'] > 0
    
    def test_default_planning_prompt_json_structure(self):
        """Test that template includes proper JSON structure guidance."""
        template = DefaultPlanningPrompt.template
        
        # Check for JSON structure elements
        json_elements = [
            '"plan_id"',
            '"task_description"',
            '"steps"',
            '"step_id"',
            '"flow_name"',
            '"step_intent"',
            '"rationale"',
            '"overall_rationale"'
        ]
        
        for element in json_elements:
            assert element in template, f"Template missing JSON element: {element}"
    
    def test_default_planning_prompt_critical_requirements(self):
        """Test that template includes critical requirements."""
        template = DefaultPlanningPrompt.template
        
        # Check for critical requirement mentions
        requirements = [
            'CRITICAL REQUIREMENTS',
            'MUST select a flow',
            'Do NOT invent flows',
            'JSON object'
        ]
        
        for req in requirements:
            assert req in template, f"Template missing requirement: {req}"


class TestConversationalPlanningPrompt:
    """Test ConversationalPlanningPrompt class."""
    
    def test_conversational_planning_prompt_creation(self):
        """Test creating ConversationalPlanningPrompt instance."""
        prompt = ConversationalPlanningPrompt(name="test_conv_planning", type=ResourceType.PROMPT_CONFIG)
        
        assert prompt.name == "test_conv_planning"
        assert prompt.type == ResourceType.PROMPT_CONFIG
        assert isinstance(prompt, ResourceBase)
    
    def test_conversational_planning_prompt_template_exists(self):
        """Test that template exists and is properly defined."""
        assert hasattr(ConversationalPlanningPrompt, 'template')
        assert isinstance(ConversationalPlanningPrompt.template, str)
        assert len(ConversationalPlanningPrompt.template.strip()) > 0
    
    def test_conversational_planning_prompt_template_content(self):
        """Test that template contains expected content for conversational context."""
        template = ConversationalPlanningPrompt.template
        
        # Check for conversational-specific variables
        expected_variables = [
            '{{task_description}}',
            '{{cycle}}',
            '{{available_flows_text}}',
            '{{execution_history_text}}',
            '{{memory_context}}'
        ]
        
        for var in expected_variables:
            assert var in template, f"Template missing variable: {var}"
        
        # Check for conversational context
        assert "conversational agent" in template.lower()
        assert "user message" in template.lower()
    
    def test_conversational_planning_prompt_config(self):
        """Test that config has appropriate values for conversational use."""
        config = ConversationalPlanningPrompt.config
        
        # Should have lower max_tokens than default (more focused)
        assert config['max_tokens'] <= DefaultPlanningPrompt.config['max_tokens']
        
        # Should have low temperature for consistent responses
        assert config['temperature'] <= 0.5


class TestDefaultInputGenerationPrompt:
    """Test DefaultInputGenerationPrompt class."""
    
    def test_default_input_generation_prompt_creation(self):
        """Test creating DefaultInputGenerationPrompt instance."""
        prompt = DefaultInputGenerationPrompt(name="test_input_gen", type=ResourceType.PROMPT_CONFIG)
        
        assert prompt.name == "test_input_gen"
        assert prompt.type == ResourceType.PROMPT_CONFIG
        assert isinstance(prompt, ResourceBase)
    
    def test_default_input_generation_prompt_template_exists(self):
        """Test that template exists and is properly defined."""
        assert hasattr(DefaultInputGenerationPrompt, 'template')
        assert isinstance(DefaultInputGenerationPrompt.template, str)
        assert len(DefaultInputGenerationPrompt.template.strip()) > 0
    
    def test_default_input_generation_prompt_template_content(self):
        """Test that template contains expected content."""
        template = DefaultInputGenerationPrompt.template
        
        # Check for expected variables
        expected_variables = [
            '{{task_description}}',
            '{{flow_name}}',
            '{{step_intent}}',
            '{{flow_description}}',
            '{{input_schema}}',
            '{{planning_rationale}}',
            '{{execution_history_text}}',
            '{{memory_context_summary}}'
        ]
        
        for var in expected_variables:
            assert var in template, f"Template missing variable: {var}"
    
    def test_default_input_generation_prompt_guidelines(self):
        """Test that template includes proper guidelines."""
        template = DefaultInputGenerationPrompt.template
        
        # Check for guideline sections
        guidelines = [
            'GUIDELINES',
            'input schema',
            'shell command flows',
            'conversation flows',
            'DO NOT use placeholders'
        ]
        
        for guideline in guidelines:
            assert guideline in template, f"Template missing guideline: {guideline}"
    
    def test_default_input_generation_prompt_config(self):
        """Test that config has appropriate values for input generation."""
        config = DefaultInputGenerationPrompt.config
        
        # Should have higher temperature than planning for more creative inputs
        assert config['temperature'] > DefaultPlanningPrompt.config['temperature']
        
        # Check all required config keys
        expected_keys = ['max_tokens', 'temperature', 'top_p', 'top_k']
        for key in expected_keys:
            assert key in config


class TestConversationalInputGenerationPrompt:
    """Test ConversationalInputGenerationPrompt class."""
    
    def test_conversational_input_generation_prompt_creation(self):
        """Test creating ConversationalInputGenerationPrompt instance."""
        prompt = ConversationalInputGenerationPrompt(
            name="test_conv_input_gen", 
            type=ResourceType.PROMPT_CONFIG
        )
        
        assert prompt.name == "test_conv_input_gen"
        assert prompt.type == ResourceType.PROMPT_CONFIG
        assert isinstance(prompt, ResourceBase)
    
    def test_conversational_input_generation_prompt_template_exists(self):
        """Test that template exists and is properly defined."""
        assert hasattr(ConversationalInputGenerationPrompt, 'template')
        assert isinstance(ConversationalInputGenerationPrompt.template, str)
        assert len(ConversationalInputGenerationPrompt.template.strip()) > 0
    
    def test_conversational_input_generation_prompt_template_content(self):
        """Test that template contains expected content for conversational input generation."""
        template = ConversationalInputGenerationPrompt.template
        
        # Check for conversational-specific content
        assert "conversational agent" in template.lower()
        assert "user message" in template.lower()
        
        # Should contain task_description variable
        assert '{{task_description}}' in template


class TestPromptDecorators:
    """Test that prompts are properly decorated."""
    
    def test_default_planning_prompt_decorated(self):
        """Test that DefaultPlanningPrompt is properly decorated."""
        # Check that the prompt decorator was applied
        assert hasattr(DefaultPlanningPrompt, '__resource_name__')
        assert hasattr(DefaultPlanningPrompt, '__resource_type__')
        
        assert DefaultPlanningPrompt.__resource_name__ == "planning_default"
        assert DefaultPlanningPrompt.__resource_type__ == ResourceType.PROMPT_CONFIG
    
    def test_conversational_planning_prompt_decorated(self):
        """Test that ConversationalPlanningPrompt is properly decorated."""
        assert hasattr(ConversationalPlanningPrompt, '__resource_name__')
        assert hasattr(ConversationalPlanningPrompt, '__resource_type__')
        
        assert ConversationalPlanningPrompt.__resource_name__ == "conversational_planning"
        assert ConversationalPlanningPrompt.__resource_type__ == ResourceType.PROMPT_CONFIG
    
    def test_default_input_generation_prompt_decorated(self):
        """Test that DefaultInputGenerationPrompt is properly decorated."""
        assert hasattr(DefaultInputGenerationPrompt, '__resource_name__')
        assert hasattr(DefaultInputGenerationPrompt, '__resource_type__')
        
        assert DefaultInputGenerationPrompt.__resource_name__ == "input_generation_default"
        assert DefaultInputGenerationPrompt.__resource_type__ == ResourceType.PROMPT_CONFIG
    
    def test_conversational_input_generation_prompt_decorated(self):
        """Test that ConversationalInputGenerationPrompt is properly decorated."""
        assert hasattr(ConversationalInputGenerationPrompt, '__resource_name__')
        assert hasattr(ConversationalInputGenerationPrompt, '__resource_type__')
        
        assert ConversationalInputGenerationPrompt.__resource_name__ == "conversational_input_generation"
        assert ConversationalInputGenerationPrompt.__resource_type__ == ResourceType.PROMPT_CONFIG


class TestPromptIntegration:
    """Test integration aspects of the prompts."""
    
    def test_all_prompts_inherit_from_resource_base(self):
        """Test that all prompt classes inherit from ResourceBase."""
        prompt_classes = [
            DefaultPlanningPrompt,
            ConversationalPlanningPrompt,
            DefaultInputGenerationPrompt,
            ConversationalInputGenerationPrompt
        ]
        
        for prompt_class in prompt_classes:
            assert issubclass(prompt_class, ResourceBase)
    
    def test_prompt_template_variable_consistency(self):
        """Test that related prompts use consistent variable naming."""
        # Planning prompts should use similar variable names
        default_planning_vars = set()
        conv_planning_vars = set()
        
        # Extract variables from templates (simplified check)
        for template in [DefaultPlanningPrompt.template, ConversationalPlanningPrompt.template]:
            import re
            variables = re.findall(r'\{\{(\w+)\}\}', template)
            if template == DefaultPlanningPrompt.template:
                default_planning_vars.update(variables)
            else:
                conv_planning_vars.update(variables)
        
        # Should have some common variables
        common_vars = default_planning_vars.intersection(conv_planning_vars)
        assert len(common_vars) > 0, "Planning prompts should share some common variables"
    
    def test_prompt_config_structure_consistency(self):
        """Test that all prompts have consistent config structure."""
        prompt_classes = [
            DefaultPlanningPrompt,
            ConversationalPlanningPrompt,
            DefaultInputGenerationPrompt,
            ConversationalInputGenerationPrompt
        ]
        
        for prompt_class in prompt_classes:
            config = prompt_class.config
            
            # All should have these basic keys
            basic_keys = ['max_tokens', 'temperature']
            for key in basic_keys:
                assert key in config, f"{prompt_class.__name__} missing config key: {key}"
    
    def test_prompt_template_format(self):
        """Test that templates follow expected format conventions."""
        prompt_classes = [
            DefaultPlanningPrompt,
            ConversationalPlanningPrompt,
            DefaultInputGenerationPrompt,
            ConversationalInputGenerationPrompt
        ]
        
        for prompt_class in prompt_classes:
            template = prompt_class.template
            
            # Should not be empty
            assert len(template.strip()) > 0
            
            # Should contain at least one variable
            assert '{{' in template and '}}' in template
            
            # Should not contain obvious syntax errors
            assert template.count('{{') == template.count('}}')


class TestPromptInstantiation:
    """Test that prompts can be properly instantiated."""
    
    def test_create_all_prompt_instances(self):
        """Test creating instances of all prompt classes."""
        prompt_data = [
            (DefaultPlanningPrompt, "planning_default"),
            (ConversationalPlanningPrompt, "conversational_planning"),
            (DefaultInputGenerationPrompt, "input_generation_default"),
            (ConversationalInputGenerationPrompt, "conversational_input_generation")
        ]
        
        for prompt_class, name in prompt_data:
            # Should be able to create instance
            instance = prompt_class(name=name, type=ResourceType.PROMPT_CONFIG)
            
            assert instance.name == name
            assert instance.type == ResourceType.PROMPT_CONFIG
            assert hasattr(instance, 'template')
            assert hasattr(instance, 'config')
    
    def test_prompt_instances_have_template_access(self):
        """Test that prompt instances can access their templates."""
        prompt = DefaultPlanningPrompt(name="test", type=ResourceType.PROMPT_CONFIG)
        
        # Should be able to access template from instance
        assert hasattr(prompt, 'template')
        assert prompt.template == DefaultPlanningPrompt.template
    
    def test_prompt_instances_have_config_access(self):
        """Test that prompt instances can access their configs."""
        prompt = DefaultPlanningPrompt(name="test", type=ResourceType.PROMPT_CONFIG)
        
        # Should be able to access config from instance
        assert hasattr(prompt, 'config')
        assert prompt.config == DefaultPlanningPrompt.config