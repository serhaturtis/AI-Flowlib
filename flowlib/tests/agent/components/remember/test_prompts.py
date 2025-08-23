"""Tests for remember prompts."""

import pytest
from flowlib.agent.components.remember.prompts import ContextAnalysisPrompt
from flowlib.resources.models.base import ResourceBase


class TestContextAnalysisPrompt:
    """Test ContextAnalysisPrompt resource."""
    
    def test_prompt_exists(self):
        """Test that ContextAnalysisPrompt exists and has required attributes."""
        assert hasattr(ContextAnalysisPrompt, 'template')
        assert hasattr(ContextAnalysisPrompt, 'config')
        assert isinstance(ContextAnalysisPrompt.template, str)
        assert isinstance(ContextAnalysisPrompt.config, dict)
    
    def test_prompt_is_resource(self):
        """Test that ContextAnalysisPrompt is a ResourceBase."""
        assert issubclass(ContextAnalysisPrompt, ResourceBase)
    
    def test_prompt_template_content(self):
        """Test template contains expected placeholders and structure."""
        template = ContextAnalysisPrompt.template
        
        # Check for key placeholders
        assert "{{context}}" in template
        assert "{{query}}" in template
        
        # Check for analysis instructions
        assert "Analyze the following context and query" in template
        assert "determine the best memory recall strategy" in template
        
        # Check for strategy explanations
        assert "CONTEXTUAL:" in template
        assert "ENTITY:" in template
        assert "TEMPORAL:" in template
        assert "SEMANTIC:" in template
        
        # Check for expected output format
        assert "JSON object" in template
        assert '"analysis"' in template
        assert '"recommended_strategy"' in template
        assert '"key_concepts"' in template
        assert '"confidence"' in template
    
    def test_prompt_config(self):
        """Test prompt configuration settings."""
        config = ContextAnalysisPrompt.config
        
        assert config['max_tokens'] == 512
        assert config['temperature'] == 0.3
        assert config['top_p'] == 0.95
    
    def test_prompt_decorator_metadata(self):
        """Test that the prompt decorator adds metadata."""
        assert hasattr(ContextAnalysisPrompt, '__resource_name__')
        assert ContextAnalysisPrompt.__resource_name__ == 'context_analysis'
        assert hasattr(ContextAnalysisPrompt, '__resource_type__')
        assert ContextAnalysisPrompt.__resource_type__ == 'prompt_config'
    
    def test_prompt_strategy_descriptions(self):
        """Test that all strategies are properly described."""
        template = ContextAnalysisPrompt.template
        
        # Check each strategy has a description
        strategies = [
            ("CONTEXTUAL", "current conversation or situation"),
            ("ENTITY", "specific people, places, or things"),
            ("TEMPORAL", "events, sequences, or time-based"),
            ("SEMANTIC", "concepts, meanings, or related ideas")
        ]
        
        for strategy, description in strategies:
            assert strategy in template
            assert description in template
    
    def test_prompt_json_structure(self):
        """Test that the expected JSON structure is documented."""
        template = ContextAnalysisPrompt.template
        
        # Check for JSON structure example
        assert '{' in template
        assert '}' in template
        assert '"analysis":' in template
        assert '"recommended_strategy":' in template
        assert '"key_concepts":' in template
        assert '"confidence":' in template
        
        # Check for data types in example
        assert '["list", "of", "key", "concepts"]' in template
        assert '0.85' in template  # Example confidence value