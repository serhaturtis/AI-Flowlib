"""Tests for reflection prompts."""

import pytest
from flowlib.agent.components.reflection.prompts.default import (
    DefaultReflectionPrompt,
    TaskCompletionReflectionPrompt,
    DEFAULT_REFLECTION_TEMPLATE
)


class TestDefaultReflectionPrompt:
    """Test DefaultReflectionPrompt resource."""
    
    def test_prompt_exists(self):
        """Test that DefaultReflectionPrompt exists and has required attributes."""
        assert hasattr(DefaultReflectionPrompt, 'template')
        assert hasattr(DefaultReflectionPrompt, 'output_model')
        assert isinstance(DefaultReflectionPrompt.template, str)
    
    def test_prompt_template_content(self):
        """Test template contains expected placeholders and structure."""
        template = DefaultReflectionPrompt.template
        
        # Check for key placeholders
        assert "{{task_description}}" in template
        assert "{{plan_status}}" in template
        assert "{{plan_error}}" in template
        assert "{{step_reflections_summary}}" in template
        assert "{{execution_history_text}}" in template
        assert "{{state_summary}}" in template
        assert "{{current_progress}}" in template
        
        # Check for key analysis sections
        assert "Overall Task:" in template
        assert "Plan Execution Outcome:" in template
        assert "Summary of Step Reflections:" in template
        assert "Analysis Task:" in template
        assert "JSON object" in template
    
    def test_prompt_output_model(self):
        """Test prompt output model."""
        from flowlib.agent.components.reflection.models import ReflectionResult
        assert DefaultReflectionPrompt.output_model == ReflectionResult
    
    def test_prompt_decorator_metadata(self):
        """Test that the prompt decorator adds metadata."""
        # Check for decorator metadata
        assert hasattr(DefaultReflectionPrompt, '__resource_name__')
        assert DefaultReflectionPrompt.__resource_name__ == 'reflection_default'
        assert hasattr(DefaultReflectionPrompt, '__resource_type__')
        assert DefaultReflectionPrompt.__resource_type__ == 'prompt_config'
    
    def test_prompt_purpose(self):
        """Test that prompt clearly states its purpose."""
        template = DefaultReflectionPrompt.template
        assert "reflection assistant" in template.lower()
        assert "analyze the outcome" in template.lower()


class TestTaskCompletionReflectionPrompt:
    """Test TaskCompletionReflectionPrompt resource."""
    
    def test_prompt_exists(self):
        """Test that TaskCompletionReflectionPrompt exists and has required attributes."""
        assert hasattr(TaskCompletionReflectionPrompt, 'template')
        assert hasattr(TaskCompletionReflectionPrompt, 'config')
        assert isinstance(TaskCompletionReflectionPrompt.template, str)
        assert isinstance(TaskCompletionReflectionPrompt.config, dict)
    
    def test_prompt_template_content(self):
        """Test template contains expected placeholders and focus."""
        template = TaskCompletionReflectionPrompt.template
        
        # Check for key placeholders
        assert "{{task_description}}" in template
        assert "{{flow_name}}" in template
        assert "{{flow_result}}" in template
        assert "{{execution_history_text}}" in template
        
        # Check for task completion focus
        assert "task evaluation system" in template
        assert "determine if the current task has been completed" in template
        assert "GUIDELINES" in template
        
        # Check for evaluation criteria
        assert "Be strict in your evaluation" in template
        assert "multi-step tasks" in template
        assert "information requests" in template
        assert "action tasks" in template
    
    def test_prompt_config(self):
        """Test prompt configuration settings."""
        config = TaskCompletionReflectionPrompt.config
        
        assert config['max_tokens'] == 1024
        assert config['temperature'] == 0.2  # Lower temperature for more focused evaluation
        assert config['top_p'] == 0.9
        assert config['top_k'] == 30
    
    def test_prompt_decorator_metadata(self):
        """Test that the prompt decorator adds metadata."""
        assert hasattr(TaskCompletionReflectionPrompt, '__resource_name__')
        assert TaskCompletionReflectionPrompt.__resource_name__ == 'task_completion_reflection'
        assert hasattr(TaskCompletionReflectionPrompt, '__resource_type__')
        assert TaskCompletionReflectionPrompt.__resource_type__ == 'prompt_config'
    
    def test_both_prompts_compatibility(self):
        """Test that both prompts have compatible structures."""
        # Both should have string templates
        assert isinstance(DefaultReflectionPrompt.template, str)
        assert isinstance(TaskCompletionReflectionPrompt.template, str)
        
        # Both should be ResourceBase subclasses
        from flowlib.resources.models.base import ResourceBase
        assert issubclass(DefaultReflectionPrompt, ResourceBase)
        assert issubclass(TaskCompletionReflectionPrompt, ResourceBase)