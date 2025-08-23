"""Tests for Todo Generation prompts."""

import pytest
from flowlib.agent.components.planning.todo_generation.prompts import TodoGenerationPrompt


class TestTodoGenerationPrompt:
    """Test the TodoGenerationPrompt class."""

    def test_prompt_attributes(self):
        """Test that prompt has correct attributes."""
        # The prompt class itself has the attributes as ClassVars
        assert hasattr(TodoGenerationPrompt, 'template')
        assert isinstance(TodoGenerationPrompt.template, str)
        assert "You are a task management expert" in TodoGenerationPrompt.template
        assert "{{task_description}}" in TodoGenerationPrompt.template
        assert "{{plan_steps}}" in TodoGenerationPrompt.template
        assert "{{context}}" in TodoGenerationPrompt.template

    def test_prompt_template_structure(self):
        """Test the structure of the prompt template."""
        template = TodoGenerationPrompt.template
        
        # Check for key sections
        assert "Original Task:" in template
        assert "Multi-Step Plan:" in template
        assert "Context:" in template
        assert "Your task:" in template
        assert "Guidelines:" in template
        
        # Check for important instructions
        assert "Convert each plan step into" in template
        assert "Identify dependencies" in template
        assert "Assign appropriate priorities" in template
        assert "Estimate effort" in template
        assert "execution strategy" in template
        
        # Check for JSON structure requirement
        assert "JSON object" in template
        assert '"todos":' in template
        assert '"execution_strategy":' in template
        assert '"reasoning":' in template

    def test_prompt_guidelines(self):
        """Test that prompt includes important guidelines."""
        template = TodoGenerationPrompt.template
        
        # Check for specific guidelines
        assert "Make TODOs specific and actionable" in template
        assert "Identify logical dependencies" in template
        assert "Higher priority for foundational" in template
        assert "Consider parallel execution" in template
        assert "Provide realistic time estimates" in template

    def test_prompt_json_schema(self):
        """Test that prompt specifies correct JSON schema."""
        template = TodoGenerationPrompt.template
        
        # Check for required todo fields
        assert '"id":' in template
        assert '"content":' in template
        assert '"priority":' in template
        assert '"status":' in template
        assert '"depends_on":' in template
        assert '"assigned_tool":' in template
        assert '"execution_context":' in template
        assert '"estimated_duration":' in template
        assert '"tags":' in template
        
        # Check for output fields
        assert '"execution_strategy":' in template
        assert '"estimated_duration":' in template
        assert '"dependency_map":' in template
        assert '"reasoning":' in template

    def test_prompt_priority_values(self):
        """Test that prompt specifies correct priority values."""
        template = TodoGenerationPrompt.template
        
        # Check for priority enum values
        assert "LOW|MEDIUM|HIGH|URGENT" in template

    def test_prompt_execution_strategies(self):
        """Test that prompt mentions execution strategies."""
        template = TodoGenerationPrompt.template
        
        # Check for execution strategy options
        assert "sequential|parallel|hybrid" in template

    def test_prompt_variable_placeholders(self):
        """Test that all expected template variables are present."""
        template = TodoGenerationPrompt.template
        
        expected_variables = [
            "{{task_description}}",
            "{{plan_steps}}",
            "{{context}}"
        ]
        
        for variable in expected_variables:
            assert variable in template

    def test_prompt_render_simulation(self):
        """Test rendering the prompt with variables."""
        template = TodoGenerationPrompt.template
        
        # Test manual template substitution
        variables = {
            'task_description': 'Build a web application',
            'plan_steps': 'Step 1: Setup\nStep 2: Development\nStep 3: Testing',
            'context': '{"framework": "React", "deadline": "2024-01-01"}'
        }
        
        # Simulate template rendering manually
        rendered = template
        for key, value in variables.items():
            rendered = rendered.replace(f'{{{{{key}}}}}', str(value))
        
        # Check that variables were substituted
        assert 'Build a web application' in rendered
        assert 'Step 1: Setup' in rendered
        assert 'Step 2: Development' in rendered
        assert 'Step 3: Testing' in rendered
        assert '"framework": "React"' in rendered
        assert '"deadline": "2024-01-01"' in rendered
        
        # Check that no template variables remain
        assert '{{' not in rendered
        assert '}}' not in rendered

    def test_prompt_decorator(self):
        """Test that the prompt decorator is properly applied."""
        # Check that the class has the resource metadata
        assert hasattr(TodoGenerationPrompt, '__resource_name__')
        assert hasattr(TodoGenerationPrompt, '__resource_type__')
        assert hasattr(TodoGenerationPrompt, '__resource_metadata__')
        
        assert TodoGenerationPrompt.__resource_name__ == 'todo-generation-prompt'
        assert TodoGenerationPrompt.__resource_type__ == 'prompt_config'

    def test_prompt_completion_requirements(self):
        """Test that prompt emphasizes completion requirements."""
        template = TodoGenerationPrompt.template
        
        # Check for completion and quality requirements
        assert "actionable" in template.lower()
        assert "logical" in template.lower()
        assert "realistic" in template.lower()

    def test_prompt_context_handling(self):
        """Test that prompt properly handles context information."""
        template = TodoGenerationPrompt.template
        
        # Check that context is mentioned and utilized
        assert "Context: {{context}}" in template
        
        # Context should be part of the input considerations
        context_mentions = template.count("context")
        assert context_mentions >= 1

    def test_prompt_dependency_emphasis(self):
        """Test that prompt emphasizes dependency management."""
        template = TodoGenerationPrompt.template
        
        # Check for dependency-related instructions
        assert "dependencies" in template.lower()
        assert "prerequisite" in template.lower() or "depends_on" in template
        assert "must complete before" in template.lower() or "must be completed before" in template.lower()

    def test_prompt_time_estimation(self):
        """Test that prompt includes time estimation guidance."""
        template = TodoGenerationPrompt.template
        
        # Check for time-related instructions
        assert "estimated_duration" in template
        assert "PT30M" in template  # ISO 8601 duration format example
        assert "time estimates" in template.lower() or "estimate effort" in template.lower()

    def test_prompt_execution_context(self):
        """Test that prompt includes execution context guidance."""
        template = TodoGenerationPrompt.template
        
        # Check for execution context fields
        assert "execution_context" in template
        assert "flow_name" in template
        assert "flow_inputs" in template
        assert "original_step_id" in template

    def test_prompt_ensures_actionability(self):
        """Test that prompt emphasizes creating actionable todos."""
        template = TodoGenerationPrompt.template
        
        # Check for actionability requirements
        assert "actionable" in template.lower()
        assert "specific" in template.lower()
        assert "not vague" in template or "actionable" in template