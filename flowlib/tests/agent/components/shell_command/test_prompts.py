"""Tests for shell command prompt definitions."""

import pytest
from flowlib.agent.components.shell_command.prompts import ShellCommandGenerationPrompt


class TestShellCommandGenerationPrompt:
    """Test the ShellCommandGenerationPrompt class."""

    def test_prompt_attributes(self):
        """Test that prompt has correct attributes."""
        # The prompt class itself has the attributes as ClassVars
        assert hasattr(ShellCommandGenerationPrompt, 'template')
        assert isinstance(ShellCommandGenerationPrompt.template, str)
        assert "You are an expert shell command generator" in ShellCommandGenerationPrompt.template
        assert "{{intent}}" in ShellCommandGenerationPrompt.template
        assert "{{target_resource}}" in ShellCommandGenerationPrompt.template
        assert "{{parameters}}" in ShellCommandGenerationPrompt.template
        assert "{{output_description}}" in ShellCommandGenerationPrompt.template
        assert "{{available_commands_list}}" in ShellCommandGenerationPrompt.template
        
        # Check config
        assert hasattr(ShellCommandGenerationPrompt, 'config')
        assert isinstance(ShellCommandGenerationPrompt.config, dict)
        assert ShellCommandGenerationPrompt.config['max_tokens'] == 512
        assert ShellCommandGenerationPrompt.config['temperature'] == 0.3
        assert ShellCommandGenerationPrompt.config['top_p'] == 0.95

    def test_prompt_template_structure(self):
        """Test the structure of the prompt template."""
        template = ShellCommandGenerationPrompt.template
        
        # Check for key sections
        assert "Intent:" in template
        assert "Target Resource:" in template
        assert "Parameters:" in template
        assert "Desired Output:" in template
        assert "Available Commands:" in template
        assert "Guidelines:" in template
        
        # Check for important guidelines
        assert "single command line" in template.lower()
        assert "ONLY commands from the 'Available Commands' list" in template
        assert "DO NOT use placeholders" in template
        assert "quoting and escaping" in template
        
        # Check for JSON structure requirement
        assert '"command":' in template
        assert '"reasoning":' in template

    def test_prompt_render(self):
        """Test rendering the prompt with variables."""
        # Get the template
        template = ShellCommandGenerationPrompt.template
        
        # Test manual template substitution (since ResourceBase doesn't have render method)
        variables = {
            'intent': 'Create a file with hello world content',
            'target_resource': '/tmp/hello.txt',
            'parameters': '{"content": "Hello, World!"}',
            'output_description': 'Confirmation of file creation',
            'available_commands_list': '- echo\n- cat\n- touch\n- ls'
        }
        
        # Simulate template rendering manually
        rendered = template
        for key, value in variables.items():
            rendered = rendered.replace(f'{{{{{key}}}}}', str(value))
        
        # Check that variables were substituted
        assert 'Create a file with hello world content' in rendered
        assert '/tmp/hello.txt' in rendered
        assert '{"content": "Hello, World!"}' in rendered
        assert 'Confirmation of file creation' in rendered
        assert '- echo\n- cat\n- touch\n- ls' in rendered
        
        # Check that no template variables remain
        assert '{{' not in rendered
        assert '}}' not in rendered

    def test_prompt_decorator(self):
        """Test that the prompt decorator is properly applied."""
        # Check that the class has the resource metadata  
        assert hasattr(ShellCommandGenerationPrompt, '__resource_name__')
        assert hasattr(ShellCommandGenerationPrompt, '__resource_type__')
        assert hasattr(ShellCommandGenerationPrompt, '__resource_metadata__')
        
        assert ShellCommandGenerationPrompt.__resource_name__ == 'shell_command_generation'
        assert ShellCommandGenerationPrompt.__resource_type__ == 'prompt_config'

    def test_prompt_error_case(self):
        """Test the error case mentioned in guidelines."""
        template = ShellCommandGenerationPrompt.template
        
        # Check that error handling is mentioned
        assert 'Error: Cannot achieve intent with available tools' in template

    def test_prompt_json_format(self):
        """Test that the prompt specifies JSON format correctly."""
        template = ShellCommandGenerationPrompt.template
        
        # Check for JSON structure specification
        assert 'JSON object' in template
        assert 'exact structure' in template
        assert '{' in template
        assert '}' in template
        
        # Check field names are specified
        assert '"command"' in template
        assert '"reasoning"' in template

    def test_prompt_config_values(self):
        """Test that config values are appropriate for command generation."""
        # Temperature should be low for consistent command generation
        assert ShellCommandGenerationPrompt.config['temperature'] <= 0.5
        
        # Max tokens should be sufficient for command + reasoning
        assert ShellCommandGenerationPrompt.config['max_tokens'] >= 256
        
        # Top_p should be reasonable
        assert 0.8 <= ShellCommandGenerationPrompt.config['top_p'] <= 1.0

    def test_prompt_safety_guidelines(self):
        """Test that prompt includes safety guidelines."""
        template = ShellCommandGenerationPrompt.template
        
        # Check for safety-related guidelines
        assert "ONLY" in template  # Emphasis on using only allowed commands
        assert "Available Commands" in template
        assert "cannot be achieved" in template  # Error handling