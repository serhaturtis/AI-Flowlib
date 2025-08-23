"""Tests for entity recall prompts."""

import pytest
from unittest.mock import MagicMock

from flowlib.agent.components.remember.entity_recall.prompts import EntityValidationPrompt
from flowlib.resources.models.base import ResourceBase


class TestEntityValidationPrompt:
    """Test cases for EntityValidationPrompt."""

    def test_prompt_inheritance(self):
        """Test that EntityValidationPrompt inherits from ResourceBase."""
        assert issubclass(EntityValidationPrompt, ResourceBase)

    def test_prompt_decorator(self):
        """Test that prompt has correct decorator metadata."""
        assert hasattr(EntityValidationPrompt, '__resource_name__')
        assert hasattr(EntityValidationPrompt, '__resource_type__')
        
        assert EntityValidationPrompt.__resource_name__ == 'entity_validation'
        # Check for the actual resource type used by @prompt decorator
        from flowlib.resources.models.constants import ResourceType
        assert EntityValidationPrompt.__resource_type__ == ResourceType.PROMPT_CONFIG

    def test_prompt_template_exists(self):
        """Test that template attribute exists and is not empty."""
        assert hasattr(EntityValidationPrompt, 'template')
        assert isinstance(EntityValidationPrompt.template, str)
        assert len(EntityValidationPrompt.template.strip()) > 0

    def test_prompt_template_structure(self):
        """Test prompt template structure and content."""
        template = EntityValidationPrompt.template
        
        # Check for required placeholder variables
        assert "{{entity_id}}" in template
        assert "{{entity_type}}" in template
        assert "{{context}}" in template
        
        # Check for key sections
        assert "Validate the following entity" in template
        assert "Return your validation as a JSON object" in template

    def test_prompt_template_json_format(self):
        """Test that template includes proper JSON format specification."""
        template = EntityValidationPrompt.template
        
        # Check for JSON structure elements
        assert "is_valid" in template
        assert "validation_message" in template
        assert "suggested_alternatives" in template
        assert "confidence" in template
        
        # Check for JSON formatting
        assert '{\n' in template
        assert '}' in template
        assert '"' in template

    def test_prompt_config_exists(self):
        """Test that config attribute exists."""
        assert hasattr(EntityValidationPrompt, 'config')
        assert isinstance(EntityValidationPrompt.config, dict)

    def test_prompt_config_structure(self):
        """Test prompt configuration structure."""
        config = EntityValidationPrompt.config
        
        # Check for expected configuration keys
        assert 'max_tokens' in config
        assert 'temperature' in config
        assert 'top_p' in config

    def test_prompt_config_values(self):
        """Test prompt configuration values."""
        config = EntityValidationPrompt.config
        
        # Verify configuration values
        assert config['max_tokens'] == 256
        assert config['temperature'] == 0.2
        assert config['top_p'] == 0.95
        
        # Verify types
        assert isinstance(config['max_tokens'], int)
        assert isinstance(config['temperature'], (int, float))
        assert isinstance(config['top_p'], (int, float))

    def test_prompt_config_reasoning(self):
        """Test that configuration values are reasonable for validation task."""
        config = EntityValidationPrompt.config
        
        # Temperature should be low for consistent validation
        assert config['temperature'] <= 0.3
        
        # Max tokens should be sufficient for JSON response
        assert config['max_tokens'] >= 200
        
        # Top_p should allow for focused responses
        assert 0.8 <= config['top_p'] <= 1.0

    def test_prompt_instantiation(self):
        """Test that prompt can be instantiated."""
        from flowlib.resources.models.constants import ResourceType
        
        prompt = EntityValidationPrompt(name="entity_validation", type=ResourceType.PROMPT_CONFIG)
        
        assert isinstance(prompt, EntityValidationPrompt)
        assert isinstance(prompt, ResourceBase)
        assert prompt.name == "entity_validation"
        assert prompt.type == ResourceType.PROMPT_CONFIG

    def test_prompt_template_rendering_structure(self):
        """Test template structure for variable substitution."""
        template = EntityValidationPrompt.template
        
        # Check that variables are in proper Jinja2 format
        import re
        variables = re.findall(r'\{\{([^}]+)\}\}', template)
        
        expected_variables = ['entity_id', 'entity_type', 'context']
        for var in expected_variables:
            assert var.strip() in [v.strip() for v in variables]

    def test_prompt_template_whitespace_handling(self):
        """Test template handles whitespace appropriately."""
        template = EntityValidationPrompt.template
        
        # Template should be well-formatted
        lines = template.split('\n')
        
        # Should not start or end with excessive whitespace
        assert not template.startswith('\n\n\n')
        assert not template.endswith('\n\n\n')
        
        # Should have reasonable structure
        assert len(lines) > 5  # Multi-line template

    def test_prompt_template_json_example_validity(self):
        """Test that JSON example in template is valid structure."""
        template = EntityValidationPrompt.template
        
        # Extract JSON portion (simplified check)
        json_start = template.find('{')
        json_end = template.rfind('}') + 1
        
        assert json_start != -1
        assert json_end > json_start
        
        json_section = template[json_start:json_end]
        
        # Basic JSON structure checks
        assert json_section.count('{') == json_section.count('}')
        assert '"is_valid"' in json_section
        assert 'true/false' in json_section

    def test_prompt_class_variables_are_class_vars(self):
        """Test that template and config are properly declared as ClassVar."""
        import typing
        
        annotations = EntityValidationPrompt.__annotations__
        
        assert 'template' in annotations
        assert 'config' in annotations
        
        # Check that they're ClassVar (at least in annotation)
        template_annotation = annotations['template']
        config_annotation = annotations['config']
        
        # Should be ClassVar types
        assert hasattr(template_annotation, '__origin__') or 'ClassVar' in str(template_annotation)
        assert hasattr(config_annotation, '__origin__') or 'ClassVar' in str(config_annotation)

    def test_prompt_template_instructions_clarity(self):
        """Test that template provides clear instructions."""
        template = EntityValidationPrompt.template
        
        # Should contain clear directive words
        assert any(word in template.lower() for word in ['validate', 'determine', 'return'])
        
        # Should explain what to do with entity
        assert 'entity' in template.lower()
        assert 'valid' in template.lower()
        
        # Should specify return format
        assert 'json' in template.lower()

    def test_prompt_template_variable_descriptions(self):
        """Test that template appropriately describes variable usage."""
        template = EntityValidationPrompt.template
        
        # Variables should be contextualized
        entity_id_context = template[template.find('{{entity_id}}') - 20:template.find('{{entity_id}}') + 50]
        assert 'Entity ID' in entity_id_context
        
        entity_type_context = template[template.find('{{entity_type}}') - 20:template.find('{{entity_type}}') + 50]
        assert 'Entity Type' in entity_type_context

    def test_prompt_expected_output_format(self):
        """Test that template specifies expected output format clearly."""
        template = EntityValidationPrompt.template
        
        # Should specify JSON format
        assert 'JSON object' in template
        
        # Should list expected fields
        expected_fields = ['is_valid', 'validation_message', 'suggested_alternatives', 'confidence']
        for field in expected_fields:
            assert field in template

    def test_prompt_confidence_field_specification(self):
        """Test that confidence field is properly specified."""
        template = EntityValidationPrompt.template
        
        # Should mention confidence score
        assert 'confidence' in template
        
        # Should indicate it's a number (0.95 example)
        assert '0.95' in template or '0.' in template

    def test_prompt_alternatives_field_specification(self):
        """Test that suggested_alternatives field is properly specified."""
        template = EntityValidationPrompt.template
        
        # Should specify it's a list
        assert 'suggested_alternatives' in template
        assert '[' in template and ']' in template
        
        # Should show list format
        list_example = template[template.find('suggested_alternatives'):template.find('suggested_alternatives') + 100]
        assert '"list"' in list_example or '["' in template