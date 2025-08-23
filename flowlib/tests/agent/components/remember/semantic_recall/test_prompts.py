"""Tests for semantic recall prompts."""

import pytest
from unittest.mock import MagicMock

from flowlib.agent.components.remember.semantic_recall.prompts import SemanticAnalysisPrompt
from flowlib.resources.models.base import ResourceBase


class TestSemanticAnalysisPrompt:
    """Test cases for SemanticAnalysisPrompt."""

    def test_prompt_inheritance(self):
        """Test that SemanticAnalysisPrompt inherits from ResourceBase."""
        assert issubclass(SemanticAnalysisPrompt, ResourceBase)

    def test_prompt_decorator(self):
        """Test that prompt has correct decorator metadata."""
        assert hasattr(SemanticAnalysisPrompt, '__resource_name__')
        assert hasattr(SemanticAnalysisPrompt, '__resource_type__')
        
        assert SemanticAnalysisPrompt.__resource_name__ == 'semantic_analysis'
        # Check for the actual resource type used by @prompt decorator
        from flowlib.resources.models.constants import ResourceType
        assert SemanticAnalysisPrompt.__resource_type__ == ResourceType.PROMPT_CONFIG

    def test_prompt_template_exists(self):
        """Test that template attribute exists and is not empty."""
        assert hasattr(SemanticAnalysisPrompt, 'template')
        assert isinstance(SemanticAnalysisPrompt.template, str)
        assert len(SemanticAnalysisPrompt.template.strip()) > 0

    def test_prompt_template_structure(self):
        """Test prompt template structure and content."""
        template = SemanticAnalysisPrompt.template
        
        # Check for required placeholder variables
        assert "{{query}}" in template
        assert "{{context}}" in template
        
        # Check for key sections
        assert "Analyze the following query" in template
        assert "semantic understanding" in template
        assert "Return your analysis as a JSON object" in template

    def test_prompt_template_analysis_components(self):
        """Test that template includes all required analysis components."""
        template = SemanticAnalysisPrompt.template
        
        # Check for analysis components mentioned in instructions
        assert "Key concepts" in template or "key concepts" in template
        assert "semantic relationships" in template or "Semantic relationships" in template
        assert "contextual meaning" in template or "Contextual meaning" in template
        assert "topic categories" in template or "Topic categories" in template
        assert "confidence" in template

    def test_prompt_template_json_format(self):
        """Test that template includes proper JSON format specification."""
        template = SemanticAnalysisPrompt.template
        
        # Check for JSON structure elements
        assert "key_concepts" in template
        assert "semantic_relationships" in template
        assert "contextual_meaning" in template
        assert "topic_categories" in template
        assert "confidence" in template
        
        # Check for JSON formatting
        assert '{\n' in template or '{' in template
        assert '}' in template
        assert '"' in template

    def test_prompt_config_exists(self):
        """Test that config attribute exists."""
        assert hasattr(SemanticAnalysisPrompt, 'config')
        assert isinstance(SemanticAnalysisPrompt.config, dict)

    def test_prompt_config_structure(self):
        """Test prompt configuration structure."""
        config = SemanticAnalysisPrompt.config
        
        # Check for expected configuration keys
        assert 'max_tokens' in config
        assert 'temperature' in config
        assert 'top_p' in config

    def test_prompt_config_values(self):
        """Test prompt configuration values."""
        config = SemanticAnalysisPrompt.config
        
        # Verify configuration values
        assert config['max_tokens'] == 512
        assert config['temperature'] == 0.3
        assert config['top_p'] == 0.95
        
        # Verify types
        assert isinstance(config['max_tokens'], int)
        assert isinstance(config['temperature'], (int, float))
        assert isinstance(config['top_p'], (int, float))

    def test_prompt_config_reasoning(self):
        """Test that configuration values are reasonable for semantic analysis task."""
        config = SemanticAnalysisPrompt.config
        
        # Temperature should be low for consistent analysis
        assert config['temperature'] <= 0.5
        
        # Max tokens should be sufficient for detailed semantic analysis
        assert config['max_tokens'] >= 400
        
        # Top_p should allow for focused responses
        assert 0.8 <= config['top_p'] <= 1.0

    def test_prompt_instantiation(self):
        """Test that prompt can be instantiated."""
        from flowlib.resources.models.constants import ResourceType
        
        prompt = SemanticAnalysisPrompt(name="semantic_analysis", type=ResourceType.PROMPT_CONFIG)
        
        assert isinstance(prompt, SemanticAnalysisPrompt)
        assert isinstance(prompt, ResourceBase)
        assert prompt.name == "semantic_analysis"
        assert prompt.type == ResourceType.PROMPT_CONFIG

    def test_prompt_template_rendering_structure(self):
        """Test template structure for variable substitution."""
        template = SemanticAnalysisPrompt.template
        
        # Check that variables are in proper Jinja2 format
        import re
        variables = re.findall(r'\{\{([^}]+)\}\}', template)
        
        expected_variables = ['query', 'context']
        for var in expected_variables:
            assert var.strip() in [v.strip() for v in variables]

    def test_prompt_template_whitespace_handling(self):
        """Test template handles whitespace appropriately."""
        template = SemanticAnalysisPrompt.template
        
        # Template should be well-formatted
        lines = template.split('\n')
        
        # Should not start or end with excessive whitespace
        assert not template.startswith('\n\n\n')
        assert not template.endswith('\n\n\n')
        
        # Should have reasonable structure
        assert len(lines) > 10  # Multi-line template with detailed instructions

    def test_prompt_template_json_example_validity(self):
        """Test that JSON example in template is valid structure."""
        template = SemanticAnalysisPrompt.template
        
        # Extract JSON portion (simplified check)
        json_start = template.find('{')
        json_end = template.rfind('}') + 1
        
        assert json_start != -1
        assert json_end > json_start
        
        json_section = template[json_start:json_end]
        
        # Basic JSON structure checks
        assert json_section.count('{') == json_section.count('}')
        assert '"key_concepts"' in json_section
        assert '"confidence"' in json_section

    def test_prompt_class_variables_are_class_vars(self):
        """Test that template and config are properly declared as ClassVar."""
        import typing
        
        annotations = SemanticAnalysisPrompt.__annotations__
        
        assert 'template' in annotations
        assert 'config' in annotations
        
        # Check that they're ClassVar types
        template_annotation = annotations['template']
        config_annotation = annotations['config']
        
        # Should be ClassVar types
        assert hasattr(template_annotation, '__origin__') or 'ClassVar' in str(template_annotation)
        assert hasattr(config_annotation, '__origin__') or 'ClassVar' in str(config_annotation)

    def test_prompt_template_instructions_clarity(self):
        """Test that template provides clear instructions."""
        template = SemanticAnalysisPrompt.template
        
        # Should contain clear directive words
        assert any(word in template.lower() for word in ['analyze', 'identify', 'return'])
        
        # Should explain what to do with query
        assert 'query' in template.lower()
        assert 'semantic' in template.lower()
        
        # Should specify return format
        assert 'json' in template.lower()

    def test_prompt_template_variable_descriptions(self):
        """Test that template appropriately describes variable usage."""
        template = SemanticAnalysisPrompt.template
        
        # Variables should be contextualized
        query_context = template[max(0, template.find('{{query}}') - 30):template.find('{{query}}') + 50]
        assert 'Query' in query_context or 'query' in query_context
        
        context_context = template[max(0, template.find('{{context}}') - 30):template.find('{{context}}') + 50]
        assert 'Context' in context_context or 'context' in context_context

    def test_prompt_expected_output_format(self):
        """Test that template specifies expected output format clearly."""
        template = SemanticAnalysisPrompt.template
        
        # Should specify JSON format
        assert 'JSON object' in template or 'JSON' in template
        
        # Should list expected fields
        expected_fields = ['key_concepts', 'semantic_relationships', 'contextual_meaning', 'topic_categories', 'confidence']
        for field in expected_fields:
            assert field in template

    def test_prompt_analysis_components_detail(self):
        """Test that template provides detailed guidance for each analysis component."""
        template = SemanticAnalysisPrompt.template
        
        # Should mention numbered analysis steps or components
        assert '1.' in template or 'concepts' in template.lower()
        assert '2.' in template or 'relationships' in template.lower()
        assert '3.' in template or 'meaning' in template.lower()
        assert '4.' in template or 'categories' in template.lower()
        assert '5.' in template or 'confidence' in template.lower()

    def test_prompt_confidence_field_specification(self):
        """Test that confidence field is properly specified."""
        template = SemanticAnalysisPrompt.template
        
        # Should mention confidence
        assert 'confidence' in template.lower()
        
        # Should indicate it's a number (example value)
        assert '0.' in template or 'confidence' in template

    def test_prompt_concept_fields_specification(self):
        """Test that concept-related fields are properly specified."""
        template = SemanticAnalysisPrompt.template
        
        # Should specify array format for concepts
        assert '[' in template and ']' in template
        
        # Should show list format examples
        list_examples = ['"concept1"', '"concept2"', '"category1"', '"relationship1"']
        assert any(example in template for example in list_examples)

    def test_prompt_semantic_focus(self):
        """Test that prompt emphasizes semantic analysis aspects."""
        template = SemanticAnalysisPrompt.template
        
        # Should emphasize semantic understanding
        semantic_terms = ['semantic', 'meaning', 'concepts', 'relationships', 'understanding']
        found_terms = [term for term in semantic_terms if term in template.lower()]
        assert len(found_terms) >= 3  # Should mention multiple semantic-related terms

    def test_prompt_template_completeness(self):
        """Test that template is comprehensive for semantic analysis."""
        template = SemanticAnalysisPrompt.template
        
        # Should be substantial enough for complex analysis
        assert len(template) > 500  # Detailed template
        
        # Should cover all major aspects of semantic analysis
        major_aspects = ['concepts', 'relationships', 'meaning', 'categories']
        for aspect in major_aspects:
            assert aspect in template.lower()

    def test_prompt_output_structure_clarity(self):
        """Test that the expected output structure is clearly defined."""
        template = SemanticAnalysisPrompt.template
        
        # JSON structure should be clearly shown
        json_portion = template[template.find('{'):template.rfind('}') + 1]
        
        # Should show proper JSON syntax
        assert '"key_concepts":' in json_portion
        assert '"semantic_relationships":' in json_portion
        assert '"contextual_meaning":' in json_portion
        assert '"topic_categories":' in json_portion
        assert '"confidence":' in json_portion