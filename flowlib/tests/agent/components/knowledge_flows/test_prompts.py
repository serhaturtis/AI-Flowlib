"""Tests for agent knowledge flows prompts."""

import pytest
from typing import ClassVar

from flowlib.agent.components.knowledge_flows.prompts import (
    KnowledgeExtractionPrompt,
    DomainDetectionPrompt,
    KnowledgeSynthesisPrompt
)
from flowlib.resources.models.base import ResourceBase
from flowlib.resources.models.constants import ResourceType


class TestKnowledgeExtractionPrompt:
    """Test KnowledgeExtractionPrompt class."""
    
    def test_knowledge_extraction_prompt_creation(self):
        """Test creating KnowledgeExtractionPrompt instance."""
        prompt = KnowledgeExtractionPrompt(name="test_extraction", type=ResourceType.PROMPT)
        
        assert prompt.name == "test_extraction"
        assert prompt.type == ResourceType.PROMPT
        assert isinstance(prompt, ResourceBase)
    
    def test_knowledge_extraction_prompt_template_exists(self):
        """Test that template exists and is properly defined."""
        assert hasattr(KnowledgeExtractionPrompt, 'template')
        assert isinstance(KnowledgeExtractionPrompt.template, str)
        assert len(KnowledgeExtractionPrompt.template.strip()) > 0
    
    def test_knowledge_extraction_prompt_template_content(self):
        """Test that template contains expected content."""
        template = KnowledgeExtractionPrompt.template
        
        # Check for expected variables
        expected_variables = [
            '{{text}}',
            '{{context}}',
            '{{domain_hint}}',
            '{{extract_personal}}'
        ]
        
        for var in expected_variables:
            assert var in template, f"Template missing variable: {var}"
    
    def test_knowledge_extraction_prompt_knowledge_types(self):
        """Test that template includes all knowledge types."""
        template = KnowledgeExtractionPrompt.template
        
        knowledge_types = [
            'FACTUAL',
            'PROCEDURAL', 
            'CONCEPTUAL',
            'PERSONAL',
            'TECHNICAL'
        ]
        
        for ktype in knowledge_types:
            assert ktype in template, f"Template missing knowledge type: {ktype}"
    
    def test_knowledge_extraction_prompt_knowledge_type_descriptions(self):
        """Test that template includes descriptions for knowledge types."""
        template = KnowledgeExtractionPrompt.template
        
        type_descriptions = [
            'Concrete facts',
            'How-to information',
            'Ideas, theories',
            'User preferences',
            'Domain-specific technical'
        ]
        
        for description in type_descriptions:
            assert description in template, f"Template missing type description: {description}"
    
    def test_knowledge_extraction_prompt_extraction_criteria(self):
        """Test that template includes extraction criteria."""
        template = KnowledgeExtractionPrompt.template
        
        criteria = [
            'Confidence level',
            'Key entities',
            'metadata',
            'useful for future reference',
            'trivial conversational'
        ]
        
        for criterion in criteria:
            assert criterion in template, f"Template missing criterion: {criterion}"
    
    def test_knowledge_extraction_prompt_json_structure(self):
        """Test that template includes proper JSON structure."""
        template = KnowledgeExtractionPrompt.template
        
        json_elements = [
            '"extracted_knowledge"',
            '"content"',
            '"knowledge_type"',
            '"domain"',
            '"confidence"',
            '"source_context"',
            '"entities"',
            '"metadata"',
            '"processing_notes"',
            '"domains_detected"'
        ]
        
        for element in json_elements:
            assert element in template, f"Template missing JSON element: {element}"
    
    def test_knowledge_extraction_prompt_output_format(self):
        """Test that template specifies JSON output format."""
        template = KnowledgeExtractionPrompt.template
        
        assert 'JSON' in template
        assert '{' in template and '}' in template
        assert 'factual|procedural|conceptual|personal|technical' in template


class TestDomainDetectionPrompt:
    """Test DomainDetectionPrompt class."""
    
    def test_domain_detection_prompt_creation(self):
        """Test creating DomainDetectionPrompt instance."""
        prompt = DomainDetectionPrompt(name="test_domain", type=ResourceType.PROMPT)
        
        assert prompt.name == "test_domain"
        assert prompt.type == ResourceType.PROMPT
        assert isinstance(prompt, ResourceBase)
    
    def test_domain_detection_prompt_template_exists(self):
        """Test that template exists and is properly defined."""
        assert hasattr(DomainDetectionPrompt, 'template')
        assert isinstance(DomainDetectionPrompt.template, str)
        assert len(DomainDetectionPrompt.template.strip()) > 0
    
    def test_domain_detection_prompt_template_content(self):
        """Test that template contains expected content."""
        template = DomainDetectionPrompt.template
        
        # Check for expected variable
        assert '{{text}}' in template
    
    def test_domain_detection_prompt_domain_examples(self):
        """Test that template includes domain examples."""
        template = DomainDetectionPrompt.template
        
        domain_examples = [
            'chemistry',
            'physics',
            'biology',
            'technology',
            'programming',
            'business',
            'finance',
            'personal',
            'education'
        ]
        
        for domain in domain_examples:
            assert domain in template, f"Template missing domain example: {domain}"
    
    def test_domain_detection_prompt_output_format(self):
        """Test that template specifies correct output format."""
        template = DomainDetectionPrompt.template
        
        assert 'JSON list' in template
        assert '["domain1", "domain2", "domain3"]' in template
    
    def test_domain_detection_prompt_guidance(self):
        """Test that template includes detection guidance."""
        template = DomainDetectionPrompt.template
        
        guidance_keywords = [
            'top 3',
            'clearly present',
            'fewer than 3',
            'relevant'
        ]
        
        for keyword in guidance_keywords:
            assert keyword in template, f"Template missing guidance keyword: {keyword}"


class TestKnowledgeSynthesisPrompt:
    """Test KnowledgeSynthesisPrompt class."""
    
    def test_knowledge_synthesis_prompt_creation(self):
        """Test creating KnowledgeSynthesisPrompt instance."""
        prompt = KnowledgeSynthesisPrompt(name="test_synthesis", type=ResourceType.PROMPT)
        
        assert prompt.name == "test_synthesis"
        assert prompt.type == ResourceType.PROMPT
        assert isinstance(prompt, ResourceBase)
    
    def test_knowledge_synthesis_prompt_template_exists(self):
        """Test that template exists and is properly defined."""
        assert hasattr(KnowledgeSynthesisPrompt, 'template')
        assert isinstance(KnowledgeSynthesisPrompt.template, str)
        assert len(KnowledgeSynthesisPrompt.template.strip()) > 0
    
    def test_knowledge_synthesis_prompt_template_content(self):
        """Test that template contains expected content."""
        template = KnowledgeSynthesisPrompt.template
        
        # Check for expected variables
        expected_variables = [
            '{{query}}',
            '{{knowledge_items}}',
            '{{sources_searched}}'
        ]
        
        for var in expected_variables:
            assert var in template, f"Template missing variable: {var}"
    
    def test_knowledge_synthesis_prompt_synthesis_requirements(self):
        """Test that template includes synthesis requirements."""
        template = KnowledgeSynthesisPrompt.template
        
        requirements = [
            'Direct answers',
            'Additional relevant context',
            'Source attribution',
            'Confidence assessment'
        ]
        
        for requirement in requirements:
            assert requirement in template, f"Template missing requirement: {requirement}"
    
    def test_knowledge_synthesis_prompt_json_structure(self):
        """Test that template includes proper JSON structure."""
        template = KnowledgeSynthesisPrompt.template
        
        json_elements = [
            '"search_summary"',
            '"synthesized_answer"',
            '"confidence"',
            '"sources_used"',
            '"gaps"'
        ]
        
        for element in json_elements:
            assert element in template, f"Template missing JSON element: {element}"
    
    def test_knowledge_synthesis_prompt_output_guidance(self):
        """Test that template provides output guidance."""
        template = KnowledgeSynthesisPrompt.template
        
        guidance_keywords = [
            'comprehensive answer',
            'Brief summary',
            'information gaps',
            'limitations'
        ]
        
        for keyword in guidance_keywords:
            assert keyword in template, f"Template missing guidance keyword: {keyword}"


class TestPromptDecorators:
    """Test that prompts are properly decorated."""
    
    def test_knowledge_extraction_prompt_decorated(self):
        """Test that KnowledgeExtractionPrompt is properly decorated."""
        assert hasattr(KnowledgeExtractionPrompt, '__resource_name__')
        assert hasattr(KnowledgeExtractionPrompt, '__resource_type__')
        
        assert KnowledgeExtractionPrompt.__resource_name__ == "knowledge_extraction"
        assert KnowledgeExtractionPrompt.__resource_type__ == ResourceType.PROMPT_CONFIG
    
    def test_domain_detection_prompt_decorated(self):
        """Test that DomainDetectionPrompt is properly decorated."""
        assert hasattr(DomainDetectionPrompt, '__resource_name__')
        assert hasattr(DomainDetectionPrompt, '__resource_type__')
        
        assert DomainDetectionPrompt.__resource_name__ == "domain_detection"
        assert DomainDetectionPrompt.__resource_type__ == ResourceType.PROMPT_CONFIG
    
    def test_knowledge_synthesis_prompt_decorated(self):
        """Test that KnowledgeSynthesisPrompt is properly decorated."""
        assert hasattr(KnowledgeSynthesisPrompt, '__resource_name__')
        assert hasattr(KnowledgeSynthesisPrompt, '__resource_type__')
        
        assert KnowledgeSynthesisPrompt.__resource_name__ == "knowledge_synthesis"
        assert KnowledgeSynthesisPrompt.__resource_type__ == ResourceType.PROMPT_CONFIG


class TestPromptIntegration:
    """Test integration aspects of the prompts."""
    
    def test_all_prompts_inherit_from_resource_base(self):
        """Test that all prompt classes inherit from ResourceBase."""
        prompt_classes = [
            KnowledgeExtractionPrompt,
            DomainDetectionPrompt,
            KnowledgeSynthesisPrompt
        ]
        
        for prompt_class in prompt_classes:
            assert issubclass(prompt_class, ResourceBase)
    
    def test_prompt_template_format_consistency(self):
        """Test that all prompts follow consistent template format."""
        prompt_classes = [
            KnowledgeExtractionPrompt,
            DomainDetectionPrompt,
            KnowledgeSynthesisPrompt
        ]
        
        for prompt_class in prompt_classes:
            template = prompt_class.template
            
            # Should not be empty
            assert len(template.strip()) > 0
            
            # Should contain at least one variable
            assert '{{' in template and '}}' in template
            
            # Should not contain obvious syntax errors
            assert template.count('{{') == template.count('}}')
    
    def test_prompt_variable_format(self):
        """Test that prompt variables follow expected format."""
        prompt_classes = [
            KnowledgeExtractionPrompt,
            DomainDetectionPrompt,
            KnowledgeSynthesisPrompt
        ]
        
        for prompt_class in prompt_classes:
            template = prompt_class.template
            
            # Extract variables from template
            import re
            variables = re.findall(r'\{\{(\w+)\}\}', template)
            
            # Variables should be lowercase with underscores
            for var in variables:
                assert var.islower() or '_' in var, f"Variable {var} should be lowercase/snake_case"
    
    def test_knowledge_workflow_prompt_consistency(self):
        """Test that prompts are consistent for knowledge workflow."""
        # Extraction prompt should mention knowledge types
        extraction_template = KnowledgeExtractionPrompt.template
        
        # Synthesis prompt should handle multiple sources
        synthesis_template = KnowledgeSynthesisPrompt.template
        
        # Both should handle confidence
        assert 'confidence' in extraction_template
        assert 'confidence' in synthesis_template
        
        # Both should handle domains
        assert 'domain' in extraction_template
        domain_template = DomainDetectionPrompt.template
        assert 'domain' in domain_template


class TestPromptInstantiation:
    """Test that prompts can be properly instantiated."""
    
    def test_create_all_prompt_instances(self):
        """Test creating instances of all prompt classes."""
        prompt_data = [
            (KnowledgeExtractionPrompt, "knowledge_extraction"),
            (DomainDetectionPrompt, "domain_detection"),
            (KnowledgeSynthesisPrompt, "knowledge_synthesis")
        ]
        
        for prompt_class, name in prompt_data:
            # Should be able to create instance
            instance = prompt_class(name=name, type=ResourceType.PROMPT)
            
            assert instance.name == name
            assert instance.type == ResourceType.PROMPT
            assert hasattr(instance, 'template')
    
    def test_prompt_instances_have_template_access(self):
        """Test that prompt instances can access their templates."""
        prompt = KnowledgeExtractionPrompt(name="test", type=ResourceType.PROMPT)
        
        # Should be able to access template from instance
        assert hasattr(prompt, 'template')
        assert prompt.template == KnowledgeExtractionPrompt.template
    
    def test_prompt_template_immutability(self):
        """Test that templates are class variables and not modified per instance."""
        prompt1 = KnowledgeExtractionPrompt(name="test1", type=ResourceType.PROMPT)
        prompt2 = KnowledgeExtractionPrompt(name="test2", type=ResourceType.PROMPT)
        
        # Both instances should have the same template
        assert prompt1.template == prompt2.template
        assert prompt1.template is prompt2.template


class TestPromptContent:
    """Test specific content aspects of the prompts."""
    
    def test_knowledge_extraction_prompt_instructions(self):
        """Test that extraction prompt provides clear instructions."""
        template = KnowledgeExtractionPrompt.template
        
        instructions = [
            'Knowledge Extraction Assistant',
            'identify and extract',
            'useful knowledge',
            'categories'
        ]
        
        for instruction in instructions:
            assert instruction in template, f"Template missing instruction: {instruction}"
    
    def test_domain_detection_prompt_specificity(self):
        """Test that domain detection prompt is specific about requirements."""
        template = DomainDetectionPrompt.template
        
        specificity_keywords = [
            'primary knowledge domains',
            'clearly present',
            'top 3',
            'relevant'
        ]
        
        for keyword in specificity_keywords:
            assert keyword in template, f"Template missing specificity keyword: {keyword}"
    
    def test_knowledge_synthesis_prompt_comprehensiveness(self):
        """Test that synthesis prompt covers comprehensive requirements."""
        template = KnowledgeSynthesisPrompt.template
        
        comprehensive_elements = [
            'Combine and summarize',
            'comprehensive answer',
            'Direct answers',
            'Additional relevant context',
            'Source attribution',
            'Confidence assessment'
        ]
        
        for element in comprehensive_elements:
            assert element in template, f"Template missing comprehensive element: {element}"
    
    def test_extraction_prompt_knowledge_filtering(self):
        """Test that extraction prompt includes knowledge filtering guidance."""
        template = KnowledgeExtractionPrompt.template
        
        filtering_guidance = [
            'useful for future reference',
            'Avoid extracting trivial',
            'conversational elements'
        ]
        
        for guidance in filtering_guidance:
            assert guidance in template, f"Template missing filtering guidance: {guidance}"
    
    def test_synthesis_prompt_gap_identification(self):
        """Test that synthesis prompt includes gap identification."""
        template = KnowledgeSynthesisPrompt.template
        
        assert '"gaps"' in template
        assert 'information gaps' in template or 'limitations' in template
    
    def test_confidence_handling_consistency(self):
        """Test that confidence is handled consistently across prompts."""
        extraction_template = KnowledgeExtractionPrompt.template
        synthesis_template = KnowledgeSynthesisPrompt.template
        
        # Both should mention confidence
        assert 'confidence' in extraction_template
        assert 'confidence' in synthesis_template
        
        # Should specify confidence range in extraction
        assert '0.0 to 1.0' in extraction_template or '(0.0-1.0)' in extraction_template