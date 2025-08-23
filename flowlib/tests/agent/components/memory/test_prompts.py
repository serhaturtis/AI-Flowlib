"""Comprehensive tests for agent memory prompts module."""

import pytest
from typing import List, ClassVar
from unittest.mock import Mock, patch
from pydantic import ValidationError

from flowlib.agent.components.memory.prompts import (
    MemoryFusionPrompt,
    FusedMemoryResult,
    KGQueryExtractionPrompt,
    ExtractedKGQueryTerms
)
from flowlib.resources.models.base import ResourceBase
from flowlib.resources.models.constants import ResourceType
from flowlib.resources.registry.registry import resource_registry


class TestMemoryFusionPrompt:
    """Test MemoryFusionPrompt class."""
    
    def test_memory_fusion_prompt_inheritance(self):
        """Test that MemoryFusionPrompt inherits from ResourceBase."""
        prompt = resource_registry.get("memory_fusion")
        assert isinstance(prompt, ResourceBase)
    
    def test_memory_fusion_prompt_decorator(self):
        """Test that the prompt decorator is applied."""
        # Check that the class has the expected resource name
        assert hasattr(MemoryFusionPrompt, '__resource_name__')
        # The decorator should set this
        assert MemoryFusionPrompt.__resource_name__ == "memory_fusion"
    
    def test_memory_fusion_prompt_template_exists(self):
        """Test that the template exists and is a string."""
        assert hasattr(MemoryFusionPrompt, 'template')
        assert isinstance(MemoryFusionPrompt.template, str)
        assert len(MemoryFusionPrompt.template.strip()) > 0
    
    def test_memory_fusion_prompt_template_content(self):
        """Test the content of the template."""
        template = MemoryFusionPrompt.template
        
        # Check for key components
        assert "Memory Fusion Assistant" in template
        assert "{{query}}" in template
        assert "{{vector_results}}" in template
        assert "{{knowledge_results}}" in template
        assert "{{working_results}}" in template
        assert "{{plugin_results}}" in template
        assert "relevant_items" in template
        assert "summary" in template
        assert "JSON" in template
    
    def test_memory_fusion_prompt_template_structure(self):
        """Test the structure of the template."""
        template = MemoryFusionPrompt.template
        
        # Check that sections are present
        sections = [
            "User Query:",
            "### Semantic Search Results",
            "### Knowledge Graph Results", 
            "### Working Memory Results",
            "### Knowledge Plugin Results",
            "### Synthesized Output"
        ]
        
        for section in sections:
            assert section in template
    
    def test_memory_fusion_prompt_template_variables(self):
        """Test that all expected template variables are present."""
        template = MemoryFusionPrompt.template
        
        expected_variables = [
            "{{query}}",
            "{{vector_results}}",
            "{{knowledge_results}}",
            "{{working_results}}",
            "{{plugin_results}}"
        ]
        
        for variable in expected_variables:
            assert variable in template
    
    def test_memory_fusion_prompt_creation(self):
        """Test creating MemoryFusionPrompt instance."""
        prompt = resource_registry.get("memory_fusion")
        assert prompt is not None
        assert hasattr(prompt, 'template')
    
    def test_memory_fusion_prompt_template_is_class_var(self):
        """Test that template is a ClassVar."""
        # The template should be the same across instances
        prompt1 = resource_registry.get("memory_fusion")
        prompt2 = resource_registry.get("memory_fusion")
        
        assert prompt1.template == prompt2.template
        assert prompt1.template is prompt2.template  # Same object reference
    
    def test_memory_fusion_prompt_template_formatting_instructions(self):
        """Test that template contains clear formatting instructions."""
        template = MemoryFusionPrompt.template
        
        # Check for JSON formatting instructions
        assert "JSON format" in template or "JSON object" in template
        assert "relevant_items" in template
        assert "List[str]" in template
        assert "summary" in template
    
    def test_memory_fusion_prompt_handles_no_results(self):
        """Test that template addresses handling of no results."""
        template = MemoryFusionPrompt.template
        
        # Should mention handling when no results are found
        assert "No relevant" in template or "no results" in template


class TestFusedMemoryResult:
    """Test FusedMemoryResult model."""
    
    def test_fused_memory_result_creation_minimal(self):
        """Test creating FusedMemoryResult with minimal required fields."""
        result = FusedMemoryResult(
            relevant_items=["item1", "item2"],
            summary="Test summary"
        )
        assert result.relevant_items == ["item1", "item2"]
        assert result.summary == "Test summary"
    
    def test_fused_memory_result_creation_empty_items(self):
        """Test creating FusedMemoryResult with empty items list."""
        result = FusedMemoryResult(
            relevant_items=[],
            summary="No relevant items found"
        )
        assert result.relevant_items == []
        assert result.summary == "No relevant items found"
    
    def test_fused_memory_result_creation_complex_items(self):
        """Test creating FusedMemoryResult with complex items."""
        complex_items = [
            "User prefers dark mode interface",
            "Previous project involved machine learning model training",
            "Frequently uses Python and JavaScript",
            "Has experience with database optimization",
            "Interested in cloud computing solutions"
        ]
        
        result = FusedMemoryResult(
            relevant_items=complex_items,
            summary="User profile shows strong technical background with preferences for modern development tools."
        )
        assert result.relevant_items == complex_items
        assert len(result.relevant_items) == 5
        assert "technical background" in result.summary
    
    def test_fused_memory_result_validation_relevant_items_required(self):
        """Test that relevant_items is required."""
        with pytest.raises(ValidationError) as exc_info:
            FusedMemoryResult(summary="Test summary")
        
        assert "relevant_items" in str(exc_info.value)
    
    def test_fused_memory_result_validation_summary_required(self):
        """Test that summary is required."""
        with pytest.raises(ValidationError) as exc_info:
            FusedMemoryResult(relevant_items=["item1"])
        
        assert "summary" in str(exc_info.value)
    
    def test_fused_memory_result_validation_relevant_items_list(self):
        """Test that relevant_items must be a list."""
        with pytest.raises(ValidationError) as exc_info:
            FusedMemoryResult(
                relevant_items="not a list",
                summary="Test summary"
            )
        
        assert "Input should be a valid list" in str(exc_info.value)
    
    def test_fused_memory_result_validation_summary_string(self):
        """Test that summary must be a string."""
        with pytest.raises(ValidationError) as exc_info:
            FusedMemoryResult(
                relevant_items=["item1"],
                summary=123
            )
        
        assert "Input should be a valid string" in str(exc_info.value)
    
    def test_fused_memory_result_validation_summary_non_empty(self):
        """Test that summary cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            FusedMemoryResult(
                relevant_items=["item1"],
                summary=""
            )
        
        assert "String should have at least 1 character" in str(exc_info.value)
    
    def test_fused_memory_result_items_with_various_content(self):
        """Test relevant_items with various content types."""
        items = [
            "Simple fact",
            "Complex fact with multiple details and context",
            "Fact with numbers: 42, 3.14, 100%",
            "Fact with special characters: @#$%^&*()",
            "Fact with unicode: 你好 world 🌍"
        ]
        
        result = FusedMemoryResult(
            relevant_items=items,
            summary="Mixed content types handled successfully"
        )
        assert result.relevant_items == items
        assert all(isinstance(item, str) for item in result.relevant_items)
    
    def test_fused_memory_result_field_descriptions(self):
        """Test that field descriptions are properly set."""
        fields = FusedMemoryResult.model_fields
        
        assert "relevant_items" in fields
        assert "synthesized from different memory sources" in fields["relevant_items"].description
        
        assert "summary" in fields
        assert "brief summary" in fields["summary"].description
        assert "acknowledging sources" in fields["summary"].description
    
    def test_fused_memory_result_serialization(self):
        """Test serialization capabilities."""
        result = FusedMemoryResult(
            relevant_items=["item1", "item2", "item3"],
            summary="Test serialization summary"
        )
        
        # Test model_dump
        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["relevant_items"] == ["item1", "item2", "item3"]
        assert data["summary"] == "Test serialization summary"
        
        # Test model_dump_json
        json_str = result.model_dump_json()
        assert isinstance(json_str, str)
        assert "item1" in json_str
        assert "Test serialization" in json_str
    
    def test_fused_memory_result_deserialization(self):
        """Test creating result from dictionary."""
        data = {
            "relevant_items": ["restored_item1", "restored_item2"],
            "summary": "Restored summary"
        }
        
        result = FusedMemoryResult(**data)
        assert result.relevant_items == ["restored_item1", "restored_item2"]
        assert result.summary == "Restored summary"


class TestKGQueryExtractionPrompt:
    """Test KGQueryExtractionPrompt class."""
    
    def test_kg_query_extraction_prompt_inheritance(self):
        """Test that KGQueryExtractionPrompt inherits from ResourceBase."""
        prompt = resource_registry.get("kg_query_extraction")
        assert isinstance(prompt, ResourceBase)
    
    def test_kg_query_extraction_prompt_decorator(self):
        """Test that the prompt decorator is applied."""
        assert hasattr(KGQueryExtractionPrompt, '__resource_name__')
        assert KGQueryExtractionPrompt.__resource_name__ == "kg_query_extraction"
    
    def test_kg_query_extraction_prompt_template_exists(self):
        """Test that the template exists and is a string."""
        assert hasattr(KGQueryExtractionPrompt, 'template')
        assert isinstance(KGQueryExtractionPrompt.template, str)
        assert len(KGQueryExtractionPrompt.template.strip()) > 0
    
    def test_kg_query_extraction_prompt_template_content(self):
        """Test the content of the template."""
        template = KGQueryExtractionPrompt.template
        
        # Check for key components
        assert "Knowledge Graph Query Assistant" in template
        assert "{{query}}" in template
        assert "{{context}}" in template
        assert "entities" in template
        assert "keywords" in template
        assert "JSON" in template
    
    def test_kg_query_extraction_prompt_template_structure(self):
        """Test the structure of the template."""
        template = KGQueryExtractionPrompt.template
        
        # Check that key sections are present
        sections = [
            "User Query:",
            "Context (Optional):",
            "Relevant Keywords/Entities"
        ]
        
        for section in sections:
            assert section in template
    
    def test_kg_query_extraction_prompt_template_variables(self):
        """Test that all expected template variables are present."""
        template = KGQueryExtractionPrompt.template
        
        expected_variables = [
            "{{query}}",
            "{{context}}"
        ]
        
        for variable in expected_variables:
            assert variable in template
    
    def test_kg_query_extraction_prompt_creation(self):
        """Test creating KGQueryExtractionPrompt instance."""
        prompt = resource_registry.get("kg_query_extraction")
        assert prompt is not None
        assert hasattr(prompt, 'template')
    
    def test_kg_query_extraction_prompt_template_is_class_var(self):
        """Test that template is a ClassVar."""
        prompt1 = resource_registry.get("kg_query_extraction")
        prompt2 = resource_registry.get("kg_query_extraction")
        
        assert prompt1.template == prompt2.template
        assert prompt1.template is prompt2.template
    
    def test_kg_query_extraction_prompt_instructions(self):
        """Test that template contains clear instructions."""
        template = KGQueryExtractionPrompt.template
        
        # Check for key instructions
        assert "nouns" in template or "proper nouns" in template
        assert "distinct items" in template or "specific" in template
        assert "empty list" in template  # Handling no results
        assert "JSON list" in template  # Output format
    
    def test_kg_query_extraction_prompt_guidance(self):
        """Test that template provides good guidance."""
        template = KGQueryExtractionPrompt.template
        
        # Should guide away from generic terms
        assert "Avoid" in template and ("generic" in template or "conversational" in template)
        # Should focus on searchable terms
        assert "searchable" in template or "search" in template


class TestExtractedKGQueryTerms:
    """Test ExtractedKGQueryTerms model."""
    
    def test_extracted_kg_query_terms_creation_minimal(self):
        """Test creating ExtractedKGQueryTerms with minimal fields."""
        terms = ExtractedKGQueryTerms(terms=["entity1", "concept2"])
        assert terms.terms == ["entity1", "concept2"]
    
    def test_extracted_kg_query_terms_creation_empty(self):
        """Test creating ExtractedKGQueryTerms with empty terms."""
        terms = ExtractedKGQueryTerms(terms=[])
        assert terms.terms == []
    
    def test_extracted_kg_query_terms_creation_complex(self):
        """Test creating ExtractedKGQueryTerms with complex terms."""
        complex_terms = [
            "Python programming",
            "machine learning",
            "neural networks",
            "TensorFlow",
            "data preprocessing",
            "model evaluation"
        ]
        
        terms = ExtractedKGQueryTerms(terms=complex_terms)
        assert terms.terms == complex_terms
        assert len(terms.terms) == 6
    
    def test_extracted_kg_query_terms_validation_terms_required(self):
        """Test that terms is required."""
        with pytest.raises(ValidationError) as exc_info:
            ExtractedKGQueryTerms()
        
        assert "terms" in str(exc_info.value)
    
    def test_extracted_kg_query_terms_validation_terms_list(self):
        """Test that terms must be a list."""
        with pytest.raises(ValidationError) as exc_info:
            ExtractedKGQueryTerms(terms="not a list")
        
        assert "Input should be a valid list" in str(exc_info.value)
    
    def test_extracted_kg_query_terms_validation_terms_strings(self):
        """Test that terms must be strings."""
        with pytest.raises(ValidationError) as exc_info:
            ExtractedKGQueryTerms(terms=[123, 456])
        
        assert "Input should be a valid string" in str(exc_info.value)
    
    def test_extracted_kg_query_terms_mixed_content(self):
        """Test terms with various string content."""
        varied_terms = [
            "simple_term",
            "Term with spaces",
            "UPPERCASE_TERM",
            "term_with_numbers_123",
            "term-with-dashes",
            "term.with.dots",
            "unicode_term_😀"
        ]
        
        terms = ExtractedKGQueryTerms(terms=varied_terms)
        assert terms.terms == varied_terms
        assert all(isinstance(term, str) for term in terms.terms)
    
    def test_extracted_kg_query_terms_field_description(self):
        """Test that field description is properly set."""
        fields = ExtractedKGQueryTerms.model_fields
        
        assert "terms" in fields
        assert "extracted keywords" in fields["terms"].description
        assert "entity names" in fields["terms"].description
        assert "KG search" in fields["terms"].description
    
    def test_extracted_kg_query_terms_serialization(self):
        """Test serialization capabilities."""
        terms = ExtractedKGQueryTerms(
            terms=["serialization", "test", "terms"]
        )
        
        # Test model_dump
        data = terms.model_dump()
        assert isinstance(data, dict)
        assert data["terms"] == ["serialization", "test", "terms"]
        
        # Test model_dump_json
        json_str = terms.model_dump_json()
        assert isinstance(json_str, str)
        assert "serialization" in json_str
    
    def test_extracted_kg_query_terms_deserialization(self):
        """Test creating terms from dictionary."""
        data = {
            "terms": ["restored", "terms", "list"]
        }
        
        terms = ExtractedKGQueryTerms(**data)
        assert terms.terms == ["restored", "terms", "list"]


class TestMemoryPromptsIntegration:
    """Test integration between memory prompt models."""
    
    def test_memory_fusion_workflow(self):
        """Test a complete memory fusion workflow."""
        # Simulate the workflow of using both prompts together
        
        # 1. First extract KG query terms
        kg_terms = ExtractedKGQueryTerms(
            terms=["Python", "machine learning", "TensorFlow"]
        )
        
        # 2. Then create fusion result
        fusion_result = FusedMemoryResult(
            relevant_items=[
                "User has experience with Python programming",
                "Previously worked on machine learning projects",
                "Familiar with TensorFlow framework",
                "Prefers hands-on coding approach"
            ],
            summary="User profile shows strong background in Python ML development with TensorFlow experience."
        )
        
        # Verify the workflow makes sense
        assert len(kg_terms.terms) == 3
        assert len(fusion_result.relevant_items) == 4
        assert "Python" in str(kg_terms.terms)
        assert "Python" in fusion_result.relevant_items[0]
    
    def test_prompt_templates_consistency(self):
        """Test that prompt templates are consistent."""
        memory_fusion_template = MemoryFusionPrompt.template
        kg_extraction_template = KGQueryExtractionPrompt.template
        
        # Both should reference query
        assert "{{query}}" in memory_fusion_template
        assert "{{query}}" in kg_extraction_template
        
        # Both should mention JSON output
        assert "JSON" in memory_fusion_template
        assert "JSON" in kg_extraction_template
        
        # Both should be well-formed templates
        assert len(memory_fusion_template.strip()) > 100
        assert len(kg_extraction_template.strip()) > 100
    
    def test_prompt_model_compatibility(self):
        """Test that prompt models are compatible with their expected outputs."""
        # The FusedMemoryResult should match what MemoryFusionPrompt expects
        fusion_fields = FusedMemoryResult.model_fields
        assert "relevant_items" in fusion_fields
        assert "summary" in fusion_fields
        
        # The ExtractedKGQueryTerms should match what KGQueryExtractionPrompt expects
        kg_fields = ExtractedKGQueryTerms.model_fields
        assert "terms" in kg_fields
        
        # Field types should be appropriate
        assert fusion_fields["relevant_items"].annotation == List[str]
        assert fusion_fields["summary"].annotation == str
        assert kg_fields["terms"].annotation == List[str]
    
    def test_empty_results_handling(self):
        """Test handling of empty results."""
        # Empty KG terms (no entities found)
        empty_kg_terms = ExtractedKGQueryTerms(terms=[])
        assert empty_kg_terms.terms == []
        
        # Empty fusion result (no relevant items)
        empty_fusion = FusedMemoryResult(
            relevant_items=[],
            summary="No relevant information found across all memory sources."
        )
        assert empty_fusion.relevant_items == []
        assert "No relevant" in empty_fusion.summary
    
    def test_prompts_are_resources(self):
        """Test that both prompts are properly configured as resources."""
        # Retrieve prompts from resource registry
        memory_fusion = resource_registry.get("memory_fusion")
        kg_extraction = resource_registry.get("kg_query_extraction")
        
        assert isinstance(memory_fusion, ResourceBase)
        assert isinstance(kg_extraction, ResourceBase)
        
        # Both should have resource names from decorators
        assert hasattr(MemoryFusionPrompt, '__resource_name__')
        assert hasattr(KGQueryExtractionPrompt, '__resource_name__')
        
        assert MemoryFusionPrompt.__resource_name__ == "memory_fusion"
        assert KGQueryExtractionPrompt.__resource_name__ == "kg_query_extraction"
    
    def test_template_variable_coverage(self):
        """Test that all template variables are covered."""
        # Memory fusion template should cover all memory types
        memory_template = MemoryFusionPrompt.template
        memory_types = [
            "{{vector_results}}",
            "{{knowledge_results}}",
            "{{working_results}}",
            "{{plugin_results}}"
        ]
        
        for memory_type in memory_types:
            assert memory_type in memory_template
        
        # KG extraction should cover query and context
        kg_template = KGQueryExtractionPrompt.template
        assert "{{query}}" in kg_template
        assert "{{context}}" in kg_template