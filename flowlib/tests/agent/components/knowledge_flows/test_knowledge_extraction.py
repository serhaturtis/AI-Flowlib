"""Tests for knowledge extraction flow."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import logging

from flowlib.agent.components.knowledge_flows.knowledge_extraction import KnowledgeExtractionFlow
from flowlib.agent.components.knowledge_flows.models import (
    KnowledgeExtractionInput,
    KnowledgeExtractionOutput,
    ExtractedKnowledge,
    KnowledgeType
)
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.resources.models.constants import ResourceType


class TestKnowledgeExtractionFlow:
    """Test KnowledgeExtractionFlow class."""
    
    def test_knowledge_extraction_flow_decoration(self):
        """Test that KnowledgeExtractionFlow is properly decorated."""
        # Check flow decorator attributes
        assert hasattr(KnowledgeExtractionFlow, '__flow_name__')
        assert hasattr(KnowledgeExtractionFlow, '__flow_description__')
        assert hasattr(KnowledgeExtractionFlow, '__is_infrastructure__')
        
        assert KnowledgeExtractionFlow.__flow_name__ == "knowledge-extraction"
        assert "Extract knowledge from conversations" in KnowledgeExtractionFlow.__flow_description__
        assert KnowledgeExtractionFlow.__is_infrastructure__ is False
    
    def test_knowledge_extraction_flow_pipeline_decoration(self):
        """Test that run_pipeline method is properly decorated."""
        # Check that run_pipeline has pipeline decoration
        assert hasattr(KnowledgeExtractionFlow.run_pipeline, '__pipeline_input_model__')
        assert hasattr(KnowledgeExtractionFlow.run_pipeline, '__pipeline_output_model__')
        
        assert KnowledgeExtractionFlow.run_pipeline.__pipeline_input_model__ == KnowledgeExtractionInput
        assert KnowledgeExtractionFlow.run_pipeline.__pipeline_output_model__ == KnowledgeExtractionOutput
    
    def test_knowledge_extraction_flow_instantiation(self):
        """Test creating KnowledgeExtractionFlow instance."""
        flow = KnowledgeExtractionFlow("knowledge-extraction")
        
        assert isinstance(flow, KnowledgeExtractionFlow)
        assert hasattr(flow, 'run_pipeline')
        assert hasattr(flow, '_enhance_knowledge')
        assert hasattr(flow, '_detect_domains')
    
    @pytest.mark.asyncio
    async def test_run_pipeline_successful_extraction(self):
        """Test successful knowledge extraction."""
        flow = KnowledgeExtractionFlow("knowledge-extraction")
        
        input_data = KnowledgeExtractionInput(
            text="Python is a high-level programming language known for its simplicity and readability.",
            context="Programming language discussion",
            domain_hint="technology",
            extract_personal=False
        )
        
        with patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.provider_registry') as mock_provider_registry, \
             patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.resource_registry') as mock_resource_registry, \
             patch.object(flow, '_detect_domains', new_callable=AsyncMock, return_value=["programming"]) as mock_detect_domains:
            
            # Mock LLM provider
            mock_llm = AsyncMock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            
            # Mock prompt resource
            mock_prompt = MagicMock()
            mock_resource_registry.get.return_value = mock_prompt
            
            # Mock LLM response
            mock_extraction_result = KnowledgeExtractionOutput(
                extracted_knowledge=[
                    ExtractedKnowledge(
                        content="Python is a high-level programming language",
                        knowledge_type=KnowledgeType.TECHNICAL,
                        domain="programming",
                        confidence=0.95,
                        source_context="Programming language discussion",
                        entities=["Python", "programming language"],
                        metadata={"language_feature": "high-level"}
                    )
                ],
                processing_notes="Extracted technical knowledge about Python",
                domains_detected=["programming"]
            )
            mock_llm.generate_structured.return_value = mock_extraction_result
            
            result = await flow.run_pipeline(input_data)
            
            assert isinstance(result, KnowledgeExtractionOutput)
            assert len(result.extracted_knowledge) == 1
            assert result.extracted_knowledge[0].content == "Python is a high-level programming language"
            assert result.extracted_knowledge[0].knowledge_type == KnowledgeType.TECHNICAL
            assert "programming" in result.domains_detected
            
            # Verify calls
            mock_provider_registry.get_by_config.assert_called_once_with("default-llm")
            mock_resource_registry.get.assert_called_with(
                name="knowledge_extraction",
                resource_type=ResourceType.PROMPT
            )
            mock_llm.generate_structured.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_pipeline_no_llm_provider(self):
        """Test handling when LLM provider is not available."""
        flow = KnowledgeExtractionFlow("knowledge-extraction")
        
        input_data = KnowledgeExtractionInput(
            text="Test text",
            context="Test context"
        )
        
        with patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.provider_registry') as mock_provider_registry:
            # Mock provider registry returning None
            mock_provider_registry.get_by_config = AsyncMock(return_value=None)
            
            result = await flow.run_pipeline(input_data)
            
            # Should return empty result with error message
            assert isinstance(result, KnowledgeExtractionOutput)
            assert len(result.extracted_knowledge) == 0
            assert "failed" in result.processing_notes.lower()
            assert result.domains_detected == []
    
    @pytest.mark.asyncio
    async def test_run_pipeline_llm_exception(self):
        """Test handling when LLM generation fails."""
        flow = KnowledgeExtractionFlow("knowledge-extraction")
        
        input_data = KnowledgeExtractionInput(
            text="Test text",
            context="Test context"
        )
        
        with patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.provider_registry') as mock_provider_registry, \
             patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.resource_registry') as mock_resource_registry:
            
            mock_llm = AsyncMock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            mock_resource_registry.get.return_value = MagicMock()
            
            # Mock LLM to raise exception
            mock_llm.generate_structured.side_effect = Exception("LLM generation failed")
            
            result = await flow.run_pipeline(input_data)
            
            # Should return empty result with error message
            assert isinstance(result, KnowledgeExtractionOutput)
            assert len(result.extracted_knowledge) == 0
            assert "failed" in result.processing_notes.lower()
    
    @pytest.mark.asyncio
    async def test_run_pipeline_prompt_variables(self):
        """Test that prompt variables are correctly prepared."""
        flow = KnowledgeExtractionFlow("knowledge-extraction")
        
        input_data = KnowledgeExtractionInput(
            text="Machine learning uses algorithms",
            context="AI discussion",
            domain_hint="ai",
            extract_personal=True
        )
        
        with patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.provider_registry') as mock_provider_registry, \
             patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.resource_registry') as mock_resource_registry:
            
            mock_llm = AsyncMock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            mock_resource_registry.get.return_value = MagicMock()
            
            # Mock empty extraction result
            mock_llm.generate_structured.return_value = KnowledgeExtractionOutput(
                extracted_knowledge=[],
                processing_notes="Test",
                domains_detected=[]
            )
            
            await flow.run_pipeline(input_data)
            
            # Check that generate_structured was called with correct prompt variables
            call_args = mock_llm.generate_structured.call_args
            prompt_vars = call_args[1]['prompt_variables']
            
            assert prompt_vars['text'] == "Machine learning uses algorithms"
            assert prompt_vars['context'] == "AI discussion"
            assert prompt_vars['domain_hint'] == "ai"
            assert prompt_vars['extract_personal'] is True
    
    @pytest.mark.asyncio
    async def test_run_pipeline_no_domain_hint(self):
        """Test handling when no domain hint is provided."""
        flow = KnowledgeExtractionFlow("knowledge-extraction")
        
        input_data = KnowledgeExtractionInput(
            text="Test text",
            context="Test context",
            domain_hint=None
        )
        
        with patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.provider_registry') as mock_provider_registry, \
             patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.resource_registry') as mock_resource_registry:
            
            mock_llm = AsyncMock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            mock_resource_registry.get.return_value = MagicMock()
            
            mock_llm.generate_structured.return_value = KnowledgeExtractionOutput(
                extracted_knowledge=[],
                processing_notes="Test",
                domains_detected=[]
            )
            
            await flow.run_pipeline(input_data)
            
            call_args = mock_llm.generate_structured.call_args
            prompt_vars = call_args[1]['prompt_variables']
            
            # Should default to "general"
            assert prompt_vars['domain_hint'] == "general"


class TestEnhanceKnowledge:
    """Test _enhance_knowledge method."""
    
    @pytest.mark.asyncio
    async def test_enhance_knowledge_successful(self):
        """Test successful knowledge enhancement."""
        flow = KnowledgeExtractionFlow("knowledge-extraction")
        
        input_data = KnowledgeExtractionInput(
            text="Original text for context",
            context="Test context"
        )
        
        knowledge = ExtractedKnowledge(
            content="Python is object-oriented",
            knowledge_type=KnowledgeType.TECHNICAL,
            domain="programming",
            confidence=0.9,
            source_context="Programming discussion",
            entities=["Python"],
            metadata={"source": "tutorial"}
        )
        
        enhanced = await flow._enhance_knowledge(knowledge, input_data)
        
        assert enhanced is not None
        assert enhanced.content == knowledge.content
        assert enhanced.knowledge_type == knowledge.knowledge_type
        assert enhanced.confidence == knowledge.confidence
        
        # Check enhanced metadata
        assert "extraction_context" in enhanced.metadata
        assert enhanced.metadata["extraction_context"] == "Test context"
        assert "source_length" in enhanced.metadata
        assert enhanced.metadata["source_length"] == len(input_data.text)
        assert "source" in enhanced.metadata  # Original metadata preserved
    
    @pytest.mark.asyncio
    async def test_enhance_knowledge_low_confidence(self):
        """Test that low confidence knowledge is filtered out."""
        flow = KnowledgeExtractionFlow("knowledge-extraction")
        
        input_data = KnowledgeExtractionInput(text="Test", context="Test")
        
        knowledge = ExtractedKnowledge(
            content="Low confidence knowledge",
            knowledge_type=KnowledgeType.FACTUAL,
            domain="test",
            confidence=0.2,  # Below 0.3 threshold
            source_context="Test"
        )
        
        enhanced = await flow._enhance_knowledge(knowledge, input_data)
        
        assert enhanced is None
    
    @pytest.mark.asyncio
    async def test_enhance_knowledge_short_content(self):
        """Test that too-short content is filtered out."""
        flow = KnowledgeExtractionFlow("knowledge-extraction")
        
        input_data = KnowledgeExtractionInput(text="Test", context="Test")
        
        knowledge = ExtractedKnowledge(
            content="Short",  # Less than 10 characters
            knowledge_type=KnowledgeType.FACTUAL,
            domain="test",
            confidence=0.8,
            source_context="Test"
        )
        
        enhanced = await flow._enhance_knowledge(knowledge, input_data)
        
        assert enhanced is None
    
    @pytest.mark.asyncio
    async def test_enhance_knowledge_exception_handling(self):
        """Test exception handling in knowledge enhancement."""
        flow = KnowledgeExtractionFlow("knowledge-extraction")
        
        input_data = KnowledgeExtractionInput(text="Test", context="Test")
        
        # Create knowledge that might cause issues during enhancement
        knowledge = ExtractedKnowledge(
            content="Valid knowledge content",
            knowledge_type=KnowledgeType.FACTUAL,
            domain="test",
            confidence=0.8,
            source_context="Test"
        )
        
        # Mock logger to have no handlers to trigger exception path
        with patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.logger') as mock_logger:
            mock_logger.handlers = []
            
            enhanced = await flow._enhance_knowledge(knowledge, input_data)
            
            # Should return original knowledge if enhancement fails
            assert enhanced is not None
            assert enhanced.content == knowledge.content


class TestDetectDomains:
    """Test _detect_domains method."""
    
    @pytest.mark.asyncio
    async def test_detect_domains_success(self):
        """Test successful domain detection."""
        flow = KnowledgeExtractionFlow("knowledge-extraction")
        
        with patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.provider_registry') as mock_provider_registry, \
             patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.resource_registry') as mock_resource_registry:
            
            mock_provider_registry.get_by_config = AsyncMock(return_value=AsyncMock())
            mock_resource_registry.get.return_value = MagicMock()
            
            # Mock the domain detection result parsing
            with patch('json.loads') as mock_json_loads:
                mock_json_loads.return_value = ["programming", "technology"]
                
                domains = await flow._detect_domains("Python programming text")
                
                assert domains == ["programming", "technology"]
    
    @pytest.mark.asyncio
    async def test_detect_domains_invalid_json(self):
        """Test handling of invalid JSON from domain detection."""
        flow = KnowledgeExtractionFlow("knowledge-extraction")
        
        with patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.provider_registry') as mock_provider_registry, \
             patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.resource_registry') as mock_resource_registry:
            
            mock_provider_registry.get_by_config = AsyncMock(return_value=AsyncMock())
            mock_resource_registry.get.return_value = MagicMock()
            
            # Mock json.loads to raise JSONDecodeError
            with patch('json.loads') as mock_json_loads:
                import json
                mock_json_loads.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
                
                domains = await flow._detect_domains("Test text")
                
                assert domains == []
    
    @pytest.mark.asyncio
    async def test_detect_domains_non_list_result(self):
        """Test handling when domain detection doesn't return a list."""
        flow = KnowledgeExtractionFlow("knowledge-extraction")
        
        with patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.provider_registry') as mock_provider_registry, \
             patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.resource_registry') as mock_resource_registry:
            
            mock_provider_registry.get_by_config = AsyncMock(return_value=AsyncMock())
            mock_resource_registry.get.return_value = MagicMock()
            
            with patch('json.loads') as mock_json_loads:
                # Return non-list result
                mock_json_loads.return_value = {"domain": "programming"}
                
                domains = await flow._detect_domains("Test text")
                
                assert domains == []
    
    @pytest.mark.asyncio
    async def test_detect_domains_exception_handling(self):
        """Test exception handling in domain detection."""
        flow = KnowledgeExtractionFlow("knowledge-extraction")
        
        with patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.provider_registry') as mock_provider_registry:
            # Mock provider registry to raise exception
            mock_provider_registry.get_by_config = AsyncMock(side_effect=Exception("Provider error"))
            
            domains = await flow._detect_domains("Test text")
            
            assert domains == []


class TestFlowIntegration:
    """Test integration aspects of the knowledge extraction flow."""
    
    @pytest.mark.asyncio
    async def test_full_extraction_workflow(self):
        """Test complete extraction workflow with multiple knowledge items."""
        flow = KnowledgeExtractionFlow("knowledge-extraction")
        
        input_data = KnowledgeExtractionInput(
            text="Python is a programming language. Machine learning is a subset of AI. "
                 "Django is a Python web framework.",
            context="Technical discussion about Python ecosystem",
            domain_hint="technology"
        )
        
        with patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.provider_registry') as mock_provider_registry, \
             patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.resource_registry') as mock_resource_registry:
            
            mock_llm = AsyncMock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            mock_resource_registry.get.return_value = MagicMock()
            
            # Mock multiple knowledge items
            mock_extraction_result = KnowledgeExtractionOutput(
                extracted_knowledge=[
                    ExtractedKnowledge(
                        content="Python is a programming language",
                        knowledge_type=KnowledgeType.FACTUAL,
                        domain="programming",
                        confidence=0.95,
                        source_context="Technical discussion"
                    ),
                    ExtractedKnowledge(
                        content="Machine learning is a subset of AI",
                        knowledge_type=KnowledgeType.CONCEPTUAL,
                        domain="ai",
                        confidence=0.9,
                        source_context="Technical discussion"
                    ),
                    ExtractedKnowledge(
                        content="Low quality knowledge",
                        knowledge_type=KnowledgeType.FACTUAL,
                        domain="test",
                        confidence=0.1,  # Low confidence - should be filtered
                        source_context="Test"
                    )
                ],
                processing_notes="Extracted multiple knowledge items",
                domains_detected=["programming", "ai"]
            )
            mock_llm.generate_structured.return_value = mock_extraction_result
            
            result = await flow.run_pipeline(input_data)
            
            # Should have 2 items (one filtered out due to low confidence)
            assert len(result.extracted_knowledge) == 2
            assert result.extracted_knowledge[0].confidence >= 0.3
            assert result.extracted_knowledge[1].confidence >= 0.3
            
            # Check that metadata was enhanced
            for knowledge in result.extracted_knowledge:
                assert "extraction_context" in knowledge.metadata
                assert knowledge.metadata["extraction_context"] == "Technical discussion about Python ecosystem"
    
    @pytest.mark.asyncio
    async def test_empty_extraction_result(self):
        """Test handling when no knowledge is extracted."""
        flow = KnowledgeExtractionFlow("knowledge-extraction")
        
        input_data = KnowledgeExtractionInput(
            text="Hello, how are you?",
            context="Casual conversation"
        )
        
        with patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.provider_registry') as mock_provider_registry, \
             patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.resource_registry') as mock_resource_registry:
            
            mock_llm = AsyncMock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            mock_resource_registry.get.return_value = MagicMock()
            
            # Mock empty extraction result
            mock_extraction_result = KnowledgeExtractionOutput(
                extracted_knowledge=[],
                processing_notes="No significant knowledge found",
                domains_detected=[]
            )
            mock_llm.generate_structured.return_value = mock_extraction_result
            
            result = await flow.run_pipeline(input_data)
            
            assert len(result.extracted_knowledge) == 0
            assert "0 knowledge items" in result.processing_notes
    
    def test_flow_class_attributes(self):
        """Test that flow class has expected attributes."""
        assert KnowledgeExtractionFlow.__flow_name__ == "knowledge-extraction"
        assert "Extract knowledge" in KnowledgeExtractionFlow.__flow_description__
        assert KnowledgeExtractionFlow.__is_infrastructure__ is False
    
    def test_pipeline_method_attributes(self):
        """Test that pipeline method has expected attributes."""
        run_pipeline = KnowledgeExtractionFlow.run_pipeline
        assert hasattr(run_pipeline, '__pipeline_input_model__')
        assert hasattr(run_pipeline, '__pipeline_output_model__')
        assert run_pipeline.__pipeline_input_model__ == KnowledgeExtractionInput
        assert run_pipeline.__pipeline_output_model__ == KnowledgeExtractionOutput


class TestLogging:
    """Test logging functionality in the flow."""
    
    @pytest.mark.asyncio
    async def test_logging_calls(self):
        """Test that appropriate logging calls are made."""
        flow = KnowledgeExtractionFlow("knowledge-extraction")
        
        input_data = KnowledgeExtractionInput(
            text="Test text",
            context="Test context"
        )
        
        with patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.logger') as mock_logger, \
             patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.provider_registry') as mock_provider_registry, \
             patch('flowlib.agent.components.knowledge_flows.knowledge_extraction.resource_registry') as mock_resource_registry:
            
            mock_llm = AsyncMock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_llm)
            mock_resource_registry.get.return_value = MagicMock()
            
            mock_llm.generate_structured.return_value = KnowledgeExtractionOutput(
                extracted_knowledge=[],
                processing_notes="Test",
                domains_detected=[]
            )
            
            await flow.run_pipeline(input_data)
            
            # Check that logging calls were made
            mock_logger.info.assert_called()
            
            # Check specific log messages
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("Starting knowledge extraction" in msg for msg in info_calls)
            assert any("Knowledge extraction completed" in msg for msg in info_calls)