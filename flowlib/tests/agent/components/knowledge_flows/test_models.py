"""Tests for agent knowledge flows models."""

import pytest
from typing import Dict, List, Any
from pydantic import ValidationError

from flowlib.agent.components.knowledge_flows.models import (
    KnowledgeType,
    ExtractedKnowledge,
    KnowledgeExtractionInput,
    KnowledgeExtractionOutput,
    KnowledgeRetrievalInput,
    RetrievedKnowledge,
    KnowledgeRetrievalOutput
)


class TestKnowledgeType:
    """Test KnowledgeType enum."""
    
    def test_knowledge_type_values(self):
        """Test KnowledgeType enum values."""
        assert KnowledgeType.FACTUAL.value == "factual"
        assert KnowledgeType.PROCEDURAL.value == "procedural"
        assert KnowledgeType.CONCEPTUAL.value == "conceptual"
        assert KnowledgeType.PERSONAL.value == "personal"
        assert KnowledgeType.TECHNICAL.value == "technical"
    
    def test_knowledge_type_membership(self):
        """Test KnowledgeType membership."""
        all_types = [
            KnowledgeType.FACTUAL,
            KnowledgeType.PROCEDURAL,
            KnowledgeType.CONCEPTUAL,
            KnowledgeType.PERSONAL,
            KnowledgeType.TECHNICAL
        ]
        
        assert len(set(all_types)) == 5  # All unique
        
        for ktype in all_types:
            assert isinstance(ktype, KnowledgeType)
    
    def test_knowledge_type_string_inheritance(self):
        """Test that KnowledgeType inherits from str."""
        assert isinstance(KnowledgeType.FACTUAL, str)
        assert KnowledgeType.FACTUAL == "factual"


class TestExtractedKnowledge:
    """Test ExtractedKnowledge model."""
    
    def test_extracted_knowledge_creation_minimal(self):
        """Test creating ExtractedKnowledge with minimal required fields."""
        knowledge = ExtractedKnowledge(
            content="Water has the chemical formula H2O",
            knowledge_type=KnowledgeType.FACTUAL,
            domain="chemistry",
            confidence=0.95,
            source_context="Discussion about water molecules"
        )
        
        assert knowledge.content == "Water has the chemical formula H2O"
        assert knowledge.knowledge_type == KnowledgeType.FACTUAL
        assert knowledge.domain == "chemistry"
        assert knowledge.confidence == 0.95
        assert knowledge.source_context == "Discussion about water molecules"
        assert knowledge.entities == []
        assert knowledge.metadata == {}
    
    def test_extracted_knowledge_creation_complete(self):
        """Test creating ExtractedKnowledge with all fields."""
        metadata = {"source": "chemistry_textbook", "page": 42}
        entities = ["water", "H2O", "hydrogen", "oxygen"]
        
        knowledge = ExtractedKnowledge(
            content="Water consists of two hydrogen atoms and one oxygen atom",
            knowledge_type=KnowledgeType.FACTUAL,
            domain="chemistry",
            confidence=0.98,
            source_context="Chapter on molecular composition",
            entities=entities,
            metadata=metadata
        )
        
        assert knowledge.entities == entities
        assert knowledge.metadata == metadata
    
    def test_extracted_knowledge_confidence_validation(self):
        """Test ExtractedKnowledge confidence validation."""
        # Valid confidence values
        knowledge_min = ExtractedKnowledge(
            content="Test knowledge",
            knowledge_type=KnowledgeType.FACTUAL,
            domain="test",
            confidence=0.0,
            source_context="Test context"
        )
        assert knowledge_min.confidence == 0.0
        
        knowledge_max = ExtractedKnowledge(
            content="Test knowledge",
            knowledge_type=KnowledgeType.FACTUAL,
            domain="test",
            confidence=1.0,
            source_context="Test context"
        )
        assert knowledge_max.confidence == 1.0
        
        # Invalid confidence values should raise ValidationError
        with pytest.raises(ValidationError):
            ExtractedKnowledge(
                content="Test knowledge",
                knowledge_type=KnowledgeType.FACTUAL,
                domain="test",
                confidence=-0.1,  # Below 0
                source_context="Test context"
            )
        
        with pytest.raises(ValidationError):
            ExtractedKnowledge(
                content="Test knowledge",
                knowledge_type=KnowledgeType.FACTUAL,
                domain="test",
                confidence=1.1,  # Above 1
                source_context="Test context"
            )
    
    def test_extracted_knowledge_missing_required_fields(self):
        """Test ExtractedKnowledge validation with missing required fields."""
        with pytest.raises(ValidationError):
            ExtractedKnowledge()
        
        with pytest.raises(ValidationError):
            ExtractedKnowledge(content="Test")
        
        with pytest.raises(ValidationError):
            ExtractedKnowledge(
                content="Test",
                knowledge_type=KnowledgeType.FACTUAL
            )
    
    def test_extracted_knowledge_serialization(self):
        """Test ExtractedKnowledge serialization."""
        knowledge = ExtractedKnowledge(
            content="Python is a programming language",
            knowledge_type=KnowledgeType.TECHNICAL,
            domain="programming",
            confidence=0.92,
            source_context="Programming tutorial",
            entities=["Python", "programming"],
            metadata={"language": "english", "level": "beginner"}
        )
        
        data = knowledge.model_dump()
        
        assert data["content"] == "Python is a programming language"
        assert data["knowledge_type"] == "technical"
        assert data["domain"] == "programming"
        assert data["confidence"] == 0.92
        assert data["entities"] == ["Python", "programming"]
        assert data["metadata"]["language"] == "english"


class TestKnowledgeExtractionInput:
    """Test KnowledgeExtractionInput model."""
    
    def test_knowledge_extraction_input_minimal(self):
        """Test creating KnowledgeExtractionInput with minimal fields."""
        input_data = KnowledgeExtractionInput(
            text="Python is great for data science",
            context="User asking about programming languages"
        )
        
        assert input_data.text == "Python is great for data science"
        assert input_data.context == "User asking about programming languages"
        assert input_data.domain_hint is None
        assert input_data.extract_personal is True
    
    def test_knowledge_extraction_input_complete(self):
        """Test creating KnowledgeExtractionInput with all fields."""
        input_data = KnowledgeExtractionInput(
            text="I prefer using PyTorch for machine learning projects",
            context="Discussion about ML frameworks",
            domain_hint="technology",
            extract_personal=False
        )
        
        assert input_data.text == "I prefer using PyTorch for machine learning projects"
        assert input_data.context == "Discussion about ML frameworks"
        assert input_data.domain_hint == "technology"
        assert input_data.extract_personal is False
    
    def test_knowledge_extraction_input_validation(self):
        """Test KnowledgeExtractionInput validation."""
        # Missing required fields
        with pytest.raises(ValidationError):
            KnowledgeExtractionInput()
        
        with pytest.raises(ValidationError):
            KnowledgeExtractionInput(text="Test text")
        
        with pytest.raises(ValidationError):
            KnowledgeExtractionInput(context="Test context")


class TestKnowledgeExtractionOutput:
    """Test KnowledgeExtractionOutput model."""
    
    @pytest.fixture
    def sample_extracted_knowledge(self):
        """Create sample extracted knowledge for testing."""
        return [
            ExtractedKnowledge(
                content="Machine learning is a subset of AI",
                knowledge_type=KnowledgeType.CONCEPTUAL,
                domain="technology",
                confidence=0.95,
                source_context="AI discussion"
            ),
            ExtractedKnowledge(
                content="NumPy is used for numerical computing in Python",
                knowledge_type=KnowledgeType.TECHNICAL,
                domain="programming",
                confidence=0.92,
                source_context="Python libraries discussion"
            )
        ]
    
    def test_knowledge_extraction_output_creation(self, sample_extracted_knowledge):
        """Test creating KnowledgeExtractionOutput."""
        output = KnowledgeExtractionOutput(
            extracted_knowledge=sample_extracted_knowledge,
            processing_notes="Successfully extracted 2 knowledge items",
            domains_detected=["technology", "programming"]
        )
        
        assert len(output.extracted_knowledge) == 2
        assert output.processing_notes == "Successfully extracted 2 knowledge items"
        assert output.domains_detected == ["technology", "programming"]
    
    def test_knowledge_extraction_output_empty_knowledge(self):
        """Test KnowledgeExtractionOutput with no extracted knowledge."""
        output = KnowledgeExtractionOutput(
            extracted_knowledge=[],
            processing_notes="No significant knowledge found"
        )
        
        assert len(output.extracted_knowledge) == 0
        assert output.domains_detected == []
    
    def test_knowledge_extraction_output_user_display_no_knowledge(self):
        """Test user display with no extracted knowledge."""
        output = KnowledgeExtractionOutput(
            extracted_knowledge=[],
            processing_notes="No knowledge found"
        )
        
        display = output.get_user_display()
        assert "No knowledge extracted" in display
        assert "🔍" in display
    
    def test_knowledge_extraction_output_user_display_with_knowledge(self, sample_extracted_knowledge):
        """Test user display with extracted knowledge."""
        output = KnowledgeExtractionOutput(
            extracted_knowledge=sample_extracted_knowledge,
            processing_notes="Extraction completed successfully",
            domains_detected=["technology", "programming"]
        )
        
        display = output.get_user_display()
        assert "🧠 Extracted 2 knowledge items" in display
        assert "across 2 domains: technology, programming" in display
        assert "1 conceptual, 1 technical" in display
        assert "Extraction completed successfully" in display
    
    def test_knowledge_extraction_output_user_display_single_item(self):
        """Test user display with single knowledge item."""
        knowledge = [
            ExtractedKnowledge(
                content="Test knowledge",
                knowledge_type=KnowledgeType.FACTUAL,
                domain="test",
                confidence=0.8,
                source_context="Test"
            )
        ]
        
        output = KnowledgeExtractionOutput(
            extracted_knowledge=knowledge,
            processing_notes="Single item extracted"
        )
        
        display = output.get_user_display()
        assert "Extracted 1 knowledge item" in display  # No 's' for singular


class TestKnowledgeRetrievalInput:
    """Test KnowledgeRetrievalInput model."""
    
    def test_knowledge_retrieval_input_minimal(self):
        """Test creating KnowledgeRetrievalInput with minimal fields."""
        input_data = KnowledgeRetrievalInput(query="What is machine learning?")
        
        assert input_data.query == "What is machine learning?"
        assert input_data.domain is None
        assert input_data.max_results == 10
        assert input_data.include_plugins is True
        assert input_data.include_memory is True
    
    def test_knowledge_retrieval_input_complete(self):
        """Test creating KnowledgeRetrievalInput with all fields."""
        input_data = KnowledgeRetrievalInput(
            query="Python data structures",
            domain="programming",
            max_results=5,
            include_plugins=False,
            include_memory=True
        )
        
        assert input_data.query == "Python data structures"
        assert input_data.domain == "programming"
        assert input_data.max_results == 5
        assert input_data.include_plugins is False
        assert input_data.include_memory is True
    
    def test_knowledge_retrieval_input_max_results_validation(self):
        """Test max_results validation."""
        # Valid range
        input_data = KnowledgeRetrievalInput(query="test", max_results=1)
        assert input_data.max_results == 1
        
        input_data = KnowledgeRetrievalInput(query="test", max_results=50)
        assert input_data.max_results == 50
        
        # Invalid range
        with pytest.raises(ValidationError):
            KnowledgeRetrievalInput(query="test", max_results=0)
        
        with pytest.raises(ValidationError):
            KnowledgeRetrievalInput(query="test", max_results=51)


class TestRetrievedKnowledge:
    """Test RetrievedKnowledge model."""
    
    def test_retrieved_knowledge_creation(self):
        """Test creating RetrievedKnowledge."""
        knowledge = RetrievedKnowledge(
            content="Pandas is a data manipulation library",
            source="memory",
            domain="programming",
            confidence=0.88,
            metadata={"timestamp": "2024-01-15", "conversation_id": "conv_123"}
        )
        
        assert knowledge.content == "Pandas is a data manipulation library"
        assert knowledge.source == "memory"
        assert knowledge.domain == "programming"
        assert knowledge.confidence == 0.88
        assert knowledge.metadata["timestamp"] == "2024-01-15"
    
    def test_retrieved_knowledge_confidence_validation(self):
        """Test RetrievedKnowledge confidence validation."""
        # Valid confidence
        knowledge = RetrievedKnowledge(
            content="Test",
            source="test",
            domain="test",
            confidence=0.5
        )
        assert knowledge.confidence == 0.5
        
        # Invalid confidence
        with pytest.raises(ValidationError):
            RetrievedKnowledge(
                content="Test",
                source="test",
                domain="test",
                confidence=1.5
            )


class TestKnowledgeRetrievalOutput:
    """Test KnowledgeRetrievalOutput model."""
    
    @pytest.fixture
    def sample_retrieved_knowledge(self):
        """Create sample retrieved knowledge for testing."""
        return [
            RetrievedKnowledge(
                content="Python is an interpreted language",
                source="memory",
                domain="programming",
                confidence=0.92
            ),
            RetrievedKnowledge(
                content="Variables store data in Python",
                source="knowledge_plugin",
                domain="programming",
                confidence=0.89
            ),
            RetrievedKnowledge(
                content="Lists are mutable sequences",
                source="memory",
                domain="programming",
                confidence=0.85
            )
        ]
    
    def test_knowledge_retrieval_output_creation(self, sample_retrieved_knowledge):
        """Test creating KnowledgeRetrievalOutput."""
        output = KnowledgeRetrievalOutput(
            retrieved_knowledge=sample_retrieved_knowledge,
            search_summary="Found 3 relevant items about Python programming",
            sources_searched=["memory", "knowledge_plugin"],
            total_results=3
        )
        
        assert len(output.retrieved_knowledge) == 3
        assert output.search_summary == "Found 3 relevant items about Python programming"
        assert output.sources_searched == ["memory", "knowledge_plugin"]
        assert output.total_results == 3
    
    def test_knowledge_retrieval_output_no_results(self):
        """Test KnowledgeRetrievalOutput with no results."""
        output = KnowledgeRetrievalOutput(
            retrieved_knowledge=[],
            search_summary="No relevant knowledge found",
            sources_searched=["memory", "plugins"],
            total_results=0
        )
        
        assert len(output.retrieved_knowledge) == 0
        assert output.total_results == 0
    
    def test_knowledge_retrieval_output_user_display_no_results(self):
        """Test user display with no results."""
        output = KnowledgeRetrievalOutput(
            retrieved_knowledge=[],
            search_summary="No matches found",
            total_results=0
        )
        
        display = output.get_user_display()
        assert "No knowledge found" in display
        assert "🔍" in display
    
    def test_knowledge_retrieval_output_user_display_with_results(self, sample_retrieved_knowledge):
        """Test user display with results."""
        output = KnowledgeRetrievalOutput(
            retrieved_knowledge=sample_retrieved_knowledge,
            search_summary="Successfully found Python programming knowledge",
            sources_searched=["memory", "knowledge_plugin"],
            total_results=3
        )
        
        display = output.get_user_display()
        assert "🎯 Found 3 relevant knowledge items" in display
        assert "Domains: programming" in display
        assert "2 from memory, 1 from knowledge_plugin" in display
        assert "Successfully found Python programming knowledge" in display
    
    def test_knowledge_retrieval_output_user_display_truncated_results(self, sample_retrieved_knowledge):
        """Test user display when showing fewer results than total."""
        output = KnowledgeRetrievalOutput(
            retrieved_knowledge=sample_retrieved_knowledge[:2],  # Show only 2 of 3
            search_summary="Showing top results",
            total_results=5
        )
        
        display = output.get_user_display()
        assert "Found 2 relevant knowledge items" in display
        assert "(showing top 2 of 5 total)" in display
    
    def test_knowledge_retrieval_output_user_display_multiple_domains(self):
        """Test user display with multiple domains."""
        knowledge = [
            RetrievedKnowledge(
                content="Python programming",
                source="memory",
                domain="programming",
                confidence=0.9
            ),
            RetrievedKnowledge(
                content="Chemistry facts",
                source="plugin",
                domain="chemistry",
                confidence=0.8
            )
        ]
        
        output = KnowledgeRetrievalOutput(
            retrieved_knowledge=knowledge,
            search_summary="Multi-domain search",
            total_results=2
        )
        
        display = output.get_user_display()
        assert "Domains: chemistry, programming" in display  # Should be sorted


class TestModelIntegration:
    """Test integration between knowledge flow models."""
    
    def test_extraction_to_retrieval_workflow(self):
        """Test typical workflow from extraction to retrieval."""
        # Simulate extraction process
        extraction_input = KnowledgeExtractionInput(
            text="Machine learning uses algorithms to find patterns in data",
            context="Data science tutorial"
        )
        
        extracted_knowledge = [
            ExtractedKnowledge(
                content="Machine learning uses algorithms to find patterns",
                knowledge_type=KnowledgeType.CONCEPTUAL,
                domain="data_science",
                confidence=0.9,
                source_context="Data science tutorial"
            )
        ]
        
        extraction_output = KnowledgeExtractionOutput(
            extracted_knowledge=extracted_knowledge,
            processing_notes="Extracted ML concept",
            domains_detected=["data_science"]
        )
        
        # Simulate retrieval process
        retrieval_input = KnowledgeRetrievalInput(
            query="machine learning algorithms",
            domain="data_science"
        )
        
        retrieved_knowledge = [
            RetrievedKnowledge(
                content="Machine learning uses algorithms to find patterns",
                source="memory",
                domain="data_science",
                confidence=0.9
            )
        ]
        
        retrieval_output = KnowledgeRetrievalOutput(
            retrieved_knowledge=retrieved_knowledge,
            search_summary="Found relevant ML information",
            total_results=1
        )
        
        # Verify consistency
        original_content = extraction_output.extracted_knowledge[0].content
        retrieved_content = retrieval_output.retrieved_knowledge[0].content
        assert original_content == retrieved_content
    
    def test_knowledge_type_consistency(self):
        """Test that knowledge types are consistently used."""
        for ktype in KnowledgeType:
            knowledge = ExtractedKnowledge(
                content=f"Test {ktype.value} knowledge",
                knowledge_type=ktype,
                domain="test",
                confidence=0.8,
                source_context="Test context"
            )
            
            assert knowledge.knowledge_type == ktype
            assert knowledge.knowledge_type.value == ktype.value
    
    def test_serialization_roundtrip_integration(self):
        """Test complete serialization/deserialization roundtrip."""
        # Create extraction output
        original_extraction = KnowledgeExtractionOutput(
            extracted_knowledge=[
                ExtractedKnowledge(
                    content="Python is interpreted",
                    knowledge_type=KnowledgeType.TECHNICAL,
                    domain="programming",
                    confidence=0.95,
                    source_context="Programming discussion",
                    entities=["Python"],
                    metadata={"type": "language_feature"}
                )
            ],
            processing_notes="Extracted technical knowledge",
            domains_detected=["programming"]
        )
        
        # Serialize and deserialize
        data = original_extraction.model_dump()
        restored_extraction = KnowledgeExtractionOutput(**data)
        
        assert len(restored_extraction.extracted_knowledge) == 1
        assert restored_extraction.extracted_knowledge[0].content == "Python is interpreted"
        assert restored_extraction.extracted_knowledge[0].knowledge_type == KnowledgeType.TECHNICAL
        
        # Create retrieval output
        original_retrieval = KnowledgeRetrievalOutput(
            retrieved_knowledge=[
                RetrievedKnowledge(
                    content="Python is interpreted",
                    source="memory",
                    domain="programming",
                    confidence=0.95
                )
            ],
            search_summary="Found Python information",
            sources_searched=["memory"],
            total_results=1
        )
        
        # Serialize and deserialize
        data = original_retrieval.model_dump()
        restored_retrieval = KnowledgeRetrievalOutput(**data)
        
        assert len(restored_retrieval.retrieved_knowledge) == 1
        assert restored_retrieval.retrieved_knowledge[0].content == "Python is interpreted"