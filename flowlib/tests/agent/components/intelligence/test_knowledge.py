"""Tests for knowledge models and data structures."""

import pytest
from datetime import datetime
from dataclasses import FrozenInstanceError

from flowlib.agent.components.intelligence.knowledge import (
    ConfidenceLevel,
    Entity,
    Concept,
    Relationship,
    Pattern,
    KnowledgeSet,
    ContentAnalysis,
    LearningResult
)


class TestConfidenceLevel:
    """Test ConfidenceLevel enum."""
    
    def test_confidence_level_values(self):
        """Test confidence level enum values."""
        assert ConfidenceLevel.LOW == "low"
        assert ConfidenceLevel.MEDIUM == "medium"
        assert ConfidenceLevel.HIGH == "high"
        assert ConfidenceLevel.VERY_HIGH == "very_high"
    
    def test_confidence_level_membership(self):
        """Test confidence level membership."""
        levels = list(ConfidenceLevel)
        assert len(levels) == 4
        assert ConfidenceLevel.LOW in levels
        assert ConfidenceLevel.MEDIUM in levels
        assert ConfidenceLevel.HIGH in levels
        assert ConfidenceLevel.VERY_HIGH in levels


class TestEntity:
    """Test Entity dataclass."""
    
    def test_entity_creation_minimal(self):
        """Test entity creation with minimal data."""
        entity = Entity(name="Test Entity", type="concept")
        
        assert entity.name == "Test Entity"
        assert entity.type == "concept"
        assert entity.description is None
        assert entity.properties == {}
        assert entity.confidence == 1.0
        assert entity.source == ""
        assert entity.aliases == []
    
    def test_entity_creation_full(self):
        """Test entity creation with all fields."""
        properties = {"category": "technology", "importance": "high"}
        aliases = ["alias1", "alias2"]
        
        entity = Entity(
            name="Full Entity",
            type="organization",
            description="A complete entity",
            properties=properties,
            confidence=0.8,
            source="test_source",
            aliases=aliases
        )
        
        assert entity.name == "Full Entity"
        assert entity.type == "organization"
        assert entity.description == "A complete entity"
        assert entity.properties == properties
        assert entity.confidence == 0.8
        assert entity.source == "test_source"
        assert entity.aliases == aliases
    
    def test_entity_validation_empty_name(self):
        """Test entity validation with empty name."""
        with pytest.raises(ValueError, match="Entity name cannot be empty"):
            Entity(name="", type="concept")
        
        with pytest.raises(ValueError, match="Entity name cannot be empty"):
            Entity(name="   ", type="concept")
    
    def test_entity_validation_confidence_range(self):
        """Test entity validation with invalid confidence values."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            Entity(name="Test", type="concept", confidence=-0.1)
        
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            Entity(name="Test", type="concept", confidence=1.1)
        
        # Valid edge cases should work
        entity_low = Entity(name="Test", type="concept", confidence=0.0)
        assert entity_low.confidence == 0.0
        
        entity_high = Entity(name="Test", type="concept", confidence=1.0)
        assert entity_high.confidence == 1.0
    
    def test_entity_confidence_level_property(self):
        """Test entity confidence level property."""
        entity_low = Entity(name="Test", type="concept", confidence=0.2)
        assert entity_low.confidence_level == ConfidenceLevel.LOW
        
        entity_medium = Entity(name="Test", type="concept", confidence=0.5)
        assert entity_medium.confidence_level == ConfidenceLevel.MEDIUM
        
        entity_high = Entity(name="Test", type="concept", confidence=0.8)
        assert entity_high.confidence_level == ConfidenceLevel.HIGH
        
        entity_very_high = Entity(name="Test", type="concept", confidence=0.95)
        assert entity_very_high.confidence_level == ConfidenceLevel.VERY_HIGH
    
    def test_entity_confidence_level_boundaries(self):
        """Test entity confidence level boundary cases."""
        # Test boundary values
        entity_039 = Entity(name="Test", type="concept", confidence=0.39)
        assert entity_039.confidence_level == ConfidenceLevel.LOW
        
        entity_040 = Entity(name="Test", type="concept", confidence=0.4)
        assert entity_040.confidence_level == ConfidenceLevel.MEDIUM
        
        entity_069 = Entity(name="Test", type="concept", confidence=0.69)
        assert entity_069.confidence_level == ConfidenceLevel.MEDIUM
        
        entity_070 = Entity(name="Test", type="concept", confidence=0.7)
        assert entity_070.confidence_level == ConfidenceLevel.HIGH
        
        entity_089 = Entity(name="Test", type="concept", confidence=0.89)
        assert entity_089.confidence_level == ConfidenceLevel.HIGH
        
        entity_090 = Entity(name="Test", type="concept", confidence=0.9)
        assert entity_090.confidence_level == ConfidenceLevel.VERY_HIGH


class TestConcept:
    """Test Concept dataclass."""
    
    def test_concept_creation_minimal(self):
        """Test concept creation with minimal data."""
        concept = Concept(name="Test Concept", description="A test concept")
        
        assert concept.name == "Test Concept"
        assert concept.description == "A test concept"
        assert concept.category is None
        assert concept.examples == []
        assert concept.related_concepts == []
        assert concept.confidence == 1.0
        assert concept.abstraction_level == 1
    
    def test_concept_creation_full(self):
        """Test concept creation with all fields."""
        examples = ["example1", "example2"]
        related = ["concept1", "concept2"]
        
        concept = Concept(
            name="Full Concept",
            description="A complete concept",
            category="technical",
            examples=examples,
            related_concepts=related,
            confidence=0.7,
            abstraction_level=3
        )
        
        assert concept.name == "Full Concept"
        assert concept.description == "A complete concept"
        assert concept.category == "technical"
        assert concept.examples == examples
        assert concept.related_concepts == related
        assert concept.confidence == 0.7
        assert concept.abstraction_level == 3
    
    def test_concept_validation_empty_name(self):
        """Test concept validation with empty name."""
        with pytest.raises(ValueError, match="Concept name cannot be empty"):
            Concept(name="", description="test")
        
        with pytest.raises(ValueError, match="Concept name cannot be empty"):
            Concept(name="   ", description="test")
    
    def test_concept_validation_empty_description(self):
        """Test concept validation with empty description."""
        with pytest.raises(ValueError, match="Concept description cannot be empty"):
            Concept(name="test", description="")
        
        with pytest.raises(ValueError, match="Concept description cannot be empty"):
            Concept(name="test", description="   ")
    
    def test_concept_validation_abstraction_level(self):
        """Test concept validation with invalid abstraction levels."""
        with pytest.raises(ValueError, match="Abstraction level must be between 1 and 5"):
            Concept(name="test", description="test", abstraction_level=0)
        
        with pytest.raises(ValueError, match="Abstraction level must be between 1 and 5"):
            Concept(name="test", description="test", abstraction_level=6)
        
        # Valid edge cases should work
        concept_low = Concept(name="test", description="test", abstraction_level=1)
        assert concept_low.abstraction_level == 1
        
        concept_high = Concept(name="test", description="test", abstraction_level=5)
        assert concept_high.abstraction_level == 5
    
    def test_concept_confidence_level_property(self):
        """Test concept confidence level property."""
        concept_low = Concept(name="test", description="test", confidence=0.3)
        assert concept_low.confidence_level == ConfidenceLevel.LOW
        
        concept_medium = Concept(name="test", description="test", confidence=0.6)
        assert concept_medium.confidence_level == ConfidenceLevel.MEDIUM
        
        concept_high = Concept(name="test", description="test", confidence=0.8)
        assert concept_high.confidence_level == ConfidenceLevel.HIGH
        
        concept_very_high = Concept(name="test", description="test", confidence=0.95)
        assert concept_very_high.confidence_level == ConfidenceLevel.VERY_HIGH


class TestRelationship:
    """Test Relationship dataclass."""
    
    def test_relationship_creation_minimal(self):
        """Test relationship creation with minimal data."""
        rel = Relationship(source="Entity A", target="Entity B", type="relates_to")
        
        assert rel.source == "Entity A"
        assert rel.target == "Entity B"
        assert rel.type == "relates_to"
        assert rel.description is None
        assert rel.confidence == 1.0
        assert rel.bidirectional is False
        assert rel.strength == "medium"
    
    def test_relationship_creation_full(self):
        """Test relationship creation with all fields."""
        rel = Relationship(
            source="Source Entity",
            target="Target Entity",
            type="is_part_of",
            description="A complete relationship",
            confidence=0.85,
            bidirectional=True,
            strength="strong"
        )
        
        assert rel.source == "Source Entity"
        assert rel.target == "Target Entity"
        assert rel.type == "is_part_of"
        assert rel.description == "A complete relationship"
        assert rel.confidence == 0.85
        assert rel.bidirectional is True
        assert rel.strength == "strong"
    
    def test_relationship_validation_empty_source(self):
        """Test relationship validation with empty source."""
        with pytest.raises(ValueError, match="Relationship source cannot be empty"):
            Relationship(source="", target="target", type="type")
        
        with pytest.raises(ValueError, match="Relationship source cannot be empty"):
            Relationship(source="   ", target="target", type="type")
    
    def test_relationship_validation_empty_target(self):
        """Test relationship validation with empty target."""
        with pytest.raises(ValueError, match="Relationship target cannot be empty"):
            Relationship(source="source", target="", type="type")
        
        with pytest.raises(ValueError, match="Relationship target cannot be empty"):
            Relationship(source="source", target="   ", type="type")
    
    def test_relationship_validation_empty_type(self):
        """Test relationship validation with empty type."""
        with pytest.raises(ValueError, match="Relationship type cannot be empty"):
            Relationship(source="source", target="target", type="")
        
        with pytest.raises(ValueError, match="Relationship type cannot be empty"):
            Relationship(source="source", target="target", type="   ")
    
    def test_relationship_validation_strength(self):
        """Test relationship validation with invalid strength."""
        with pytest.raises(ValueError, match="Strength must be 'weak', 'medium', or 'strong'"):
            Relationship(source="source", target="target", type="type", strength="invalid")
        
        # Valid strengths should work
        rel_weak = Relationship(source="s", target="t", type="type", strength="weak")
        assert rel_weak.strength == "weak"
        
        rel_medium = Relationship(source="s", target="t", type="type", strength="medium")
        assert rel_medium.strength == "medium"
        
        rel_strong = Relationship(source="s", target="t", type="type", strength="strong")
        assert rel_strong.strength == "strong"
    
    def test_relationship_confidence_level_property(self):
        """Test relationship confidence level property."""
        rel_low = Relationship(source="s", target="t", type="type", confidence=0.1)
        assert rel_low.confidence_level == ConfidenceLevel.LOW
        
        rel_medium = Relationship(source="s", target="t", type="type", confidence=0.5)
        assert rel_medium.confidence_level == ConfidenceLevel.MEDIUM
        
        rel_high = Relationship(source="s", target="t", type="type", confidence=0.8)
        assert rel_high.confidence_level == ConfidenceLevel.HIGH
        
        rel_very_high = Relationship(source="s", target="t", type="type", confidence=0.95)
        assert rel_very_high.confidence_level == ConfidenceLevel.VERY_HIGH


class TestPattern:
    """Test Pattern dataclass."""
    
    def test_pattern_creation_minimal(self):
        """Test pattern creation with minimal data."""
        pattern = Pattern(name="Test Pattern", description="A test pattern")
        
        assert pattern.name == "Test Pattern"
        assert pattern.description == "A test pattern"
        assert pattern.pattern_type == "general"
        assert pattern.examples == []
        assert pattern.variables == []
        assert pattern.confidence == 1.0
        assert pattern.frequency == "unknown"
    
    def test_pattern_creation_full(self):
        """Test pattern creation with all fields."""
        examples = ["example1", "example2"]
        variables = ["var1", "var2"]
        
        pattern = Pattern(
            name="Full Pattern",
            description="A complete pattern",
            pattern_type="sequence",
            examples=examples,
            variables=variables,
            confidence=0.75,
            frequency="common"
        )
        
        assert pattern.name == "Full Pattern"
        assert pattern.description == "A complete pattern"
        assert pattern.pattern_type == "sequence"
        assert pattern.examples == examples
        assert pattern.variables == variables
        assert pattern.confidence == 0.75
        assert pattern.frequency == "common"
    
    def test_pattern_validation_empty_name(self):
        """Test pattern validation with empty name."""
        with pytest.raises(ValueError, match="Pattern name cannot be empty"):
            Pattern(name="", description="test")
        
        with pytest.raises(ValueError, match="Pattern name cannot be empty"):
            Pattern(name="   ", description="test")
    
    def test_pattern_validation_empty_description(self):
        """Test pattern validation with empty description."""
        with pytest.raises(ValueError, match="Pattern description cannot be empty"):
            Pattern(name="test", description="")
        
        with pytest.raises(ValueError, match="Pattern description cannot be empty"):
            Pattern(name="test", description="   ")
    
    def test_pattern_validation_frequency(self):
        """Test pattern validation with invalid frequency."""
        with pytest.raises(ValueError, match="Frequency must be one of: unknown, rare, occasional, common, frequent"):
            Pattern(name="test", description="test", frequency="invalid")
        
        # Valid frequencies should work
        valid_frequencies = ["unknown", "rare", "occasional", "common", "frequent"]
        for freq in valid_frequencies:
            pattern = Pattern(name="test", description="test", frequency=freq)
            assert pattern.frequency == freq
    
    def test_pattern_confidence_level_property(self):
        """Test pattern confidence level property."""
        pattern_low = Pattern(name="test", description="test", confidence=0.2)
        assert pattern_low.confidence_level == ConfidenceLevel.LOW
        
        pattern_medium = Pattern(name="test", description="test", confidence=0.6)
        assert pattern_medium.confidence_level == ConfidenceLevel.MEDIUM
        
        pattern_high = Pattern(name="test", description="test", confidence=0.8)
        assert pattern_high.confidence_level == ConfidenceLevel.HIGH
        
        pattern_very_high = Pattern(name="test", description="test", confidence=0.95)
        assert pattern_very_high.confidence_level == ConfidenceLevel.VERY_HIGH


class TestKnowledgeSet:
    """Test KnowledgeSet dataclass."""
    
    def test_knowledge_set_creation_empty(self):
        """Test knowledge set creation with no items."""
        ks = KnowledgeSet()
        
        assert ks.entities == []
        assert ks.concepts == []
        assert ks.relationships == []
        assert ks.patterns == []
        assert ks.summary == ""
        assert ks.confidence == 1.0
        assert isinstance(ks.extracted_at, datetime)
        assert ks.source_content == ""
        assert ks.processing_notes == []
    
    def test_knowledge_set_creation_with_items(self):
        """Test knowledge set creation with items."""
        entities = [Entity(name="Entity1", type="person")]
        concepts = [Concept(name="Concept1", description="Description1")]
        relationships = [Relationship(source="A", target="B", type="relates_to")]
        patterns = [Pattern(name="Pattern1", description="Description1")]
        notes = ["note1", "note2"]
        
        ks = KnowledgeSet(
            entities=entities,
            concepts=concepts,
            relationships=relationships,
            patterns=patterns,
            summary="Test summary",
            confidence=0.8,
            source_content="Test content",
            processing_notes=notes
        )
        
        assert ks.entities == entities
        assert ks.concepts == concepts
        assert ks.relationships == relationships
        assert ks.patterns == patterns
        assert ks.summary == "Test summary"
        assert ks.confidence == 0.8
        assert ks.source_content == "Test content"
        assert ks.processing_notes == notes
    
    def test_knowledge_set_total_items_property(self):
        """Test knowledge set total_items property."""
        # Empty set
        ks_empty = KnowledgeSet()
        assert ks_empty.total_items == 0
        
        # Set with items
        entities = [Entity(name="E1", type="person"), Entity(name="E2", type="place")]
        concepts = [Concept(name="C1", description="D1")]
        relationships = [Relationship(source="A", target="B", type="relates_to")]
        patterns = [Pattern(name="P1", description="D1"), Pattern(name="P2", description="D2")]
        
        ks = KnowledgeSet(
            entities=entities,
            concepts=concepts,
            relationships=relationships,
            patterns=patterns
        )
        
        assert ks.total_items == 6  # 2 + 1 + 1 + 2
    
    def test_knowledge_set_confidence_level_property(self):
        """Test knowledge set confidence level property."""
        ks_low = KnowledgeSet(confidence=0.3)
        assert ks_low.confidence_level == ConfidenceLevel.LOW
        
        ks_medium = KnowledgeSet(confidence=0.6)
        assert ks_medium.confidence_level == ConfidenceLevel.MEDIUM
        
        ks_high = KnowledgeSet(confidence=0.8)
        assert ks_high.confidence_level == ConfidenceLevel.HIGH
        
        ks_very_high = KnowledgeSet(confidence=0.95)
        assert ks_very_high.confidence_level == ConfidenceLevel.VERY_HIGH
    
    def test_knowledge_set_get_stats(self):
        """Test knowledge set get_stats method."""
        entities = [Entity(name="E1", type="person", confidence=0.8)]
        concepts = [Concept(name="C1", description="D1", confidence=0.6)]
        relationships = [Relationship(source="A", target="B", type="relates_to", confidence=0.9)]
        patterns = [Pattern(name="P1", description="D1", confidence=0.7)]
        
        ks = KnowledgeSet(
            entities=entities,
            concepts=concepts,
            relationships=relationships,
            patterns=patterns,
            confidence=0.75
        )
        
        stats = ks.get_stats()
        
        assert stats["total_items"] == 4
        assert stats["entities_count"] == 1
        assert stats["concepts_count"] == 1
        assert stats["relationships_count"] == 1
        assert stats["patterns_count"] == 1
        assert stats["confidence_level"] == "high"
        assert stats["average_confidence"] == 0.75  # (0.8 + 0.6 + 0.9 + 0.7) / 4
        assert "extraction_time" in stats
    
    def test_knowledge_set_calculate_average_confidence_empty(self):
        """Test average confidence calculation with no items."""
        ks = KnowledgeSet(confidence=0.8)
        assert ks._calculate_average_confidence() == 0.8
    
    def test_knowledge_set_calculate_average_confidence_with_items(self):
        """Test average confidence calculation with items."""
        entities = [Entity(name="E1", type="person", confidence=0.8)]
        concepts = [Concept(name="C1", description="D1", confidence=0.6)]
        
        ks = KnowledgeSet(entities=entities, concepts=concepts)
        assert ks._calculate_average_confidence() == 0.7  # (0.8 + 0.6) / 2


class TestContentAnalysis:
    """Test ContentAnalysis dataclass."""
    
    def test_content_analysis_creation_minimal(self):
        """Test content analysis creation with minimal data."""
        analysis = ContentAnalysis(content_type="technical")
        
        assert analysis.content_type == "technical"
        assert analysis.key_topics == []
        assert analysis.complexity_level == "medium"
        assert analysis.language == "en"
        assert analysis.length_category == "medium"
        assert analysis.structure_type == "unstructured"
        assert analysis.suggested_focus == []
        assert analysis.confidence == 1.0
    
    def test_content_analysis_creation_full(self):
        """Test content analysis creation with all fields."""
        topics = ["topic1", "topic2"]
        focus = ["entities", "concepts"]
        
        analysis = ContentAnalysis(
            content_type="narrative",
            key_topics=topics,
            complexity_level="complex",
            language="es",
            length_category="long",
            structure_type="structured",
            suggested_focus=focus,
            confidence=0.85
        )
        
        assert analysis.content_type == "narrative"
        assert analysis.key_topics == topics
        assert analysis.complexity_level == "complex"
        assert analysis.language == "es"
        assert analysis.length_category == "long"
        assert analysis.structure_type == "structured"
        assert analysis.suggested_focus == focus
        assert analysis.confidence == 0.85
    
    def test_content_analysis_validation_complexity_level(self):
        """Test content analysis validation with invalid complexity level."""
        with pytest.raises(ValueError, match="Complexity level must be 'simple', 'medium', or 'complex'"):
            ContentAnalysis(content_type="test", complexity_level="invalid")
        
        # Valid levels should work
        for level in ["simple", "medium", "complex"]:
            analysis = ContentAnalysis(content_type="test", complexity_level=level)
            assert analysis.complexity_level == level
    
    def test_content_analysis_validation_length_category(self):
        """Test content analysis validation with invalid length category."""
        with pytest.raises(ValueError, match="Length category must be 'short', 'medium', or 'long'"):
            ContentAnalysis(content_type="test", length_category="invalid")
        
        # Valid categories should work
        for category in ["short", "medium", "long"]:
            analysis = ContentAnalysis(content_type="test", length_category=category)
            assert analysis.length_category == category
    
    def test_content_analysis_validation_structure_type(self):
        """Test content analysis validation with invalid structure type."""
        with pytest.raises(ValueError, match="Structure type must be 'structured', 'semi-structured', or 'unstructured'"):
            ContentAnalysis(content_type="test", structure_type="invalid")
        
        # Valid types should work
        for struct_type in ["structured", "semi-structured", "unstructured"]:
            analysis = ContentAnalysis(content_type="test", structure_type=struct_type)
            assert analysis.structure_type == struct_type


class TestLearningResult:
    """Test LearningResult dataclass."""
    
    def test_learning_result_creation_success(self):
        """Test learning result creation for success case."""
        knowledge = KnowledgeSet()
        result = LearningResult(success=True, knowledge=knowledge)
        
        assert result.success is True
        assert result.knowledge == knowledge
        assert result.processing_time_seconds == 0.0
        assert result.message == ""
        assert result.errors == []
        assert result.warnings == []
    
    def test_learning_result_creation_full(self):
        """Test learning result creation with all fields."""
        knowledge = KnowledgeSet(summary="Test knowledge")
        errors = ["error1", "error2"]
        warnings = ["warning1"]
        
        result = LearningResult(
            success=False,
            knowledge=knowledge,
            processing_time_seconds=2.5,
            message="Processing failed",
            errors=errors,
            warnings=warnings
        )
        
        assert result.success is False
        assert result.knowledge == knowledge
        assert result.processing_time_seconds == 2.5
        assert result.message == "Processing failed"
        assert result.errors == errors
        assert result.warnings == warnings
    
    def test_learning_result_has_errors_property(self):
        """Test learning result has_errors property."""
        knowledge = KnowledgeSet()
        
        result_no_errors = LearningResult(success=True, knowledge=knowledge)
        assert result_no_errors.has_errors is False
        
        result_with_errors = LearningResult(success=False, knowledge=knowledge, errors=["error1"])
        assert result_with_errors.has_errors is True
    
    def test_learning_result_has_warnings_property(self):
        """Test learning result has_warnings property."""
        knowledge = KnowledgeSet()
        
        result_no_warnings = LearningResult(success=True, knowledge=knowledge)
        assert result_no_warnings.has_warnings is False
        
        result_with_warnings = LearningResult(success=True, knowledge=knowledge, warnings=["warning1"])
        assert result_with_warnings.has_warnings is True
    
    def test_learning_result_get_summary_success(self):
        """Test learning result get_summary for success case."""
        entities = [Entity(name="E1", type="person")]
        concepts = [Concept(name="C1", description="D1")]
        knowledge = KnowledgeSet(entities=entities, concepts=concepts, confidence=0.8)
        
        result = LearningResult(success=True, knowledge=knowledge)
        summary = result.get_summary()
        
        assert "Successfully learned 2 items" in summary
        assert "1 entities, 1 concepts" in summary
        assert "0 relationships, 0 patterns" in summary
        assert "high confidence" in summary
    
    def test_learning_result_get_summary_failure(self):
        """Test learning result get_summary for failure case."""
        knowledge = KnowledgeSet()
        result = LearningResult(
            success=False, 
            knowledge=knowledge, 
            message="Failed to process content"
        )
        
        summary = result.get_summary()
        assert summary == "Learning failed: Failed to process content"
    
    def test_learning_result_get_summary_complex_knowledge(self):
        """Test learning result get_summary with complex knowledge."""
        entities = [Entity(name="E1", type="person"), Entity(name="E2", type="place")]
        concepts = [Concept(name="C1", description="D1")]
        relationships = [Relationship(source="A", target="B", type="relates_to")]
        patterns = [Pattern(name="P1", description="D1"), Pattern(name="P2", description="D2")]
        
        knowledge = KnowledgeSet(
            entities=entities,
            concepts=concepts, 
            relationships=relationships,
            patterns=patterns,
            confidence=0.95
        )
        
        result = LearningResult(success=True, knowledge=knowledge)
        summary = result.get_summary()
        
        assert "Successfully learned 6 items" in summary
        assert "2 entities, 1 concepts, 1 relationships, 2 patterns" in summary
        assert "very_high confidence" in summary


class TestKnowledgeModelsEdgeCases:
    """Test edge cases and boundary conditions for knowledge models."""
    
    def test_entity_with_unicode_characters(self):
        """Test entity with unicode characters."""
        entity = Entity(
            name="Entity with émojis 🚀",
            type="test",
            description="Description with 中文 characters",
            properties={"unicode_key": "unicode_value_🌍"}
        )
        
        assert "émojis 🚀" in entity.name
        assert "中文" in entity.description
        assert "unicode_value_🌍" in entity.properties["unicode_key"]
    
    def test_knowledge_set_with_mixed_confidence_levels(self):
        """Test knowledge set with items having different confidence levels."""
        entities = [
            Entity(name="E1", type="person", confidence=0.1),  # LOW
            Entity(name="E2", type="place", confidence=0.5),   # MEDIUM
        ]
        concepts = [
            Concept(name="C1", description="D1", confidence=0.8),  # HIGH
            Concept(name="C2", description="D2", confidence=0.95), # VERY_HIGH
        ]
        
        ks = KnowledgeSet(entities=entities, concepts=concepts)
        stats = ks.get_stats()
        
        # Average should be (0.1 + 0.5 + 0.8 + 0.95) / 4 = 0.5875
        assert abs(stats["average_confidence"] - 0.5875) < 0.001
    
    def test_relationship_bidirectional_implications(self):
        """Test relationship bidirectional property."""
        rel_unidirectional = Relationship(
            source="A", target="B", type="leads_to", bidirectional=False
        )
        rel_bidirectional = Relationship(
            source="A", target="B", type="connected_to", bidirectional=True
        )
        
        assert rel_unidirectional.bidirectional is False
        assert rel_bidirectional.bidirectional is True
    
    def test_pattern_variables_and_examples(self):
        """Test pattern with variables and examples."""
        pattern = Pattern(
            name="Template Pattern",
            description="A template with variables",
            pattern_type="template",
            examples=["Example {var1} with {var2}", "Another {var1} example"],
            variables=["var1", "var2"],
            frequency="common"
        )
        
        assert len(pattern.examples) == 2
        assert len(pattern.variables) == 2
        assert "var1" in pattern.variables
        assert "var2" in pattern.variables
        assert "{var1}" in pattern.examples[0]
    
    def test_knowledge_set_with_large_content(self):
        """Test knowledge set with large amounts of content."""
        # Create large lists of items
        entities = [Entity(name=f"Entity_{i}", type="test") for i in range(100)]
        concepts = [Concept(name=f"Concept_{i}", description=f"Description_{i}") for i in range(50)]
        
        ks = KnowledgeSet(entities=entities, concepts=concepts)
        
        assert ks.total_items == 150
        assert len(ks.entities) == 100
        assert len(ks.concepts) == 50
        
        stats = ks.get_stats()
        assert stats["total_items"] == 150
        assert stats["entities_count"] == 100
        assert stats["concepts_count"] == 50