"""Simplified tests for intelligent memory interface."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from typing import List, Dict, Any

from flowlib.agent.components.intelligence.knowledge import (
    Entity, Concept, Relationship, Pattern, KnowledgeSet
)


class TestIntelligentMemoryCore:
    """Test core memory functionality without decorators."""
    
    def test_memory_import(self):
        """Test that memory module can be imported."""
        from flowlib.agent.components.intelligence.memory import IntelligentMemory
        memory = IntelligentMemory()
        assert memory is not None
    
    def test_memory_private_methods(self):
        """Test memory private methods that don't use decorators."""
        from flowlib.agent.components.intelligence.memory import IntelligentMemory
        memory = IntelligentMemory()
        
        # Test deduplication methods
        entities = [
            Entity(name="Entity1", type="test"),
            Entity(name="Entity1", type="test"),  # Duplicate
            Entity(name="Entity2", type="test")
        ]
        
        unique_entities = memory._deduplicate_by_name(entities)
        assert len(unique_entities) == 2
        assert unique_entities[0].name == "Entity1"
        assert unique_entities[1].name == "Entity2"
    
    def test_relationship_deduplication(self):
        """Test relationship deduplication."""
        from flowlib.agent.components.intelligence.memory import IntelligentMemory
        memory = IntelligentMemory()
        
        relationships = [
            Relationship(source="A", target="B", type="connects"),
            Relationship(source="A", target="B", type="connects"),  # Duplicate
            Relationship(source="A", target="B", type="different"),  # Different type
            Relationship(source="B", target="C", type="connects")
        ]
        
        unique_rels = memory._deduplicate_relationships(relationships)
        assert len(unique_rels) == 3
    
    def test_knowledge_filtering(self):
        """Test knowledge type filtering."""
        from flowlib.agent.components.intelligence.memory import IntelligentMemory
        memory = IntelligentMemory()
        
        knowledge = KnowledgeSet(
            entities=[Entity(name="E1", type="test")],
            concepts=[Concept(name="C1", description="desc")],
            relationships=[Relationship(source="A", target="B", type="connects")],
            patterns=[Pattern(name="P1", description="pattern")]
        )
        
        # Filter for only entities
        filtered = memory._filter_by_types(knowledge, ['entities'])
        assert len(filtered.entities) == 1
        assert len(filtered.concepts) == 0
        assert len(filtered.relationships) == 0
        assert len(filtered.patterns) == 0
    
    def test_result_combination(self):
        """Test result combination logic."""
        from flowlib.agent.components.intelligence.memory import IntelligentMemory
        memory = IntelligentMemory()
        
        vector_results = KnowledgeSet(
            entities=[Entity(name="Entity1", type="test")],
            concepts=[Concept(name="Concept1", description="desc")]
        )
        
        graph_results = KnowledgeSet(
            relationships=[Relationship(source="A", target="B", type="connects")]
        )
        
        combined = memory._combine_results(vector_results, graph_results)
        assert len(combined.entities) == 1
        assert len(combined.concepts) == 1
        assert len(combined.relationships) == 1
        assert "Combined search results" in combined.summary
    
    @pytest.mark.asyncio
    async def test_working_memory_storage(self):
        """Test working memory storage fallback."""
        from flowlib.agent.components.intelligence.memory import IntelligentMemory
        memory = IntelligentMemory()
        
        knowledge = KnowledgeSet(
            entities=[Entity(name="Test", type="test")],
            summary="Test knowledge"
        )
        
        # Should not raise any errors
        await memory._store_in_working_memory(knowledge)


class TestKnowledgeManipulation:
    """Test knowledge manipulation and processing."""
    
    def test_entity_properties(self):
        """Test entity creation and properties."""
        entity = Entity(
            name="Python",
            type="programming_language",
            description="High-level programming language",
            confidence=0.9,
            properties={"category": "interpreted", "typing": "dynamic"}
        )
        
        assert entity.name == "Python"
        assert entity.type == "programming_language"
        assert entity.confidence == 0.9
        assert entity.properties["category"] == "interpreted"
        assert entity.confidence_level.value == "very_high"
    
    def test_concept_creation(self):
        """Test concept creation and validation."""
        concept = Concept(
            name="Machine Learning",
            description="AI subset focused on learning from data",
            category="artificial_intelligence",
            confidence=0.8,
            abstraction_level=3
        )
        
        assert concept.name == "Machine Learning"
        assert concept.category == "artificial_intelligence"
        assert concept.abstraction_level == 3
        assert concept.confidence_level.value == "high"
    
    def test_relationship_properties(self):
        """Test relationship creation and properties."""
        relationship = Relationship(
            source="Python",
            target="Machine Learning",
            type="used_for",
            description="Python is commonly used for ML",
            confidence=0.85,
            bidirectional=False,
            strength="strong"
        )
        
        assert relationship.source == "Python"
        assert relationship.target == "Machine Learning"
        assert relationship.type == "used_for"
        assert relationship.strength == "strong"
        assert relationship.bidirectional is False
    
    def test_pattern_creation(self):
        """Test pattern creation and validation."""
        pattern = Pattern(
            name="MVC Pattern",
            description="Model-View-Controller architectural pattern",
            pattern_type="architectural",
            examples=["Django MVC", "Rails MVC"],
            variables=["Model", "View", "Controller"],
            confidence=0.9,
            frequency="common"
        )
        
        assert pattern.name == "MVC Pattern"
        assert pattern.pattern_type == "architectural"
        assert len(pattern.examples) == 2
        assert len(pattern.variables) == 3
        assert pattern.frequency == "common"
    
    def test_knowledge_set_statistics(self):
        """Test knowledge set statistics and aggregation."""
        entities = [Entity(name="E1", type="test", confidence=0.8)]
        concepts = [Concept(name="C1", description="desc", confidence=0.6)]
        relationships = [Relationship(source="A", target="B", type="connects", confidence=0.9)]
        
        knowledge = KnowledgeSet(
            entities=entities,
            concepts=concepts,
            relationships=relationships,
            confidence=0.75
        )
        
        stats = knowledge.get_stats()
        assert stats["total_items"] == 3
        assert stats["entities_count"] == 1
        assert stats["concepts_count"] == 1
        assert stats["relationships_count"] == 1
        assert stats["confidence_level"] == "high"
        
        # Average confidence: (0.8 + 0.6 + 0.9) / 3 = 0.767
        assert abs(stats["average_confidence"] - 0.7666666666666667) < 0.001


class TestMemoryAPIMocking:
    """Test memory API with proper mocking."""
    
    @pytest.mark.asyncio 
    async def test_global_memory_api(self):
        """Test global memory API functions."""
        from flowlib.agent.components.intelligence.memory import get_memory, remember, recall
        
        # Test singleton behavior
        memory1 = await get_memory()
        memory2 = await get_memory() 
        assert memory1 is memory2
    
    @pytest.mark.asyncio
    async def test_remember_recall_api_mocking(self):
        """Test remember/recall API with mocking."""
        from flowlib.agent.components.intelligence.memory import remember, recall, IntelligentMemory
        
        knowledge = KnowledgeSet(
            entities=[Entity(name="Test Entity", type="test")],
            summary="Test knowledge"
        )
        
        # Mock the memory methods
        with patch.object(IntelligentMemory, 'store_knowledge') as mock_store, \
             patch.object(IntelligentMemory, 'retrieve_knowledge') as mock_retrieve:
            
            # Test remember
            await remember(knowledge)
            mock_store.assert_called_once_with(knowledge)
            
            # Test recall
            expected_result = KnowledgeSet(summary="Retrieved")
            mock_retrieve.return_value = expected_result
            
            result = await recall("test query", ['entities'], 5)
            assert result == expected_result
            mock_retrieve.assert_called_once_with("test query", ['entities'], 5)


class TestMemoryErrorHandling:
    """Test memory error handling and edge cases."""
    
    def test_empty_knowledge_handling(self):
        """Test handling of empty knowledge sets."""
        empty_knowledge = KnowledgeSet()
        assert empty_knowledge.total_items == 0
        assert len(empty_knowledge.entities) == 0
        assert len(empty_knowledge.concepts) == 0
        assert len(empty_knowledge.relationships) == 0
        assert len(empty_knowledge.patterns) == 0
    
    def test_knowledge_with_unicode(self):
        """Test knowledge with unicode characters."""
        entity = Entity(
            name="测试实体",
            type="test",
            description="Unicode description with émojis 🚀",
            properties={"unicode_prop": "值"}
        )
        
        knowledge = KnowledgeSet(entities=[entity])
        assert knowledge.total_items == 1
        assert "测试实体" in knowledge.entities[0].name
        assert "🚀" in knowledge.entities[0].description
    
    def test_large_knowledge_set(self):
        """Test handling of large knowledge sets."""
        # Create large number of entities
        entities = [Entity(name=f"Entity_{i}", type="test") for i in range(1000)]
        large_knowledge = KnowledgeSet(entities=entities)
        
        assert large_knowledge.total_items == 1000
        assert len(large_knowledge.entities) == 1000
        
        # Test statistics calculation
        stats = large_knowledge.get_stats()
        assert stats["entities_count"] == 1000
        assert stats["total_items"] == 1000
    
    def test_knowledge_confidence_calculations(self):
        """Test confidence level calculations across different ranges."""
        # Test different confidence levels
        low_entity = Entity(name="Low", type="test", confidence=0.2)
        medium_entity = Entity(name="Medium", type="test", confidence=0.5) 
        high_entity = Entity(name="High", type="test", confidence=0.8)
        very_high_entity = Entity(name="VeryHigh", type="test", confidence=0.95)
        
        knowledge = KnowledgeSet(entities=[low_entity, medium_entity, high_entity, very_high_entity])
        
        # Average confidence: (0.2 + 0.5 + 0.8 + 0.95) / 4 = 0.6125
        stats = knowledge.get_stats()
        assert abs(stats["average_confidence"] - 0.6125) < 0.001
    
    def test_knowledge_edge_cases(self):
        """Test various edge cases in knowledge processing."""
        # Test with None values handled gracefully by dataclass defaults
        entity = Entity(name="Test", type="test")
        assert entity.description is None
        assert entity.confidence == 1.0
        assert entity.properties == {}
        
        # Test relationship with defaults
        rel = Relationship(source="A", target="B", type="connects")
        assert rel.description is None
        assert rel.confidence == 1.0
        assert rel.bidirectional is False
        assert rel.strength == "medium"