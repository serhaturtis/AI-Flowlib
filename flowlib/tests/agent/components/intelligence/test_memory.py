"""Tests for intelligence memory module."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from flowlib.agent.components.intelligence.memory import (
    IntelligentMemory,
    get_memory,
    remember,
    recall
)
from flowlib.agent.components.intelligence.knowledge import (
    KnowledgeSet,
    Entity,
    Concept,
    Relationship,
    Pattern
)


class TestIntelligentMemory:
    """Test IntelligentMemory class."""
    
    @pytest.fixture
    def memory(self):
        """Create memory instance for testing."""
        return IntelligentMemory()
    
    @pytest.fixture
    def sample_knowledge(self):
        """Create sample knowledge set."""
        return KnowledgeSet(
            entities=[
                Entity(name="Python", type="language", description="Programming language", confidence=0.9),
                Entity(name="AI", type="technology", description="Artificial Intelligence", confidence=0.95)
            ],
            concepts=[
                Concept(name="Machine Learning", description="ML concept", confidence=0.85),
                Concept(name="Deep Learning", description="DL concept", confidence=0.88)
            ],
            relationships=[
                Relationship(source="Python", target="AI", type="used_for", 
                           description="Python is used for AI", confidence=0.9),
                Relationship(source="Machine Learning", target="Deep Learning", 
                           type="contains", description="ML contains DL", confidence=0.95)
            ],
            patterns=[
                Pattern(name="Growth Pattern", description="AI growth", confidence=0.8)
            ],
            summary="Sample knowledge set"
        )
    
    @pytest.fixture
    def mock_vector_provider(self):
        """Mock vector provider."""
        provider = Mock()
        provider.add_vectors = AsyncMock()
        provider.search = AsyncMock(return_value=[
            {
                'document': 'Python: Programming language',
                'metadata': {'type': 'Entity', 'name': 'Python', 'confidence': 0.9},
                'score': 0.95
            }
        ])
        return provider
    
    @pytest.fixture
    def mock_graph_provider(self):
        """Mock graph provider."""
        provider = Mock()
        provider.add_relationship = AsyncMock()
        return provider
    
    @pytest.mark.asyncio
    async def test_store_knowledge_all_providers(self, memory, sample_knowledge, 
                                                mock_vector_provider, mock_graph_provider):
        """Test storing knowledge with all providers available."""
        # Bypass the @inject decorator by calling the underlying method directly
        await memory.store_knowledge.__wrapped__(
            memory,
            sample_knowledge,
            vector_provider=mock_vector_provider,
            graph_provider=mock_graph_provider
        )
        
        # Verify vector storage was called
        assert mock_vector_provider.add_vectors.called
        call_args = mock_vector_provider.add_vectors.call_args[1]
        assert len(call_args['ids']) == 4  # 2 entities + 2 concepts
        assert len(call_args['documents']) == 4
        assert len(call_args['metadatas']) == 4
        
        # Verify graph storage was called
        assert mock_graph_provider.add_relationship.call_count == 2
    
    @pytest.mark.asyncio
    async def test_store_knowledge_vector_only(self, memory, sample_knowledge, mock_vector_provider):
        """Test storing knowledge with only vector provider."""
        await memory.store_knowledge.__wrapped__(
            memory,
            sample_knowledge,
            vector_provider=mock_vector_provider,
            graph_provider=None
        )
        
        # Verify vector storage was called
        assert mock_vector_provider.add_vectors.called
        
    @pytest.mark.asyncio
    async def test_store_knowledge_graph_only(self, memory, sample_knowledge, mock_graph_provider):
        """Test storing knowledge with only graph provider."""
        await memory.store_knowledge.__wrapped__(
            memory,
            sample_knowledge,
            vector_provider=None,
            graph_provider=mock_graph_provider
        )
        
        # Verify graph storage was called
        assert mock_graph_provider.add_relationship.call_count == 2
    
    @pytest.mark.asyncio
    async def test_store_knowledge_no_providers(self, memory, sample_knowledge):
        """Test storing knowledge with no providers (fallback to working memory)."""
        # Should not raise exception
        await memory.store_knowledge.__wrapped__(
            memory,
            sample_knowledge,
            vector_provider=None,
            graph_provider=None
        )
    
    @pytest.mark.asyncio
    async def test_store_knowledge_provider_failure(self, memory, sample_knowledge):
        """Test graceful degradation when providers fail."""
        failing_vector = Mock()
        failing_vector.add_vectors = AsyncMock(side_effect=Exception("Vector DB error"))
        
        failing_graph = Mock()
        failing_graph.add_relationship = AsyncMock(side_effect=Exception("Graph DB error"))
        
        # Should not raise exception due to graceful degradation
        await memory.store_knowledge.__wrapped__(
            memory,
            sample_knowledge,
            vector_provider=failing_vector,
            graph_provider=failing_graph
        )
    
    @pytest.mark.asyncio
    async def test_retrieve_knowledge_basic(self, memory):
        """Test basic knowledge retrieval."""
        with patch.object(memory, '_search_vector', AsyncMock(return_value=KnowledgeSet(
            entities=[Entity(name="Python", type="language", confidence=0.9)],
            summary="Vector results"
        ))):
            with patch.object(memory, '_search_graph', AsyncMock(return_value=KnowledgeSet())):
                result = await memory.retrieve_knowledge("Python")
                
                assert len(result.entities) == 1
                assert result.entities[0].name == "Python"
    
    @pytest.mark.asyncio
    async def test_retrieve_knowledge_with_filters(self, memory):
        """Test knowledge retrieval with type filters."""
        test_knowledge = KnowledgeSet(
            entities=[Entity(name="Python", type="language", confidence=0.9)],
            concepts=[Concept(name="ML", description="Machine Learning", confidence=0.85)],
            relationships=[Relationship(source="A", target="B", type="related", confidence=0.8)]
        )
        
        with patch.object(memory, '_search_vector', AsyncMock(return_value=test_knowledge)):
            with patch.object(memory, '_search_graph', AsyncMock(return_value=KnowledgeSet())):
                # Filter for entities only
                result = await memory.retrieve_knowledge("test", knowledge_types=['entities'])
                assert len(result.entities) == 1
                assert len(result.concepts) == 0
                assert len(result.relationships) == 0
                
                # Filter for concepts and relationships
                result = await memory.retrieve_knowledge("test", knowledge_types=['concepts', 'relationships'])
                assert len(result.entities) == 0
                assert len(result.concepts) == 1
                assert len(result.relationships) == 1
    
    @pytest.mark.asyncio
    async def test_store_in_vector_empty_items(self, memory, mock_vector_provider):
        """Test vector storage with empty items list."""
        await memory._store_in_vector([], mock_vector_provider)
        assert not mock_vector_provider.add_vectors.called
    
    @pytest.mark.asyncio
    async def test_store_in_vector_metadata_creation(self, memory, mock_vector_provider):
        """Test proper metadata creation during vector storage."""
        items = [
            Entity(name="Test Entity", type="test", description="A test entity", confidence=0.75),
            Concept(name="Test Concept", description="A test concept", confidence=0.82)
        ]
        
        await memory._store_in_vector(items, mock_vector_provider)
        
        call_args = mock_vector_provider.add_vectors.call_args[1]
        assert call_args['metadatas'][0]['type'] == 'Entity'
        assert call_args['metadatas'][0]['confidence'] == 0.75
        assert call_args['metadatas'][0]['name'] == 'Test Entity'
        
        assert call_args['metadatas'][1]['type'] == 'Concept'
        assert call_args['metadatas'][1]['confidence'] == 0.82
        assert call_args['metadatas'][1]['name'] == 'Test Concept'
    
    @pytest.mark.asyncio
    async def test_store_in_graph_properties(self, memory, mock_graph_provider):
        """Test graph storage with relationship properties."""
        relationships = [
            Relationship(
                source="A", target="B", type="connected",
                description="A is connected to B", confidence=0.85,
                bidirectional=True
            )
        ]
        
        await memory._store_in_graph(relationships, mock_graph_provider)
        
        call_args = mock_graph_provider.add_relationship.call_args[1]
        assert call_args['source_id'] == "A"
        assert call_args['target_id'] == "B"
        assert call_args['relationship_type'] == "connected"
        assert call_args['properties']['description'] == "A is connected to B"
        assert call_args['properties']['confidence'] == 0.85
        assert call_args['properties']['bidirectional'] is True
    
    @pytest.mark.asyncio
    async def test_search_vector_conversion(self, memory):
        """Test conversion of vector search results to knowledge objects."""
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config', 
                  AsyncMock(return_value=Mock(search=AsyncMock(return_value=[
                      {
                          'document': 'Entity description',
                          'metadata': {'type': 'Entity', 'name': 'TestEntity', 'confidence': 0.9}
                      },
                      {
                          'document': 'Concept description',
                          'metadata': {'type': 'Concept', 'name': 'TestConcept', 'confidence': 0.85}
                      }
                  ])))):
            result = await memory._search_vector("test query", limit=10)
            
            assert len(result.entities) == 1
            assert result.entities[0].name == 'TestEntity'
            assert result.entities[0].confidence == 0.9
            
            assert len(result.concepts) == 1
            assert result.concepts[0].name == 'TestConcept'
            assert result.concepts[0].confidence == 0.85
    
    def test_combine_results(self, memory):
        """Test combining results from multiple sources."""
        vector_results = KnowledgeSet(
            entities=[Entity(name="A", type="type1", confidence=0.9)],
            concepts=[Concept(name="Concept1", description="desc", confidence=0.8)]
        )
        
        graph_results = KnowledgeSet(
            entities=[Entity(name="B", type="type2", confidence=0.85)],
            relationships=[Relationship(source="A", target="B", type="related", confidence=0.7)]
        )
        
        combined = memory._combine_results(vector_results, graph_results)
        
        assert len(combined.entities) == 2
        assert len(combined.concepts) == 1
        assert len(combined.relationships) == 1
    
    def test_deduplicate_by_name(self, memory):
        """Test deduplication of items by name."""
        items = [
            Entity(name="A", type="type1", confidence=0.9),
            Entity(name="B", type="type2", confidence=0.8),
            Entity(name="A", type="type3", confidence=0.7),  # Duplicate name
        ]
        
        unique = memory._deduplicate_by_name(items)
        assert len(unique) == 2
        assert unique[0].name == "A"
        assert unique[0].confidence == 0.9  # First occurrence kept
        assert unique[1].name == "B"
    
    def test_deduplicate_relationships(self, memory):
        """Test deduplication of relationships."""
        relationships = [
            Relationship(source="A", target="B", type="related", confidence=0.9),
            Relationship(source="C", target="D", type="connected", confidence=0.8),
            Relationship(source="A", target="B", type="related", confidence=0.7),  # Duplicate
        ]
        
        unique = memory._deduplicate_relationships(relationships)
        assert len(unique) == 2
        assert unique[0].confidence == 0.9  # First occurrence kept
    
    def test_filter_by_types(self, memory):
        """Test filtering knowledge by types."""
        knowledge = KnowledgeSet(
            entities=[Entity(name="E1", type="t1", confidence=0.9)],
            concepts=[Concept(name="C1", description="d1", confidence=0.8)],
            relationships=[Relationship(source="A", target="B", type="r1", confidence=0.7)],
            patterns=[Pattern(name="P1", description="p1", confidence=0.6)]
        )
        
        # Test single type filter
        filtered = memory._filter_by_types(knowledge, ['entities'])
        assert len(filtered.entities) == 1
        assert len(filtered.concepts) == 0
        assert len(filtered.relationships) == 0
        assert len(filtered.patterns) == 0
        
        # Test multiple type filter
        filtered = memory._filter_by_types(knowledge, ['concepts', 'patterns'])
        assert len(filtered.entities) == 0
        assert len(filtered.concepts) == 1
        assert len(filtered.relationships) == 0
        assert len(filtered.patterns) == 1


class TestMemoryAPIs:
    """Test global memory APIs."""
    
    @pytest.mark.asyncio
    async def test_get_memory_singleton(self):
        """Test that get_memory returns singleton instance."""
        # Reset global instance
        import flowlib.agent.components.intelligence.memory as mem_module
        mem_module._memory_instance = None
        
        memory1 = await get_memory()
        memory2 = await get_memory()
        
        assert memory1 is memory2
        assert isinstance(memory1, IntelligentMemory)
    
    @pytest.mark.asyncio
    async def test_remember_api(self):
        """Test simple remember API."""
        knowledge = KnowledgeSet(
            entities=[Entity(name="Test", type="test", confidence=0.9)]
        )
        
        with patch('flowlib.agent.components.intelligence.memory.get_memory', 
                  AsyncMock(return_value=Mock(store_knowledge=AsyncMock()))):
            await remember(knowledge)
    
    @pytest.mark.asyncio
    async def test_recall_api(self):
        """Test simple recall API."""
        expected_knowledge = KnowledgeSet(
            entities=[Entity(name="Result", type="test", confidence=0.85)]
        )
        
        with patch('flowlib.agent.components.intelligence.memory.get_memory',
                  AsyncMock(return_value=Mock(
                      retrieve_knowledge=AsyncMock(return_value=expected_knowledge)
                  ))):
            result = await recall("test query", knowledge_types=['entities'], limit=5)
            assert result == expected_knowledge