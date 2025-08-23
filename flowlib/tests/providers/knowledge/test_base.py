"""Tests for knowledge provider base functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any
from pydantic import ValidationError

from flowlib.providers.knowledge.base import (
    Knowledge, KnowledgeProvider, MultiDatabaseKnowledgeProvider
)


class TestKnowledge:
    """Test Knowledge model."""
    
    def test_knowledge_creation_valid(self):
        """Test creating Knowledge with valid data."""
        knowledge = Knowledge(
            content="Python is a programming language",
            source="vector",
            domain="programming",
            confidence=0.95,
            metadata={"type": "definition"}
        )
        
        assert knowledge.content == "Python is a programming language"
        assert knowledge.source == "vector"
        assert knowledge.domain == "programming"
        assert knowledge.confidence == 0.95
        assert knowledge.metadata == {"type": "definition"}
    
    def test_knowledge_creation_minimal(self):
        """Test creating Knowledge with minimal required fields."""
        knowledge = Knowledge(
            content="Test content",
            source="graph",
            domain="test",
            confidence=0.5
        )
        
        assert knowledge.content == "Test content"
        assert knowledge.source == "graph"
        assert knowledge.domain == "test"
        assert knowledge.confidence == 0.5
        assert knowledge.metadata == {}  # Default empty dict
    
    def test_knowledge_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence values
        Knowledge(
            content="Test",
            source="vector",
            domain="test",
            confidence=0.0  # Minimum valid
        )
        
        Knowledge(
            content="Test",
            source="vector",
            domain="test",
            confidence=1.0  # Maximum valid
        )
        
        # Invalid confidence values
        with pytest.raises(ValidationError):
            Knowledge(
                content="Test",
                source="vector",
                domain="test",
                confidence=-0.1  # Below minimum
            )
        
        with pytest.raises(ValidationError):
            Knowledge(
                content="Test",
                source="vector",
                domain="test",
                confidence=1.1  # Above maximum
            )
    
    def test_knowledge_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        # Missing content
        with pytest.raises(ValidationError) as exc_info:
            Knowledge(
                source="vector",
                domain="test",
                confidence=0.5
            )
        assert any(error['loc'] == ('content',) for error in exc_info.value.errors())
        
        # Missing source
        with pytest.raises(ValidationError) as exc_info:
            Knowledge(
                content="Test",
                domain="test",
                confidence=0.5
            )
        assert any(error['loc'] == ('source',) for error in exc_info.value.errors())
        
        # Missing domain
        with pytest.raises(ValidationError) as exc_info:
            Knowledge(
                content="Test",
                source="vector",
                confidence=0.5
            )
        assert any(error['loc'] == ('domain',) for error in exc_info.value.errors())
        
        # Missing confidence
        with pytest.raises(ValidationError) as exc_info:
            Knowledge(
                content="Test",
                source="vector",
                domain="test"
            )
        assert any(error['loc'] == ('confidence',) for error in exc_info.value.errors())
    
    def test_knowledge_serialization(self):
        """Test Knowledge serialization and deserialization."""
        original = Knowledge(
            content="Serialization test",
            source="hybrid",
            domain="testing",
            confidence=0.8,
            metadata={"id": 123, "tags": ["test", "serialize"]}
        )
        
        # Test dict conversion
        data = original.model_dump()
        expected = {
            "content": "Serialization test",
            "source": "hybrid",
            "domain": "testing",
            "confidence": 0.8,
            "metadata": {"id": 123, "tags": ["test", "serialize"]}
        }
        assert data == expected
        
        # Test recreation from dict
        recreated = Knowledge.model_validate(data)
        assert recreated == original
    
    def test_knowledge_json_serialization(self):
        """Test Knowledge JSON serialization."""
        knowledge = Knowledge(
            content="JSON test",
            source="vector",
            domain="json",
            confidence=0.75,
            metadata={"nested": {"key": "value"}}
        )
        
        json_str = knowledge.model_dump_json()
        assert isinstance(json_str, str)
        
        recreated = Knowledge.model_validate_json(json_str)
        assert recreated == knowledge
    
    def test_knowledge_complex_metadata(self):
        """Test Knowledge with complex metadata."""
        complex_metadata = {
            "entities": ["entity1", "entity2"],
            "relations": {
                "type": "is_a",
                "confidence": 0.9
            },
            "sources": [
                {"name": "source1", "url": "http://example.com"},
                {"name": "source2", "url": "http://example.org"}
            ],
            "processing": {
                "timestamp": "2023-01-01T00:00:00Z",
                "version": "1.0",
                "features": [1.0, 2.0, 3.0]
            }
        }
        
        knowledge = Knowledge(
            content="Complex metadata test",
            source="vector",
            domain="complex",
            confidence=0.9,
            metadata=complex_metadata
        )
        
        assert knowledge.metadata["entities"] == ["entity1", "entity2"]
        assert knowledge.metadata["relations"]["type"] == "is_a"
        assert len(knowledge.metadata["sources"]) == 2
        assert knowledge.metadata["processing"]["features"] == [1.0, 2.0, 3.0]


class MockKnowledgeProvider(KnowledgeProvider):
    """Mock implementation of KnowledgeProvider for testing."""
    
    domains = ["test", "mock"]
    
    def __init__(self):
        self.initialized = False
        self.config = None
        self.query_calls = []
    
    async def initialize(self, config: Dict[str, Any]):
        """Mock initialization."""
        self.initialized = True
        self.config = config
    
    async def query(self, domain: str, query: str, limit: int = 10) -> List[Knowledge]:
        """Mock query implementation."""
        self.query_calls.append((domain, query, limit))
        
        if domain not in self.domains:
            raise ValueError(f"Domain '{domain}' not supported")
        
        # Return mock results
        return [
            Knowledge(
                content=f"Mock result for '{query}' in domain '{domain}'",
                source="mock",
                domain=domain,
                confidence=0.8,
                metadata={"query": query, "limit": limit}
            )
        ]


class TestKnowledgeProvider:
    """Test KnowledgeProvider abstract base class."""
    
    def test_knowledge_provider_is_abstract(self):
        """Test that KnowledgeProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            KnowledgeProvider()
    
    def test_knowledge_provider_concrete_implementation(self):
        """Test concrete implementation of KnowledgeProvider."""
        provider = MockKnowledgeProvider()
        
        assert provider.domains == ["test", "mock"]
        assert not provider.initialized
    
    @pytest.mark.asyncio
    async def test_knowledge_provider_initialize(self):
        """Test provider initialization."""
        provider = MockKnowledgeProvider()
        config = {"param": "value"}
        
        await provider.initialize(config)
        
        assert provider.initialized
        assert provider.config == config
    
    @pytest.mark.asyncio
    async def test_knowledge_provider_query(self):
        """Test provider query functionality."""
        provider = MockKnowledgeProvider()
        await provider.initialize({})
        
        results = await provider.query("test", "test query", 5)
        
        assert len(results) == 1
        assert results[0].content == "Mock result for 'test query' in domain 'test'"
        assert results[0].domain == "test"
        assert results[0].confidence == 0.8
        
        # Check query was recorded
        assert len(provider.query_calls) == 1
        assert provider.query_calls[0] == ("test", "test query", 5)
    
    @pytest.mark.asyncio
    async def test_knowledge_provider_unsupported_domain(self):
        """Test query with unsupported domain."""
        provider = MockKnowledgeProvider()
        await provider.initialize({})
        
        with pytest.raises(ValueError, match="Domain 'unsupported' not supported"):
            await provider.query("unsupported", "test query")
    
    def test_knowledge_provider_supports_domain(self):
        """Test supports_domain method."""
        provider = MockKnowledgeProvider()
        
        assert provider.supports_domain("test") is True
        assert provider.supports_domain("mock") is True
        assert provider.supports_domain("unsupported") is False
    
    @pytest.mark.asyncio
    async def test_knowledge_provider_shutdown(self):
        """Test provider shutdown (default implementation)."""
        provider = MockKnowledgeProvider()
        
        # Should not raise exception
        await provider.shutdown()


class MockMultiDatabaseProvider(MultiDatabaseKnowledgeProvider):
    """Mock implementation of MultiDatabaseKnowledgeProvider."""
    
    domains = ["chemistry", "biology"]
    
    async def _query_vector(self, domain: str, query: str, limit: int) -> List[Knowledge]:
        """Mock vector query."""
        return [
            Knowledge(
                content=f"Vector result for '{query}'",
                source="vector",
                domain=domain,
                confidence=0.9,
                metadata={"type": "vector"}
            )
        ]
    
    async def _query_graph(self, domain: str, query: str, limit: int) -> List[Knowledge]:
        """Mock graph query."""
        return [
            Knowledge(
                content=f"Graph result for '{query}'",
                source="graph",
                domain=domain,
                confidence=0.85,
                metadata={"type": "graph"}
            )
        ]


class TestMultiDatabaseKnowledgeProvider:
    """Test MultiDatabaseKnowledgeProvider."""
    
    def test_multi_database_provider_initialization(self):
        """Test MultiDatabaseKnowledgeProvider initialization."""
        provider = MockMultiDatabaseProvider()
        
        assert provider.vector_db is None
        assert provider.graph_db is None
        assert provider._config == {}
    
    @pytest.mark.asyncio
    async def test_multi_database_provider_initialize_empty_config(self):
        """Test initialization with empty config."""
        provider = MockMultiDatabaseProvider()
        
        await provider.initialize({})
        
        assert provider.vector_db is None
        assert provider.graph_db is None
        assert provider._config == {}
    
    @pytest.mark.asyncio
    @patch('flowlib.providers.knowledge.base.ChromaDBProvider')
    @patch('flowlib.providers.knowledge.base.ChromaDBProviderSettings')
    async def test_multi_database_provider_initialize_vector_db(
        self, mock_settings_class, mock_provider_class
    ):
        """Test initialization with vector database config."""
        provider = MockMultiDatabaseProvider()
        
        # Mock the settings and provider
        mock_settings = Mock()
        mock_settings_class.return_value = mock_settings
        mock_vector_db = AsyncMock()
        mock_provider_class.return_value = mock_vector_db
        
        config = {
            "chromadb": {
                "connection": {
                    "host": "localhost",
                    "port": 8000
                }
            }
        }
        
        await provider.initialize(config)
        
        # Verify settings were created
        mock_settings_class.assert_called_once_with(host="localhost", port=8000)
        
        # Verify provider was created and initialized
        mock_provider_class.assert_called_once()
        mock_vector_db.initialize.assert_called_once()
        
        assert provider.vector_db == mock_vector_db
    
    @pytest.mark.asyncio
    @patch('flowlib.providers.graph.neo4j.provider.Neo4jProvider')
    @patch('flowlib.providers.graph.neo4j.provider.Neo4jProviderSettings')
    async def test_multi_database_provider_initialize_graph_db(
        self, mock_settings_class, mock_provider_class
    ):
        """Test initialization with graph database config."""
        provider = MockMultiDatabaseProvider()
        
        # Mock the settings and provider
        mock_settings = Mock()
        mock_settings_class.return_value = mock_settings
        mock_graph_db = AsyncMock()
        mock_provider_class.return_value = mock_graph_db
        
        config = {
            "neo4j": {
                "connection": {
                    "uri": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "password"
                }
            }
        }
        
        await provider.initialize(config)
        
        # Verify settings were created
        mock_settings_class.assert_called_once_with(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password"
        )
        
        # Verify provider was created and initialized
        mock_provider_class.assert_called_once()
        mock_graph_db.initialize.assert_called_once()
        
        assert provider.graph_db == mock_graph_db
    
    @pytest.mark.asyncio
    async def test_multi_database_provider_query_with_both_dbs(self):
        """Test query with both databases available."""
        provider = MockMultiDatabaseProvider()
        
        # Mock both databases
        provider.vector_db = AsyncMock()
        provider.graph_db = AsyncMock()
        
        await provider.initialize({})
        
        results = await provider.query("chemistry", "caffeine", 10)
        
        # Should get results from both vector and graph queries
        assert len(results) == 2
        
        # Results should be sorted by confidence (vector: 0.9, graph: 0.85)
        assert results[0].source == "vector"
        assert results[0].confidence == 0.9
        assert results[1].source == "graph"
        assert results[1].confidence == 0.85
    
    @pytest.mark.asyncio
    async def test_multi_database_provider_query_vector_only(self):
        """Test query with only vector database."""
        provider = MockMultiDatabaseProvider()
        provider.vector_db = AsyncMock()
        # graph_db remains None
        
        await provider.initialize({})
        
        results = await provider.query("chemistry", "caffeine", 10)
        
        assert len(results) == 1
        assert results[0].source == "vector"
    
    @pytest.mark.asyncio
    async def test_multi_database_provider_query_graph_only(self):
        """Test query with only graph database."""
        provider = MockMultiDatabaseProvider()
        provider.graph_db = AsyncMock()
        # vector_db remains None
        
        await provider.initialize({})
        
        results = await provider.query("chemistry", "caffeine", 10)
        
        assert len(results) == 1
        assert results[0].source == "graph"
    
    @pytest.mark.asyncio
    async def test_multi_database_provider_query_unsupported_domain(self):
        """Test query with unsupported domain."""
        provider = MockMultiDatabaseProvider()
        await provider.initialize({})
        
        with pytest.raises(ValueError, match="Domain 'unsupported' not supported"):
            await provider.query("unsupported", "test query")
    
    @pytest.mark.asyncio
    async def test_multi_database_provider_query_error_handling(self):
        """Test query error handling."""
        provider = MockMultiDatabaseProvider()
        
        # Mock databases that raise errors
        provider.vector_db = AsyncMock()
        provider.graph_db = AsyncMock()
        
        # Override the mock methods to raise exceptions
        async def failing_vector_query(domain, query, limit):
            raise Exception("Vector DB error")
        
        async def failing_graph_query(domain, query, limit):
            raise Exception("Graph DB error")
        
        provider._query_vector = failing_vector_query
        provider._query_graph = failing_graph_query
        
        await provider.initialize({})
        
        # Should not raise exception, but return empty results
        results = await provider.query("chemistry", "test", 10)
        assert results == []
    
    def test_multi_database_provider_fuse_and_rank_results(self):
        """Test result fusion and ranking."""
        provider = MockMultiDatabaseProvider()
        
        results = [
            Knowledge(
                content="High confidence result",
                source="vector",
                domain="test",
                confidence=0.95
            ),
            Knowledge(
                content="Medium confidence result",
                source="graph",
                domain="test",
                confidence=0.7
            ),
            Knowledge(
                content="Low confidence result",
                source="vector",
                domain="test",
                confidence=0.4
            ),
            Knowledge(
                content="High confidence result",  # Duplicate content
                source="graph",
                domain="test",
                confidence=0.8  # Lower than first instance
            )
        ]
        
        fused = provider._fuse_and_rank_results(results, 5)
        
        # Should deduplicate and sort by confidence
        assert len(fused) == 3  # One duplicate removed
        assert fused[0].confidence == 0.95  # Highest confidence first
        assert fused[1].confidence == 0.7
        assert fused[2].confidence == 0.4
    
    def test_multi_database_provider_fuse_empty_results(self):
        """Test fusion with empty results."""
        provider = MockMultiDatabaseProvider()
        
        fused = provider._fuse_and_rank_results([], 10)
        assert fused == []
    
    def test_multi_database_provider_fuse_with_limit(self):
        """Test fusion with result limit."""
        provider = MockMultiDatabaseProvider()
        
        results = [
            Knowledge(
                content=f"Result {i}",
                source="vector",
                domain="test",
                confidence=0.9 - i * 0.1
            )
            for i in range(10)
        ]
        
        fused = provider._fuse_and_rank_results(results, 3)
        
        assert len(fused) == 3
        # Should be top 3 by confidence
        assert fused[0].confidence == 0.9
        assert fused[1].confidence == 0.8
        assert fused[2].confidence == 0.7
    
    @pytest.mark.asyncio
    async def test_multi_database_provider_shutdown(self):
        """Test provider shutdown."""
        provider = MockMultiDatabaseProvider()
        
        # Mock databases
        mock_vector_db = AsyncMock()
        mock_graph_db = AsyncMock()
        provider.vector_db = mock_vector_db
        provider.graph_db = mock_graph_db
        
        await provider.shutdown()
        
        # Verify shutdown was called on both databases
        mock_vector_db.shutdown.assert_called_once()
        mock_graph_db.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multi_database_provider_shutdown_with_errors(self):
        """Test provider shutdown with errors."""
        provider = MockMultiDatabaseProvider()
        
        # Mock databases that raise errors on shutdown
        mock_vector_db = AsyncMock()
        mock_graph_db = AsyncMock()
        mock_vector_db.shutdown.side_effect = Exception("Vector shutdown error")
        mock_graph_db.shutdown.side_effect = Exception("Graph shutdown error")
        
        provider.vector_db = mock_vector_db
        provider.graph_db = mock_graph_db
        
        # Should not raise exception
        await provider.shutdown()
        
        # Verify both shutdowns were attempted
        mock_vector_db.shutdown.assert_called_once()
        mock_graph_db.shutdown.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])