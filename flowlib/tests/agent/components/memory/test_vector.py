"""
Tests for the modernized VectorMemory component.

These tests verify:
1. Config-driven provider access
2. Semantic search capabilities
3. Vector embedding operations
4. Provider lifecycle management
5. Error handling and fallbacks
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from flowlib.agent.components.memory.vector import VectorMemory, VectorMemoryConfig
from flowlib.agent.components.memory.models import MemoryItem, MemoryStoreRequest, MemoryRetrieveRequest, MemorySearchRequest
from flowlib.agent.core.errors import MemoryError


@pytest.fixture
def mock_vector_provider():
    """Create a mock vector database provider."""
    provider = AsyncMock()
    provider.list_collections.return_value = []
    provider.create_collection.return_value = None
    provider.add_documents.return_value = None
    provider.query.return_value = {
        'documents': [['test document']],
        'metadatas': [[{'context': 'test', 'key': 'test_key'}]],
        'distances': [[0.1]],
        'ids': [['test_id']]
    }
    provider.delete.return_value = None
    return provider


@pytest.fixture
def mock_embedding_provider():
    """Create a mock embedding provider."""
    provider = AsyncMock()
    provider.embed_texts.return_value = [[0.1, 0.2, 0.3, 0.4]]  # Mock embedding vector
    return provider


def create_mock_registry(mock_vector_provider, mock_embedding_provider):
    """Helper to create properly configured mock registry."""
    async def mock_get_by_config(config_name):
        if "vector" in config_name:
            return mock_vector_provider
        elif "embedding" in config_name:
            return mock_embedding_provider
        return None
    return mock_get_by_config


class TestVectorMemoryConfig:
    """Test VectorMemoryConfig validation and defaults."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = VectorMemoryConfig()
        assert config.vector_provider_config == "default-vector-db"
        assert config.embedding_provider_config == "default-embedding"
        assert config.collection_name == "agent_memory"
        assert config.similarity_threshold == 0.7
        assert config.max_results == 50
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = VectorMemoryConfig(
            vector_provider_config="custom-vector",
            embedding_provider_config="custom-embedding",
            collection_name="custom_collection",
            similarity_threshold=0.8,
            max_results=100,
            embedding_dimensions=512
        )
        assert config.vector_provider_config == "custom-vector"
        assert config.embedding_provider_config == "custom-embedding"
        assert config.collection_name == "custom_collection"
        assert config.similarity_threshold == 0.8
        assert config.max_results == 100
        assert config.embedding_dimensions == 512
    
    def test_config_validation(self):
        """Test configuration validation constraints."""
        with pytest.raises(ValueError):
            VectorMemoryConfig(similarity_threshold=1.5)  # Above maximum
        
        with pytest.raises(ValueError):
            VectorMemoryConfig(similarity_threshold=-0.1)  # Below minimum
        
        with pytest.raises(ValueError):
            VectorMemoryConfig(max_results=0)  # Below minimum


class TestVectorMemoryInitialization:
    """Test VectorMemory initialization and provider management."""
    
    @pytest.mark.asyncio
    async def test_successful_initialization(self, mock_vector_provider, mock_embedding_provider):
        """Test successful initialization with mock providers."""
        config = VectorMemoryConfig()
        memory = VectorMemory(config)
        
        with patch('flowlib.agent.components.memory.vector.provider_registry') as mock_registry:
            mock_registry.get_by_config.side_effect = create_mock_registry(mock_vector_provider, mock_embedding_provider)
            
            await memory.initialize()
            
            assert memory.initialized is True
            assert memory._vector_provider == mock_vector_provider
            assert memory._embedding_provider == mock_embedding_provider
            
            # Verify provider registry calls
            assert mock_registry.get_by_config.call_count == 2
            mock_registry.get_by_config.assert_any_call("default-vector-db")
            mock_registry.get_by_config.assert_any_call("default-embedding")
            
            await memory.shutdown()
    
    @pytest.mark.asyncio
    async def test_initialization_missing_vector_provider(self):
        """Test initialization failure when vector provider is missing."""
        memory = VectorMemory(VectorMemoryConfig())
        
        with patch('flowlib.agent.components.memory.vector.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                if "vector" in config_name:
                    return None
                elif "embedding" in config_name:
                    return mock_embedding_provider
                return None
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            with pytest.raises(MemoryError, match="Vector provider not found"):
                await memory.initialize()
            
            assert memory.initialized is False
    
    @pytest.mark.asyncio
    async def test_initialization_missing_embedding_provider(self, mock_vector_provider):
        """Test initialization failure when embedding provider is missing."""
        memory = VectorMemory(VectorMemoryConfig())
        
        with patch('flowlib.agent.components.memory.vector.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                if "vector" in config_name:
                    return mock_vector_provider
                elif "embedding" in config_name:
                    return None
                return None
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            with pytest.raises(MemoryError, match="Embedding provider not found"):
                await memory.initialize()
            
            assert memory.initialized is False
    
    @pytest.mark.asyncio
    async def test_double_initialization(self, mock_vector_provider, mock_embedding_provider):
        """Test that double initialization is handled gracefully."""
        memory = VectorMemory(VectorMemoryConfig())
        
        with patch('flowlib.agent.components.memory.vector.provider_registry') as mock_registry:
            mock_registry.get_by_config.side_effect = create_mock_registry(mock_vector_provider, mock_embedding_provider)
            
            await memory.initialize()
            assert memory.initialized is True
            
            # Second initialization should be no-op
            await memory.initialize()
            assert memory.initialized is True
            
            # Should only call registry once per provider
            assert mock_registry.get_by_config.call_count == 2
            
            await memory.shutdown()


class TestVectorMemoryCollectionManagement:
    """Test vector collection creation and management."""
    
    @pytest.mark.asyncio
    async def test_collection_creation(self, mock_vector_provider, mock_embedding_provider):
        """Test that collections are created if they don't exist."""
        config = VectorMemoryConfig(collection_name="test_collection", embedding_dimensions=384)
        memory = VectorMemory(config)
        
        # Mock empty collection list
        mock_vector_provider.list_collections.return_value = []
        
        with patch('flowlib.agent.components.memory.vector.provider_registry') as mock_registry:
            mock_registry.get_by_config.side_effect = create_mock_registry(mock_vector_provider, mock_embedding_provider)
            
            await memory.initialize()
            
            # Verify collection creation was called
            mock_vector_provider.create_collection.assert_called_once_with(
                name="test_collection",
                dimension=384
            )
            
            await memory.shutdown()
    
    @pytest.mark.asyncio
    async def test_existing_collection(self, mock_vector_provider, mock_embedding_provider):
        """Test that existing collections are not recreated."""
        config = VectorMemoryConfig(collection_name="existing_collection")
        memory = VectorMemory(config)
        
        # Mock existing collection
        mock_vector_provider.list_collections.return_value = ["existing_collection"]
        
        with patch('flowlib.agent.components.memory.vector.provider_registry') as mock_registry:
            mock_registry.get_by_config.side_effect = create_mock_registry(mock_vector_provider, mock_embedding_provider)
            
            await memory.initialize()
            
            # Verify collection creation was NOT called
            mock_vector_provider.create_collection.assert_not_called()
            
            await memory.shutdown()


class TestVectorMemoryStorage:
    """Test vector memory storage operations."""
    
    @pytest.mark.asyncio
    async def test_store_memory_item(self, mock_vector_provider, mock_embedding_provider):
        """Test storing a memory item with embedding generation."""
        memory = VectorMemory(VectorMemoryConfig())
        
        with patch('flowlib.agent.components.memory.vector.provider_registry') as mock_registry:
            mock_registry.get_by_config.side_effect = create_mock_registry(mock_vector_provider, mock_embedding_provider)
            
            await memory.initialize()
            
            try:
                # Store a memory item
                request = MemoryStoreRequest(key="test_key", value="Test content for embedding", context="test_context", metadata={"source": "test"})
                
                result = await memory.store(request)
                
                assert result == "test_key"
                
                # Verify embedding generation
                mock_embedding_provider.embed_texts.assert_called_once_with(["Test content for embedding"])
                
                # Verify vector storage
                mock_vector_provider.add_documents.assert_called_once()
                call_args = mock_vector_provider.add_documents.call_args
                
                assert call_args[1]["collection_name"] == "agent_memory"
                assert call_args[1]["documents"] == ["Test content for embedding"]
                assert call_args[1]["embeddings"] == [[0.1, 0.2, 0.3, 0.4]]
                assert call_args[1]["ids"] == ["test_context_test_key"]
                
                # Check metadata
                metadata = call_args[1]["metadatas"][0]
                assert metadata["context"] == "test_context"
                assert metadata["key"] == "test_key"
                assert metadata["content"] == "Test content for embedding"
                assert metadata["source"] == "test"
                
            finally:
                await memory.shutdown()
    
    @pytest.mark.asyncio
    async def test_store_uninitialized(self):
        """Test that storing fails on uninitialized memory."""
        memory = VectorMemory(VectorMemoryConfig())
        
        request = MemoryStoreRequest(key="test", value="test", context="test")
        
        with pytest.raises(MemoryError, match="not initialized"):
            await memory.store(request)
    
    @pytest.mark.asyncio
    async def test_store_embedding_failure(self, mock_vector_provider, mock_embedding_provider):
        """Test handling of embedding generation failures."""
        memory = VectorMemory(VectorMemoryConfig())
        
        # Mock embedding failure
        mock_embedding_provider.embed_texts.return_value = []  # Empty result
        
        with patch('flowlib.agent.components.memory.vector.provider_registry') as mock_registry:
            mock_registry.get_by_config.side_effect = create_mock_registry(mock_vector_provider, mock_embedding_provider)
            
            await memory.initialize()
            
            try:
                request = MemoryStoreRequest(key="test", value="test", context="test")
                
                with pytest.raises(MemoryError, match="Embedding provider returned empty result"):
                    await memory.store(request)
                    
            finally:
                await memory.shutdown()


class TestVectorMemoryRetrieval:
    """Test vector memory retrieval operations."""
    
    @pytest.mark.asyncio
    async def test_retrieve_existing_item(self, mock_vector_provider, mock_embedding_provider):
        """Test retrieving an existing memory item."""
        memory = VectorMemory(VectorMemoryConfig())
        
        # Mock query result
        mock_vector_provider.query.return_value = {
            'documents': [['Retrieved content']],
            'metadatas': [[{
                'context': 'test_context',
                'key': 'test_key',
                'content': 'Retrieved content',
                'source': 'test'
            }]],
            'distances': [[0.0]],
            'ids': [['test_context_test_key']]
        }
        
        with patch('flowlib.agent.components.memory.vector.provider_registry') as mock_registry:
            mock_registry.get_by_config.side_effect = create_mock_registry(mock_vector_provider, mock_embedding_provider)
            
            await memory.initialize()
            
            try:
                request = MemoryRetrieveRequest(key="test_key", context="test_context")
                result = await memory.retrieve(request)
                
                assert result is not None
                assert result.value == "Retrieved content"
                assert result.metadata['source'] == 'test'
                assert result.metadata['context'] == 'test_context'
                
                # Verify query parameters
                mock_vector_provider.query.assert_called_once()
                call_args = mock_vector_provider.query.call_args
                assert call_args[1]["collection_name"] == "agent_memory"
                assert call_args[1]["query_embeddings"] is None  # Exact match query
                assert call_args[1]["where"] == {"context": "test_context", "key": "test_key"}
                
            finally:
                await memory.shutdown()
    
    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_item(self, mock_vector_provider, mock_embedding_provider):
        """Test retrieving a non-existent memory item."""
        memory = VectorMemory(VectorMemoryConfig())
        
        # Mock empty query result
        mock_vector_provider.query.return_value = {
            'documents': [],
            'metadatas': [],
            'distances': [],
            'ids': []
        }
        
        with patch('flowlib.agent.components.memory.vector.provider_registry') as mock_registry:
            mock_registry.get_by_config.side_effect = create_mock_registry(mock_vector_provider, mock_embedding_provider)
            
            await memory.initialize()
            
            try:
                request = MemoryRetrieveRequest(key="nonexistent", context="test")
                result = await memory.retrieve(request)
                
                assert result is None
                
            finally:
                await memory.shutdown()


class TestVectorMemorySearch:
    """Test vector memory search operations."""
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, mock_vector_provider, mock_embedding_provider):
        """Test semantic search functionality."""
        memory = VectorMemory(VectorMemoryConfig(similarity_threshold=0.5))
        
        # Mock search results
        mock_vector_provider.query.return_value = {
            'documents': [['First result', 'Second result']],
            'metadatas': [[
                {'key': 'item1', 'context': 'test', 'source': 'test1'},
                {'key': 'item2', 'context': 'test', 'source': 'test2'}
            ]],
            'distances': [[0.3, 0.4]],  # Good similarity scores
            'ids': [['id1', 'id2']]
        }
        
        with patch('flowlib.agent.components.memory.vector.provider_registry') as mock_registry:
            mock_registry.get_by_config.side_effect = create_mock_registry(mock_vector_provider, mock_embedding_provider)
            
            await memory.initialize()
            
            try:
                request = MemorySearchRequest(query="search query", context="test", limit=10)
                results = await memory.search(request)
                
                assert len(results) == 2
                
                # Check first result
                assert results[0].item.key == "item1"
                assert results[0].item.value == "First result"
                assert results[0].score == 0.7  # 1 - 0.3 distance
                assert results[0].metadata["search_type"] == "semantic"
                
                # Check second result
                assert results[1].item.key == "item2"
                assert results[1].item.value == "Second result"
                assert results[1].score == 0.6  # 1 - 0.4 distance
                
                # Verify embedding generation for query
                mock_embedding_provider.embed_texts.assert_called_once_with(["search query"])
                
                # Verify vector search
                mock_vector_provider.query.assert_called_once()
                call_args = mock_vector_provider.query.call_args
                assert call_args[1]["query_embeddings"] == [[0.1, 0.2, 0.3, 0.4]]
                assert call_args[1]["n_results"] == 10
                
            finally:
                await memory.shutdown()
    
    @pytest.mark.asyncio
    async def test_search_similarity_threshold(self, mock_vector_provider, mock_embedding_provider):
        """Test that similarity threshold filters results."""
        memory = VectorMemory(VectorMemoryConfig(similarity_threshold=0.8))
        
        # Mock results with varying similarity
        mock_vector_provider.query.return_value = {
            'documents': [['High similarity', 'Low similarity']],
            'metadatas': [[
                {'key': 'good', 'context': 'test'},
                {'key': 'bad', 'context': 'test'}
            ]],
            'distances': [[0.1, 0.9]],  # One good, one bad similarity
            'ids': [['id1', 'id2']]
        }
        
        with patch('flowlib.agent.components.memory.vector.provider_registry') as mock_registry:
            mock_registry.get_by_config.side_effect = create_mock_registry(mock_vector_provider, mock_embedding_provider)
            
            await memory.initialize()
            
            try:
                request = MemorySearchRequest(query="test", context="test")
                results = await memory.search(request)
                
                # Only high similarity result should pass threshold
                assert len(results) == 1
                assert results[0].item.key == "good"
                assert results[0].score == 0.9  # 1 - 0.1 distance
                
            finally:
                await memory.shutdown()
    
    @pytest.mark.asyncio
    async def test_search_context_filter(self, mock_vector_provider, mock_embedding_provider):
        """Test search with context filtering."""
        memory = VectorMemory(VectorMemoryConfig())
        
        with patch('flowlib.agent.components.memory.vector.provider_registry') as mock_registry:
            mock_registry.get_by_config.side_effect = create_mock_registry(mock_vector_provider, mock_embedding_provider)
            
            await memory.initialize()
            
            try:
                request = MemorySearchRequest(query="test", context="specific_context", limit=5)
                await memory.search(request)
                
                # Verify context filter was applied
                call_args = mock_vector_provider.query.call_args
                assert call_args[1]["where"] == {"context": "specific_context"}
                
            finally:
                await memory.shutdown()


class TestVectorMemoryContextManagement:
    """Test context creation and management."""
    
    @pytest.mark.asyncio
    async def test_create_context(self, mock_vector_provider, mock_embedding_provider):
        """Test context creation."""
        memory = VectorMemory(VectorMemoryConfig())
        
        with patch('flowlib.agent.components.memory.vector.provider_registry') as mock_registry:
            mock_registry.get_by_config.side_effect = create_mock_registry(mock_vector_provider, mock_embedding_provider)
            
            await memory.initialize()
            
            try:
                context_name = await memory.create_context("test_context", {"meta": "data"})
                assert context_name == "test_context"
                
                # Check stats
                stats = memory.get_stats()
                assert "test_context" in stats["contexts"]
                assert stats["context_count"] == 1
                
            finally:
                await memory.shutdown()
    
    @pytest.mark.asyncio
    async def test_wipe_context(self, mock_vector_provider, mock_embedding_provider):
        """Test wiping a specific context."""
        memory = VectorMemory(VectorMemoryConfig())
        
        # Mock query to find items in context
        mock_vector_provider.query.return_value = {
            'ids': [['id1', 'id2', 'id3']],
            'documents': [['doc1', 'doc2', 'doc3']],
            'metadatas': [[{}, {}, {}]],
            'distances': [[0.1, 0.2, 0.3]]
        }
        
        with patch('flowlib.agent.components.memory.vector.provider_registry') as mock_registry:
            mock_registry.get_by_config.side_effect = create_mock_registry(mock_vector_provider, mock_embedding_provider)
            
            await memory.initialize()
            
            try:
                await memory.create_context("test_context")
                await memory.wipe_context("test_context")
                
                # Verify query to find items
                mock_vector_provider.query.assert_called()
                query_call = mock_vector_provider.query.call_args
                assert query_call[1]["where"] == {"context": "test_context"}
                assert query_call[1]["n_results"] == 10000  # Large number to get all
                
                # Verify deletion
                mock_vector_provider.delete.assert_called_once_with(
                    collection_name="agent_memory",
                    ids=['id1', 'id2', 'id3']
                )
                
                # Check context was removed from tracking
                stats = memory.get_stats()
                assert "test_context" not in stats["contexts"]
                
            finally:
                await memory.shutdown()


class TestVectorMemoryRetrieveRelevant:
    """Test retrieve_relevant functionality."""
    
    @pytest.mark.asyncio
    async def test_retrieve_relevant(self, mock_vector_provider, mock_embedding_provider):
        """Test retrieve_relevant method."""
        memory = VectorMemory(VectorMemoryConfig())
        
        # Mock search results
        mock_vector_provider.query.return_value = {
            'documents': [['Relevant doc 1', 'Relevant doc 2']],
            'metadatas': [[
                {'key': 'doc1', 'context': 'test'},
                {'key': 'doc2', 'context': 'test'}
            ]],
            'distances': [[0.2, 0.3]],
            'ids': [['id1', 'id2']]
        }
        
        with patch('flowlib.agent.components.memory.vector.provider_registry') as mock_registry:
            mock_registry.get_by_config.side_effect = create_mock_registry(mock_vector_provider, mock_embedding_provider)
            
            await memory.initialize()
            
            try:
                relevant_items = await memory.retrieve_relevant(
                    query="relevant content",
                    context="test",
                    limit=5
                )
                
                assert len(relevant_items) == 2
                assert relevant_items[0] == "doc1: Relevant doc 1"
                assert relevant_items[1] == "doc2: Relevant doc 2"
                
            finally:
                await memory.shutdown()


class TestVectorMemoryErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_uninitialized_operations(self):
        """Test that operations fail on uninitialized memory."""
        memory = VectorMemory(VectorMemoryConfig())
        
        store_request = MemoryStoreRequest(key="test", value="test", context="test")
        retrieve_request = MemoryRetrieveRequest(key="test", context="test")
        search_request = MemorySearchRequest(query="test", context="test")
        
        with pytest.raises(MemoryError, match="not initialized"):
            await memory.store(store_request)
        
        with pytest.raises(MemoryError, match="not initialized"):
            await memory.retrieve(retrieve_request)
        
        with pytest.raises(MemoryError, match="not initialized"):
            await memory.search(search_request)
        
        with pytest.raises(MemoryError, match="not initialized"):
            await memory.retrieve_relevant("query")
        
        with pytest.raises(MemoryError, match="not initialized"):
            await memory.wipe_context("test")
    
    @pytest.mark.asyncio
    async def test_graceful_error_handling(self, mock_vector_provider, mock_embedding_provider):
        """Test that errors fail fast instead of masking problems."""
        memory = VectorMemory(VectorMemoryConfig())
        
        with patch('flowlib.agent.components.memory.vector.provider_registry') as mock_registry:
            mock_registry.get_by_config.side_effect = create_mock_registry(mock_vector_provider, mock_embedding_provider)
            
            await memory.initialize()
            
            try:
                # Mock provider error
                mock_vector_provider.query.side_effect = Exception("Provider error")
                
                # Retrieve should raise MemoryError instead of masking the problem
                request = MemoryRetrieveRequest(key="test", context="test")
                with pytest.raises(MemoryError, match="Vector retrieval failed"):
                    await memory.retrieve(request)
                
                # Search should raise MemoryError instead of masking the problem
                search_request = MemorySearchRequest(query="test", context="test")
                with pytest.raises(MemoryError, match="Vector search failed"):
                    await memory.search(search_request)
                
            finally:
                await memory.shutdown()


class TestVectorMemoryStats:
    """Test statistics and monitoring."""
    
    @pytest.mark.asyncio
    async def test_get_stats(self, mock_vector_provider, mock_embedding_provider):
        """Test statistics reporting."""
        config = VectorMemoryConfig(
            vector_provider_config="custom-vector",
            embedding_provider_config="custom-embedding"
        )
        memory = VectorMemory(config)
        
        # Stats before initialization
        stats = memory.get_stats()
        assert stats["initialized"] is False
        assert stats["context_count"] == 0
        
        with patch('flowlib.agent.components.memory.vector.provider_registry') as mock_registry:
            mock_registry.get_by_config.side_effect = create_mock_registry(mock_vector_provider, mock_embedding_provider)
            
            await memory.initialize()
            
            try:
                # Stats after initialization
                stats = memory.get_stats()
                assert stats["initialized"] is True
                assert stats["providers"]["vector"] == "custom-vector"
                assert stats["providers"]["embedding"] == "custom-embedding"
                
                # Add context and check stats
                await memory.create_context("test_ctx")
                stats = memory.get_stats()
                assert stats["context_count"] == 1
                assert "test_ctx" in stats["contexts"]
                
            finally:
                await memory.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])