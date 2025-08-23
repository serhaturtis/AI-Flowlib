"""
Tests for the modernized KnowledgeMemory component.

These tests verify:
1. Config-driven graph provider access
2. Entity and relationship management
3. Graph database operations
4. Knowledge search capabilities
5. Error handling and provider lifecycle
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from flowlib.agent.components.memory.knowledge import KnowledgeMemory, KnowledgeMemoryConfig
from flowlib.agent.components.memory.models import MemoryItem, MemoryStoreRequest, MemoryRetrieveRequest, MemorySearchRequest
from flowlib.agent.core.errors import MemoryError
from flowlib.providers.graph.models import Entity, EntityAttribute, EntityRelationship


@pytest.fixture
def mock_graph_provider():
    """Create a mock graph database provider."""
    provider = AsyncMock()
    provider.add_entity.return_value = None
    provider.get_entity.return_value = None
    provider.delete_entity.return_value = None
    provider.search_entities.return_value = []
    return provider


@pytest.fixture
def sample_entity():
    """Create a sample entity for testing."""
    return Entity(
        id="test_entity_1",
        type="person",
        attributes={
            "name": EntityAttribute(name="name", value="John Doe", source="test"),
            "age": EntityAttribute(name="age", value="30", source="test")
        },
        relationships=[
            EntityRelationship(
                relation_type="works_at",
                target_entity="company_1",
                confidence=0.9,
                source="test"
            )
        ],
        tags=["employee", "developer"],
        importance=0.8
    )


class TestKnowledgeMemoryConfig:
    """Test KnowledgeMemoryConfig validation and defaults."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = KnowledgeMemoryConfig()
        assert config.graph_provider_config == "default-graph"
        assert config.default_importance == 0.7
        assert config.max_search_results == 50
        assert config.default_context == "knowledge"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = KnowledgeMemoryConfig(
            graph_provider_config="custom-graph",
            default_importance=0.9,
            max_search_results=100,
            default_context="custom"
        )
        assert config.graph_provider_config == "custom-graph"
        assert config.default_importance == 0.9
        assert config.max_search_results == 100
        assert config.default_context == "custom"
    
    def test_config_validation(self):
        """Test configuration validation constraints."""
        with pytest.raises(ValueError):
            KnowledgeMemoryConfig(default_importance=1.5)  # Above maximum
        
        with pytest.raises(ValueError):
            KnowledgeMemoryConfig(default_importance=-0.1)  # Below minimum
        
        with pytest.raises(ValueError):
            KnowledgeMemoryConfig(max_search_results=0)  # Below minimum


class TestKnowledgeMemoryInitialization:
    """Test KnowledgeMemory initialization and provider management."""
    
    @pytest.mark.asyncio
    async def test_successful_initialization(self, mock_graph_provider):
        """Test successful initialization with mock provider."""
        config = KnowledgeMemoryConfig()
        memory = KnowledgeMemory(config)
        
        with patch('flowlib.agent.components.memory.knowledge.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                return mock_graph_provider
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            await memory.initialize()
            
            assert memory.initialized is True
            assert memory._graph_provider == mock_graph_provider
            
            # Verify provider registry call
            mock_registry.get_by_config.assert_called_once_with("default-graph")
            
            await memory.shutdown()
    
    @pytest.mark.asyncio
    async def test_initialization_missing_provider(self):
        """Test initialization failure when graph provider is missing."""
        memory = KnowledgeMemory(KnowledgeMemoryConfig())
        
        with patch('flowlib.agent.components.memory.knowledge.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                return None
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            with pytest.raises(MemoryError, match="Graph provider not found"):
                await memory.initialize()
            
            assert memory.initialized is False
    
    @pytest.mark.asyncio
    async def test_double_initialization(self, mock_graph_provider):
        """Test that double initialization is handled gracefully."""
        memory = KnowledgeMemory(KnowledgeMemoryConfig())
        
        with patch('flowlib.agent.components.memory.knowledge.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                return mock_graph_provider
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            await memory.initialize()
            assert memory.initialized is True
            
            # Second initialization should be no-op
            await memory.initialize()
            assert memory.initialized is True
            
            # Should only call registry once
            assert mock_registry.get_by_config.call_count == 1
            
            await memory.shutdown()


class TestKnowledgeMemoryEntityStorage:
    """Test entity storage operations."""
    
    @pytest.mark.asyncio
    async def test_store_entity_directly(self, mock_graph_provider, sample_entity):
        """Test storing an Entity object directly."""
        memory = KnowledgeMemory(KnowledgeMemoryConfig())
        
        with patch('flowlib.agent.components.memory.knowledge.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                return mock_graph_provider
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            await memory.initialize()
            
            try:
                # Store the entity
                request = MemoryStoreRequest(key="entity_key", value=sample_entity, context="test")
                result = await memory.store(request)
                
                assert result == "entity_key"
                
                # Verify entity was added to graph
                mock_graph_provider.add_entity.assert_called_once_with(sample_entity)
                
            finally:
                await memory.shutdown()
    
    @pytest.mark.asyncio
    async def test_store_memory_item_as_entity(self, mock_graph_provider):
        """Test storing a MemoryItem as an entity."""
        memory = KnowledgeMemory(KnowledgeMemoryConfig())
        
        with patch('flowlib.agent.components.memory.knowledge.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                return mock_graph_provider
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            await memory.initialize()
            
            try:
                # Store a regular memory item
                request = MemoryStoreRequest(
                    key="person_1", 
                    value="John Doe is a software engineer", 
                    context="people",
                    metadata={"source": "test", "type": "person"}
                )
                
                result = await memory.store(request)
                
                assert result == "person_1"
                
                # Verify entity creation and storage
                mock_graph_provider.add_entity.assert_called_once()
                stored_entity = mock_graph_provider.add_entity.call_args[0][0]
                
                assert stored_entity.id == "person_1"
                assert stored_entity.type == "people"
                assert "content" in stored_entity.attributes
                assert stored_entity.attributes["content"].value == "John Doe is a software engineer"
                
            finally:
                await memory.shutdown()
    
    @pytest.mark.asyncio
    async def test_store_uninitialized(self):
        """Test that storing fails on uninitialized memory."""
        memory = KnowledgeMemory(KnowledgeMemoryConfig())
        
        request = MemoryStoreRequest(key="test", value="test", context="test")
        
        with pytest.raises(MemoryError, match="not initialized"):
            await memory.store(request)


class TestKnowledgeMemoryEntityRetrieval:
    """Test entity retrieval operations."""
    
    @pytest.mark.asyncio
    async def test_retrieve_existing_entity(self, mock_graph_provider, sample_entity):
        """Test retrieving an existing entity."""
        memory = KnowledgeMemory(KnowledgeMemoryConfig())
        
        # Mock graph provider to return the entity
        mock_graph_provider.get_entity.return_value = sample_entity
        
        with patch('flowlib.agent.components.memory.knowledge.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                return mock_graph_provider
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            await memory.initialize()
            
            try:
                request = MemoryRetrieveRequest(key="test_entity_1", context="test")
                result = await memory.retrieve(request)
                
                assert result is not None
                assert result.metadata["entity_type"] == "person"
                assert result.metadata["importance"] == 0.8
                assert result.metadata["tags"] == ["employee", "developer"]
                assert "test_entity_1" in result.value  # Entity ID should be in the string representation
                
                # Verify correct entity was requested
                mock_graph_provider.get_entity.assert_called_once_with("test_entity_1")
                
            finally:
                await memory.shutdown()
    
    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_entity(self, mock_graph_provider):
        """Test retrieving a non-existent entity."""
        memory = KnowledgeMemory(KnowledgeMemoryConfig())
        
        # Mock graph provider to return None
        mock_graph_provider.get_entity.return_value = None
        
        with patch('flowlib.agent.components.memory.knowledge.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                return mock_graph_provider
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            await memory.initialize()
            
            try:
                request = MemoryRetrieveRequest(key="nonexistent", context="test")
                result = await memory.retrieve(request)
                
                assert result is None
                
            finally:
                await memory.shutdown()
    
    @pytest.mark.asyncio
    async def test_get_entity_method(self, mock_graph_provider, sample_entity):
        """Test the get_entity convenience method."""
        memory = KnowledgeMemory(KnowledgeMemoryConfig())
        
        mock_graph_provider.get_entity.return_value = sample_entity
        
        with patch('flowlib.agent.components.memory.knowledge.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                return mock_graph_provider
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            await memory.initialize()
            
            try:
                entity = await memory.get_entity("test_entity_1")
                
                assert entity == sample_entity
                mock_graph_provider.get_entity.assert_called_once_with("test_entity_1")
                
            finally:
                await memory.shutdown()


class TestKnowledgeMemorySearch:
    """Test knowledge search operations."""
    
    @pytest.mark.asyncio
    async def test_search_entities(self, mock_graph_provider, sample_entity):
        """Test searching for entities."""
        memory = KnowledgeMemory(KnowledgeMemoryConfig())
        
        # Mock search results
        mock_graph_provider.search_entities.return_value = [sample_entity]
        
        with patch('flowlib.agent.components.memory.knowledge.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                return mock_graph_provider
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            await memory.initialize()
            
            try:
                request = MemorySearchRequest(query="John Doe", context="people", limit=10)
                results = await memory.search(request)
                
                assert len(results) == 1
                assert results[0].item.key == "test_entity_1"
                assert results[0].item.metadata["entity_type"] == "person"
                assert results[0].score == 0.8  # Entity importance used as score
                assert results[0].metadata["search_type"] == "graph_search"
                
                # Verify search parameters
                mock_graph_provider.search_entities.assert_called_once_with(
                    query="John Doe",
                    entity_type="people",
                    limit=10
                )
                
            finally:
                await memory.shutdown()
    
    @pytest.mark.asyncio
    async def test_search_no_results(self, mock_graph_provider):
        """Test search with no results."""
        memory = KnowledgeMemory(KnowledgeMemoryConfig())
        
        # Mock empty search results
        mock_graph_provider.search_entities.return_value = []
        
        with patch('flowlib.agent.components.memory.knowledge.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                return mock_graph_provider
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            await memory.initialize()
            
            try:
                request = MemorySearchRequest(query="nonexistent", context="test", limit=10)
                results = await memory.search(request)
                
                assert len(results) == 0
                
            finally:
                await memory.shutdown()
    
    @pytest.mark.asyncio
    async def test_search_provider_not_supported(self, mock_graph_provider):
        """Test search when provider doesn't support search_entities."""
        memory = KnowledgeMemory(KnowledgeMemoryConfig())
        
        # Remove search_entities method from mock
        del mock_graph_provider.search_entities
        
        with patch('flowlib.agent.components.memory.knowledge.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                return mock_graph_provider
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            await memory.initialize()
            
            try:
                request = MemorySearchRequest(query="test", context="test", limit=10)
                results = await memory.search(request)
                
                # Should return empty results gracefully
                assert len(results) == 0
                
            finally:
                await memory.shutdown()


class TestKnowledgeMemoryRelationships:
    """Test relationship management."""
    
    @pytest.mark.asyncio
    async def test_add_relationship(self, mock_graph_provider, sample_entity):
        """Test adding relationships between entities."""
        memory = KnowledgeMemory(KnowledgeMemoryConfig())
        
        # Mock getting the source entity
        mock_graph_provider.get_entity.return_value = sample_entity
        
        with patch('flowlib.agent.components.memory.knowledge.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                return mock_graph_provider
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            await memory.initialize()
            
            try:
                await memory.add_relationship(
                    source_id="test_entity_1",
                    relation_type="knows",
                    target_id="test_entity_2",
                    confidence=0.95,
                    source="test_system"
                )
                
                # Verify entity was retrieved
                mock_graph_provider.get_entity.assert_called_once_with("test_entity_1")
                
                # Verify entity was updated with new relationship
                mock_graph_provider.add_entity.assert_called_once()
                updated_entity = mock_graph_provider.add_entity.call_args[0][0]
                
                # Check that new relationship was added
                assert len(updated_entity.relationships) == 2  # Original + new
                new_relationship = updated_entity.relationships[-1]
                assert new_relationship.relation_type == "knows"
                assert new_relationship.target_entity == "test_entity_2"
                assert new_relationship.confidence == 0.95
                assert new_relationship.source == "test_system"
                
            finally:
                await memory.shutdown()
    
    @pytest.mark.asyncio
    async def test_add_relationship_missing_source(self, mock_graph_provider):
        """Test adding relationship when source entity doesn't exist."""
        memory = KnowledgeMemory(KnowledgeMemoryConfig())
        
        # Mock no entity found
        mock_graph_provider.get_entity.return_value = None
        
        with patch('flowlib.agent.components.memory.knowledge.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                return mock_graph_provider
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            await memory.initialize()
            
            try:
                with pytest.raises(MemoryError, match="Source entity .* not found"):
                    await memory.add_relationship(
                        source_id="nonexistent",
                        relation_type="knows",
                        target_id="target"
                    )
                    
            finally:
                await memory.shutdown()


class TestKnowledgeMemoryContextManagement:
    """Test context creation and management."""
    
    @pytest.mark.asyncio
    async def test_create_context(self, mock_graph_provider):
        """Test context creation."""
        memory = KnowledgeMemory(KnowledgeMemoryConfig())
        
        with patch('flowlib.agent.components.memory.knowledge.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                return mock_graph_provider
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            await memory.initialize()
            
            try:
                context_name = await memory.create_context("knowledge_context", {"meta": "data"})
                assert context_name == "knowledge_context"
                
                # Check stats
                stats = memory.get_stats()
                assert "knowledge_context" in stats["contexts"]
                assert stats["context_count"] == 1
                
            finally:
                await memory.shutdown()
    
    @pytest.mark.asyncio
    async def test_wipe_context(self, mock_graph_provider, sample_entity):
        """Test wiping a specific context."""
        memory = KnowledgeMemory(KnowledgeMemoryConfig())
        
        # Mock search to find entities in context
        mock_graph_provider.search_entities.return_value = [sample_entity]
        
        with patch('flowlib.agent.components.memory.knowledge.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                return mock_graph_provider
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            await memory.initialize()
            
            try:
                await memory.create_context("test_context")
                await memory.wipe_context("test_context")
                
                # Verify search for entities in context
                mock_graph_provider.search_entities.assert_called_with(
                    query="*",
                    entity_type="test_context",
                    limit=50  # Capped by max_search_results config
                )
                
                # Verify entity deletion (if provider supports it)
                if hasattr(mock_graph_provider, 'delete_entity'):
                    mock_graph_provider.delete_entity.assert_called_with("test_entity_1")
                
                # Check context was removed from tracking
                stats = memory.get_stats()
                assert "test_context" not in stats["contexts"]
                
            finally:
                await memory.shutdown()


class TestKnowledgeMemoryRetrieveRelevant:
    """Test retrieve_relevant functionality."""
    
    @pytest.mark.asyncio
    async def test_retrieve_relevant(self, mock_graph_provider, sample_entity):
        """Test retrieve_relevant method."""
        memory = KnowledgeMemory(KnowledgeMemoryConfig())
        
        # Mock search results
        mock_graph_provider.search_entities.return_value = [sample_entity]
        
        with patch('flowlib.agent.components.memory.knowledge.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                return mock_graph_provider
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            await memory.initialize()
            
            try:
                relevant_items = await memory.retrieve_relevant(
                    query="software engineer",
                    context="people",
                    limit=5
                )
                
                assert len(relevant_items) == 1
                assert relevant_items[0].startswith("test_entity_1:")
                assert "Entity" in relevant_items[0]  # Contains entity representation
                
            finally:
                await memory.shutdown()


class TestKnowledgeMemoryErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_uninitialized_operations(self):
        """Test that operations fail on uninitialized memory."""
        memory = KnowledgeMemory(KnowledgeMemoryConfig())
        
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
        
        with pytest.raises(MemoryError, match="not initialized"):
            await memory.add_relationship("source", "relation", "target")
        
        with pytest.raises(MemoryError, match="not initialized"):
            await memory.get_entity("entity_id")
    
    @pytest.mark.asyncio
    async def test_graceful_error_handling(self, mock_graph_provider):
        """Test that errors are handled gracefully without crashing."""
        memory = KnowledgeMemory(KnowledgeMemoryConfig())
        
        with patch('flowlib.agent.components.memory.knowledge.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                return mock_graph_provider
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            await memory.initialize()
            
            try:
                # Mock provider errors
                mock_graph_provider.get_entity.side_effect = Exception("Provider error")
                mock_graph_provider.search_entities.side_effect = Exception("Search error")
                
                # get_entity should raise MemoryError instead of masking the problem
                with pytest.raises(MemoryError, match="Failed to get entity"):
                    await memory.get_entity("test")
                
                # retrieve should raise MemoryError instead of masking the problem
                request = MemoryRetrieveRequest(key="test", context="test")
                with pytest.raises(MemoryError, match="Knowledge retrieval failed"):
                    await memory.retrieve(request)
                
                # search should raise MemoryError instead of masking the problem
                search_request = MemorySearchRequest(query="test", context="test")
                with pytest.raises(MemoryError, match="Knowledge search failed"):
                    await memory.search(search_request)
                
            finally:
                await memory.shutdown()


class TestKnowledgeMemoryContentExtraction:
    """Test content extraction from different item types."""
    
    @pytest.mark.asyncio
    async def test_extract_content_from_memory_item(self, mock_graph_provider):
        """Test content extraction from MemoryItem."""
        memory = KnowledgeMemory(KnowledgeMemoryConfig())
        
        with patch('flowlib.agent.components.memory.knowledge.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                return mock_graph_provider
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            await memory.initialize()
            
            try:
                # Test different content types
                test_cases = [
                    "test content",
                    type('MockItem', (), {'text': 'test text'})(),
                    type('MockItem', (), {'message': 'test message'})(),
                    "plain string"
                ]
                
                for i, value in enumerate(test_cases):
                    request = MemoryStoreRequest(key=f"test_{i}", value=value, context="test")
                    result = await memory.store(request)
                    assert result == f"test_{i}"
                
                # Verify all items were stored
                assert mock_graph_provider.add_entity.call_count == len(test_cases)
                
            finally:
                await memory.shutdown()


class TestKnowledgeMemoryStats:
    """Test statistics and monitoring."""
    
    @pytest.mark.asyncio
    async def test_get_stats(self, mock_graph_provider):
        """Test statistics reporting."""
        config = KnowledgeMemoryConfig(
            graph_provider_config="custom-graph",
            default_importance=0.9
        )
        memory = KnowledgeMemory(config)
        
        # Stats before initialization
        stats = memory.get_stats()
        assert stats["initialized"] is False
        assert stats["context_count"] == 0
        
        with patch('flowlib.agent.components.memory.knowledge.provider_registry') as mock_registry:
            async def mock_get_by_config(config_name):
                return mock_graph_provider
            mock_registry.get_by_config.side_effect = mock_get_by_config
            
            await memory.initialize()
            
            try:
                # Stats after initialization
                stats = memory.get_stats()
                assert stats["initialized"] is True
                assert stats["provider"]["graph"] == "custom-graph"
                assert stats["config"]["default_importance"] == 0.9
                
                # Add context and check stats
                await memory.create_context("knowledge_ctx")
                stats = memory.get_stats()
                assert stats["context_count"] == 1
                assert "knowledge_ctx" in stats["contexts"]
                
            finally:
                await memory.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])