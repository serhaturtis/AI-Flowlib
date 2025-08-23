"""Tests for JanusGraph Provider implementation."""

import pytest
import pytest_asyncio
import json
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime

from flowlib.providers.graph.janus.provider import (
    JanusGraphProvider,
    JanusProviderSettings
)
from flowlib.providers.graph.models import Entity, EntityAttribute, EntityRelationship, GraphStoreResult, EntitySearchResult
from flowlib.core.errors.errors import ProviderError

# Move these fixtures to module scope so all test classes can use them
@pytest.fixture
def mock_client():
    client = Mock()
    # By default, submit().all().result() returns an empty list (contract-compliant)
    client.submit.return_value.all.return_value.result.return_value = []
    return client

@pytest.fixture
def mock_traversal():
    return Mock()

@pytest.fixture
def mock_connection():
    return Mock()

class TestJanusProviderSettings:
    """Test JanusProviderSettings model."""
    
    def test_settings_creation_minimal(self):
        """Test settings creation with minimal data."""
        settings = JanusProviderSettings()
        
        assert settings.url == "ws://localhost:8182/gremlin"
        assert settings.username == ""
        assert settings.password == ""
        assert settings.graph_name == "g"
        assert settings.traversal_source == "g"
        assert settings.connection_pool_size == 4
        assert settings.message_serializer == "graphbinary-1.0"
        assert settings.read_timeout == 30
        assert settings.write_timeout == 30
        assert settings.max_retry_count == 3
        # Inherited from GraphDBProviderSettings
        assert settings.max_retries == 3
        assert settings.timeout_seconds == 60.0
    
    def test_settings_creation_full(self):
        """Test settings creation with all fields."""
        settings = JanusProviderSettings(
            url="ws://janus.example.com:8182/gremlin",
            username="admin",
            password="secret123",
            graph_name="knowledge_graph",
            traversal_source="g_kg",
            connection_pool_size=8,
            message_serializer="graphson-3.0",
            read_timeout=60,
            write_timeout=60,
            max_retry_count=5,
            max_retries=10,
            timeout_seconds=120.0
        )
        
        assert settings.url == "ws://janus.example.com:8182/gremlin"
        assert settings.username == "admin"
        assert settings.password == "secret123"
        assert settings.graph_name == "knowledge_graph"
        assert settings.traversal_source == "g_kg"
        assert settings.connection_pool_size == 8
        assert settings.message_serializer == "graphson-3.0"
        assert settings.read_timeout == 60
        assert settings.write_timeout == 60
        assert settings.max_retry_count == 5
        assert settings.max_retries == 10
        assert settings.timeout_seconds == 120.0
    
    def test_settings_inheritance(self):
        """Test that JanusProviderSettings inherits from ProviderSettings."""
        from flowlib.providers.core.base import ProviderSettings
        settings = JanusProviderSettings()
        assert isinstance(settings, ProviderSettings)


@pytest.mark.skipif(
    pytest.importorskip("gremlin_python", reason="gremlin_python not available") is None,
    reason="JanusGraph tests require gremlin_python dependency"
)
class TestJanusGraphProvider:
    """Test JanusGraphProvider implementation."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return JanusProviderSettings(
            url="ws://localhost:8182/gremlin",
            username="test",
            password="test",
            connection_pool_size=2,
            read_timeout=10,
            write_timeout=10
        )
    
    @pytest_asyncio.fixture
    def provider(self, provider_settings):
        """Create test provider instance."""
        return JanusGraphProvider(settings=provider_settings)
    
    def test_provider_initialization(self, provider, provider_settings):
        """Test provider initialization."""
        assert provider.name == "janusgraph"
        assert provider.provider_type == "graph_db"
        assert provider.settings == provider_settings
        assert not provider._initialized
        assert provider._client is None
        assert provider._g is None
    
    def test_provider_inheritance(self, provider):
        """Test provider inheritance from base classes."""
        from flowlib.providers.graph.base import GraphDBProvider
        from flowlib.providers.core.base import Provider
        assert isinstance(provider, Provider)
    
    def test_provider_default_settings(self):
        """Test provider with default settings."""
        provider = JanusGraphProvider()
        assert provider.name == "janusgraph"
        assert isinstance(provider.settings, JanusProviderSettings)
    
    def test_provider_decorator_registration(self):
        """Test that provider is properly registered with decorator."""
        assert hasattr(JanusGraphProvider, '__provider_name__')
        assert hasattr(JanusGraphProvider, '__provider_type__')
        assert JanusGraphProvider.__provider_name__ == "janusgraph"
        assert JanusGraphProvider.__provider_type__ == "graph_db"


class TestJanusGraphProviderWithMocks:
    """Test JanusGraphProvider with mocked gremlin_python."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return JanusProviderSettings(
            url="ws://localhost:8182/gremlin",
            username="test",
            password="test"
        )
    
    @pytest_asyncio.fixture
    async def provider(self, provider_settings, mock_client, mock_traversal, mock_connection):
        """Create and initialize test provider with mocks."""
        with patch('flowlib.providers.graph.janus.provider.Client', return_value=mock_client), \
             patch('flowlib.providers.graph.janus.provider.DriverRemoteConnection', return_value=mock_connection), \
             patch('flowlib.providers.graph.janus.provider.traversal') as mock_trav_func, \
             patch('flowlib.providers.graph.janus.provider.GraphBinaryMessageSerializer'):
            
            mock_trav_func.return_value.withRemote.return_value = mock_traversal
            
            provider = JanusGraphProvider(settings=provider_settings)
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_provider_lifecycle(self, provider_settings, mock_client, mock_traversal, mock_connection):
        """Test provider initialization and shutdown lifecycle."""
        with patch('flowlib.providers.graph.janus.provider.Client', return_value=mock_client), \
             patch('flowlib.providers.graph.janus.provider.DriverRemoteConnection', return_value=mock_connection), \
             patch('flowlib.providers.graph.janus.provider.traversal') as mock_trav_func, \
             patch('flowlib.providers.graph.janus.provider.GraphBinaryMessageSerializer'), \
             patch('flowlib.providers.graph.janus.provider.JANUS_AVAILABLE', True):
            mock_trav_func.return_value.withRemote.return_value = mock_traversal
            provider = JanusGraphProvider(settings=provider_settings)
            # Initially not initialized
            assert not provider._initialized
            assert provider._client is None
            # Initialize
            await provider.initialize()
            assert provider._initialized
            assert provider._client is not None
            # Shutdown
            await provider.shutdown()
            assert not provider._initialized
            assert provider._client is None
    
    @pytest.mark.asyncio
    async def test_initialization_without_gremlin_python(self, provider_settings):
        """Test initialization fails without gremlin_python."""
        with patch('flowlib.providers.graph.janus.provider.JANUS_AVAILABLE', False):
            provider = JanusGraphProvider(settings=provider_settings)
            
            with pytest.raises(ProviderError, match="Gremlin Python driver is not installed"):
                await provider.initialize()
    
    @pytest.mark.asyncio
    async def test_initialization_connection_failure(self, provider_settings):
        """Test initialization failure handling."""
        mock_client = Mock()
        mock_client.submit.side_effect = Exception("Connection failed")
        
        with patch('flowlib.providers.graph.janus.provider.JANUS_AVAILABLE', True), \
             patch('flowlib.providers.graph.janus.provider.Client', return_value=mock_client), \
             patch('flowlib.providers.graph.janus.provider.DriverRemoteConnection'), \
             patch('flowlib.providers.graph.janus.provider.traversal'), \
             patch('flowlib.providers.graph.janus.provider.GraphBinaryMessageSerializer'):
            
            provider = JanusGraphProvider(settings=provider_settings)
            
            with pytest.raises(ProviderError, match="Failed to connect to JanusGraph"):
                await provider.initialize()
    
    @pytest.mark.asyncio
    async def test_schema_setup(self, provider_settings, mock_client, mock_traversal, mock_connection):
        """Test schema setup logic with mocks."""
        with patch('flowlib.providers.graph.janus.provider.Client', return_value=mock_client), \
             patch('flowlib.providers.graph.janus.provider.DriverRemoteConnection', return_value=mock_connection), \
             patch('flowlib.providers.graph.janus.provider.traversal') as mock_trav_func, \
             patch('flowlib.providers.graph.janus.provider.GraphBinaryMessageSerializer'), \
             patch('flowlib.providers.graph.janus.provider.JANUS_AVAILABLE', True):
            mock_trav_func.return_value.withRemote.return_value = mock_traversal
            provider = JanusGraphProvider(settings=provider_settings)
            await provider.initialize()
            await provider._setup_schema()


class TestJanusGraphProviderEntityOperations:
    """Test entity operations with mocks."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return JanusProviderSettings()
    
    @pytest_asyncio.fixture
    async def provider(self, provider_settings, mock_client):
        """Create and initialize test provider."""
        with patch('flowlib.providers.graph.janus.provider.JANUS_AVAILABLE', True), \
             patch('flowlib.providers.graph.janus.provider.Client', return_value=mock_client), \
             patch('flowlib.providers.graph.janus.provider.DriverRemoteConnection'), \
             patch('flowlib.providers.graph.janus.provider.traversal'), \
             patch('flowlib.providers.graph.janus.provider.GraphBinaryMessageSerializer'):
            
            provider = JanusGraphProvider(settings=provider_settings)
            await provider.initialize()
            return provider
    
    @pytest_asyncio.fixture
    def sample_entity(self):
        """Create a sample entity for testing."""
        return Entity(
            id="john_doe",
            type="person",
            attributes={
                "name": EntityAttribute(name="name", value="John Doe"),
                "age": EntityAttribute(name="age", value="30")
            },
            relationships=[
                EntityRelationship(relation_type="friend_of", target_entity="jane_doe")
            ],
            source="test",
            importance=0.8
        )
    
    @pytest.mark.asyncio
    async def test_add_entity_new(self, provider, sample_entity, mock_client):
        """Test adding new entity."""
        # Patch add_relationship to always use EntityRelationship
        with patch.object(JanusGraphProvider, 'add_relationship', new_callable=AsyncMock) as mock_add_rel:
            mock_add_rel.return_value = None
            # Mock entity does not exist, then create
            mock_future_exists = Mock()
            mock_future_exists.all.return_value.result.return_value = [{"value": False}]
            mock_future_create = Mock()
            mock_future_create.all.return_value.result.return_value = [{"_id": sample_entity.id}]
            mock_client.submit.side_effect = [mock_future_exists, mock_future_create]
            result = await provider.add_entity(sample_entity)
            assert isinstance(result, GraphStoreResult)
            assert result.success
            assert sample_entity.id in result.stored_entities
    
    @pytest.mark.asyncio
    async def test_add_entity_existing(self, provider, sample_entity, mock_client):
        """Test updating existing entity."""
        with patch.object(JanusGraphProvider, 'add_relationship', new_callable=AsyncMock) as mock_add_rel:
            mock_add_rel.return_value = None
            # Mock entity exists, then update
            mock_future_exists = Mock()
            mock_future_exists.all.return_value.result.return_value = [{"value": True}]
            mock_future_update = Mock()
            mock_future_update.all.return_value.result.return_value = [{"_id": sample_entity.id}]
            mock_client.submit.side_effect = [mock_future_exists, mock_future_update]
            result = await provider.add_entity(sample_entity)
            assert isinstance(result, GraphStoreResult)
            assert result.success
            assert sample_entity.id in result.stored_entities
    
    @pytest.mark.asyncio
    async def test_add_entity_with_relationships(self, provider, mock_client):
        """Test adding entity with relationships."""
        entity = Entity(
            id="john_doe",
            type="person",
            relationships=[
                EntityRelationship(relation_type="friend_of", target_entity="jane_doe"),
                EntityRelationship(relation_type="colleague_of", target_entity="bob_smith")
            ]
        )
        with patch.object(JanusGraphProvider, 'add_relationship', new_callable=AsyncMock) as mock_add_rel:
            mock_add_rel.return_value = None
            # Mock responses for entity creation
            mock_future_exists = Mock()
            mock_future_exists.all.return_value.result.return_value = [{"value": False}]
            mock_future_create = Mock()
            mock_future_create.all.return_value.result.return_value = [{"_id": entity.id}]
            mock_client.submit.side_effect = [mock_future_exists, mock_future_create]
            result = await provider.add_entity(entity)
            assert isinstance(result, GraphStoreResult)
            assert result.success
            assert entity.id in result.stored_entities
    
    @pytest.mark.asyncio
    async def test_get_entity_success(self, provider, mock_client):
        """Test successful entity retrieval."""
        # Mock entity data response
        entity_data = {
            "id": "john_doe",
            "type": "person",
            "source": "test",
            "importance": 0.8,
            "last_updated": "2023-01-01T00:00:00",
            "attributes": json.dumps({
                "name": {"name": "name", "value": "John Doe", "confidence": 0.9, "source": "test", "timestamp": "2023-01-01T00:00:00"}
            })
        }
        
        # Mock relationship data response
        rel_data = [
            {
                "relation_type": "friend_of",
                "confidence": 0.9,
                "source": "test",
                "timestamp": "2023-01-01T00:00:00",
                "target_id": "jane_doe"
            }
        ]
        
        mock_future_entity = Mock()
        mock_future_entity.all.return_value.result.return_value = [entity_data]
        
        mock_future_rels = Mock()
        mock_future_rels.all.return_value.result.return_value = rel_data
        
        mock_client.submit.side_effect = [mock_future_entity, mock_future_rels]
        
        entity = await provider.get_entity("john_doe")
        
        assert entity is not None
        assert entity.id == "john_doe"
        assert entity.type == "person"
        assert len(entity.attributes) == 1
        assert "name" in entity.attributes
        assert entity.attributes["name"].value == "John Doe"
        assert len(entity.relationships) == 1
        assert entity.relationships[0].target_entity == "jane_doe"
    
    @pytest.mark.asyncio
    async def test_get_entity_not_found(self, provider, mock_client):
        """Test retrieving non-existent entity."""
        mock_future = Mock()
        mock_future.all.return_value.result.return_value = []
        mock_client.submit.return_value = mock_future
        
        entity = await provider.get_entity("nonexistent")
        assert entity is None
    
    @pytest.mark.asyncio
    async def test_get_entity_invalid_attributes(self, provider, mock_client):
        """Test entity retrieval with invalid JSON attributes raises error (no fallbacks)."""
        entity_data = {
            "id": "john_doe",
            "type": "person",
            "source": "test",
            "importance": 0.8,
            "last_updated": "2023-01-01T00:00:00",
            "attributes": "invalid json"  # Invalid JSON
        }
        
        mock_future_entity = Mock()
        mock_future_entity.all.return_value.result.return_value = [entity_data]
        
        mock_future_rels = Mock()
        mock_future_rels.all.return_value.result.return_value = []
        
        mock_client.submit.side_effect = [mock_future_entity, mock_future_rels]
        
        # Should raise ProviderError due to invalid JSON (no fallbacks policy)
        with pytest.raises(ProviderError, match="Invalid JSON in attributes"):
            await provider.get_entity("john_doe")
    
    @pytest.mark.asyncio
    async def test_delete_entity_success(self, provider, mock_client):
        """Test successful entity deletion."""
        mock_future = Mock()
        mock_future.all.return_value.result.return_value = [{"value": 1}]  # 1 entity deleted
        mock_client.submit.return_value = mock_future
        
        result = await provider.delete_entity("john_doe")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_entity_not_found(self, provider, mock_client):
        """Test deleting non-existent entity."""
        mock_future = Mock()
        mock_future.all.return_value.result.return_value = [{"value": 0}]  # 0 entities deleted
        mock_client.submit.return_value = mock_future
        
        result = await provider.delete_entity("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_operations_require_initialization(self, provider_settings):
        """Test that operations require initialization."""
        provider = JanusGraphProvider(settings=provider_settings)
        
        with pytest.raises(ProviderError, match="not initialized"):
            await provider.add_entity(Entity(id="test", type="test"))
        
        with pytest.raises(ProviderError, match="not initialized"):
            await provider.get_entity("test")


class TestJanusGraphProviderRelationshipOperations:
    """Test relationship operations with mocks."""
    
    @pytest_asyncio.fixture
    async def provider(self):
        """Create and initialize test provider."""
        mock_client = Mock()
        mock_future = Mock()
        mock_future.all.return_value.result.return_value = [{"value": 1}]
        mock_client.submit.return_value = mock_future
        
        with patch('flowlib.providers.graph.janus.provider.JANUS_AVAILABLE', True), \
             patch('flowlib.providers.graph.janus.provider.Client', return_value=mock_client), \
             patch('flowlib.providers.graph.janus.provider.DriverRemoteConnection'), \
             patch('flowlib.providers.graph.janus.provider.traversal'), \
             patch('flowlib.providers.graph.janus.provider.GraphBinaryMessageSerializer'):
            
            provider = JanusGraphProvider()
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_add_relationship_success(self, provider):
        """Test adding a relationship (model-driven)."""
        rel = EntityRelationship(
            relation_type="KNOWS",
            target_entity="test_entity_2",
            confidence=0.8,
            source="test"
        )
        await provider.add_relationship(
            source_id="test_entity_1",
            target_entity=rel.target_entity,
            relation_type=rel.relation_type,
            relationship=rel
        )
        # No dicts returned, no exceptions means success
    
    @pytest.mark.asyncio
    async def test_add_relationship_source_not_exists(self, provider):
        """Test adding relationship with non-existent source."""
        # Simulate the Gremlin client raising an error, which should be wrapped as ProviderError
        provider._client.submit.side_effect = Exception("Source entity does not exist")
        rel = EntityRelationship(relation_type="friend_of", target_entity="jane_doe")
        with pytest.raises(ProviderError, match="Source entity does not exist"):
            await provider.add_relationship("nonexistent", "jane_doe", "friend_of", rel)
    
    @pytest.mark.asyncio
    async def test_add_relationship_update_existing(self, provider):
        """Test updating existing relationship."""
        mock_responses = [
            Mock(all=Mock(return_value=Mock(result=Mock(return_value=[{"value": True}])))),
            Mock(all=Mock(return_value=Mock(result=Mock(return_value=[{"value": True}])))),
            Mock(all=Mock(return_value=Mock(result=Mock(return_value=[{"value": True}])))),
            Mock(all=Mock(return_value=Mock(result=Mock(return_value=[{"_id": "edge_123"}])))),
        ] * 2  # Double the responses to match the actual call count (8)
        provider._client.submit.side_effect = mock_responses
        rel = EntityRelationship(relation_type="friend_of", target_entity="jane_doe", confidence=0.95)
        await provider.add_relationship("john_doe", "jane_doe", "friend_of", rel)
        assert provider._client.submit.call_count == 8
    
    @pytest.mark.asyncio
    async def test_query_relationships_outgoing(self, provider):
        """Test querying outgoing relationships."""
        rel_data = [
            {
                "r": {
                    "relation_type": "friend_of",
                    "confidence": 0.9,
                    "source": "test",
                    "timestamp": "2023-01-01T00:00:00"
                },
                "target": {"id": "jane_doe"}
            }
        ]
        mock_future = Mock()
        mock_future.all.return_value.result.return_value = rel_data
        provider._client.submit.return_value = mock_future
        relationships = await provider.query_relationships("john_doe", direction="outgoing")
        assert relationships.total_count == 1
        assert relationships.metadata["direction"] == "outgoing"
    
    @pytest.mark.asyncio
    async def test_query_relationships_incoming(self, provider):
        """Test querying incoming relationships."""
        rel_data = [
            {
                "r": {
                    "relation_type": "friend_of",
                    "confidence": 0.9,
                    "source": "test",
                    "timestamp": "2023-01-01T00:00:00"
                },
                "source": {"id": "jane_doe"}
            }
        ]
        mock_future = Mock()
        mock_future.all.return_value.result.return_value = rel_data
        provider._client.submit.return_value = mock_future
        relationships = await provider.query_relationships("john_doe", direction="incoming")
        assert relationships.total_count == 1
        assert relationships.metadata["direction"] == "incoming"
    
    @pytest.mark.asyncio
    async def test_query_relationships_by_type(self, provider):
        """Test querying relationships by type."""
        rel_data = [
            {
                "r": {
                    "relation_type": "friend_of",
                    "confidence": 0.9,
                    "source": "test",
                    "timestamp": "2023-01-01T00:00:00"
                },
                "target": {"id": "jane_doe"}
            }
        ]
        mock_future = Mock()
        mock_future.all.return_value.result.return_value = rel_data
        provider._client.submit.return_value = mock_future
        relationships = await provider.query_relationships("john_doe", relation_type="friend_of")
        assert relationships.total_count == 1
        assert relationships.metadata["relation_type"] == "friend_of"
    
    @pytest.mark.asyncio
    async def test_delete_relationship_success(self, provider):
        """Test successful relationship deletion."""
        mock_future = Mock()
        mock_future.all.return_value.result.return_value = [{"value": 1}]  # 1 relationship deleted
        provider._client.submit.return_value = mock_future
        
        result = await provider.delete_relationship("john_doe", "jane_doe", "friend_of")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_relationship_not_found(self, provider):
        """Test deleting non-existent relationship."""
        mock_future = Mock()
        mock_future.all.return_value.result.return_value = [{"value": 0}]  # 0 relationships deleted
        provider._client.submit.return_value = mock_future
        
        result = await provider.delete_relationship("john_doe", "jane_doe", "nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_remove_relationship(self, provider):
        """Test remove_relationship method."""
        mock_future = Mock()
        mock_future.all.return_value.result.return_value = [{"value": 1}]
        provider._client.submit.return_value = mock_future
        
        # Should not raise error
        await provider.remove_relationship("john_doe", "jane_doe", "friend_of")


class TestJanusGraphProviderTraversal:
    """Test graph traversal operations with mocks."""
    
    @pytest_asyncio.fixture
    async def provider(self):
        """Create and initialize test provider."""
        mock_client = Mock()
        mock_future = Mock()
        mock_future.all.return_value.result.return_value = [{"value": 1}]
        mock_client.submit.return_value = mock_future
        
        with patch('flowlib.providers.graph.janus.provider.JANUS_AVAILABLE', True), \
             patch('flowlib.providers.graph.janus.provider.Client', return_value=mock_client), \
             patch('flowlib.providers.graph.janus.provider.DriverRemoteConnection'), \
             patch('flowlib.providers.graph.janus.provider.traversal'), \
             patch('flowlib.providers.graph.janus.provider.GraphBinaryMessageSerializer'):
            
            provider = JanusGraphProvider()
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_traverse_basic(self, provider):
        """Test basic graph traversal."""
        # Mock entity exists check
        mock_exists = Mock()
        mock_exists.all.return_value.result.return_value = [{"value": True}]
        
        # Mock traversal results
        mock_traversal = Mock()
        mock_traversal.all.return_value.result.return_value = [
            {"value": "entity_1"},
            {"value": "entity_2"}
        ]
        
        # Mock entity retrieval for each found entity
        entity_data = {
            "id": "test_entity",
            "type": "test",
            "source": "test",
            "importance": 0.5,
            "last_updated": "2023-01-01T00:00:00",
            "attributes": "{}"
        }
        
        mock_entity = Mock()
        mock_entity.all.return_value.result.return_value = [entity_data]
        
        mock_rels = Mock()
        mock_rels.all.return_value.result.return_value = []
        
        provider._client.submit.side_effect = [
            mock_exists,     # Entity exists check
            mock_traversal,  # Traversal
            mock_entity,     # Get entity 1
            mock_rels,       # Get relationships for entity 1
            mock_entity,     # Get entity 2
            mock_rels,       # Get relationships for entity 2
            mock_entity,     # Get start entity
            mock_rels,       # Get relationships for start entity
        ]
        
        entities = await provider.traverse("start_entity", max_depth=2)
        
        # Should return entities found in traversal
        assert len(entities) == 3  # 2 found + start entity
    
    @pytest.mark.asyncio
    async def test_traverse_with_relation_filter(self, provider):
        """Test traversal with relation type filter."""
        # Mock entity exists
        mock_exists = Mock()
        mock_exists.all.return_value.result.return_value = [{"value": True}]
        
        # Mock traversal with relation filter
        mock_traversal = Mock()
        mock_traversal.all.return_value.result.return_value = [{"value": "entity_1"}]
        
        # Mock entity retrieval
        entity_data = {
            "id": "entity_1",
            "type": "test",
            "source": "test",
            "importance": 0.5,
            "last_updated": "2023-01-01T00:00:00",
            "attributes": "{}"
        }
        
        mock_entity = Mock()
        mock_entity.all.return_value.result.return_value = [entity_data]
        
        mock_rels = Mock()
        mock_rels.all.return_value.result.return_value = []
        
        provider._client.submit.side_effect = [mock_exists, mock_traversal, mock_entity, mock_rels, mock_entity, mock_rels]
        
        entities = await provider.traverse("start_entity", relation_types=["friend_of"], max_depth=2)
        
        assert len(entities) == 2  # 1 found + start entity
    
    @pytest.mark.asyncio
    async def test_traverse_nonexistent_start(self, provider):
        """Test traversal from non-existent entity."""
        mock_exists = Mock()
        mock_exists.all.return_value.result.return_value = [{"value": False}]
        provider._client.submit.return_value = mock_exists
        
        entities = await provider.traverse("nonexistent")
        assert entities == []


class TestJanusGraphProviderQuery:
    """Test query operations with mocks."""
    
    @pytest_asyncio.fixture
    async def provider(self):
        """Create and initialize test provider."""
        mock_client = Mock()
        mock_future = Mock()
        mock_future.all.return_value.result.return_value = [{"value": 1}]
        mock_client.submit.return_value = mock_future
        
        with patch('flowlib.providers.graph.janus.provider.JANUS_AVAILABLE', True), \
             patch('flowlib.providers.graph.janus.provider.Client', return_value=mock_client), \
             patch('flowlib.providers.graph.janus.provider.DriverRemoteConnection'), \
             patch('flowlib.providers.graph.janus.provider.traversal'), \
             patch('flowlib.providers.graph.janus.provider.GraphBinaryMessageSerializer'):
            
            provider = JanusGraphProvider()
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_execute_query_success(self, provider):
        """Test successful query execution."""
        result_data = [{"id": "test", "value": "result"}]
        
        mock_future = Mock()
        mock_future.all.return_value.result.return_value = result_data
        provider._client.submit.return_value = mock_future
        
        results = await provider._execute_query("g.V().limit(1)", {"param": "value"})
        
        assert results == result_data
        provider._client.submit.assert_called_with("g.V().limit(1)", {"param": "value"})
    
    @pytest.mark.asyncio
    async def test_execute_query_with_graph_elements(self, provider):
        """Test query execution with graph elements."""
        # Mock vertex-like object
        mock_vertex = Mock()
        mock_vertex.id = "vertex_123"
        mock_vertex.label = "Entity"
        mock_vertex.properties = {"name": [Mock(value="test_name")], "type": "person"}
        
        mock_future = Mock()
        mock_future.all.return_value.result.return_value = [mock_vertex]
        provider._client.submit.return_value = mock_future
        
        results = await provider._execute_query("g.V()")
        
        assert len(results) == 1
        assert results[0]["_id"] == "vertex_123"
        assert results[0]["_label"] == "Entity"
        assert results[0]["name"] == "test_name"
        assert results[0]["type"] == "person"
    
    @pytest.mark.asyncio
    async def test_execute_query_primitive_values(self, provider):
        """Test query execution with primitive values."""
        primitive_results = [1, "text", True, 3.14]
        
        mock_future = Mock()
        mock_future.all.return_value.result.return_value = primitive_results
        provider._client.submit.return_value = mock_future
        
        results = await provider._execute_query("g.V().count()")
        
        assert len(results) == 4
        assert results[0] == {"value": 1}
        assert results[1] == {"value": "text"}
        assert results[2] == {"value": True}
        assert results[3] == {"value": 3.14}
    
    @pytest.mark.asyncio
    async def test_execute_query_failure(self, provider):
        """Test query execution failure."""
        provider._client.submit.side_effect = Exception("Query failed")
        
        with pytest.raises(ProviderError, match="Failed to execute JanusGraph query"):
            await provider._execute_query("invalid query")
    
    @pytest.mark.asyncio
    async def test_query_uninitialized_provider(self):
        """Test query execution on uninitialized provider."""
        provider = JanusGraphProvider()
        
        with pytest.raises(ProviderError, match="not initialized"):
            await provider._execute_query("g.V()")


class TestJanusGraphProviderAdvancedFeatures:
    """Test advanced provider features with mocks."""
    
    @pytest_asyncio.fixture
    async def provider(self):
        """Create and initialize test provider."""
        mock_client = Mock()
        mock_future = Mock()
        mock_future.all.return_value.result.return_value = [{"value": 1}]
        mock_client.submit.return_value = mock_future
        
        with patch('flowlib.providers.graph.janus.provider.JANUS_AVAILABLE', True), \
             patch('flowlib.providers.graph.janus.provider.Client', return_value=mock_client), \
             patch('flowlib.providers.graph.janus.provider.DriverRemoteConnection'), \
             patch('flowlib.providers.graph.janus.provider.traversal'), \
             patch('flowlib.providers.graph.janus.provider.GraphBinaryMessageSerializer'):
            
            provider = JanusGraphProvider()
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_bulk_add_entities(self, provider):
        """Test bulk entity addition."""
        entities = [Entity(id=f"entity_{i}", type="bulk") for i in range(5)]
        # Patch add_entity to return a unique contract-compliant GraphStoreResult for each entity
        def add_entity_side_effect(entity):
            return GraphStoreResult(
                success=True,
                stored_entities=[entity.id],
                stored_relationships=[],
                failed_entities=[],
                failed_relationships=[],
                error_details={},
                execution_time_ms=None
            )
        with patch.object(JanusGraphProvider, 'add_entity', new_callable=AsyncMock) as mock_add:
            mock_add.side_effect = add_entity_side_effect
            result = await provider.bulk_add_entities(entities)
            assert isinstance(result, GraphStoreResult)
            assert result.success
            assert len(result.stored_entities) == 5
            assert all(e.id in result.stored_entities for e in entities)
    
    @pytest.mark.asyncio
    async def test_element_to_dict_vertex(self, provider):
        """Test converting vertex to dictionary."""
        mock_vertex = Mock()
        mock_vertex.id = "vertex_123"
        mock_vertex.label = "Entity"
        mock_vertex.properties = {
            "name": [Mock(value="John Doe")],
            "age": [Mock(value=30)]
        }
        
        result = provider._element_to_dict(mock_vertex)
        
        assert result["_id"] == "vertex_123"
        assert result["_label"] == "Entity"
        assert result["name"] == "John Doe"
        assert result["age"] == 30
    
    @pytest.mark.asyncio
    async def test_element_to_dict_multi_valued_properties(self, provider):
        """Test converting element with multi-valued properties."""
        mock_element = Mock()
        mock_element.id = "element_123"
        mock_element.label = "MultiValue"
        mock_element.properties = {
            "tags": [Mock(value="tag1"), Mock(value="tag2")],
            "single": [Mock(value="value")]
        }
        
        result = provider._element_to_dict(mock_element)
        
        assert result["tags"] == ["tag1", "tag2"]
        assert result["single"] == "value"
    
    @pytest.mark.asyncio
    async def test_entity_exists_helper(self, provider):
        """Test _entity_exists helper method."""
        mock_future = Mock()
        mock_future.all.return_value.result.return_value = [{"value": True}]
        provider._client.submit.return_value = mock_future
        
        exists = await provider._entity_exists("test_entity")
        assert exists is True
        
        mock_future.all.return_value.result.return_value = [{"value": False}]
        exists = await provider._entity_exists("nonexistent")
        assert exists is False
    
    @pytest.mark.asyncio
    async def test_search_entities_not_implemented(self, provider):
        """Test that search_entities returns a contract-compliant EntitySearchResult."""
        result = await provider.search_entities("test")
        from flowlib.providers.graph.models import EntitySearchResult
        assert isinstance(result, EntitySearchResult)
        assert hasattr(result, "success")
        assert hasattr(result, "entities")


class TestJanusGraphProviderErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return JanusProviderSettings()
    
    @pytest.mark.asyncio
    async def test_operations_without_gremlin_python(self, provider_settings):
        """Test operations without gremlin_python package."""
        with patch('flowlib.providers.graph.janus.provider.JANUS_AVAILABLE', False):
            provider = JanusGraphProvider(settings=provider_settings)
            
            with pytest.raises(ProviderError, match="Gremlin Python driver is not installed"):
                await provider.initialize()
    
    @pytest.mark.asyncio
    async def test_shutdown_error_handling(self, provider_settings):
        """Test shutdown handles errors gracefully."""
        mock_client = Mock()
        mock_client.close.side_effect = Exception("Close failed")
        provider = JanusGraphProvider(settings=provider_settings)
        provider._client = mock_client
        provider._initialized = True
        try:
            await provider.shutdown()
        except Exception as e:
            assert str(e) == "Close failed"
    
    @pytest.mark.asyncio
    async def test_schema_setup_error_handling(self, provider_settings):
        """Test schema setup handles errors gracefully."""
        mock_client = Mock()
        # First call succeeds (connection test), subsequent calls fail (schema setup)
        mock_future_success = Mock()
        mock_future_success.all.return_value.result.return_value = [{"value": 1}]
        
        mock_future_fail = Mock()
        mock_future_fail.all.return_value.result.side_effect = Exception("Schema error")
        
        mock_client.submit.side_effect = [mock_future_success] + [mock_future_fail] * 10
        
        with patch('flowlib.providers.graph.janus.provider.JANUS_AVAILABLE', True), \
             patch('flowlib.providers.graph.janus.provider.Client', return_value=mock_client), \
             patch('flowlib.providers.graph.janus.provider.DriverRemoteConnection'), \
             patch('flowlib.providers.graph.janus.provider.traversal'), \
             patch('flowlib.providers.graph.janus.provider.GraphBinaryMessageSerializer'):
            
            provider = JanusGraphProvider(settings=provider_settings)
            
            # Should still initialize even if schema setup fails
            await provider.initialize()
            assert provider._initialized
    
    @pytest.mark.asyncio
    async def test_provider_type_validation(self):
        """Test provider type is correctly set."""
        provider = JanusGraphProvider()
        assert provider.provider_type == "graph_db"
    
    @pytest.mark.asyncio
    async def test_connection_configuration(self):
        """Test connection configuration parameters."""
        # Create new settings instance with desired values
        provider_settings = JanusProviderSettings(
            username="admin",
            password="secret",
            connection_pool_size=8
        )
        
        with patch('flowlib.providers.graph.janus.provider.Client') as mock_client_class, \
             patch('flowlib.providers.graph.janus.provider.DriverRemoteConnection'), \
             patch('flowlib.providers.graph.janus.provider.traversal'), \
             patch('flowlib.providers.graph.janus.provider.GraphBinaryMessageSerializer'), \
             patch('flowlib.providers.graph.janus.provider.JANUS_AVAILABLE', True):
            
            mock_client = Mock()
            mock_future = Mock()
            mock_future.all.return_value.result.return_value = [{"value": 1}]
            mock_client.submit.return_value = mock_future
            mock_client_class.return_value = mock_client
            
            provider = JanusGraphProvider(settings=provider_settings)
            await provider.initialize()
            
            # Verify client was created with correct parameters
            mock_client_class.assert_called_once()
            call_args = mock_client_class.call_args[1]
            assert call_args["username"] == "admin"
            assert call_args["password"] == "secret"
            assert call_args["pool_size"] == 8
    
    @pytest.mark.asyncio
    async def test_attribute_serialization_edge_cases(self, provider_settings):
        """Test attribute serialization handles edge cases."""
        mock_client = Mock()
        mock_future = Mock()
        mock_future.all.return_value.result.return_value = [{"value": False}]  # Entity doesn't exist
        mock_client.submit.return_value = mock_future
        with patch('flowlib.providers.graph.janus.provider.JANUS_AVAILABLE', True), \
             patch('flowlib.providers.graph.janus.provider.Client', return_value=mock_client), \
             patch('flowlib.providers.graph.janus.provider.DriverRemoteConnection'), \
             patch('flowlib.providers.graph.janus.provider.traversal'), \
             patch('flowlib.providers.graph.janus.provider.GraphBinaryMessageSerializer'):
            provider = JanusGraphProvider(settings=provider_settings)
            await provider.initialize()
            # Test entity with complex attributes
            entity = Entity(
                id="complex_entity",
                type="test",
                attributes={
                    "unicode": EntityAttribute(name="unicode", value="测试"),
                    "special_chars": EntityAttribute(name="special_chars", value="!@#$%^&*()"),
                    "empty": EntityAttribute(name="empty", value=""),
                }
            )
            # Should handle complex attributes without error
            result = await provider.add_entity(entity)
            from flowlib.providers.graph.models import GraphStoreResult
            assert isinstance(result, GraphStoreResult)
            assert result.success
            assert entity.id in result.stored_entities

    @pytest_asyncio.fixture
    async def provider_fully_mocked(self, provider_settings, mock_client, mock_traversal, mock_connection):
        with patch('flowlib.providers.graph.janus.provider.Client', return_value=mock_client), \
             patch('flowlib.providers.graph.janus.provider.DriverRemoteConnection', return_value=mock_connection), \
             patch('flowlib.providers.graph.janus.provider.traversal') as mock_trav_func, \
             patch('flowlib.providers.graph.janus.provider.GraphBinaryMessageSerializer'), \
             patch('flowlib.providers.graph.janus.provider.JANUS_AVAILABLE', True):
            mock_trav_func.return_value.withRemote.return_value = mock_traversal
            provider = JanusGraphProvider(settings=provider_settings)
            await provider.initialize()
            yield provider

    @pytest.mark.asyncio
    async def test_query_model_driven(self, provider_fully_mocked):
        """Test query returns EntitySearchResult and only model-driven data."""
        # Pass contract-compliant params with entity_type
        params = {'entity_type': 'person'}
        result = await provider_fully_mocked.query("find_entities", params=params)
        assert isinstance(result, type(provider_fully_mocked).query.__annotations__['return'])
        assert result.success
        for entity in result.entities:
            assert isinstance(entity, Entity)
        assert not isinstance(result, dict)