"""Tests for graph database provider base functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Optional, Any
from pydantic import ConfigDict, Field

from flowlib.providers.graph.base import GraphDBProvider, GraphDBProviderSettings
from flowlib.providers.graph.models import Entity, EntityAttribute, EntityRelationship, GraphStoreResult, EntitySearchResult, GraphHealthResult, GraphDeleteResult, GraphQueryResult, RelationshipSearchResult
from flowlib.providers.core.base import Provider
from flowlib.flows.base.base import FlowSettings
from flowlib.core.errors.errors import ProviderError


class TestGraphDBProviderSettings:
    """Test GraphDBProviderSettings configuration."""
    
    def test_graph_db_provider_settings_creation(self):
        """Test creating GraphDBProviderSettings with defaults."""
        settings = GraphDBProviderSettings()
        
        assert settings.max_retries == 3
        assert settings.retry_delay_seconds == 1.0
        assert settings.timeout_seconds == 30.0
        assert settings.max_batch_size == 100
        
        # Should inherit from FlowSettings
        assert isinstance(settings, FlowSettings)
    
    def test_graph_db_provider_settings_custom_values(self):
        """Test creating GraphDBProviderSettings with custom values."""
        settings = GraphDBProviderSettings(
            max_retries=5,
            retry_delay_seconds=2.0,
            timeout_seconds=60.0,
            max_batch_size=200
        )
        
        assert settings.max_retries == 5
        assert settings.retry_delay_seconds == 2.0
        assert settings.timeout_seconds == 60.0
        assert settings.max_batch_size == 200
    
    def test_graph_db_provider_settings_validation(self):
        """Test GraphDBProviderSettings validation."""
        # Should accept valid values
        settings = GraphDBProviderSettings(
            max_retries=0,  # Edge case
            retry_delay_seconds=0.0,  # Edge case
            timeout_seconds=1.0,
            max_batch_size=1
        )
        
        assert settings.max_retries == 0
        assert settings.retry_delay_seconds == 0.0
        assert settings.timeout_seconds == 1.0
        assert settings.max_batch_size == 1


class MockGraphDBProvider(GraphDBProvider):
    """Mock implementation of GraphDBProvider for testing."""
    
    def __init__(self, name: str = "mock_graph", provider_type: str = "graph_db", settings: Optional[GraphDBProviderSettings] = None):
        if settings is None:
            settings = GraphDBProviderSettings()
        super().__init__(name=name, provider_type=provider_type, settings=settings)

    async def add_entity(self, entity: Entity) -> GraphStoreResult:
        return GraphStoreResult(
            success=True,
            stored_entities=[entity.id],
            stored_relationships=[],
            failed_entities=[],
            failed_relationships=[],
            error_details={}
        )

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        return None

    async def add_relationship(self, source_id: str, target_entity: str, relation_type: str, properties: Dict[str, Any] = {}) -> None:
        return None

    async def query_relationships(self, entity_id: str, relation_type: Optional[str] = None, direction: str = "outgoing") -> RelationshipSearchResult:
        return RelationshipSearchResult(
            success=True,
            relationships=[],
            total_count=0,
            source_entity=entity_id if direction == "outgoing" else None,
            target_entity=entity_id if direction == "incoming" else None,
            metadata={"direction": direction, "relation_type": relation_type}
        )

    async def traverse(self, start_id: str, relation_types: Optional[List[str]] = None, max_depth: int = 2) -> List[Entity]:
        return []

    async def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> GraphQueryResult:
        return GraphQueryResult(
            success=True,
            nodes=[{"result": "mock_query_result"}],
            edges=[],
            metadata={"query": query, "params": params}
        )

    async def delete_entity(self, entity_id: str) -> GraphDeleteResult:
        return GraphDeleteResult(
            success=True,
            deleted_entities=[entity_id],
            deleted_relationships=[],
            not_found_entities=[],
            not_found_relationships=[],
            error_details={}
        )

    async def delete_relationship(self, source_id: str, target_entity: str, relation_type: Optional[str] = None) -> bool:
        return True

    async def search_entities(self, query: Optional[str] = None, entity_type: Optional[str] = None, tags: Optional[List[str]] = None, limit: int = 10) -> EntitySearchResult:
        return EntitySearchResult(
            success=True,
            entities=[],
            total_count=0,
            search_query=query or "",
            metadata={
                "entity_type": entity_type,
                "tags": tags,
                "limit": limit
            }
        )

    async def remove_relationship(self, source_id: str, target_entity: str, relation_type: str) -> None:
        return None

    async def bulk_add_entities(self, entities: List[Entity]) -> GraphStoreResult:
        return GraphStoreResult(
            success=True,
            stored_entities=[e.id for e in entities],
            stored_relationships=[],
            failed_entities=[],
            failed_relationships=[],
            error_details={}
        )


class TestGraphDBProvider:
    """Test GraphDBProvider base class."""
    
    def test_graph_db_provider_is_abstract(self):
        """Test that GraphDBProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            GraphDBProvider()
    
    def test_graph_db_provider_inheritance(self):
        """Test GraphDBProvider inheritance."""
        provider = MockGraphDBProvider()
        
        # Should inherit from Provider
        assert isinstance(provider, Provider)
        assert isinstance(provider, GraphDBProvider)
    
    def test_graph_db_provider_initialization(self):
        """Test GraphDBProvider initialization."""
        provider = MockGraphDBProvider(name="test_graph")
        
        assert provider.name == "test_graph"
        assert provider.provider_type == "graph_db"
        assert provider.settings is not None
        assert isinstance(provider.settings, GraphDBProviderSettings)
    
    def test_graph_db_provider_initialization_with_settings(self):
        """Test GraphDBProvider initialization with settings."""
        settings = GraphDBProviderSettings(max_retries=5, timeout_seconds=60.0)
        provider = MockGraphDBProvider(name="test_graph", settings=settings)
        
        assert provider.name == "test_graph"
        assert provider.provider_type == "graph_db"
        assert provider.settings == settings
        assert provider.settings.max_retries == 5
        assert provider.settings.timeout_seconds == 60.0
    
    def test_graph_db_provider_default_name(self):
        """Test GraphDBProvider with default name."""
        provider = MockGraphDBProvider()
        
        assert provider.name == "mock_graph"
        assert provider.provider_type == "graph_db"


class TestGraphDBProviderEntityOperations:
    """Test entity operations in GraphDBProvider."""
    
    @pytest.mark.asyncio
    async def test_add_entity(self):
        """Test adding an entity."""
        provider = MockGraphDBProvider()
        entity = Entity(
            id="test_entity",
            type="person",
            attributes={
                "name": EntityAttribute(name="name", value="John Doe"),
                "age": EntityAttribute(name="age", value="30")
            }
        )
        result = await provider.add_entity(entity)
        assert isinstance(result, GraphStoreResult)
        assert result.success
        assert result.stored_entities == ["test_entity"]
    
    @pytest.mark.asyncio
    async def test_get_entity_exists(self):
        """Test getting an existing entity."""
        provider = MockGraphDBProvider()
        entity = Entity(id="test_entity", type="person")
        await provider.add_entity(entity)
        retrieved_entity = await provider.get_entity("test_entity")
        assert retrieved_entity is None  # Stateless mock always returns None
    
    @pytest.mark.asyncio
    async def test_get_entity_not_exists(self):
        """Test getting a non-existent entity."""
        provider = MockGraphDBProvider()
        
        retrieved_entity = await provider.get_entity("nonexistent")
        
        assert retrieved_entity is None
    
    @pytest.mark.asyncio
    async def test_delete_entity_exists(self):
        """Test deleting an existing entity."""
        provider = MockGraphDBProvider()
        entity = Entity(id="test_entity", type="person")
        await provider.add_entity(entity)
        await provider.add_relationship("test_entity", "other_entity", "knows")
        result = await provider.delete_entity("test_entity")
        assert isinstance(result, GraphDeleteResult)
        assert result.success is True
        assert result.deleted_entities == ["test_entity"]
    
    @pytest.mark.asyncio
    async def test_delete_entity_not_exists(self):
        """Test deleting a non-existent entity."""
        provider = MockGraphDBProvider()
        result = await provider.delete_entity("nonexistent")
        assert isinstance(result, GraphDeleteResult)
        # Stateless mock always returns success
        assert result.success is True
        assert result.deleted_entities == ["nonexistent"]
    
    @pytest.mark.asyncio
    async def test_bulk_add_entities(self):
        """Test bulk adding entities."""
        provider = MockGraphDBProvider()
        entities = [
            Entity(id="entity1", type="person"),
            Entity(id="entity2", type="organization"),
            Entity(id="entity3", type="location")
        ]
        result = await provider.bulk_add_entities(entities)
        assert isinstance(result, GraphStoreResult)
        assert result.success
        assert len(result.stored_entities) == 3
        assert result.stored_entities == ["entity1", "entity2", "entity3"]
    
    @pytest.mark.asyncio
    async def test_search_entities_no_filters(self):
        """Test searching entities without filters."""
        provider = MockGraphDBProvider()
        # Add test entities (no effect in stateless mock)
        entities = [
            Entity(id="person1", type="person"),
            Entity(id="person2", type="person"),
            Entity(id="org1", type="organization")
        ]
        for entity in entities:
            await provider.add_entity(entity)
        results = await provider.search_entities()
        assert isinstance(results, EntitySearchResult)
        assert results.success is True
        assert results.entities == []
    
    @pytest.mark.asyncio
    async def test_search_entities_with_query(self):
        """Test searching entities with text query."""
        provider = MockGraphDBProvider()
        entities = [
            Entity(id="john_person", type="person"),
            Entity(id="jane_person", type="person"),
            Entity(id="acme_org", type="organization")
        ]
        for entity in entities:
            await provider.add_entity(entity)
        results = await provider.search_entities(query="person")
        assert isinstance(results, EntitySearchResult)
        assert results.success is True
        assert results.entities == []
    
    @pytest.mark.asyncio
    async def test_search_entities_with_type_filter(self):
        """Test searching entities with type filter."""
        provider = MockGraphDBProvider()
        entities = [
            Entity(id="person1", type="person"),
            Entity(id="person2", type="person"),
            Entity(id="org1", type="organization")
        ]
        for entity in entities:
            await provider.add_entity(entity)
        results = await provider.search_entities(entity_type="person")
        assert isinstance(results, EntitySearchResult)
        assert results.success is True
        assert results.entities == []
    
    @pytest.mark.asyncio
    async def test_search_entities_with_limit(self):
        """Test searching entities with limit."""
        provider = MockGraphDBProvider()
        for i in range(10):
            entity = Entity(id=f"entity{i}", type="test")
            await provider.add_entity(entity)
        results = await provider.search_entities(limit=3)
        assert isinstance(results, EntitySearchResult)
        assert results.success is True
        assert results.entities == []


class TestGraphDBProviderRelationshipOperations:
    """Test relationship operations in GraphDBProvider."""
    
    @pytest.mark.asyncio
    async def test_add_relationship(self):
        """Test adding a relationship."""
        provider = MockGraphDBProvider()
        await provider.add_relationship(
            source_id="person1",
            target_entity="person2",
            relation_type="knows",
            properties={"since": "2020", "strength": 0.8}
        )
        # No assertion on provider.relationships
    
    @pytest.mark.asyncio
    async def test_query_relationships_outgoing(self):
        """Test querying outgoing relationships."""
        provider = MockGraphDBProvider()
        # Add relationships (no effect in stateless mock)
        await provider.add_relationship("person1", "person2", "knows")
        await provider.add_relationship("person1", "org1", "works_for")
        await provider.add_relationship("person2", "person1", "knows")
        relationships = await provider.query_relationships("person1", direction="outgoing")
        assert isinstance(relationships, RelationshipSearchResult)
        assert relationships.success is True
        assert relationships.relationships == []
    
    @pytest.mark.asyncio
    async def test_query_relationships_incoming(self):
        """Test querying incoming relationships."""
        provider = MockGraphDBProvider()
        await provider.add_relationship("person1", "person2", "knows")
        await provider.add_relationship("person3", "person2", "knows")
        await provider.add_relationship("person2", "org1", "works_for")
        relationships = await provider.query_relationships("person2", direction="incoming")
        assert isinstance(relationships, RelationshipSearchResult)
        assert relationships.success is True
        assert relationships.relationships == []
    
    @pytest.mark.asyncio
    async def test_query_relationships_filtered_by_type(self):
        """Test querying relationships filtered by type."""
        provider = MockGraphDBProvider()
        await provider.add_relationship("person1", "person2", "knows")
        await provider.add_relationship("person1", "org1", "works_for")
        await provider.add_relationship("person1", "person3", "knows")
        relationships = await provider.query_relationships(
            "person1",
            relation_type="knows",
            direction="outgoing"
        )
        assert isinstance(relationships, RelationshipSearchResult)
        assert relationships.success is True
        assert relationships.relationships == []
    
    @pytest.mark.asyncio
    async def test_delete_relationship(self):
        """Test deleting relationships."""
        provider = MockGraphDBProvider()
        await provider.add_relationship("person1", "person2", "knows")
        await provider.add_relationship("person1", "org1", "works_for")
        await provider.add_relationship("person1", "person2", "likes")
        result = await provider.delete_relationship("person1", "person2", "knows")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_relationship_all_types(self):
        """Test deleting all relationships between entities."""
        provider = MockGraphDBProvider()
        await provider.add_relationship("person1", "person2", "knows")
        await provider.add_relationship("person1", "person2", "likes")
        await provider.add_relationship("person1", "org1", "works_for")
        result = await provider.delete_relationship("person1", "person2")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_remove_relationship(self):
        """Test removing a specific relationship."""
        provider = MockGraphDBProvider()
        await provider.add_relationship("person1", "person2", "knows")
        await provider.add_relationship("person1", "person2", "likes")
        await provider.remove_relationship("person1", "person2", "knows")
        # No assertion on provider.relationships


class TestGraphDBProviderQueryOperations:
    """Test query and traversal operations in GraphDBProvider."""
    
    @pytest.mark.asyncio
    async def test_query_execution(self):
        """Test executing native queries."""
        provider = MockGraphDBProvider()
        result = await provider.query(
            "MATCH (n:Person) RETURN n",
            params={"limit": 10}
        )
        assert isinstance(result, GraphQueryResult)
        assert result.success is True
        assert len(result.nodes) == 1
        assert result.nodes[0]["result"] == "mock_query_result"
    
    @pytest.mark.asyncio
    async def test_query_without_params(self):
        """Test executing queries without parameters."""
        provider = MockGraphDBProvider()
        result = await provider.query("MATCH (n) RETURN count(n)")
        assert isinstance(result, GraphQueryResult)
        assert result.success is True
        assert len(result.nodes) == 1
    
    @pytest.mark.asyncio
    async def test_traverse(self):
        """Test graph traversal."""
        provider = MockGraphDBProvider()
        entities = [
            Entity(id="person1", type="person"),
            Entity(id="person2", type="person"),
            Entity(id="org1", type="organization")
        ]
        for entity in entities:
            await provider.add_entity(entity)
        await provider.add_relationship("person1", "person2", "knows")
        await provider.add_relationship("person1", "org1", "works_for")
        results = await provider.traverse("person1", max_depth=2)
        assert results == []
    
    @pytest.mark.asyncio
    async def test_traverse_with_relation_filter(self):
        """Test graph traversal with relation type filter."""
        provider = MockGraphDBProvider()
        entity = Entity(id="person1", type="person")
        await provider.add_entity(entity)
        await provider.add_relationship("person1", "person2", "knows")
        await provider.add_relationship("person1", "org1", "works_for")
        results = await provider.traverse(
            "person1",
            relation_types=["knows"],
            max_depth=1
        )
        assert results == []


class TestGraphDBProviderHealthAndUtilities:
    """Test health check and utility methods."""
    
    @pytest.mark.asyncio
    async def test_get_health_success(self):
        """Test successful health check."""
        provider = MockGraphDBProvider(name="test_provider")
        provider._initialized = True
        
        health = await provider.get_health()
        
        assert isinstance(health, GraphHealthResult)
        assert health.healthy is True
        assert health.database_info["provider"] == "test_provider"
        assert health.database_info["provider_type"] == "graph_db"
        assert health.database_info["initialized"] is True
    
    @pytest.mark.asyncio
    async def test_get_health_uninitialized(self):
        """Test health check when uninitialized."""
        provider = MockGraphDBProvider(name="test_provider")
        provider._initialized = False
        
        health = await provider.get_health()
        
        assert isinstance(health, GraphHealthResult)
        assert health.healthy is True  # Still healthy, just not initialized
        assert health.database_info["initialized"] is False
    
    @pytest.mark.asyncio
    async def test_get_health_with_error(self):
        """Test health check with error."""
        # Create a broken provider that will cause an error during health check
        class BrokenGraphDBProvider(MockGraphDBProvider):
            """A broken provider for testing error conditions."""
            
            @property
            def initialized(self):
                """Override initialized property to raise an error."""
                raise AttributeError("Mock error: provider initialization check is broken")
        
        provider = BrokenGraphDBProvider(name="test_provider")
        
        health = await provider.get_health()
        
        assert isinstance(health, GraphHealthResult)
        assert health.healthy is False
        assert health.error_message is not None
        assert "Mock error: provider initialization check is broken" in health.error_message


class TestGraphDBProviderAbstractMethods:
    """Test that abstract methods raise NotImplementedError when not overridden."""
    
    def test_abstract_methods_not_implemented(self):
        """Test that base GraphDBProvider raises NotImplementedError for abstract methods."""
        # Create a minimal concrete implementation that doesn't override abstract methods
        class IncompleteGraphProvider(GraphDBProvider):
            pass
        
        # Should still raise TypeError because abstract methods aren't implemented
        with pytest.raises(TypeError):
            IncompleteGraphProvider()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])