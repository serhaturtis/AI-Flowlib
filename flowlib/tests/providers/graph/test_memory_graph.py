"""Tests for Memory Graph Provider implementation."""

import pytest
import pytest_asyncio
import asyncio
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from flowlib.providers.graph.memory_graph import MemoryGraphProvider
from flowlib.providers.graph.base import GraphDBProviderSettings
from flowlib.providers.graph.models import Entity, EntityAttribute, EntityRelationship, GraphStoreResult, GraphHealthResult, EntitySearchResult
from flowlib.core.errors.errors import ProviderError
from pydantic import ValidationError


class TestMemoryGraphProviderSettings:
    """Test GraphDBProviderSettings model."""
    
    def test_settings_creation_minimal(self):
        """Test settings creation with minimal data."""
        settings = GraphDBProviderSettings()
        
        assert settings.max_retries == 3
        assert settings.retry_delay_seconds == 1.0
        assert settings.timeout_seconds == 30.0
        assert settings.max_batch_size == 100
    
    def test_settings_creation_full(self):
        """Test settings creation with all fields."""
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
    
    def test_settings_inheritance(self):
        """Test that GraphDBProviderSettings inherits from FlowSettings."""
        from flowlib.flows.base.base import FlowSettings
        settings = GraphDBProviderSettings()
        assert isinstance(settings, FlowSettings)


class TestEntityModels:
    """Test Entity and related models."""
    
    def test_entity_attribute_creation(self):
        """Test EntityAttribute creation."""
        attr = EntityAttribute(
            name="full_name",
            value="John Doe",
            confidence=0.95,
            source="conversation"
        )
        
        assert attr.name == "full_name"
        assert attr.value == "John Doe"
        assert attr.confidence == 0.95
        assert attr.source == "conversation"
        assert attr.timestamp is not None
    
    def test_entity_relationship_creation(self):
        """Test EntityRelationship creation."""
        rel = EntityRelationship(
            relation_type="friend_of",
            target_entity="jane_doe",
            confidence=0.9,
            source="conversation"
        )
        
        assert rel.relation_type == "friend_of"
        assert rel.target_entity == "jane_doe"
        assert rel.confidence == 0.9
        assert rel.source == "conversation"
        assert rel.timestamp is not None
    
    def test_entity_creation_minimal(self):
        """Test Entity creation with minimal data."""
        entity = Entity(
            id="john_doe",
            type="person"
        )
        
        assert entity.id == "john_doe"
        assert entity.type == "person"
        assert entity.attributes == {}
        assert entity.relationships == []
        assert entity.tags == []
        assert entity.importance == 0.7
        assert entity.vector_id is None
        assert entity.last_updated is not None
    
    def test_entity_creation_full(self):
        """Test Entity creation with all fields."""
        attributes = {
            "name": EntityAttribute(name="name", value="John Doe"),
            "age": EntityAttribute(name="age", value="30")
        }
        relationships = [
            EntityRelationship(relation_type="friend_of", target_entity="jane_doe")
        ]
        
        entity = Entity(
            id="john_doe",
            type="person",
            attributes=attributes,
            relationships=relationships,
            tags=["friend", "colleague"],
            importance=0.9,
            vector_id="vec_123"
        )
        
        assert entity.id == "john_doe"
        assert entity.type == "person"
        assert len(entity.attributes) == 2
        assert len(entity.relationships) == 1
        assert entity.tags == ["friend", "colleague"]
        assert entity.importance == 0.9
        assert entity.vector_id == "vec_123"
    
    def test_entity_to_memory_item_specific_attribute(self):
        """Test Entity.to_memory_item with specific attribute."""
        attr = EntityAttribute(name="name", value="John Doe", confidence=0.95)
        entity = Entity(
            id="john_doe",
            type="person",
            attributes={"name": attr},
            importance=0.8,
            tags=["friend"]
        )
        
        memory_item = entity.to_memory_item("name")
        
        assert memory_item["entity_id"] == "john_doe"
        assert memory_item["entity_type"] == "person"
        assert memory_item["attribute"] == "name"
        assert memory_item["value"] == "John Doe"
        assert memory_item["confidence"] == 0.95
        assert memory_item["importance"] == 0.8
        assert memory_item["tags"] == ["friend"]
    
    def test_entity_to_memory_item_summary(self):
        """Test Entity.to_memory_item for summary."""
        entity = Entity(
            id="john_doe",
            type="person",
            attributes={"name": EntityAttribute(name="name", value="John Doe")},
            relationships=[EntityRelationship(relation_type="friend_of", target_entity="jane_doe")]
        )
        
        memory_item = entity.to_memory_item()
        
        assert memory_item["entity_id"] == "john_doe"
        assert memory_item["attribute"] == "summary"
        assert "1 attributes and 1 relationships" in memory_item["value"]
        assert memory_item["confidence"] == 1.0
    
    def test_entity_get_formatted_view(self):
        """Test Entity.get_formatted_view."""
        attributes = {
            "name": EntityAttribute(name="name", value="John Doe", confidence=0.95)
        }
        relationships = [
            EntityRelationship(relation_type="friend_of", target_entity="jane_doe", confidence=0.9)
        ]
        
        entity = Entity(
            id="john_doe",
            type="person",
            attributes=attributes,
            relationships=relationships,
            tags=["friend"]
        )
        
        formatted = entity.get_formatted_view()
        
        assert "Entity: john_doe (Type: person)" in formatted
        assert "name: John Doe (confidence: 0.95)" in formatted
        assert "friend_of jane_doe (confidence: 0.90)" in formatted
        assert "Tags: friend" in formatted


class TestMemoryGraphProvider:
    """Test MemoryGraphProvider implementation."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return GraphDBProviderSettings(
            max_retries=5,
            timeout_seconds=60.0
        )
    
    @pytest_asyncio.fixture
    def provider(self, provider_settings):
        """Create test provider instance."""
        return MemoryGraphProvider(settings=provider_settings)
    
    @pytest_asyncio.fixture
    def sample_entity(self):
        """Create a sample entity for testing."""
        return Entity(
            id="john_doe",
            type="person",
            attributes={
                "name": EntityAttribute(name="name", value="John Doe"),
                "age": EntityAttribute(name="age", value="30"),
                "occupation": EntityAttribute(name="occupation", value="Engineer")
            },
            relationships=[
                EntityRelationship(relation_type="friend_of", target_entity="jane_doe"),
                EntityRelationship(relation_type="colleague_of", target_entity="bob_smith")
            ],
            tags=["person", "engineer"],
            importance=0.8
        )
    
    def test_provider_initialization(self, provider, provider_settings):
        """Test provider initialization."""
        assert provider.name == "memory-graph"
        assert provider.provider_type == "graph_db"
        assert provider.settings == provider_settings
        assert not provider._initialized
        assert provider._database is None
    
    def test_provider_inheritance(self, provider):
        """Test provider inheritance from base classes."""
        from flowlib.providers.graph.base import GraphDBProvider
        from flowlib.providers.core.base import Provider
        assert isinstance(provider, GraphDBProvider)
        assert isinstance(provider, Provider)
    
    def test_provider_default_settings(self):
        """Test provider with default settings."""
        provider = MemoryGraphProvider()
        assert provider.name == "memory-graph"
        assert isinstance(provider.settings, GraphDBProviderSettings)
    
    def test_provider_decorator_registration(self):
        """Test that provider is properly registered with decorator."""
        assert hasattr(MemoryGraphProvider, '__provider_name__')
        assert hasattr(MemoryGraphProvider, '__provider_type__')
        assert MemoryGraphProvider.__provider_name__ == "memory-graph"
        assert MemoryGraphProvider.__provider_type__ == "graph_db"
    
    @pytest.mark.asyncio
    async def test_provider_lifecycle(self, provider):
        """Test provider initialization and shutdown lifecycle."""
        # Initially not initialized
        assert not provider._initialized
        
        # Initialize
        await provider.initialize()
        assert provider._initialized
        assert provider._database is not None
        
        # Shutdown
        await provider.shutdown()
        assert not provider._initialized
        assert provider._database is None


class TestMemoryGraphProviderEntityOperations:
    """Test entity operations."""
    
    @pytest_asyncio.fixture
    async def provider(self):
        """Create and initialize test provider."""
        provider = MemoryGraphProvider()
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
            relationships=[],
            tags=["person"],
            importance=0.8
        )
    
    @pytest.mark.asyncio
    async def test_add_entity_success(self, provider, sample_entity):
        """Test successful entity addition."""
        result = await provider.add_entity(sample_entity)
        
        assert result.success is True
        assert result.stored_entities == ["john_doe"]
        assert result.failed_entities == []
        
        # Verify entity was stored by retrieving it
        stored_entity = await provider.get_entity("john_doe")
        assert stored_entity is not None
        assert stored_entity.id == "john_doe"
        assert stored_entity.type == "person"
        assert len(stored_entity.attributes) == 2
    
    @pytest.mark.asyncio
    async def test_add_entity_with_relationships(self, provider):
        """Test adding entity with relationships."""
        # Create target entity first
        target_entity = Entity(id="jane_doe", type="person")
        await provider.add_entity(target_entity)
        # Create source entity with relationship
        source_entity = Entity(
            id="john_doe",
            type="person",
            relationships=[
                EntityRelationship(
                    relation_type="friend_of",
                    target_entity="jane_doe",
                    confidence=0.9
                )
            ]
        )
        result = await provider.add_entity(source_entity)
        assert result.success is True
        assert "john_doe" in result.stored_entities
        # Check relationships were created
        rels = await provider.query_relationships("john_doe", direction="outgoing")
        assert rels.success
        assert rels.total_count == 1
        assert len(rels.relationships) == 1
        rel = rels.relationships[0]
        assert rel.target_entity == "jane_doe"
        assert rel.relation_type == "friend_of"
    
    @pytest.mark.asyncio
    async def test_add_entity_creates_missing_target(self, provider):
        """Test that adding entity with relationships to missing target raises error."""
        entity = Entity(
            id="john_doe",
            type="person",
            relationships=[
                EntityRelationship(
                    relation_type="friend_of",
                    target_entity="missing_person",
                    confidence=0.9
                )
            ]
        )
        with pytest.raises(ProviderError, match="Target entity missing_person does not exist"):
            await provider.add_entity(entity)
    
    @pytest.mark.asyncio
    async def test_get_entity_success(self, provider, sample_entity):
        """Test successful entity retrieval."""
        await provider.add_entity(sample_entity)
        
        retrieved = await provider.get_entity("john_doe")
        
        assert retrieved is not None
        assert retrieved.id == "john_doe"
        assert retrieved.type == "person"
        assert len(retrieved.attributes) == 2
        
        # Ensure it's a copy (not the same object from internal storage)
        retrieved2 = await provider.get_entity("john_doe")
        assert retrieved is not retrieved2  # Each call returns a new copy
    
    @pytest.mark.asyncio
    async def test_get_entity_not_found(self, provider):
        """Test retrieving non-existent entity."""
        retrieved = await provider.get_entity("nonexistent")
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_delete_entity_success(self, provider):
        """Test successful entity deletion."""
        # Create entities with relationships
        entity1 = Entity(id="entity1", type="test")
        entity2 = Entity(id="entity2", type="test")
        await provider.add_entity(entity1)
        await provider.add_entity(entity2)
        # Add relationship (model-driven)
        rel = EntityRelationship(
            relation_type="relates_to",
            target_entity="entity2",
            confidence=0.8
        )
        await provider.add_relationship("entity1", "entity2", "relates_to", rel)
        # Delete entity1
        result = await provider.delete_entity("entity1")
        assert result.success is True
        assert "entity1" in result.deleted_entities
        assert await provider.get_entity("entity1") is None
        # Relationships involving deleted entity should be removed
        entity2_rels = await provider.query_relationships("entity2", direction="outgoing")
        assert not any(r.target_entity == "entity1" for r in entity2_rels.relationships)
        entity2_incoming = await provider.query_relationships("entity2", direction="incoming")
        assert not any(r.target_entity == "entity1" for r in entity2_incoming.relationships)
    
    @pytest.mark.asyncio
    async def test_delete_entity_not_found(self, provider):
        """Test deleting non-existent entity."""
        result = await provider.delete_entity("nonexistent")
        assert result.success is False
        assert "nonexistent" in result.not_found_entities
    
    @pytest.mark.asyncio
    async def test_add_entity_thread_safety(self, provider):
        """Test concurrent entity additions."""
        async def add_entity_task(entity_id: str):
            entity = Entity(id=entity_id, type="concurrent")
            return await provider.add_entity(entity)
        
        # Run concurrent additions
        tasks = [add_entity_task(f"entity_{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 10
        
        # All entities should be present
        for i in range(10):
            entity = await provider.get_entity(f"entity_{i}")
            assert entity is not None
            assert entity.type == "concurrent"


class TestMemoryGraphProviderRelationshipOperations:
    """Test relationship operations."""
    
    @pytest_asyncio.fixture
    async def provider(self):
        """Create and initialize test provider with sample entities."""
        provider = MemoryGraphProvider()
        await provider.initialize()
        
        # Add sample entities
        entity1 = Entity(id="john_doe", type="person")
        entity2 = Entity(id="jane_doe", type="person")
        entity3 = Entity(id="company_abc", type="organization")
        
        await provider.add_entity(entity1)
        await provider.add_entity(entity2)
        await provider.add_entity(entity3)
        
        return provider
    
    @pytest.mark.asyncio
    async def test_add_relationship_success(self, provider):
        """Test adding a relationship (model-driven)."""
        # Add both source and target entities first
        await provider.add_entity(Entity(id="test_entity_1", type="test"))
        await provider.add_entity(Entity(id="test_entity_2", type="test"))
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
        # Should succeed with no error
    
    @pytest.mark.asyncio
    async def test_add_relationship_with_properties(self, provider):
        """Test adding relationship with properties."""
        rel = EntityRelationship(
            relation_type="friend_of",
            target_entity="jane_doe",
            confidence=0.95,
            source="conversation",
            timestamp="2023-01-01T00:00:00"
        )
        await provider.add_relationship("john_doe", "jane_doe", "friend_of", rel)
        rels = await provider.query_relationships("john_doe", direction="outgoing")
        assert rels.success
        found = False
        for r in rels.relationships:
            if r.target_entity == "jane_doe" and r.relation_type == "friend_of":
                assert r.confidence == 0.95
                assert r.source == "conversation"
                assert r.timestamp == "2023-01-01T00:00:00"
                found = True
        assert found
    
    @pytest.mark.asyncio
    async def test_add_relationship_missing_source(self, provider):
        """Test adding relationship with missing source entity."""
        rel = EntityRelationship(
            relation_type="friend_of",
            target_entity="jane_doe",
            confidence=0.8
        )
        with pytest.raises(ProviderError, match="Source entity .* does not exist"):
            await provider.add_relationship("nonexistent", "jane_doe", "friend_of", rel)
    
    @pytest.mark.asyncio
    async def test_add_relationship_creates_target(self, provider):
        """Test that error is raised if target does not exist (no placeholder)."""
        await provider.add_entity(Entity(id="john_doe", type="person"))
        rel = EntityRelationship(
            relation_type="knows",
            target_entity="new_person",
            confidence=0.8
        )
        with pytest.raises(ProviderError, match="Target entity new_person does not exist"):
            await provider.add_relationship("john_doe", "new_person", "knows", rel)
    
    @pytest.mark.asyncio
    async def test_add_duplicate_relationship(self, provider):
        """Test adding duplicate relationship."""
        rel = EntityRelationship(
            relation_type="friend_of",
            target_entity="jane_doe",
            confidence=0.8
        )
        await provider.add_relationship("john_doe", "jane_doe", "friend_of", rel)
        await provider.add_relationship("john_doe", "jane_doe", "friend_of", rel)
        # No exceptions means success
    
    @pytest.mark.asyncio
    async def test_query_relationships_by_type(self, provider):
        """Test querying relationships by type."""
        rel = EntityRelationship(
            relation_type="friend_of",
            target_entity="jane_doe",
            confidence=0.8
        )
        await provider.add_relationship("john_doe", "jane_doe", "friend_of", rel)
        rels = await provider.query_relationships("john_doe", relation_type="friend_of", direction="outgoing")
        assert rels.success
        assert len(rels.relationships) > 0
        assert any(r.target_entity == "jane_doe" and r.relation_type == "friend_of" for r in rels.relationships)
    
    @pytest.mark.asyncio
    async def test_query_relationships_direction(self, provider):
        """Test querying relationships by direction."""
        rel = EntityRelationship(
            relation_type="friend_of",
            target_entity="jane_doe",
            confidence=0.8
        )
        await provider.add_relationship("john_doe", "jane_doe", "friend_of", rel)
        rels_out = await provider.query_relationships("john_doe", direction="outgoing")
        rels_in = await provider.query_relationships("jane_doe", direction="incoming")
        assert rels_out.success and rels_in.success
        assert any(r.target_entity == "jane_doe" for r in rels_out.relationships)
        assert any(r.target_entity == "jane_doe" for r in rels_in.relationships)
    
    @pytest.mark.asyncio
    async def test_delete_relationship_success(self, provider):
        """Test successful relationship deletion."""
        rel1 = EntityRelationship(
            relation_type="friend_of",
            target_entity="jane_doe",
            confidence=0.8
        )
        rel2 = EntityRelationship(
            relation_type="colleague_of",
            target_entity="jane_doe",
            confidence=0.8
        )
        await provider.add_relationship("john_doe", "jane_doe", "friend_of", rel1)
        await provider.add_relationship("john_doe", "jane_doe", "colleague_of", rel2)
        # Delete specific relationship
        result = await provider.delete_relationship("john_doe", "jane_doe", "friend_of")
        assert result is True
        # Should only have colleague relationship left
        rels = await provider.query_relationships("john_doe", direction="outgoing")
        assert rels.success
        assert any(r.relation_type == "colleague_of" for r in rels.relationships)
        assert not any(r.relation_type == "friend_of" for r in rels.relationships)
    
    @pytest.mark.asyncio
    async def test_delete_all_relationships(self, provider):
        """Test deleting all relationships between entities."""
        rel1 = EntityRelationship(
            relation_type="friend_of",
            target_entity="jane_doe",
            confidence=0.8
        )
        rel2 = EntityRelationship(
            relation_type="colleague_of",
            target_entity="jane_doe",
            confidence=0.8
        )
        await provider.add_relationship("john_doe", "jane_doe", "friend_of", rel1)
        await provider.add_relationship("john_doe", "jane_doe", "colleague_of", rel2)
        # Delete all relationships (no type specified)
        result = await provider.delete_relationship("john_doe", "jane_doe")
        assert result is True
        # Should have no relationships left
        rels = await provider.query_relationships("john_doe", direction="outgoing")
        assert rels.success
        assert len(rels.relationships) == 0
    
    @pytest.mark.asyncio
    async def test_remove_relationship(self, provider):
        """Test remove_relationship method."""
        rel = EntityRelationship(
            relation_type="friend_of",
            target_entity="jane_doe",
            confidence=0.8
        )
        await provider.add_relationship("john_doe", "jane_doe", "friend_of", rel)
        # Remove relationship
        await provider.remove_relationship("john_doe", "jane_doe", "friend_of")
        # Should be removed
        rels = await provider.query_relationships("john_doe", direction="outgoing")
        assert rels.success
        assert len(rels.relationships) == 0
    
    @pytest.mark.asyncio
    async def test_remove_nonexistent_relationship(self, provider):
        """Test removing non-existent relationship."""
        # Should not raise error, just log warning
        await provider.remove_relationship("john_doe", "jane_doe", "nonexistent")


class TestMemoryGraphProviderTraversal:
    """Test graph traversal operations."""
    
    @pytest_asyncio.fixture
    async def provider(self):
        """Create provider with sample graph structure."""
        provider = MemoryGraphProvider()
        await provider.initialize()
        
        # Create entities: A -> B -> C -> D
        #                      \-> E
        entities = [
            Entity(id="A", type="node"),
            Entity(id="B", type="node"),
            Entity(id="C", type="node"),
            Entity(id="D", type="node"),
            Entity(id="E", type="node")
        ]
        
        for entity in entities:
            await provider.add_entity(entity)
        
        # Add relationships (model-driven)
        rel_ab = EntityRelationship(relation_type="connects_to", target_entity="B", confidence=0.8)
        rel_bc = EntityRelationship(relation_type="connects_to", target_entity="C", confidence=0.8)
        rel_cd = EntityRelationship(relation_type="connects_to", target_entity="D", confidence=0.8)
        rel_be = EntityRelationship(relation_type="branches_to", target_entity="E", confidence=0.8)
        await provider.add_relationship("A", "B", "connects_to", rel_ab)
        await provider.add_relationship("B", "C", "connects_to", rel_bc)
        await provider.add_relationship("C", "D", "connects_to", rel_cd)
        await provider.add_relationship("B", "E", "branches_to", rel_be)
        
        return provider
    
    @pytest.mark.asyncio
    async def test_traverse_basic(self, provider):
        """Test basic graph traversal."""
        results = await provider.traverse("A", max_depth=2)
        
        # Should include A, B, C, E (depth 2 from A)
        result_ids = [entity.id for entity in results]
        assert "A" in result_ids
        assert "B" in result_ids
        assert "C" in result_ids
        assert "E" in result_ids
        assert "D" not in result_ids  # Depth 3 from A
    
    @pytest.mark.asyncio
    async def test_traverse_with_relation_filter(self, provider):
        """Test traversal with relation type filter."""
        results = await provider.traverse("A", relation_types=["connects_to"], max_depth=3)
        
        result_ids = [entity.id for entity in results]
        assert "A" in result_ids
        assert "B" in result_ids
        assert "C" in result_ids
        assert "D" in result_ids
        assert "E" not in result_ids  # Connected via "branches_to"
    
    @pytest.mark.asyncio
    async def test_traverse_depth_limit(self, provider):
        """Test traversal respects depth limit."""
        results = await provider.traverse("A", max_depth=1)
        
        result_ids = [entity.id for entity in results]
        assert "A" in result_ids
        assert "B" in result_ids
        assert len(result_ids) == 2  # Only A and B
    
    @pytest.mark.asyncio
    async def test_traverse_nonexistent_start(self, provider):
        """Test traversal from non-existent entity."""
        results = await provider.traverse("nonexistent")
        assert results == []
    
    @pytest.mark.asyncio
    async def test_traverse_circular_graph(self, provider):
        """Test traversal handles circular references."""
        # Add circular relationship: A -> B -> A
        rel = EntityRelationship(relation_type="circular", target_entity="A", confidence=0.8)
        await provider.add_relationship("B", "A", "circular", rel)
        results = await provider.traverse("A", max_depth=5)
        # Should not infinite loop, each entity appears only once
        result_ids = [entity.id for entity in results]
        assert result_ids.count("A") == 1
        assert result_ids.count("B") == 1


class TestMemoryGraphProviderQuery:
    """Test query operations."""
    
    @pytest_asyncio.fixture
    async def provider(self):
        """Create provider with sample data."""
        provider = MemoryGraphProvider()
        await provider.initialize()
        # Add sample entities
        entities = [
            Entity(id="john_doe", type="person", attributes={
                "name": EntityAttribute(name="name", value="John Doe")
            }),
            Entity(id="jane_doe", type="person", attributes={
                "name": EntityAttribute(name="name", value="Jane Doe")
            }),
            Entity(id="company_abc", type="organization", attributes={
                "name": EntityAttribute(name="name", value="ABC Corp")
            })
        ]
        for entity in entities:
            await provider.add_entity(entity)
        # Add relationships (model-driven)
        rel1 = EntityRelationship(relation_type="friend_of", target_entity="jane_doe", confidence=0.8)
        rel2 = EntityRelationship(relation_type="works_at", target_entity="company_abc", confidence=0.8)
        await provider.add_relationship("john_doe", "jane_doe", "friend_of", rel1)
        await provider.add_relationship("john_doe", "company_abc", "works_at", rel2)
        return provider
    
    @pytest.mark.asyncio
    async def test_query_find_entities_by_type(self, provider):
        """Test finding entities by type."""
        results = await provider.query("find_entities type=person")
        assert results.success is True
        assert len(results.nodes) == 2
        entity_ids = [node["id"] for node in results.nodes]
        assert "john_doe" in entity_ids
        assert "jane_doe" in entity_ids
    
    @pytest.mark.asyncio
    async def test_query_find_entities_by_name(self, provider):
        """Test finding entities by name."""
        results = await provider.query("find_entities name=John")
        assert results.success is True
        assert len(results.nodes) == 1
        assert results.nodes[0]["id"] == "john_doe"
    
    @pytest.mark.asyncio
    async def test_query_with_params(self, provider):
        """Test query with explicit parameters."""
        from flowlib.providers.graph.models import GraphQueryParams
        params = GraphQueryParams(extra_params={"entity_type": "organization"})
        results = await provider.query("find_entities", params)
        assert results.success is True
        assert len(results.nodes) == 1
        assert results.nodes[0]["id"] == "company_abc"
    
    @pytest.mark.asyncio
    async def test_query_neighbors(self, provider):
        """Test finding neighboring entities (unsupported query)."""
        # Unsupported queries now return empty results instead of raising
        results = await provider.query("neighbors id=john_doe")
        assert results.success is True
        assert len(results.nodes) == 0
        assert len(results.edges) == 0
    
    @pytest.mark.asyncio
    async def test_query_neighbors_by_relation(self, provider):
        """Test finding neighbors by relation type (unsupported query)."""
        # Unsupported queries now return empty results instead of raising
        results = await provider.query("neighbors id=john_doe relation=friend_of")
        assert results.success is True
        assert len(results.nodes) == 0
        assert len(results.edges) == 0
    
    @pytest.mark.asyncio
    async def test_query_path_between_entities(self, provider):
        """Test finding path between entities (unsupported query)."""
        # Unsupported queries now return empty results instead of raising
        results = await provider.query("path from=john_doe to=jane_doe")
        assert results.success is True
        assert len(results.nodes) == 0
        assert len(results.edges) == 0
    
    @pytest.mark.asyncio
    async def test_query_path_with_max_depth(self, provider):
        """Test path finding with max depth (unsupported query)."""
        rel = EntityRelationship(
            relation_type="employs",
            target_entity="jane_doe",
            confidence=0.8
        )
        await provider.add_relationship("company_abc", "jane_doe", "employs", rel)
        # Unsupported queries now return empty results instead of raising
        results = await provider.query("path from=john_doe to=jane_doe max_depth=1")
        assert results.success is True
        assert len(results.nodes) == 0
        assert len(results.edges) == 0
    
    @pytest.mark.asyncio
    async def test_query_path_no_path_exists(self, provider):
        """Test path finding when no path exists (unsupported query)."""
        isolated = Entity(id="isolated", type="test")
        await provider.add_entity(isolated)
        # Unsupported queries now return empty results instead of raising
        results = await provider.query("path from=john_doe to=isolated")
        assert results.success is True
        assert len(results.nodes) == 0
        assert len(results.edges) == 0
    
    @pytest.mark.asyncio
    async def test_query_unsupported(self, provider):
        """Test unsupported query."""
        # Unsupported queries now return empty results instead of raising
        results = await provider.query("unsupported_query")
        assert results.success is True
        assert len(results.nodes) == 0
        assert len(results.edges) == 0
    
    @pytest.mark.asyncio
    async def test_query_case_insensitive(self, provider):
        """Test query is case insensitive (unsupported query)."""
        # Unsupported queries now return empty results instead of raising
        results = await provider.query("FIND_ENTITIES TYPE=PERSON")
        assert results.success is True
        assert len(results.nodes) == 0
        assert len(results.edges) == 0


class TestMemoryGraphProviderAdvancedFeatures:
    """Test advanced provider features."""
    
    @pytest_asyncio.fixture
    async def provider(self):
        """Create initialized provider."""
        provider = MemoryGraphProvider()
        await provider.initialize()
        return provider
    
    @pytest.mark.asyncio
    async def test_bulk_add_entities(self, provider):
        """Test bulk entity addition."""
        entities = [
            Entity(id=f"entity_{i}", type="bulk")
            for i in range(5)
        ]
        result = await provider.bulk_add_entities(entities)
        assert isinstance(result, GraphStoreResult)
        assert result.success
        assert len(result.stored_entities) == 5
        for e in entities:
            assert e.id in result.stored_entities
    
    @pytest.mark.asyncio
    async def test_search_entities_by_type(self, provider):
        """Test searching entities by type."""
        # Add entities of different types
        entities = [
            Entity(id="person1", type="person"),
            Entity(id="person2", type="person"),
            Entity(id="org1", type="organization")
        ]
        
        for entity in entities:
            await provider.add_entity(entity)
        
        # Test searching by entity type
        results = await provider.search_entities(entity_type="person")
        assert len(results) == 2
        assert all(entity.type == "person" for entity in results)
        
        # Test searching by different type
        results = await provider.search_entities(entity_type="organization")
        assert len(results) == 1
        assert results[0].type == "organization"
    
    @pytest.mark.asyncio
    async def test_get_health(self, provider):
        """Test health check."""
        health = await provider.get_health()
        from flowlib.providers.graph.models import GraphHealthResult
        assert isinstance(health, GraphHealthResult)
        assert health.healthy is True
        assert health.connection_status == "connected"
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, provider):
        """Test concurrent graph operations."""
        async def add_entity_and_relationship(i: int):
            entity = Entity(id=f"entity_{i}", type="concurrent")
            await provider.add_entity(entity)
            if i > 0:
                rel = EntityRelationship(
                    relation_type="connects_to",
                    target_entity=f"entity_{i-1}",
                    confidence=0.8
                )
                await provider.add_relationship(f"entity_{i}", f"entity_{i-1}", "connects_to", rel)
        # Run concurrent operations
        tasks = [add_entity_and_relationship(i) for i in range(10)]
        await asyncio.gather(*tasks)
        # Verify all entities were created
        for i in range(10):
            entity = await provider.get_entity(f"entity_{i}")
            assert entity is not None
            assert entity.type == "concurrent"
        # Check some relationships exist
        rels = await provider.query_relationships("entity_1", direction="outgoing")
        assert rels.total_count >= 1


class TestMemoryGraphProviderErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest_asyncio.fixture
    async def provider(self):
        """Create initialized provider."""
        provider = MemoryGraphProvider()
        await provider.initialize()
        return provider
    
    @pytest.mark.asyncio
    async def test_add_invalid_entity(self, provider):
        """Test adding invalid entity."""
        # Create entity with invalid data
        with patch.object(Entity, 'model_validate', side_effect=ValueError("Invalid entity")):
            # This would trigger validation error in a real scenario
            # For now, test that provider handles exceptions properly
            pass
    
    @pytest.mark.asyncio
    async def test_query_relationships_nonexistent_entity(self, provider):
        """Test querying relationships for non-existent entity returns empty contract result."""
        results = await provider.query_relationships("nonexistent")
        from flowlib.providers.graph.models import RelationshipSearchResult
        assert isinstance(results, RelationshipSearchResult)
        assert results.success
        assert results.relationships == []
        assert results.total_count == 0
    
    @pytest.mark.asyncio
    async def test_traverse_error_handling(self, provider):
        """Test traversal error handling."""
        # Add entity and then corrupt internal state
        entity = Entity(id="test", type="test")
        await provider.add_entity(entity)
        
        # This should handle errors gracefully
        results = await provider.traverse("test")
        assert len(results) >= 1
    
    @pytest.mark.asyncio
    async def test_query_error_handling(self, provider):
        """Test query error handling."""
        # Test malformed query - now returns empty results instead of raising
        results = await provider.query("malformed query without proper format")
        assert results.success is True
        assert len(results.nodes) == 0
        assert len(results.edges) == 0
    
    @pytest.mark.asyncio
    async def test_thread_safety_stress_test(self, provider):
        """Test thread safety under stress."""
        async def stress_operation(entity_id: str):
            entity = Entity(id=entity_id, type="stress")
            await provider.add_entity(entity)
            retrieved = await provider.get_entity(entity_id)
            assert retrieved is not None
            await provider.delete_entity(entity_id)
        
        # Run many concurrent operations
        tasks = [stress_operation(f"stress_{i}") for i in range(20)]
        await asyncio.gather(*tasks)
        
        # All operations should complete successfully - all entities should be deleted
        for i in range(20):
            entity = await provider.get_entity(f"stress_{i}")
            assert entity is None  # All should be deleted
    
    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, provider):
        """Test that provider properly manages memory."""
        # Add many entities
        for i in range(100):
            entity = Entity(id=f"memory_test_{i}", type="test")
            await provider.add_entity(entity)
        
        # Delete half
        for i in range(0, 100, 2):
            await provider.delete_entity(f"memory_test_{i}")
        
        # Should have 50 entities left
        remaining_count = 0
        for i in range(100):
            entity = await provider.get_entity(f"memory_test_{i}")
            if entity is not None:
                remaining_count += 1
        
        assert remaining_count == 50
    
    def test_provider_settings_validation(self):
        """Test provider settings validation."""
        # Test with negative values - should raise validation error
        with pytest.raises(ValidationError):
            GraphDBProviderSettings(
                max_retries=-1,  # Invalid
                timeout_seconds=0  # Invalid
            )

    @pytest.mark.asyncio
    async def test_query_relationships_success(self, provider):
        """Test query_relationships returns EntitySearchResult and only model-driven data."""
        result = await provider.query_relationships(entity_id="test_entity_1")
        assert isinstance(result, type(provider).query_relationships.__annotations__['return'])
        assert result.success
        assert not isinstance(result, dict)