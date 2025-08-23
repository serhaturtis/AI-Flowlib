"""Tests for entity recall models."""

import pytest
from pydantic import ValidationError

from flowlib.agent.components.remember.entity_recall.models import (
    EntityRecallRequest,
    EntityRecallResponse,
    RecallRequest,
    RecallResponse,
    RecallStrategy,
    MemoryMatch
)


class TestEntityRecallRequest:
    """Test cases for EntityRecallRequest model."""

    def test_basic_entity_recall_request(self):
        """Test creating a basic entity recall request."""
        request = EntityRecallRequest(
            query="test query",
            strategy=RecallStrategy.ENTITY,
            entity_id="entity-123"
        )
        
        assert request.query == "test query"
        assert request.strategy == RecallStrategy.ENTITY
        assert request.entity_id == "entity-123"
        assert request.entity_type is None
        assert request.include_relationships is True
        assert request.limit == 10
        assert request.memory_types == []

    def test_entity_recall_request_with_entity_type(self):
        """Test entity recall request with entity type specified."""
        request = EntityRecallRequest(
            query="test query",
            strategy=RecallStrategy.ENTITY,
            entity_id="entity-123",
            entity_type="person"
        )
        
        assert request.entity_type == "person"
        assert request.include_relationships is True

    def test_entity_recall_request_without_relationships(self):
        """Test entity recall request without relationships."""
        request = EntityRecallRequest(
            query="test query",
            strategy=RecallStrategy.ENTITY,
            entity_id="entity-123",
            include_relationships=False
        )
        
        assert request.include_relationships is False

    def test_entity_recall_request_with_all_fields(self):
        """Test entity recall request with all fields populated."""
        request = EntityRecallRequest(
            query="comprehensive entity query",
            strategy=RecallStrategy.ENTITY,
            context="specific context",
            entity_id="entity-456",
            entity_type="organization",
            include_relationships=True,
            limit=25,
            memory_types=["episodic", "semantic"]
        )
        
        assert request.query == "comprehensive entity query"
        assert request.strategy == RecallStrategy.ENTITY
        assert request.context == "specific context"
        assert request.entity_id == "entity-456"
        assert request.entity_type == "organization"
        assert request.include_relationships is True
        assert request.limit == 25
        assert request.memory_types == ["episodic", "semantic"]

    def test_entity_recall_request_inheritance(self):
        """Test that EntityRecallRequest inherits from RecallRequest."""
        request = EntityRecallRequest(
            query="test query",
            strategy=RecallStrategy.ENTITY,
            entity_id="entity-123"
        )
        
        assert isinstance(request, RecallRequest)
        assert hasattr(request, 'query')
        assert hasattr(request, 'strategy')
        assert hasattr(request, 'entity_id')

    def test_entity_recall_request_field_validation(self):
        """Test field validation for EntityRecallRequest."""
        # Test required fields
        with pytest.raises(ValidationError) as exc_info:
            EntityRecallRequest()
        
        error_fields = [error['loc'][0] for error in exc_info.value.errors()]
        assert 'query' in error_fields
        assert 'strategy' in error_fields

    def test_entity_recall_request_strategy_validation(self):
        """Test strategy field validation."""
        with pytest.raises(ValidationError) as exc_info:
            EntityRecallRequest(
                query="test query",
                strategy="invalid_strategy",
                entity_id="entity-123"
            )
        
        assert "invalid_strategy" in str(exc_info.value)

    def test_entity_recall_request_limit_validation(self):
        """Test limit field validation."""
        # Valid limit
        request = EntityRecallRequest(
            query="test query",
            strategy=RecallStrategy.ENTITY,
            entity_id="entity-123",
            limit=100
        )
        assert request.limit == 100
        
        # Zero limit should be valid
        request = EntityRecallRequest(
            query="test query",
            strategy=RecallStrategy.ENTITY,
            entity_id="entity-123",
            limit=0
        )
        assert request.limit == 0

    def test_entity_recall_request_optional_fields(self):
        """Test optional fields have correct defaults."""
        request = EntityRecallRequest(
            query="test query",
            strategy=RecallStrategy.ENTITY,
            entity_id="entity-123"
        )
        
        assert request.context is None
        assert request.entity_type is None
        assert request.include_relationships is True
        assert request.limit == 10
        assert request.memory_types == []

    def test_entity_recall_request_serialization(self):
        """Test EntityRecallRequest serialization."""
        request = EntityRecallRequest(
            query="test query",
            strategy=RecallStrategy.ENTITY,
            entity_id="entity-123",
            entity_type="person",
            include_relationships=False
        )
        
        # Test dict conversion
        request_dict = request.model_dump()
        assert request_dict['query'] == "test query"
        assert request_dict['strategy'] == "entity"
        assert request_dict['entity_id'] == "entity-123"
        assert request_dict['entity_type'] == "person"
        assert request_dict['include_relationships'] is False
        
        # Test reconstruction from dict
        reconstructed = EntityRecallRequest(**request_dict)
        assert reconstructed == request


class TestEntityRecallResponse:
    """Test cases for EntityRecallResponse model."""

    @pytest.fixture
    def sample_memory_match(self):
        """Create a sample memory match."""
        return MemoryMatch(
            memory_id="mem-123",
            content="sample memory content",
            memory_type="episodic",
            relevance_score=0.85,
            metadata={"timestamp": "2024-01-01", "source": "conversation"}
        )

    def test_basic_entity_recall_response(self, sample_memory_match):
        """Test creating a basic entity recall response."""
        response = EntityRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.ENTITY,
            total_matches=1
        )
        
        assert len(response.matches) == 1
        assert response.strategy_used == RecallStrategy.ENTITY
        assert response.total_matches == 1
        assert response.entity_properties == {}
        assert response.relationship_count == 0

    def test_entity_recall_response_with_properties(self, sample_memory_match):
        """Test entity recall response with entity properties."""
        entity_props = {
            "name": "John Doe",
            "type": "person",
            "attributes": ["intelligent", "helpful"]
        }
        
        response = EntityRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.ENTITY,
            total_matches=1,
            entity_properties=entity_props,
            relationship_count=5
        )
        
        assert response.entity_properties == entity_props
        assert response.relationship_count == 5

    def test_entity_recall_response_inheritance(self, sample_memory_match):
        """Test that EntityRecallResponse inherits from RecallResponse."""
        response = EntityRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.ENTITY,
            total_matches=1
        )
        
        assert isinstance(response, RecallResponse)
        assert hasattr(response, 'matches')
        assert hasattr(response, 'strategy_used')
        assert hasattr(response, 'total_matches')

    def test_entity_recall_response_with_multiple_matches(self):
        """Test entity recall response with multiple memory matches."""
        matches = [
            MemoryMatch(
                memory_id=f"mem-{i}",
                content=f"memory content {i}",
                memory_type="episodic",
                relevance_score=0.8 - (i * 0.1),
                metadata={"index": i}
            )
            for i in range(3)
        ]
        
        response = EntityRecallResponse(
            matches=matches,
            strategy_used=RecallStrategy.ENTITY,
            total_matches=3,
            relationship_count=10
        )
        
        assert len(response.matches) == 3
        assert response.total_matches == 3
        assert response.relationship_count == 10

    def test_entity_recall_response_empty_matches(self):
        """Test entity recall response with no matches."""
        response = EntityRecallResponse(
            matches=[],
            strategy_used=RecallStrategy.ENTITY,
            total_matches=0
        )
        
        assert response.matches == []
        assert response.total_matches == 0
        assert response.entity_properties == {}
        assert response.relationship_count == 0

    def test_entity_recall_response_field_validation(self):
        """Test field validation for EntityRecallResponse."""
        # Test required fields
        with pytest.raises(ValidationError) as exc_info:
            EntityRecallResponse()
        
        error_fields = [error['loc'][0] for error in exc_info.value.errors()]
        assert 'matches' in error_fields
        assert 'strategy_used' in error_fields
        assert 'total_matches' in error_fields

    def test_entity_recall_response_strategy_validation(self):
        """Test strategy field validation."""
        with pytest.raises(ValidationError) as exc_info:
            EntityRecallResponse(
                matches=[],
                strategy_used="invalid_strategy",
                total_matches=0
            )
        
        assert "invalid_strategy" in str(exc_info.value)

    def test_entity_recall_response_relationship_count_validation(self):
        """Test relationship count validation."""
        # Valid relationship count
        response = EntityRecallResponse(
            matches=[],
            strategy_used=RecallStrategy.ENTITY,
            total_matches=0,
            relationship_count=100
        )
        assert response.relationship_count == 100
        
        # Zero relationship count
        response = EntityRecallResponse(
            matches=[],
            strategy_used=RecallStrategy.ENTITY,
            total_matches=0,
            relationship_count=0
        )
        assert response.relationship_count == 0

    def test_entity_recall_response_with_query_analysis(self, sample_memory_match):
        """Test entity recall response with query analysis."""
        query_analysis = {
            "entity_id": "entity-123",
            "analysis_confidence": 0.95,
            "processing_time": 0.15
        }
        
        response = EntityRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.ENTITY,
            total_matches=1,
            query_analysis=query_analysis
        )
        
        assert response.query_analysis == query_analysis

    def test_entity_recall_response_serialization(self, sample_memory_match):
        """Test EntityRecallResponse serialization."""
        entity_props = {"name": "Test Entity", "type": "concept"}
        
        response = EntityRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.ENTITY,
            total_matches=1,
            entity_properties=entity_props,
            relationship_count=3
        )
        
        # Test dict conversion
        response_dict = response.model_dump()
        assert response_dict['total_matches'] == 1
        assert response_dict['strategy_used'] == "entity"
        assert response_dict['entity_properties'] == entity_props
        assert response_dict['relationship_count'] == 3
        assert len(response_dict['matches']) == 1
        
        # Test reconstruction from dict
        reconstructed = EntityRecallResponse(**response_dict)
        assert reconstructed.entity_properties == response.entity_properties
        assert reconstructed.relationship_count == response.relationship_count

    def test_entity_recall_response_optional_fields_defaults(self, sample_memory_match):
        """Test that optional fields have correct defaults."""
        response = EntityRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.ENTITY,
            total_matches=1
        )
        
        assert response.entity_properties == {}
        assert response.relationship_count == 0
        assert response.query_analysis is None


class TestModelExports:
    """Test model exports and imports."""

    def test_model_exports(self):
        """Test that all expected models are exported."""
        from flowlib.agent.components.remember.entity_recall.models import __all__
        
        expected_exports = [
            "EntityRecallRequest",
            "EntityRecallResponse", 
            "RecallRequest",
            "RecallResponse",
            "RecallStrategy",
            "MemoryMatch"
        ]
        
        for model in expected_exports:
            assert model in __all__

    def test_model_imports(self):
        """Test that models can be imported correctly."""
        # Test direct imports
        from flowlib.agent.components.remember.entity_recall.models import (
            EntityRecallRequest,
            EntityRecallResponse,
            RecallRequest,
            RecallResponse,
            RecallStrategy,
            MemoryMatch
        )
        
        # Verify they are classes/enums
        assert isinstance(RecallStrategy.ENTITY, RecallStrategy)
        
        # Test instantiation
        request = EntityRecallRequest(
            query="test",
            strategy=RecallStrategy.ENTITY,
            entity_id="test-entity"
        )
        assert isinstance(request, EntityRecallRequest)
        assert isinstance(request, RecallRequest)

    def test_recall_strategy_values(self):
        """Test RecallStrategy enum values."""
        assert RecallStrategy.ENTITY == "entity"
        assert RecallStrategy.SEMANTIC == "semantic"
        assert RecallStrategy.TEMPORAL == "temporal"
        assert RecallStrategy.CONTEXTUAL == "contextual"