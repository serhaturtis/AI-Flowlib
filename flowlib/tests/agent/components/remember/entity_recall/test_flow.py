"""Tests for entity recall flow."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import ValidationError

from flowlib.agent.components.remember.entity_recall.flow import EntityRecallFlow
from flowlib.agent.components.remember.models import (
    RecallRequest,
    RecallResponse,
    RecallStrategy,
    MemoryMatch
)
from flowlib.agent.core.errors import ExecutionError
from flowlib.core.context.context import Context


class TestEntityRecallFlow:
    """Test cases for EntityRecallFlow."""

    @pytest.fixture
    def flow(self):
        """Create an EntityRecallFlow instance."""
        return EntityRecallFlow()

    @pytest.fixture
    def valid_entity_request(self):
        """Create a valid entity recall request."""
        return RecallRequest(
            query="information about entity",
            strategy=RecallStrategy.ENTITY,
            entity_id="entity-123",
            limit=10
        )

    @pytest.fixture
    def invalid_entity_request(self):
        """Create an invalid entity recall request without entity_id."""
        return RecallRequest(
            query="information about entity",
            strategy=RecallStrategy.ENTITY,
            limit=10
        )

    def test_flow_metadata(self, flow):
        """Test that flow has correct metadata."""
        assert hasattr(EntityRecallFlow, '__flow_metadata__')
        metadata = EntityRecallFlow.__flow_metadata__
        
        assert metadata['name'] == 'entity-recall'
        assert metadata['is_infrastructure'] is True

    def test_flow_inheritance(self, flow):
        """Test that flow properly inherits from BaseRecallFlow."""
        from flowlib.agent.components.remember.flows import BaseRecallFlow
        assert isinstance(flow, BaseRecallFlow)

    def test_flow_methods(self, flow):
        """Test that flow has required methods."""
        # Check that flow has the required methods
        assert hasattr(flow, 'validate_request')  # From BaseRecallFlow
        assert hasattr(flow, 'validate_entity')
        assert hasattr(flow, 'recall_entity_knowledge')
        assert hasattr(flow, 'run_pipeline')
        
        # Check that methods are callable
        assert callable(flow.validate_request)
        assert callable(flow.validate_entity)
        assert callable(flow.recall_entity_knowledge)
        assert callable(flow.run_pipeline)

    @pytest.mark.asyncio
    async def test_validate_entity_with_valid_request(self, flow, valid_entity_request):
        """Test entity validation with valid entity ID."""
        result = await flow.validate_entity(valid_entity_request)
        
        assert result == valid_entity_request
        assert result.entity_id == "entity-123"

    @pytest.mark.asyncio
    async def test_validate_entity_without_entity_id(self, flow, invalid_entity_request):
        """Test entity validation fails without entity ID."""
        with pytest.raises(ExecutionError) as exc_info:
            await flow.validate_entity(invalid_entity_request)
        
        assert "Entity ID is required for entity recall" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_validate_entity_with_none_entity_id(self, flow):
        """Test entity validation fails with None entity ID."""
        request = RecallRequest(
            query="test query",
            strategy=RecallStrategy.ENTITY,
            entity_id=None,
            limit=5
        )
        
        with pytest.raises(ExecutionError) as exc_info:
            await flow.validate_entity(request)
        
        assert "Entity ID is required for entity recall" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_validate_entity_with_empty_entity_id(self, flow):
        """Test entity validation fails with empty entity ID."""
        request = RecallRequest(
            query="test query",
            strategy=RecallStrategy.ENTITY,
            entity_id="",
            limit=5
        )
        
        with pytest.raises(ExecutionError) as exc_info:
            await flow.validate_entity(request)
        
        assert "Entity ID is required for entity recall" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_recall_entity_knowledge_basic(self, flow, valid_entity_request):
        """Test basic entity knowledge recall."""
        result = await flow.recall_entity_knowledge(valid_entity_request)
        
        assert isinstance(result, RecallResponse)
        assert result.strategy_used == RecallStrategy.ENTITY
        assert result.total_matches == 0  # Empty placeholder implementation
        assert result.matches == []
        assert result.query_analysis["entity_id"] == "entity-123"

    @pytest.mark.asyncio
    async def test_recall_entity_knowledge_response_structure(self, flow, valid_entity_request):
        """Test entity knowledge recall response structure."""
        result = await flow.recall_entity_knowledge(valid_entity_request)
        
        # Verify response structure
        assert hasattr(result, 'matches')
        assert hasattr(result, 'strategy_used')
        assert hasattr(result, 'total_matches')
        assert hasattr(result, 'query_analysis')
        
        # Verify types
        assert isinstance(result.matches, list)
        assert isinstance(result.strategy_used, RecallStrategy)
        assert isinstance(result.total_matches, int)
        assert isinstance(result.query_analysis, dict)

    @pytest.mark.asyncio
    async def test_run_pipeline_success(self, flow, valid_entity_request):
        """Test successful pipeline execution."""
        # Mock the base validation method
        flow.validate_request = AsyncMock(return_value=valid_entity_request)
        
        result = await flow.run_pipeline(valid_entity_request)
        
        assert isinstance(result, RecallResponse)
        assert result.strategy_used == RecallStrategy.ENTITY
        assert result.query_analysis["entity_id"] == "entity-123"
        
        # Verify validation was called
        flow.validate_request.assert_called_once_with(valid_entity_request)

    @pytest.mark.asyncio
    async def test_run_pipeline_validation_failure(self, flow, invalid_entity_request):
        """Test pipeline failure during validation."""
        # Mock base validation to pass, so we test entity validation
        flow.validate_request = AsyncMock(return_value=invalid_entity_request)
        
        with pytest.raises(ExecutionError) as exc_info:
            await flow.run_pipeline(invalid_entity_request)
        
        assert "Entity ID is required for entity recall" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_pipeline_with_context(self, flow):
        """Test pipeline execution with additional context."""
        request = RecallRequest(
            query="detailed entity information",
            strategy=RecallStrategy.ENTITY,
            entity_id="entity-456",
            context="specific context for recall",
            limit=15
        )
        
        flow.validate_request = AsyncMock(return_value=request)
        
        result = await flow.run_pipeline(request)
        
        assert result.query_analysis["entity_id"] == "entity-456"
        assert result.strategy_used == RecallStrategy.ENTITY

    @pytest.mark.asyncio
    async def test_pipeline_stage_sequence(self, flow, valid_entity_request):
        """Test that pipeline stages execute in correct sequence."""
        # Mock stages to track call order
        call_order = []
        
        async def mock_validate_request(request):
            call_order.append("validate_request")
            return request
        
        async def mock_validate_entity(request):
            call_order.append("validate_entity")
            return request
        
        async def mock_recall_knowledge(request):
            call_order.append("recall_entity_knowledge")
            return RecallResponse(
                matches=[],
                strategy_used=RecallStrategy.ENTITY,
                total_matches=0,
                query_analysis={"entity_id": request.entity_id}
            )
        
        flow.validate_request = mock_validate_request
        flow.validate_entity = mock_validate_entity
        flow.recall_entity_knowledge = mock_recall_knowledge
        
        await flow.run_pipeline(valid_entity_request)
        
        assert call_order == ["validate_request", "validate_entity", "recall_entity_knowledge"]

    @pytest.mark.asyncio
    async def test_pipeline_preserves_request_data(self, flow, valid_entity_request):
        """Test that pipeline preserves request data through stages."""
        flow.validate_request = AsyncMock(return_value=valid_entity_request)
        
        result = await flow.run_pipeline(valid_entity_request)
        
        # Verify original request data is preserved in response
        assert result.query_analysis["entity_id"] == valid_entity_request.entity_id
        assert result.strategy_used == valid_entity_request.strategy

    def test_flow_immutability(self, flow):
        """Test that flow instances are stateless."""
        # Flows should not maintain state between calls
        assert not hasattr(flow, '_state')
        assert not hasattr(flow, '_cache')
        assert not hasattr(flow, '_memory')

    @pytest.mark.asyncio
    async def test_concurrent_pipeline_execution(self, flow):
        """Test that multiple pipeline executions can run concurrently."""
        import asyncio
        
        requests = [
            RecallRequest(
                query=f"query {i}",
                strategy=RecallStrategy.ENTITY,
                entity_id=f"entity-{i}",
                limit=5
            )
            for i in range(3)
        ]
        
        flow.validate_request = AsyncMock(side_effect=lambda x: x)
        
        # Execute pipelines concurrently
        results = await asyncio.gather(*[
            flow.run_pipeline(req) for req in requests
        ])
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.query_analysis["entity_id"] == f"entity-{i}"
            assert result.strategy_used == RecallStrategy.ENTITY

    @pytest.mark.asyncio
    async def test_memory_match_structure_compatibility(self, flow, valid_entity_request):
        """Test that response matches are compatible with MemoryMatch structure."""
        result = await flow.recall_entity_knowledge(valid_entity_request)
        
        # Even with empty matches, verify structure compatibility
        for match in result.matches:
            assert isinstance(match, MemoryMatch)
            assert hasattr(match, 'memory_id')
            assert hasattr(match, 'content')
            assert hasattr(match, 'memory_type')
            assert hasattr(match, 'relevance_score')
            assert hasattr(match, 'metadata')

    @pytest.mark.asyncio
    async def test_error_propagation(self, flow, valid_entity_request):
        """Test that errors propagate correctly through pipeline."""
        # Mock validate_request to raise an error
        from flowlib.core.errors.errors import ErrorContext
        error_context = ErrorContext.create(
            flow_name="entity-recall",
            error_type="ValidationError", 
            error_location="test_error_propagation",
            component="EntityRecallFlow",
            operation="validate_request"
        )
        flow.validate_request = AsyncMock(side_effect=ExecutionError("Base validation failed", context=error_context))
        
        with pytest.raises(ExecutionError) as exc_info:
            await flow.run_pipeline(valid_entity_request)
        
        assert "Base validation failed" in str(exc_info.value)