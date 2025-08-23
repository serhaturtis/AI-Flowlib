"""Tests for temporal recall flow."""

import pytest
from unittest.mock import AsyncMock
from datetime import datetime, timedelta

from flowlib.agent.components.remember.temporal_recall.flow import TemporalRecallFlow
from flowlib.agent.components.remember.models import (
    RecallRequest,
    RecallResponse,
    RecallStrategy,
    MemoryMatch
)
from flowlib.core.errors.errors import ExecutionError, ErrorContext


class TestTemporalRecallFlow:
    """Test cases for TemporalRecallFlow."""

    @pytest.fixture
    def flow(self):
        """Create a TemporalRecallFlow instance."""
        return TemporalRecallFlow()

    @pytest.fixture
    def valid_temporal_request(self):
        """Create a valid temporal recall request."""
        return RecallRequest(
            query="events from last week",
            strategy=RecallStrategy.TEMPORAL,
            context="temporal context",
            limit=10
        )

    @pytest.fixture
    def temporal_request_with_time_context(self):
        """Create a temporal recall request with time-related context."""
        return RecallRequest(
            query="what happened yesterday",
            strategy=RecallStrategy.TEMPORAL,
            context="Looking for recent events and conversations",
            limit=15
        )

    def test_flow_metadata(self, flow):
        """Test that flow has correct metadata."""
        assert hasattr(TemporalRecallFlow, '__flow_metadata__')
        metadata = TemporalRecallFlow.__flow_metadata__
        
        assert metadata['name'] == 'temporal-recall'
        assert metadata['is_infrastructure'] is True

    def test_flow_inheritance(self, flow):
        """Test that flow properly inherits from BaseRecallFlow."""
        from flowlib.agent.components.remember.flows import BaseRecallFlow
        assert isinstance(flow, BaseRecallFlow)

    def test_flow_methods(self, flow):
        """Test that flow has required methods."""
        # Check that flow has the required methods
        assert hasattr(flow, 'validate_request')  # From BaseRecallFlow
        assert hasattr(flow, 'analyze_temporal_context')
        assert hasattr(flow, 'execute_temporal_recall')
        assert hasattr(flow, 'run_pipeline')
        
        # Check that methods are callable
        assert callable(flow.validate_request)
        assert callable(flow.analyze_temporal_context)
        assert callable(flow.execute_temporal_recall)
        assert callable(flow.run_pipeline)

    @pytest.mark.asyncio
    async def test_analyze_temporal_context_basic(self, flow, valid_temporal_request):
        """Test basic temporal context analysis."""
        result = await flow.analyze_temporal_context(valid_temporal_request)
        
        assert isinstance(result, dict)
        assert "temporal_analysis" in result
        assert result["temporal_analysis"] == "placeholder"

    @pytest.mark.asyncio
    async def test_analyze_temporal_context_with_different_queries(self, flow):
        """Test temporal context analysis with different query types."""
        test_cases = [
            "events from yesterday",
            "what happened last month",
            "recent conversations",
            "activities in the morning",
            "documents created today"
        ]
        
        for query in test_cases:
            request = RecallRequest(
                query=query,
                strategy=RecallStrategy.TEMPORAL,
                limit=10
            )
            
            result = await flow.analyze_temporal_context(request)
            
            assert isinstance(result, dict)
            assert "temporal_analysis" in result
            # Current implementation returns placeholder for all queries
            assert result["temporal_analysis"] == "placeholder"

    @pytest.mark.asyncio
    async def test_analyze_temporal_context_with_context(self, flow, temporal_request_with_time_context):
        """Test temporal context analysis with additional context."""
        result = await flow.analyze_temporal_context(temporal_request_with_time_context)
        
        assert isinstance(result, dict)
        assert "temporal_analysis" in result
        assert result["temporal_analysis"] == "placeholder"

    @pytest.mark.asyncio
    async def test_analyze_temporal_context_return_type(self, flow, valid_temporal_request):
        """Test that temporal context analysis returns correct type."""
        result = await flow.analyze_temporal_context(valid_temporal_request)
        
        # Should return a dictionary
        assert isinstance(result, dict)
        
        # Should contain at least the temporal_analysis key
        assert len(result) >= 1
        assert "temporal_analysis" in result

    @pytest.mark.asyncio
    async def test_execute_temporal_recall_basic(self, flow, valid_temporal_request):
        """Test basic temporal recall execution."""
        temporal_analysis = {"temporal_analysis": "test analysis"}
        
        result = await flow.execute_temporal_recall(valid_temporal_request, temporal_analysis)
        
        assert isinstance(result, RecallResponse)
        assert result.strategy_used == RecallStrategy.TEMPORAL
        assert result.total_matches == 0  # Empty placeholder implementation
        assert result.matches == []
        assert result.query_analysis == temporal_analysis

    @pytest.mark.asyncio
    async def test_execute_temporal_recall_with_complex_analysis(self, flow, valid_temporal_request):
        """Test temporal recall execution with complex analysis data."""
        temporal_analysis = {
            "temporal_analysis": "complex analysis",
            "time_period": "last_week",
            "temporal_keywords": ["yesterday", "recent", "last"],
            "time_confidence": 0.85
        }
        
        result = await flow.execute_temporal_recall(valid_temporal_request, temporal_analysis)
        
        assert isinstance(result, RecallResponse)
        assert result.strategy_used == RecallStrategy.TEMPORAL
        assert result.total_matches == 0
        assert result.matches == []
        assert result.query_analysis == temporal_analysis

    @pytest.mark.asyncio
    async def test_execute_temporal_recall_response_structure(self, flow, valid_temporal_request):
        """Test temporal recall response structure."""
        temporal_analysis = {"temporal_analysis": "structure test"}
        
        result = await flow.execute_temporal_recall(valid_temporal_request, temporal_analysis)
        
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
    async def test_run_pipeline_success(self, flow, valid_temporal_request):
        """Test successful pipeline execution."""
        # Mock the base validation method
        flow.validate_request = AsyncMock(return_value=valid_temporal_request)
        
        result = await flow.run_pipeline(valid_temporal_request)
        
        assert isinstance(result, RecallResponse)
        assert result.strategy_used == RecallStrategy.TEMPORAL
        assert result.query_analysis["temporal_analysis"] == "placeholder"
        
        # Verify validation was called
        flow.validate_request.assert_called_once_with(valid_temporal_request)

    @pytest.mark.asyncio
    async def test_run_pipeline_validation_failure(self, flow):
        """Test pipeline failure during validation."""
        invalid_request = RecallRequest(
            query="test query",
            strategy=RecallStrategy.ENTITY,  # Wrong strategy
            entity_id=None,  # Missing entity_id for entity strategy
            limit=5
        )
        
        # Mock base validation to fail for entity strategy without entity_id
        flow.validate_request = AsyncMock(side_effect=ExecutionError(
            "Entity ID required for entity-based recall",
            ErrorContext.create(
                flow_name="temporal_recall",
                error_type="ValidationError",
                error_location="validate_request",
                component="TemporalRecallFlow",
                operation="validation"
            )
        ))
        
        with pytest.raises(ExecutionError) as exc_info:
            await flow.run_pipeline(invalid_request)
        
        assert "Entity ID required for entity-based recall" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_pipeline_with_context(self, flow, temporal_request_with_time_context):
        """Test pipeline execution with temporal context."""
        flow.validate_request = AsyncMock(return_value=temporal_request_with_time_context)
        
        result = await flow.run_pipeline(temporal_request_with_time_context)
        
        assert result.strategy_used == RecallStrategy.TEMPORAL
        assert result.query_analysis["temporal_analysis"] == "placeholder"

    @pytest.mark.asyncio
    async def test_pipeline_stage_sequence(self, flow, valid_temporal_request):
        """Test that pipeline stages execute in correct sequence."""
        # Mock stages to track call order
        call_order = []
        
        async def mock_validate_request(request):
            call_order.append("validate_request")
            return request
        
        async def mock_analyze_temporal(request):
            call_order.append("analyze_temporal_context")
            return {"temporal_analysis": "sequence test"}
        
        async def mock_execute_recall(request, analysis):
            call_order.append("execute_temporal_recall")
            return RecallResponse(
                matches=[],
                strategy_used=RecallStrategy.TEMPORAL,
                total_matches=0,
                query_analysis=analysis
            )
        
        flow.validate_request = mock_validate_request
        flow.analyze_temporal_context = mock_analyze_temporal
        flow.execute_temporal_recall = mock_execute_recall
        
        await flow.run_pipeline(valid_temporal_request)
        
        assert call_order == ["validate_request", "analyze_temporal_context", "execute_temporal_recall"]

    @pytest.mark.asyncio
    async def test_pipeline_preserves_request_data(self, flow, valid_temporal_request):
        """Test that pipeline preserves request data through stages."""
        flow.validate_request = AsyncMock(return_value=valid_temporal_request)
        
        result = await flow.run_pipeline(valid_temporal_request)
        
        # Verify original request data influences the result
        assert result.strategy_used == valid_temporal_request.strategy

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
                query=f"temporal query {i}",
                strategy=RecallStrategy.TEMPORAL,
                context=f"temporal context {i}",
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
        for result in results:
            assert result.strategy_used == RecallStrategy.TEMPORAL
            assert result.query_analysis["temporal_analysis"] == "placeholder"

    @pytest.mark.asyncio
    async def test_memory_match_structure_compatibility(self, flow, valid_temporal_request):
        """Test that response matches are compatible with MemoryMatch structure."""
        temporal_analysis = {"temporal_analysis": "compatibility test"}
        result = await flow.execute_temporal_recall(valid_temporal_request, temporal_analysis)
        
        # Even with empty matches, verify structure compatibility
        for match in result.matches:
            assert isinstance(match, MemoryMatch)
            assert hasattr(match, 'memory_id')
            assert hasattr(match, 'content')
            assert hasattr(match, 'memory_type')
            assert hasattr(match, 'relevance_score')
            assert hasattr(match, 'metadata')

    @pytest.mark.asyncio
    async def test_error_propagation(self, flow, valid_temporal_request):
        """Test that errors propagate correctly through pipeline."""
        # Mock validate_request to raise an error
        flow.validate_request = AsyncMock(side_effect=ExecutionError(
            "Base validation failed",
            ErrorContext.create(
                flow_name="temporal_recall",
                error_type="ValidationError",
                error_location="validate_request",
                component="TemporalRecallFlow",
                operation="validation"
            )
        ))
        
        with pytest.raises(ExecutionError) as exc_info:
            await flow.run_pipeline(valid_temporal_request)
        
        assert "Base validation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_temporal_analysis_integration(self, flow, valid_temporal_request):
        """Test integration between temporal analysis and recall execution."""
        # Create a realistic temporal analysis
        analysis = {
            "temporal_analysis": "placeholder",
            "additional_data": "integration test"
        }
        
        # Mock analyze_temporal_context to return custom analysis
        flow.analyze_temporal_context = AsyncMock(return_value=analysis)
        flow.validate_request = AsyncMock(return_value=valid_temporal_request)
        
        result = await flow.run_pipeline(valid_temporal_request)
        
        # Verify the analysis is properly integrated into the response
        assert result.query_analysis == analysis
        assert result.strategy_used == RecallStrategy.TEMPORAL

    @pytest.mark.asyncio
    async def test_temporal_recall_with_time_related_queries(self, flow):
        """Test temporal recall with various time-related queries."""
        time_queries = [
            "what happened this morning",
            "events from last Tuesday", 
            "recent activities",
            "yesterday's conversations",
            "files created today",
            "meetings from last week"
        ]
        
        flow.validate_request = AsyncMock(side_effect=lambda x: x)
        
        for query in time_queries:
            request = RecallRequest(
                query=query,
                strategy=RecallStrategy.TEMPORAL,
                limit=10
            )
            
            result = await flow.run_pipeline(request)
            
            assert result.strategy_used == RecallStrategy.TEMPORAL
            assert result.total_matches == 0  # Placeholder implementation
            assert "temporal_analysis" in result.query_analysis

    @pytest.mark.asyncio
    async def test_temporal_analysis_method_isolation(self, flow, valid_temporal_request):
        """Test that temporal analysis method can be called independently."""
        # Test that the method works independently of the pipeline
        result = await flow.analyze_temporal_context(valid_temporal_request)
        
        assert isinstance(result, dict)
        assert "temporal_analysis" in result
        
        # Should be consistent when called multiple times
        result2 = await flow.analyze_temporal_context(valid_temporal_request)
        assert result == result2

    @pytest.mark.asyncio
    async def test_execute_temporal_recall_method_isolation(self, flow, valid_temporal_request):
        """Test that execute temporal recall method can be called independently."""
        analysis = {"temporal_analysis": "isolated test"}
        
        # Test that the method works independently of the pipeline
        result = await flow.execute_temporal_recall(valid_temporal_request, analysis)
        
        assert isinstance(result, RecallResponse)
        assert result.strategy_used == RecallStrategy.TEMPORAL
        assert result.query_analysis == analysis

    @pytest.mark.asyncio
    async def test_flow_constructor(self, flow):
        """Test that flow constructor initializes correctly."""
        # Test that flow can be constructed
        new_flow = TemporalRecallFlow()
        
        assert isinstance(new_flow, TemporalRecallFlow)
        assert hasattr(new_flow, 'run_pipeline')
        assert hasattr(new_flow, 'analyze_temporal_context')
        assert hasattr(new_flow, 'execute_temporal_recall')

    @pytest.mark.asyncio
    async def test_placeholder_implementation_behavior(self, flow, valid_temporal_request):
        """Test the current placeholder implementation behavior."""
        # Since the current implementation is a placeholder, test its consistent behavior
        
        # Multiple calls should return the same placeholder result
        analysis1 = await flow.analyze_temporal_context(valid_temporal_request)
        analysis2 = await flow.analyze_temporal_context(valid_temporal_request)
        
        assert analysis1 == analysis2
        assert analysis1["temporal_analysis"] == "placeholder"
        
        # Recall execution should always return empty results with placeholder
        recall_result = await flow.execute_temporal_recall(valid_temporal_request, analysis1)
        
        assert recall_result.matches == []
        assert recall_result.total_matches == 0
        assert recall_result.strategy_used == RecallStrategy.TEMPORAL