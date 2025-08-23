"""Tests for temporal recall models."""

import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError

from flowlib.agent.components.remember.temporal_recall.models import (
    TemporalRecallRequest,
    TemporalRecallResponse,
    RecallRequest,
    RecallResponse,
    RecallStrategy,
    MemoryMatch
)


class TestTemporalRecallRequest:
    """Test cases for TemporalRecallRequest model."""

    def test_basic_temporal_recall_request(self):
        """Test creating a basic temporal recall request."""
        request = TemporalRecallRequest(
            query="temporal query",
            strategy=RecallStrategy.TEMPORAL
        )
        
        assert request.query == "temporal query"
        assert request.strategy == RecallStrategy.TEMPORAL
        assert request.time_range_start is None  # Default value
        assert request.time_range_end is None    # Default value
        assert request.temporal_relation is None # Default value
        assert request.limit == 10  # From parent class
        assert request.memory_types == []  # From parent class

    def test_temporal_recall_request_with_time_range(self):
        """Test temporal recall request with time range specified."""
        start_time = datetime(2024, 1, 1, 9, 0, 0)
        end_time = datetime(2024, 1, 1, 17, 0, 0)
        
        request = TemporalRecallRequest(
            query="events during work hours",
            strategy=RecallStrategy.TEMPORAL,
            time_range_start=start_time,
            time_range_end=end_time
        )
        
        assert request.time_range_start == start_time
        assert request.time_range_end == end_time

    def test_temporal_recall_request_with_temporal_relation(self):
        """Test temporal recall request with temporal relation."""
        request = TemporalRecallRequest(
            query="events before meeting",
            strategy=RecallStrategy.TEMPORAL,
            temporal_relation="before"
        )
        
        assert request.temporal_relation == "before"

    def test_temporal_recall_request_with_all_fields(self):
        """Test temporal recall request with all fields populated."""
        start_time = datetime(2024, 1, 15, 8, 0, 0)
        end_time = datetime(2024, 1, 15, 18, 0, 0)
        
        request = TemporalRecallRequest(
            query="comprehensive temporal query",
            strategy=RecallStrategy.TEMPORAL,
            context="temporal context",
            entity_id=None,
            time_range_start=start_time,
            time_range_end=end_time,
            temporal_relation="during",
            limit=25,
            memory_types=["episodic", "temporal"]
        )
        
        assert request.query == "comprehensive temporal query"
        assert request.strategy == RecallStrategy.TEMPORAL
        assert request.context == "temporal context"
        assert request.time_range_start == start_time
        assert request.time_range_end == end_time
        assert request.temporal_relation == "during"
        assert request.limit == 25
        assert request.memory_types == ["episodic", "temporal"]

    def test_temporal_recall_request_inheritance(self):
        """Test that TemporalRecallRequest inherits from RecallRequest."""
        request = TemporalRecallRequest(
            query="test query",
            strategy=RecallStrategy.TEMPORAL
        )
        
        assert isinstance(request, RecallRequest)
        assert hasattr(request, 'query')
        assert hasattr(request, 'strategy')
        assert hasattr(request, 'time_range_start')

    def test_temporal_recall_request_field_validation(self):
        """Test field validation for TemporalRecallRequest."""
        # Test required fields from parent
        with pytest.raises(ValidationError) as exc_info:
            TemporalRecallRequest()
        
        error_fields = [error['loc'][0] for error in exc_info.value.errors()]
        assert 'query' in error_fields
        assert 'strategy' in error_fields

    def test_temporal_recall_request_strategy_validation(self):
        """Test strategy field validation."""
        with pytest.raises(ValidationError) as exc_info:
            TemporalRecallRequest(
                query="test query",
                strategy="invalid_strategy"
            )
        
        assert "invalid_strategy" in str(exc_info.value)

    def test_temporal_recall_request_datetime_validation(self):
        """Test datetime field validation."""
        # Valid datetime objects
        now = datetime.now()
        later = now + timedelta(hours=1)
        
        request = TemporalRecallRequest(
            query="test query",
            strategy=RecallStrategy.TEMPORAL,
            time_range_start=now,
            time_range_end=later
        )
        
        assert request.time_range_start == now
        assert request.time_range_end == later

    def test_temporal_recall_request_temporal_relations(self):
        """Test various temporal relation values."""
        relations = ["before", "after", "during", "overlaps", "adjacent"]
        
        for relation in relations:
            request = TemporalRecallRequest(
                query="test query",
                strategy=RecallStrategy.TEMPORAL,
                temporal_relation=relation
            )
            assert request.temporal_relation == relation

    def test_temporal_recall_request_optional_fields_defaults(self):
        """Test that optional fields have correct defaults."""
        request = TemporalRecallRequest(
            query="test query",
            strategy=RecallStrategy.TEMPORAL
        )
        
        assert request.context is None
        assert request.entity_id is None
        assert request.time_range_start is None
        assert request.time_range_end is None
        assert request.temporal_relation is None
        assert request.limit == 10
        assert request.memory_types == []

    def test_temporal_recall_request_serialization(self):
        """Test TemporalRecallRequest serialization."""
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 15, 0, 0)
        
        request = TemporalRecallRequest(
            query="temporal test query",
            strategy=RecallStrategy.TEMPORAL,
            time_range_start=start_time,
            time_range_end=end_time,
            temporal_relation="during"
        )
        
        # Test dict conversion
        request_dict = request.model_dump()
        assert request_dict['query'] == "temporal test query"
        assert request_dict['strategy'] == "temporal"
        assert request_dict['temporal_relation'] == "during"
        
        # Test reconstruction from dict
        reconstructed = TemporalRecallRequest(**request_dict)
        assert reconstructed.time_range_start == request.time_range_start
        assert reconstructed.time_range_end == request.time_range_end
        assert reconstructed.temporal_relation == request.temporal_relation

    def test_temporal_recall_request_time_range_logic(self):
        """Test logical time range scenarios."""
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        
        # Test same start and end time
        request = TemporalRecallRequest(
            query="instant query",
            strategy=RecallStrategy.TEMPORAL,
            time_range_start=base_time,
            time_range_end=base_time
        )
        assert request.time_range_start == request.time_range_end
        
        # Test start time only
        request = TemporalRecallRequest(
            query="from time query",
            strategy=RecallStrategy.TEMPORAL,
            time_range_start=base_time
        )
        assert request.time_range_start == base_time
        assert request.time_range_end is None
        
        # Test end time only
        request = TemporalRecallRequest(
            query="until time query",
            strategy=RecallStrategy.TEMPORAL,
            time_range_end=base_time
        )
        assert request.time_range_start is None
        assert request.time_range_end == base_time


class TestTemporalRecallResponse:
    """Test cases for TemporalRecallResponse model."""

    @pytest.fixture
    def sample_memory_match(self):
        """Create a sample memory match."""
        return MemoryMatch(
            memory_id="mem-123",
            content="sample temporal memory content",
            memory_type="temporal",
            relevance_score=0.85,
            metadata={"timestamp": "2024-01-01T12:00:00", "event_type": "meeting"}
        )

    def test_basic_temporal_recall_response(self, sample_memory_match):
        """Test creating a basic temporal recall response."""
        response = TemporalRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.TEMPORAL,
            total_matches=1
        )
        
        assert len(response.matches) == 1
        assert response.strategy_used == RecallStrategy.TEMPORAL
        assert response.total_matches == 1
        assert response.timeline == []  # Default value
        assert response.temporal_clusters == []  # Default value

    def test_temporal_recall_response_with_timeline(self, sample_memory_match):
        """Test temporal recall response with timeline data."""
        timeline = [
            {"time": "2024-01-01T09:00:00", "event": "start work", "memory_id": "mem-1"},
            {"time": "2024-01-01T12:00:00", "event": "lunch meeting", "memory_id": "mem-2"},
            {"time": "2024-01-01T17:00:00", "event": "end work", "memory_id": "mem-3"}
        ]
        
        response = TemporalRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.TEMPORAL,
            total_matches=1,
            timeline=timeline
        )
        
        assert response.timeline == timeline
        assert len(response.timeline) == 3

    def test_temporal_recall_response_with_temporal_clusters(self, sample_memory_match):
        """Test temporal recall response with temporal clusters."""
        clusters = [
            {"cluster_id": "morning", "time_range": "09:00-12:00", "event_count": 5},
            {"cluster_id": "afternoon", "time_range": "13:00-17:00", "event_count": 8}
        ]
        
        response = TemporalRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.TEMPORAL,
            total_matches=1,
            temporal_clusters=clusters
        )
        
        assert response.temporal_clusters == clusters
        assert len(response.temporal_clusters) == 2

    def test_temporal_recall_response_inheritance(self, sample_memory_match):
        """Test that TemporalRecallResponse inherits from RecallResponse."""
        response = TemporalRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.TEMPORAL,
            total_matches=1
        )
        
        assert isinstance(response, RecallResponse)
        assert hasattr(response, 'matches')
        assert hasattr(response, 'strategy_used')
        assert hasattr(response, 'total_matches')
        assert hasattr(response, 'timeline')

    def test_temporal_recall_response_with_multiple_matches(self):
        """Test temporal recall response with multiple memory matches."""
        matches = [
            MemoryMatch(
                memory_id=f"mem-{i}",
                content=f"temporal memory content {i}",
                memory_type="temporal",
                relevance_score=0.9 - (i * 0.1),
                metadata={"timestamp": f"2024-01-0{i+1}T12:00:00"}
            )
            for i in range(3)
        ]
        
        timeline = [
            {"time": f"2024-01-0{i+1}T12:00:00", "memory_id": f"mem-{i}"}
            for i in range(3)
        ]
        
        response = TemporalRecallResponse(
            matches=matches,
            strategy_used=RecallStrategy.TEMPORAL,
            total_matches=3,
            timeline=timeline
        )
        
        assert len(response.matches) == 3
        assert response.total_matches == 3
        assert len(response.timeline) == 3

    def test_temporal_recall_response_empty_matches(self):
        """Test temporal recall response with no matches."""
        response = TemporalRecallResponse(
            matches=[],
            strategy_used=RecallStrategy.TEMPORAL,
            total_matches=0
        )
        
        assert response.matches == []
        assert response.total_matches == 0
        assert response.timeline == []
        assert response.temporal_clusters == []

    def test_temporal_recall_response_field_validation(self):
        """Test field validation for TemporalRecallResponse."""
        # Test required fields
        with pytest.raises(ValidationError) as exc_info:
            TemporalRecallResponse()
        
        error_fields = [error['loc'][0] for error in exc_info.value.errors()]
        assert 'matches' in error_fields
        assert 'strategy_used' in error_fields
        assert 'total_matches' in error_fields

    def test_temporal_recall_response_strategy_validation(self):
        """Test strategy field validation."""
        with pytest.raises(ValidationError) as exc_info:
            TemporalRecallResponse(
                matches=[],
                strategy_used="invalid_strategy",
                total_matches=0
            )
        
        assert "invalid_strategy" in str(exc_info.value)

    def test_temporal_recall_response_with_query_analysis(self, sample_memory_match):
        """Test temporal recall response with query analysis."""
        query_analysis = {
            "temporal_analysis": "placeholder",
            "time_period": "last_week",
            "confidence": 0.82
        }
        
        response = TemporalRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.TEMPORAL,
            total_matches=1,
            query_analysis=query_analysis
        )
        
        assert response.query_analysis == query_analysis

    def test_temporal_recall_response_serialization(self, sample_memory_match):
        """Test TemporalRecallResponse serialization."""
        timeline = [{"time": "2024-01-01T12:00:00", "event": "test"}]
        clusters = [{"cluster": "test_cluster", "size": 1}]
        
        response = TemporalRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.TEMPORAL,
            total_matches=1,
            timeline=timeline,
            temporal_clusters=clusters
        )
        
        # Test dict conversion
        response_dict = response.model_dump()
        assert response_dict['total_matches'] == 1
        assert response_dict['strategy_used'] == "temporal"
        assert response_dict['timeline'] == timeline
        assert response_dict['temporal_clusters'] == clusters
        assert len(response_dict['matches']) == 1
        
        # Test reconstruction from dict
        reconstructed = TemporalRecallResponse(**response_dict)
        assert reconstructed.timeline == response.timeline
        assert reconstructed.temporal_clusters == response.temporal_clusters

    def test_temporal_recall_response_optional_fields_defaults(self, sample_memory_match):
        """Test that optional fields have correct defaults."""
        response = TemporalRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.TEMPORAL,
            total_matches=1
        )
        
        assert response.timeline == []
        assert response.temporal_clusters == []
        assert response.query_analysis is None

    def test_temporal_recall_response_complex_timeline(self, sample_memory_match):
        """Test temporal recall response with complex timeline."""
        timeline = [
            {
                "time": "2024-01-01T09:00:00",
                "event": "project_start",
                "memory_id": "mem-1",
                "duration": "2h",
                "participants": ["alice", "bob"]
            },
            {
                "time": "2024-01-01T14:00:00", 
                "event": "review_meeting",
                "memory_id": "mem-2",
                "duration": "1h",
                "participants": ["alice", "carol", "david"]
            }
        ]
        
        response = TemporalRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.TEMPORAL,
            total_matches=1,
            timeline=timeline
        )
        
        assert len(response.timeline) == 2
        assert response.timeline[0]["participants"] == ["alice", "bob"]
        assert response.timeline[1]["duration"] == "1h"

    def test_temporal_recall_response_complex_clusters(self, sample_memory_match):
        """Test temporal recall response with complex temporal clusters."""
        clusters = [
            {
                "cluster_id": "work_hours",
                "time_range": {"start": "09:00", "end": "17:00"},
                "event_count": 12,
                "avg_relevance": 0.85,
                "dominant_activity": "meetings"
            },
            {
                "cluster_id": "evening",
                "time_range": {"start": "18:00", "end": "22:00"},
                "event_count": 5,
                "avg_relevance": 0.65,
                "dominant_activity": "personal"
            }
        ]
        
        response = TemporalRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.TEMPORAL,
            total_matches=1,
            temporal_clusters=clusters
        )
        
        assert len(response.temporal_clusters) == 2
        assert response.temporal_clusters[0]["dominant_activity"] == "meetings"
        assert response.temporal_clusters[1]["event_count"] == 5


class TestModelExports:
    """Test model exports and imports."""

    def test_model_exports(self):
        """Test that all expected models are exported."""
        from flowlib.agent.components.remember.temporal_recall.models import __all__
        
        expected_exports = [
            "TemporalRecallRequest",
            "TemporalRecallResponse",
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
        from flowlib.agent.components.remember.temporal_recall.models import (
            TemporalRecallRequest,
            TemporalRecallResponse,
            RecallRequest,
            RecallResponse,
            RecallStrategy,
            MemoryMatch
        )
        
        # Verify they are classes/enums
        assert isinstance(RecallStrategy.TEMPORAL, RecallStrategy)
        
        # Test instantiation
        request = TemporalRecallRequest(
            query="test",
            strategy=RecallStrategy.TEMPORAL
        )
        assert isinstance(request, TemporalRecallRequest)
        assert isinstance(request, RecallRequest)

    def test_recall_strategy_values(self):
        """Test RecallStrategy enum values."""
        assert RecallStrategy.TEMPORAL == "temporal"
        assert RecallStrategy.ENTITY == "entity"
        assert RecallStrategy.SEMANTIC == "semantic"
        assert RecallStrategy.CONTEXTUAL == "contextual"

    def test_temporal_models_real_world_scenario(self):
        """Test temporal models with real-world scenario data."""
        # Create a realistic temporal recall scenario
        start_time = datetime(2024, 1, 15, 9, 0, 0)
        end_time = datetime(2024, 1, 15, 17, 0, 0)
        
        request = TemporalRecallRequest(
            query="What meetings did I have during work hours on January 15th?",
            strategy=RecallStrategy.TEMPORAL,
            context="Looking for calendar events and meeting notes from that day",
            time_range_start=start_time,
            time_range_end=end_time,
            temporal_relation="during",
            limit=20,
            memory_types=["calendar", "notes", "conversations"]
        )
        
        # Simulate memory matches
        matches = [
            MemoryMatch(
                memory_id="meeting-1",
                content="Daily standup with development team",
                memory_type="calendar",
                relevance_score=0.95,
                metadata={"start": "09:30", "duration": "30min", "participants": 5}
            ),
            MemoryMatch(
                memory_id="meeting-2", 
                content="Product review with stakeholders",
                memory_type="calendar",
                relevance_score=0.88,
                metadata={"start": "14:00", "duration": "90min", "participants": 8}
            )
        ]
        
        # Timeline of events
        timeline = [
            {"time": "09:30", "event": "Daily standup", "memory_id": "meeting-1"},
            {"time": "14:00", "event": "Product review", "memory_id": "meeting-2"}
        ]
        
        # Temporal clusters
        clusters = [
            {"cluster_id": "morning", "time_range": "09:00-12:00", "meeting_count": 1},
            {"cluster_id": "afternoon", "time_range": "13:00-17:00", "meeting_count": 1}
        ]
        
        response = TemporalRecallResponse(
            matches=matches,
            strategy_used=RecallStrategy.TEMPORAL,
            total_matches=2,
            timeline=timeline,
            temporal_clusters=clusters,
            query_analysis={
                "temporal_analysis": "work day meeting query",
                "time_period": "single_day",
                "detected_time_markers": ["January 15th", "work hours"]
            }
        )
        
        # Verify the complex scenario
        assert len(request.memory_types) == 3
        assert request.temporal_relation == "during"
        assert len(response.matches) == 2
        assert len(response.timeline) == 2
        assert len(response.temporal_clusters) == 2
        assert response.query_analysis["time_period"] == "single_day"