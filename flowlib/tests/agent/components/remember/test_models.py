"""Tests for remember models."""

import pytest
from pydantic import ValidationError
from flowlib.agent.components.remember.models import (
    RecallStrategy,
    ContextAnalysis,
    RecallRequest,
    MemoryMatch,
    RecallResponse
)


class TestRecallStrategy:
    """Test RecallStrategy enum."""
    
    def test_recall_strategy_values(self):
        """Test that all expected strategies exist."""
        assert RecallStrategy.CONTEXTUAL == "contextual"
        assert RecallStrategy.ENTITY == "entity"
        assert RecallStrategy.TEMPORAL == "temporal"
        assert RecallStrategy.SEMANTIC == "semantic"
    
    def test_recall_strategy_is_enum(self):
        """Test that RecallStrategy is a proper enum."""
        strategies = list(RecallStrategy)
        assert len(strategies) == 4
        assert RecallStrategy.CONTEXTUAL in strategies


class TestContextAnalysis:
    """Test ContextAnalysis model."""
    
    def test_context_analysis_minimal(self):
        """Test creating ContextAnalysis with required fields."""
        analysis = ContextAnalysis(
            analysis="This is an analysis",
            recommended_strategy=RecallStrategy.CONTEXTUAL,
            confidence=0.85
        )
        
        assert analysis.analysis == "This is an analysis"
        assert analysis.recommended_strategy == RecallStrategy.CONTEXTUAL
        assert analysis.confidence == 0.85
        assert analysis.key_concepts == []  # Default empty list
    
    def test_context_analysis_full(self):
        """Test creating ContextAnalysis with all fields."""
        analysis = ContextAnalysis(
            analysis="Detailed context analysis",
            recommended_strategy=RecallStrategy.ENTITY,
            key_concepts=["person", "meeting", "project"],
            confidence=0.92
        )
        
        assert len(analysis.key_concepts) == 3
        assert "meeting" in analysis.key_concepts
    
    def test_context_analysis_validation(self):
        """Test ContextAnalysis validation."""
        # Missing required fields
        with pytest.raises(ValidationError) as exc_info:
            ContextAnalysis(analysis="Test")
        assert "recommended_strategy" in str(exc_info.value)
        assert "confidence" in str(exc_info.value)
        
        # Invalid confidence value (should be float)
        with pytest.raises(ValidationError):
            ContextAnalysis(
                analysis="Test",
                recommended_strategy=RecallStrategy.SEMANTIC,
                confidence="high"  # Should be float
            )
    
    def test_context_analysis_confidence_range(self):
        """Test confidence value range."""
        # Valid range
        analysis = ContextAnalysis(
            analysis="Test",
            recommended_strategy=RecallStrategy.TEMPORAL,
            confidence=0.0
        )
        assert analysis.confidence == 0.0
        
        analysis = ContextAnalysis(
            analysis="Test",
            recommended_strategy=RecallStrategy.TEMPORAL,
            confidence=1.0
        )
        assert analysis.confidence == 1.0


class TestRecallRequest:
    """Test RecallRequest model."""
    
    def test_recall_request_minimal(self):
        """Test creating RecallRequest with minimal fields."""
        request = RecallRequest(
            query="Find information about the project",
            strategy=RecallStrategy.SEMANTIC
        )
        
        assert request.query == "Find information about the project"
        assert request.strategy == RecallStrategy.SEMANTIC
        assert request.context is None
        assert request.entity_id is None
        assert request.limit == 10  # Default
        assert request.memory_types == []  # Default
    
    def test_recall_request_full(self):
        """Test creating RecallRequest with all fields."""
        request = RecallRequest(
            query="What did John say about the budget?",
            strategy=RecallStrategy.ENTITY,
            context="During yesterday's meeting",
            entity_id="entity_john_doe",
            limit=5,
            memory_types=["conversation", "meeting"]
        )
        
        assert request.context == "During yesterday's meeting"
        assert request.entity_id == "entity_john_doe"
        assert request.limit == 5
        assert len(request.memory_types) == 2
        assert "meeting" in request.memory_types
    
    def test_recall_request_validation(self):
        """Test RecallRequest validation."""
        # Missing required fields
        with pytest.raises(ValidationError) as exc_info:
            RecallRequest(query="Test")
        assert "strategy" in str(exc_info.value)
        
        # Invalid strategy
        with pytest.raises(ValidationError):
            RecallRequest(
                query="Test",
                strategy="invalid_strategy"
            )


class TestMemoryMatch:
    """Test MemoryMatch model."""
    
    def test_memory_match_minimal(self):
        """Test creating MemoryMatch with required fields."""
        match = MemoryMatch(
            memory_id="mem_123",
            content="This is a memory about the project",
            memory_type="conversation",
            relevance_score=0.87
        )
        
        assert match.memory_id == "mem_123"
        assert match.content == "This is a memory about the project"
        assert match.memory_type == "conversation"
        assert match.relevance_score == 0.87
        assert match.metadata == {}  # Default empty dict
    
    def test_memory_match_with_metadata(self):
        """Test creating MemoryMatch with metadata."""
        match = MemoryMatch(
            memory_id="mem_456",
            content="Budget discussion",
            memory_type="meeting",
            relevance_score=0.95,
            metadata={
                "timestamp": "2024-01-15T10:00:00",
                "participants": ["John", "Jane"],
                "tags": ["budget", "Q1"]
            }
        )
        
        assert match.metadata["timestamp"] == "2024-01-15T10:00:00"
        assert len(match.metadata["participants"]) == 2
        assert "budget" in match.metadata["tags"]
    
    def test_memory_match_validation(self):
        """Test MemoryMatch validation."""
        # Missing required fields
        with pytest.raises(ValidationError) as exc_info:
            MemoryMatch(memory_id="test", content="test")
        assert "memory_type" in str(exc_info.value)
        assert "relevance_score" in str(exc_info.value)


class TestRecallResponse:
    """Test RecallResponse model."""
    
    def test_recall_response_empty(self):
        """Test creating RecallResponse with no matches."""
        response = RecallResponse(
            matches=[],
            strategy_used=RecallStrategy.CONTEXTUAL,
            total_matches=0
        )
        
        assert response.matches == []
        assert response.strategy_used == RecallStrategy.CONTEXTUAL
        assert response.total_matches == 0
        assert response.query_analysis is None
    
    def test_recall_response_with_matches(self):
        """Test creating RecallResponse with matches."""
        matches = [
            MemoryMatch(
                memory_id="mem_1",
                content="First memory",
                memory_type="note",
                relevance_score=0.9
            ),
            MemoryMatch(
                memory_id="mem_2",
                content="Second memory",
                memory_type="conversation",
                relevance_score=0.85
            )
        ]
        
        response = RecallResponse(
            matches=matches,
            strategy_used=RecallStrategy.SEMANTIC,
            total_matches=10,  # Total found, not just returned
            query_analysis={
                "concepts": ["memory", "recall"],
                "confidence": 0.88
            }
        )
        
        assert len(response.matches) == 2
        assert response.matches[0].memory_id == "mem_1"
        assert response.total_matches == 10
        assert response.query_analysis["confidence"] == 0.88
    
    def test_recall_response_validation(self):
        """Test RecallResponse validation."""
        # Missing required fields
        with pytest.raises(ValidationError) as exc_info:
            RecallResponse(matches=[])
        assert "strategy_used" in str(exc_info.value)
        assert "total_matches" in str(exc_info.value)
        
        # Invalid matches type
        with pytest.raises(ValidationError):
            RecallResponse(
                matches="not_a_list",
                strategy_used=RecallStrategy.ENTITY,
                total_matches=0
            )