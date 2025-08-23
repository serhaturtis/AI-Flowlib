"""Tests for semantic recall models."""

import pytest
from pydantic import ValidationError

from flowlib.agent.components.remember.semantic_recall.models import (
    SemanticAnalysis,
    SemanticRecallRequest,
    SemanticRecallResponse,
    RecallRequest,
    RecallResponse,
    RecallStrategy,
    MemoryMatch
)


class TestSemanticAnalysis:
    """Test cases for SemanticAnalysis model."""

    def test_basic_semantic_analysis(self):
        """Test creating a basic semantic analysis."""
        analysis = SemanticAnalysis(
            key_concepts=["machine learning", "algorithms"],
            semantic_relationships=["ML uses algorithms"],
            contextual_meaning="Query about machine learning techniques",
            topic_categories=["computer science", "AI"],
            confidence=0.85
        )
        
        assert analysis.key_concepts == ["machine learning", "algorithms"]
        assert analysis.semantic_relationships == ["ML uses algorithms"]
        assert analysis.contextual_meaning == "Query about machine learning techniques"
        assert analysis.topic_categories == ["computer science", "AI"]
        assert analysis.confidence == 0.85

    def test_semantic_analysis_with_defaults(self):
        """Test semantic analysis with default values."""
        analysis = SemanticAnalysis(
            contextual_meaning="Test meaning",
            confidence=0.9
        )
        
        assert analysis.key_concepts == []
        assert analysis.semantic_relationships == []
        assert analysis.contextual_meaning == "Test meaning"
        assert analysis.topic_categories == []
        assert analysis.confidence == 0.9

    def test_semantic_analysis_field_validation(self):
        """Test field validation for SemanticAnalysis."""
        # Test required fields
        with pytest.raises(ValidationError) as exc_info:
            SemanticAnalysis()
        
        error_fields = [error['loc'][0] for error in exc_info.value.errors()]
        assert 'contextual_meaning' in error_fields
        assert 'confidence' in error_fields

    def test_semantic_analysis_confidence_validation(self):
        """Test confidence field validation."""
        # Valid confidence values
        for confidence in [0.0, 0.5, 1.0]:
            analysis = SemanticAnalysis(
                contextual_meaning="test",
                confidence=confidence
            )
            assert analysis.confidence == confidence

    def test_semantic_analysis_list_fields(self):
        """Test list field behavior."""
        analysis = SemanticAnalysis(
            key_concepts=["concept1", "concept2", "concept3"],
            semantic_relationships=["rel1", "rel2"],
            contextual_meaning="test meaning",
            topic_categories=["cat1", "cat2", "cat3", "cat4"],
            confidence=0.75
        )
        
        assert len(analysis.key_concepts) == 3
        assert len(analysis.semantic_relationships) == 2
        assert len(analysis.topic_categories) == 4

    def test_semantic_analysis_serialization(self):
        """Test SemanticAnalysis serialization."""
        analysis = SemanticAnalysis(
            key_concepts=["neural networks", "deep learning"],
            semantic_relationships=["neural networks enable deep learning"],
            contextual_meaning="Query about neural network architectures",
            topic_categories=["machine learning", "artificial intelligence"],
            confidence=0.92
        )
        
        # Test dict conversion
        analysis_dict = analysis.model_dump()
        assert analysis_dict['key_concepts'] == ["neural networks", "deep learning"]
        assert analysis_dict['contextual_meaning'] == "Query about neural network architectures"
        assert analysis_dict['confidence'] == 0.92
        
        # Test reconstruction from dict
        reconstructed = SemanticAnalysis(**analysis_dict)
        assert reconstructed == analysis


class TestSemanticRecallRequest:
    """Test cases for SemanticRecallRequest model."""

    def test_basic_semantic_recall_request(self):
        """Test creating a basic semantic recall request."""
        request = SemanticRecallRequest(
            query="semantic search query",
            strategy=RecallStrategy.SEMANTIC
        )
        
        assert request.query == "semantic search query"
        assert request.strategy == RecallStrategy.SEMANTIC
        assert request.similarity_threshold == 0.7  # Default value
        assert request.include_related_concepts is True  # Default value
        assert request.limit == 10  # From parent class
        assert request.memory_types == []  # From parent class

    def test_semantic_recall_request_with_custom_threshold(self):
        """Test semantic recall request with custom similarity threshold."""
        request = SemanticRecallRequest(
            query="test query",
            strategy=RecallStrategy.SEMANTIC,
            similarity_threshold=0.85
        )
        
        assert request.similarity_threshold == 0.85

    def test_semantic_recall_request_without_related_concepts(self):
        """Test semantic recall request excluding related concepts."""
        request = SemanticRecallRequest(
            query="test query",
            strategy=RecallStrategy.SEMANTIC,
            include_related_concepts=False
        )
        
        assert request.include_related_concepts is False

    def test_semantic_recall_request_with_all_fields(self):
        """Test semantic recall request with all fields populated."""
        request = SemanticRecallRequest(
            query="comprehensive semantic query",
            strategy=RecallStrategy.SEMANTIC,
            context="research context",
            entity_id=None,
            similarity_threshold=0.9,
            include_related_concepts=True,
            limit=20,
            memory_types=["episodic", "semantic"]
        )
        
        assert request.query == "comprehensive semantic query"
        assert request.strategy == RecallStrategy.SEMANTIC
        assert request.context == "research context"
        assert request.similarity_threshold == 0.9
        assert request.include_related_concepts is True
        assert request.limit == 20
        assert request.memory_types == ["episodic", "semantic"]

    def test_semantic_recall_request_inheritance(self):
        """Test that SemanticRecallRequest inherits from RecallRequest."""
        request = SemanticRecallRequest(
            query="test query",
            strategy=RecallStrategy.SEMANTIC
        )
        
        assert isinstance(request, RecallRequest)
        assert hasattr(request, 'query')
        assert hasattr(request, 'strategy')
        assert hasattr(request, 'similarity_threshold')

    def test_semantic_recall_request_field_validation(self):
        """Test field validation for SemanticRecallRequest."""
        # Test required fields from parent
        with pytest.raises(ValidationError) as exc_info:
            SemanticRecallRequest()
        
        error_fields = [error['loc'][0] for error in exc_info.value.errors()]
        assert 'query' in error_fields
        assert 'strategy' in error_fields

    def test_semantic_recall_request_strategy_validation(self):
        """Test strategy field validation."""
        with pytest.raises(ValidationError) as exc_info:
            SemanticRecallRequest(
                query="test query",
                strategy="invalid_strategy"
            )
        
        assert "invalid_strategy" in str(exc_info.value)

    def test_semantic_recall_request_threshold_validation(self):
        """Test similarity threshold validation."""
        # Valid thresholds
        for threshold in [0.0, 0.5, 0.9, 1.0]:
            request = SemanticRecallRequest(
                query="test query",
                strategy=RecallStrategy.SEMANTIC,
                similarity_threshold=threshold
            )
            assert request.similarity_threshold == threshold

    def test_semantic_recall_request_optional_fields_defaults(self):
        """Test that optional fields have correct defaults."""
        request = SemanticRecallRequest(
            query="test query",
            strategy=RecallStrategy.SEMANTIC
        )
        
        assert request.context is None
        assert request.entity_id is None
        assert request.similarity_threshold == 0.7
        assert request.include_related_concepts is True
        assert request.limit == 10
        assert request.memory_types == []

    def test_semantic_recall_request_serialization(self):
        """Test SemanticRecallRequest serialization."""
        request = SemanticRecallRequest(
            query="semantic test query",
            strategy=RecallStrategy.SEMANTIC,
            similarity_threshold=0.8,
            include_related_concepts=False
        )
        
        # Test dict conversion
        request_dict = request.model_dump()
        assert request_dict['query'] == "semantic test query"
        assert request_dict['strategy'] == "semantic"
        assert request_dict['similarity_threshold'] == 0.8
        assert request_dict['include_related_concepts'] is False
        
        # Test reconstruction from dict
        reconstructed = SemanticRecallRequest(**request_dict)
        assert reconstructed == request


class TestSemanticRecallResponse:
    """Test cases for SemanticRecallResponse model."""

    @pytest.fixture
    def sample_memory_match(self):
        """Create a sample memory match."""
        return MemoryMatch(
            memory_id="mem-123",
            content="sample semantic memory content",
            memory_type="semantic",
            relevance_score=0.85,
            metadata={"timestamp": "2024-01-01", "concepts": ["test", "semantic"]}
        )

    def test_basic_semantic_recall_response(self, sample_memory_match):
        """Test creating a basic semantic recall response."""
        response = SemanticRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.SEMANTIC,
            total_matches=1
        )
        
        assert len(response.matches) == 1
        assert response.strategy_used == RecallStrategy.SEMANTIC
        assert response.total_matches == 1
        assert response.semantic_clusters == []  # Default value
        assert response.concept_coverage == {}  # Default value

    def test_semantic_recall_response_with_clusters(self, sample_memory_match):
        """Test semantic recall response with semantic clusters."""
        clusters = [
            {"cluster_id": "cluster1", "concepts": ["AI", "ML"], "size": 5},
            {"cluster_id": "cluster2", "concepts": ["algorithms"], "size": 3}
        ]
        
        response = SemanticRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.SEMANTIC,
            total_matches=1,
            semantic_clusters=clusters
        )
        
        assert response.semantic_clusters == clusters
        assert len(response.semantic_clusters) == 2

    def test_semantic_recall_response_with_concept_coverage(self, sample_memory_match):
        """Test semantic recall response with concept coverage."""
        coverage = {
            "machine_learning": 0.95,
            "algorithms": 0.85,
            "neural_networks": 0.70
        }
        
        response = SemanticRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.SEMANTIC,
            total_matches=1,
            concept_coverage=coverage
        )
        
        assert response.concept_coverage == coverage
        assert response.concept_coverage["machine_learning"] == 0.95

    def test_semantic_recall_response_inheritance(self, sample_memory_match):
        """Test that SemanticRecallResponse inherits from RecallResponse."""
        response = SemanticRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.SEMANTIC,
            total_matches=1
        )
        
        assert isinstance(response, RecallResponse)
        assert hasattr(response, 'matches')
        assert hasattr(response, 'strategy_used')
        assert hasattr(response, 'total_matches')
        assert hasattr(response, 'semantic_clusters')

    def test_semantic_recall_response_with_multiple_matches(self):
        """Test semantic recall response with multiple memory matches."""
        matches = [
            MemoryMatch(
                memory_id=f"mem-{i}",
                content=f"semantic memory content {i}",
                memory_type="semantic",
                relevance_score=0.9 - (i * 0.1),
                metadata={"concept_group": f"group_{i}"}
            )
            for i in range(3)
        ]
        
        response = SemanticRecallResponse(
            matches=matches,
            strategy_used=RecallStrategy.SEMANTIC,
            total_matches=3,
            semantic_clusters=[{"cluster": "main", "size": 3}]
        )
        
        assert len(response.matches) == 3
        assert response.total_matches == 3
        assert len(response.semantic_clusters) == 1

    def test_semantic_recall_response_empty_matches(self):
        """Test semantic recall response with no matches."""
        response = SemanticRecallResponse(
            matches=[],
            strategy_used=RecallStrategy.SEMANTIC,
            total_matches=0
        )
        
        assert response.matches == []
        assert response.total_matches == 0
        assert response.semantic_clusters == []
        assert response.concept_coverage == {}

    def test_semantic_recall_response_field_validation(self):
        """Test field validation for SemanticRecallResponse."""
        # Test required fields
        with pytest.raises(ValidationError) as exc_info:
            SemanticRecallResponse()
        
        error_fields = [error['loc'][0] for error in exc_info.value.errors()]
        assert 'matches' in error_fields
        assert 'strategy_used' in error_fields
        assert 'total_matches' in error_fields

    def test_semantic_recall_response_strategy_validation(self):
        """Test strategy field validation."""
        with pytest.raises(ValidationError) as exc_info:
            SemanticRecallResponse(
                matches=[],
                strategy_used="invalid_strategy",
                total_matches=0
            )
        
        assert "invalid_strategy" in str(exc_info.value)

    def test_semantic_recall_response_with_query_analysis(self, sample_memory_match):
        """Test semantic recall response with query analysis."""
        query_analysis = {
            "semantic_concepts": ["machine learning", "algorithms"],
            "confidence": 0.88,
            "processing_time": 0.25
        }
        
        response = SemanticRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.SEMANTIC,
            total_matches=1,
            query_analysis=query_analysis
        )
        
        assert response.query_analysis == query_analysis

    def test_semantic_recall_response_serialization(self, sample_memory_match):
        """Test SemanticRecallResponse serialization."""
        clusters = [{"cluster": "test", "size": 1}]
        coverage = {"test_concept": 0.95}
        
        response = SemanticRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.SEMANTIC,
            total_matches=1,
            semantic_clusters=clusters,
            concept_coverage=coverage
        )
        
        # Test dict conversion
        response_dict = response.model_dump()
        assert response_dict['total_matches'] == 1
        assert response_dict['strategy_used'] == "semantic"
        assert response_dict['semantic_clusters'] == clusters
        assert response_dict['concept_coverage'] == coverage
        assert len(response_dict['matches']) == 1
        
        # Test reconstruction from dict
        reconstructed = SemanticRecallResponse(**response_dict)
        assert reconstructed.semantic_clusters == response.semantic_clusters
        assert reconstructed.concept_coverage == response.concept_coverage

    def test_semantic_recall_response_optional_fields_defaults(self, sample_memory_match):
        """Test that optional fields have correct defaults."""
        response = SemanticRecallResponse(
            matches=[sample_memory_match],
            strategy_used=RecallStrategy.SEMANTIC,
            total_matches=1
        )
        
        assert response.semantic_clusters == []
        assert response.concept_coverage == {}
        assert response.query_analysis is None


class TestModelExports:
    """Test model exports and imports."""

    def test_model_exports(self):
        """Test that all expected models are exported."""
        from flowlib.agent.components.remember.semantic_recall.models import __all__
        
        expected_exports = [
            "SemanticAnalysis",
            "SemanticRecallRequest",
            "SemanticRecallResponse",
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
        from flowlib.agent.components.remember.semantic_recall.models import (
            SemanticAnalysis,
            SemanticRecallRequest,
            SemanticRecallResponse,
            RecallRequest,
            RecallResponse,
            RecallStrategy,
            MemoryMatch
        )
        
        # Verify they are classes/enums
        assert isinstance(RecallStrategy.SEMANTIC, RecallStrategy)
        
        # Test instantiation
        analysis = SemanticAnalysis(
            contextual_meaning="test",
            confidence=0.8
        )
        assert isinstance(analysis, SemanticAnalysis)
        
        request = SemanticRecallRequest(
            query="test",
            strategy=RecallStrategy.SEMANTIC
        )
        assert isinstance(request, SemanticRecallRequest)
        assert isinstance(request, RecallRequest)

    def test_recall_strategy_values(self):
        """Test RecallStrategy enum values."""
        assert RecallStrategy.SEMANTIC == "semantic"
        assert RecallStrategy.ENTITY == "entity"
        assert RecallStrategy.TEMPORAL == "temporal"
        assert RecallStrategy.CONTEXTUAL == "contextual"

    def test_semantic_analysis_complex_scenario(self):
        """Test SemanticAnalysis with complex real-world data."""
        analysis = SemanticAnalysis(
            key_concepts=[
                "natural language processing",
                "transformer architecture", 
                "attention mechanism",
                "BERT",
                "fine-tuning"
            ],
            semantic_relationships=[
                "transformers use attention mechanisms",
                "BERT is based on transformer architecture",
                "fine-tuning improves model performance",
                "NLP models process natural language"
            ],
            contextual_meaning="User is asking about transformer-based NLP models, specifically BERT and fine-tuning techniques for downstream tasks",
            topic_categories=[
                "natural language processing",
                "machine learning",
                "deep learning",
                "artificial intelligence",
                "computer science"
            ],
            confidence=0.94
        )
        
        assert len(analysis.key_concepts) == 5
        assert len(analysis.semantic_relationships) == 4
        assert len(analysis.topic_categories) == 5
        assert analysis.confidence == 0.94
        assert "transformer" in analysis.contextual_meaning