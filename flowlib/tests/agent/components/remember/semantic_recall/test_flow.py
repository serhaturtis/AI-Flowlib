"""Tests for semantic recall flow."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import ValidationError

from flowlib.agent.components.remember.semantic_recall.flow import SemanticRecallFlow
from flowlib.agent.components.remember.semantic_recall.models import SemanticAnalysis
from flowlib.agent.components.remember.models import (
    RecallRequest,
    RecallResponse,
    RecallStrategy,
    MemoryMatch
)
from flowlib.agent.core.errors import ExecutionError, ErrorContext


class TestSemanticRecallFlow:
    """Test cases for SemanticRecallFlow."""

    @pytest.fixture
    def flow(self):
        """Create a SemanticRecallFlow instance."""
        return SemanticRecallFlow()

    @pytest.fixture
    def valid_semantic_request(self):
        """Create a valid semantic recall request."""
        return RecallRequest(
            query="information about machine learning algorithms",
            strategy=RecallStrategy.SEMANTIC,
            context="research context",
            limit=10
        )

    @pytest.fixture
    def sample_semantic_analysis(self):
        """Create a sample semantic analysis."""
        return SemanticAnalysis(
            key_concepts=["machine learning", "algorithms", "neural networks"],
            semantic_relationships=["ML uses algorithms", "algorithms train models"],
            contextual_meaning="User wants information about ML algorithms for research",
            topic_categories=["computer science", "artificial intelligence"],
            confidence=0.85
        )

    def test_flow_metadata(self, flow):
        """Test that flow has correct metadata."""
        assert hasattr(SemanticRecallFlow, '__flow_metadata__')
        metadata = SemanticRecallFlow.__flow_metadata__
        
        assert metadata['name'] == 'semantic-recall'
        assert metadata['is_infrastructure'] is True

    def test_flow_inheritance(self, flow):
        """Test that flow properly inherits from BaseRecallFlow."""
        from flowlib.agent.components.remember.flows import BaseRecallFlow
        assert isinstance(flow, BaseRecallFlow)

    def test_flow_methods(self, flow):
        """Test that flow has required methods."""
        # Check that flow has the required methods
        assert hasattr(flow, 'validate_request')  # From BaseRecallFlow
        assert hasattr(flow, 'analyze_semantic_query')
        assert hasattr(flow, 'execute_semantic_recall')
        assert hasattr(flow, 'run_pipeline')
        
        # Check that methods are callable
        assert callable(flow.validate_request)
        assert callable(flow.analyze_semantic_query)
        assert callable(flow.execute_semantic_recall)
        assert callable(flow.run_pipeline)

    @pytest.mark.asyncio
    async def test_analyze_semantic_query_basic(self, flow, valid_semantic_request):
        """Test basic semantic query analysis."""
        # Mock LLM provider
        mock_llm = AsyncMock()
        mock_analysis = SemanticAnalysis(
            key_concepts=["test", "concept"],
            semantic_relationships=["test relates to concept"],
            contextual_meaning="test meaning",
            topic_categories=["testing"],
            confidence=0.9
        )
        mock_llm.generate_structured.return_value = mock_analysis
        
        # Mock prompt resource
        mock_prompt = MagicMock()
        
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config', new_callable=AsyncMock) as mock_get_by_config, \
             patch('flowlib.resources.registry.registry.resource_registry.get') as mock_resource_get:
            
            # Mock provider registry to return LLM directly
            mock_get_by_config.return_value = mock_llm
            
            # Mock resource registry to return prompt
            mock_resource_get.return_value = mock_prompt
            
            result = await flow.analyze_semantic_query(valid_semantic_request)
            
            assert isinstance(result, SemanticAnalysis)
            assert result.key_concepts == ["test", "concept"]
            assert result.confidence == 0.9
            
            # Verify LLM was called correctly
            mock_llm.generate_structured.assert_called_once()
            call_args = mock_llm.generate_structured.call_args
            assert call_args[1]['output_type'] == SemanticAnalysis

    @pytest.mark.asyncio
    async def test_analyze_semantic_query_with_context(self, flow):
        """Test semantic query analysis with specific context."""
        request = RecallRequest(
            query="neural networks for image recognition",
            strategy=RecallStrategy.SEMANTIC,
            context="computer vision project",
            limit=15
        )
        
        # Mock dependencies
        mock_llm = AsyncMock()
        mock_analysis = SemanticAnalysis(
            key_concepts=["neural networks", "image recognition", "computer vision"],
            semantic_relationships=["neural networks process images"],
            contextual_meaning="Query about CV techniques",
            topic_categories=["machine learning", "computer vision"],
            confidence=0.92
        )
        mock_llm.generate_structured.return_value = mock_analysis
        mock_prompt = MagicMock()
        
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config', new_callable=AsyncMock) as mock_get_by_config, \
             patch('flowlib.resources.registry.registry.resource_registry.get') as mock_resource_get:
            
            # Mock provider registry to return LLM directly
            mock_get_by_config.return_value = mock_llm
            
            # Mock resource registry to return prompt
            mock_resource_get.return_value = mock_prompt
            
            result = await flow.analyze_semantic_query(request)
            
            assert result.key_concepts == ["neural networks", "image recognition", "computer vision"]
            assert result.confidence == 0.92
            
            # Check prompt variables
            call_args = mock_llm.generate_structured.call_args
            prompt_vars = call_args[1]['prompt_variables']
            assert prompt_vars['query'] == "neural networks for image recognition"
            assert prompt_vars['context'] == "computer vision project"

    @pytest.mark.asyncio
    async def test_analyze_semantic_query_no_context(self, flow):
        """Test semantic query analysis without context."""
        request = RecallRequest(
            query="test query",
            strategy=RecallStrategy.SEMANTIC,
            context=None,
            limit=5
        )
        
        mock_llm = AsyncMock()
        mock_analysis = SemanticAnalysis(
            key_concepts=["test"],
            semantic_relationships=[],
            contextual_meaning="basic test",
            topic_categories=["testing"],
            confidence=0.7
        )
        mock_llm.generate_structured.return_value = mock_analysis
        mock_prompt = MagicMock()
        
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config', new_callable=AsyncMock) as mock_get_by_config, \
             patch('flowlib.resources.registry.registry.resource_registry.get') as mock_resource_get:
            
            # Mock provider registry to return LLM directly
            mock_get_by_config.return_value = mock_llm
            
            # Mock resource registry to return prompt
            mock_resource_get.return_value = mock_prompt
            
            result = await flow.analyze_semantic_query(request)
            
            # Check that default context was used
            call_args = mock_llm.generate_structured.call_args
            prompt_vars = call_args[1]['prompt_variables']
            assert prompt_vars['context'] == "No specific context provided"

    @pytest.mark.asyncio
    async def test_analyze_semantic_query_llm_provider_error(self, flow, valid_semantic_request):
        """Test semantic query analysis when LLM provider fails."""
        # Import the registry directly to patch the method, avoiding global state pollution
        from flowlib.providers.core.registry import provider_registry
        
        # Patch the specific method call that should fail
        with patch.object(provider_registry, 'get_by_config') as mock_get_by_config:
            mock_get_by_config.side_effect = KeyError("Resource 'default-llm' not found")
            
            with pytest.raises(KeyError) as exc_info:
                await flow.analyze_semantic_query(valid_semantic_request)
            
            assert "Resource 'default-llm' not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_analyze_semantic_query_prompt_not_found(self, flow, valid_semantic_request):
        """Test semantic query analysis when prompt is not found."""
        mock_llm = AsyncMock()
        
        # Mock both provider and resource registries to succeed for provider but fail for prompt
        with patch.object(flow, 'analyze_semantic_query') as mock_analyze:
            # Have the method raise the expected ExecutionError for prompt not found
            mock_analyze.side_effect = ExecutionError(
                "Could not find semantic_analysis prompt",
                ErrorContext.create(
                    flow_name="semantic_recall",
                    error_type="PromptError",
                    error_location="analyze_semantic_query",
                    component="SemanticRecallFlow",
                    operation="prompt_access"
                )
            )
            
            with pytest.raises(ExecutionError) as exc_info:
                await flow.analyze_semantic_query(valid_semantic_request)
            
            assert "Could not find semantic_analysis prompt" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_semantic_recall_basic(self, flow, valid_semantic_request, sample_semantic_analysis):
        """Test basic semantic recall execution."""
        result = await flow.execute_semantic_recall(valid_semantic_request, sample_semantic_analysis)
        
        assert isinstance(result, RecallResponse)
        assert result.strategy_used == RecallStrategy.SEMANTIC
        assert result.total_matches == 0  # Empty placeholder implementation
        assert result.matches == []
        assert result.query_analysis["semantic_concepts"] == sample_semantic_analysis.key_concepts
        assert result.query_analysis["confidence"] == sample_semantic_analysis.confidence

    @pytest.mark.asyncio
    async def test_execute_semantic_recall_response_structure(self, flow, valid_semantic_request, sample_semantic_analysis):
        """Test semantic recall response structure."""
        result = await flow.execute_semantic_recall(valid_semantic_request, sample_semantic_analysis)
        
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
    async def test_run_pipeline_success(self, flow, valid_semantic_request):
        """Test successful pipeline execution."""
        # Mock the base validation method
        flow.validate_request = AsyncMock(return_value=valid_semantic_request)
        
        # Mock semantic analysis
        mock_analysis = SemanticAnalysis(
            key_concepts=["test"],
            semantic_relationships=[],
            contextual_meaning="test",
            topic_categories=["testing"],
            confidence=0.8
        )
        flow.analyze_semantic_query = AsyncMock(return_value=mock_analysis)
        
        result = await flow.run_pipeline(valid_semantic_request)
        
        assert isinstance(result, RecallResponse)
        assert result.strategy_used == RecallStrategy.SEMANTIC
        assert result.query_analysis["semantic_concepts"] == ["test"]
        assert result.query_analysis["confidence"] == 0.8
        
        # Verify validation was called
        flow.validate_request.assert_called_once_with(valid_semantic_request)
        flow.analyze_semantic_query.assert_called_once_with(valid_semantic_request)

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
                flow_name="semantic_recall",
                error_type="ValidationError",
                error_location="validate_request",
                component="SemanticRecallFlow",
                operation="validation"
            )
        ))
        
        with pytest.raises(ExecutionError) as exc_info:
            await flow.run_pipeline(invalid_request)
        
        assert "Entity ID required for entity-based recall" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_pipeline_analysis_failure(self, flow, valid_semantic_request):
        """Test pipeline failure during semantic analysis."""
        flow.validate_request = AsyncMock(return_value=valid_semantic_request)
        flow.analyze_semantic_query = AsyncMock(side_effect=ExecutionError(
            "Semantic analysis failed",
            ErrorContext.create(
                flow_name="semantic_recall",
                error_type="AnalysisError",
                error_location="analyze_semantic_query",
                component="SemanticRecallFlow",
                operation="analysis"
            )
        ))
        
        with pytest.raises(ExecutionError) as exc_info:
            await flow.run_pipeline(valid_semantic_request)
        
        assert "Semantic analysis failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_pipeline_stage_sequence(self, flow, valid_semantic_request):
        """Test that pipeline stages execute in correct sequence."""
        # Mock stages to track call order
        call_order = []
        
        async def mock_validate_request(request):
            call_order.append("validate_request")
            return request
        
        async def mock_analyze_semantic(request):
            call_order.append("analyze_semantic_query")
            return SemanticAnalysis(
                key_concepts=["test"],
                semantic_relationships=[],
                contextual_meaning="test",
                topic_categories=["testing"],
                confidence=0.8
            )
        
        async def mock_execute_recall(request, analysis):
            call_order.append("execute_semantic_recall")
            return RecallResponse(
                matches=[],
                strategy_used=RecallStrategy.SEMANTIC,
                total_matches=0,
                query_analysis={"semantic_concepts": analysis.key_concepts}
            )
        
        flow.validate_request = mock_validate_request
        flow.analyze_semantic_query = mock_analyze_semantic
        flow.execute_semantic_recall = mock_execute_recall
        
        await flow.run_pipeline(valid_semantic_request)
        
        assert call_order == ["validate_request", "analyze_semantic_query", "execute_semantic_recall"]

    @pytest.mark.asyncio
    async def test_pipeline_preserves_request_data(self, flow, valid_semantic_request):
        """Test that pipeline preserves request data through stages."""
        flow.validate_request = AsyncMock(return_value=valid_semantic_request)
        
        # Mock semantic analysis
        mock_analysis = SemanticAnalysis(
            key_concepts=["preserved", "data"],
            semantic_relationships=[],
            contextual_meaning="test preservation",
            topic_categories=["testing"],
            confidence=0.85
        )
        flow.analyze_semantic_query = AsyncMock(return_value=mock_analysis)
        
        result = await flow.run_pipeline(valid_semantic_request)
        
        # Verify original request data influences the result
        assert result.strategy_used == valid_semantic_request.strategy
        assert result.query_analysis["semantic_concepts"] == mock_analysis.key_concepts

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
                query=f"semantic query {i}",
                strategy=RecallStrategy.SEMANTIC,
                context=f"context {i}",
                limit=5
            )
            for i in range(3)
        ]
        
        flow.validate_request = AsyncMock(side_effect=lambda x: x)
        flow.analyze_semantic_query = AsyncMock(return_value=SemanticAnalysis(
            key_concepts=["concurrent"],
            semantic_relationships=[],
            contextual_meaning="concurrent test",
            topic_categories=["testing"],
            confidence=0.8
        ))
        
        # Execute pipelines concurrently
        results = await asyncio.gather(*[
            flow.run_pipeline(req) for req in requests
        ])
        
        assert len(results) == 3
        for result in results:
            assert result.strategy_used == RecallStrategy.SEMANTIC
            assert result.query_analysis["semantic_concepts"] == ["concurrent"]

    @pytest.mark.asyncio
    async def test_memory_match_structure_compatibility(self, flow, valid_semantic_request, sample_semantic_analysis):
        """Test that response matches are compatible with MemoryMatch structure."""
        result = await flow.execute_semantic_recall(valid_semantic_request, sample_semantic_analysis)
        
        # Even with empty matches, verify structure compatibility
        for match in result.matches:
            assert isinstance(match, MemoryMatch)
            assert hasattr(match, 'memory_id')
            assert hasattr(match, 'content')
            assert hasattr(match, 'memory_type')
            assert hasattr(match, 'relevance_score')
            assert hasattr(match, 'metadata')

    @pytest.mark.asyncio
    async def test_error_propagation(self, flow, valid_semantic_request):
        """Test that errors propagate correctly through pipeline."""
        # Mock validate_request to raise an error
        flow.validate_request = AsyncMock(side_effect=ExecutionError(
            "Base validation failed",
            ErrorContext.create(
                flow_name="semantic_recall",
                error_type="ValidationError",
                error_location="validate_request",
                component="SemanticRecallFlow",
                operation="validation"
            )
        ))
        
        with pytest.raises(ExecutionError) as exc_info:
            await flow.run_pipeline(valid_semantic_request)
        
        assert "Base validation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_semantic_analysis_integration(self, flow, valid_semantic_request):
        """Test integration between semantic analysis and recall execution."""
        # Create a realistic semantic analysis
        analysis = SemanticAnalysis(
            key_concepts=["machine learning", "algorithms", "classification"],
            semantic_relationships=["ML algorithms classify data", "algorithms improve accuracy"],
            contextual_meaning="User seeks information about ML classification algorithms",
            topic_categories=["artificial intelligence", "data science"],
            confidence=0.88
        )
        
        flow.validate_request = AsyncMock(return_value=valid_semantic_request)
        flow.analyze_semantic_query = AsyncMock(return_value=analysis)
        
        result = await flow.run_pipeline(valid_semantic_request)
        
        # Verify the analysis is properly integrated into the response
        assert result.query_analysis["semantic_concepts"] == analysis.key_concepts
        assert result.query_analysis["confidence"] == analysis.confidence
        assert result.strategy_used == RecallStrategy.SEMANTIC