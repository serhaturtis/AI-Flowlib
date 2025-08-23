"""Tests for remember flows."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from flowlib.agent.components.remember.flows import (
    BaseRecallFlow,
    ContextualRecallFlow
)
from flowlib.agent.components.remember.models import (
    RecallRequest,
    RecallResponse,
    RecallStrategy,
    ContextAnalysis,
    MemoryMatch
)
from flowlib.agent.core.errors import ExecutionError
from flowlib.resources.models.constants import ResourceType


class TestBaseRecallFlow:
    """Test BaseRecallFlow class."""
    
    @pytest.fixture
    def base_flow(self):
        """Create BaseRecallFlow instance."""
        return BaseRecallFlow()
    
    def test_base_flow_initialization(self, base_flow):
        """Test BaseRecallFlow initialization."""
        assert base_flow.name == "BaseRecallFlow"
        assert base_flow.metadata.get("is_infrastructure") is True
    
    @pytest.mark.asyncio
    async def test_validate_request_valid(self, base_flow):
        """Test validate_request with valid request."""
        request = RecallRequest(
            query="test query",
            strategy=RecallStrategy.CONTEXTUAL
        )
        
        result = await base_flow.validate_request(request)
        assert result == request
    
    @pytest.mark.asyncio
    async def test_validate_request_entity_without_id(self, base_flow):
        """Test validate_request with entity strategy but no entity_id."""
        request = RecallRequest(
            query="test query",
            strategy=RecallStrategy.ENTITY
        )
        
        with pytest.raises(ExecutionError, match="Entity ID required"):
            await base_flow.validate_request(request)
    
    @pytest.mark.asyncio
    async def test_validate_request_entity_with_id(self, base_flow):
        """Test validate_request with entity strategy and entity_id."""
        request = RecallRequest(
            query="test query",
            strategy=RecallStrategy.ENTITY,
            entity_id="entity_123"
        )
        
        result = await base_flow.validate_request(request)
        assert result == request
    
    @pytest.mark.asyncio
    async def test_execute_recall_not_implemented(self, base_flow):
        """Test that execute_recall raises NotImplementedError."""
        request = RecallRequest(
            query="test",
            strategy=RecallStrategy.SEMANTIC
        )
        
        with pytest.raises(NotImplementedError):
            await base_flow.execute_recall(request)
    
    @pytest.mark.asyncio
    async def test_run_pipeline(self, base_flow):
        """Test run_pipeline method."""
        request = RecallRequest(
            query="test query",
            strategy=RecallStrategy.TEMPORAL
        )
        
        mock_response = RecallResponse(
            matches=[],
            strategy_used=RecallStrategy.TEMPORAL,
            total_matches=0
        )
        
        # Mock execute_recall
        base_flow.execute_recall = AsyncMock(return_value=mock_response)
        
        result = await base_flow.run_pipeline(request)
        
        assert result == mock_response
        base_flow.execute_recall.assert_called_once()


class TestContextualRecallFlow:
    """Test ContextualRecallFlow class."""
    
    @pytest.fixture
    def contextual_flow(self):
        """Create ContextualRecallFlow instance."""
        return ContextualRecallFlow()
    
    def test_contextual_flow_initialization(self, contextual_flow):
        """Test ContextualRecallFlow initialization."""
        assert isinstance(contextual_flow, BaseRecallFlow)
    
    @pytest.mark.asyncio
    async def test_analyze_context_success(self, contextual_flow):
        """Test analyze_context with successful LLM call."""
        request = RecallRequest(
            query="What was discussed about budgets?",
            strategy=RecallStrategy.CONTEXTUAL,
            context="In the last team meeting"
        )
        
        mock_analysis = ContextAnalysis(
            analysis="Query about budget discussions in team meeting",
            recommended_strategy=RecallStrategy.CONTEXTUAL,
            key_concepts=["budget", "team meeting", "discussion"],
            confidence=0.9
        )
        
        mock_llm = Mock()
        mock_llm.generate_structured = AsyncMock(return_value=mock_analysis)
        
        mock_prompt = Mock()
        
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config', 
                  AsyncMock(return_value=mock_llm)):
            with patch('flowlib.resources.registry.registry.resource_registry.get',
                      return_value=mock_prompt):
                
                result = await contextual_flow.analyze_context(request)
                
                assert result == mock_analysis
                mock_llm.generate_structured.assert_called_once()
                
                # Check prompt variables
                call_kwargs = mock_llm.generate_structured.call_args[1]
                assert call_kwargs['prompt_variables']['query'] == request.query
                assert call_kwargs['prompt_variables']['context'] == request.context
    
    @pytest.mark.asyncio
    async def test_analyze_context_no_prompt(self, contextual_flow):
        """Test analyze_context when prompt is not found."""
        request = RecallRequest(
            query="test",
            strategy=RecallStrategy.CONTEXTUAL
        )
        
        mock_llm = Mock()
        
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config',
                  AsyncMock(return_value=mock_llm)):
            with patch('flowlib.resources.registry.registry.resource_registry.get',
                      return_value=None):
                
                with pytest.raises(ExecutionError, match="Could not find context_analysis prompt"):
                    await contextual_flow.analyze_context(request)
    
    @pytest.mark.asyncio
    async def test_analyze_context_no_context_provided(self, contextual_flow):
        """Test analyze_context with no context in request."""
        request = RecallRequest(
            query="Find something",
            strategy=RecallStrategy.CONTEXTUAL,
            context=None
        )
        
        mock_analysis = ContextAnalysis(
            analysis="Generic query analysis",
            recommended_strategy=RecallStrategy.SEMANTIC,
            confidence=0.6
        )
        
        mock_llm = Mock()
        mock_llm.generate_structured = AsyncMock(return_value=mock_analysis)
        mock_prompt = Mock()
        
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config',
                  AsyncMock(return_value=mock_llm)):
            with patch('flowlib.resources.registry.registry.resource_registry.get',
                      return_value=mock_prompt):
                
                result = await contextual_flow.analyze_context(request)
                
                # Check that "No specific context provided" was used
                call_kwargs = mock_llm.generate_structured.call_args[1]
                assert call_kwargs['prompt_variables']['context'] == "No specific context provided"
    
    @pytest.mark.asyncio
    async def test_execute_recall(self, contextual_flow):
        """Test execute_recall method."""
        request = RecallRequest(
            query="test query",
            strategy=RecallStrategy.CONTEXTUAL,
            context="test context"
        )
        
        result = await contextual_flow.execute_recall(request)
        
        assert isinstance(result, RecallResponse)
        assert result.matches == []  # Currently returns empty list
        assert result.strategy_used == RecallStrategy.CONTEXTUAL
        assert result.total_matches == 0
        assert result.query_analysis == {"context_analysis": "test context"}
    
    @pytest.mark.asyncio
    async def test_run_pipeline_full(self, contextual_flow):
        """Test full pipeline execution."""
        request = RecallRequest(
            query="What about the project deadline?",
            strategy=RecallStrategy.CONTEXTUAL,
            context="Recent project discussions"
        )
        
        mock_analysis = ContextAnalysis(
            analysis="Query about project deadline",
            recommended_strategy=RecallStrategy.TEMPORAL,
            key_concepts=["project", "deadline"],
            confidence=0.85
        )
        
        # Mock the methods
        contextual_flow.analyze_context = AsyncMock(return_value=mock_analysis)
        
        mock_response = RecallResponse(
            matches=[
                MemoryMatch(
                    memory_id="mem_1",
                    content="Project deadline is next Friday",
                    memory_type="meeting",
                    relevance_score=0.95
                )
            ],
            strategy_used=RecallStrategy.CONTEXTUAL,
            total_matches=1
        )
        
        contextual_flow.execute_recall = AsyncMock(return_value=mock_response)
        
        result = await contextual_flow.run_pipeline(request)
        
        assert result == mock_response
        contextual_flow.analyze_context.assert_called_once()
        contextual_flow.execute_recall.assert_called_once()
    
    def test_flow_decorators(self):
        """Test that flows have proper decorators."""
        # Check that flows are Flow subclasses
        from flowlib.flows.base.base import Flow
        assert issubclass(BaseRecallFlow, Flow)
        assert issubclass(ContextualRecallFlow, Flow)
        
        # Check initialization metadata
        base_flow = BaseRecallFlow()
        assert base_flow.name == "BaseRecallFlow"
        assert base_flow.metadata.get("is_infrastructure") is True
        
        contextual_flow = ContextualRecallFlow()
        assert contextual_flow.name == "BaseRecallFlow"  # Inherits from BaseRecallFlow