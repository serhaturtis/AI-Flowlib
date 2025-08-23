"""Tests for agent core context system."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from dataclasses import dataclass
from typing import Protocol

from flowlib.agent.core.context import (
    FlowContext, 
    ProcessingOptions, 
    FlowProvider,
    DEFAULT_CONTEXT,
    get_default_processing_options,
    create_flow_context
)


class MockProvider:
    """Mock provider for testing."""
    
    def __init__(self, name: str = "mock"):
        self.name = name
        self.initialized = False
    
    async def initialize(self):
        """Initialize the provider."""
        self.initialized = True


class TestFlowContext:
    """Test FlowContext functionality."""
    
    def test_initialization_defaults(self):
        """Test FlowContext initialization with defaults."""
        context = FlowContext()
        
        assert context.confidence_threshold == 0.7
        assert context.model_preference == "balanced"
        assert context._llm_provider is None
        assert context._graph_provider is None
        assert context._vector_provider is None
        assert context._cache_provider is None
    
    def test_initialization_with_values(self):
        """Test FlowContext initialization with custom values."""
        context = FlowContext(
            confidence_threshold=0.8,
            model_preference="fast"
        )
        
        assert context.confidence_threshold == 0.8
        assert context.model_preference == "fast"
    
    @pytest.mark.asyncio
    async def test_llm_provider_lazy_loading(self):
        """Test LLM provider lazy loading."""
        context = FlowContext(model_preference="fast")
        mock_provider = MockProvider("llm")
        
        with patch('flowlib.agent.core.context.provider_registry') as mock_registry:
            mock_registry.get_by_config = AsyncMock(return_value=mock_provider)
            
            # First call should load provider
            provider = await context.llm()
            assert provider is mock_provider
            mock_registry.get_by_config.assert_called_once_with("fast-llm")
            
            # Second call should return cached provider
            mock_registry.get_by_config.reset_mock()
            provider2 = await context.llm()
            assert provider2 is mock_provider
            mock_registry.get_by_config.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_graph_provider_lazy_loading(self):
        """Test graph provider lazy loading."""
        context = FlowContext()
        mock_provider = MockProvider("graph")
        
        with patch('flowlib.agent.core.context.provider_registry') as mock_registry:
            mock_registry.get_by_config = AsyncMock(return_value=mock_provider)
            
            provider = await context.graph()
            assert provider is mock_provider
            mock_registry.get_by_config.assert_called_once_with("default-graph-db")
    
    @pytest.mark.asyncio
    async def test_vector_provider_lazy_loading(self):
        """Test vector provider lazy loading."""
        context = FlowContext()
        mock_provider = MockProvider("vector")
        
        with patch('flowlib.agent.core.context.provider_registry') as mock_registry:
            mock_registry.get_by_config = AsyncMock(return_value=mock_provider)
            
            provider = await context.vector()
            assert provider is mock_provider
            mock_registry.get_by_config.assert_called_once_with("default-vector-db")
    
    @pytest.mark.asyncio
    async def test_cache_provider_lazy_loading(self):
        """Test cache provider lazy loading."""
        context = FlowContext()
        mock_provider = MockProvider("cache")
        
        with patch('flowlib.agent.core.context.provider_registry') as mock_registry:
            mock_registry.get_by_config = AsyncMock(return_value=mock_provider)
            
            provider = await context.cache()
            assert provider is mock_provider
            mock_registry.get_by_config.assert_called_once_with("default-cache")
    
    def test_get_llm_config_name_fast(self):
        """Test LLM config name for fast preference."""
        context = FlowContext(model_preference="fast")
        assert context._get_llm_config_name() == "fast-llm"
    
    def test_get_llm_config_name_quality(self):
        """Test LLM config name for quality preference."""
        context = FlowContext(model_preference="quality")
        assert context._get_llm_config_name() == "quality-llm"
    
    def test_get_llm_config_name_balanced(self):
        """Test LLM config name for balanced preference."""
        context = FlowContext(model_preference="balanced")
        assert context._get_llm_config_name() == "default-llm"
    
    def test_get_llm_config_name_unknown(self):
        """Test LLM config name for unknown preference defaults to balanced."""
        context = FlowContext(model_preference="unknown")
        assert context._get_llm_config_name() == "default-llm"


class TestProcessingOptions:
    """Test ProcessingOptions functionality."""
    
    def test_initialization_defaults(self):
        """Test ProcessingOptions initialization with defaults."""
        options = ProcessingOptions()
        
        assert options.confidence_threshold == 0.7
        assert options.model_preference == "balanced"
        assert options.max_results == 10
        assert options.timeout_seconds == 60
    
    def test_initialization_with_values(self):
        """Test ProcessingOptions initialization with custom values."""
        options = ProcessingOptions(
            confidence_threshold=0.8,
            model_preference="fast",
            max_results=20,
            timeout_seconds=30
        )
        
        assert options.confidence_threshold == 0.8
        assert options.model_preference == "fast"
        assert options.max_results == 20
        assert options.timeout_seconds == 30
    
    def test_confidence_threshold_validation(self):
        """Test confidence threshold validation."""
        # Valid values
        ProcessingOptions(confidence_threshold=0.0)
        ProcessingOptions(confidence_threshold=1.0)
        ProcessingOptions(confidence_threshold=0.5)
        
        # Invalid values should raise validation error
        with pytest.raises(ValueError):
            ProcessingOptions(confidence_threshold=-0.1)
        
        with pytest.raises(ValueError):
            ProcessingOptions(confidence_threshold=1.1)
    
    def test_max_results_validation(self):
        """Test max results validation."""
        # Valid values
        ProcessingOptions(max_results=1)
        ProcessingOptions(max_results=100)
        ProcessingOptions(max_results=50)
        
        # Invalid values should raise validation error
        with pytest.raises(ValueError):
            ProcessingOptions(max_results=0)
        
        with pytest.raises(ValueError):
            ProcessingOptions(max_results=101)
    
    def test_timeout_seconds_validation(self):
        """Test timeout seconds validation."""
        # Valid values
        ProcessingOptions(timeout_seconds=1)
        ProcessingOptions(timeout_seconds=300)
        ProcessingOptions(timeout_seconds=150)
        
        # Invalid values should raise validation error
        with pytest.raises(ValueError):
            ProcessingOptions(timeout_seconds=0)
        
        with pytest.raises(ValueError):
            ProcessingOptions(timeout_seconds=301)
    
    def test_create_context(self):
        """Test creating FlowContext from ProcessingOptions."""
        options = ProcessingOptions(
            confidence_threshold=0.8,
            model_preference="fast"
        )
        
        context = options.create_context()
        
        assert isinstance(context, FlowContext)
        assert context.confidence_threshold == 0.8
        assert context.model_preference == "fast"


class TestModuleFunctions:
    """Test module-level functions."""
    
    def test_get_default_processing_options(self):
        """Test getting default processing options."""
        options = get_default_processing_options()
        
        assert isinstance(options, ProcessingOptions)
        assert options.confidence_threshold == 0.7
        assert options.model_preference == "balanced"
        assert options.max_results == 10
        assert options.timeout_seconds == 60
    
    @pytest.mark.asyncio
    async def test_create_flow_context_with_options(self):
        """Test creating flow context with processing options."""
        options = ProcessingOptions(
            confidence_threshold=0.9,
            model_preference="quality"
        )
        
        context = await create_flow_context(options)
        
        assert isinstance(context, FlowContext)
        assert context.confidence_threshold == 0.9
        assert context.model_preference == "quality"
    
    @pytest.mark.asyncio
    async def test_create_flow_context_without_options(self):
        """Test creating flow context without processing options."""
        context = await create_flow_context()
        
        assert isinstance(context, FlowContext)
        assert context.confidence_threshold == 0.7  # Default
        assert context.model_preference == "balanced"  # Default
    
    def test_default_context_exists(self):
        """Test that DEFAULT_CONTEXT is available."""
        assert DEFAULT_CONTEXT is not None
        assert isinstance(DEFAULT_CONTEXT, FlowContext)
        assert DEFAULT_CONTEXT.confidence_threshold == 0.7
        assert DEFAULT_CONTEXT.model_preference == "balanced"


class TestFlowProviderProtocol:
    """Test FlowProvider protocol."""
    
    def test_mock_provider_implements_protocol(self):
        """Test that our mock provider implements the protocol."""
        provider = MockProvider()
        
        # Should have initialize method
        assert hasattr(provider, 'initialize')
        assert callable(provider.initialize)
    
    @pytest.mark.asyncio
    async def test_mock_provider_initialize(self):
        """Test mock provider initialization."""
        provider = MockProvider()
        
        assert not provider.initialized
        await provider.initialize()
        assert provider.initialized


class TestContextIntegration:
    """Test integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_multiple_provider_access(self):
        """Test accessing multiple providers in sequence."""
        context = FlowContext()
        
        mock_llm = MockProvider("llm")
        mock_graph = MockProvider("graph")
        mock_vector = MockProvider("vector")
        mock_cache = MockProvider("cache")
        
        with patch('flowlib.agent.core.context.provider_registry') as mock_registry:
            mock_registry.get_by_config = AsyncMock(side_effect=[
                mock_llm, mock_graph, mock_vector, mock_cache
            ])
            
            # Access all providers
            llm = await context.llm()
            graph = await context.graph()
            vector = await context.vector()
            cache = await context.cache()
            
            # Verify all providers are different instances
            assert llm is mock_llm
            assert graph is mock_graph
            assert vector is mock_vector
            assert cache is mock_cache
            
            # Verify correct config names were used
            expected_calls = [
                (("default-llm",),),
                (("default-graph-db",),),
                (("default-vector-db",),),
                (("default-cache",),)
            ]
            actual_calls = mock_registry.get_by_config.call_args_list
            assert len(actual_calls) == 4
    
    @pytest.mark.asyncio
    async def test_context_from_options_end_to_end(self):
        """Test complete flow from options to provider access."""
        options = ProcessingOptions(
            confidence_threshold=0.95,
            model_preference="quality",
            max_results=5,
            timeout_seconds=120
        )
        
        context = await create_flow_context(options)
        mock_provider = MockProvider("quality-llm")
        
        with patch('flowlib.agent.core.context.provider_registry') as mock_registry:
            mock_registry.get_by_config = AsyncMock(return_value=mock_provider)
            
            provider = await context.llm()
            
            assert provider is mock_provider
            assert context.confidence_threshold == 0.95
            mock_registry.get_by_config.assert_called_once_with("quality-llm")