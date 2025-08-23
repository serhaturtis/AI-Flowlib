"""Tests for MCP-aware flow decorators."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any
from pydantic import BaseModel

from flowlib.flows.decorators.mcp_decorators import (
    mcp_tool,
    mcp_resource,
    mcp_client_aware,
    get_mcp_metadata,
    is_mcp_tool,
    is_mcp_resource
)
from flowlib.flows.decorators.decorators import pipeline


class InputModel(BaseModel):
    text: str
    option: bool = False


class OutputModel(BaseModel):
    result: str
    confidence: float = 0.5


class TestMcpToolDecorator:
    """Test mcp_tool decorator functionality."""
    
    def test_mcp_tool_basic_decoration(self):
        """Test basic mcp_tool decorator usage."""
        @mcp_tool()
        class TestToolFlow:
            @pipeline(input_model=InputModel, output_model=OutputModel)
            async def run_pipeline(self, input_data: InputModel) -> OutputModel:
                return OutputModel(result="tool executed")
        
        # Should have MCP metadata
        assert hasattr(TestToolFlow, '__mcp_metadata__')
        assert TestToolFlow.__mcp_metadata__['expose_as_tool'] is True
    
    def test_mcp_tool_with_parameters(self):
        """Test mcp_tool decorator with parameters."""
        @mcp_tool(
            name="test_tool",
            description="Test tool description",
            input_schema={"type": "object"},
            expose_by_default=False
        )
        class TestToolFlow:
            @pipeline(input_model=InputModel, output_model=OutputModel)
            async def run_pipeline(self, input_data: InputModel) -> OutputModel:
                return OutputModel(result="tool executed")
        
        metadata = TestToolFlow.__mcp_metadata__
        assert metadata['expose_as_tool'] is True
        assert metadata['tool_name'] == "test_tool"
        assert metadata['tool_description'] == "Test tool description"
        assert metadata['input_schema'] == {"type": "object"}
        assert metadata['expose_by_default'] is False
    
    def test_mcp_tool_default_parameters(self):
        """Test mcp_tool decorator with default parameters."""
        @mcp_tool()
        class TestToolFlow:
            @pipeline(input_model=InputModel, output_model=OutputModel)
            async def run_pipeline(self, input_data: InputModel) -> OutputModel:
                return OutputModel(result="tool executed")
        
        metadata = TestToolFlow.__mcp_metadata__
        assert metadata['expose_as_tool'] is True
        assert metadata['tool_name'] is None
        assert metadata['tool_description'] == ""
        assert metadata['input_schema'] is None
        assert metadata['expose_by_default'] is True
    
    @patch('flowlib.flows.decorators.mcp_decorators.flow')
    def test_mcp_tool_applies_flow_decorator(self, mock_flow):
        """Test that mcp_tool applies flow decorator first."""
        mock_flow.return_value = lambda cls: cls
        
        @mcp_tool(name="test", description="test desc")
        class TestToolFlow:
            @pipeline(input_model=InputModel, output_model=OutputModel)
            async def run_pipeline(self, input_data: InputModel) -> OutputModel:
                return OutputModel(result="tool executed")
        
        # Verify flow decorator was called
        mock_flow.assert_called_once_with(name="test", description="test desc", is_infrastructure=True)


class TestMcpResourceDecorator:
    """Test mcp_resource decorator functionality."""
    
    def test_mcp_resource_decoration(self):
        """Test mcp_resource decorator usage."""
        @mcp_resource(
            uri="memory://test_resource",
            name="Test Resource",
            description="Test resource description",
            mime_type="application/json"
        )
        class TestResourceFlow:
            @pipeline(input_model=InputModel, output_model=OutputModel)
            async def run_pipeline(self, input_data: InputModel) -> OutputModel:
                return OutputModel(result="resource accessed")
        
        # Should have MCP resource metadata
        assert hasattr(TestResourceFlow, '__mcp_resource_metadata__')
        metadata = TestResourceFlow.__mcp_resource_metadata__
        
        assert metadata['expose_as_resource'] is True
        assert metadata['resource_uri'] == "memory://test_resource"
        assert metadata['resource_name'] == "Test Resource"
        assert metadata['resource_description'] == "Test resource description"
        assert metadata['mime_type'] == "application/json"
    
    def test_mcp_resource_default_mime_type(self):
        """Test mcp_resource decorator with default mime type."""
        @mcp_resource(
            uri="test://resource",
            name="Test Resource"
        )
        class TestResourceFlow:
            @pipeline(input_model=InputModel, output_model=OutputModel)
            async def run_pipeline(self, input_data: InputModel) -> OutputModel:
                return OutputModel(result="resource accessed")
        
        metadata = TestResourceFlow.__mcp_resource_metadata__
        assert metadata['mime_type'] == "application/json"
    
    @patch('flowlib.flows.decorators.mcp_decorators.flow')
    def test_mcp_resource_applies_flow_decorator(self, mock_flow):
        """Test that mcp_resource applies flow decorator first."""
        mock_flow.return_value = lambda cls: cls
        
        @mcp_resource(
            uri="test://resource",
            name="Test Resource"
        )
        class TestResourceFlow:
            @pipeline(input_model=InputModel, output_model=OutputModel)
            async def run_pipeline(self, input_data: InputModel) -> OutputModel:
                return OutputModel(result="resource accessed")
        
        # Verify flow decorator was called
        mock_flow.assert_called_once_with(description="", is_infrastructure=True)


class TestMcpClientAwareDecorator:
    """Test mcp_client_aware decorator functionality."""
    
    def test_mcp_client_aware_decoration(self):
        """Test mcp_client_aware decorator usage."""
        @mcp_client_aware("test_client")
        class TestClientFlow:
            def __init__(self):
                pass
        
        flow_instance = TestClientFlow()
        
        # Should have MCP client attributes
        assert hasattr(flow_instance, '_mcp_client_name')
        assert hasattr(flow_instance, '_mcp_client')
        assert flow_instance._mcp_client_name == "test_client"
        assert flow_instance._mcp_client is None
        
        # Should have MCP methods
        assert hasattr(flow_instance, 'get_mcp_client')
        assert hasattr(flow_instance, 'call_mcp_tool')
        assert hasattr(flow_instance, 'read_mcp_resource')
    
    def test_mcp_client_aware_preserves_init(self):
        """Test that mcp_client_aware preserves original __init__."""
        @mcp_client_aware("test_client")
        class TestClientFlow:
            def __init__(self, value):
                self.value = value
        
        flow_instance = TestClientFlow("test_value")
        
        assert flow_instance.value == "test_value"
        assert flow_instance._mcp_client_name == "test_client"
    
    @pytest.mark.asyncio
    async def test_get_mcp_client_success(self):
        """Test successful MCP client retrieval."""
        @mcp_client_aware("test_client")
        class TestClientFlow:
            def __init__(self):
                pass
        
        flow_instance = TestClientFlow()
        mock_client = Mock()
        
        with patch('flowlib.providers.core.registry.provider_registry') as mock_registry:
            mock_registry.get_by_config = AsyncMock(return_value=mock_client)
            
            client = await flow_instance.get_mcp_client()
            
            assert client == mock_client
            assert flow_instance._mcp_client == mock_client
            mock_registry.get_by_config.assert_called_once_with("test_client")
    
    @pytest.mark.asyncio
    async def test_get_mcp_client_cached(self):
        """Test that MCP client is cached after first retrieval."""
        @mcp_client_aware("test_client")
        class TestClientFlow:
            def __init__(self):
                pass
        
        flow_instance = TestClientFlow()
        mock_client = Mock()
        flow_instance._mcp_client = mock_client  # Pre-cache
        
        with patch('flowlib.providers.core.registry.provider_registry') as mock_registry:
            mock_registry.get_by_config = AsyncMock()
            
            client = await flow_instance.get_mcp_client()
            
            assert client == mock_client
            mock_registry.get_by_config.assert_not_called()  # Should not call registry
    
    @pytest.mark.asyncio
    async def test_get_mcp_client_failure(self):
        """Test MCP client retrieval failure."""
        @mcp_client_aware("test_client")
        class TestClientFlow:
            def __init__(self):
                pass
        
        flow_instance = TestClientFlow()
        
        with patch('flowlib.providers.core.registry.provider_registry') as mock_registry, \
             patch('flowlib.flows.decorators.mcp_decorators.logger') as mock_logger:
            
            mock_registry.get_by_config = AsyncMock(side_effect=Exception("Client not found"))
            
            client = await flow_instance.get_mcp_client()
            
            assert client is None
            mock_logger.warning.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_call_mcp_tool_success(self):
        """Test successful MCP tool call."""
        @mcp_client_aware("test_client")
        class TestClientFlow:
            def __init__(self):
                pass
        
        flow_instance = TestClientFlow()
        mock_client = Mock()
        mock_client.call_tool = AsyncMock(return_value={"result": "success"})
        
        with patch.object(flow_instance, 'get_mcp_client', return_value=mock_client):
            result = await flow_instance.call_mcp_tool("test_tool", {"arg": "value"})
            
            assert result == {"result": "success"}
            mock_client.call_tool.assert_called_once_with("test_tool", {"arg": "value"})
    
    @pytest.mark.asyncio
    async def test_call_mcp_tool_no_client(self):
        """Test MCP tool call when no client available."""
        @mcp_client_aware("test_client")
        class TestClientFlow:
            def __init__(self):
                pass
        
        flow_instance = TestClientFlow()
        
        with patch.object(flow_instance, 'get_mcp_client', return_value=None):
            with pytest.raises(RuntimeError, match="MCP client 'test_client' not available"):
                await flow_instance.call_mcp_tool("test_tool", {})
    
    @pytest.mark.asyncio
    async def test_read_mcp_resource_success(self):
        """Test successful MCP resource reading."""
        @mcp_client_aware("test_client")
        class TestClientFlow:
            def __init__(self):
                pass
        
        flow_instance = TestClientFlow()
        mock_client = Mock()
        mock_client.read_resource = AsyncMock(return_value={"content": "resource_data"})
        
        with patch.object(flow_instance, 'get_mcp_client', return_value=mock_client):
            result = await flow_instance.read_mcp_resource("test://resource")
            
            assert result == {"content": "resource_data"}
            mock_client.read_resource.assert_called_once_with("test://resource")
    
    @pytest.mark.asyncio
    async def test_read_mcp_resource_no_client(self):
        """Test MCP resource reading when no client available."""
        @mcp_client_aware("test_client")
        class TestClientFlow:
            def __init__(self):
                pass
        
        flow_instance = TestClientFlow()
        
        with patch.object(flow_instance, 'get_mcp_client', return_value=None):
            with pytest.raises(RuntimeError, match="MCP client 'test_client' not available"):
                await flow_instance.read_mcp_resource("test://resource")


class TestHelperFunctions:
    """Test helper functions for MCP metadata."""
    
    def test_get_mcp_metadata_tool_only(self):
        """Test getting metadata from tool flow."""
        @mcp_tool(name="test_tool", description="Test tool")
        class TestToolFlow:
            @pipeline(input_model=InputModel, output_model=OutputModel)
            async def run_pipeline(self, input_data: InputModel) -> OutputModel:
                return OutputModel(result="tool executed")
        
        metadata = get_mcp_metadata(TestToolFlow)
        
        assert 'expose_as_tool' in metadata
        assert 'tool_name' in metadata
        assert metadata['tool_name'] == "test_tool"
    
    def test_get_mcp_metadata_resource_only(self):
        """Test getting metadata from resource flow."""
        @mcp_resource(uri="test://resource", name="Test Resource")
        class TestResourceFlow:
            @pipeline(input_model=InputModel, output_model=OutputModel)
            async def run_pipeline(self, input_data: InputModel) -> OutputModel:
                return OutputModel(result="resource accessed")
        
        metadata = get_mcp_metadata(TestResourceFlow)
        
        assert 'expose_as_resource' in metadata
        assert 'resource_uri' in metadata
        assert metadata['resource_uri'] == "test://resource"
    
    def test_get_mcp_metadata_both(self):
        """Test getting metadata from flow with both tool and resource decorators."""
        @mcp_resource(uri="test://resource", name="Test Resource")
        @mcp_tool(name="test_tool", description="Test tool")
        class TestBothFlow:
            @pipeline(input_model=InputModel, output_model=OutputModel)
            async def run_pipeline(self, input_data: InputModel) -> OutputModel:
                return OutputModel(result="both executed")
        
        metadata = get_mcp_metadata(TestBothFlow)
        
        assert 'expose_as_tool' in metadata
        assert 'expose_as_resource' in metadata
        assert metadata['tool_name'] == "test_tool"
        assert metadata['resource_uri'] == "test://resource"
    
    def test_get_mcp_metadata_none(self):
        """Test getting metadata from flow with no MCP decorators."""
        class TestPlainFlow:
            pass
        
        metadata = get_mcp_metadata(TestPlainFlow)
        
        assert metadata == {}
    
    def test_is_mcp_tool_true(self):
        """Test is_mcp_tool with MCP tool flow."""
        @mcp_tool()
        class TestToolFlow:
            @pipeline(input_model=InputModel, output_model=OutputModel)
            async def run_pipeline(self, input_data: InputModel) -> OutputModel:
                return OutputModel(result="tool executed")
        
        assert is_mcp_tool(TestToolFlow) is True
    
    def test_is_mcp_tool_false(self):
        """Test is_mcp_tool with non-MCP flow."""
        class TestPlainFlow:
            pass
        
        assert is_mcp_tool(TestPlainFlow) is False
    
    def test_is_mcp_resource_true(self):
        """Test is_mcp_resource with MCP resource flow."""
        @mcp_resource(uri="test://resource", name="Test Resource")
        class TestResourceFlow:
            @pipeline(input_model=InputModel, output_model=OutputModel)
            async def run_pipeline(self, input_data: InputModel) -> OutputModel:
                return OutputModel(result="resource accessed")
        
        assert is_mcp_resource(TestResourceFlow) is True
    
    def test_is_mcp_resource_false(self):
        """Test is_mcp_resource with non-MCP flow."""
        class TestPlainFlow:
            pass
        
        assert is_mcp_resource(TestPlainFlow) is False


class TestExampleUsage:
    """Test the example usage patterns in the module."""
    
    def test_example_imports_available(self):
        """Test that example classes can be imported and work."""
        # This test ensures the example patterns in __main__ are syntactically correct
        # We test the patterns separately rather than executing __main__
        
        from pydantic import BaseModel
        
        class AnalysisInput(BaseModel):
            text: str
            sentiment_analysis: bool = True
            entity_extraction: bool = False
        
        class AnalysisOutput(BaseModel):
            sentiment: str
            entities: list = []
            confidence: float
        
        @mcp_tool(
            name="analyze_text",
            description="Analyze text for sentiment and entities",
            expose_by_default=True
        )
        class TextAnalysisFlow:
            @pipeline(input_model=AnalysisInput, output_model=AnalysisOutput)
            async def run_pipeline(self, input_data: AnalysisInput) -> AnalysisOutput:
                return AnalysisOutput(sentiment="positive", entities=[], confidence=0.85)
        
        @mcp_resource(
            uri="memory://recent_conversations",
            name="Recent Conversations",
            description="Access to recent conversation history"
        )
        class ConversationHistoryFlow:
            @pipeline(input_model=InputModel, output_model=OutputModel)
            async def run_pipeline(self, input_data: InputModel) -> OutputModel:
                return OutputModel(result="conversations retrieved")
        
        @mcp_client_aware("github_client")
        class GitHubIntegrationFlow:
            def __init__(self):
                pass
        
        # Verify the decorators worked
        assert is_mcp_tool(TextAnalysisFlow)
        assert is_mcp_resource(ConversationHistoryFlow)
        assert hasattr(GitHubIntegrationFlow(), '_mcp_client_name')
        
        # Verify metadata
        tool_metadata = get_mcp_metadata(TextAnalysisFlow)
        assert tool_metadata['tool_name'] == "analyze_text"
        assert tool_metadata['expose_by_default'] is True
        
        resource_metadata = get_mcp_metadata(ConversationHistoryFlow)
        assert resource_metadata['resource_uri'] == "memory://recent_conversations"
        assert resource_metadata['resource_name'] == "Recent Conversations"