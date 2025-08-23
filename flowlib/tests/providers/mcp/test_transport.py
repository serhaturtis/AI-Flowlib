"""Comprehensive tests for MCP transport module."""

import pytest
import json
import asyncio
from typing import Optional, Dict, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import tempfile
import subprocess

from flowlib.providers.mcp.transport import (
    StdioTransport,
    SSETransport,
    WebSocketTransport,
    create_transport
)
from flowlib.providers.mcp.base import MCPMessage, MCPTransport, MCPConnectionError


# Test helper classes and fixtures
@pytest.fixture
def sample_mcp_message():
    """Sample MCP message for testing."""
    return MCPMessage(
        id="test_123",
        type="request",
        method="test_method",
        params={"param1": "value1", "param2": 42}
    )


class MockProcess:
    """Mock subprocess for testing STDIO transport."""
    
    def __init__(self):
        self.stdin = AsyncMock()
        self.stdout = AsyncMock()
        self.stderr = AsyncMock()
        self._terminated = False
        self._return_code = 0
        self.terminate = Mock()
    
    async def wait(self):
        """Mock wait method."""
        return self._return_code
    
    async def communicate(self):
        """Mock communicate method."""
        return b"", b""


class TestStdioTransport:
    """Test StdioTransport class."""
    
    def test_stdio_transport_creation(self):
        """Test creating STDIO transport instance."""
        transport = StdioTransport("test_command", ["arg1", "arg2"])
        
        assert transport.server_command == "test_command"
        assert transport.server_args == ["arg1", "arg2"]
        assert transport.process is None
        assert transport._closed is False
    
    def test_stdio_transport_creation_no_args(self):
        """Test creating STDIO transport with no arguments."""
        transport = StdioTransport("test_command")
        
        assert transport.server_command == "test_command"
        assert transport.server_args == []
    
    @pytest.mark.asyncio
    async def test_stdio_transport_connect_success(self):
        """Test successful STDIO transport connection."""
        transport = StdioTransport("test_command")
        mock_process = MockProcess()
        
        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_process
            
            await transport.connect()
            
            assert transport.process == mock_process
            mock_create.assert_called_once_with(
                "test_command",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
    
    @pytest.mark.asyncio
    async def test_stdio_transport_connect_with_args(self):
        """Test STDIO transport connection with arguments."""
        transport = StdioTransport("test_command", ["arg1", "arg2"])
        mock_process = MockProcess()
        
        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_process
            
            await transport.connect()
            
            mock_create.assert_called_once_with(
                "test_command",
                "arg1",
                "arg2",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
    
    @pytest.mark.asyncio
    async def test_stdio_transport_connect_failure(self):
        """Test STDIO transport connection failure."""
        transport = StdioTransport("nonexistent_command")
        
        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = FileNotFoundError("Command not found")
            
            with pytest.raises(MCPConnectionError) as exc_info:
                await transport.connect()
            
            assert "Failed to start MCP server process" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_stdio_transport_send_success(self, sample_mcp_message):
        """Test successful message sending via STDIO."""
        transport = StdioTransport("test_command")
        mock_process = MockProcess()
        mock_process.stdin.drain = AsyncMock()
        transport.process = mock_process
        
        await transport.send(sample_mcp_message)
        
        # Verify stdin write was called
        mock_process.stdin.write.assert_called_once()
        mock_process.stdin.drain.assert_called_once()
        
        # Check the written data
        written_data = mock_process.stdin.write.call_args[0][0]
        written_str = written_data.decode()
        assert written_str.endswith('\n')
        
        # Parse and verify JSON
        json_data = json.loads(written_str.strip())
        assert json_data["id"] == "test_123"
        assert json_data["method"] == "test_method"
    
    @pytest.mark.asyncio
    async def test_stdio_transport_send_closed_connection(self, sample_mcp_message):
        """Test sending message with closed connection."""
        transport = StdioTransport("test_command")
        transport._closed = True
        
        with pytest.raises(MCPConnectionError) as exc_info:
            await transport.send(sample_mcp_message)
        
        assert "Connection closed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_stdio_transport_send_no_process(self, sample_mcp_message):
        """Test sending message with no process."""
        transport = StdioTransport("test_command")
        transport.process = None
        
        with pytest.raises(MCPConnectionError) as exc_info:
            await transport.send(sample_mcp_message)
        
        assert "Connection closed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_stdio_transport_send_error(self, sample_mcp_message):
        """Test sending message with write error."""
        transport = StdioTransport("test_command")
        mock_process = MockProcess()
        # Make write fail when called
        mock_process.stdin.write = Mock(side_effect=BrokenPipeError("Pipe broken"))
        mock_process.stdin.drain = AsyncMock()
        transport.process = mock_process
        
        # The error happens on write
        with pytest.raises(MCPConnectionError) as exc_info:
            await transport.send(sample_mcp_message)
        
        assert "Failed to send message" in str(exc_info.value)
        # Verify write was attempted but failed
        mock_process.stdin.write.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stdio_transport_receive_success(self):
        """Test successful message receiving via STDIO."""
        transport = StdioTransport("test_command")
        mock_process = MockProcess()
        
        # Mock response data
        response_data = {
            "id": "response_123",
            "type": "response",
            "result": {"status": "success"}
        }
        response_line = json.dumps(response_data) + "\n"
        mock_process.stdout.readline.return_value = response_line.encode()
        transport.process = mock_process
        
        message = await transport.receive()
        
        assert isinstance(message, MCPMessage)
        assert message.id == "response_123"
        assert message.result == {"status": "success"}
    
    @pytest.mark.asyncio
    async def test_stdio_transport_receive_empty_line(self):
        """Test receiving with empty line (process ended)."""
        transport = StdioTransport("test_command")
        mock_process = MockProcess()
        mock_process.stdout.readline.return_value = b""
        transport.process = mock_process
        
        with pytest.raises(MCPConnectionError) as exc_info:
            await transport.receive()
        
        assert "Server process ended" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_stdio_transport_receive_invalid_json(self):
        """Test receiving invalid JSON."""
        transport = StdioTransport("test_command")
        mock_process = MockProcess()
        mock_process.stdout.readline.return_value = b"invalid json\n"
        transport.process = mock_process
        
        with pytest.raises(MCPConnectionError) as exc_info:
            await transport.receive()
        
        assert "Invalid JSON received" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_stdio_transport_receive_closed_connection(self):
        """Test receiving with closed connection."""
        transport = StdioTransport("test_command")
        transport._closed = True
        
        with pytest.raises(MCPConnectionError) as exc_info:
            await transport.receive()
        
        assert "Connection closed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_stdio_transport_close_graceful(self):
        """Test graceful close of STDIO transport."""
        transport = StdioTransport("test_command")
        mock_process = MockProcess()
        transport.process = mock_process
        
        # Mock stdin close and wait
        mock_process.stdin.wait_closed = AsyncMock()
        
        await transport.close()
        
        assert transport._closed is True
        assert transport.process is None
        mock_process.stdin.close.assert_called_once()
        mock_process.stdin.wait_closed.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stdio_transport_close_force_terminate(self):
        """Test force termination on close timeout."""
        transport = StdioTransport("test_command")
        mock_process = MockProcess()
        transport.process = mock_process
        
        # Mock stdin close
        mock_process.stdin.wait_closed = AsyncMock()
        
        # Mock wait to timeout  
        async def slow_wait():
            await asyncio.sleep(10)
            return 0
            
        mock_process.wait = AsyncMock(side_effect=slow_wait)
        
        with patch('asyncio.wait_for') as mock_wait_for:
            # First call should timeout, second should succeed
            mock_wait_for.side_effect = [asyncio.TimeoutError(), 0]
            await transport.close()
        
        mock_process.terminate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stdio_transport_close_already_closed(self):
        """Test closing already closed transport."""
        transport = StdioTransport("test_command")
        transport._closed = True
        
        # Should not raise any exception
        await transport.close()
    
    @pytest.mark.asyncio
    async def test_stdio_transport_close_with_errors(self):
        """Test close with errors."""
        transport = StdioTransport("test_command")
        mock_process = MockProcess()
        mock_process.stdin.close.side_effect = Exception("Close error")
        transport.process = mock_process
        
        # Should not raise exception, just log warning
        await transport.close()
        
        assert transport._closed is True
        assert transport.process is None


class TestSSETransport:
    """Test SSETransport class."""
    
    def test_sse_transport_creation(self):
        """Test creating SSE transport instance."""
        headers = {"Custom-Header": "value"}
        transport = SSETransport(
            "https://example.com",
            auth_token="test_token",
            headers=headers
        )
        
        assert transport.server_url == "https://example.com"
        assert transport.auth_token == "test_token"
        assert transport.headers == headers
        assert transport._session is None
        assert transport._closed is False
    
    def test_sse_transport_creation_defaults(self):
        """Test creating SSE transport with defaults."""
        transport = SSETransport("https://example.com")
        
        assert transport.server_url == "https://example.com"
        assert transport.auth_token is None
        assert transport.headers == {}
    
    @pytest.mark.asyncio
    async def test_sse_transport_connect_success(self):
        """Test successful SSE transport connection."""
        transport = SSETransport("https://example.com", auth_token="test_token")
        
        mock_session = Mock()
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with patch('asyncio.create_task') as mock_task:
                await transport.connect()
                
                assert transport._session == mock_session
                mock_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sse_transport_connect_no_aiohttp(self):
        """Test SSE transport connection without aiohttp."""
        transport = SSETransport("https://example.com")
        
        with patch('aiohttp.ClientSession', side_effect=ImportError):
            with pytest.raises(MCPConnectionError) as exc_info:
                await transport.connect()
            
            assert "aiohttp is required for SSE transport" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_sse_transport_connect_error(self):
        """Test SSE transport connection error."""
        transport = SSETransport("https://example.com")
        
        with patch('aiohttp.ClientSession', side_effect=Exception("Connection failed")):
            with pytest.raises(MCPConnectionError) as exc_info:
                await transport.connect()
            
            assert "Failed to connect to SSE server" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_sse_transport_send_success(self, sample_mcp_message):
        """Test successful message sending via SSE."""
        transport = SSETransport("https://example.com", auth_token="test_token")
        
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.post.return_value = mock_response
        transport._session = mock_session
        
        await transport.send(sample_mcp_message)
        
        # Verify POST was called with correct parameters
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "https://example.com/messages"
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer test_token"
        assert call_args.kwargs["headers"]["Content-Type"] == "application/json"
    
    @pytest.mark.asyncio
    async def test_sse_transport_send_http_error(self, sample_mcp_message):
        """Test sending message with HTTP error."""
        transport = SSETransport("https://example.com")
        
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.post.return_value = mock_response
        transport._session = mock_session
        
        with pytest.raises(MCPConnectionError) as exc_info:
            await transport.send(sample_mcp_message)
        
        assert "HTTP 500" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_sse_transport_send_closed_connection(self, sample_mcp_message):
        """Test sending message with closed connection."""
        transport = SSETransport("https://example.com")
        transport._closed = True
        
        with pytest.raises(MCPConnectionError) as exc_info:
            await transport.send(sample_mcp_message)
        
        assert "Connection closed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_sse_transport_receive_success(self):
        """Test successful message receiving via SSE."""
        transport = SSETransport("https://example.com")
        
        # Mock message in queue
        test_message = MCPMessage(id="test", type="request", method="test_method")
        await transport._message_queue.put(test_message)
        
        message = await transport.receive()
        
        assert message.id == test_message.id
        assert message.method == test_message.method
    
    @pytest.mark.asyncio
    async def test_sse_transport_receive_exception(self):
        """Test receiving exception from queue."""
        transport = SSETransport("https://example.com")
        
        # Put exception in queue
        test_error = MCPConnectionError("Test error")
        await transport._message_queue.put(test_error)
        
        with pytest.raises(MCPConnectionError) as exc_info:
            await transport.receive()
        
        assert "Test error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_sse_transport_receive_closed(self):
        """Test receiving with closed connection."""
        transport = SSETransport("https://example.com")
        transport._closed = True
        
        with pytest.raises(MCPConnectionError) as exc_info:
            await transport.receive()
        
        assert "Connection closed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_sse_transport_listen_for_events(self):
        """Test SSE event listening."""
        transport = SSETransport("https://example.com")
        
        # Mock response content
        mock_response = Mock()
        mock_content = Mock()
        
        # Simulate SSE data
        sse_lines = [
            b"data: " + json.dumps({"id": "test1", "method": "method1"}).encode() + b"\n",
            b"data: " + json.dumps({"id": "test2", "method": "method2"}).encode() + b"\n"
        ]
        
        async def mock_content_iter():
            for line in sse_lines:
                yield line
        
        mock_content.__aiter__ = mock_content_iter
        mock_response.content = mock_content
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        headers = {"Accept": "text/event-stream"}
        
        # Mock session get
        mock_session = Mock()
        mock_session.get.return_value = mock_response
        transport._session = mock_session
        
        # Start listening
        listen_task = asyncio.create_task(transport._listen_for_events(headers))
        
        # Give it time to process
        await asyncio.sleep(0.1)
        listen_task.cancel()
        
        # Check messages were queued
        assert not transport._message_queue.empty()
    
    @pytest.mark.asyncio
    async def test_sse_transport_close(self):
        """Test SSE transport close."""
        transport = SSETransport("https://example.com")
        
        mock_session = Mock()
        mock_session.close = AsyncMock()
        transport._session = mock_session
        
        await transport.close()
        
        assert transport._closed is True
        assert transport._session is None
        mock_session.close.assert_called_once()


class TestWebSocketTransport:
    """Test WebSocketTransport class."""
    
    def test_websocket_transport_creation(self):
        """Test creating WebSocket transport instance."""
        headers = {"Custom-Header": "value"}
        transport = WebSocketTransport(
            "wss://example.com",
            auth_token="test_token",
            headers=headers
        )
        
        assert transport.server_url == "wss://example.com"
        assert transport.auth_token == "test_token"
        assert transport.headers == headers
        assert transport._websocket is None
        assert transport._closed is False
    
    @pytest.mark.asyncio
    async def test_websocket_transport_connect_success(self):
        """Test successful WebSocket transport connection."""
        transport = WebSocketTransport("https://example.com", auth_token="test_token")
        
        mock_websocket = Mock()
        
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = mock_websocket
            
            await transport.connect()
            
            assert transport._websocket == mock_websocket
            mock_connect.assert_called_once()
            
            # Check URL conversion
            call_args = mock_connect.call_args
            assert call_args[0][0] == "wss://example.com"
            assert call_args.kwargs["extra_headers"]["Authorization"] == "Bearer test_token"
    
    @pytest.mark.asyncio
    async def test_websocket_transport_connect_url_conversion(self):
        """Test WebSocket URL conversion."""
        test_cases = [
            ("http://example.com", "ws://example.com"),
            ("https://example.com", "wss://example.com"),
            ("ws://example.com", "ws://example.com"),
            ("wss://example.com", "wss://example.com")
        ]
        
        for input_url, expected_url in test_cases:
            transport = WebSocketTransport(input_url)
            
            with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
                await transport.connect()
                
                call_args = mock_connect.call_args
                assert call_args[0][0] == expected_url
    
    @pytest.mark.asyncio
    async def test_websocket_transport_connect_no_websockets(self):
        """Test WebSocket transport connection without websockets library."""
        transport = WebSocketTransport("wss://example.com")
        
        with patch('websockets.connect', side_effect=ImportError):
            with pytest.raises(MCPConnectionError) as exc_info:
                await transport.connect()
            
            assert "websockets is required for WebSocket transport" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_websocket_transport_connect_error(self):
        """Test WebSocket transport connection error."""
        transport = WebSocketTransport("wss://example.com")
        
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")
            
            with pytest.raises(MCPConnectionError) as exc_info:
                await transport.connect()
            
            assert "Failed to connect to WebSocket server" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_websocket_transport_send_success(self, sample_mcp_message):
        """Test successful message sending via WebSocket."""
        transport = WebSocketTransport("wss://example.com")
        
        mock_websocket = Mock()
        mock_websocket.send = AsyncMock()
        transport._websocket = mock_websocket
        
        await transport.send(sample_mcp_message)
        
        mock_websocket.send.assert_called_once()
        sent_data = mock_websocket.send.call_args[0][0]
        
        # Parse and verify JSON
        json_data = json.loads(sent_data)
        assert json_data["id"] == "test_123"
        assert json_data["method"] == "test_method"
    
    @pytest.mark.asyncio
    async def test_websocket_transport_send_closed_connection(self, sample_mcp_message):
        """Test sending message with closed connection."""
        transport = WebSocketTransport("wss://example.com")
        transport._closed = True
        
        with pytest.raises(MCPConnectionError) as exc_info:
            await transport.send(sample_mcp_message)
        
        assert "Connection closed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_websocket_transport_send_error(self, sample_mcp_message):
        """Test sending message with WebSocket error."""
        transport = WebSocketTransport("wss://example.com")
        
        mock_websocket = Mock()
        mock_websocket.send = AsyncMock(side_effect=Exception("Send failed"))
        transport._websocket = mock_websocket
        
        with pytest.raises(MCPConnectionError) as exc_info:
            await transport.send(sample_mcp_message)
        
        assert "Failed to send message" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_websocket_transport_receive_success(self):
        """Test successful message receiving via WebSocket."""
        transport = WebSocketTransport("wss://example.com")
        
        mock_websocket = Mock()
        response_data = {"id": "response_123", "type": "response", "result": {"status": "success"}}
        mock_websocket.recv = AsyncMock(return_value=json.dumps(response_data))
        transport._websocket = mock_websocket
        
        message = await transport.receive()
        
        assert isinstance(message, MCPMessage)
        assert message.id == "response_123"
        assert message.result == {"status": "success"}
    
    @pytest.mark.asyncio
    async def test_websocket_transport_receive_invalid_json(self):
        """Test receiving invalid JSON via WebSocket."""
        transport = WebSocketTransport("wss://example.com")
        
        mock_websocket = Mock()
        mock_websocket.recv = AsyncMock(return_value="invalid json")
        transport._websocket = mock_websocket
        
        with pytest.raises(MCPConnectionError) as exc_info:
            await transport.receive()
        
        assert "Invalid JSON received" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_websocket_transport_receive_closed_connection(self):
        """Test receiving with closed connection."""
        transport = WebSocketTransport("wss://example.com")
        transport._closed = True
        
        with pytest.raises(MCPConnectionError) as exc_info:
            await transport.receive()
        
        assert "Connection closed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_websocket_transport_close(self):
        """Test WebSocket transport close."""
        transport = WebSocketTransport("wss://example.com")
        
        mock_websocket = Mock()
        mock_websocket.close = AsyncMock()
        transport._websocket = mock_websocket
        
        await transport.close()
        
        assert transport._closed is True
        assert transport._websocket is None
        mock_websocket.close.assert_called_once()


class TestCreateTransport:
    """Test create_transport factory function."""
    
    @pytest.mark.asyncio
    async def test_create_stdio_transport(self):
        """Test creating STDIO transport."""
        with patch('flowlib.providers.mcp.transport.StdioTransport') as mock_stdio:
            mock_instance = Mock()
            mock_instance.connect = AsyncMock()
            mock_stdio.return_value = mock_instance
            
            transport = await create_transport(
                MCPTransport.STDIO,
                "dummy_uri",
                server_command="test_command",
                server_args=["arg1", "arg2"],
                timeout=10.0
            )
            
            assert transport == mock_instance
            mock_stdio.assert_called_once_with("test_command", ["arg1", "arg2"])
            mock_instance.connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_stdio_transport_missing_command(self):
        """Test creating STDIO transport without server command."""
        with pytest.raises(MCPConnectionError) as exc_info:
            await create_transport(
                MCPTransport.STDIO,
                "dummy_uri"
            )
        
        assert "server_command required for STDIO transport" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_create_sse_transport(self):
        """Test creating SSE transport."""
        with patch('flowlib.providers.mcp.transport.SSETransport') as mock_sse:
            mock_instance = Mock()
            mock_instance.connect = AsyncMock()
            mock_sse.return_value = mock_instance
            
            transport = await create_transport(
                MCPTransport.SSE,
                "https://example.com",
                auth_token="test_token",
                headers={"Custom": "Header"}
            )
            
            assert transport == mock_instance
            mock_sse.assert_called_once_with(
                "https://example.com",
                "test_token", 
                {"Custom": "Header"}
            )
            mock_instance.connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_websocket_transport(self):
        """Test creating WebSocket transport."""
        with patch('flowlib.providers.mcp.transport.WebSocketTransport') as mock_ws:
            mock_instance = Mock()
            mock_instance.connect = AsyncMock()
            mock_ws.return_value = mock_instance
            
            transport = await create_transport(
                MCPTransport.WEBSOCKET,
                "wss://example.com",
                auth_token="test_token",
                headers={"Custom": "Header"}
            )
            
            assert transport == mock_instance
            mock_ws.assert_called_once_with(
                "wss://example.com",
                "test_token",
                {"Custom": "Header"}
            )
            mock_instance.connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_transport_unsupported_type(self):
        """Test creating transport with unsupported type."""
        with pytest.raises(MCPConnectionError) as exc_info:
            await create_transport(
                "INVALID_TRANSPORT",
                "dummy_uri"
            )
        
        assert "Unsupported transport type" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_create_transport_timeout(self):
        """Test transport creation with timeout."""
        async def slow_connect():
            await asyncio.sleep(10)
            
        with patch('flowlib.providers.mcp.transport.StdioTransport') as mock_stdio:
            mock_instance = Mock()
            mock_instance.connect = AsyncMock(side_effect=slow_connect)
            mock_instance.close = AsyncMock()
            mock_stdio.return_value = mock_instance
            
            with pytest.raises(MCPConnectionError) as exc_info:
                await create_transport(
                    MCPTransport.STDIO,
                    "dummy_uri",
                    server_command="test_command",
                    timeout=0.1  # Very short timeout
                )
            
            assert "Connection timeout" in str(exc_info.value)
            mock_instance.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_transport_connection_error(self):
        """Test transport creation with connection error."""
        with patch('flowlib.providers.mcp.transport.StdioTransport') as mock_stdio:
            mock_instance = Mock()
            mock_instance.connect = AsyncMock(side_effect=Exception("Connection failed"))
            mock_instance.close = AsyncMock()
            mock_stdio.return_value = mock_instance
            
            with pytest.raises(Exception):  # Original exception should be raised
                await create_transport(
                    MCPTransport.STDIO,
                    "dummy_uri", 
                    server_command="test_command"
                )
            
            mock_instance.close.assert_called_once()


class TestMCPTransportIntegration:
    """Test integration aspects of MCP transport."""
    
    @pytest.mark.asyncio
    async def test_stdio_transport_full_workflow(self):
        """Test complete STDIO transport workflow."""
        transport = StdioTransport("echo", ["test"])
        
        # Mock process
        mock_process = MockProcess()
        
        # Mock message data
        request_message = MCPMessage(id="req1", type="request", method="test_method", params={"test": True})
        response_data = {"id": "req1", "type": "response", "result": {"success": True}}
        response_line = json.dumps(response_data) + "\n"
        
        mock_process.stdout.readline.return_value = response_line.encode()
        
        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_process
            
            # Connect
            await transport.connect()
            assert transport.process == mock_process
            
            # Send message
            await transport.send(request_message)
            mock_process.stdin.write.assert_called_once()
            
            # Receive response
            response = await transport.receive()
            assert response.id == "req1"
            assert response.result == {"success": True}
            
            # Close
            mock_process.stdin.wait_closed = AsyncMock()
            await transport.close()
            assert transport._closed is True
    
    @pytest.mark.asyncio
    async def test_websocket_transport_concurrent_operations(self):
        """Test WebSocket transport with concurrent send/receive."""
        transport = WebSocketTransport("wss://example.com")
        
        mock_websocket = Mock()
        mock_websocket.send = AsyncMock()
        mock_websocket.recv = AsyncMock()
        transport._websocket = mock_websocket
        
        # Simulate concurrent operations
        message1 = MCPMessage(id="msg1", type="request", method="method1")
        message2 = MCPMessage(id="msg2", type="request", method="method2")
        
        # Mock responses
        response1 = {"id": "msg1", "type": "response", "result": {"data": "response1"}}
        response2 = {"id": "msg2", "type": "response", "result": {"data": "response2"}}
        
        mock_websocket.recv.side_effect = [
            json.dumps(response1),
            json.dumps(response2)
        ]
        
        # Send and receive concurrently
        send_task1 = asyncio.create_task(transport.send(message1))
        send_task2 = asyncio.create_task(transport.send(message2))
        recv_task1 = asyncio.create_task(transport.receive())
        recv_task2 = asyncio.create_task(transport.receive())
        
        # Wait for all operations
        await asyncio.gather(send_task1, send_task2, recv_task1, recv_task2)
        
        # Verify sends were called
        assert mock_websocket.send.call_count == 2
        
        # Verify receives got correct data
        result1 = recv_task1.result()
        result2 = recv_task2.result()
        assert result1.id == "msg1"
        assert result2.id == "msg2"
    
    def test_transport_error_handling_consistency(self):
        """Test that all transports handle errors consistently."""
        transports = [
            StdioTransport("test_command"),
            SSETransport("https://example.com"),
            WebSocketTransport("wss://example.com")
        ]
        
        for transport in transports:
            # All should raise MCPConnectionError for closed connections
            transport._closed = True
            
            with pytest.raises(MCPConnectionError):
                asyncio.run(transport.send(MCPMessage(id="test", type="request", method="test")))
            
            with pytest.raises(MCPConnectionError):
                asyncio.run(transport.receive())
    
    @pytest.mark.asyncio
    async def test_transport_message_serialization_consistency(self):
        """Test that all transports serialize messages consistently."""
        message = MCPMessage(
            id="test_123",
            type="request",
            method="test_method",
            params={"param1": "value1", "param2": 42, "param3": None}
        )
        
        # Test STDIO serialization
        stdio_transport = StdioTransport("test_command")
        mock_process = MockProcess()
        mock_process.stdin.drain = AsyncMock()
        stdio_transport.process = mock_process
        
        await stdio_transport.send(message)
        written_data = mock_process.stdin.write.call_args[0][0]
        stdio_json = json.loads(written_data.decode().strip())
        
        # Test WebSocket serialization
        ws_transport = WebSocketTransport("wss://example.com")
        mock_websocket = Mock()
        mock_websocket.send = AsyncMock()
        ws_transport._websocket = mock_websocket
        
        await ws_transport.send(message)
        ws_data = mock_websocket.send.call_args[0][0]
        ws_json = json.loads(ws_data)
        
        # Both should produce equivalent JSON
        assert stdio_json == ws_json
        assert stdio_json["id"] == "test_123"
        assert stdio_json["method"] == "test_method"
        assert stdio_json["params"]["param1"] == "value1"
        assert stdio_json["params"]["param2"] == 42
        # param3 should be None as explicitly set in model
        assert stdio_json["params"]["param3"] is None