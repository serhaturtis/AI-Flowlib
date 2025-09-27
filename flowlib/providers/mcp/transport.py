"""MCP transport implementations."""

import asyncio
import json
import logging
from typing import Optional, Dict, List, Union, Any

import aiohttp

from .base import MCPConnection, MCPMessage, MCPTransport, MCPConnectionError

logger = logging.getLogger(__name__)


class StdioTransport(MCPConnection):
    """STDIO transport for MCP."""
    
    def __init__(self, server_command: str, server_args: Optional[List[str]] = None):
        self.server_command = server_command
        self.server_args = server_args or []
        self.process: Optional[asyncio.subprocess.Process] = None
        self._closed = False
    
    async def connect(self) -> None:
        """Start the server process and connect."""
        try:
            self.process = await asyncio.create_subprocess_exec(
                self.server_command,
                *self.server_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            logger.debug(f"Started MCP server process: {self.server_command}")
            
        except Exception as e:
            raise MCPConnectionError(f"Failed to start MCP server process: {e}")
    
    async def send(self, message: MCPMessage) -> None:
        """Send message via stdin."""
        if self._closed or not self.process or not self.process.stdin:
            raise MCPConnectionError("Connection closed")
        
        try:
            # Serialize message to JSON
            data = message.model_dump(exclude_none=True)
            json_str = json.dumps(data) + "\n"
            
            # Send via stdin
            self.process.stdin.write(json_str.encode())
            await self.process.stdin.drain()
            
        except Exception as e:
            raise MCPConnectionError(f"Failed to send message: {e}")
    
    async def receive(self) -> MCPMessage:
        """Receive message from stdout."""
        if self._closed or not self.process or not self.process.stdout:
            raise MCPConnectionError("Connection closed")
        
        try:
            # Read line from stdout
            line = await self.process.stdout.readline()
            if not line:
                raise MCPConnectionError("Server process ended")
            
            # Parse JSON
            data = json.loads(line.decode().strip())
            return MCPMessage(**data)
            
        except json.JSONDecodeError as e:
            raise MCPConnectionError(f"Invalid JSON received: {e}")
        except Exception as e:
            raise MCPConnectionError(f"Failed to receive message: {e}")
    
    async def close(self) -> None:
        """Close the connection and terminate process."""
        if self._closed:
            return
        
        self._closed = True
        
        if self.process:
            try:
                # Close stdin
                if self.process.stdin:
                    self.process.stdin.close()
                    await self.process.stdin.wait_closed()
                
                # Wait for process to exit
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    # Force terminate if it doesn't exit gracefully
                    self.process.terminate()
                    await self.process.wait()
                
                logger.debug("MCP server process terminated")
                
            except Exception as e:
                logger.warning(f"Error closing MCP process: {e}")
            finally:
                self.process = None


class SSETransport(MCPConnection):
    """Server-Sent Events transport for MCP."""
    
    def __init__(
        self, 
        server_url: str, 
        auth_token: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        self.server_url = server_url
        self.auth_token = auth_token
        self.headers = headers or {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._event_source = None
        self._message_queue: asyncio.Queue[Union[MCPMessage, Exception]] = asyncio.Queue()
        self._closed = False
    
    async def connect(self) -> None:
        """Connect to SSE endpoint."""
        try:
            # Setup headers
            headers = self.headers.copy()
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            headers["Accept"] = "text/event-stream"
            headers["Cache-Control"] = "no-cache"
            
            # Create session and connect
            self._session = aiohttp.ClientSession()
            
            # Start listening for events
            asyncio.create_task(self._listen_for_events(headers))
            
            logger.debug(f"Connected to MCP SSE server: {self.server_url}")
            
        except ImportError:
            raise MCPConnectionError("aiohttp is required for SSE transport")
        except Exception as e:
            raise MCPConnectionError(f"Failed to connect to SSE server: {e}")
    
    async def _listen_for_events(self, headers: Dict[str, str]) -> None:
        """Listen for SSE events."""
        if not self._session:
            raise RuntimeError("Session not initialized. Call connect() first.")

        try:
            async with self._session.get(
                f"{self.server_url}/events",
                headers=headers
            ) as response:
                async for line_bytes in response.content:
                    if self._closed:
                        break

                    line = line_bytes.decode().strip()
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        try:
                            message_data = json.loads(data)
                            message = MCPMessage(**message_data)
                            await self._message_queue.put(message)
                        except Exception as e:
                            logger.warning(f"Invalid SSE message: {e}")
                            
        except Exception as e:
            if not self._closed:
                logger.error(f"Error listening for SSE events: {e}")
            await self._message_queue.put(MCPConnectionError("SSE connection lost"))
    
    async def send(self, message: MCPMessage) -> None:
        """Send message via HTTP POST."""
        if self._closed or not self._session:
            raise MCPConnectionError("Connection closed")
        
        try:
            headers = self.headers.copy()
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            headers["Content-Type"] = "application/json"
            
            data = message.model_dump(exclude_none=True)
            
            async with self._session.post(
                f"{self.server_url}/messages",
                json=data,
                headers=headers
            ) as response:
                if response.status >= 400:
                    raise MCPConnectionError(f"HTTP {response.status}: {await response.text()}")
                    
        except Exception as e:
            raise MCPConnectionError(f"Failed to send message: {e}")
    
    async def receive(self) -> MCPMessage:
        """Receive message from event queue."""
        if self._closed:
            raise MCPConnectionError("Connection closed")
        
        message = await self._message_queue.get()
        
        if isinstance(message, Exception):
            raise message
        
        return message
    
    async def close(self) -> None:
        """Close the SSE connection."""
        if self._closed:
            return
        
        self._closed = True
        
        if self._session:
            try:
                await self._session.close()
            except Exception as e:
                logger.warning(f"Error closing SSE session: {e}")
            finally:
                self._session = None


class WebSocketTransport(MCPConnection):
    """WebSocket transport for MCP."""
    
    def __init__(
        self, 
        server_url: str,
        auth_token: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        self.server_url = server_url
        self.auth_token = auth_token
        self.headers = headers or {}
        self._websocket: Optional[Any] = None
        self._closed = False
    
    async def connect(self) -> None:
        """Connect to WebSocket."""
        try:
            import websockets
            
            # Setup headers
            headers = self.headers.copy()
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            # Convert HTTP(S) URL to WS(S)
            ws_url = self.server_url.replace("http://", "ws://").replace("https://", "wss://")
            
            # Connect
            self._websocket = await websockets.connect(
                ws_url,
                extra_headers=headers
            )
            
            logger.debug(f"Connected to MCP WebSocket server: {ws_url}")
            
        except ImportError:
            raise MCPConnectionError("websockets is required for WebSocket transport")
        except Exception as e:
            raise MCPConnectionError(f"Failed to connect to WebSocket server: {e}")
    
    async def send(self, message: MCPMessage) -> None:
        """Send message via WebSocket."""
        if self._closed or not self._websocket:
            raise MCPConnectionError("Connection closed")
        
        try:
            data = message.model_dump(exclude_none=True)
            json_str = json.dumps(data)
            await self._websocket.send(json_str)
            
        except Exception as e:
            raise MCPConnectionError(f"Failed to send message: {e}")
    
    async def receive(self) -> MCPMessage:
        """Receive message from WebSocket."""
        if self._closed or not self._websocket:
            raise MCPConnectionError("Connection closed")
        
        try:
            message_str = await self._websocket.recv()
            data = json.loads(message_str)
            return MCPMessage(**data)
            
        except json.JSONDecodeError as e:
            raise MCPConnectionError(f"Invalid JSON received: {e}")
        except Exception as e:
            raise MCPConnectionError(f"Failed to receive message: {e}")
    
    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self._closed:
            return
        
        self._closed = True
        
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            finally:
                self._websocket = None


async def create_transport(
    transport_type: MCPTransport,
    server_uri: str,
    server_command: Optional[str] = None,
    server_args: Optional[List[str]] = None,
    timeout: float = 30.0,
    auth_token: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None
) -> MCPConnection:
    """Create and connect appropriate transport."""
    
    transport: MCPConnection
    
    if transport_type == MCPTransport.STDIO:
        if not server_command:
            raise MCPConnectionError("server_command required for STDIO transport")
        transport = StdioTransport(server_command, server_args or [])
        
    elif transport_type == MCPTransport.SSE:
        transport = SSETransport(server_uri, auth_token, headers)
        
    elif transport_type == MCPTransport.WEBSOCKET:
        transport = WebSocketTransport(server_uri, auth_token, headers)
        
    else:
        raise MCPConnectionError(f"Unsupported transport type: {transport_type}")
    
    # Connect with timeout
    try:
        await asyncio.wait_for(transport.connect(), timeout=timeout)
        return transport
    except asyncio.TimeoutError:
        await transport.close()
        raise MCPConnectionError(f"Connection timeout after {timeout} seconds")
    except Exception:
        await transport.close()
        raise