"""Tests for REPL command handlers."""

import pytest
import pytest_asyncio
import json
from typing import Dict, Any, List
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from flowlib.agent.runners.repl.handlers import (
    DefaultCommandHandler,
    AgentCommandHandler,
    MCPCommandHandler,
    ToolCommandHandler,
    TodoCommandHandler
)
from flowlib.agent.runners.repl.commands import Command, CommandType, CommandRegistry
from flowlib.agent.components.task_execution import ToolResult


class TestDefaultCommandHandler:
    """Test DefaultCommandHandler implementation."""
    
    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        return DefaultCommandHandler()
    
    @pytest.fixture
    def user_command(self):
        """Create user command."""
        return Command(CommandType.USER, "", [], "Hello, how are you?")
    
    @pytest.fixture
    def system_command(self):
        """Create system command."""
        return Command(CommandType.SYSTEM, "clear", [], "clear")
    
    @pytest.fixture
    def slash_command(self):
        """Create slash command."""
        return Command(CommandType.SLASH, "help", [], "/help")
    
    def test_can_handle_user_commands(self, handler, user_command):
        """Test handler can handle user commands."""
        assert handler.can_handle(user_command) is True
    
    def test_can_handle_system_commands(self, handler, system_command):
        """Test handler can handle system commands."""
        assert handler.can_handle(system_command) is True
    
    def test_cannot_handle_slash_commands(self, handler, slash_command):
        """Test handler cannot handle slash commands."""
        assert handler.can_handle(slash_command) is False
    
    @pytest.mark.asyncio
    async def test_handle_user_command(self, handler, user_command):
        """Test handling user commands."""
        context = {}
        result = await handler.handle(user_command, context)
        assert result is None  # Passes through
    
    @pytest.mark.asyncio
    async def test_handle_system_command(self, handler, system_command):
        """Test handling system commands."""
        context = {}
        result = await handler.handle(system_command, context)
        assert result is None  # System commands handled by registry


class TestAgentCommandHandler:
    """Test AgentCommandHandler implementation."""
    
    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        return AgentCommandHandler()
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent."""
        agent = Mock()
        agent.execute_flow = AsyncMock()
        agent._state_manager = Mock()
        agent._state_manager.current_state = Mock()
        agent._state_manager.current_state.status = "active"
        agent._state_manager.current_state.mode = "chat"
        agent._state_manager.current_state.active_flows = []
        agent._state_manager.current_state.context = {}
        
        # Add state property for backward compatibility with handler
        mock_state = Mock()
        mock_state.status = "active"
        mock_state.mode = "chat"
        mock_state.active_flows = []
        mock_state.context = {}
        agent.state = mock_state
        
        agent.memory = Mock()
        agent.flow_runner = Mock()
        agent.tools = {}
        return agent
    
    @pytest.fixture
    def context_with_agent(self, mock_agent):
        """Create context with agent."""
        return {
            "agent": mock_agent,
            "session_stats": {
                "flows_executed": 0,
                "message_count": 0,
                "tokens_used": 0
            }
        }
    
    def test_can_handle_flow_commands(self, handler):
        """Test handler can handle flow commands."""
        flow_command = Command(CommandType.USER, "", [], "!test_flow arg1 arg2")
        assert handler.can_handle(flow_command) is True
    
    def test_can_handle_introspection_commands(self, handler):
        """Test handler can handle introspection commands."""
        intro_command = Command(CommandType.USER, "", [], "?state")
        assert handler.can_handle(intro_command) is True
    
    def test_cannot_handle_regular_commands(self, handler):
        """Test handler cannot handle regular commands."""
        regular_command = Command(CommandType.USER, "", [], "Hello there")
        assert handler.can_handle(regular_command) is False
    
    @pytest.mark.asyncio
    async def test_execute_flow_success(self, handler, context_with_agent):
        """Test successful flow execution."""
        command = Command(CommandType.USER, "", [], "!test_flow {\"input\": \"test\"}")
        context_with_agent["agent"].execute_flow.return_value = "Flow result"
        
        result = await handler.handle(command, context_with_agent)
        
        assert "Flow 'test_flow' executed successfully" in result
        assert "Flow result" in result
        context_with_agent["agent"].execute_flow.assert_called_once_with("test_flow", {"input": "test"})
        assert context_with_agent["session_stats"]["flows_executed"] == 1
    
    @pytest.mark.asyncio
    async def test_execute_flow_string_args(self, handler, context_with_agent):
        """Test flow execution with string arguments."""
        command = Command(CommandType.USER, "", [], "!test_flow simple string arg")
        context_with_agent["agent"].execute_flow.return_value = "Flow result"
        
        result = await handler.handle(command, context_with_agent)
        
        context_with_agent["agent"].execute_flow.assert_called_once_with("test_flow", {"input": "simple string arg"})
    
    @pytest.mark.asyncio
    async def test_execute_flow_no_args(self, handler, context_with_agent):
        """Test flow execution without arguments."""
        command = Command(CommandType.USER, "", [], "!test_flow")
        context_with_agent["agent"].execute_flow.return_value = "Flow result"
        
        result = await handler.handle(command, context_with_agent)
        
        context_with_agent["agent"].execute_flow.assert_called_once_with("test_flow", {})
    
    @pytest.mark.asyncio
    async def test_execute_flow_error(self, handler, context_with_agent):
        """Test flow execution error handling."""
        command = Command(CommandType.USER, "", [], "!test_flow")
        context_with_agent["agent"].execute_flow.side_effect = Exception("Flow failed")
        
        result = await handler.handle(command, context_with_agent)
        
        assert "Error executing flow 'test_flow': Flow failed" in result
    
    @pytest.mark.asyncio
    async def test_execute_flow_invalid_command(self, handler, context_with_agent):
        """Test flow execution with invalid command."""
        command = Command(CommandType.USER, "", [], "!")
        
        result = await handler.handle(command, context_with_agent)
        
        assert "Usage: !flow_name [arguments]" in result
    
    @pytest.mark.asyncio
    async def test_introspect_state(self, handler, context_with_agent):
        """Test state introspection."""
        command = Command(CommandType.USER, "", [], "?state")
        
        result = await handler.handle(command, context_with_agent)
        
        assert "**Agent State:**" in result
        assert "Status: active" in result
        assert "Mode: chat" in result
    
    @pytest.mark.asyncio
    async def test_introspect_memory(self, handler, context_with_agent):
        """Test memory introspection."""
        command = Command(CommandType.USER, "", [], "?memory")
        
        # Mock memory components
        working_memory = Mock()
        working_memory.get_recent = AsyncMock(return_value=[
            {"content": "Test memory item 1"},
            {"content": "Test memory item 2"}
        ])
        
        vector_memory = Mock()
        vector_memory.get_stats = AsyncMock(return_value={"count": 100, "collections": 5})
        
        knowledge_graph = Mock()
        knowledge_graph.get_stats = AsyncMock(return_value={
            "nodes": 50, "edges": 75, "node_types": ["person", "document"]
        })
        
        context_with_agent["agent"].memory.working_memory = working_memory
        context_with_agent["agent"].memory.vector_memory = vector_memory
        context_with_agent["agent"].memory.knowledge_graph = knowledge_graph
        
        result = await handler.handle(command, context_with_agent)
        
        assert "**Working Memory**" in result
        assert "**Vector Memory**" in result
        assert "**Knowledge Graph**" in result
        assert "Embeddings: 100" in result
        assert "Nodes: 50" in result
    
    @pytest.mark.asyncio
    async def test_introspect_context(self, handler, context_with_agent):
        """Test context introspection."""
        command = Command(CommandType.USER, "", [], "?context")
        context_with_agent["test_key"] = "test_value"
        
        result = await handler.handle(command, context_with_agent)
        
        assert "**Current Context:**" in result
        assert "test_key" in result
        # Sensitive keys should be filtered out
        assert "agent" not in result
    
    @pytest.mark.asyncio
    async def test_introspect_flows(self, handler, context_with_agent):
        """Test flows introspection."""
        command = Command(CommandType.USER, "", [], "?flows")
        
        context_with_agent["agent"].flow_runner.list_available_flows = AsyncMock(return_value=[
            {"name": "test_flow", "description": "Test flow"},
            {"name": "another_flow", "description": "Another flow"}
        ])
        
        result = await handler.handle(command, context_with_agent)
        
        assert "**Available Flows:**" in result
        assert "test_flow: Test flow" in result
        assert "another_flow: Another flow" in result
    
    @pytest.mark.asyncio
    async def test_introspect_tools(self, handler, context_with_agent):
        """Test tools introspection."""
        command = Command(CommandType.USER, "", [], "?tools")
        
        mock_tool = Mock()
        mock_tool.description = "Test tool description"
        context_with_agent["agent"].tools = {"test_tool": mock_tool}
        
        result = await handler.handle(command, context_with_agent)
        
        assert "**Available Tools:**" in result
        assert "test_tool: Test tool description" in result
    
    @pytest.mark.asyncio
    async def test_introspect_invalid_query(self, handler, context_with_agent):
        """Test invalid introspection query."""
        command = Command(CommandType.USER, "", [], "?invalid")
        
        result = await handler.handle(command, context_with_agent)
        
        assert "Unknown introspection query: invalid" in result
    
    @pytest.mark.asyncio
    async def test_introspect_no_query(self, handler, context_with_agent):
        """Test introspection without query."""
        command = Command(CommandType.USER, "", [], "?")
        
        result = await handler.handle(command, context_with_agent)
        
        assert "Usage: ?[state|memory|context|flows|tools]" in result
    
    @pytest.mark.asyncio
    async def test_no_agent_available(self, handler):
        """Test handler behavior when no agent is available."""
        command = Command(CommandType.USER, "", [], "!test_flow")
        context = {}
        
        result = await handler.handle(command, context)
        
        assert "No agent available." in result


class TestMCPCommandHandler:
    """Test MCPCommandHandler implementation."""
    
    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        return MCPCommandHandler()
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent with MCP client."""
        agent = Mock()
        agent.mcp_client = Mock()
        agent.mcp_client.connected = True
        agent.mcp_client.connect = AsyncMock()
        agent.mcp_client.disconnect = AsyncMock()
        agent.mcp_client.get_status = AsyncMock()
        agent.mcp_client.list_servers = AsyncMock()
        agent.mcp_client.list_tools = AsyncMock()
        agent.mcp_client.list_resources = AsyncMock()
        return agent
    
    @pytest.fixture
    def context_with_agent(self, mock_agent):
        """Create context with agent."""
        return {"agent": mock_agent}
    
    def test_can_handle_mcp_commands(self, handler):
        """Test handler can handle MCP commands."""
        mcp_command = Command(CommandType.SLASH, "mcp", ["status"], "/mcp status")
        assert handler.can_handle(mcp_command) is True
        
        tools_command = Command(CommandType.SLASH, "tools", [], "/tools")
        assert handler.can_handle(tools_command) is True
        
        resources_command = Command(CommandType.SLASH, "resources", [], "/resources")
        assert handler.can_handle(resources_command) is True
    
    def test_cannot_handle_other_commands(self, handler):
        """Test handler cannot handle other commands."""
        other_command = Command(CommandType.SLASH, "help", [], "/help")
        assert handler.can_handle(other_command) is False
    
    @pytest.mark.asyncio
    async def test_mcp_help(self, handler, context_with_agent):
        """Test MCP help display."""
        command = Command(CommandType.SLASH, "mcp", [], "/mcp")
        
        result = await handler.handle(command, context_with_agent)
        
        assert "**MCP Commands:**" in result
        assert "/mcp connect <server>" in result
        assert "/mcp disconnect" in result
        assert "/mcp status" in result
        assert "/mcp servers" in result
    
    @pytest.mark.asyncio
    async def test_mcp_connect_success(self, handler, context_with_agent):
        """Test successful MCP server connection."""
        command = Command(CommandType.SLASH, "mcp", ["connect", "test_server"], "/mcp connect test_server")
        
        result = await handler.handle(command, context_with_agent)
        
        assert "Connected to MCP server: test_server" in result
        context_with_agent["agent"].mcp_client.connect.assert_called_once_with("test_server")
    
    @pytest.mark.asyncio
    async def test_mcp_connect_error(self, handler, context_with_agent):
        """Test MCP server connection error."""
        command = Command(CommandType.SLASH, "mcp", ["connect", "test_server"], "/mcp connect test_server")
        context_with_agent["agent"].mcp_client.connect.side_effect = Exception("Connection failed")
        
        result = await handler.handle(command, context_with_agent)
        
        assert "Failed to connect to MCP server: Connection failed" in result
    
    @pytest.mark.asyncio
    async def test_mcp_connect_missing_server(self, handler, context_with_agent):
        """Test MCP connect without server argument."""
        command = Command(CommandType.SLASH, "mcp", ["connect"], "/mcp connect")
        
        result = await handler.handle(command, context_with_agent)
        
        assert "Usage: /mcp connect <server>" in result
    
    @pytest.mark.asyncio
    async def test_mcp_disconnect(self, handler, context_with_agent):
        """Test MCP server disconnection."""
        command = Command(CommandType.SLASH, "mcp", ["disconnect"], "/mcp disconnect")
        
        result = await handler.handle(command, context_with_agent)
        
        assert "Disconnected from MCP server." in result
        context_with_agent["agent"].mcp_client.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mcp_status(self, handler, context_with_agent):
        """Test MCP status display."""
        command = Command(CommandType.SLASH, "mcp", ["status"], "/mcp status")
        context_with_agent["agent"].mcp_client.get_status.return_value = {
            "connected": True,
            "server": "test_server",
            "protocol_version": "1.0",
            "tool_count": 5,
            "resource_count": 3
        }
        
        result = await handler.handle(command, context_with_agent)
        
        assert "**MCP Status:**" in result
        assert "Connected: True" in result
        assert "Server: test_server" in result
        assert "Protocol Version: 1.0" in result
        assert "Tools Available: 5" in result
        assert "Resources Available: 3" in result
    
    @pytest.mark.asyncio
    async def test_mcp_list_servers(self, handler, context_with_agent):
        """Test MCP servers listing."""
        command = Command(CommandType.SLASH, "mcp", ["servers"], "/mcp servers")
        context_with_agent["agent"].mcp_client.list_servers.return_value = [
            {"name": "server1", "description": "First server"},
            {"name": "server2", "description": "Second server"}
        ]
        
        result = await handler.handle(command, context_with_agent)
        
        assert "**Available MCP Servers:**" in result
        assert "server1: First server" in result
        assert "server2: Second server" in result
    
    @pytest.mark.asyncio
    async def test_list_mcp_tools(self, handler, context_with_agent):
        """Test MCP tools listing."""
        command = Command(CommandType.SLASH, "tools", [], "/tools")
        context_with_agent["agent"].mcp_client.list_tools.return_value = [
            {"name": "tool1", "description": "First tool"},
            {"name": "tool2", "description": "Second tool", "parameters": {"param1": "value1"}}
        ]
        
        result = await handler.handle(command, context_with_agent)
        
        assert "**MCP Tools:**" in result
        assert "tool1: First tool" in result
        assert "tool2: Second tool" in result
    
    @pytest.mark.asyncio
    async def test_list_mcp_tools_verbose(self, handler, context_with_agent):
        """Test MCP tools listing with verbose output."""
        command = Command(CommandType.SLASH, "tools", [], "/tools")
        context_with_agent["verbose"] = True
        context_with_agent["agent"].mcp_client.list_tools.return_value = [
            {"name": "tool1", "description": "First tool", "parameters": {"param1": "value1"}}
        ]
        
        result = await handler.handle(command, context_with_agent)
        
        assert "Parameters: {'param1': 'value1'}" in result
    
    @pytest.mark.asyncio
    async def test_list_mcp_tools_no_connection(self, handler, context_with_agent):
        """Test MCP tools listing without connection."""
        command = Command(CommandType.SLASH, "tools", [], "/tools")
        context_with_agent["agent"].mcp_client.connected = False
        
        result = await handler.handle(command, context_with_agent)
        
        assert "No MCP connection. Use /mcp connect <server> first." in result
    
    @pytest.mark.asyncio
    async def test_list_mcp_resources(self, handler, context_with_agent):
        """Test MCP resources listing."""
        command = Command(CommandType.SLASH, "resources", [], "/resources")
        context_with_agent["agent"].mcp_client.list_resources.return_value = [
            {"uri": "file://test.txt", "description": "Test file"},
            {"uri": "https://example.com", "description": "Example website", "mimeType": "text/html"}
        ]
        
        result = await handler.handle(command, context_with_agent)
        
        assert "**MCP Resources:**" in result
        assert "file://test.txt: Test file" in result
        assert "https://example.com: Example website" in result
    
    @pytest.mark.asyncio
    async def test_no_agent_available(self, handler):
        """Test handler behavior when no agent is available."""
        command = Command(CommandType.SLASH, "mcp", ["status"], "/mcp status")
        context = {}
        
        result = await handler.handle(command, context)
        
        assert "No agent available." in result
    
    @pytest.mark.asyncio
    async def test_agent_no_mcp_support(self, handler):
        """Test handler behavior when agent doesn't support MCP."""
        command = Command(CommandType.SLASH, "mcp", ["status"], "/mcp status")
        agent = Mock()
        del agent.mcp_client  # Remove MCP client attribute
        context = {"agent": agent}
        
        result = await handler.handle(command, context)
        
        assert "Agent does not support MCP." in result


class TestToolCommandHandler:
    """Test ToolCommandHandler implementation."""
    
    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        return ToolCommandHandler()
    
    @pytest.fixture
    def mock_tool(self):
        """Create mock tool."""
        tool = Mock(spec=REPLTool)
        tool.name = "test_tool"
        tool.description = "Test tool description"
        tool.parameters = [
            ToolParameter(name="param1", type="str", description="First parameter", required=True),
            ToolParameter(name="param2", type="int", description="Second parameter", required=False, default=42)
        ]
        return tool
    
    @pytest.fixture
    def context(self):
        """Create context."""
        return {"verbose": False}
    
    def test_can_handle_tool_slash_command(self, handler):
        """Test handler can handle tool slash commands."""
        command = Command(CommandType.SLASH, "tool", ["list"], "/tool list")
        assert handler.can_handle(command) is True
    
    def test_can_handle_direct_tool_execution(self, handler):
        """Test handler can handle direct tool execution."""
        with patch('flowlib.agent.runners.repl.handlers.tool_registry') as mock_registry:
            mock_registry.get.return_value = Mock()
            
            command = Command(CommandType.USER, "", [], "@test_tool param=value")
            assert handler.can_handle(command) is True
    
    def test_cannot_handle_direct_tool_execution_unknown(self, handler):
        """Test handler cannot handle unknown direct tool execution."""
        with patch('flowlib.agent.runners.repl.handlers.tool_registry') as mock_registry:
            mock_registry.get.return_value = None
            
            command = Command(CommandType.USER, "", [], "@unknown_tool param=value")
            assert handler.can_handle(command) is False
    
    def test_cannot_handle_other_commands(self, handler):
        """Test handler cannot handle other commands."""
        command = Command(CommandType.SLASH, "help", [], "/help")
        assert handler.can_handle(command) is False
    
    @pytest.mark.asyncio
    async def test_tool_help(self, handler, context):
        """Test tool help display."""
        command = Command(CommandType.SLASH, "tool", [], "/tool")
        
        result = await handler.handle(command, context)
        
        assert "**Tool Commands:**" in result
        assert "/tool list" in result
        assert "/tool describe <tool_name>" in result
        assert "/tool execute <tool_name> <args>" in result
        assert "@<tool_name> <args>" in result
    
    @pytest.mark.asyncio
    async def test_tool_list(self, handler, context):
        """Test tool listing."""
        command = Command(CommandType.SLASH, "tool", ["list"], "/tool list")
        
        with patch('flowlib.agent.runners.repl.handlers.tool_registry') as mock_registry:
            mock_registry.list_tools.return_value = ["tool1", "tool2"]
            mock_registry.get.side_effect = lambda name: Mock(description=f"{name} description")
            
            result = await handler.handle(command, context)
            
            assert "**Available Tools:**" in result
            assert "tool1: tool1 description" in result
            assert "tool2: tool2 description" in result
    
    @pytest.mark.asyncio
    async def test_tool_list_empty(self, handler, context):
        """Test tool listing when no tools available."""
        command = Command(CommandType.SLASH, "tool", ["list"], "/tool list")
        
        with patch('flowlib.agent.runners.repl.handlers.tool_registry') as mock_registry:
            mock_registry.list_tools.return_value = []
            
            result = await handler.handle(command, context)
            
            assert "No tools available." in result
    
    @pytest.mark.asyncio
    async def test_tool_describe(self, handler, context, mock_tool):
        """Test tool description."""
        command = Command(CommandType.SLASH, "tool", ["describe", "test_tool"], "/tool describe test_tool")
        
        with patch('flowlib.agent.runners.repl.handlers.tool_registry') as mock_registry:
            mock_registry.get.return_value = mock_tool
            
            result = await handler.handle(command, context)
            
            assert "**Tool: test_tool**" in result
            assert "Description: Test tool description" in result
            assert "param1 (str) (required)" in result
            assert "param2 (int) (optional) [default: 42]" in result
            assert "**Usage Examples:**" in result
    
    @pytest.mark.asyncio
    async def test_tool_describe_not_found(self, handler, context):
        """Test tool description for non-existent tool."""
        command = Command(CommandType.SLASH, "tool", ["describe", "unknown_tool"], "/tool describe unknown_tool")
        
        with patch('flowlib.agent.runners.repl.handlers.tool_registry') as mock_registry:
            mock_registry.get.return_value = None
            
            result = await handler.handle(command, context)
            
            assert "Tool 'unknown_tool' not found." in result
    
    @pytest.mark.asyncio
    async def test_tool_describe_missing_tool_name(self, handler, context):
        """Test tool description without tool name."""
        command = Command(CommandType.SLASH, "tool", ["describe"], "/tool describe")
        
        result = await handler.handle(command, context)
        
        assert "Usage: /tool describe <tool_name>" in result
    
    @pytest.mark.asyncio
    async def test_tool_execute_success(self, handler, context):
        """Test successful tool execution."""
        command = Command(CommandType.SLASH, "tool", ["execute", "test_tool", "param1=value1"], "/tool execute test_tool param1=value1")
        
        mock_result = ToolResult(status=ToolResultStatus.SUCCESS, content="Tool output")
        
        with patch('flowlib.agent.runners.repl.handlers.tool_registry') as mock_registry:
            mock_registry.get.return_value = Mock()
            mock_registry.execute_tool = AsyncMock(return_value=mock_result)
            
            result = await handler.handle(command, context)
            
            assert "✅ **test_tool** executed successfully" in result
            assert "Tool output" in result
            mock_registry.execute_tool.assert_called_once_with("test_tool", param1="value1")
    
    @pytest.mark.asyncio
    async def test_tool_execute_warning(self, handler, context):
        """Test tool execution with warning."""
        command = Command(CommandType.SLASH, "tool", ["execute", "test_tool"], "/tool execute test_tool")
        
        mock_result = ToolResult(status=ToolResultStatus.WARNING, content="Tool output", error="Warning message")
        
        with patch('flowlib.agent.runners.repl.handlers.tool_registry') as mock_registry:
            mock_registry.get.return_value = Mock()
            mock_registry.execute_tool = AsyncMock(return_value=mock_result)
            
            result = await handler.handle(command, context)
            
            assert "⚠️ **test_tool** completed with warnings" in result
            assert "Tool output" in result
            assert "**Warning:** Warning message" in result
    
    @pytest.mark.asyncio
    async def test_tool_execute_error(self, handler, context):
        """Test tool execution with error."""
        command = Command(CommandType.SLASH, "tool", ["execute", "test_tool"], "/tool execute test_tool")
        
        mock_result = ToolResult(status=ToolResultStatus.ERROR, content="Error output", error="Error message")
        
        with patch('flowlib.agent.runners.repl.handlers.tool_registry') as mock_registry:
            mock_registry.get.return_value = Mock()
            mock_registry.execute_tool = AsyncMock(return_value=mock_result)
            
            result = await handler.handle(command, context)
            
            assert "❌ **test_tool** failed" in result
            assert "**Error:** Error message" in result
            assert "**Output:** Error output" in result
    
    @pytest.mark.asyncio
    async def test_tool_execute_exception(self, handler, context):
        """Test tool execution with exception."""
        command = Command(CommandType.SLASH, "tool", ["execute", "test_tool"], "/tool execute test_tool")
        
        with patch('flowlib.agent.runners.repl.handlers.tool_registry') as mock_registry:
            mock_registry.get.return_value = Mock()
            mock_registry.execute_tool = AsyncMock(side_effect=Exception("Execution failed"))
            
            result = await handler.handle(command, context)
            
            assert "❌ Error executing tool 'test_tool': Execution failed" in result
    
    @pytest.mark.asyncio
    async def test_direct_tool_execution(self, handler, context):
        """Test direct tool execution with @ syntax."""
        command = Command(CommandType.USER, "", [], "@test_tool param1=value1 param2=123")
        
        mock_result = ToolResult(status=ToolResultStatus.SUCCESS, content="Tool output")
        
        with patch('flowlib.agent.runners.repl.handlers.tool_registry') as mock_registry:
            mock_registry.get.return_value = Mock()
            mock_registry.execute_tool = AsyncMock(return_value=mock_result)
            
            result = await handler.handle(command, context)
            
            assert "✅ **test_tool** executed successfully" in result
            mock_registry.execute_tool.assert_called_once_with("test_tool", param1="value1", param2=123)
    
    @pytest.mark.asyncio
    async def test_direct_tool_execution_no_args(self, handler, context):
        """Test direct tool execution without arguments."""
        command = Command(CommandType.USER, "", [], "@test_tool")
        
        mock_result = ToolResult(status=ToolResultStatus.SUCCESS, content="Tool output")
        
        with patch('flowlib.agent.runners.repl.handlers.tool_registry') as mock_registry:
            mock_registry.get.return_value = Mock()
            mock_registry.execute_tool = AsyncMock(return_value=mock_result)
            
            result = await handler.handle(command, context)
            
            mock_registry.execute_tool.assert_called_once_with("test_tool")
    
    def test_parse_tool_args_key_value_pairs(self, handler):
        """Test parsing key=value arguments."""
        args = handler._parse_tool_args('param1="string value" param2=123 param3=true')
        
        assert args["param1"] == "string value"
        assert args["param2"] == 123
        assert args["param3"] is True
    
    def test_parse_tool_args_json_values(self, handler):
        """Test parsing JSON values."""
        args = handler._parse_tool_args('param1=["item1", "item2"] param2={"key": "value"}')
        
        assert args["param1"] == ["item1", "item2"]
        assert args["param2"] == {"key": "value"}
    
    def test_parse_tool_args_positional(self, handler):
        """Test parsing positional arguments."""
        args = handler._parse_tool_args('positional_value')
        
        assert args["content"] == "positional_value"
    
    def test_parse_tool_args_empty(self, handler):
        """Test parsing empty arguments."""
        args = handler._parse_tool_args("")
        
        assert args == {}
    
    def test_parse_tool_args_quoted_values(self, handler):
        """Test parsing quoted values."""
        args = handler._parse_tool_args("param1='single quoted' param2=\"double quoted\"")
        
        assert args["param1"] == "single quoted"
        assert args["param2"] == "double quoted"


class TestTodoCommandHandler:
    """Test TodoCommandHandler implementation."""
    
    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        return TodoCommandHandler()
    
    @pytest.fixture
    def mock_todo_manager(self):
        """Create mock todo manager."""
        manager = Mock()
        manager.get_current_list = Mock()
        manager.add_todo = Mock()
        manager.mark_todo_completed = Mock()
        manager.mark_todo_failed = Mock()
        manager.export_list = Mock()
        manager.create_list = Mock()
        return manager
    
    @pytest.fixture
    def mock_todo_list(self):
        """Create mock todo list."""
        todo_list = Mock()
        todo_list.items = {}
        todo_list.get_todos_by_status = Mock(return_value=[])
        todo_list.get_progress_summary = Mock(return_value={
            "total": 5, "completed": 2, "in_progress": 1, 
            "pending": 1, "failed": 1, "blocked": 0, "progress": 0.4
        })
        todo_list.delete_todo = Mock()
        return todo_list
    
    @pytest.fixture
    def mock_agent(self, mock_todo_manager):
        """Create mock agent with todo support."""
        agent = Mock()
        agent._engine = Mock()
        agent._engine.get_todo_manager.return_value = mock_todo_manager
        return agent
    
    @pytest.fixture
    def context_with_agent(self, mock_agent):
        """Create context with agent."""
        return {"agent": mock_agent}
    
    def test_can_handle_todo_slash_commands(self, handler):
        """Test handler can handle todo slash commands."""
        todo_command = Command(CommandType.SLASH, "todo", ["list"], "/todo list")
        assert handler.can_handle(todo_command) is True
        
        todos_command = Command(CommandType.SLASH, "todos", [], "/todos")
        assert handler.can_handle(todos_command) is True
    
    def test_can_handle_quick_todo_commands(self, handler):
        """Test handler can handle quick todo commands."""
        quick_command = Command(CommandType.USER, "", [], "#todo Fix the bug")
        assert handler.can_handle(quick_command) is True
    
    def test_cannot_handle_other_commands(self, handler):
        """Test handler cannot handle other commands."""
        other_command = Command(CommandType.SLASH, "help", [], "/help")
        assert handler.can_handle(other_command) is False
    
    @pytest.mark.asyncio
    async def test_todo_help(self, handler, context_with_agent):
        """Test todo help display."""
        command = Command(CommandType.SLASH, "todo", [], "/todo")
        
        result = await handler.handle(command, context_with_agent)
        
        assert "**TODO Commands:**" in result
        assert "/todo list" in result
        assert "/todo add <content>" in result
        assert "/todo complete <id>" in result
        assert "#todo <content>" in result
    
    @pytest.mark.asyncio
    async def test_todo_list_empty(self, handler, context_with_agent, mock_todo_manager, mock_todo_list):
        """Test listing empty todo list."""
        command = Command(CommandType.SLASH, "todo", ["list"], "/todo list")
        mock_todo_list.items = {}
        mock_todo_manager.get_current_list.return_value = mock_todo_list
        
        result = await handler.handle(command, context_with_agent)
        
        assert "No TODOs in current list." in result
    
    @pytest.mark.asyncio
    async def test_todo_list_with_items(self, handler, context_with_agent, mock_todo_manager, mock_todo_list):
        """Test listing todos with items."""
        from flowlib.agent.components.task_decomposition.todo import TodoStatus, TodoPriority
        
        # Mock todo items
        mock_todo = Mock()
        mock_todo.id = "abc123def456"
        mock_todo.content = "Test todo item"
        mock_todo.priority = TodoPriority.HIGH
        mock_todo.created_at = "2023-01-01T00:00:00"
        mock_todo.error_message = None
        
        command = Command(CommandType.SLASH, "todo", ["list"], "/todo list")
        mock_todo_list.items = {"abc123def456": mock_todo}
        mock_todo_list.get_todos_by_status.return_value = [mock_todo]
        mock_todo_manager.get_current_list.return_value = mock_todo_list
        
        with patch('flowlib.agent.components.task_decomposition.todo.TodoStatus', TodoStatus):
            result = await handler.handle(command, context_with_agent)
            
            assert "**Current TODOs:**" in result
            assert "`abc123de`" in result  # Shortened ID
            assert "Test todo item" in result
    
    @pytest.mark.asyncio
    async def test_todo_add(self, handler, context_with_agent, mock_todo_manager):
        """Test adding a todo."""
        command = Command(CommandType.SLASH, "todo", ["add", "Fix", "the", "bug"], "/todo add Fix the bug")
        mock_todo_manager.add_todo.return_value = "abc123def456"
        
        result = await handler.handle(command, context_with_agent)
        
        assert "✅ Added TODO `abc123de`: Fix the bug" in result
        mock_todo_manager.add_todo.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_todo_add_priority_detection(self, handler, context_with_agent, mock_todo_manager):
        """Test adding todo with priority detection."""
        from flowlib.agent.components.task_decomposition.todo import TodoPriority
        
        command = Command(CommandType.SLASH, "todo", ["add", "urgent", "task"], "/todo add urgent task")
        mock_todo_manager.add_todo.return_value = "abc123def456"
        
        with patch('flowlib.agent.components.task_decomposition.todo.TodoPriority', TodoPriority):
            result = await handler.handle(command, context_with_agent)
            
            # Should detect "urgent" and set high priority
            call_args = mock_todo_manager.add_todo.call_args
            assert call_args[1]["priority"] == TodoPriority.URGENT
    
    @pytest.mark.asyncio
    async def test_todo_complete(self, handler, context_with_agent, mock_todo_manager, mock_todo_list):
        """Test completing a todo."""
        mock_todo = Mock()
        mock_todo.id = "abc123def456"
        mock_todo.content = "Test todo"
        
        command = Command(CommandType.SLASH, "todo", ["complete", "abc123"], "/todo complete abc123")
        mock_todo_list.items = {"abc123def456": mock_todo}
        mock_todo_manager.get_current_list.return_value = mock_todo_list
        mock_todo_manager.mark_todo_completed.return_value = True
        
        result = await handler.handle(command, context_with_agent)
        
        assert "✅ Marked TODO as completed: Test todo" in result
        mock_todo_manager.mark_todo_completed.assert_called_once_with("abc123def456")
    
    @pytest.mark.asyncio
    async def test_todo_fail(self, handler, context_with_agent, mock_todo_manager, mock_todo_list):
        """Test failing a todo."""
        mock_todo = Mock()
        mock_todo.id = "abc123def456"
        mock_todo.content = "Test todo"
        
        command = Command(CommandType.SLASH, "todo", ["fail", "abc123", "Not", "feasible"], "/todo fail abc123 Not feasible")
        mock_todo_list.items = {"abc123def456": mock_todo}
        mock_todo_manager.get_current_list.return_value = mock_todo_list
        mock_todo_manager.mark_todo_failed.return_value = True
        
        result = await handler.handle(command, context_with_agent)
        
        assert "❌ Marked TODO as failed: Test todo" in result
        assert "Reason: Not feasible" in result
        mock_todo_manager.mark_todo_failed.assert_called_once_with("abc123def456", "Not feasible")
    
    @pytest.mark.asyncio
    async def test_todo_delete(self, handler, context_with_agent, mock_todo_manager, mock_todo_list):
        """Test deleting a todo."""
        mock_todo = Mock()
        mock_todo.id = "abc123def456"
        mock_todo.content = "Test todo"
        
        command = Command(CommandType.SLASH, "todo", ["delete", "abc123"], "/todo delete abc123")
        mock_todo_list.items = {"abc123def456": mock_todo}
        mock_todo_list.delete_todo.return_value = True
        mock_todo_manager.get_current_list.return_value = mock_todo_list
        
        result = await handler.handle(command, context_with_agent)
        
        assert "🗑️ Deleted TODO: Test todo" in result
        mock_todo_list.delete_todo.assert_called_once_with("abc123def456")
    
    @pytest.mark.asyncio
    async def test_todo_progress(self, handler, context_with_agent, mock_todo_manager, mock_todo_list):
        """Test showing todo progress."""
        command = Command(CommandType.SLASH, "todo", ["progress"], "/todo progress")
        mock_todo_manager.get_current_list.return_value = mock_todo_list
        
        result = await handler.handle(command, context_with_agent)
        
        assert "**TODO Progress Summary:**" in result
        assert "Progress:" in result
        assert "40%" in result  # Based on mock progress (0.4)
        assert "Total: 5" in result
        assert "Completed: 2" in result
    
    @pytest.mark.asyncio
    async def test_todo_export(self, handler, context_with_agent, mock_todo_manager, mock_todo_list):
        """Test exporting todos."""
        command = Command(CommandType.SLASH, "todo", ["export", "json"], "/todo export json")
        mock_todo_manager.get_current_list.return_value = mock_todo_list
        mock_todo_manager.export_list.return_value = '{"todos": []}'
        
        result = await handler.handle(command, context_with_agent)
        
        assert "**Exported TODOs (json):**" in result
        assert '{"todos": []}' in result
        mock_todo_manager.export_list.assert_called_once_with("json")
    
    @pytest.mark.asyncio
    async def test_todo_clear(self, handler, context_with_agent, mock_todo_manager, mock_todo_list):
        """Test clearing todos."""
        command = Command(CommandType.SLASH, "todo", ["clear"], "/todo clear")
        mock_todo_list.items = {"1": Mock(), "2": Mock(), "3": Mock()}
        mock_todo_manager.get_current_list.return_value = mock_todo_list
        
        result = await handler.handle(command, context_with_agent)
        
        assert "🗑️ Cleared 3 TODOs from the list." in result
        mock_todo_manager.create_list.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_quick_todo_creation(self, handler, context_with_agent, mock_todo_manager):
        """Test quick todo creation with #todo."""
        command = Command(CommandType.USER, "", [], "#todo Fix the critical bug")
        mock_todo_manager.add_todo.return_value = "abc123def456"
        
        result = await handler.handle(command, context_with_agent)
        
        assert "✅ Added TODO `abc123de`: Fix the critical bug" in result
    
    @pytest.mark.asyncio
    async def test_quick_todo_empty_content(self, handler, context_with_agent):
        """Test quick todo with empty content."""
        command = Command(CommandType.USER, "", [], "#todo")
        
        result = await handler.handle(command, context_with_agent)
        
        assert "Usage: #todo <content>" in result
    
    @pytest.mark.asyncio
    async def test_no_agent_available(self, handler):
        """Test handler behavior when no agent is available."""
        command = Command(CommandType.SLASH, "todo", ["list"], "/todo list")
        context = {}
        
        result = await handler.handle(command, context)
        
        assert "No agent with TODO support available." in result
    
    @pytest.mark.asyncio
    async def test_todo_not_found(self, handler, context_with_agent, mock_todo_manager, mock_todo_list):
        """Test operations on non-existent todo."""
        command = Command(CommandType.SLASH, "todo", ["complete", "nonexistent"], "/todo complete nonexistent")
        mock_todo_list.items = {}
        mock_todo_manager.get_current_list.return_value = mock_todo_list
        
        result = await handler.handle(command, context_with_agent)
        
        assert "TODO not found: nonexistent" in result