"""Tests for REPL commands."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from flowlib.agent.runners.repl.commands import (
    CommandType,
    Command,
    CommandHandler,
    CommandRegistry
)


class TestCommandType:
    """Test CommandType enum."""
    
    def test_command_types(self):
        """Test all command types exist."""
        assert CommandType.SLASH.value == "slash"
        assert CommandType.META.value == "meta"
        assert CommandType.SYSTEM.value == "system"
        assert CommandType.USER.value == "user"


class TestCommand:
    """Test Command dataclass."""
    
    def test_command_creation(self):
        """Test creating a command."""
        cmd = Command(
            type=CommandType.SLASH,
            name="help",
            args=[],
            raw_input="/help"
        )
        
        assert cmd.type == CommandType.SLASH
        assert cmd.name == "help"
        assert cmd.args == []
        assert cmd.raw_input == "/help"
    
    def test_command_with_args(self):
        """Test command with arguments."""
        cmd = Command(
            type=CommandType.SLASH,
            name="save",
            args=["myfile.json", "--format", "json"],
            raw_input="/save myfile.json --format json"
        )
        
        assert len(cmd.args) == 3
        assert cmd.args[0] == "myfile.json"
        assert cmd.args[2] == "json"


class TestCommandHandler:
    """Test CommandHandler abstract base class."""
    
    def test_abstract_methods(self):
        """Test that CommandHandler is abstract."""
        with pytest.raises(TypeError):
            CommandHandler()
    
    def test_concrete_implementation(self):
        """Test implementing CommandHandler."""
        class TestHandler(CommandHandler):
            def can_handle(self, command: Command) -> bool:
                return command.name == "test"
            
            async def handle(self, command: Command, context: dict) -> str:
                return "handled"
        
        handler = TestHandler()
        cmd = Command(CommandType.USER, "test", [], "test")
        
        assert handler.can_handle(cmd) is True
        assert handler.can_handle(Command(CommandType.USER, "other", [], "other")) is False


class TestCommandRegistry:
    """Test CommandRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create a command registry."""
        return CommandRegistry()
    
    def test_initialization(self, registry):
        """Test registry initialization."""
        assert isinstance(registry.handlers, list)
        assert isinstance(registry.slash_commands, dict)
        assert isinstance(registry.meta_commands, dict)
        
        # Check built-in commands are registered
        assert "help" in registry.slash_commands
        assert "clear" in registry.slash_commands
        assert "debug" in registry.meta_commands
    
    def test_parse_slash_command(self, registry):
        """Test parsing slash commands."""
        cmd = registry.parse_command("/help")
        assert cmd.type == CommandType.SLASH
        assert cmd.name == "help"
        assert cmd.args == []
        
        cmd = registry.parse_command("/save myfile.json")
        assert cmd.type == CommandType.SLASH
        assert cmd.name == "save"
        assert cmd.args == ["myfile.json"]
        
        cmd = registry.parse_command("/mode task verbose")
        assert cmd.type == CommandType.SLASH
        assert cmd.name == "mode"
        assert cmd.args == ["task", "verbose"]
    
    def test_parse_meta_command(self, registry):
        """Test parsing meta commands."""
        cmd = registry.parse_command("@debug")
        assert cmd.type == CommandType.META
        assert cmd.name == "debug"
        assert cmd.args == []
        
        cmd = registry.parse_command("@tool read file.txt")
        assert cmd.type == CommandType.META
        assert cmd.name == "tool"
        assert cmd.args == ["read", "file.txt"]
    
    def test_parse_system_command(self, registry):
        """Test parsing system commands."""
        for sys_cmd in ["exit", "quit", "clear", "cls"]:
            cmd = registry.parse_command(sys_cmd)
            assert cmd.type == CommandType.SYSTEM
            assert cmd.name == sys_cmd
            assert cmd.args == []
        
        # Test case insensitive
        cmd = registry.parse_command("EXIT")
        assert cmd.type == CommandType.SYSTEM
        assert cmd.name == "exit"
    
    def test_parse_user_input(self, registry):
        """Test parsing regular user input."""
        cmd = registry.parse_command("Hello, how are you?")
        assert cmd.type == CommandType.USER
        assert cmd.name == ""
        assert cmd.args == []
        assert cmd.raw_input == "Hello, how are you?"
    
    def test_parse_empty_input(self, registry):
        """Test parsing empty input."""
        cmd = registry.parse_command("")
        assert cmd.type == CommandType.USER
        assert cmd.name == ""
        assert cmd.args == []
        
        cmd = registry.parse_command("   ")
        assert cmd.type == CommandType.USER
    
    def test_register_handler(self, registry):
        """Test registering a command handler."""
        handler = Mock(spec=CommandHandler)
        registry.register_handler(handler)
        
        assert handler in registry.handlers
    
    def test_register_slash_command(self, registry):
        """Test registering a slash command."""
        handler = Mock()
        registry.register_slash_command("test", handler)
        
        assert "test" in registry.slash_commands
        assert registry.slash_commands["test"] == handler
    
    def test_register_meta_command(self, registry):
        """Test registering a meta command."""
        handler = Mock()
        registry.register_meta_command("test", handler)
        
        assert "test" in registry.meta_commands
        assert registry.meta_commands["test"] == handler
    
    @pytest.mark.asyncio
    async def test_execute_slash_command(self, registry):
        """Test executing a slash command."""
        context = {}
        cmd = Command(CommandType.SLASH, "help", [], "/help")
        
        result = await registry.execute_command(cmd, context)
        
        assert isinstance(result, str)
        assert "Available Commands" in result
    
    @pytest.mark.asyncio
    async def test_execute_unknown_slash_command(self, registry):
        """Test executing unknown slash command."""
        context = {}
        cmd = Command(CommandType.SLASH, "unknown", [], "/unknown")
        
        result = await registry.execute_command(cmd, context)
        
        assert "Unknown command: /unknown" in result
    
    @pytest.mark.asyncio
    async def test_execute_meta_command(self, registry):
        """Test executing a meta command."""
        context = {"debug": False}
        cmd = Command(CommandType.META, "debug", [], "@debug")
        
        result = await registry.execute_command(cmd, context)
        
        assert "Debug mode: ON" in result
        assert context["debug"] is True
    
    @pytest.mark.asyncio
    async def test_execute_system_exit_command(self, registry):
        """Test executing system exit command."""
        context = {}
        cmd = Command(CommandType.SYSTEM, "exit", [], "exit")
        
        result = await registry.execute_command(cmd, context)
        
        assert result == "Goodbye!"
        assert context["should_exit"] is True
    
    @pytest.mark.asyncio
    async def test_execute_system_clear_command(self, registry):
        """Test executing system clear command."""
        context = {}
        cmd = Command(CommandType.SYSTEM, "clear", [], "clear")
        
        with patch('os.system') as mock_system:
            result = await registry.execute_command(cmd, context)
            mock_system.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_with_handler(self, registry):
        """Test executing command with custom handler."""
        handler = Mock(spec=CommandHandler)
        handler.can_handle.return_value = True
        handler.handle = AsyncMock(return_value="custom handled")
        
        registry.register_handler(handler)
        
        context = {}
        cmd = Command(CommandType.USER, "", [], "custom command")
        
        result = await registry.execute_command(cmd, context)
        
        assert result == "custom handled"
        handler.can_handle.assert_called_once_with(cmd)
        handler.handle.assert_called_once_with(cmd, context)
    
    @pytest.mark.asyncio
    async def test_call_async_handler(self, registry):
        """Test calling async handler."""
        async def async_handler(cmd, ctx):
            return "async result"
        
        registry.register_slash_command("async_test", async_handler)
        
        cmd = Command(CommandType.SLASH, "async_test", [], "/async_test")
        result = await registry.execute_command(cmd, {})
        
        assert result == "async result"
    
    @pytest.mark.asyncio
    async def test_call_sync_handler(self, registry):
        """Test calling sync handler."""
        def sync_handler(cmd, ctx):
            return "sync result"
        
        registry.register_slash_command("sync_test", sync_handler)
        
        cmd = Command(CommandType.SLASH, "sync_test", [], "/sync_test")
        result = await registry.execute_command(cmd, {})
        
        assert result == "sync result"
    
    def test_show_help_command(self, registry):
        """Test help command output."""
        cmd = Command(CommandType.SLASH, "help", [], "/help")
        result = registry._show_help(cmd, {})
        
        assert "Available Commands" in result
        assert "/help" in result
        assert "@debug" in result
        assert "Tool Execution" in result
    
    def test_show_history_command(self, registry):
        """Test history command."""
        context = {"command_history": ["/help", "/clear", "hello"]}
        cmd = Command(CommandType.SLASH, "history", [], "/history")
        
        result = registry._show_history(cmd, context)
        
        assert "Command History" in result
        assert "/help" in result
        assert "hello" in result
    
    def test_show_history_empty(self, registry):
        """Test history command with no history."""
        context = {}
        cmd = Command(CommandType.SLASH, "history", [], "/history")
        
        result = registry._show_history(cmd, context)
        
        assert result == "No command history yet."
    
    def test_change_mode_command(self, registry):
        """Test mode change command."""
        context = {"mode": "chat"}
        
        # Test showing current mode
        cmd = Command(CommandType.SLASH, "mode", [], "/mode")
        result = registry._change_mode(cmd, context)
        assert "Current mode: chat" in result
        
        # Test changing mode
        cmd = Command(CommandType.SLASH, "mode", ["task"], "/mode task")
        result = registry._change_mode(cmd, context)
        assert "Mode changed to: task" in result
        assert context["mode"] == "task"
        
        # Test invalid mode
        cmd = Command(CommandType.SLASH, "mode", ["invalid"], "/mode invalid")
        result = registry._change_mode(cmd, context)
        assert "Invalid mode: invalid" in result
    
    def test_toggle_commands(self, registry):
        """Test toggle commands."""
        context = {"debug": False, "verbose": True, "stream": True}
        
        # Test debug toggle
        cmd = Command(CommandType.META, "debug", [], "@debug")
        result = registry._toggle_debug(cmd, context)
        assert "Debug mode: ON" in result
        assert context["debug"] is True
        
        # Test verbose toggle
        cmd = Command(CommandType.META, "verbose", [], "@verbose")
        result = registry._toggle_verbose(cmd, context)
        assert "Verbose mode: OFF" in result
        assert context["verbose"] is False
        
        # Test stream toggle
        cmd = Command(CommandType.META, "stream", [], "@stream")
        result = registry._toggle_streaming(cmd, context)
        assert "Response streaming: OFF" in result
        assert context["stream"] is False
    
    def test_show_stats_command(self, registry):
        """Test stats command."""
        context = {
            "session_stats": {
                "message_count": 10,
                "tokens_used": 1500,
                "tools_called": 3,
                "flows_executed": 2,
                "duration": "15m 30s"
            }
        }
        
        cmd = Command(CommandType.META, "stats", [], "@stats")
        result = registry._show_stats(cmd, context)
        
        assert "Session Statistics" in result
        assert "Messages: 10" in result
        assert "Tokens Used: 1500" in result
        assert "Duration: 15m 30s" in result
    
    @pytest.mark.asyncio
    async def test_list_flows_command(self, registry):
        """Test list flows command."""
        mock_flows = {
            "flow1": Mock(description="Test flow 1"),
            "flow2": Mock(description="Test flow 2")
        }
        
        with patch('flowlib.flows.registry.flow_registry.get_all_flow_metadata', return_value=mock_flows):
            cmd = Command(CommandType.SLASH, "flows", [], "/flows")
            result = await registry._list_flows(cmd, {})
            
            assert "Available Flows" in result
            assert "flow1" in result
            assert "flow2" in result
    
    @pytest.mark.asyncio
    async def test_memory_status_command(self, registry):
        """Test memory status command."""
        mock_agent = Mock()
        mock_agent.memory.get_stats = AsyncMock(return_value={
            "working_memory_items": 5,
            "vector_memory_items": 100,
            "knowledge_graph_nodes": 50,
            "knowledge_graph_edges": 75,
            "total_storage_mb": 12.5
        })
        
        context = {"agent": mock_agent}
        cmd = Command(CommandType.SLASH, "memory", [], "/memory")
        
        result = await registry._memory_status(cmd, context)
        
        assert "Memory Status" in result
        assert "Working Memory: 5 items" in result
        assert "Vector Memory: 100 items" in result
        assert "Total Storage: 12.50 MB" in result