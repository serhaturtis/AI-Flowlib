"""Comprehensive tests for agent CLI interface."""

import pytest
import asyncio
import json
import tempfile
import os
import argparse
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from flowlib.agent.runners.cli import main, AgentCLI, create_parser, async_main


class TestAgentCLI:
    """Test AgentCLI class functionality."""
    
    @pytest.fixture
    def cli_instance(self):
        """Create AgentCLI instance for testing."""
        return AgentCLI()
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".flowlib"
            agents_dir = config_dir / "agents"
            agents_dir.mkdir(parents=True)
            yield config_dir, agents_dir
    
    def test_cli_initialization(self, cli_instance):
        """Test CLI initialization."""
        assert isinstance(cli_instance.home_dir, Path)
        assert cli_instance.config_dir == cli_instance.home_dir / ".flowlib"
        assert cli_instance.agents_dir == cli_instance.config_dir / "agents"
    
    def test_load_agent_config_nonexistent(self, cli_instance, temp_config_dir):
        """Test loading non-existent agent config."""
        config_dir, agents_dir = temp_config_dir
        cli_instance.config_dir = config_dir
        cli_instance.agents_dir = agents_dir
        
        result = cli_instance.load_agent_config("nonexistent")
        assert result is None
    
    def test_load_agent_config_valid(self, cli_instance, temp_config_dir):
        """Test loading valid agent config."""
        config_dir, agents_dir = temp_config_dir
        cli_instance.config_dir = config_dir
        cli_instance.agents_dir = agents_dir
        
        # Create test config
        test_config = {
            "name": "test-agent",
            "persona": {"name": "assistant", "system_prompt": "You are helpful"},
            "mode": {"name": "chat", "settings": {}},
            "provider": {"provider_type": "llamacpp"}
        }
        
        config_file = agents_dir / "test-agent.json"
        with open(config_file, 'w') as f:
            json.dump(test_config, f)
        
        result = cli_instance.load_agent_config("test-agent")
        assert result == test_config
    
    def test_load_agent_config_invalid_json(self, cli_instance, temp_config_dir):
        """Test loading agent config with invalid JSON."""
        config_dir, agents_dir = temp_config_dir
        cli_instance.config_dir = config_dir
        cli_instance.agents_dir = agents_dir
        
        # Create invalid JSON file
        config_file = agents_dir / "invalid.json"
        with open(config_file, 'w') as f:
            f.write("invalid json {")
        
        result = cli_instance.load_agent_config("invalid")
        assert result is None
    
    @patch('flowlib.agent.runners.cli.BaseAgent')
    def test_create_agent_from_config(self, mock_orchestrator, cli_instance):
        """Test creating agent from config."""
        test_config = {
            "name": "test-agent",
            "persona": {"name": "assistant", "system_prompt": "You are helpful"},
            "mode": {"settings": {"memory_enabled": True, "max_turns": 10}},
            "provider": {"provider_type": "llamacpp"}
        }
        
        result = cli_instance.create_agent_from_config(test_config)
        
        mock_orchestrator.assert_called_once()
        call_args = mock_orchestrator.call_args[0][0]  # AgentConfig argument
        assert call_args.name == "test-agent"
        assert call_args.persona == "assistant"
        assert call_args.provider_name == "llamacpp"
        assert call_args.system_prompt == "You are helpful"
        assert call_args.max_iterations == 10
    
    def test_print_welcome(self, cli_instance, capsys):
        """Test welcome message printing."""
        test_config = {
            "name": "test-agent",
            "persona": {"name": "assistant", "personality": "friendly"},
            "mode": {"name": "chat"},
            "interface": "repl",
            "provider": {"provider_type": "llamacpp"}
        }
        
        cli_instance.print_welcome(test_config)
        
        captured = capsys.readouterr()
        assert "test-agent Agent" in captured.out
        assert "assistant" in captured.out
        assert "chat" in captured.out
        assert "REPL" in captured.out
        assert "llamacpp" in captured.out
        assert "friendly" in captured.out
    
    def test_list_agents_no_directory(self, cli_instance, temp_config_dir, capsys):
        """Test listing agents when no agents directory exists."""
        config_dir, _ = temp_config_dir
        cli_instance.config_dir = config_dir
        cli_instance.agents_dir = config_dir / "nonexistent"
        
        cli_instance.list_agents()
        
        captured = capsys.readouterr()
        assert "No agents configured yet" in captured.out
    
    def test_list_agents_empty_directory(self, cli_instance, temp_config_dir, capsys):
        """Test listing agents with empty directory."""
        config_dir, agents_dir = temp_config_dir
        cli_instance.config_dir = config_dir
        cli_instance.agents_dir = agents_dir
        
        cli_instance.list_agents()
        
        captured = capsys.readouterr()
        assert "No valid agents found" in captured.out
    
    def test_list_agents_with_agents(self, cli_instance, temp_config_dir, capsys):
        """Test listing agents with valid agents."""
        config_dir, agents_dir = temp_config_dir
        cli_instance.config_dir = config_dir
        cli_instance.agents_dir = agents_dir
        
        # Create test agents
        agents = [
            {
                "name": "agent1",
                "persona": {"name": "assistant"},
                "mode": {"name": "chat"},
                "interface": "repl",
                "provider": {"provider_type": "llamacpp"}
            },
            {
                "name": "agent2",
                "persona": {"name": "researcher"},
                "mode": {"name": "analysis"},
                "provider": {"provider_type": "google_ai"}
            }
        ]
        
        for i, agent in enumerate(agents, 1):
            config_file = agents_dir / f"agent{i}.json"
            with open(config_file, 'w') as f:
                json.dump(agent, f)
        
        cli_instance.list_agents()
        
        captured = capsys.readouterr()
        assert "Available agents:" in captured.out
        assert "agent1" in captured.out
        assert "agent2" in captured.out
        assert "assistant" in captured.out
        assert "researcher" in captured.out


class TestArgumentParser:
    """Test argument parser functionality."""
    
    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        assert "AI-Flowlib Agent CLI" in parser.description
    
    def test_parser_agent_argument(self):
        """Test agent argument parsing."""
        parser = create_parser()
        args = parser.parse_args(["test-agent"])
        assert args.agent == "test-agent"
    
    def test_parser_message_argument(self):
        """Test message argument parsing."""
        parser = create_parser()
        args = parser.parse_args(["agent", "-m", "Hello world"])
        assert args.message == "Hello world"
        
        args = parser.parse_args(["agent", "--message", "Hello world"])
        assert args.message == "Hello world"
    
    def test_parser_file_argument(self):
        """Test file argument parsing."""
        parser = create_parser()
        args = parser.parse_args(["agent", "-f", "input.txt"])
        assert args.file == "input.txt"
        
        args = parser.parse_args(["agent", "--file", "input.txt"])
        assert args.file == "input.txt"
    
    def test_parser_list_argument(self):
        """Test list argument parsing."""
        parser = create_parser()
        args = parser.parse_args(["-l"])
        assert args.list is True
        
        args = parser.parse_args(["--list"])
        assert args.list is True
    
    def test_parser_quiet_argument(self):
        """Test quiet argument parsing."""
        parser = create_parser()
        args = parser.parse_args(["agent", "-q"])
        assert args.quiet is True
        
        args = parser.parse_args(["agent", "--quiet"])
        assert args.quiet is True
    
    def test_parser_debug_argument(self):
        """Test debug argument parsing."""
        parser = create_parser()
        args = parser.parse_args(["agent", "--debug"])
        assert args.debug is True


class TestAsyncMain:
    """Test async main function."""
    
    @pytest.fixture
    def mock_cli(self):
        """Mock AgentCLI class."""
        with patch('flowlib.agent.runners.cli.AgentCLI') as mock:
            mock_instance = Mock()
            mock.return_value = mock_instance
            yield mock_instance
    
    @pytest.mark.asyncio
    async def test_async_main_with_agent_name(self, mock_cli):
        """Test async_main with specific agent name."""
        mock_cli.list_agents.return_value = None
        mock_cli.run_agent = AsyncMock(return_value=0)
        
        with patch('sys.argv', ['script', '--list']):
            result = await async_main("test-agent")
            assert result == 0
            mock_cli.list_agents.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_main_no_args(self, mock_cli):
        """Test async_main with no arguments."""
        with patch('sys.argv', ['script']):
            result = await async_main()
            assert result == 1  # Should return 1 when no agent specified
    
    @pytest.mark.asyncio
    async def test_async_main_list_agents(self, mock_cli):
        """Test async_main with list option."""
        mock_cli.list_agents.return_value = None
        
        with patch('sys.argv', ['script', '--list']):
            result = await async_main()
            assert result == 0
            mock_cli.list_agents.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_main_run_agent(self, mock_cli):
        """Test async_main running an agent."""
        mock_cli.run_agent = AsyncMock(return_value=0)
        
        with patch('sys.argv', ['script', 'test-agent']):
            result = await async_main()
            assert result == 0
            mock_cli.run_agent.assert_called_once_with('test-agent', mock_cli.run_agent.call_args[0][1])


class TestMainFunction:
    """Test main entry point function."""
    
    @patch('flowlib.agent.runners.cli.asyncio.run')
    def test_main_success(self, mock_run):
        """Test main function success."""
        mock_run.return_value = 0
        
        result = main()
        assert result == 0
        mock_run.assert_called_once()
    
    @patch('flowlib.agent.runners.cli.asyncio.run')
    def test_main_keyboard_interrupt(self, mock_run):
        """Test main function with KeyboardInterrupt."""
        mock_run.side_effect = KeyboardInterrupt()
        
        result = main()
        assert result == 0  # Should return 0 on KeyboardInterrupt
    
    @patch('flowlib.agent.runners.cli.asyncio.run')
    def test_main_exception(self, mock_run):
        """Test main function with exception."""
        mock_run.side_effect = Exception("Test error")
        
        result = main()
        assert result == 1  # Should return 1 on exception
    
    @patch('flowlib.agent.runners.cli.asyncio.run')
    def test_main_with_agent_name(self, mock_run):
        """Test main function with specific agent name."""
        mock_run.return_value = 0
        
        result = main("test-agent")
        assert result == 0
        mock_run.assert_called_once()


class TestAgentRunning:
    """Test agent running functionality."""
    
    @pytest.fixture
    def cli_with_config(self):
        """Create CLI with mock config setup."""
        cli = AgentCLI()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".flowlib"
            agents_dir = config_dir / "agents"
            agents_dir.mkdir(parents=True)
            cli.config_dir = config_dir
            cli.agents_dir = agents_dir
            
            # Create test config
            test_config = {
                "name": "test-agent",
                "persona": {
                    "name": "assistant",
                    "system_prompt": "You are helpful",
                    "personality": "friendly"
                },
                "mode": {
                    "name": "chat",
                    "settings": {"memory_enabled": True, "max_turns": 10}
                },
                "interface": "repl",
                "provider": {"provider_type": "llamacpp"}
            }
            
            config_file = agents_dir / "test-agent.json"
            with open(config_file, 'w') as f:
                json.dump(test_config, f)
            
            yield cli, test_config
    
    @pytest.mark.asyncio
    @patch('flowlib.agent.runners.cli.BaseAgent')
    @patch('flowlib.agent.runners.repl.start_agent_repl')
    @patch('flowlib.agent.runners.cli.run_interactive_session')
    async def test_run_agent_interactive(self, mock_interactive, mock_repl, mock_orchestrator, cli_with_config):
        """Test running agent in interactive mode."""
        cli, config = cli_with_config
        
        # Setup mocks
        mock_agent = Mock()
        mock_agent.initialize = AsyncMock()
        mock_agent.shutdown = AsyncMock()
        mock_orchestrator.return_value = mock_agent
        mock_repl.return_value = None
        mock_interactive.return_value = None
        
        # Create args
        args = argparse.Namespace(
            message=None, file=None, quiet=False, debug=False
        )
        
        result = await cli.run_agent("test-agent", args)
        
        assert result == 0
        mock_agent.initialize.assert_called_once()
        mock_agent.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('flowlib.agent.runners.cli.BaseAgent')
    async def test_run_agent_single_message(self, mock_orchestrator, cli_with_config):
        """Test running agent with single message."""
        cli, config = cli_with_config
        
        # Setup mocks
        mock_agent = Mock()
        mock_agent.initialize = AsyncMock()
        mock_agent.shutdown = AsyncMock()
        mock_agent.process_message = AsyncMock(return_value="Test response")
        mock_orchestrator.return_value = mock_agent
        
        # Create args
        args = argparse.Namespace(
            message="Hello", file=None, quiet=True, debug=False
        )
        
        result = await cli.run_agent("test-agent", args)
        
        assert result == 0
        mock_agent.process_message.assert_called_once_with("Hello")
    
    @pytest.mark.asyncio
    @patch('flowlib.agent.runners.cli.BaseAgent')
    async def test_run_agent_file_input(self, mock_orchestrator, cli_with_config):
        """Test running agent with file input."""
        cli, config = cli_with_config
        
        # Setup mocks
        mock_agent = Mock()
        mock_agent.initialize = AsyncMock()
        mock_agent.shutdown = AsyncMock()
        mock_agent.process_message = AsyncMock(return_value="File response")
        mock_orchestrator.return_value = mock_agent
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Test file content")
            test_file = f.name
        
        try:
            # Create args
            args = argparse.Namespace(
                message=None, file=test_file, quiet=True, debug=False
            )
            
            result = await cli.run_agent("test-agent", args)
            
            assert result == 0
            mock_agent.process_message.assert_called_once_with("Test file content")
        finally:
            os.unlink(test_file)
    
    @pytest.mark.asyncio
    async def test_run_agent_nonexistent(self, cli_with_config):
        """Test running non-existent agent."""
        cli, config = cli_with_config
        
        args = argparse.Namespace(
            message=None, file=None, quiet=False, debug=False
        )
        
        result = await cli.run_agent("nonexistent", args)
        assert result == 1
    
    @pytest.mark.asyncio
    @patch('flowlib.agent.runners.cli.BaseAgent')
    @patch('flowlib.agent.runners.repl.start_agent_repl', side_effect=KeyboardInterrupt())
    async def test_run_agent_keyboard_interrupt(self, mock_repl, mock_orchestrator, cli_with_config):
        """Test running agent with KeyboardInterrupt."""
        cli, config = cli_with_config
        
        # Setup mocks
        mock_agent = Mock()
        mock_agent.initialize = AsyncMock()
        mock_agent.shutdown = AsyncMock()
        mock_orchestrator.return_value = mock_agent
        
        args = argparse.Namespace(
            message=None, file=None, quiet=False, debug=False
        )
        
        result = await cli.run_agent("test-agent", args)
        assert result == 0  # Should handle KeyboardInterrupt gracefully
    
    @pytest.mark.asyncio
    @patch('flowlib.agent.runners.cli.BaseAgent')
    async def test_run_agent_exception(self, mock_orchestrator, cli_with_config):
        """Test running agent with exception."""
        cli, config = cli_with_config
        
        # Setup mocks to raise exception
        mock_orchestrator.side_effect = Exception("Test error")
        
        args = argparse.Namespace(
            message=None, file=None, quiet=False, debug=False
        )
        
        result = await cli.run_agent("test-agent", args)
        assert result == 1  # Should return 1 on exception
    
    @pytest.mark.asyncio
    @patch('flowlib.agent.runners.cli.BaseAgent')
    async def test_run_agent_exception_with_debug(self, mock_orchestrator, cli_with_config):
        """Test running agent with exception and debug enabled."""
        cli, config = cli_with_config
        
        # Setup mocks to raise exception
        mock_orchestrator.side_effect = Exception("Test error")
        
        args = argparse.Namespace(
            message=None, file=None, quiet=False, debug=True
        )
        
        with patch('traceback.print_exc') as mock_traceback:
            result = await cli.run_agent("test-agent", args)
            assert result == 1
            mock_traceback.assert_called_once()  # Should print traceback in debug mode


if __name__ == "__main__":
    pytest.main([__file__, "-v"])