"""
Comprehensive tests for the REPL (Read-Eval-Print Loop) system.

Tests cover the interactive agent session functionality, command processing,
and user interaction workflows.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime

from flowlib.agent.runners.repl.agent_repl import AgentREPL, InteractiveAgent
from flowlib.agent.runners.repl.commands import CommandRegistry, CommandType, Command
from flowlib.agent.runners.repl.handlers import DefaultCommandHandler, AgentCommandHandler
from flowlib.agent.models.config import AgentConfig
from flowlib.agent.models.state import AgentState
from flowlib.agent.core.base_agent import BaseAgent
from flowlib.agent.runners.repl.agent_repl import start_agent_repl


class TestAgentREPLInitialization:
    """Test REPL initialization and configuration."""

    def test_repl_creation_with_agent(self):
        """Test creating REPL with existing agent."""
        mock_agent = MagicMock()
        
        repl = AgentREPL(agent=mock_agent)
        
        assert repl.agent is mock_agent
        assert repl.context["agent"] is mock_agent
        assert isinstance(repl.command_registry, CommandRegistry)
        assert not repl.context["should_exit"]

    def test_repl_creation_with_config(self):
        """Test creating REPL with config."""
        config = AgentConfig(name="test_agent", persona="Test agent", provider_name="mock")
        
        with patch('flowlib.agent.core.Agent') as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            
            repl = AgentREPL(config=config)
            
            assert repl.config is config
            assert repl.agent is mock_agent_instance
            assert repl.context["config"] is config

    def test_repl_creation_no_agent_no_config(self):
        """Test creating REPL without agent or config."""
        repl = AgentREPL()
        
        assert repl.agent is None
        assert repl.config is None
        assert repl.context["agent"] is None

    def test_repl_history_setup(self):
        """Test command history setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = os.path.join(temp_dir, "test_history")
            
            repl = AgentREPL(history_file=history_file)
            
            assert repl.history_file == history_file

    def test_repl_default_context(self):
        """Test default context values."""
        repl = AgentREPL()
        
        context = repl.context
        assert context["should_exit"] is False
        assert context["command_history"] == []
        assert context["mode"] == "chat"
        assert context["debug"] is False
        assert context["verbose"] is False
        assert context["stream"] is True
        assert "session_stats" in context
        assert context["session_stats"]["message_count"] == 0

    def test_repl_session_stats_initialization(self):
        """Test session statistics initialization."""
        repl = AgentREPL()
        
        stats = repl.context["session_stats"]
        assert stats["message_count"] == 0
        assert stats["tokens_used"] == 0
        assert stats["tools_called"] == 0
        assert stats["flows_executed"] == 0
        assert isinstance(stats["start_time"], datetime)


class TestAgentREPLCommandHandling:
    """Test REPL command processing."""

    @pytest.fixture
    def mock_repl(self):
        """Create a REPL with mocked components."""
        mock_agent = MagicMock()
        mock_agent.process_message = AsyncMock(return_value={
            "content": "Test response",
            "stats": {"flows_executed": 1}
        })
        
        repl = AgentREPL(agent=mock_agent)
        repl.console = MagicMock()
        return repl

    @pytest.mark.asyncio
    async def test_user_message_handling(self, mock_repl):
        """Test handling of user messages."""
        message = "Hello, how are you?"
        
        # Mock the command registry to return a user command
        mock_command = Command(
            type=CommandType.USER,
            name="user_input",
            args=[],
            raw_input=message
        )
        mock_repl.command_registry.parse_command = MagicMock(return_value=mock_command)
        mock_repl.command_registry.execute_command = AsyncMock(return_value=None)
        
        await mock_repl._handle_user_message(message)
        
        # Verify agent was called
        mock_repl.agent.process_message.assert_called_once_with(
            message=message,
            context=mock_repl.context
        )
        
        # Verify stats were updated
        assert mock_repl.context["session_stats"]["message_count"] == 1
        assert mock_repl.context["session_stats"]["flows_executed"] == 1

    @pytest.mark.asyncio
    async def test_agent_response_formatting(self, mock_repl):
        """Test agent response formatting and display."""
        message = "Test message"
        
        # Mock agent response with activity and content
        mock_response = {
            "content": "Agent response",
            "activity": "Planning: Analyzing request\nExecution: Processing task",
            "stats": {"flows_executed": 2}
        }
        mock_repl.agent.process_message = AsyncMock(return_value=mock_response)
        
        response = await mock_repl._get_agent_response(message)
        
        # Should combine activity and content
        expected_lines = [
            "Planning: Analyzing request\nExecution: Processing task",
            "",  # Empty line separator
            "Agent response"
        ]
        assert response == "\n".join(expected_lines)

    @pytest.mark.asyncio
    async def test_agent_response_error_handling(self, mock_repl):
        """Test error handling in agent responses."""
        message = "Test message"
        
        # Mock agent to raise an exception
        mock_repl.agent.process_message = AsyncMock(side_effect=Exception("Test error"))
        
        response = await mock_repl._get_agent_response(message)
        
        assert "Error getting agent response: Test error" in response

    @pytest.mark.asyncio
    async def test_agent_response_debug_mode(self, mock_repl):
        """Test agent response error handling in debug mode."""
        message = "Test message"
        mock_repl.repl_context.debug = True
        mock_repl.context["debug"] = True
        
        # Mock agent to raise an exception
        mock_repl.agent.process_message = AsyncMock(side_effect=Exception("Test error"))
        
        response = await mock_repl._get_agent_response(message)
        
        assert "Error getting agent response: Test error" in response
        assert "Traceback" in response

    @pytest.mark.asyncio
    async def test_no_agent_message_handling(self, mock_repl):
        """Test handling messages when no agent is available."""
        mock_repl.agent = None
        
        await mock_repl._handle_user_message("test")
        
        # Should print error message
        mock_repl.console.print.assert_called_with("[red]No agent initialized![/red]")

    def test_display_string_output(self, mock_repl):
        """Test displaying string output."""
        output = "Test string output"
        
        mock_repl._display_output(output)
        
        # Should display as Markdown
        from rich.markdown import Markdown
        mock_repl.console.print.assert_called_once()
        call_args = mock_repl.console.print.call_args[0][0]
        assert isinstance(call_args, Markdown)

    def test_display_empty_string_output(self, mock_repl):
        """Test displaying empty string output."""
        output = "   "  # Whitespace only
        
        mock_repl._display_output(output)
        
        # Should not display empty strings
        mock_repl.console.print.assert_not_called()

    def test_display_non_string_output(self, mock_repl):
        """Test displaying non-string output."""
        output = {"key": "value"}
        
        mock_repl._display_output(output)
        
        # Should display object directly
        mock_repl.console.print.assert_called_once_with(output)


class TestAgentREPLActivityStreaming:
    """Test real-time activity streaming."""

    @pytest.fixture
    def streaming_repl(self):
        """Create REPL with streaming enabled."""
        mock_agent = MagicMock()
        mock_agent.process_message = AsyncMock(return_value={
            "content": "Response",
            "activity": "Activity log"
        })
        mock_agent.set_activity_stream_handler = MagicMock()
        
        repl = AgentREPL(agent=mock_agent)
        repl.console = MagicMock()
        repl.context["stream"] = True
        return repl

    @pytest.mark.asyncio
    async def test_activity_streaming_enabled(self, streaming_repl):
        """Test activity streaming when enabled."""
        message = "Test message"
        
        await streaming_repl._handle_user_message(message)
        
        # Should set up activity stream handler
        streaming_repl.agent.set_activity_stream_handler.assert_called_once()
        
        # Check that handler function was passed
        call_args = streaming_repl.agent.set_activity_stream_handler.call_args[0][0]
        assert callable(call_args)

    @pytest.mark.asyncio
    async def test_activity_streaming_disabled(self, streaming_repl):
        """Test when activity streaming is disabled."""
        streaming_repl.repl_context.stream = False
        streaming_repl.context["stream"] = False
        message = "Test message"
        
        await streaming_repl._handle_user_message(message)
        
        # Should not set up activity stream handler when disabled
        streaming_repl.agent.set_activity_stream_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_activity_handler_function(self, streaming_repl):
        """Test the activity handler function."""
        message = "Test message"
        
        await streaming_repl._handle_user_message(message)
        
        # Get the handler function that was passed
        handler = streaming_repl.agent.set_activity_stream_handler.call_args[0][0]
        
        # Test that it prints activity
        test_activity = "Test activity message"
        handler(test_activity)
        
        streaming_repl.console.print.assert_called_with(test_activity)


class TestAgentREPLMainLoop:
    """Test REPL main loop functionality."""

    @pytest.fixture
    def loop_repl(self):
        """Create REPL for loop testing."""
        repl = AgentREPL()
        repl.console = MagicMock()
        repl.command_registry = MagicMock()
        return repl

    @pytest.mark.asyncio
    async def test_repl_loop_exit_command(self, loop_repl):
        """Test REPL loop with exit command."""
        # Mock input to return exit command
        with patch('rich.prompt.Prompt.ask', side_effect=["exit", EOFError]):
            # Mock command that sets should_exit
            mock_command = MagicMock()
            loop_repl.command_registry.parse_command.return_value = mock_command
            
            def set_exit_flag(command, context):
                context["should_exit"] = True
                return "Goodbye!"
            
            loop_repl.command_registry.execute_command = AsyncMock(side_effect=set_exit_flag)
            
            await loop_repl._repl_loop()
            
            assert loop_repl.context["should_exit"]

    @pytest.mark.asyncio
    async def test_repl_loop_keyboard_interrupt(self, loop_repl):
        """Test REPL loop handling keyboard interrupt."""
        with patch('rich.prompt.Prompt.ask', side_effect=KeyboardInterrupt):
            loop_repl.command_registry.parse_command.return_value = MagicMock()
            loop_repl.command_registry.execute_command = AsyncMock()
            
            # Should handle KeyboardInterrupt gracefully
            await loop_repl._repl_loop()

    @pytest.mark.asyncio
    async def test_repl_loop_exception_handling(self, loop_repl):
        """Test REPL loop exception handling."""
        with patch('rich.prompt.Prompt.ask', side_effect=["test", EOFError]):
            loop_repl.command_registry.parse_command.side_effect = Exception("Test error")
            
            await loop_repl._repl_loop()
            
            # Should print error message
            loop_repl.console.print.assert_any_call("\n[red]Error: Test error[/red]")

    @pytest.mark.asyncio
    async def test_repl_loop_debug_exception(self, loop_repl):
        """Test REPL loop exception handling in debug mode."""
        loop_repl.context["debug"] = True
        
        with patch('rich.prompt.Prompt.ask', side_effect=["test", EOFError]):
            loop_repl.command_registry.parse_command.side_effect = Exception("Test error")
            
            await loop_repl._repl_loop()
            
            # Should print error and traceback
            calls = [call[0][0] for call in loop_repl.console.print.call_args_list]
            error_printed = any("\n[red]Error: Test error[/red]" in str(call) for call in calls)
            assert error_printed

    def test_get_input_prompt_modes(self, loop_repl):
        """Test input prompt for different modes."""
        test_cases = [
            ("chat", "💬"),
            ("task", "🎯"),
            ("debug", "🐛"),
            ("unknown", "💬")  # fallback
        ]
        
        for mode, expected_emoji in test_cases:
            loop_repl.repl_context.mode = mode
            loop_repl.context["mode"] = mode
            
            with patch('rich.prompt.Prompt.ask', return_value="test") as mock_ask:
                result = loop_repl._get_input()
                
                assert result == "test"
                # Check that the prompt contains the expected emoji
                prompt_arg = mock_ask.call_args[0][0]
                assert expected_emoji in prompt_arg

    def test_get_input_eof_handling(self, loop_repl):
        """Test input handling for EOF."""
        with patch('rich.prompt.Prompt.ask', side_effect=EOFError):
            result = loop_repl._get_input()
            assert result is None

    def test_get_input_keyboard_interrupt_handling(self, loop_repl):
        """Test input handling for keyboard interrupt."""
        with patch('rich.prompt.Prompt.ask', side_effect=KeyboardInterrupt):
            result = loop_repl._get_input()
            assert result is None


class TestAgentREPLSessionManagement:
    """Test REPL session management."""

    @pytest.fixture
    def session_repl(self):
        """Create REPL for session testing."""
        repl = AgentREPL()
        repl.console = MagicMock()
        return repl

    @pytest.mark.asyncio
    async def test_start_session_welcome(self, session_repl):
        """Test session start with welcome message."""
        with patch.object(session_repl, '_repl_loop') as mock_loop:
            mock_loop.return_value = None
            
            await session_repl.start()
            
            # Should call _repl_loop
            mock_loop.assert_called_once()
            
            # Should print welcome (check that console.print was called)
            assert session_repl.console.print.called

    @pytest.mark.asyncio
    async def test_session_keyboard_interrupt(self, session_repl):
        """Test session handling keyboard interrupt."""
        with patch.object(session_repl, '_repl_loop', side_effect=KeyboardInterrupt):
            with patch.object(session_repl, '_save_history') as mock_save:
                await session_repl.start()
                
                # Should save history on exit
                mock_save.assert_called_once()
                
                # Should print interrupt message
                calls = [str(call) for call in session_repl.console.print.call_args_list]
                interrupt_printed = any("Interrupted by user" in call for call in calls)
                assert interrupt_printed

    @pytest.mark.asyncio
    async def test_session_eof(self, session_repl):
        """Test session handling EOF."""
        with patch.object(session_repl, '_repl_loop', side_effect=EOFError):
            with patch.object(session_repl, '_save_history') as mock_save:
                await session_repl.start()
                
                # Should save history on exit
                mock_save.assert_called_once()

    def test_show_goodbye_stats(self, session_repl):
        """Test goodbye message with session stats."""
        # Set up some session stats
        session_repl.context["session_stats"].update({
            "message_count": 5,
            "tokens_used": 1000,
            "tools_called": 3,
            "flows_executed": 2
        })
        
        session_repl._show_goodbye()
        
        # Should print stats table
        assert session_repl.console.print.called
        
        # Check that print was called multiple times (for table and goodbye)
        assert session_repl.console.print.call_count >= 2

    def test_history_save_success(self, session_repl):
        """Test successful history saving."""
        with patch('flowlib.agent.runners.repl.agent_repl.readline') as mock_readline:
            mock_readline.write_history_file = MagicMock()
            
            session_repl._save_history()
            
            mock_readline.write_history_file.assert_called_once_with(session_repl.history_file)

    def test_history_save_no_readline(self, session_repl):
        """Test history saving when readline is not available."""
        with patch('flowlib.agent.runners.repl.agent_repl.readline', None):
            # Should not raise an exception
            session_repl._save_history()

    def test_history_save_error(self, session_repl):
        """Test history saving with error."""
        with patch('flowlib.agent.runners.repl.agent_repl.readline') as mock_readline:
            mock_readline.write_history_file.side_effect = Exception("Write error")
            
            # Should not raise an exception
            session_repl._save_history()


class TestInteractiveAgent:
    """Test InteractiveAgent factory class."""

    def test_create_repl_default(self):
        """Test creating REPL with default agent type."""
        with patch('flowlib.agent.registry.agent_registry') as mock_registry, \
             patch('flowlib.flows.registry.flow_registry') as mock_flow_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            
            # Setup registry mocks
            mock_flow_registry.get_agent_selectable_flows.return_value = {}
            mock_resource_registry.get.return_value = MagicMock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=MagicMock())
            
            mock_agent_class = MagicMock()
            mock_agent_instance = MagicMock()
            mock_agent_class.return_value = mock_agent_instance
            mock_registry.get.return_value = mock_agent_class
            
            repl = InteractiveAgent.create_repl()
            
            assert isinstance(repl, AgentREPL)
            assert repl.agent is mock_agent_instance
            mock_registry.get.assert_called_once_with("default")

    def test_create_repl_custom_type(self):
        """Test creating REPL with custom agent type."""
        with patch('flowlib.agent.registry.agent_registry') as mock_registry, \
             patch('flowlib.flows.registry.flow_registry') as mock_flow_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            
            # Setup registry mocks
            mock_flow_registry.get_agent_selectable_flows.return_value = {}
            mock_resource_registry.get.return_value = MagicMock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=MagicMock())
            
            mock_agent_class = MagicMock()
            mock_agent_instance = MagicMock()
            mock_agent_class.return_value = mock_agent_instance
            mock_registry.get.return_value = mock_agent_class
            
            custom_config = {"name": "custom_agent", "provider_name": "mock"}
            repl = InteractiveAgent.create_repl(
                agent_type="custom",
                config=custom_config
            )
            
            mock_registry.get.assert_called_once_with("custom")
            mock_agent_class.assert_called_once()

    def test_create_repl_unknown_agent_type(self):
        """Test creating REPL with unknown agent type."""
        with patch('flowlib.agent.registry.agent_registry') as mock_registry, \
             patch('flowlib.flows.registry.flow_registry') as mock_flow_registry, \
             patch('flowlib.resources.registry.registry.resource_registry') as mock_resource_registry, \
             patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            
            # Setup registry mocks
            mock_flow_registry.get_agent_selectable_flows.return_value = {}
            mock_resource_registry.get.return_value = MagicMock()
            mock_provider_registry.get_by_config = AsyncMock(return_value=MagicMock())
            
            mock_registry.get.return_value = None
            
            with pytest.raises(ValueError, match="Unknown agent type: unknown"):
                InteractiveAgent.create_repl(agent_type="unknown")

    @pytest.mark.asyncio
    async def test_start_session(self):
        """Test starting an interactive session."""
        with patch.object(InteractiveAgent, 'create_repl') as mock_create:
            mock_repl = MagicMock()
            mock_repl.start = AsyncMock()
            mock_create.return_value = mock_repl
            
            await InteractiveAgent.start_session(
                agent_type="test",
                config={"name": "test"}
            )
            
            mock_create.assert_called_once_with("test", {"name": "test"})
            mock_repl.start.assert_called_once()


class TestREPLIntegration:
    """Integration tests for REPL system."""

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test a complete conversation flow through REPL."""
        # Create a mock agent with proper response structure
        mock_agent = MagicMock()
        mock_agent.process_message = AsyncMock(return_value={
            "content": "Hello! How can I help you?",
            "activity": "Processing: Analyzing user greeting",
            "stats": {"flows_executed": 1}
        })
        
        repl = AgentREPL(agent=mock_agent)
        repl.console = MagicMock()
        
        # Mock command registry to simulate user input
        mock_command = Command(
            type=CommandType.USER,
            name="user_input",
            args=[],
            raw_input="Hello!"
        )
        repl.command_registry.parse_command = MagicMock(return_value=mock_command)
        repl.command_registry.execute_command = AsyncMock(return_value=None)
        
        # Handle the user message
        await repl._handle_user_message("Hello!")
        
        # Verify the full flow
        mock_agent.process_message.assert_called_once()
        assert repl.context["session_stats"]["message_count"] == 1
        assert repl.context["session_stats"]["flows_executed"] == 1

    @pytest.mark.asyncio
    async def test_command_and_message_handling(self):
        """Test handling of both commands and regular messages."""
        mock_agent = MagicMock()
        mock_agent.process_message = AsyncMock(return_value={"content": "Response"})
        
        repl = AgentREPL(agent=mock_agent)
        repl.console = MagicMock()
        
        # Test command handling
        command_cmd = Command(type=CommandType.SLASH, name="help", args=[], raw_input="/help")
        repl.command_registry.parse_command = MagicMock(return_value=command_cmd)
        repl.command_registry.execute_command = AsyncMock(return_value="Help text")
        
        # Simulate command input
        with patch.object(repl, '_get_input', return_value="/help"):
            # Process one iteration of the loop
            user_input = repl._get_input()
            command = repl.command_registry.parse_command(user_input)
            result = await repl.command_registry.execute_command(command, repl.context)
            
            assert result == "Help text"
            assert command.type == CommandType.SLASH

    def test_context_persistence(self):
        """Test that context persists across interactions."""
        repl = AgentREPL()
        
        # Modify context
        repl.context["custom_key"] = "custom_value"
        repl.context["session_stats"]["message_count"] = 5
        
        # Verify persistence
        assert repl.context["custom_key"] == "custom_value"
        assert repl.context["session_stats"]["message_count"] == 5
        assert repl.context["should_exit"] is False  # Original value preserved


class TestREPLHistoryManagement:
    """Test command history functionality."""

    def test_history_file_default_location(self):
        """Test default history file location."""
        repl = AgentREPL()
        
        expected_path = os.path.expanduser("~/.flowlib_agent_history")
        assert repl.history_file == expected_path

    def test_history_file_custom_location(self):
        """Test custom history file location."""
        custom_path = "/tmp/custom_history"
        repl = AgentREPL(history_file=custom_path)
        
        assert repl.history_file == custom_path

    def test_history_setup_with_readline(self):
        """Test history setup when readline is available."""
        with patch('flowlib.agent.runners.repl.agent_repl.readline') as mock_readline:
            mock_readline.parse_and_bind = MagicMock()
            mock_readline.read_history_file = MagicMock()
            mock_readline.set_history_length = MagicMock()
            
            with tempfile.NamedTemporaryFile() as temp_file:
                repl = AgentREPL(history_file=temp_file.name)
                
                # Should configure readline
                assert mock_readline.parse_and_bind.called
                assert mock_readline.set_history_length.called

    def test_history_setup_without_readline(self):
        """Test history setup when readline is not available."""
        with patch('flowlib.agent.runners.repl.agent_repl.readline', None):
            # Should not raise an exception
            repl = AgentREPL()
            assert repl.history_file is not None


# Convenience function test
@pytest.mark.asyncio
async def test_start_agent_repl_function():
    """Test the convenience start_agent_repl function."""
    mock_agent = MagicMock()
    
    with patch.object(AgentREPL, 'start') as mock_start:
        mock_start.return_value = None
        
        await start_agent_repl(agent=mock_agent)
        
        mock_start.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])