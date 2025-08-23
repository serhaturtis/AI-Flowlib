"""Tests for shell command execution flow."""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch, PropertyMock
import os
import time

from flowlib.agent.components.shell_command.flow import ShellCommandFlow
from flowlib.agent.components.shell_command.models import (
    ShellCommandIntentInput,
    ShellCommandOutput,
    GeneratedCommand
)


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = AsyncMock()
    provider.generate_structured = AsyncMock()
    return provider


@pytest.fixture
def mock_prompt_resource():
    """Create a mock prompt resource."""
    prompt = Mock()
    prompt.render = Mock(return_value="Test prompt")
    return prompt


@pytest.fixture
def shell_command_flow():
    """Create a ShellCommandFlow instance."""
    return ShellCommandFlow()


class TestShellCommandFlow:
    """Test the ShellCommandFlow class."""

    @pytest.mark.asyncio
    async def test_is_command_available(self, shell_command_flow):
        """Test checking if a command is available."""
        # Common commands that should exist
        assert await shell_command_flow._is_command_available("echo") is True
        assert await shell_command_flow._is_command_available("ls") is True
        
        # Command that should not exist
        assert await shell_command_flow._is_command_available("nonexistentcommand123") is False

    def test_parse_primary_command(self, shell_command_flow):
        """Test parsing primary command from command string."""
        # Simple commands
        assert shell_command_flow._parse_primary_command("ls -la") == "ls"
        assert shell_command_flow._parse_primary_command("echo hello") == "echo"
        assert shell_command_flow._parse_primary_command("git status") == "git"
        
        # Commands with pipes and operators
        assert shell_command_flow._parse_primary_command("ls | grep test") == "ls"
        assert shell_command_flow._parse_primary_command("cd /tmp && ls") == "cd"
        
        # Edge cases
        assert shell_command_flow._parse_primary_command("") == ""
        assert shell_command_flow._parse_primary_command("   ") == ""
        assert shell_command_flow._parse_primary_command("   echo   ") == "echo"

    @pytest.mark.asyncio
    async def test_get_available_commands(self, shell_command_flow):
        """Test getting available commands."""
        with patch.object(shell_command_flow, '_is_command_available') as mock_is_available:
            # Mock some commands as available
            def side_effect(cmd):
                return cmd in ["echo", "ls", "grep", "cat"]
            
            mock_is_available.side_effect = side_effect
            
            available = await shell_command_flow._get_available_commands()
            
            # Should include only available commands
            assert "echo" in available
            assert "ls" in available
            assert "grep" in available
            assert "cat" in available
            
            # Should not include unavailable commands
            assert "docker" not in available
            assert "kubectl" not in available

    @pytest.mark.asyncio
    async def test_generate_command_success(self, shell_command_flow, mock_llm_provider, mock_prompt_resource):
        """Test successful command generation."""
        # Setup input
        input_data = ShellCommandIntentInput(
            intent="List all Python files",
            target_resource="/tmp",
            parameters={"recursive": True},
            output_description="List of Python files"
        )
        
        # Mock LLM response
        generated_command = GeneratedCommand(
            command="find /tmp -name '*.py' -type f",
            reasoning="Using find command to recursively search for Python files"
        )
        mock_llm_provider.generate_structured.return_value = generated_command
        
        # Mock dependencies
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config', return_value=mock_llm_provider), \
             patch('flowlib.resources.registry.registry.resource_registry.get', return_value=mock_prompt_resource), \
             patch.object(shell_command_flow, '_get_available_commands', return_value=["find", "ls", "echo"]):
            
            result = await shell_command_flow._generate_command(input_data)
            
            assert result.command == "find /tmp -name '*.py' -type f"
            assert result.reasoning == "Using find command to recursively search for Python files"
            
            # Verify LLM was called with correct parameters
            mock_llm_provider.generate_structured.assert_called_once()
            call_args = mock_llm_provider.generate_structured.call_args
            assert call_args.kwargs['prompt'] == mock_prompt_resource
            assert call_args.kwargs['output_type'] == GeneratedCommand
            assert 'prompt_variables' in call_args.kwargs

    @pytest.mark.asyncio
    async def test_generate_command_no_llm_provider(self, shell_command_flow):
        """Test command generation when LLM provider is not available."""
        input_data = ShellCommandIntentInput(intent="Test command")
        
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config', return_value=None):
            with pytest.raises(RuntimeError, match="Could not get LLM provider"):
                await shell_command_flow._generate_command(input_data)

    @pytest.mark.asyncio
    async def test_generate_command_no_prompt(self, shell_command_flow, mock_llm_provider):
        """Test command generation when prompt is not available."""
        input_data = ShellCommandIntentInput(intent="Test command")
        
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config', return_value=mock_llm_provider), \
             patch('flowlib.resources.registry.registry.resource_registry.get', return_value=None), \
             patch.object(shell_command_flow, '_get_available_commands', return_value=["echo"]):
            with pytest.raises(RuntimeError, match="Could not find shell_command_generation prompt"):
                await shell_command_flow._generate_command(input_data)

    @pytest.mark.asyncio
    async def test_execute_command_success(self, shell_command_flow):
        """Test successful command execution."""
        # Mock process
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"Hello, World!", b"")
        
        with patch.object(shell_command_flow, '_get_available_commands', return_value=["echo"]), \
             patch('asyncio.create_subprocess_shell', return_value=mock_process):
            
            result = await shell_command_flow._execute_command(
                command="echo 'Hello, World!'",
                working_dir="/tmp",
                timeout=30
            )
            
            assert result.command == "echo 'Hello, World!'"
            assert result.exit_code == 0
            assert result.stdout == "Hello, World!"
            assert result.stderr == ""
            assert result.success is True
            assert result.working_dir == "/tmp"
            assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_execute_command_failure(self, shell_command_flow):
        """Test command execution failure."""
        # Mock process with non-zero exit code
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b"", b"Command not found")
        
        with patch.object(shell_command_flow, '_get_available_commands', return_value=["ls"]), \
             patch('asyncio.create_subprocess_shell', return_value=mock_process):
            
            result = await shell_command_flow._execute_command(
                command="ls /nonexistent",
                working_dir="/tmp",
                timeout=30
            )
            
            assert result.exit_code == 1
            assert result.stdout == ""
            assert result.stderr == "Command not found"
            assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_command_not_allowed(self, shell_command_flow):
        """Test execution of command not in allowed list."""
        with patch.object(shell_command_flow, '_get_available_commands', return_value=["echo", "ls"]):
            result = await shell_command_flow._execute_command(
                command="rm -rf /",  # Dangerous command not in allowed list
                working_dir="/tmp",
                timeout=30
            )
            
            assert result.exit_code == -1
            assert result.success is False
            assert "not available or not allowed" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_command_timeout(self, shell_command_flow):
        """Test command execution timeout."""
        # Mock process that times out
        mock_process = AsyncMock()
        mock_process.communicate.side_effect = asyncio.TimeoutError()
        # kill() is not async in real subprocess, so use regular Mock
        mock_process.kill = Mock()
        mock_process.wait = AsyncMock(return_value=0)
        
        with patch.object(shell_command_flow, '_get_available_commands', return_value=["sleep"]), \
             patch('asyncio.create_subprocess_shell', return_value=mock_process):
            
            result = await shell_command_flow._execute_command(
                command="sleep 100",
                working_dir="/tmp",
                timeout=1
            )
            
            assert result.exit_code == -1
            assert result.success is False
            assert "timed out" in result.stderr
            mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_command_exception(self, shell_command_flow):
        """Test command execution with exception."""
        with patch.object(shell_command_flow, '_get_available_commands', return_value=["echo"]), \
             patch('asyncio.create_subprocess_shell', side_effect=Exception("Test error")):
            
            result = await shell_command_flow._execute_command(
                command="echo test",
                working_dir="/tmp",
                timeout=30
            )
            
            assert result.exit_code == -1
            assert result.success is False
            assert "Execution error: Test error" in result.stderr

    @pytest.mark.asyncio
    async def test_run_pipeline_success(self, shell_command_flow, mock_llm_provider, mock_prompt_resource):
        """Test successful pipeline execution."""
        # Setup input
        input_data = ShellCommandIntentInput(
            intent="Show current directory",
            output_description="Display current working directory"
        )
        
        # Mock command generation
        generated_command = GeneratedCommand(
            command="pwd",
            reasoning="pwd shows the current working directory"
        )
        mock_llm_provider.generate_structured.return_value = generated_command
        
        # Mock process
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"/home/user", b"")
        
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config', return_value=mock_llm_provider), \
             patch('flowlib.resources.registry.registry.resource_registry.get', return_value=mock_prompt_resource), \
             patch.object(shell_command_flow, '_get_available_commands', return_value=["pwd"]), \
             patch('asyncio.create_subprocess_shell', return_value=mock_process):
            
            result = await shell_command_flow.run_pipeline(input_data)
            
            assert result.command == "pwd"
            assert result.stdout == "/home/user"
            assert result.success is True
            assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_run_pipeline_with_working_dir(self, shell_command_flow, mock_llm_provider, mock_prompt_resource):
        """Test pipeline execution with custom working directory."""
        # Setup input with working directory
        input_data = ShellCommandIntentInput(
            intent="List files",
            working_dir="/usr/local",
            timeout=45
        )
        
        # Mock command generation
        generated_command = GeneratedCommand(
            command="ls -la",
            reasoning="List all files with details"
        )
        mock_llm_provider.generate_structured.return_value = generated_command
        
        # Mock process
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"file1\nfile2", b"")
        
        with patch('flowlib.providers.core.registry.provider_registry.get_by_config', return_value=mock_llm_provider), \
             patch('flowlib.resources.registry.registry.resource_registry.get', return_value=mock_prompt_resource), \
             patch.object(shell_command_flow, '_get_available_commands', return_value=["ls"]), \
             patch('asyncio.create_subprocess_shell', return_value=mock_process) as mock_subprocess:
            
            result = await shell_command_flow.run_pipeline(input_data)
            
            # Verify subprocess was called with correct working directory
            mock_subprocess.assert_called_once()
            assert mock_subprocess.call_args.kwargs['cwd'] == "/usr/local"
            
            assert result.working_dir == "/usr/local"
            assert result.success is True

    @pytest.mark.asyncio
    async def test_run_pipeline_error_handling(self, shell_command_flow):
        """Test pipeline error handling."""
        input_data = ShellCommandIntentInput(intent="Test command")
        
        # Mock command generation to raise exception
        with patch.object(shell_command_flow, '_generate_command', side_effect=Exception("Generation failed")):
            result = await shell_command_flow.run_pipeline(input_data)
            
            assert result.command == ""
            assert result.exit_code == -1
            assert result.success is False
            assert "Flow error: Generation failed" in result.stderr
            assert result.execution_time == 0.0

    @pytest.mark.asyncio
    async def test_flow_decorators(self):
        """Test that flow decorators are properly applied."""
        # Verify the flow decorator
        assert hasattr(ShellCommandFlow, '__flow_metadata__')
        assert ShellCommandFlow.__flow_metadata__['name'] == 'shell-command'
        assert ShellCommandFlow.__flow_metadata__['is_infrastructure'] is False
        
        # Check that the flow has get_description method
        flow_instance = ShellCommandFlow()
        assert hasattr(flow_instance, 'get_description')
        assert flow_instance.get_description() == "Execute shell commands on the local system based on high-level intent"
        
        # Verify the pipeline decorator
        assert hasattr(ShellCommandFlow.run_pipeline, '__pipeline__')
        assert ShellCommandFlow.run_pipeline.__pipeline__ is True
        assert hasattr(ShellCommandFlow.run_pipeline, '__input_model__')
        assert hasattr(ShellCommandFlow.run_pipeline, '__output_model__')
        assert ShellCommandFlow.run_pipeline.__input_model__ == ShellCommandIntentInput
        assert ShellCommandFlow.run_pipeline.__output_model__ == ShellCommandOutput

    def test_shell_command_output_display(self):
        """Test the get_user_display method of ShellCommandOutput."""
        # Success with output
        output = ShellCommandOutput(
            command="echo 'Hello'",
            exit_code=0,
            stdout="Hello",
            stderr="",
            execution_time=0.5,
            success=True,
            working_dir="/tmp"
        )
        display = output.get_user_display()
        assert "Command executed successfully" in display
        assert "echo 'Hello'" in display
        assert "Hello" in display
        
        # Failure with error
        output = ShellCommandOutput(
            command="bad_command",
            exit_code=1,
            stdout="",
            stderr="command not found",
            execution_time=0.1,
            success=False,
            working_dir="/tmp"
        )
        display = output.get_user_display()
        assert "Command failed" in display
        assert "bad_command" in display
        assert "command not found" in display
        
        # Success with no output
        output = ShellCommandOutput(
            command="touch file.txt",
            exit_code=0,
            stdout="",
            stderr="",
            execution_time=0.2,
            success=True,
            working_dir="/tmp"
        )
        display = output.get_user_display()
        assert "Command executed" in display
        assert "(No output)" in display
        
        # No command
        output = ShellCommandOutput(
            command="",
            exit_code=-1,
            stdout="",
            stderr="",
            execution_time=0.0,
            success=False,
            working_dir="/tmp"
        )
        display = output.get_user_display()
        assert "Shell command executed" in display


class TestShellCommandIntegration:
    """Integration tests for shell command flow."""

    @pytest.mark.asyncio
    async def test_real_command_execution(self, shell_command_flow):
        """Test executing a real command (integration test)."""
        # This test actually executes a command
        result = await shell_command_flow._execute_command(
            command="echo 'Integration Test'",
            working_dir=os.getcwd(),
            timeout=5
        )
        
        assert result.success is True
        assert result.exit_code == 0
        assert "Integration Test" in result.stdout
        assert result.stderr == ""
        assert result.execution_time < 5.0

    @pytest.mark.asyncio
    async def test_command_with_unicode_output(self, shell_command_flow):
        """Test handling unicode in command output."""
        # Test command that outputs unicode
        result = await shell_command_flow._execute_command(
            command="echo '你好世界 🌍'",
            working_dir=os.getcwd(),
            timeout=5
        )
        
        assert result.success is True
        assert "你好世界" in result.stdout
        assert "🌍" in result.stdout