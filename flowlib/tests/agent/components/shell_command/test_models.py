"""Tests for shell command models."""

import pytest
from pydantic import ValidationError

from flowlib.agent.components.shell_command.models import (
    ShellCommandIntentInput,
    GeneratedCommand,
    ShellCommandOutput
)


class TestShellCommandIntentInput:
    """Test the ShellCommandIntentInput model."""

    def test_valid_input_minimal(self):
        """Test creating valid input with minimal fields."""
        input_data = ShellCommandIntentInput(
            intent="List files in current directory"
        )
        
        assert input_data.intent == "List files in current directory"
        assert input_data.target_resource is None
        assert input_data.parameters == {}
        assert input_data.output_description == "Return the standard output."
        assert input_data.working_dir is None
        assert input_data.timeout == 60

    def test_valid_input_full(self):
        """Test creating valid input with all fields."""
        input_data = ShellCommandIntentInput(
            intent="Create a file with specific content",
            target_resource="/tmp/test.txt",
            parameters={"content": "Hello, World!", "permissions": "644"},
            output_description="Return confirmation of file creation",
            working_dir="/tmp",
            timeout=30
        )
        
        assert input_data.intent == "Create a file with specific content"
        assert input_data.target_resource == "/tmp/test.txt"
        assert input_data.parameters == {"content": "Hello, World!", "permissions": "644"}
        assert input_data.output_description == "Return confirmation of file creation"
        assert input_data.working_dir == "/tmp"
        assert input_data.timeout == 30

    def test_missing_required_field(self):
        """Test that intent is required."""
        with pytest.raises(ValidationError) as exc_info:
            ShellCommandIntentInput()
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]['loc'] == ('intent',)
        assert errors[0]['type'] == 'missing'

    def test_empty_intent(self):
        """Test that empty intent is valid but empty."""
        # Empty string is technically valid for Pydantic string fields
        input_data = ShellCommandIntentInput(intent="")
        assert input_data.intent == ""

    def test_parameter_types(self):
        """Test various parameter types."""
        # String parameters
        input_data = ShellCommandIntentInput(
            intent="Test",
            parameters={"key": "value"}
        )
        assert input_data.parameters == {"key": "value"}
        
        # Numeric parameters
        input_data = ShellCommandIntentInput(
            intent="Test",
            parameters={"count": 10, "ratio": 0.5}
        )
        assert input_data.parameters == {"count": 10, "ratio": 0.5}
        
        # Mixed parameters
        input_data = ShellCommandIntentInput(
            intent="Test",
            parameters={"name": "test", "count": 5, "enabled": True}
        )
        assert input_data.parameters == {"name": "test", "count": 5, "enabled": True}


class TestGeneratedCommand:
    """Test the GeneratedCommand model."""

    def test_valid_command(self):
        """Test creating a valid generated command."""
        cmd = GeneratedCommand(
            command="ls -la /tmp",
            reasoning="List all files in /tmp directory with details"
        )
        
        assert cmd.command == "ls -la /tmp"
        assert cmd.reasoning == "List all files in /tmp directory with details"

    def test_missing_required_fields(self):
        """Test that all fields are required."""
        # Missing command
        with pytest.raises(ValidationError) as exc_info:
            GeneratedCommand(reasoning="Test reasoning")
        errors = exc_info.value.errors()
        assert any(error['loc'] == ('command',) for error in errors)
        
        # Missing reasoning
        with pytest.raises(ValidationError) as exc_info:
            GeneratedCommand(command="echo test")
        errors = exc_info.value.errors()
        assert any(error['loc'] == ('reasoning',) for error in errors)

    def test_empty_values(self):
        """Test that empty values are valid but empty."""
        # Empty strings are technically valid for Pydantic string fields
        cmd = GeneratedCommand(command="", reasoning="")
        assert cmd.command == ""
        assert cmd.reasoning == ""


class TestShellCommandOutput:
    """Test the ShellCommandOutput model."""

    def test_valid_output_success(self):
        """Test creating valid output for successful command."""
        output = ShellCommandOutput(
            command="echo 'Hello'",
            exit_code=0,
            stdout="Hello\n",
            stderr="",
            execution_time=0.123,
            success=True,
            working_dir="/home/user"
        )
        
        assert output.command == "echo 'Hello'"
        assert output.exit_code == 0
        assert output.stdout == "Hello\n"
        assert output.stderr == ""
        assert output.execution_time == 0.123
        assert output.success is True
        assert output.working_dir == "/home/user"

    def test_valid_output_failure(self):
        """Test creating valid output for failed command."""
        output = ShellCommandOutput(
            command="ls /nonexistent",
            exit_code=1,
            stdout="",
            stderr="ls: cannot access '/nonexistent': No such file or directory",
            execution_time=0.05,
            success=False,
            working_dir="/tmp"
        )
        
        assert output.exit_code == 1
        assert output.stdout == ""
        assert "No such file or directory" in output.stderr
        assert output.success is False

    # Removed redundant default values test - handled by Pydantic
        """Test default values for optional fields."""
        output = ShellCommandOutput(
            command="test",
            exit_code=0,
            execution_time=1.0,
            success=True,
            working_dir="/tmp"
        )
        
        assert output.stdout == ""
        assert output.stderr == ""

    def test_missing_required_fields(self):
        """Test that required fields must be provided."""
        # Test each required field
        required_fields = [
            'command', 'exit_code', 'execution_time', 'success', 'working_dir'
        ]
        
        for field in required_fields:
            data = {
                'command': 'test',
                'exit_code': 0,
                'execution_time': 1.0,
                'success': True,
                'working_dir': '/tmp'
            }
            del data[field]
            
            with pytest.raises(ValidationError) as exc_info:
                ShellCommandOutput(**data)
            
            errors = exc_info.value.errors()
            assert any(error['loc'] == (field,) for error in errors)

    def test_get_user_display_success_with_output(self):
        """Test user display for successful command with output."""
        output = ShellCommandOutput(
            command="ls -la",
            exit_code=0,
            stdout="total 16\ndrwxr-xr-x 2 user user 4096\n",
            stderr="",
            execution_time=0.1,
            success=True,
            working_dir="/tmp"
        )
        
        display = output.get_user_display()
        assert "Command executed successfully" in display
        assert "```bash" in display
        assert "$ ls -la" in display
        assert "total 16" in display

    def test_get_user_display_failure_with_error(self):
        """Test user display for failed command with error."""
        output = ShellCommandOutput(
            command="rm /protected/file",
            exit_code=1,
            stdout="",
            stderr="rm: cannot remove '/protected/file': Permission denied",
            execution_time=0.05,
            success=False,
            working_dir="/tmp"
        )
        
        display = output.get_user_display()
        assert "Command failed" in display
        assert "```bash" in display
        assert "$ rm /protected/file" in display
        assert "Permission denied" in display

    def test_get_user_display_success_no_output(self):
        """Test user display for successful command with no output."""
        output = ShellCommandOutput(
            command="touch newfile.txt",
            exit_code=0,
            stdout="",
            stderr="",
            execution_time=0.02,
            success=True,
            working_dir="/tmp"
        )
        
        display = output.get_user_display()
        assert "Command executed" in display
        assert "```bash" in display
        assert "$ touch newfile.txt" in display
        assert "(No output)" in display

    def test_get_user_display_no_command(self):
        """Test user display when command is empty."""
        output = ShellCommandOutput(
            command="",
            exit_code=-1,
            stdout="",
            stderr="Error occurred",
            execution_time=0.0,
            success=False,
            working_dir="/tmp"
        )
        
        display = output.get_user_display()
        # When command is empty but there's stderr, it shows the failure format
        assert "Command failed" in display
        assert "Error: Error occurred" in display
        
        # Test truly empty case
        output_empty = ShellCommandOutput(
            command="",
            exit_code=-1,
            stdout="",
            stderr="",
            execution_time=0.0,
            success=False,
            working_dir="/tmp"
        )
        
        display_empty = output_empty.get_user_display()
        assert display_empty == "Shell command executed"

    def test_numeric_field_validation(self):
        """Test validation of numeric fields."""
        # Negative execution time should be allowed (edge case)
        output = ShellCommandOutput(
            command="test",
            exit_code=-1,
            execution_time=-0.5,  # Negative time (edge case)
            success=False,
            working_dir="/tmp"
        )
        assert output.execution_time == -0.5
        
        # Large exit codes
        output = ShellCommandOutput(
            command="test",
            exit_code=255,
            execution_time=1.0,
            success=False,
            working_dir="/tmp"
        )
        assert output.exit_code == 255

    def test_output_with_special_characters(self):
        """Test handling special characters in output."""
        output = ShellCommandOutput(
            command="echo $'Hello\\nWorld\\t\\x41'",
            exit_code=0,
            stdout="Hello\nWorld\tA",
            stderr="",
            execution_time=0.1,
            success=True,
            working_dir="/tmp"
        )
        
        assert "Hello\nWorld\tA" in output.stdout
        display = output.get_user_display()
        assert "Hello" in display