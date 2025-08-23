"""Tests for execution REPL tools."""

import os
import sys
import pytest
import asyncio
import tempfile
from unittest.mock import Mock, patch, MagicMock, AsyncMock, create_autospec

from flowlib.agent.runners.repl.tools.execution import (
    BashTool,
    PythonExecuteTool,
    ScriptExecuteTool
)
from flowlib.agent.runners.repl.tools.base import ToolResultStatus


class TestBashTool:
    """Test suite for BashTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = BashTool()
    
    def test_parameter_definitions(self):
        """Test parameter definitions."""
        params = self.tool._define_parameters()
        assert len(params) == 4
        
        # Check command parameter
        cmd_param = next(p for p in params if p.name == "command")
        assert cmd_param.type == "str"
        assert cmd_param.required is True
        
        # Check timeout parameter
        timeout_param = next(p for p in params if p.name == "timeout")
        assert timeout_param.type == "int"
        assert timeout_param.required is False
        assert timeout_param.default == 30
        
        # Check working_dir parameter
        dir_param = next(p for p in params if p.name == "working_dir")
        assert dir_param.type == "str"
        assert dir_param.required is False
        
        # Check capture_output parameter
        capture_param = next(p for p in params if p.name == "capture_output")
        assert capture_param.type == "bool"
        assert capture_param.required is False
        assert capture_param.default is True
    
    @pytest.mark.asyncio
    async def test_dangerous_command_blocking(self):
        """Test that dangerous commands are blocked."""
        # These commands should be blocked by the tool
        dangerous_commands = [
            "rm -rf /",
            ":(){ :|:& };:",  # Fork bomb
            "mkfs /dev/sda",
            "fdisk /dev/sda",
            "format c:",
            "dd if=/dev/zero > /dev/sda"  # Note: pattern checks for "> /dev/sda"
        ]
        
        for cmd in dangerous_commands:
            result = await self.tool.execute(command=cmd)
            assert result.status == ToolResultStatus.ERROR
            assert "dangerous command blocked" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_working_dir_not_exists(self):
        """Test error handling for non-existent working directory."""
        with patch('os.path.exists', return_value=False):
            result = await self.tool.execute(
                command="ls",
                working_dir="/nonexistent"
            )
        
        assert result.status == ToolResultStatus.ERROR
        assert "Working directory does not exist" in result.error
    
    @pytest.mark.asyncio
    async def test_successful_command_execution(self):
        """Test successful command execution."""
        # Create mock process
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Hello World\n", b""))
        
        with patch('asyncio.create_subprocess_shell', return_value=mock_process):
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.time.side_effect = [0.0, 0.5]  # Start and end time
                
                result = await self.tool.execute(command="echo 'Hello World'")
        
        assert result.status == ToolResultStatus.SUCCESS
        assert result.content == "Hello World\n"
        assert result.error is None
        assert result.metadata["return_code"] == 0
        assert result.metadata["execution_time"] == 0.5
        assert result.metadata["stdout_length"] == 12
        assert result.metadata["stderr_length"] == 0
    
    @pytest.mark.asyncio
    async def test_failed_command_execution(self):
        """Test failed command execution."""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Command not found\n"))
        
        with patch('asyncio.create_subprocess_shell', return_value=mock_process):
            result = await self.tool.execute(command="nonexistentcommand")
        
        assert result.status == ToolResultStatus.ERROR
        assert result.error == "Command not found\n"
        assert result.metadata["return_code"] == 1
    
    @pytest.mark.asyncio
    async def test_command_timeout(self):
        """Test command timeout handling."""
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        
        with patch('asyncio.create_subprocess_shell', return_value=mock_process):
            with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError()):
                result = await self.tool.execute(command="sleep 100", timeout=1)
        
        assert result.status == ToolResultStatus.ERROR
        assert "timed out after 1 seconds" in result.error
    
    @pytest.mark.asyncio
    async def test_without_output_capture(self):
        """Test execution without output capture."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.wait = AsyncMock(return_value=0)
        
        with patch('asyncio.create_subprocess_shell', return_value=mock_process):
            result = await self.tool.execute(
                command="echo 'test'",
                capture_output=False
            )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert result.content == ""  # No output captured
        assert mock_process.wait.called
    
    @pytest.mark.asyncio
    async def test_with_working_directory(self):
        """Test command execution with custom working directory."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Success\n", b""))
        
        with patch('os.path.exists', return_value=True):
            with patch('asyncio.create_subprocess_shell', return_value=mock_process) as mock_create:
                result = await self.tool.execute(
                    command="pwd",
                    working_dir="/tmp"
                )
        
        assert result.status == ToolResultStatus.SUCCESS
        # Verify cwd was passed to subprocess
        assert mock_create.call_args[1]["cwd"] == "/tmp"
        assert result.metadata["working_dir"] == "/tmp"
    
    @pytest.mark.asyncio
    async def test_general_exception_handling(self):
        """Test handling of general exceptions."""
        with patch('asyncio.create_subprocess_shell', side_effect=Exception("Test error")):
            result = await self.tool.execute(command="test")
        
        assert result.status == ToolResultStatus.ERROR
        assert "Error executing command" in result.error
        assert "Test error" in result.error


class TestPythonExecuteTool:
    """Test suite for PythonExecuteTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = PythonExecuteTool()
    
    def test_parameter_definitions(self):
        """Test parameter definitions."""
        params = self.tool._define_parameters()
        assert len(params) == 3
        
        # Check code parameter
        code_param = next(p for p in params if p.name == "code")
        assert code_param.type == "str"
        assert code_param.required is True
        
        # Check timeout parameter
        timeout_param = next(p for p in params if p.name == "timeout")
        assert timeout_param.type == "int"
        assert timeout_param.default == 30
        
        # Check capture_output parameter
        capture_param = next(p for p in params if p.name == "capture_output")
        assert capture_param.type == "bool"
        assert capture_param.default is True
    
    @pytest.mark.asyncio
    async def test_successful_python_execution(self):
        """Test successful Python code execution."""
        code = "print('Hello from Python')\nprint(2 + 2)"
        
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"Hello from Python\n4\n", b"")
        )
        
        with patch('tempfile.NamedTemporaryFile') as mock_tempfile:
            mock_file = MagicMock()
            mock_file.name = "/tmp/test.py"
            mock_tempfile.return_value.__enter__.return_value = mock_file
            
            with patch('asyncio.create_subprocess_exec', return_value=mock_process):
                with patch('os.unlink') as mock_unlink:
                    result = await self.tool.execute(code=code)
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "Hello from Python" in result.content
        assert "4" in result.content
        assert result.metadata["return_code"] == 0
        assert result.metadata["code_length"] == len(code)
        
        # Verify temp file was cleaned up
        mock_unlink.assert_called_once_with("/tmp/test.py")
    
    @pytest.mark.asyncio
    async def test_python_syntax_error(self):
        """Test Python code with syntax error."""
        code = "print('Missing closing quote)"
        
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"SyntaxError: unterminated string literal\n")
        )
        
        with patch('tempfile.NamedTemporaryFile') as mock_tempfile:
            mock_file = MagicMock()
            mock_file.name = "/tmp/test.py"
            mock_tempfile.return_value.__enter__.return_value = mock_file
            
            with patch('asyncio.create_subprocess_exec', return_value=mock_process):
                with patch('os.unlink'):
                    result = await self.tool.execute(code=code)
        
        assert result.status == ToolResultStatus.ERROR
        assert "SyntaxError" in result.error
        assert result.metadata["return_code"] == 1
    
    @pytest.mark.asyncio
    async def test_python_execution_timeout(self):
        """Test Python execution timeout."""
        code = "import time\ntime.sleep(100)"
        
        with patch('tempfile.NamedTemporaryFile') as mock_tempfile:
            mock_file = MagicMock()
            mock_file.name = "/tmp/test.py"
            mock_tempfile.return_value.__enter__.return_value = mock_file
            
            with patch('asyncio.create_subprocess_exec'):
                with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError()):
                    with patch('os.path.exists', return_value=True):
                        with patch('os.unlink'):
                            result = await self.tool.execute(code=code, timeout=1)
        
        assert result.status == ToolResultStatus.ERROR
        assert "timed out after 1 seconds" in result.error
    
    @pytest.mark.asyncio
    async def test_without_output_capture(self):
        """Test Python execution without output capture."""
        code = "print('This should not be captured')"
        
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(None, b""))
        
        with patch('tempfile.NamedTemporaryFile') as mock_tempfile:
            mock_file = MagicMock()
            mock_file.name = "/tmp/test.py"
            mock_tempfile.return_value.__enter__.return_value = mock_file
            
            with patch('asyncio.create_subprocess_exec', return_value=mock_process) as mock_exec:
                with patch('os.unlink'):
                    result = await self.tool.execute(code=code, capture_output=False)
        
        assert result.status == ToolResultStatus.SUCCESS
        assert result.content == ""
        # Verify stdout was not captured
        assert mock_exec.call_args[1]["stdout"] is None
    
    @pytest.mark.asyncio
    async def test_temp_file_cleanup_on_error(self):
        """Test that temp file is cleaned up on error."""
        code = "print('test')"
        temp_file = "/tmp/test.py"
        
        with patch('tempfile.NamedTemporaryFile') as mock_tempfile:
            mock_file = MagicMock()
            mock_file.name = temp_file
            mock_tempfile.return_value.__enter__.return_value = mock_file
            
            with patch('asyncio.create_subprocess_exec', side_effect=Exception("Test error")):
                with patch('os.path.exists', return_value=True) as mock_exists:
                    with patch('os.unlink') as mock_unlink:
                        result = await self.tool.execute(code=code)
        
        assert result.status == ToolResultStatus.ERROR
        assert "Error executing Python code" in result.error
        # Verify cleanup was attempted
        mock_exists.assert_called_with(temp_file)
        mock_unlink.assert_called_with(temp_file)


class TestScriptExecuteTool:
    """Test suite for ScriptExecuteTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = ScriptExecuteTool()
    
    def test_parameter_definitions(self):
        """Test parameter definitions."""
        params = self.tool._define_parameters()
        assert len(params) == 4
        
        # Check required parameters
        script_param = next(p for p in params if p.name == "script")
        assert script_param.type == "str"
        assert script_param.required is True
        
        lang_param = next(p for p in params if p.name == "language")
        assert lang_param.type == "str"
        assert lang_param.required is True
        
        # Check optional parameters
        timeout_param = next(p for p in params if p.name == "timeout")
        assert timeout_param.default == 30
        
        args_param = next(p for p in params if p.name == "args")
        assert args_param.type == "list"
        assert args_param.default == []
    
    @pytest.mark.asyncio
    async def test_unsupported_language(self):
        """Test error for unsupported language."""
        result = await self.tool.execute(
            script="echo test",
            language="cobol"
        )
        
        assert result.status == ToolResultStatus.ERROR
        assert "Unsupported language" in result.error
    
    @pytest.mark.asyncio
    async def test_python_script_execution(self):
        """Test Python script execution."""
        script = "import sys\nprint('Python version:', sys.version.split()[0])"
        
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"Python version: 3.12.2\n", b"")
        )
        
        with patch('tempfile.NamedTemporaryFile') as mock_tempfile:
            mock_file = MagicMock()
            mock_file.name = "/tmp/script.py"
            mock_tempfile.return_value.__enter__.return_value = mock_file
            
            with patch('asyncio.create_subprocess_exec', return_value=mock_process) as mock_exec:
                with patch('os.unlink'):
                    result = await self.tool.execute(
                        script=script,
                        language="python"
                    )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "Python version:" in result.content
        assert result.metadata["language"] == "python"
        assert result.metadata["interpreter"] == [sys.executable]
        
        # Verify correct command was executed
        call_args = mock_exec.call_args[0]
        assert call_args[0] == sys.executable
        assert call_args[1] == "/tmp/script.py"
    
    @pytest.mark.asyncio
    async def test_bash_script_execution(self):
        """Test bash script execution."""
        script = "#!/bin/bash\necho 'Hello from Bash'"
        
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"Hello from Bash\n", b"")
        )
        
        with patch('tempfile.NamedTemporaryFile') as mock_tempfile:
            mock_file = MagicMock()
            mock_file.name = "/tmp/script.sh"
            mock_tempfile.return_value.__enter__.return_value = mock_file
            
            with patch('asyncio.create_subprocess_exec', return_value=mock_process) as mock_exec:
                with patch('os.unlink'):
                    result = await self.tool.execute(
                        script=script,
                        language="bash"
                    )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "Hello from Bash" in result.content
        assert result.metadata["language"] == "bash"
        assert result.metadata["interpreter"] == ["/bin/bash"]
    
    @pytest.mark.asyncio
    async def test_script_with_arguments(self):
        """Test script execution with arguments."""
        script = "import sys\nprint('Args:', sys.argv[1:])"
        args = ["arg1", "arg2", "arg3"]
        
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"Args: ['arg1', 'arg2', 'arg3']\n", b"")
        )
        
        with patch('tempfile.NamedTemporaryFile') as mock_tempfile:
            mock_file = MagicMock()
            mock_file.name = "/tmp/script.py"
            mock_tempfile.return_value.__enter__.return_value = mock_file
            
            with patch('asyncio.create_subprocess_exec', return_value=mock_process) as mock_exec:
                with patch('os.unlink'):
                    result = await self.tool.execute(
                        script=script,
                        language="python",
                        args=args
                    )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert result.metadata["args"] == args
        
        # Verify args were passed to subprocess
        call_args = mock_exec.call_args[0]
        assert call_args[-3:] == tuple(args)
    
    @pytest.mark.asyncio
    async def test_node_script_execution(self):
        """Test Node.js script execution."""
        script = "console.log('Hello from Node.js');"
        
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"Hello from Node.js\n", b"")
        )
        
        with patch('tempfile.NamedTemporaryFile') as mock_tempfile:
            mock_file = MagicMock()
            mock_file.name = "/tmp/script.js"
            mock_tempfile.return_value.__enter__.return_value = mock_file
            
            with patch('asyncio.create_subprocess_exec', return_value=mock_process):
                with patch('os.unlink'):
                    result = await self.tool.execute(
                        script=script,
                        language="node"
                    )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "Hello from Node.js" in result.content
        assert result.metadata["language"] == "node"
        assert result.metadata["interpreter"] == ["node"]
    
    @pytest.mark.asyncio
    async def test_script_execution_failure(self):
        """Test script execution failure."""
        script = "exit 1"
        
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"Script failed\n")
        )
        
        with patch('tempfile.NamedTemporaryFile') as mock_tempfile:
            mock_file = MagicMock()
            mock_file.name = "/tmp/script.sh"
            mock_tempfile.return_value.__enter__.return_value = mock_file
            
            with patch('asyncio.create_subprocess_exec', return_value=mock_process):
                with patch('os.unlink'):
                    result = await self.tool.execute(
                        script=script,
                        language="bash"
                    )
        
        assert result.status == ToolResultStatus.ERROR
        assert result.error == "Script failed\n"
        assert result.metadata["return_code"] == 1
    
    @pytest.mark.asyncio
    async def test_script_timeout(self):
        """Test script execution timeout."""
        script = "while True: pass"
        
        with patch('tempfile.NamedTemporaryFile') as mock_tempfile:
            mock_file = MagicMock()
            mock_file.name = "/tmp/script.py"
            mock_tempfile.return_value.__enter__.return_value = mock_file
            
            with patch('asyncio.create_subprocess_exec'):
                with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError()):
                    result = await self.tool.execute(
                        script=script,
                        language="python",
                        timeout=1
                    )
        
        assert result.status == ToolResultStatus.ERROR
        assert "timed out after 1 seconds" in result.error
    
    @pytest.mark.asyncio
    async def test_language_aliases(self):
        """Test that language aliases work correctly."""
        script = "print('test')"
        
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"test\n", b""))
        
        # Test various Python aliases
        for lang in ["python", "python3"]:
            with patch('tempfile.NamedTemporaryFile') as mock_tempfile:
                mock_file = MagicMock()
                mock_file.name = f"/tmp/script.py"
                mock_tempfile.return_value.__enter__.return_value = mock_file
                
                with patch('asyncio.create_subprocess_exec', return_value=mock_process):
                    with patch('os.unlink'):
                        result = await self.tool.execute(
                            script=script,
                            language=lang
                        )
            
            assert result.status == ToolResultStatus.SUCCESS
            assert result.metadata["interpreter"] == [sys.executable]
        
        # Test JavaScript aliases
        for lang in ["node", "nodejs", "javascript"]:
            with patch('tempfile.NamedTemporaryFile') as mock_tempfile:
                mock_file = MagicMock()
                mock_file.name = "/tmp/script.js"
                mock_tempfile.return_value.__enter__.return_value = mock_file
                
                with patch('asyncio.create_subprocess_exec', return_value=mock_process):
                    with patch('os.unlink'):
                        result = await self.tool.execute(
                            script="console.log('test')",
                            language=lang
                        )
            
            assert result.status == ToolResultStatus.SUCCESS
            assert result.metadata["interpreter"] == ["node"]