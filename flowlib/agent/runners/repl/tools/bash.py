"""Bash tool - opencode style implementation with flowlib conventions."""

import asyncio
import os
import shlex
import subprocess
from typing import Dict, Any
from .base import REPLTool, ToolResult, ToolExecutionContext
from ..permissions import request_permission


class BashTool(REPLTool):
    """Execute bash commands with security checks and permission system."""
    
    # Dangerous command patterns (opencode approach)
    DANGEROUS_PATTERNS = [
        "rm -rf /",
        ":(){ :|:& };:",  # Fork bomb
        "mkfs",
        "fdisk", 
        "format",
        "> /dev/sda",
        "dd if=",
        "chmod -R 777 /",
        "chown -R root /",
        "sudo rm -rf",
        "rm -rf ~",
        "rm -rf /*",
        "kill -9 -1",
        "shutdown",
        "reboot",
        "halt"
    ]
    
    async def init(self) -> Dict[str, Any]:
        """Initialize bash tool schema."""
        return {
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string", 
                        "description": "The bash command to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 30, max: 300)",
                        "minimum": 1,
                        "maximum": 300,
                        "default": 30
                    }
                },
                "required": ["command"],
                "additionalProperties": False
            }
        }
    
    async def execute(self, parameters: Dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        """Execute bash command."""
        command = parameters["command"]
        timeout = parameters.get("timeout", 30)
        
        try:
            # Security check - block dangerous commands
            for pattern in self.DANGEROUS_PATTERNS:
                if pattern.lower() in command.lower():
                    return ToolResult.error(f"Potentially dangerous command blocked: {pattern}")
            
            # Additional security checks
            if any(char in command for char in ["&", "|", ";", "$(", "`"]) and "rm" in command.lower():
                return ToolResult.error("Complex command with 'rm' blocked for safety")
            
            # Request permission for command execution
            permission_title = f"Execute bash command"
            permission_details = f"Command: {command[:100]}{'...' if len(command) > 100 else ''}"
            
            granted = await request_permission(
                operation_type="bash",
                title=permission_title,
                details=permission_details,
                session_id=context.session_id,
                message_id=context.message_id,
                call_id=context.call_id,
                metadata={
                    "command": command,
                    "timeout": timeout,
                    "working_directory": context.working_directory
                }
            )
            
            if not granted:
                return ToolResult.error("Bash command execution denied by user")
            
            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.working_directory
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
                
                # Decode output
                stdout_text = stdout.decode('utf-8', errors='replace')
                stderr_text = stderr.decode('utf-8', errors='replace')
                
                # Combine output
                output_parts = []
                if stdout_text.strip():
                    output_parts.append(f"STDOUT:\\n{stdout_text}")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\\n{stderr_text}")
                
                combined_output = "\\n\\n".join(output_parts)
                
                metadata = {
                    "command": command,
                    "exit_code": process.returncode,
                    "timeout": timeout,
                    "stdout_length": len(stdout_text),
                    "stderr_length": len(stderr_text),
                    "working_directory": context.working_directory
                }
                
                if process.returncode == 0:
                    return ToolResult.success(
                        combined_output or "Command completed successfully (no output)",
                        metadata
                    )
                else:
                    return ToolResult.error(
                        f"Command failed with exit code {process.returncode}:\\n{combined_output}",
                        metadata
                    )
                    
            except asyncio.TimeoutError:
                # Kill the process
                try:
                    process.kill()
                    await process.wait()
                except:
                    pass
                    
                return ToolResult.error(f"Command timed out after {timeout} seconds")
                
        except FileNotFoundError:
            return ToolResult.error("Bash not found - shell commands not available")
        except PermissionError:
            return ToolResult.error(f"Permission denied executing command: {command}")
        except Exception as e:
            return ToolResult.error(f"Error executing command: {str(e)}")
    
    def _is_safe_command(self, command: str) -> bool:
        """Additional safety check for commands."""
        # Parse command safely
        try:
            tokens = shlex.split(command)
        except ValueError:
            return False  # Invalid shell syntax
        
        if not tokens:
            return False
        
        base_command = tokens[0].lower()
        
        # Allow common safe commands
        safe_commands = {
            'ls', 'cat', 'echo', 'pwd', 'whoami', 'date', 'which', 'head', 'tail',
            'grep', 'find', 'locate', 'file', 'stat', 'wc', 'sort', 'uniq',
            'git', 'python', 'python3', 'pip', 'npm', 'node', 'cargo', 'go',
            'make', 'cmake', 'gcc', 'clang', 'javac', 'java', 'rustc'
        }
        
        return base_command in safe_commands