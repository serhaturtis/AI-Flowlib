"""Pydantic models for shell command flow."""

from typing import Dict, List, Optional, Any
from pydantic import Field
from flowlib.core.models import StrictBaseModel


class ShellCommandIntentInput(StrictBaseModel):
    """Input model describing the intent for a shell command."""
    
    intent: str = Field(..., description="A clear description of the goal (e.g., Create file with content, List files, Download file)")
    target_resource: Optional[str] = Field(None, description="The primary resource involved (e.g., file path, URL, directory)")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional parameters or options (e.g., file content, specific flags)")
    output_description: str = Field("Return the standard output.", description="How the output should be handled")
    working_dir: Optional[str] = Field(None, description="Optional working directory for command execution")
    timeout: Optional[int] = Field(60, description="Timeout in seconds for command execution")


class GeneratedCommand(StrictBaseModel):
    """Model for the generated shell command."""
    
    command: str = Field(..., description="The generated shell command string ready for execution")
    reasoning: str = Field(..., description="Brief explanation of why this command achieves the intent")


class ShellCommandOutput(StrictBaseModel):
    """Output model for shell command execution results."""
    
    command: str = Field(..., description="The command that was executed")
    exit_code: int = Field(..., description="Command exit code")
    stdout: str = Field("", description="Standard output from the command")
    stderr: str = Field("", description="Standard error from the command")
    execution_time: float = Field(..., description="Execution time in seconds")
    success: bool = Field(..., description="Whether the command executed successfully")
    working_dir: str = Field(..., description="Directory where the command was executed")
    
    def get_user_display(self) -> str:
        """Get user-friendly display text for shell command results."""
        if self.success and self.stdout.strip():
            return f"Command executed successfully:\n\n```bash\n$ {self.command}\n{self.stdout}\n```"
        elif not self.success and self.stderr.strip():
            return f"Command failed:\n\n```bash\n$ {self.command}\n\nError: {self.stderr}\n```"
        elif self.command:
            return f"Command executed:\n\n```bash\n$ {self.command}\n\n(No output)\n```"
        else:
            return "Shell command executed" 