"""Models for bash tool."""


from pydantic import Field, field_validator

from ...models import ToolParameters, ToolResult, ToolStatus


class BashParameters(ToolParameters):
    """Parameters for bash tool execution."""

    command: str = Field(..., description="Shell command to execute")
    working_directory: str = Field(default=".", description="Working directory for command")
    timeout: int = Field(default=30, description="Timeout in seconds")
    capture_output: bool = Field(default=True, description="Capture stdout and stderr")
    shell: bool = Field(default=True, description="Execute through shell")
    env_vars: dict[str, str] = Field(
        default_factory=dict, description="Environment variables to set"
    )

    @field_validator("command")
    @classmethod
    def validate_command(cls, v: str) -> str:
        """Validate command."""
        if not v or not v.strip():
            raise ValueError("Command cannot be empty")
        return v.strip()

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: float) -> float:
        """Validate timeout."""
        if v < 0:
            raise ValueError("Timeout must be non-negative (0 means no timeout)")
        if v > 300:  # 5 minutes max
            raise ValueError("Timeout cannot exceed 300 seconds")
        return v


class BashResult(ToolResult):
    """Result from bash tool execution."""

    command: str | None = Field(default=None, description="Command that was executed")
    exit_code: int | None = Field(default=None, description="Process exit code")
    stdout: str | None = Field(default=None, description="Standard output")
    stderr: str | None = Field(default=None, description="Standard error")
    execution_time: float | None = Field(default=None, description="Execution time in seconds")
    timed_out: bool | None = Field(default=None, description="Whether the command timed out")
    working_directory: str | None = Field(default=None, description="Working directory used")

    def get_display_content(self) -> str:
        """Get user-friendly display text."""
        if self.status == ToolStatus.SUCCESS:
            lines = [f"Command executed successfully: {self.command}"]
            if self.exit_code is not None:
                lines.append(f"Exit code: {self.exit_code}")
            if self.execution_time is not None:
                lines.append(f"Execution time: {self.execution_time:.2f}s")
            if self.stdout:
                lines.append(f"Output:\n{self.stdout}")
            if self.stderr:
                lines.append(f"Error output:\n{self.stderr}")
            return "\n".join(lines)
        elif self.status == ToolStatus.ERROR:
            lines = [f"Command failed: {self.command or 'unknown'}"]
            if self.exit_code is not None:
                lines.append(f"Exit code: {self.exit_code}")
            if self.timed_out:
                lines.append("Command timed out")
            lines.append(f"Error: {self.message}")
            if self.stderr:
                lines.append(f"Error output:\n{self.stderr}")
            return "\n".join(lines)
        else:
            return self.message or "Bash execution status unknown"
