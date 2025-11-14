"""Models for write tool."""


from pydantic import Field, field_validator

from ...models import ToolParameters, ToolResult, ToolStatus


class WriteParameters(ToolParameters):
    """Parameters for write tool execution.

    This tool will overwrite existing files. If the file already exists,
    you MUST use the Read tool first to read its contents.

    NEVER proactively create documentation files (*.md, README*).
    Only create documentation if explicitly requested by the user.

    ALWAYS prefer editing existing files over creating new ones.
    """

    file_path: str = Field(..., description="Absolute path to file to write")
    content: str = Field(..., description="Content to write to file")

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate file path is not empty."""
        if not v or not v.strip():
            raise ValueError("file_path cannot be empty")
        return v.strip()


class WriteResult(ToolResult):
    """Result from write tool execution."""

    file_path: str | None = Field(default=None, description="Path to written file")
    bytes_written: int | None = Field(default=None, description="Number of bytes written")
    lines_written: int | None = Field(default=None, description="Number of lines written")

    def get_display_content(self) -> str:
        """Get user-friendly display text."""
        if self.status == ToolStatus.SUCCESS:
            lines = [f"Successfully wrote to {self.file_path}"]
            if self.bytes_written is not None:
                lines.append(f"Bytes written: {self.bytes_written}")
            if self.lines_written is not None:
                lines.append(f"Lines written: {self.lines_written}")
            return "\n".join(lines)
        elif self.status == ToolStatus.ERROR:
            return f"Write error: {self.message}"
        else:
            return self.message or "Write operation status unknown"
