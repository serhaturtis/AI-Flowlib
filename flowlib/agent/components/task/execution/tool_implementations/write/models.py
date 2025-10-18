"""Models for write tool."""

from typing import Optional

from pydantic import Field, field_validator

from flowlib.core.models import StrictBaseModel

from ...models import ToolResult, ToolStatus


class WriteParameters(StrictBaseModel):
    """Parameters for write tool execution."""

    file_path: str = Field(..., description="Path to file to write")
    content: str = Field(..., description="Content to write to file")
    encoding: str = Field(default="utf-8", description="File encoding")
    create_directories: bool = Field(default=True, description="Create parent directories if they don't exist")
    backup: bool = Field(default=False, description="Create backup if file exists")

    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate file path."""
        if not v or not v.strip():
            raise ValueError("File path cannot be empty")
        return v.strip()


class WriteResult(ToolResult):
    """Result from write tool execution."""

    file_path: Optional[str] = Field(default=None, description="Path to written file")
    bytes_written: Optional[int] = Field(default=None, description="Number of bytes written")
    lines_written: Optional[int] = Field(default=None, description="Number of lines written")
    backup_path: Optional[str] = Field(default=None, description="Path to backup file if created")
    created_directories: Optional[bool] = Field(default=None, description="Whether parent directories were created")

    def get_display_content(self) -> str:
        """Get user-friendly display text."""
        if self.status == ToolStatus.SUCCESS:
            lines = [f"Successfully wrote to {self.file_path}"]
            if self.bytes_written is not None:
                lines.append(f"Bytes written: {self.bytes_written}")
            if self.lines_written is not None:
                lines.append(f"Lines written: {self.lines_written}")
            if self.backup_path:
                lines.append(f"Backup created: {self.backup_path}")
            if self.created_directories:
                lines.append("Created parent directories")
            return "\n".join(lines)
        elif self.status == ToolStatus.ERROR:
            return f"Write error: {self.message}"
        else:
            return self.message or "Write operation status unknown"
