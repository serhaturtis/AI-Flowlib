"""Models for read tool operations."""


from pydantic import Field, field_validator

from ...models import ToolParameters, ToolResult


class ReadParameters(ToolParameters):
    """Parameters for read tool execution.

    Reads files and returns content in cat -n format (with line numbers).
    By default, reads up to 2000 lines from the beginning.
    Lines longer than 2000 characters are truncated.
    """

    file_path: str = Field(description="Absolute path to file to read")
    offset: int | None = Field(
        default=None,
        description="Line number to start reading from (0-based). Only provide if file is too large to read at once."
    )
    limit: int | None = Field(
        default=None,
        description="Number of lines to read. Only provide if file is too large to read at once."
    )

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate file path is not empty."""
        if not v or not v.strip():
            raise ValueError("file_path cannot be empty")
        return v.strip()

    @field_validator("offset")
    @classmethod
    def validate_offset(cls, v: int | None) -> int | None:
        """Validate offset is non-negative."""
        if v is not None and v < 0:
            raise ValueError("offset must be >= 0")
        return v

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v: int | None) -> int | None:
        """Validate limit is positive."""
        if v is not None and v < 1:
            raise ValueError("limit must be >= 1")
        return v


class ReadResult(ToolResult):
    """Result from read tool execution.

    Content is formatted using cat -n style with line numbers starting at 1.
    Lines longer than 2000 characters are truncated.
    """

    content: str | None = Field(default=None, description="File content in cat -n format (with line numbers)")
    file_path: str = Field(description="Path to file that was read")
    lines_read: int | None = Field(default=None, description="Number of lines actually read")
    total_lines: int | None = Field(default=None, description="Total lines in file")
    file_size: int | None = Field(default=None, description="File size in bytes")
    is_empty: bool = Field(default=False, description="Whether file exists but has empty contents")

    def get_display_content(self) -> str:
        """Get displayable content from read result."""
        if self.content is not None:
            return self.content
        return self.message or f"Read file: {self.file_path}"
