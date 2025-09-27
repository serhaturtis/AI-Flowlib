"""Models for read tool operations."""

from typing import Optional
from pydantic import Field, field_validator

from ...models import ToolParameters, ToolResult


class ReadParameters(ToolParameters):
    """Parameters for read tool execution."""
    
    file_path: str = Field(description="Path to file to read")
    start_line: int = Field(default=1, description="Line number to start reading from (1-based)")
    line_count: int = Field(default=-1, description="Number of lines to read (-1 for all)")
    encoding: str = Field(default="utf-8", description="File encoding")
    
    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate file path format."""
        if not v or not v.strip():
            raise ValueError("file_path cannot be empty")
        return v.strip()

    @field_validator('start_line')
    @classmethod
    def validate_start_line(cls, v: int) -> int:
        """Validate start line is positive."""
        if v < 1:
            raise ValueError("start_line must be >= 1")
        return v

    @field_validator('line_count')
    @classmethod
    def validate_line_count(cls, v: int) -> int:
        """Validate line count is valid."""
        if v != -1 and v < 1:
            raise ValueError("line_count must be >= 1 or -1 for all")
        return v


class ReadResult(ToolResult):
    """Result from read tool execution."""
    
    content: Optional[str] = Field(default=None, description="File content that was read")
    file_path: str = Field(description="Path to file that was read")
    lines_read: Optional[int] = Field(default=None, description="Number of lines actually read")
    total_lines: Optional[int] = Field(default=None, description="Total lines in file")
    file_size: Optional[int] = Field(default=None, description="File size in bytes")
    encoding_used: str = Field(default="utf-8", description="Encoding used to read file")
    
    def get_display_content(self) -> str:
        """Get displayable content from read result."""
        if self.content is not None:
            return self.content
        return self.message or f"Read file: {self.file_path}"