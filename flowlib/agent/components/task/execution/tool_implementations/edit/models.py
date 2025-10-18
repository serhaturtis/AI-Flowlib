"""Models for edit tool."""

from typing import List, Optional

from pydantic import Field, field_validator

from flowlib.core.models import StrictBaseModel

from ...models import ToolResult, ToolStatus


class EditOperation(StrictBaseModel):
    """Single edit operation."""

    old_text: str = Field(..., description="Text to find and replace")
    new_text: str = Field(..., description="Text to replace with")
    replace_all: bool = Field(default=False, description="Replace all occurrences")


class EditParameters(StrictBaseModel):
    """Parameters for edit tool execution."""

    file_path: str = Field(..., description="Path to file to edit")
    operations: List[EditOperation] = Field(..., description="Edit operations to perform")
    encoding: str = Field(default="utf-8", description="File encoding")
    backup: bool = Field(default=True, description="Create backup before editing")

    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate file path."""
        if not v or not v.strip():
            raise ValueError("File path cannot be empty")
        return v.strip()

    @field_validator('operations')
    @classmethod
    def validate_operations(cls, v: List[EditOperation]) -> List[EditOperation]:
        """Validate edit operations."""
        if not v:
            raise ValueError("At least one edit operation required")
        for op in v:
            if op.old_text == op.new_text:
                raise ValueError("old_text and new_text cannot be the same")
        return v


class EditResult(ToolResult):
    """Result from edit tool execution."""

    file_path: Optional[str] = Field(default=None, description="Path to edited file")
    operations_applied: Optional[int] = Field(default=None, description="Number of operations successfully applied")
    total_replacements: Optional[int] = Field(default=None, description="Total number of text replacements made")
    backup_path: Optional[str] = Field(default=None, description="Path to backup file if created")
    lines_modified: Optional[int] = Field(default=None, description="Number of lines that were modified")

    def get_display_content(self) -> str:
        """Get user-friendly display text."""
        if self.status == ToolStatus.SUCCESS:
            lines = [f"Successfully edited {self.file_path}"]
            if self.operations_applied is not None:
                lines.append(f"Operations applied: {self.operations_applied}")
            if self.total_replacements is not None:
                lines.append(f"Total replacements: {self.total_replacements}")
            if self.lines_modified is not None:
                lines.append(f"Lines modified: {self.lines_modified}")
            if self.backup_path:
                lines.append(f"Backup created: {self.backup_path}")
            return "\n".join(lines)
        elif self.status == ToolStatus.ERROR:
            return f"Edit error: {self.message}"
        else:
            return self.message or "Edit operation status unknown"
