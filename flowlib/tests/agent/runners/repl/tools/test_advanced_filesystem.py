"""Tests for advanced filesystem REPL tools."""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock

from flowlib.agent.runners.repl.tools.advanced_filesystem import (
    MultiFileEditTool,
    BatchFileOperationTool
)
from flowlib.agent.runners.repl.tools.base import ToolResultStatus


class TestMultiFileEditTool:
    """Test suite for MultiFileEditTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = MultiFileEditTool()
    
    def test_parameter_definitions(self):
        """Test parameter definitions."""
        params = self.tool._define_parameters()
        assert len(params) == 3
        
        # Check operations parameter
        ops_param = next(p for p in params if p.name == "operations")
        assert ops_param.type == "list"
        assert ops_param.required is True
        assert "edit operations" in ops_param.description
        
        # Check preview parameter
        preview_param = next(p for p in params if p.name == "preview")
        assert preview_param.type == "bool"
        assert preview_param.required is False
        assert preview_param.default is False
        
        # Check backup parameter
        backup_param = next(p for p in params if p.name == "backup")
        assert backup_param.type == "bool"
        assert backup_param.required is False
        assert backup_param.default is True
    
    @pytest.mark.asyncio
    async def test_invalid_operations_type(self):
        """Test error handling for invalid operations type."""
        result = await self.tool.execute(operations="not a list")
        
        assert result.status == ToolResultStatus.ERROR
        assert "Operations must be a list" in result.error
    
    @pytest.mark.asyncio
    async def test_validation_missing_fields(self):
        """Test validation catches missing required fields."""
        operations = [
            {"file_path": "/tmp/test.txt"},  # Missing old_string and new_string
            {"old_string": "foo", "new_string": "bar"},  # Missing file_path
            {"not_a_dict": True}  # Not a dict
        ]
        
        result = await self.tool.execute(operations=operations)
        
        assert result.status == ToolResultStatus.ERROR
        assert "Validation errors" in result.error
    
    @pytest.mark.asyncio
    async def test_validation_file_not_exists(self):
        """Test validation catches non-existent files."""
        operations = [
            {
                "file_path": "/nonexistent/file.txt",
                "old_string": "foo",
                "new_string": "bar"
            }
        ]
        
        with patch('os.path.exists', return_value=False):
            result = await self.tool.execute(operations=operations)
        
        assert result.status == ToolResultStatus.ERROR
        assert "File does not exist" in result.error
    
    @pytest.mark.asyncio
    async def test_validation_string_not_found(self):
        """Test validation catches strings not in files."""
        operations = [
            {
                "file_path": "/tmp/test.txt",
                "old_string": "not_in_file",
                "new_string": "replacement"
            }
        ]
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="file content")):
                result = await self.tool.execute(operations=operations)
        
        assert result.status == ToolResultStatus.ERROR
        assert "String not found" in result.error
    
    @pytest.mark.asyncio
    async def test_preview_mode(self):
        """Test preview mode generates diffs without applying changes."""
        operations = [
            {
                "file_path": "/tmp/test1.txt",
                "old_string": "hello",
                "new_string": "goodbye"
            },
            {
                "file_path": "/tmp/test2.txt",
                "old_string": "world",
                "new_string": "universe"
            }
        ]
        
        # Mock file operations
        file_contents = {
            "/tmp/test1.txt": "hello world\nhello again",
            "/tmp/test2.txt": "world is big\nsmall world"
        }
        
        def mock_open_func(path, mode='r', **kwargs):
            if mode == 'r':
                content = file_contents.get(path, "")
                return mock_open(read_data=content)()
            return mock_open()()
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', side_effect=mock_open_func):
                with patch('difflib.unified_diff') as mock_diff:
                    mock_diff.return_value = ["--- original\n", "+++ modified\n", "-hello\n", "+goodbye\n"]
                    
                    result = await self.tool.execute(operations=operations, preview=True)
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "Multi-File Edit Preview" in result.content
        assert "Operation 1" in result.content
        assert "Operation 2" in result.content
        assert result.metadata["preview"] is True
        assert result.metadata["operations"] == 2
    
    @pytest.mark.asyncio
    async def test_apply_changes_with_backup(self):
        """Test applying changes with backup creation."""
        operations = [
            {
                "file_path": "/tmp/test.txt",
                "old_string": "hello",
                "new_string": "goodbye"
            }
        ]
        
        # Mock file operations
        read_mock = mock_open(read_data="hello world")
        write_mock = mock_open()
        
        def mock_open_func(path, mode='r', **kwargs):
            if mode == 'r':
                return read_mock()
            return write_mock()
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', side_effect=mock_open_func):
                with patch('shutil.copy2') as mock_copy:
                    result = await self.tool.execute(operations=operations, backup=True)
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "Multi-File Edit Completed" in result.content
        assert result.metadata["operations_completed"] == 1
        assert result.metadata["total_replacements"] == 1
        
        # Verify backup was created
        mock_copy.assert_called_once_with("/tmp/test.txt", "/tmp/test.txt.backup")
        
        # Verify file was written
        write_mock().write.assert_called_once_with("goodbye world")
    
    @pytest.mark.asyncio
    async def test_apply_changes_without_backup(self):
        """Test applying changes without backup."""
        operations = [
            {
                "file_path": "/tmp/test.txt",
                "old_string": "foo",
                "new_string": "bar"
            }
        ]
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="foo and foo")):
                with patch('shutil.copy2') as mock_copy:
                    result = await self.tool.execute(operations=operations, backup=False)
        
        assert result.status == ToolResultStatus.SUCCESS
        assert result.metadata["total_replacements"] == 2
        
        # Verify backup was NOT created
        mock_copy.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_error_recovery_with_backup_restore(self):
        """Test backup restoration on error."""
        operations = [
            {
                "file_path": "/tmp/test.txt",
                "old_string": "hello",
                "new_string": "goodbye"
            }
        ]
        
        # Mock successful read, then error on write
        read_data = "hello world"
        backup_created = False
        
        def mock_open_func(path, mode='r', **kwargs):
            if mode == 'r':
                return mock_open(read_data=read_data)()
            else:
                # Simulate error after backup created
                raise Exception("Write error")
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', side_effect=mock_open_func):
                with patch('shutil.copy2') as mock_copy:
                    # Track that backup was created
                    mock_copy.side_effect = lambda s, d: None
                    with patch('shutil.move') as mock_move:
                        result = await self.tool.execute(operations=operations, backup=True)
        
        assert result.status == ToolResultStatus.ERROR
        assert "Error applying changes" in result.error
        assert "Backups restored" in result.error
    
    @pytest.mark.asyncio
    async def test_multiple_edits_same_file_warning(self):
        """Test warning for multiple edits to same file."""
        operations = [
            {
                "file_path": "/tmp/test.txt",
                "old_string": "foo",
                "new_string": "bar"
            },
            {
                "file_path": "/tmp/test.txt",
                "old_string": "baz",
                "new_string": "qux"
            }
        ]
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="foo baz")):
                validation_result = self.tool._validate_operations(operations)
        
        assert len(validation_result["warnings"]) == 1
        assert "Multiple edits to same file" in validation_result["warnings"][0]


class TestBatchFileOperationTool:
    """Test suite for BatchFileOperationTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = BatchFileOperationTool()
    
    def test_parameter_definitions(self):
        """Test parameter definitions."""
        params = self.tool._define_parameters()
        assert len(params) == 5
        
        # Check required parameters
        op_param = next(p for p in params if p.name == "operation")
        assert op_param.required is True
        assert "copy, move, delete, rename" in op_param.description
        
        files_param = next(p for p in params if p.name == "files")
        assert files_param.required is True
        assert files_param.type == "list"
        
        # Check optional parameters
        dest_param = next(p for p in params if p.name == "destination")
        assert dest_param.required is False
        
        pattern_param = next(p for p in params if p.name == "pattern")
        assert pattern_param.required is False
        
        dry_run_param = next(p for p in params if p.name == "dry_run")
        assert dry_run_param.required is False
        assert dry_run_param.default is False
    
    @pytest.mark.asyncio
    async def test_invalid_files_type(self):
        """Test error handling for invalid files type."""
        result = await self.tool.execute(operation="copy", files="not a list")
        
        assert result.status == ToolResultStatus.ERROR
        assert "Files must be a list" in result.error
    
    @pytest.mark.asyncio
    async def test_invalid_operation(self):
        """Test error handling for invalid operation."""
        result = await self.tool.execute(operation="invalid", files=[])
        
        assert result.status == ToolResultStatus.ERROR
        assert "Invalid operation" in result.error
        assert "copy, move, delete, rename" in result.error
    
    @pytest.mark.asyncio
    async def test_missing_files(self):
        """Test error handling for non-existent files."""
        with patch('os.path.exists', return_value=False):
            result = await self.tool.execute(
                operation="copy",
                files=["/tmp/missing1.txt", "/tmp/missing2.txt"],
                destination="/tmp/dest"
            )
        
        assert result.status == ToolResultStatus.ERROR
        assert "Files not found" in result.error
    
    @pytest.mark.asyncio
    async def test_batch_copy_success(self):
        """Test successful batch copy operation."""
        files = ["/tmp/file1.txt", "/tmp/file2.txt"]
        destination = "/tmp/dest"
        
        with patch('os.path.exists', return_value=True):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.is_dir', return_value=True):
                    with patch('shutil.copy2') as mock_copy:
                        result = await self.tool.execute(
                            operation="copy",
                            files=files,
                            destination=destination
                        )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "Batch Copy" in result.content
        assert result.metadata["operation"] == "copy"
        assert result.metadata["files_processed"] == 2
        assert mock_copy.call_count == 2
    
    @pytest.mark.asyncio
    async def test_batch_copy_dry_run(self):
        """Test batch copy in dry run mode."""
        files = ["/tmp/file1.txt"]
        destination = "/tmp/dest"
        
        with patch('os.path.exists', return_value=True):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.is_dir', return_value=True):
                    with patch('shutil.copy2') as mock_copy:
                        result = await self.tool.execute(
                            operation="copy",
                            files=files,
                            destination=destination,
                            dry_run=True
                        )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "(Dry Run)" in result.content
        assert result.metadata["dry_run"] is True
        mock_copy.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_batch_copy_missing_destination(self):
        """Test copy operation without destination."""
        with patch('os.path.exists', return_value=True):
            result = await self.tool.execute(
                operation="copy",
                files=["/tmp/file.txt"]
            )
        
        assert result.status == ToolResultStatus.ERROR
        assert "Destination is required" in result.error
    
    @pytest.mark.asyncio
    async def test_batch_move_success(self):
        """Test successful batch move operation."""
        files = ["/tmp/file1.txt", "/tmp/file2.txt"]
        destination = "/tmp/dest"
        
        with patch('os.path.exists', return_value=True):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.is_dir', return_value=True):
                    with patch('shutil.move') as mock_move:
                        result = await self.tool.execute(
                            operation="move",
                            files=files,
                            destination=destination
                        )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "Batch Move" in result.content
        assert result.metadata["operation"] == "move"
        assert mock_move.call_count == 2
    
    @pytest.mark.asyncio
    async def test_batch_delete_success(self):
        """Test successful batch delete operation."""
        files = ["/tmp/file1.txt", "/tmp/file2.txt"]
        
        with patch('os.path.exists', return_value=True):
            with patch('os.remove') as mock_remove:
                result = await self.tool.execute(
                    operation="delete",
                    files=files
                )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "Batch Delete" in result.content
        assert "Deleted 2 files" in result.content
        assert mock_remove.call_count == 2
    
    @pytest.mark.asyncio
    async def test_batch_delete_dry_run(self):
        """Test batch delete in dry run mode."""
        files = ["/tmp/file1.txt"]
        
        with patch('os.path.exists', return_value=True):
            with patch('os.remove') as mock_remove:
                result = await self.tool.execute(
                    operation="delete",
                    files=files,
                    dry_run=True
                )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "Would delete" in result.content
        mock_remove.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_batch_rename_success(self):
        """Test successful batch rename operation."""
        files = ["/tmp/file1.txt", "/tmp/file2.txt"]
        pattern = "renamed_{i}_{name}{ext}"
        
        # Mock Path.rename to simulate successful rename
        with patch('os.path.exists', return_value=True):
            with patch('pathlib.Path.rename') as mock_rename:
                # Don't raise any errors
                mock_rename.return_value = None
                
                result = await self.tool.execute(
                    operation="rename",
                    files=files,
                    pattern=pattern
                )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "Batch Rename" in result.content
        assert pattern in result.content  # Pattern is shown in the content with markdown
        assert result.metadata["files_processed"] == 2
        assert result.metadata["errors"] == 0
    
    @pytest.mark.asyncio
    async def test_batch_rename_missing_pattern(self):
        """Test rename operation without pattern."""
        with patch('os.path.exists', return_value=True):
            result = await self.tool.execute(
                operation="rename",
                files=["/tmp/file.txt"]
            )
        
        assert result.status == ToolResultStatus.ERROR
        assert "Pattern is required" in result.error
    
    @pytest.mark.asyncio
    async def test_batch_operation_with_errors(self):
        """Test batch operation with some failures."""
        files = ["/tmp/file1.txt", "/tmp/file2.txt"]
        
        with patch('os.path.exists', return_value=True):
            # First file succeeds, second fails
            with patch('os.remove', side_effect=[None, Exception("Permission denied")]):
                result = await self.tool.execute(
                    operation="delete",
                    files=files
                )
        
        assert result.status == ToolResultStatus.WARNING
        assert "Errors" in result.content
        assert "Permission denied" in result.content
        assert result.metadata["errors"] == 1
    
    @pytest.mark.asyncio
    async def test_create_destination_directory(self):
        """Test destination directory creation."""
        files = ["/tmp/file.txt"]
        destination = "/tmp/new_dest"
        
        with patch('os.path.exists', return_value=True):
            # Mock Path.mkdir to track directory creation
            with patch('pathlib.Path.mkdir') as mock_mkdir:
                with patch('pathlib.Path.exists', return_value=False):
                    with patch('pathlib.Path.is_dir', return_value=True):
                        with patch('shutil.copy2'):
                            result = await self.tool.execute(
                                operation="copy",
                                files=files,
                                destination=destination
                            )
        
        # Verify mkdir was called
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        assert result.status == ToolResultStatus.SUCCESS
    
    @pytest.mark.asyncio
    async def test_pattern_replacements(self):
        """Test pattern replacement logic in rename."""
        files = ["/tmp/test.txt"]
        pattern = "file_{i}_{name}{ext}"
        
        # Track the rename calls
        rename_calls = []
        
        def capture_rename(self, new_path):
            rename_calls.append((str(self), str(new_path)))
        
        with patch('os.path.exists', return_value=True):
            with patch('pathlib.Path.rename', capture_rename):
                result = await self.tool.execute(
                    operation="rename",
                    files=files,
                    pattern=pattern
                )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert len(rename_calls) == 1
        # Verify the pattern was applied correctly
        source, dest = rename_calls[0]
        assert source == "/tmp/test.txt"
        assert dest == "/tmp/file_0_test.txt"