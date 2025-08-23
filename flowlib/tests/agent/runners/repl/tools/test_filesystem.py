"""Tests for filesystem REPL tools."""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock

from flowlib.agent.runners.repl.tools.filesystem import (
    ReadTool,
    WriteTool,
    EditTool,
    MultiEditTool
)
from flowlib.agent.runners.repl.tools.base import ToolResultStatus


class TestReadTool:
    """Test suite for ReadTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = ReadTool()
    
    def test_parameter_definitions(self):
        """Test parameter definitions."""
        params = self.tool._define_parameters()
        assert len(params) == 3
        
        # Check file_path parameter
        file_param = next(p for p in params if p.name == "file_path")
        assert file_param.type == "str"
        assert file_param.required is True
        
        # Check offset parameter
        offset_param = next(p for p in params if p.name == "offset")
        assert offset_param.type == "int"
        assert offset_param.required is False
        
        # Check limit parameter
        limit_param = next(p for p in params if p.name == "limit")
        assert limit_param.type == "int"
        assert limit_param.required is False
    
    @pytest.mark.asyncio
    async def test_file_not_exists(self):
        """Test error handling for non-existent file."""
        with patch('os.path.exists', return_value=False):
            result = await self.tool.execute(file_path="/nonexistent/file.txt")
        
        assert result.status == ToolResultStatus.ERROR
        assert "File does not exist" in result.error
    
    @pytest.mark.asyncio
    async def test_read_entire_file(self):
        """Test reading entire file content."""
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=content)):
                result = await self.tool.execute(file_path="/test/file.txt")
        
        assert result.status == ToolResultStatus.SUCCESS
        
        # Check line numbering
        lines = result.content.split('\n')
        assert len(lines) == 5
        assert "     1\tLine 1" in result.content
        assert "     5\tLine 5" in result.content
        
        # Check metadata
        assert result.metadata["total_lines"] == 5
        assert result.metadata["start_line"] == 1
        assert result.metadata["end_line"] == 5
    
    @pytest.mark.asyncio
    async def test_read_with_offset(self):
        """Test reading file with offset."""
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=content)):
                result = await self.tool.execute(
                    file_path="/test/file.txt",
                    offset=3
                )
        
        assert result.status == ToolResultStatus.SUCCESS
        
        # Should start from line 3
        lines = result.content.split('\n')
        assert len(lines) == 3  # Lines 3, 4, 5
        assert "     3\tLine 3" in result.content
        assert "     5\tLine 5" in result.content
        
        # Check metadata
        assert result.metadata["start_line"] == 3
        assert result.metadata["end_line"] == 5
    
    @pytest.mark.asyncio
    async def test_read_with_limit(self):
        """Test reading file with limit."""
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=content)):
                result = await self.tool.execute(
                    file_path="/test/file.txt",
                    limit=3
                )
        
        assert result.status == ToolResultStatus.SUCCESS
        
        # Should only read 3 lines
        lines = result.content.split('\n')
        assert len(lines) == 3
        assert "     1\tLine 1" in result.content
        assert "     3\tLine 3" in result.content
        assert "Line 4" not in result.content
    
    @pytest.mark.asyncio
    async def test_read_with_offset_and_limit(self):
        """Test reading file with both offset and limit."""
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=content)):
                result = await self.tool.execute(
                    file_path="/test/file.txt",
                    offset=2,
                    limit=2
                )
        
        assert result.status == ToolResultStatus.SUCCESS
        
        # Should read lines 2 and 3 only
        lines = result.content.split('\n')
        assert len(lines) == 2
        assert "     2\tLine 2" in result.content
        assert "     3\tLine 3" in result.content
    
    @pytest.mark.asyncio
    async def test_unicode_decode_error(self):
        """Test handling of Unicode decode errors."""
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', side_effect=UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid')):
                result = await self.tool.execute(file_path="/test/binary.bin")
        
        assert result.status == ToolResultStatus.ERROR
        assert "Could not decode file as UTF-8" in result.error
    
    @pytest.mark.asyncio
    async def test_empty_file(self):
        """Test reading empty file."""
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="")):
                result = await self.tool.execute(file_path="/test/empty.txt")
        
        assert result.status == ToolResultStatus.SUCCESS
        assert result.content == ""
        assert result.metadata["total_lines"] == 0


class TestWriteTool:
    """Test suite for WriteTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = WriteTool()
    
    def test_parameter_definitions(self):
        """Test parameter definitions."""
        params = self.tool._define_parameters()
        assert len(params) == 3
        
        # Check required parameters
        file_param = next(p for p in params if p.name == "file_path")
        assert file_param.type == "str"
        assert file_param.required is True
        
        content_param = next(p for p in params if p.name == "content")
        assert content_param.type == "str"
        assert content_param.required is True
        
        # Check optional parameter
        dirs_param = next(p for p in params if p.name == "create_dirs")
        assert dirs_param.type == "bool"
        assert dirs_param.required is False
        assert dirs_param.default is False
    
    @pytest.mark.asyncio
    async def test_write_file_success(self):
        """Test successful file write."""
        content = "Hello, World!\nThis is a test file."
        
        mock_stat = MagicMock()
        mock_stat.st_size = len(content.encode('utf-8'))
        
        with patch('pathlib.Path.parent') as mock_parent:
            mock_parent.exists.return_value = True
            with patch('pathlib.Path.stat', return_value=mock_stat):
                with patch('builtins.open', mock_open()) as mock_file:
                    result = await self.tool.execute(
                        file_path="/test/output.txt",
                        content=content
                    )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "File written successfully" in result.content
        assert result.metadata["lines_written"] == 2
        assert result.metadata["bytes_written"] == len(content.encode('utf-8'))
        
        # Verify content was written
        mock_file().write.assert_called_once_with(content)
    
    @pytest.mark.asyncio
    async def test_parent_directory_not_exists(self):
        """Test error when parent directory doesn't exist."""
        with patch('pathlib.Path.parent') as mock_parent:
            mock_parent.exists.return_value = False
            result = await self.tool.execute(
                file_path="/nonexistent/dir/file.txt",
                content="test"
            )
        
        assert result.status == ToolResultStatus.ERROR
        assert "Parent directory does not exist" in result.error
    
    @pytest.mark.asyncio
    async def test_create_directories(self):
        """Test creating parent directories when requested."""
        content = "Test content"
        mock_stat = MagicMock()
        mock_stat.st_size = len(content)
        
        with patch('pathlib.Path.parent') as mock_parent:
            mock_parent.mkdir = MagicMock()
            mock_parent.exists.return_value = True  # After creation
            with patch('pathlib.Path.stat', return_value=mock_stat):
                with patch('builtins.open', mock_open()):
                    result = await self.tool.execute(
                        file_path="/new/dir/file.txt",
                        content=content,
                        create_dirs=True
                    )
        
        assert result.status == ToolResultStatus.SUCCESS
        mock_parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    @pytest.mark.asyncio
    async def test_write_empty_file(self):
        """Test writing empty content."""
        mock_stat = MagicMock()
        mock_stat.st_size = 0
        
        with patch('pathlib.Path.parent') as mock_parent:
            mock_parent.exists.return_value = True
            with patch('pathlib.Path.stat', return_value=mock_stat):
                with patch('builtins.open', mock_open()):
                    result = await self.tool.execute(
                        file_path="/test/empty.txt",
                        content=""
                    )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert result.metadata["lines_written"] == 1  # Empty string counts as 1 line
        assert result.metadata["bytes_written"] == 0
    
    @pytest.mark.asyncio
    async def test_write_error_handling(self):
        """Test error handling during write."""
        with patch('pathlib.Path.parent') as mock_parent:
            mock_parent.exists.return_value = True
            with patch('builtins.open', side_effect=PermissionError("Access denied")):
                result = await self.tool.execute(
                    file_path="/protected/file.txt",
                    content="test"
                )
        
        assert result.status == ToolResultStatus.ERROR
        assert "Error writing file" in result.error
        assert "Access denied" in result.error


class TestEditTool:
    """Test suite for EditTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = EditTool()
    
    def test_parameter_definitions(self):
        """Test parameter definitions."""
        params = self.tool._define_parameters()
        assert len(params) == 4
        
        # Check required parameters
        assert any(p.name == "file_path" and p.required for p in params)
        assert any(p.name == "old_string" and p.required for p in params)
        assert any(p.name == "new_string" and p.required for p in params)
        
        # Check optional parameter
        expected_param = next(p for p in params if p.name == "expected_replacements")
        assert expected_param.required is False
        assert expected_param.default == 1
    
    @pytest.mark.asyncio
    async def test_file_not_exists(self):
        """Test error handling for non-existent file."""
        with patch('os.path.exists', return_value=False):
            result = await self.tool.execute(
                file_path="/nonexistent/file.txt",
                old_string="old",
                new_string="new"
            )
        
        assert result.status == ToolResultStatus.ERROR
        assert "File does not exist" in result.error
    
    @pytest.mark.asyncio
    async def test_successful_edit(self):
        """Test successful string replacement."""
        original_content = "Hello, old world!\nThis is the old way."
        expected_content = "Hello, new world!\nThis is the new way."
        
        with patch('os.path.exists', return_value=True):
            # Mock read
            read_mock = mock_open(read_data=original_content)
            # Mock write
            write_mock = mock_open()
            
            def open_mock(path, mode='r', **kwargs):
                if mode == 'r':
                    return read_mock()
                return write_mock()
            
            with patch('builtins.open', side_effect=open_mock):
                result = await self.tool.execute(
                    file_path="/test/file.txt",
                    old_string="old",
                    new_string="new",
                    expected_replacements=2
                )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "Successfully replaced 2 occurrence(s)" in result.content
        assert result.metadata["replacements_made"] == 2
        
        # Verify correct content was written
        write_mock().write.assert_called_once_with(expected_content)
    
    @pytest.mark.asyncio
    async def test_string_not_found(self):
        """Test error when string to replace is not found."""
        content = "Hello, world!"
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=content)):
                result = await self.tool.execute(
                    file_path="/test/file.txt",
                    old_string="missing",
                    new_string="replacement"
                )
        
        assert result.status == ToolResultStatus.ERROR
        assert "String to replace not found" in result.error
    
    @pytest.mark.asyncio
    async def test_unexpected_replacement_count(self):
        """Test error when replacement count doesn't match expectation."""
        content = "old old old"
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=content)):
                result = await self.tool.execute(
                    file_path="/test/file.txt",
                    old_string="old",
                    new_string="new",
                    expected_replacements=2  # But there are 3
                )
        
        assert result.status == ToolResultStatus.ERROR
        assert "Expected 2 replacements, found 3" in result.error
    
    @pytest.mark.asyncio
    async def test_zero_expected_replacements(self):
        """Test when expected_replacements is 0 (no validation)."""
        content = "old old old"
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=content)):
                result = await self.tool.execute(
                    file_path="/test/file.txt",
                    old_string="old",
                    new_string="new",
                    expected_replacements=0  # No validation
                )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "Successfully replaced 3 occurrence(s)" in result.content


class TestMultiEditTool:
    """Test suite for MultiEditTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = MultiEditTool()
    
    def test_parameter_definitions(self):
        """Test parameter definitions."""
        params = self.tool._define_parameters()
        assert len(params) == 2
        
        # Check parameters
        file_param = next(p for p in params if p.name == "file_path")
        assert file_param.type == "str"
        assert file_param.required is True
        
        edits_param = next(p for p in params if p.name == "edits")
        assert edits_param.type == "list"
        assert edits_param.required is True
    
    @pytest.mark.asyncio
    async def test_file_not_exists(self):
        """Test error handling for non-existent file."""
        with patch('os.path.exists', return_value=False):
            result = await self.tool.execute(
                file_path="/nonexistent/file.txt",
                edits=[{"old_string": "old", "new_string": "new"}]
            )
        
        assert result.status == ToolResultStatus.ERROR
        assert "File does not exist" in result.error
    
    @pytest.mark.asyncio
    async def test_successful_multi_edit(self):
        """Test successful multiple edits."""
        original_content = "The quick brown fox jumps over the lazy dog."
        edits = [
            {"old_string": "quick", "new_string": "slow"},
            {"old_string": "brown", "new_string": "red"},
            {"old_string": "lazy", "new_string": "sleeping"}
        ]
        expected_content = "The slow red fox jumps over the sleeping dog."
        
        with patch('os.path.exists', return_value=True):
            read_mock = mock_open(read_data=original_content)
            write_mock = mock_open()
            
            def open_mock(path, mode='r', **kwargs):
                if mode == 'r':
                    return read_mock()
                return write_mock()
            
            with patch('builtins.open', side_effect=open_mock):
                result = await self.tool.execute(
                    file_path="/test/file.txt",
                    edits=edits
                )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "Successfully applied 3 edits" in result.content
        assert result.metadata["total_edits"] == 3
        assert result.metadata["total_replacements"] == 3
        
        # Verify final content
        write_mock().write.assert_called_once_with(expected_content)
    
    @pytest.mark.asyncio
    async def test_invalid_edit_format(self):
        """Test error for invalid edit format."""
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="test content")):
                try:
                    result = await self.tool.execute(
                        file_path="/test/file.txt",
                        edits=["not a dict"]  # Invalid format
                    )
                    # Should fail validation
                    assert False, "Expected ValidationError"
                except Exception as e:
                    # Should get Pydantic validation error
                    assert "validation error" in str(e).lower() or "model_type" in str(e)
    
    @pytest.mark.asyncio
    async def test_missing_edit_fields(self):
        """Test error for missing edit fields."""
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="test content")):
                try:
                    result = await self.tool.execute(
                        file_path="/test/file.txt",
                        edits=[{"old_string": "test"}]  # Missing new_string
                    )
                    # Should fail validation
                    assert False, "Expected ValidationError"
                except Exception as e:
                    # Should get Pydantic validation error
                    assert "Field required" in str(e) or "missing" in str(e)
    
    @pytest.mark.asyncio
    async def test_edit_string_not_found(self):
        """Test error when edit string is not found."""
        content = "Hello, world!"
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=content)):
                result = await self.tool.execute(
                    file_path="/test/file.txt",
                    edits=[
                        {"old_string": "Hello", "new_string": "Hi"},
                        {"old_string": "missing", "new_string": "replacement"}
                    ]
                )
        
        assert result.status == ToolResultStatus.ERROR
        assert "Edit 2: String to replace not found" in result.error
    
    @pytest.mark.asyncio
    async def test_sequential_edits(self):
        """Test that edits are applied sequentially."""
        original_content = "AAA BBB CCC"
        edits = [
            {"old_string": "AAA", "new_string": "BBB", "expected_replacements": 1},  # AAA -> BBB
            {"old_string": "BBB", "new_string": "CCC", "expected_replacements": 2},  # Both BBBs -> CCC
            {"old_string": "CCC", "new_string": "DDD", "expected_replacements": 3}   # All CCCs -> DDD
        ]
        expected_content = "DDD DDD DDD"
        
        with patch('os.path.exists', return_value=True):
            read_mock = mock_open(read_data=original_content)
            write_mock = mock_open()
            
            def open_mock(path, mode='r', **kwargs):
                if mode == 'r':
                    return read_mock()
                return write_mock()
            
            with patch('builtins.open', side_effect=open_mock):
                result = await self.tool.execute(
                    file_path="/test/file.txt",
                    edits=edits
                )
        
        assert result.status == ToolResultStatus.SUCCESS
        # First edit: 1 replacement (AAA->BBB)
        # Second edit: 2 replacements (both BBBs->CCC)
        # Third edit: 3 replacements (all CCCs->DDD)
        assert result.metadata["total_replacements"] == 6
        
        # Verify final content
        write_mock().write.assert_called_once_with(expected_content)
    
    @pytest.mark.asyncio
    async def test_expected_replacements_validation(self):
        """Test validation of expected replacements in edits."""
        content = "old old old"
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=content)):
                result = await self.tool.execute(
                    file_path="/test/file.txt",
                    edits=[{
                        "old_string": "old",
                        "new_string": "new",
                        "expected_replacements": 2  # But there are 3
                    }]
                )
        
        assert result.status == ToolResultStatus.ERROR
        assert "Expected 2 replacements, found 3" in result.error