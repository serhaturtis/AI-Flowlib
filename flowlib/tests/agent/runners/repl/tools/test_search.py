"""Tests for search REPL tools."""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import re
import time

from flowlib.agent.runners.repl.tools.search import (
    GlobTool,
    GrepTool,
    LSDirectoryTool
)
from flowlib.agent.runners.repl.tools.base import ToolResultStatus


class TestGlobTool:
    """Test suite for GlobTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = GlobTool()
    
    def test_parameter_definitions(self):
        """Test parameter definitions."""
        params = self.tool._define_parameters()
        assert len(params) == 4
        
        # Check pattern parameter
        pattern_param = next(p for p in params if p.name == "pattern")
        assert pattern_param.type == "str"
        assert pattern_param.required is True
        assert "glob pattern" in pattern_param.description.lower()
        
        # Check path parameter
        path_param = next(p for p in params if p.name == "path")
        assert path_param.type == "str"
        assert path_param.required is False
        assert path_param.default == "."
        
        # Check include_dirs parameter
        dirs_param = next(p for p in params if p.name == "include_dirs")
        assert dirs_param.type == "bool"
        assert dirs_param.required is False
        assert dirs_param.default is False
        
        # Check max_results parameter
        max_param = next(p for p in params if p.name == "max_results")
        assert max_param.type == "int"
        assert max_param.required is False
        assert max_param.default == 100
    
    @pytest.mark.asyncio
    async def test_path_not_exists(self):
        """Test error handling for non-existent path."""
        with patch('os.path.exists', return_value=False):
            result = await self.tool.execute(pattern="*.py", path="/nonexistent")
        
        assert result.status == ToolResultStatus.ERROR
        assert "Path does not exist" in result.error
    
    @pytest.mark.asyncio
    async def test_glob_success(self):
        """Test successful glob operation."""
        mock_files = ["file1.py", "file2.py", "test/file3.py"]
        
        with patch('os.path.exists', return_value=True):
            with patch('os.getcwd', return_value="/original"):
                with patch('os.chdir') as mock_chdir:
                    with patch('glob.glob', return_value=mock_files):
                        with patch('os.path.isfile', return_value=True):
                            with patch('os.path.abspath', return_value="/search/path"):
                                result = await self.tool.execute(
                                    pattern="**/*.py",
                                    path="/search/path"
                                )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "file1.py" in result.content
        assert "file2.py" in result.content
        assert "test/file3.py" in result.content
        assert result.metadata["total_matches"] == 3
        assert result.metadata["pattern"] == "**/*.py"
        assert result.metadata["search_path"] == "/search/path"
        assert result.metadata["truncated"] is False
        
        # Verify directory change and restore
        assert mock_chdir.call_count == 2  # Change to path and back
    
    @pytest.mark.asyncio
    async def test_glob_with_directories(self):
        """Test glob with directory inclusion."""
        mock_entries = ["file1.py", "dir1", "file2.py", "dir2"]
        
        def is_file(path):
            return not path.endswith(("dir1", "dir2"))
        
        with patch('os.path.exists', return_value=True):
            with patch('os.getcwd', return_value="/original"):
                with patch('os.chdir'):
                    with patch('glob.glob', return_value=mock_entries):
                        with patch('os.path.isfile', side_effect=is_file):
                            with patch('os.path.abspath', return_value="/path"):
                                # Test without directories
                                result = await self.tool.execute(
                                    pattern="*",
                                    include_dirs=False
                                )
                                assert "dir1" not in result.content
                                assert "file1.py" in result.content
                                
                                # Test with directories
                                result = await self.tool.execute(
                                    pattern="*",
                                    include_dirs=True
                                )
                                assert "dir1" in result.content
                                assert "file1.py" in result.content
    
    @pytest.mark.asyncio
    async def test_glob_max_results(self):
        """Test glob with max results limit."""
        # Create more files than max_results
        mock_files = [f"file{i}.py" for i in range(150)]
        
        with patch('os.path.exists', return_value=True):
            with patch('os.getcwd', return_value="/original"):
                with patch('os.chdir'):
                    with patch('glob.glob', return_value=mock_files):
                        with patch('os.path.isfile', return_value=True):
                            with patch('os.path.abspath', return_value="/path"):
                                result = await self.tool.execute(
                                    pattern="*.py",
                                    max_results=100
                                )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert result.metadata["total_matches"] == 100
        assert result.metadata["truncated"] is True
        
        # Count lines in content (each file on a separate line)
        lines = result.content.strip().split('\n')
        assert len(lines) == 100
    
    @pytest.mark.asyncio
    async def test_glob_no_matches(self):
        """Test glob with no matching files."""
        with patch('os.path.exists', return_value=True):
            with patch('os.getcwd', return_value="/original"):
                with patch('os.chdir'):
                    with patch('glob.glob', return_value=[]):
                        with patch('os.path.abspath', return_value="/path"):
                            result = await self.tool.execute(pattern="*.nonexistent")
        
        assert result.status == ToolResultStatus.SUCCESS
        assert result.content == "No files found"
        assert result.metadata["total_matches"] == 0
    
    @pytest.mark.asyncio
    async def test_glob_error_handling(self):
        """Test error handling in glob operation."""
        with patch('os.path.exists', return_value=True):
            with patch('os.getcwd', return_value="/original"):
                with patch('os.chdir', side_effect=PermissionError("Access denied")):
                    result = await self.tool.execute(pattern="*.py")
        
        assert result.status == ToolResultStatus.ERROR
        assert "Error searching files" in result.error
        assert "Access denied" in result.error


class TestGrepTool:
    """Test suite for GrepTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = GrepTool()
    
    def test_parameter_definitions(self):
        """Test parameter definitions."""
        params = self.tool._define_parameters()
        assert len(params) == 7
        
        # Check required parameters
        pattern_param = next(p for p in params if p.name == "pattern")
        assert pattern_param.type == "str"
        assert pattern_param.required is True
        
        # Check optional parameters
        param_names = [p.name for p in params]
        assert "path" in param_names
        assert "include" in param_names
        assert "exclude" in param_names
        assert "case_sensitive" in param_names
        assert "max_results" in param_names
        assert "context_lines" in param_names
        
        # Check defaults
        path_param = next(p for p in params if p.name == "path")
        assert path_param.default == "."
        
        case_param = next(p for p in params if p.name == "case_sensitive")
        assert case_param.default is True
    
    @pytest.mark.asyncio
    async def test_invalid_regex_pattern(self):
        """Test error handling for invalid regex pattern."""
        result = await self.tool.execute(pattern="[invalid(regex")
        
        assert result.status == ToolResultStatus.ERROR
        assert "Invalid regex pattern" in result.error
    
    @pytest.mark.asyncio
    async def test_path_not_exists(self):
        """Test error handling for non-existent path."""
        with patch('os.path.exists', return_value=False):
            result = await self.tool.execute(pattern="test", path="/nonexistent")
        
        assert result.status == ToolResultStatus.ERROR
        assert "Path does not exist" in result.error
    
    @pytest.mark.asyncio
    async def test_grep_success(self):
        """Test successful grep operation."""
        # Mock file system
        mock_walk = [
            ("/test", ["subdir"], ["file1.py", "file2.txt"]),
            ("/test/subdir", [], ["file3.py"])
        ]
        
        file_contents = {
            "/test/file1.py": "def test_function():\n    pass\n",
            "/test/file2.txt": "This is a test file\nAnother line\n",
            "/test/subdir/file3.py": "# No matches here\n"
        }
        
        def mock_open_func(path, mode='r', **kwargs):
            if path in file_contents:
                return mock_open(read_data=file_contents[path])()
            raise FileNotFoundError()
        
        with patch('os.path.exists', return_value=True):
            with patch('os.walk', return_value=mock_walk):
                with patch('builtins.open', side_effect=mock_open_func):
                    with patch('os.path.abspath', return_value="/test"):
                        result = await self.tool.execute(
                            pattern="test",
                            path="/test"
                        )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "file1.py" in result.content
        assert "file2.txt" in result.content
        assert "file3.py" not in result.content  # No matches
        assert result.metadata["matching_files"] == 2
        assert result.metadata["total_matches"] == 2
    
    @pytest.mark.asyncio
    async def test_grep_with_filters(self):
        """Test grep with include/exclude filters."""
        mock_walk = [
            ("/test", [], ["file1.py", "file2.txt", "file3.js"])
        ]
        
        file_contents = {
            "/test/file1.py": "test content\n",
            "/test/file2.txt": "test content\n",
            "/test/file3.js": "test content\n"
        }
        
        def mock_open_func(path, mode='r', **kwargs):
            if path in file_contents:
                return mock_open(read_data=file_contents[path])()
            raise FileNotFoundError()
        
        with patch('os.path.exists', return_value=True):
            with patch('os.walk', return_value=mock_walk):
                with patch('builtins.open', side_effect=mock_open_func):
                    with patch('os.path.abspath', return_value="/test"):
                        # Test with include filter
                        result = await self.tool.execute(
                            pattern="test",
                            path="/test",
                            include="*.py"
                        )
                        assert "file1.py" in result.content
                        assert "file2.txt" not in result.content
                        
                        # Test with exclude filter
                        result = await self.tool.execute(
                            pattern="test",
                            path="/test",
                            exclude="*.txt"
                        )
                        assert "file1.py" in result.content
                        assert "file2.txt" not in result.content
                        assert "file3.js" in result.content
    
    @pytest.mark.asyncio
    async def test_grep_case_sensitivity(self):
        """Test case sensitive and insensitive search."""
        mock_walk = [("/test", [], ["file.txt"])]
        file_content = "Test line\ntest line\nTEST line\n"
        
        with patch('os.path.exists', return_value=True):
            with patch('os.walk', return_value=mock_walk):
                with patch('builtins.open', mock_open(read_data=file_content)):
                    with patch('os.path.abspath', return_value="/test"):
                        # Case sensitive (default)
                        result = await self.tool.execute(
                            pattern="Test",
                            path="/test",
                            case_sensitive=True
                        )
                        assert result.metadata["total_matches"] == 1
                        
                        # Case insensitive
                        result = await self.tool.execute(
                            pattern="Test",
                            path="/test",
                            case_sensitive=False
                        )
                        assert result.metadata["total_matches"] == 3
    
    @pytest.mark.asyncio
    async def test_grep_with_context(self):
        """Test grep with context lines."""
        mock_walk = [("/test", [], ["file.txt"])]
        file_content = "line1\nline2\nmatch line\nline4\nline5\n"
        
        with patch('os.path.exists', return_value=True):
            with patch('os.walk', return_value=mock_walk):
                with patch('builtins.open', mock_open(read_data=file_content)):
                    with patch('os.path.abspath', return_value="/test"):
                        result = await self.tool.execute(
                            pattern="match",
                            path="/test",
                            context_lines=2
                        )
        
        assert result.status == ToolResultStatus.SUCCESS
        # Check that context lines are included
        assert "line1" in result.content
        assert "line2" in result.content
        assert "match line" in result.content
        assert "line4" in result.content
        assert "line5" in result.content
    
    @pytest.mark.asyncio
    async def test_grep_max_results(self):
        """Test grep with max results limit."""
        # Create many files
        mock_walk = [("/test", [], [f"file{i}.txt" for i in range(60)])]
        
        def mock_open_func(path, mode='r', **kwargs):
            return mock_open(read_data="test content\n")()
        
        with patch('os.path.exists', return_value=True):
            with patch('os.walk', return_value=mock_walk):
                with patch('builtins.open', side_effect=mock_open_func):
                    with patch('os.path.abspath', return_value="/test"):
                        result = await self.tool.execute(
                            pattern="test",
                            path="/test",
                            max_results=50
                        )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert result.metadata["matching_files"] == 50
        assert result.metadata["truncated"] is True
    
    @pytest.mark.asyncio
    async def test_grep_skip_binary_and_hidden(self):
        """Test that grep skips binary and hidden files."""
        mock_walk = [
            ("/test", [".git"], ["normal.txt", ".hidden", "binary.bin"])
        ]
        
        def mock_open_func(path, mode='r', **kwargs):
            if "normal.txt" in path:
                return mock_open(read_data="test content\n")()
            elif ".hidden" in path:
                return mock_open(read_data="test content\n")()
            else:
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "invalid")
        
        with patch('os.path.exists', return_value=True):
            with patch('os.walk', return_value=mock_walk) as mock_walk_obj:
                with patch('builtins.open', side_effect=mock_open_func):
                    with patch('os.path.abspath', return_value="/test"):
                        result = await self.tool.execute(
                            pattern="test",
                            path="/test"
                        )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "normal.txt" in result.content
        assert ".hidden" not in result.content  # Hidden files skipped
        assert "binary.bin" not in result.content  # Binary files skipped
        
        # Check that .git directory was filtered out
        dirs_list = mock_walk_obj.return_value[0][1]
        assert ".git" not in dirs_list  # Modified in place
    
    @pytest.mark.asyncio
    async def test_grep_no_matches(self):
        """Test grep with no matches."""
        mock_walk = [("/test", [], ["file.txt"])]
        
        with patch('os.path.exists', return_value=True):
            with patch('os.walk', return_value=mock_walk):
                with patch('builtins.open', mock_open(read_data="no matches here\n")):
                    with patch('os.path.abspath', return_value="/test"):
                        result = await self.tool.execute(
                            pattern="nonexistent",
                            path="/test"
                        )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "No matches found" in result.content
        assert result.metadata["matching_files"] == 0


class TestLSDirectoryTool:
    """Test suite for LSDirectoryTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = LSDirectoryTool()
    
    def test_parameter_definitions(self):
        """Test parameter definitions."""
        params = self.tool._define_parameters()
        assert len(params) == 5
        
        # Check path parameter
        path_param = next(p for p in params if p.name == "path")
        assert path_param.type == "str"
        assert path_param.required is True
        
        # Check optional parameters
        param_names = [p.name for p in params]
        assert "show_hidden" in param_names
        assert "show_details" in param_names
        assert "recursive" in param_names
        assert "max_depth" in param_names
        
        # Check defaults
        hidden_param = next(p for p in params if p.name == "show_hidden")
        assert hidden_param.default is False
        
        depth_param = next(p for p in params if p.name == "max_depth")
        assert depth_param.default == 3
    
    @pytest.mark.asyncio
    async def test_path_not_exists(self):
        """Test error handling for non-existent path."""
        with patch('os.path.exists', return_value=False):
            result = await self.tool.execute(path="/nonexistent")
        
        assert result.status == ToolResultStatus.ERROR
        assert "Path does not exist" in result.error
    
    @pytest.mark.asyncio
    async def test_path_not_directory(self):
        """Test error handling when path is not a directory."""
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isdir', return_value=False):
                result = await self.tool.execute(path="/file.txt")
        
        assert result.status == ToolResultStatus.ERROR
        assert "Path is not a directory" in result.error
    
    @pytest.mark.asyncio
    async def test_ls_basic(self):
        """Test basic directory listing."""
        mock_entries = ["file1.txt", "file2.py", "subdir"]
        
        def mock_isdir(path):
            return path.endswith("subdir")
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isdir', side_effect=lambda p: p == "/test" or mock_isdir(p)):
                with patch('os.listdir', return_value=mock_entries):
                    with patch('os.path.abspath', return_value="/test"):
                        result = await self.tool.execute(path="/test")
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "file1.txt" in result.content
        assert "file2.py" in result.content
        assert "subdir" in result.content
        assert "📁" in result.content  # Directory marker
        assert "📄" in result.content  # File marker
        assert result.metadata["total_files"] == 2
        assert result.metadata["total_directories"] == 1
    
    @pytest.mark.asyncio
    async def test_ls_with_hidden_files(self):
        """Test listing with hidden files."""
        mock_entries = ["file.txt", ".hidden", ".git"]
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isdir', side_effect=lambda p: p == "/test" or p.endswith(".git")):
                with patch('os.listdir', return_value=mock_entries):
                    with patch('os.path.abspath', return_value="/test"):
                        # Without hidden files
                        result = await self.tool.execute(
                            path="/test",
                            show_hidden=False
                        )
                        assert ".hidden" not in result.content
                        assert ".git" not in result.content
                        
                        # With hidden files
                        result = await self.tool.execute(
                            path="/test",
                            show_hidden=True
                        )
                        assert ".hidden" in result.content
                        assert ".git" in result.content
    
    @pytest.mark.asyncio
    async def test_ls_with_details(self):
        """Test listing with detailed information."""
        mock_entries = ["file.txt", "large_file.bin"]
        
        mock_stat = MagicMock()
        mock_stat.st_size = 1536  # 1.5K
        mock_stat.st_mtime = 1609459200  # 2021-01-01 00:00:00
        
        mock_stat_large = MagicMock()
        mock_stat_large.st_size = 2097152  # 2M
        mock_stat_large.st_mtime = 1609459200
        
        def mock_stat_func(path):
            if "large_file" in path:
                return mock_stat_large
            return mock_stat
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isdir', side_effect=lambda p: p == "/test"):
                with patch('os.listdir', return_value=mock_entries):
                    with patch('os.path.isfile', return_value=True):
                        with patch('os.stat', side_effect=mock_stat_func):
                            with patch('os.path.abspath', return_value="/test"):
                                result = await self.tool.execute(
                                    path="/test",
                                    show_details=True
                                )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "1.5K" in result.content  # Size formatting
        assert "2.0M" in result.content  # Size formatting
        assert "2021-01-01" in result.content  # Date formatting
    
    @pytest.mark.asyncio
    async def test_ls_recursive(self):
        """Test recursive directory listing."""
        # Mock nested directory structure
        def mock_listdir(path):
            if path == "/test":
                return ["file1.txt", "subdir"]
            elif path.endswith("subdir"):
                return ["file2.txt", "deep"]
            elif path.endswith("deep"):
                return ["file3.txt"]
            return []
        
        def mock_isdir(path):
            return path in ["/test", "/test/subdir", "/test/subdir/deep"] or \
                   path.endswith(("subdir", "deep"))
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isdir', side_effect=mock_isdir):
                with patch('os.listdir', side_effect=mock_listdir):
                    with patch('os.path.abspath', return_value="/test"):
                        result = await self.tool.execute(
                            path="/test",
                            recursive=True,
                            max_depth=2
                        )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "file1.txt" in result.content
        assert "file2.txt" in result.content
        assert "file3.txt" in result.content
        # Check indentation for nested items
        lines = result.content.split('\n')
        # Find lines with file2.txt and file3.txt
        for line in lines:
            if "file2.txt" in line:
                assert line.startswith("  ")  # Indented
            if "file3.txt" in line:
                assert line.startswith("    ")  # Double indented
    
    @pytest.mark.asyncio
    async def test_ls_permission_error(self):
        """Test handling of permission errors."""
        def mock_listdir(path):
            if "forbidden" in path:
                raise PermissionError("Access denied")
            return ["accessible.txt", "forbidden_dir"]
        
        def mock_isdir(path):
            return path == "/test" or "forbidden_dir" in path
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isdir', side_effect=mock_isdir):
                with patch('os.listdir', side_effect=mock_listdir):
                    with patch('os.path.abspath', return_value="/test"):
                        result = await self.tool.execute(
                            path="/test",
                            recursive=True
                        )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "accessible.txt" in result.content
        assert "Permission denied" in result.content
    
    @pytest.mark.asyncio
    async def test_ls_empty_directory(self):
        """Test listing empty directory."""
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isdir', return_value=True):
                with patch('os.listdir', return_value=[]):
                    with patch('os.path.abspath', return_value="/empty"):
                        result = await self.tool.execute(path="/empty")
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "Total: 0 directories, 0 files" in result.content
        assert result.metadata["total_files"] == 0
        assert result.metadata["total_directories"] == 0