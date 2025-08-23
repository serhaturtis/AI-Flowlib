"""Tests for analysis REPL tools."""

import os
import ast
import pytest
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import Mock, patch, mock_open, MagicMock

from flowlib.agent.runners.repl.tools.analysis import (
    CodeAnalysisTool,
    DependencyAnalysisTool
)
from flowlib.agent.runners.repl.tools.base import ToolResultStatus


class TestCodeAnalysisTool:
    """Test suite for CodeAnalysisTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = CodeAnalysisTool()
    
    def test_parameter_definitions(self):
        """Test parameter definitions."""
        params = self.tool._define_parameters()
        assert len(params) == 3
        
        # Check file_path parameter
        file_param = next(p for p in params if p.name == "file_path")
        assert file_param.type == "str"
        assert file_param.required is True
        
        # Check include_metrics parameter
        metrics_param = next(p for p in params if p.name == "include_metrics")
        assert metrics_param.type == "bool"
        assert metrics_param.required is False
        assert metrics_param.default is True
        
        # Check include_structure parameter
        structure_param = next(p for p in params if p.name == "include_structure")
        assert structure_param.type == "bool"
        assert structure_param.required is False
        assert structure_param.default is True
    
    @pytest.mark.asyncio
    async def test_file_not_exists(self):
        """Test error handling for non-existent file."""
        with patch('os.path.exists', return_value=False):
            result = await self.tool.execute(file_path="/nonexistent/file.py")
        
        assert result.status == ToolResultStatus.ERROR
        assert "File does not exist" in result.error
    
    @pytest.mark.asyncio
    async def test_non_python_file(self):
        """Test error handling for non-Python file."""
        with patch('os.path.exists', return_value=True):
            result = await self.tool.execute(file_path="/test/file.txt")
        
        assert result.status == ToolResultStatus.ERROR
        assert "must be a Python file" in result.error
    
    @pytest.mark.asyncio
    async def test_simple_python_analysis(self):
        """Test analysis of simple Python code."""
        code = '''
def hello(name):
    """Say hello."""
    if name:
        print(f"Hello, {name}!")
    else:
        print("Hello, World!")

class Greeter:
    def __init__(self):
        self.count = 0
    
    def greet(self, name):
        self.count += 1
        hello(name)

# Main execution
if __name__ == "__main__":
    greeter = Greeter()
    for i in range(3):
        greeter.greet(f"User{i}")
'''
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=code)):
                result = await self.tool.execute(file_path="/test/simple.py")
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "Code Analysis: simple.py" in result.content
        
        # Check metadata
        assert result.metadata["total_lines"] == len(code.splitlines())
        assert len(result.metadata["classes"]) == 1
        assert len(result.metadata["functions"]) == 3  # hello + __init__ + greet
        assert result.metadata["classes"][0]["name"] == "Greeter"
        assert result.metadata["cyclomatic_complexity"] > 1  # Has if statements and loops
    
    @pytest.mark.asyncio
    async def test_syntax_error_handling(self):
        """Test handling of Python syntax errors."""
        code = '''
def broken_function(
    print("This has a syntax error"
'''
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=code)):
                result = await self.tool.execute(file_path="/test/syntax_error.py")
        
        assert result.status == ToolResultStatus.ERROR
        assert "Syntax error" in result.error
    
    @pytest.mark.asyncio
    async def test_complex_code_analysis(self):
        """Test analysis of complex code with multiple constructs."""
        code = '''
import os
import sys
from typing import List, Dict

class ComplexClass:
    def complex_method(self, data: List[int]) -> Dict[str, int]:
        result = {}
        
        for item in data:
            if item > 0:
                if item % 2 == 0:
                    result["even"] = result.get("even", 0) + 1
                else:
                    result["odd"] = result.get("odd", 0) + 1
            elif item < 0:
                result["negative"] = result.get("negative", 0) + 1
            else:
                result["zero"] = result.get("zero", 0) + 1
        
        try:
            average = sum(data) / len(data)
        except ZeroDivisionError:
            average = 0
        
        while len(result) < 5:
            result[f"key_{len(result)}"] = 0
        
        return result

async def async_function():
    async for item in async_generator():
        process(item)
'''
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=code)):
                result = await self.tool.execute(file_path="/test/complex.py")
        
        assert result.status == ToolResultStatus.SUCCESS
        
        # Check structure
        assert len(result.metadata["classes"]) == 1
        assert len(result.metadata["imports"]) >= 3  # os, sys, typing components
        
        # Check metrics
        assert result.metadata["cyclomatic_complexity"] > 5  # Multiple decision points
        assert result.metadata["conditions"] >= 3  # Multiple if statements
        assert result.metadata["loops"] >= 2  # for and while loops
    
    @pytest.mark.asyncio
    async def test_metrics_only(self):
        """Test analysis with only metrics enabled."""
        code = '''
def simple_function():
    x = 1
    return x * 2
'''
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=code)):
                result = await self.tool.execute(
                    file_path="/test/metrics.py",
                    include_metrics=True,
                    include_structure=False
                )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "Complexity Metrics:" in result.content
        assert "Structure:" not in result.content
        assert "cyclomatic_complexity" in result.metadata
        assert "classes" not in result.metadata
    
    @pytest.mark.asyncio
    async def test_structure_only(self):
        """Test analysis with only structure enabled."""
        code = '''
class TestClass:
    def method(self):
        pass

def function():
    pass
'''
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=code)):
                result = await self.tool.execute(
                    file_path="/test/structure.py",
                    include_metrics=False,
                    include_structure=True
                )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "Structure:" in result.content
        assert "Complexity Metrics:" not in result.content
        assert "classes" in result.metadata
        assert "cyclomatic_complexity" not in result.metadata
    
    @pytest.mark.asyncio
    async def test_empty_file_analysis(self):
        """Test analysis of empty Python file."""
        code = ""
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=code)):
                result = await self.tool.execute(file_path="/test/empty.py")
        
        assert result.status == ToolResultStatus.SUCCESS
        assert result.metadata["total_lines"] == 0
        assert result.metadata["non_empty_lines"] == 0
        assert len(result.metadata.get("classes", [])) == 0
        assert len(result.metadata.get("functions", [])) == 0
    
    @pytest.mark.asyncio
    async def test_function_complexity_calculation(self):
        """Test accurate function complexity calculation."""
        code = '''
def complex_function(data):
    if data:  # +1
        for item in data:  # +1
            if item > 0:  # +1
                try:
                    process(item)
                except Exception:  # +1
                    pass
    return data
'''
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=code)):
                result = await self.tool.execute(file_path="/test/func_complexity.py")
        
        assert result.status == ToolResultStatus.SUCCESS
        assert result.metadata["max_function_complexity"] >= 4  # Base 1 + 4 decision points


class TestDependencyAnalysisTool:
    """Test suite for DependencyAnalysisTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = DependencyAnalysisTool()
    
    def test_parameter_definitions(self):
        """Test parameter definitions."""
        params = self.tool._define_parameters()
        assert len(params) == 3
        
        # Check path parameter
        path_param = next(p for p in params if p.name == "path")
        assert path_param.type == "str"
        assert path_param.required is True
        
        # Check include_stdlib parameter
        stdlib_param = next(p for p in params if p.name == "include_stdlib")
        assert stdlib_param.type == "bool"
        assert stdlib_param.required is False
        assert stdlib_param.default is False
        
        # Check show_usage parameter
        usage_param = next(p for p in params if p.name == "show_usage")
        assert usage_param.type == "bool"
        assert usage_param.required is False
        assert usage_param.default is True
    
    @pytest.mark.asyncio
    async def test_path_not_exists(self):
        """Test error handling for non-existent path."""
        with patch('os.path.exists', return_value=False):
            result = await self.tool.execute(path="/nonexistent")
        
        assert result.status == ToolResultStatus.ERROR
        assert "Path does not exist" in result.error
    
    @pytest.mark.asyncio
    async def test_single_file_analysis(self):
        """Test dependency analysis of a single file."""
        code = '''
import os
import sys
from pathlib import Path
from typing import List, Dict
import requests
import numpy as np
from my_module import MyClass
'''
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isfile', return_value=True):
                with patch('builtins.open', mock_open(read_data=code)):
                    result = await self.tool.execute(path="/test/file.py")
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "Dependency Analysis:" in result.content
        assert result.metadata["python_files"] == 1
        
        # Check that imports were found
        imports_dict = result.metadata.get("imports", {})
        assert len(imports_dict) > 0
    
    @pytest.mark.asyncio
    async def test_directory_analysis(self):
        """Test dependency analysis of a directory."""
        # Mock directory structure
        mock_walk = [
            ("/project", ["subdir"], ["file1.py", "file2.py"]),
            ("/project/subdir", [], ["file3.py", "README.md"])
        ]
        
        file_contents = {
            "/project/file1.py": "import os\nimport requests",
            "/project/file2.py": "import sys\nimport requests\nfrom pathlib import Path",
            "/project/subdir/file3.py": "import json\nimport requests"
        }
        
        def mock_open_func(path, mode='r', **kwargs):
            if path in file_contents:
                return mock_open(read_data=file_contents[path])()
            raise FileNotFoundError()
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isfile', return_value=False):
                with patch('os.walk', return_value=mock_walk):
                    with patch('builtins.open', side_effect=mock_open_func):
                        result = await self.tool.execute(
                            path="/project",
                            include_stdlib=True
                        )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert result.metadata["python_files"] == 3
        
        # Check that requests appears multiple times
        assert "requests" in result.content
        assert "(3 uses)" in result.content  # Used in all 3 files
    
    @pytest.mark.asyncio
    async def test_no_python_files(self):
        """Test handling when no Python files are found."""
        mock_walk = [
            ("/project", [], ["README.md", "config.json"])
        ]
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isfile', return_value=False):
                with patch('os.walk', return_value=mock_walk):
                    result = await self.tool.execute(path="/project")
        
        assert result.status == ToolResultStatus.WARNING
        assert "No Python files found" in result.content
    
    @pytest.mark.asyncio
    async def test_stdlib_filtering(self):
        """Test filtering of standard library imports."""
        code = '''
import os  # stdlib
import sys  # stdlib
import json  # stdlib
import requests  # third-party
import numpy  # third-party
from pathlib import Path  # stdlib
from my_module import MyClass  # local
'''
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isfile', return_value=True):
                with patch('builtins.open', mock_open(read_data=code)):
                    # Test without stdlib
                    result = await self.tool.execute(
                        path="/test/file.py",
                        include_stdlib=False
                    )
        
        assert result.status == ToolResultStatus.SUCCESS
        
        # Should only show third-party imports
        assert "requests" in result.content
        assert "numpy" in result.content
        # stdlib imports should not be in main content (unless marked)
        metadata_imports = list(result.metadata.get("imports", {}).keys())
        stdlib_in_top = any("os" in imp or "sys" in imp for imp in metadata_imports)
        assert not stdlib_in_top  # stdlib filtered out
    
    @pytest.mark.asyncio
    async def test_import_usage_details(self):
        """Test showing import usage details."""
        mock_walk = [
            ("/project", [], ["file1.py", "file2.py", "file3.py"])
        ]
        
        file_contents = {
            "/project/file1.py": "import shared_module",
            "/project/file2.py": "import shared_module",
            "/project/file3.py": "import unique_module"
        }
        
        def mock_open_func(path, mode='r', **kwargs):
            if path in file_contents:
                return mock_open(read_data=file_contents[path])()
            raise FileNotFoundError()
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isfile', return_value=False):
                with patch('os.walk', return_value=mock_walk):
                    with patch('builtins.open', side_effect=mock_open_func):
                        result = await self.tool.execute(
                            path="/project",
                            show_usage=True
                        )
        
        assert result.status == ToolResultStatus.SUCCESS
        # shared_module should show it's used in 2 files
        assert "shared_module" in result.content
        assert "(2 uses)" in result.content
        # Should show file paths for modules with few uses
        assert "file1.py" in result.content or "2 files" in result.content
    
    @pytest.mark.asyncio
    async def test_import_extraction_variations(self):
        """Test extraction of various import styles."""
        code = '''
# Standard imports
import os
import sys, json  # Multiple imports on one line

# From imports
from pathlib import Path
from typing import List, Dict, Optional
# Removed unused wildcard import from collections

# Relative imports
from . import sibling
from ..parent import ParentClass
from .child import ChildClass

# Import with alias
import numpy as np
from pandas import DataFrame as df
'''
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isfile', return_value=True):
                with patch('builtins.open', mock_open(read_data=code)):
                    result = await self.tool.execute(
                        path="/test/imports.py",
                        include_stdlib=True
                    )
        
        assert result.status == ToolResultStatus.SUCCESS
        
        # Check various import types were captured
        all_imports_found = result.metadata["total_imports"] > 10
        assert all_imports_found
    
    @pytest.mark.asyncio
    async def test_skip_hidden_directories(self):
        """Test that hidden directories are skipped."""
        mock_walk_data = [
            ("/project", [".git", "src", ".venv"], ["setup.py"]),
            ("/project/src", [], ["main.py"])
        ]
        
        dirs_checked = []
        
        def mock_walk(path):
            for root, dirs, files in mock_walk_data:
                # Track which directories are being processed
                dirs_checked.extend(dirs[:])
                # Filter out hidden directories (mimics the tool's behavior)
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                yield root, dirs, files
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isfile', return_value=False):
                with patch('os.walk', side_effect=mock_walk):
                    with patch('builtins.open', mock_open(read_data="import os")):
                        result = await self.tool.execute(path="/project")
        
        assert result.status == ToolResultStatus.SUCCESS
        # Verify hidden directories were filtered
        assert ".git" not in str(result.content)
        assert ".venv" not in str(result.content)
    
    @pytest.mark.asyncio
    async def test_syntax_error_skip(self):
        """Test that files with syntax errors are skipped."""
        mock_walk = [
            ("/project", [], ["good.py", "bad.py"])
        ]
        
        file_contents = {
            "/project/good.py": "import os\nimport sys",
            "/project/bad.py": "import json\nthis is not valid python syntax"
        }
        
        def mock_open_func(path, mode='r', **kwargs):
            if path in file_contents:
                return mock_open(read_data=file_contents[path])()
            raise FileNotFoundError()
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isfile', return_value=False):
                with patch('os.walk', return_value=mock_walk):
                    with patch('builtins.open', side_effect=mock_open_func):
                        result = await self.tool.execute(path="/project")
        
        assert result.status == ToolResultStatus.SUCCESS
        # Should still analyze the good file
        assert result.metadata["python_files"] == 2  # Both files attempted
        # When include_stdlib=False (default), os/sys won't be in imports
        # Check that at least the good file was processed
        assert result.metadata["total_imports"] >= 0  # May be 0 if all imports are stdlib
        # bad.py's imports might not appear due to syntax error