"""Tests for REPL project analysis tools."""

import pytest
import pytest_asyncio
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, mock_open, AsyncMock

from flowlib.agent.runners.repl.tools.project import (
    ProjectAnalysisTool,
    BuildSystemTool,
    PackageManagerTool
)
from flowlib.agent.runners.repl.tools.base import ToolResult, ToolResultStatus, ToolParameter


class TestProjectAnalysisTool:
    """Test ProjectAnalysisTool implementation."""
    
    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return ProjectAnalysisTool()
    
    @pytest.fixture
    def mock_project_path(self, tmp_path):
        """Create mock project directory structure."""
        # Create basic project structure
        (tmp_path / "src").mkdir()
        (tmp_path / "test").mkdir()
        (tmp_path / "config").mkdir()
        
        # Create files
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "test-project",
            "version": "1.0.0",
            "dependencies": {"lodash": "^4.17.21"},
            "devDependencies": {"jest": "^27.0.0"},
            "scripts": {"test": "jest", "build": "webpack"}
        }))
        
        (tmp_path / "requirements.txt").write_text("requests==2.28.0\nflask==2.0.0")
        (tmp_path / ".gitignore").write_text("node_modules/\n*.pyc")
        (tmp_path / "README.md").write_text("# Test Project")
        
        # Create source files
        (tmp_path / "src" / "main.js").write_text("console.log('hello');")
        (tmp_path / "src" / "app.py").write_text("print('hello')")
        (tmp_path / "test" / "test_main.js").write_text("test('hello', () => {});")
        
        return tmp_path
    
    def test_tool_parameters(self, tool):
        """Test tool parameter definitions."""
        params = tool.parameters
        
        assert len(params) == 2
        
        path_param = next(p for p in params if p.name == "path")
        assert path_param.type == "str"
        assert path_param.required is False
        assert path_param.default == "."
        
        deps_param = next(p for p in params if p.name == "include_dependencies")
        assert deps_param.type == "bool"
        assert deps_param.required is False
        assert deps_param.default is True
    
    @pytest.mark.asyncio
    async def test_execute_success(self, tool, mock_project_path):
        """Test successful project analysis."""
        result = await tool.execute(path=str(mock_project_path), include_dependencies=True)
        
        assert result.status == ToolResultStatus.SUCCESS
        assert result.content is not None
        assert result.metadata is not None
        
        # Check metadata structure
        metadata = result.metadata
        assert "project_root" in metadata
        assert "project_type" in metadata
        assert "languages" in metadata
        assert "build_files" in metadata
        assert "structure" in metadata
        assert "dependencies" in metadata
        
        # Check project detection
        assert "node" in metadata["project_type"]
        assert "python" in metadata["project_type"]
        
        # Check language detection
        assert "javascript" in metadata["languages"]
        assert "python" in metadata["languages"]
        
        # Check build files
        assert "package.json" in metadata["build_files"]
        assert "requirements.txt" in metadata["build_files"]
    
    @pytest.mark.asyncio
    async def test_execute_without_dependencies(self, tool, mock_project_path):
        """Test project analysis without dependency analysis."""
        result = await tool.execute(path=str(mock_project_path), include_dependencies=False)
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "dependencies" not in result.metadata
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_path(self, tool):
        """Test analysis of non-existent project path."""
        result = await tool.execute(path="/nonexistent/path")
        
        assert result.status == ToolResultStatus.ERROR
        assert "does not exist" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_default_path(self, tool):
        """Test analysis with default path."""
        with patch.object(Path, 'resolve') as mock_resolve, \
             patch.object(Path, 'exists', return_value=True), \
             patch.object(tool, '_detect_project_type', return_value=["unknown"]), \
             patch.object(tool, '_detect_languages', return_value={}), \
             patch.object(tool, '_find_build_files', return_value=[]), \
             patch.object(tool, '_find_config_files', return_value=[]), \
             patch.object(tool, '_analyze_structure', return_value={
                 "total_files": 0, 
                 "total_directories": 0,
                 "source_directories": [],
                 "test_directories": [],
                 "config_directories": []
             }):
            
            mock_resolve.return_value = Path("/current/dir")
            
            result = await tool.execute()
            
            assert result.status == ToolResultStatus.SUCCESS
            mock_resolve.assert_called()
    
    def test_detect_project_type_node(self, tool, tmp_path):
        """Test Node.js project type detection."""
        (tmp_path / "package.json").write_text("{}")
        
        result = tool._detect_project_type(tmp_path)
        
        assert "node" in result
    
    def test_detect_project_type_python(self, tool, tmp_path):
        """Test Python project type detection."""
        (tmp_path / "requirements.txt").write_text("")
        
        result = tool._detect_project_type(tmp_path)
        
        assert "python" in result
    
    def test_detect_project_type_multiple(self, tool, tmp_path):
        """Test multiple project type detection."""
        (tmp_path / "package.json").write_text("{}")
        (tmp_path / "requirements.txt").write_text("")
        (tmp_path / "Cargo.toml").write_text("")
        
        result = tool._detect_project_type(tmp_path)
        
        assert "node" in result
        assert "python" in result
        assert "rust" in result
    
    def test_detect_project_type_unknown(self, tool, tmp_path):
        """Test unknown project type detection."""
        result = tool._detect_project_type(tmp_path)
        
        assert result == ["unknown"]
    
    def test_detect_languages(self, tool, tmp_path):
        """Test programming language detection."""
        # Create files with different extensions
        (tmp_path / "app.py").write_text("print('hello')")
        (tmp_path / "script.js").write_text("console.log('hello');")
        (tmp_path / "main.rs").write_text("fn main() {}")
        (tmp_path / "style.css").write_text("body { color: red; }")
        (tmp_path / "README.md").write_text("# Title")
        
        # Create subdirectory (should be ignored if starts with .)
        hidden_dir = tmp_path / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "test.py").write_text("hidden")
        
        result = tool._detect_languages(tmp_path)
        
        assert "python" in result
        assert "javascript" in result
        assert "rust" in result
        assert "css" in result
        assert "markdown" in result
        assert result["python"] == 1  # Should not count hidden files
    
    def test_find_build_files(self, tool, tmp_path):
        """Test build file detection."""
        # Create various build files
        (tmp_path / "package.json").write_text("{}")
        (tmp_path / "requirements.txt").write_text("")
        (tmp_path / "Cargo.toml").write_text("")
        (tmp_path / "Makefile").write_text("")
        (tmp_path / "docker-compose.yml").write_text("")
        (tmp_path / "unknown.txt").write_text("")  # Should not be detected
        
        result = tool._find_build_files(tmp_path)
        
        assert "package.json" in result
        assert "requirements.txt" in result
        assert "Cargo.toml" in result
        assert "Makefile" in result
        assert "docker-compose.yml" in result
        assert "unknown.txt" not in result
    
    def test_find_config_files(self, tool, tmp_path):
        """Test configuration file detection."""
        (tmp_path / ".gitignore").write_text("")
        (tmp_path / ".eslintrc.json").write_text("{}")
        (tmp_path / "tsconfig.json").write_text("{}")
        (tmp_path / ".env").write_text("")
        (tmp_path / "regular.txt").write_text("")  # Should not be detected
        
        result = tool._find_config_files(tmp_path)
        
        assert ".gitignore" in result
        assert ".eslintrc.json" in result
        assert "tsconfig.json" in result
        assert ".env" in result
        assert "regular.txt" not in result
    
    def test_analyze_structure(self, tool, tmp_path):
        """Test project structure analysis."""
        # Create directory structure
        (tmp_path / "src").mkdir()
        (tmp_path / "test").mkdir()
        (tmp_path / "config").mkdir()
        (tmp_path / "docs").mkdir()
        (tmp_path / ".hidden").mkdir()  # Should be ignored
        
        # Create files
        (tmp_path / "main.py").write_text("")
        (tmp_path / "src" / "app.py").write_text("")
        (tmp_path / "test" / "test_app.py").write_text("")
        (tmp_path / ".hidden" / "file.py").write_text("")  # Should be ignored
        
        result = tool._analyze_structure(tmp_path)
        
        assert result["total_files"] == 3  # Excludes hidden files
        assert result["total_directories"] == 4  # Excludes hidden directories
        assert "src" in result["source_directories"]
        assert "test" in result["test_directories"]
        assert "config" in result["config_directories"]
    
    def test_analyze_node_dependencies(self, tool, tmp_path):
        """Test Node.js dependency analysis."""
        package_json = {
            "name": "test-app",
            "version": "1.0.0",
            "dependencies": {"lodash": "^4.17.21", "express": "^4.18.0"},
            "devDependencies": {"jest": "^27.0.0"},
            "scripts": {"test": "jest", "start": "node app.js"},
            "engines": {"node": ">=14.0.0"}
        }
        
        (tmp_path / "package.json").write_text(json.dumps(package_json))
        
        result = tool._analyze_node_dependencies(tmp_path)
        
        assert result["name"] == "test-app"
        assert result["version"] == "1.0.0"
        assert len(result["dependencies"]) == 2
        assert "lodash" in result["dependencies"]
        assert "express" in result["dependencies"]
        assert len(result["devDependencies"]) == 1
        assert "jest" in result["devDependencies"]
        assert "test" in result["scripts"]
        assert "node" in result["engines"]
    
    def test_analyze_node_dependencies_no_file(self, tool, tmp_path):
        """Test Node.js dependency analysis with no package.json."""
        result = tool._analyze_node_dependencies(tmp_path)
        
        assert result == {}
    
    def test_analyze_node_dependencies_invalid_json(self, tool, tmp_path):
        """Test Node.js dependency analysis with invalid JSON."""
        (tmp_path / "package.json").write_text("invalid json")
        
        result = tool._analyze_node_dependencies(tmp_path)
        
        assert result == {}
    
    def test_analyze_python_dependencies_requirements(self, tool, tmp_path):
        """Test Python dependency analysis with requirements.txt."""
        requirements = "requests==2.28.0\nflask>=2.0.0\n# Comment line\nnumpy"
        (tmp_path / "requirements.txt").write_text(requirements)
        
        result = tool._analyze_python_dependencies(tmp_path)
        
        assert "requirements" in result
        reqs = result["requirements"]
        assert "requests==2.28.0" in reqs
        assert "flask>=2.0.0" in reqs
        assert "numpy" in reqs
        assert len(reqs) == 3  # Should exclude comments and empty lines
    
    def test_analyze_python_dependencies_pyproject(self, tool, tmp_path):
        """Test Python dependency analysis with pyproject.toml."""
        # Mock tomllib since it's Python 3.11+
        mock_toml_data = {
            "project": {
                "dependencies": ["requests>=2.25.0", "click>=8.0.0"]
            }
        }
        
        (tmp_path / "pyproject.toml").write_text("[project]\ndependencies = ['requests>=2.25.0', 'click>=8.0.0']")
        
        with patch('builtins.__import__') as mock_import:
            mock_tomllib = Mock()
            mock_tomllib.load.return_value = mock_toml_data
            
            def import_side_effect(name, *args, **kwargs):
                if name == 'tomllib':
                    return mock_tomllib
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            result = tool._analyze_python_dependencies(tmp_path)
            
            assert "pyproject" in result
            deps = result["pyproject"]
            assert "requests>=2.25.0" in deps
            assert "click>=8.0.0" in deps
    
    def test_analyze_rust_dependencies(self, tool, tmp_path):
        """Test Rust dependency analysis."""
        mock_toml_data = {
            "package": {"name": "test-app", "version": "0.1.0"},
            "dependencies": {"serde": "1.0", "tokio": "1.0"},
            "dev-dependencies": {"criterion": "0.3"}
        }
        
        (tmp_path / "Cargo.toml").write_text("")  # File exists
        
        with patch('builtins.__import__') as mock_import:
            mock_tomllib = Mock()
            mock_tomllib.load.return_value = mock_toml_data
            
            def import_side_effect(name, *args, **kwargs):
                if name == 'tomllib':
                    return mock_tomllib
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            result = tool._analyze_rust_dependencies(tmp_path)
            
            assert "package" in result
            assert result["package"]["name"] == "test-app"
            assert "dependencies" in result
            assert "serde" in result["dependencies"]
            assert "dev_dependencies" in result
            assert "criterion" in result["dev_dependencies"]
    
    def test_analyze_maven_dependencies(self, tool, tmp_path):
        """Test Maven dependency analysis."""
        pom_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <project xmlns="http://maven.apache.org/POM/4.0.0">
            <dependencies>
                <dependency>
                    <groupId>junit</groupId>
                    <artifactId>junit</artifactId>
                    <version>4.13.2</version>
                </dependency>
                <dependency>
                    <groupId>org.springframework</groupId>
                    <artifactId>spring-core</artifactId>
                </dependency>
            </dependencies>
        </project>"""
        
        (tmp_path / "pom.xml").write_text(pom_xml)
        
        result = tool._analyze_maven_dependencies(tmp_path)
        
        assert "dependencies" in result
        deps = result["dependencies"]
        assert len(deps) == 2
        
        junit_dep = next(d for d in deps if d["artifactId"] == "junit")
        assert junit_dep["groupId"] == "junit"
        assert junit_dep["version"] == "4.13.2"
        
        spring_dep = next(d for d in deps if d["artifactId"] == "spring-core")
        assert spring_dep["groupId"] == "org.springframework"
        assert "version" not in spring_dep  # No version specified
    
    def test_format_analysis(self, tool):
        """Test analysis result formatting."""
        analysis = {
            "project_root": "/test/project",
            "project_type": ["node", "python"],
            "languages": {"javascript": 10, "python": 5, "css": 3},
            "build_files": ["package.json", "requirements.txt"],
            "structure": {
                "total_files": 25,
                "total_directories": 8,
                "source_directories": ["src", "lib"],
                "test_directories": ["test", "__tests__"]
            },
            "dependencies": {
                "node": {"dependencies": {"lodash": "^4.17.21"}},
                "python": {"requirements": ["requests==2.28.0"]}
            }
        }
        
        result = tool._format_analysis(analysis)
        
        assert "🔍 **Project Analysis**" in result
        assert "/test/project" in result
        assert "node, python" in result
        assert "javascript: 10 files" in result
        assert "package.json" in result
        assert "Files: 25" in result
        assert "Source dirs: src, lib" in result
        assert "Test dirs: test, __tests__" in result
        assert "Node.js: 1 deps" in result
        assert "Python: 1 requirements" in result


class TestBuildSystemTool:
    """Test BuildSystemTool implementation."""
    
    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return BuildSystemTool()
    
    @pytest.fixture
    def mock_subprocess_success(self):
        """Mock successful subprocess execution."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Build successful"
        mock_result.stderr = ""
        return mock_result
    
    @pytest.fixture
    def mock_subprocess_failure(self):
        """Mock failed subprocess execution."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = "Some output"
        mock_result.stderr = "Build failed"
        return mock_result
    
    def test_tool_parameters(self, tool):
        """Test tool parameter definitions."""
        params = tool.parameters
        
        assert len(params) == 3
        
        command_param = next(p for p in params if p.name == "command")
        assert command_param.required is True
        
        path_param = next(p for p in params if p.name == "path")
        assert path_param.required is False
        assert path_param.default == "."
        
        args_param = next(p for p in params if p.name == "args")
        assert args_param.type == "list"
        assert args_param.required is False
    
    @pytest.mark.asyncio
    async def test_execute_success(self, tool, tmp_path, mock_subprocess_success):
        """Test successful build command execution."""
        (tmp_path / "package.json").write_text("{}")
        
        with patch('flowlib.agent.runners.repl.tools.project.subprocess.run', return_value=mock_subprocess_success):
            result = await tool.execute(command="build", path=str(tmp_path))
            
            assert result.status == ToolResultStatus.SUCCESS
            assert "Command completed successfully" in result.content
            assert "Build successful" in result.content
            assert result.metadata["exit_code"] == 0
    
    @pytest.mark.asyncio
    async def test_execute_failure(self, tool, tmp_path, mock_subprocess_failure):
        """Test failed build command execution."""
        (tmp_path / "package.json").write_text("{}")
        
        with patch('flowlib.agent.runners.repl.tools.project.subprocess.run', return_value=mock_subprocess_failure):
            result = await tool.execute(command="build", path=str(tmp_path))
            
            assert result.status == ToolResultStatus.ERROR
            assert "Command failed" in result.content
            assert "Build failed" in result.content
            assert result.metadata["exit_code"] == 1
    
    @pytest.mark.asyncio
    async def test_execute_timeout(self, tool, tmp_path):
        """Test build command timeout."""
        (tmp_path / "package.json").write_text("{}")
        
        with patch('flowlib.agent.runners.repl.tools.project.subprocess.run', side_effect=subprocess.TimeoutExpired("cmd", 300)):
            result = await tool.execute(command="build", path=str(tmp_path))
            
            assert result.status == ToolResultStatus.ERROR
            assert "timed out" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_path(self, tool):
        """Test build command with non-existent path."""
        result = await tool.execute(command="build", path="/nonexistent")
        
        assert result.status == ToolResultStatus.ERROR
        assert "does not exist" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_no_build_system(self, tool, tmp_path):
        """Test build command with no recognized build system."""
        result = await tool.execute(command="build", path=str(tmp_path))
        
        assert result.status == ToolResultStatus.ERROR
        assert "No build system found" in result.error
    
    def test_node_command_mapping(self, tool):
        """Test Node.js command mapping."""
        assert tool._node_command("install", []) == ["npm", "install"]
        assert tool._node_command("build", []) == ["npm", "run", "build"]
        assert tool._node_command("test", ["--verbose"]) == ["npm", "test", "--verbose"]
        assert tool._node_command("custom", []) == ["npm", "run", "custom"]  # Custom script
    
    def test_python_command_mapping(self, tool, tmp_path):
        """Test Python command mapping."""
        assert tool._python_command("test", [], tmp_path) == ["python", "-m", "pytest"]
        assert tool._python_command("lint", [], tmp_path) == ["python", "-m", "flake8"]
        
        # Test pyproject.toml install
        (tmp_path / "pyproject.toml").write_text("")
        assert tool._python_command("install", [], tmp_path) == ["pip", "install", "-e", "."]
        
        # Test custom command
        assert tool._python_command("mymodule", ["arg"], tmp_path) == ["python", "mymodule", "arg"]
    
    def test_rust_command_mapping(self, tool):
        """Test Rust command mapping."""
        assert tool._rust_command("build", []) == ["cargo", "build"]
        assert tool._rust_command("test", ["--verbose"]) == ["cargo", "test", "--verbose"]
        assert tool._rust_command("custom", []) == ["cargo", "custom"]
    
    def test_maven_command_mapping(self, tool):
        """Test Maven command mapping."""
        assert tool._maven_command("install", []) == ["mvn", "install"]
        assert tool._maven_command("test", []) == ["mvn", "test"]
        assert tool._maven_command("custom", ["arg"]) == ["mvn", "custom", "arg"]
    
    def test_gradle_command_mapping(self, tool):
        """Test Gradle command mapping."""
        assert tool._gradle_command("build", []) == ["./gradlew", "build"]
        assert tool._gradle_command("test", []) == ["./gradlew", "test"]
        assert tool._gradle_command("custom", []) == ["./gradlew", "custom"]
    
    def test_go_command_mapping(self, tool):
        """Test Go command mapping."""
        assert tool._go_command("build", []) == ["go", "build"]
        assert tool._go_command("test", []) == ["go", "test"]
        assert tool._go_command("install", []) == ["go", "mod", "download"]
        assert tool._go_command("custom", []) == ["go", "custom"]


class TestPackageManagerTool:
    """Test PackageManagerTool implementation."""
    
    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return PackageManagerTool()
    
    @pytest.fixture
    def mock_subprocess_success(self):
        """Mock successful subprocess execution."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Package installed successfully"
        mock_result.stderr = ""
        return mock_result
    
    def test_tool_parameters(self, tool):
        """Test tool parameter definitions."""
        params = tool.parameters
        
        assert len(params) == 4
        
        action_param = next(p for p in params if p.name == "action")
        assert action_param.required is True
        
        package_param = next(p for p in params if p.name == "package")
        assert package_param.required is False
        
        dev_param = next(p for p in params if p.name == "dev")
        assert dev_param.type == "bool"
        assert dev_param.default is False
    
    @pytest.mark.asyncio
    async def test_execute_npm_install(self, tool, tmp_path, mock_subprocess_success):
        """Test npm package installation."""
        (tmp_path / "package.json").write_text("{}")
        
        with patch('flowlib.agent.runners.repl.tools.project.subprocess.run', return_value=mock_subprocess_success):
            result = await tool.execute(action="install", path=str(tmp_path))
            
            assert result.status == ToolResultStatus.SUCCESS
            assert "Command completed successfully" in result.content
    
    @pytest.mark.asyncio
    async def test_execute_npm_add_package(self, tool, tmp_path, mock_subprocess_success):
        """Test npm package addition."""
        (tmp_path / "package.json").write_text("{}")
        
        with patch('flowlib.agent.runners.repl.tools.project.subprocess.run', return_value=mock_subprocess_success) as mock_run:
            result = await tool.execute(action="add", package="lodash", path=str(tmp_path))
            
            assert result.status == ToolResultStatus.SUCCESS
            mock_run.assert_called_once()
            # Verify the command includes the package
            called_cmd = mock_run.call_args[0][0]
            assert "lodash" in called_cmd
    
    @pytest.mark.asyncio
    async def test_execute_npm_add_dev_package(self, tool, tmp_path, mock_subprocess_success):
        """Test npm dev package addition."""
        (tmp_path / "package.json").write_text("{}")
        
        with patch('flowlib.agent.runners.repl.tools.project.subprocess.run', return_value=mock_subprocess_success) as mock_run:
            result = await tool.execute(action="add", package="jest", path=str(tmp_path), dev=True)
            
            assert result.status == ToolResultStatus.SUCCESS
            called_cmd = mock_run.call_args[0][0]
            assert "jest" in called_cmd
            assert "--save-dev" in called_cmd
    
    @pytest.mark.asyncio
    async def test_execute_pip_install(self, tool, tmp_path, mock_subprocess_success):
        """Test pip package installation."""
        (tmp_path / "requirements.txt").write_text("requests")
        
        with patch('flowlib.agent.runners.repl.tools.project.subprocess.run', return_value=mock_subprocess_success) as mock_run:
            result = await tool.execute(action="install", path=str(tmp_path))
            
            assert result.status == ToolResultStatus.SUCCESS
            called_cmd = mock_run.call_args[0][0]
            assert "pip" in called_cmd
            assert "requirements.txt" in called_cmd
    
    @pytest.mark.asyncio
    async def test_execute_cargo_add(self, tool, tmp_path, mock_subprocess_success):
        """Test cargo package addition."""
        (tmp_path / "Cargo.toml").write_text("")
        
        with patch('flowlib.agent.runners.repl.tools.project.subprocess.run', return_value=mock_subprocess_success) as mock_run:
            result = await tool.execute(action="add", package="serde", path=str(tmp_path))
            
            assert result.status == ToolResultStatus.SUCCESS
            called_cmd = mock_run.call_args[0][0]
            assert "cargo" in called_cmd
            assert "add" in called_cmd
            assert "serde" in called_cmd
    
    @pytest.mark.asyncio
    async def test_execute_no_package_manager(self, tool, tmp_path):
        """Test execution with no recognized package manager."""
        result = await tool.execute(action="install", path=str(tmp_path))
        
        assert result.status == ToolResultStatus.ERROR
        assert "No package manager found" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_timeout(self, tool, tmp_path):
        """Test package manager command timeout."""
        (tmp_path / "package.json").write_text("{}")
        
        with patch('flowlib.agent.runners.repl.tools.project.subprocess.run', side_effect=subprocess.TimeoutExpired("cmd", 180)):
            result = await tool.execute(action="install", path=str(tmp_path))
            
            assert result.status == ToolResultStatus.ERROR
            assert "timed out" in result.error
    
    def test_npm_package_commands(self, tool):
        """Test npm package command generation."""
        assert tool._npm_package_command("install", None, False) == ["npm", "install"]
        assert tool._npm_package_command("add", "lodash", False) == ["npm", "install", "lodash"]
        assert tool._npm_package_command("add", "jest", True) == ["npm", "install", "jest", "--save-dev"]
        assert tool._npm_package_command("remove", "lodash", False) == ["npm", "uninstall", "lodash"]
        assert tool._npm_package_command("update", None, False) == ["npm", "update"]
        assert tool._npm_package_command("update", "lodash", False) == ["npm", "update", "lodash"]
        assert tool._npm_package_command("list", None, False) == ["npm", "list"]
        assert tool._npm_package_command("invalid", None, False) is None
    
    def test_pip_package_commands(self, tool):
        """Test pip package command generation."""
        assert tool._pip_package_command("install", None) == ["pip", "install", "-r", "requirements.txt"]
        assert tool._pip_package_command("add", "requests") == ["pip", "install", "requests"]
        assert tool._pip_package_command("remove", "requests") == ["pip", "uninstall", "requests"]
        assert tool._pip_package_command("list", None) == ["pip", "list"]
        assert tool._pip_package_command("invalid", None) is None
    
    def test_cargo_package_commands(self, tool):
        """Test cargo package command generation."""
        assert tool._cargo_package_command("add", "serde") == ["cargo", "add", "serde"]
        assert tool._cargo_package_command("remove", "serde") == ["cargo", "remove", "serde"]
        assert tool._cargo_package_command("update", None) == ["cargo", "update"]
        assert tool._cargo_package_command("invalid", None) is None


class TestProjectToolsIntegration:
    """Test integration between project tools."""
    
    @pytest.fixture
    def analysis_tool(self):
        return ProjectAnalysisTool()
    
    @pytest.fixture
    def build_tool(self):
        return BuildSystemTool()
    
    @pytest.fixture
    def package_tool(self):
        return PackageManagerTool()
    
    @pytest.fixture
    def complex_project(self, tmp_path):
        """Create a complex multi-language project."""
        # Create project structure
        (tmp_path / "src").mkdir()
        (tmp_path / "test").mkdir()
        (tmp_path / "frontend").mkdir()
        (tmp_path / "backend").mkdir()
        
        # Frontend (Node.js)
        (tmp_path / "frontend" / "src").mkdir(parents=True)
        (tmp_path / "frontend" / "package.json").write_text(json.dumps({
            "name": "frontend",
            "scripts": {"build": "webpack", "test": "jest"}
        }))
        (tmp_path / "frontend" / "src" / "app.js").write_text("// Frontend app")
        
        # Backend (Python)
        (tmp_path / "backend" / "requirements.txt").write_text("flask==2.0.0")
        (tmp_path / "backend" / "app.py").write_text("# Backend app")
        
        # Root level
        (tmp_path / "README.md").write_text("# Complex Project")
        (tmp_path / "docker-compose.yml").write_text("version: '3'")
        
        return tmp_path
    
    @pytest.mark.asyncio
    async def test_comprehensive_project_analysis(self, analysis_tool, complex_project):
        """Test comprehensive analysis of complex project."""
        result = await analysis_tool.execute(path=str(complex_project))
        
        assert result.status == ToolResultStatus.SUCCESS
        
        # Should detect multiple project types at different levels
        metadata = result.metadata
        assert "javascript" in metadata["languages"]
        assert "python" in metadata["languages"]
        assert "markdown" in metadata["languages"]
        
        # Should find various build and config files
        build_files = metadata["build_files"]
        assert any("docker-compose.yml" in f for f in build_files)
        
        # Should analyze structure
        structure = metadata["structure"]
        assert structure["total_files"] >= 4
        assert structure["total_directories"] >= 4
    
    @pytest.mark.asyncio
    async def test_error_handling_consistency(self, analysis_tool, build_tool, package_tool):
        """Test consistent error handling across tools."""
        nonexistent_path = "/this/path/does/not/exist"
        
        # All tools should handle non-existent paths consistently
        analysis_result = await analysis_tool.execute(path=nonexistent_path)
        build_result = await build_tool.execute(command="build", path=nonexistent_path)
        package_result = await package_tool.execute(action="install", path=nonexistent_path)
        
        assert analysis_result.status == ToolResultStatus.ERROR
        assert build_result.status == ToolResultStatus.ERROR
        assert package_result.status == ToolResultStatus.ERROR
        
        assert "does not exist" in analysis_result.error
        assert "does not exist" in build_result.error
        assert "does not exist" in package_result.error