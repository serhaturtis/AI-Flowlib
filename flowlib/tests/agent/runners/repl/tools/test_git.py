"""Tests for REPL git integration tools."""

import pytest
import pytest_asyncio
import subprocess
import os
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, mock_open, AsyncMock

from flowlib.agent.runners.repl.tools.git import (
    GitStatusTool,
    GitDiffTool,
    GitAddTool,
    GitCommitTool,
    GitLogTool
)
from flowlib.agent.runners.repl.tools.base import ToolResult, ToolResultStatus, ToolParameter


class TestGitStatusTool:
    """Test GitStatusTool implementation."""
    
    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return GitStatusTool()
    
    @pytest.fixture
    def mock_subprocess_success(self):
        """Mock successful git status subprocess."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """On branch main
Your branch is up to date with 'origin/main'.

Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

	new file:   test.py

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   existing.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)

	untracked.py"""
        mock_result.stderr = ""
        return mock_result
    
    @pytest.fixture
    def mock_subprocess_porcelain(self):
        """Mock git status porcelain output."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """A  test.py
 M existing.py
?? untracked.py"""
        mock_result.stderr = ""
        return mock_result
    
    @pytest.fixture
    def mock_git_repo_check(self):
        """Mock git repository check."""
        return True
    
    def test_tool_parameters(self, tool):
        """Test tool parameter definitions."""
        params = tool.parameters
        
        assert len(params) == 2
        
        path_param = next(p for p in params if p.name == "path")
        assert path_param.type == "str"
        assert path_param.required is False
        assert path_param.default == "."
        
        porcelain_param = next(p for p in params if p.name == "porcelain")
        assert porcelain_param.type == "bool"
        assert porcelain_param.required is False
        assert porcelain_param.default is False
    
    @pytest.mark.asyncio
    async def test_execute_success(self, tool, mock_subprocess_success):
        """Test successful git status execution."""
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_subprocess_success):
            
            result = await tool.execute(path="/test/repo")
            
            assert result.status == ToolResultStatus.SUCCESS
            assert "Git Repository Status" in result.content
            assert "main" in result.content
            assert result.metadata["branch"] == "main"
            assert "test.py" in result.metadata["staged_files"]
            assert "existing.py" in result.metadata["modified_files"]
            assert "untracked.py" in result.metadata["untracked_files"]
    
    @pytest.mark.asyncio
    async def test_execute_porcelain_format(self, tool, mock_subprocess_porcelain):
        """Test git status with porcelain format."""
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_subprocess_porcelain):
            
            result = await tool.execute(path="/test/repo", porcelain=True)
            
            assert result.status == ToolResultStatus.SUCCESS
            assert result.content == mock_subprocess_porcelain.stdout
            assert "test.py" in result.metadata["staged_files"]
            assert "existing.py" in result.metadata["modified_files"]
            assert "untracked.py" in result.metadata["untracked_files"]
    
    @pytest.mark.asyncio
    async def test_execute_not_git_repo(self, tool):
        """Test execution when not in git repository."""
        with patch.object(tool, '_is_git_repo', return_value=False):
            
            result = await tool.execute(path="/not/git/repo")
            
            assert result.status == ToolResultStatus.ERROR
            assert "Not a git repository" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_git_command_failure(self, tool):
        """Test handling of git command failure."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "fatal: not a git repository"
        
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_result):
            
            result = await tool.execute(path="/test/repo")
            
            assert result.status == ToolResultStatus.ERROR
            assert "Git status failed" in result.error
            assert "fatal: not a git repository" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_timeout(self, tool):
        """Test handling of command timeout."""
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', side_effect=subprocess.TimeoutExpired("git", 30)):
            
            result = await tool.execute(path="/test/repo")
            
            assert result.status == ToolResultStatus.ERROR
            assert "timed out" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_default_path(self, tool, mock_subprocess_success):
        """Test execution with default path."""
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_subprocess_success):
            
            result = await tool.execute()
            
            assert result.status == ToolResultStatus.SUCCESS
            # Should use current directory as default
    
    def test_is_git_repo_with_git_dir(self, tool, tmp_path):
        """Test git repo detection with .git directory."""
        (tmp_path / ".git").mkdir()
        
        result = tool._is_git_repo(str(tmp_path))
        
        assert result is True
    
    def test_is_git_repo_with_command(self, tool, tmp_path):
        """Test git repo detection with git command."""
        mock_result = Mock()
        mock_result.returncode = 0
        
        with patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_result):
            result = tool._is_git_repo(str(tmp_path))
            
            assert result is True
    
    def test_is_git_repo_not_git(self, tool, tmp_path):
        """Test git repo detection when not a git repo."""
        mock_result = Mock()
        mock_result.returncode = 1
        
        with patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_result):
            result = tool._is_git_repo(str(tmp_path))
            
            assert result is False
    
    def test_parse_status_porcelain(self, tool):
        """Test parsing porcelain status output."""
        output = """A  new_file.py
 M modified_file.py
D  deleted_file.py
?? untracked_file.py
R  renamed_file.py"""
        
        result = tool._parse_status(output, porcelain=True)
        
        assert "new_file.py" in result["staged"]
        assert "deleted_file.py" in result["staged"]
        assert "renamed_file.py" in result["staged"]
        assert "modified_file.py" in result["modified"]
        assert "untracked_file.py" in result["untracked"]
    
    def test_parse_status_regular(self, tool):
        """Test parsing regular status output."""
        output = """On branch feature-branch
Your branch is ahead of 'origin/main' by 2 commits.

Changes to be committed:
	new file:   new_file.py
	deleted:    old_file.py

Changes not staged for commit:
	modified:   existing_file.py"""
        
        result = tool._parse_status(output, porcelain=False)
        
        assert result["branch"] == "feature-branch"
        assert "new_file.py" in result["staged"]
        assert "old_file.py" in result["staged"]
        assert "existing_file.py" in result["modified"]
    
    def test_format_status_output(self, tool):
        """Test formatting of status output."""
        output = "On branch main\nnothing to commit"
        status_info = {
            "branch": "main",
            "modified": ["file1.py"],
            "staged": ["file2.py"],
            "untracked": ["file3.py"]
        }
        
        result = tool._format_status_output(output, status_info)
        
        assert "Git Repository Status" in result
        assert "**Branch**: `main`" in result
        assert "Staged: 1 files" in result
        assert "Modified: 1 files" in result
        assert "Untracked: 1 files" in result


class TestGitDiffTool:
    """Test GitDiffTool implementation."""
    
    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return GitDiffTool()
    
    @pytest.fixture
    def mock_subprocess_diff(self):
        """Mock git diff subprocess output."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """diff --git a/test.py b/test.py
index abc123..def456 100644
--- a/test.py
+++ b/test.py
@@ -1,3 +1,4 @@
 def hello():
+    print("Hello World!")
     pass"""
        mock_result.stderr = ""
        return mock_result
    
    def test_tool_parameters(self, tool):
        """Test tool parameter definitions."""
        params = tool.parameters
        
        assert len(params) == 4
        
        path_param = next(p for p in params if p.name == "path")
        assert path_param.required is False
        assert path_param.default == "."
        
        file_param = next(p for p in params if p.name == "file_path")
        assert file_param.required is False
        
        staged_param = next(p for p in params if p.name == "staged")
        assert staged_param.type == "bool"
        assert staged_param.default is False
        
        commit_param = next(p for p in params if p.name == "commit")
        assert commit_param.required is False
    
    @pytest.mark.asyncio
    async def test_execute_success(self, tool, mock_subprocess_diff):
        """Test successful git diff execution."""
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_subprocess_diff):
            
            result = await tool.execute(path="/test/repo")
            
            assert result.status == ToolResultStatus.SUCCESS
            assert "Git Diff" in result.content
            assert "def hello():" in result.content
            assert result.metadata["has_changes"] is True
    
    @pytest.mark.asyncio
    async def test_execute_staged_diff(self, tool, mock_subprocess_diff):
        """Test git diff with staged changes."""
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_subprocess_diff) as mock_run:
            
            result = await tool.execute(path="/test/repo", staged=True)
            
            assert result.status == ToolResultStatus.SUCCESS
            assert result.metadata["staged"] is True
            
            # Verify --staged flag was used
            called_cmd = mock_run.call_args[0][0]
            assert "--staged" in called_cmd
    
    @pytest.mark.asyncio
    async def test_execute_specific_file(self, tool, mock_subprocess_diff):
        """Test git diff for specific file."""
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_subprocess_diff) as mock_run:
            
            result = await tool.execute(path="/test/repo", file_path="test.py")
            
            assert result.status == ToolResultStatus.SUCCESS
            assert result.metadata["file_path"] == "test.py"
            
            # Verify file path was included in command
            called_cmd = mock_run.call_args[0][0]
            assert "test.py" in called_cmd
    
    @pytest.mark.asyncio
    async def test_execute_commit_comparison(self, tool, mock_subprocess_diff):
        """Test git diff against specific commit."""
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_subprocess_diff) as mock_run:
            
            result = await tool.execute(path="/test/repo", commit="abc123")
            
            assert result.status == ToolResultStatus.SUCCESS
            assert result.metadata["commit"] == "abc123"
            
            # Verify commit was included in command
            called_cmd = mock_run.call_args[0][0]
            assert "abc123" in called_cmd
    
    @pytest.mark.asyncio
    async def test_execute_no_changes(self, tool):
        """Test git diff when no changes present."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_result):
            
            result = await tool.execute(path="/test/repo")
            
            assert result.status == ToolResultStatus.SUCCESS
            assert "No changes found" in result.content
            assert result.metadata["has_changes"] is False
    
    @pytest.mark.asyncio
    async def test_execute_not_git_repo(self, tool):
        """Test execution when not in git repository."""
        with patch.object(tool, '_is_git_repo', return_value=False):
            
            result = await tool.execute(path="/not/git/repo")
            
            assert result.status == ToolResultStatus.ERROR
            assert "Not a git repository" in result.error


class TestGitAddTool:
    """Test GitAddTool implementation."""
    
    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return GitAddTool()
    
    @pytest.fixture
    def mock_subprocess_add_success(self):
        """Mock successful git add subprocess."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        return mock_result
    
    @pytest.fixture
    def mock_subprocess_status(self):
        """Mock git status after add."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """A  test1.py
M  test2.py
D  test3.py"""
        mock_result.stderr = ""
        return mock_result
    
    def test_tool_parameters(self, tool):
        """Test tool parameter definitions."""
        params = tool.parameters
        
        assert len(params) == 3
        
        files_param = next(p for p in params if p.name == "files")
        assert files_param.type == "list"
        assert files_param.required is True
        
        path_param = next(p for p in params if p.name == "path")
        assert path_param.required is False
        assert path_param.default == "."
        
        all_param = next(p for p in params if p.name == "all")
        assert all_param.type == "bool"
        assert all_param.default is False
    
    @pytest.mark.asyncio
    async def test_execute_single_file(self, tool, mock_subprocess_add_success, mock_subprocess_status):
        """Test adding single file."""
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', side_effect=[mock_subprocess_add_success, mock_subprocess_status]) as mock_run:
            
            result = await tool.execute(files="test.py", path="/test/repo")
            
            assert result.status == ToolResultStatus.SUCCESS
            assert "Files added to staging area" in result.content
            assert "test.py" in result.content
            assert result.metadata["files_added"] == ["test.py"]
            
            # Verify git add command
            first_call = mock_run.call_args_list[0][0][0]
            assert "git" in first_call
            assert "add" in first_call
            assert "test.py" in first_call
    
    @pytest.mark.asyncio
    async def test_execute_multiple_files(self, tool, mock_subprocess_add_success, mock_subprocess_status):
        """Test adding multiple files."""
        files = ["test1.py", "test2.py", "test3.py"]
        
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', side_effect=[mock_subprocess_add_success, mock_subprocess_status]) as mock_run:
            
            result = await tool.execute(files=files, path="/test/repo")
            
            assert result.status == ToolResultStatus.SUCCESS
            assert result.metadata["files_added"] == files
            
            # Verify all files were included in command
            first_call = mock_run.call_args_list[0][0][0]
            for file in files:
                assert file in first_call
    
    @pytest.mark.asyncio
    async def test_execute_add_all(self, tool, mock_subprocess_add_success, mock_subprocess_status):
        """Test adding all modified files."""
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', side_effect=[mock_subprocess_add_success, mock_subprocess_status]) as mock_run:
            
            result = await tool.execute(files=[], all=True, path="/test/repo")
            
            assert result.status == ToolResultStatus.SUCCESS
            assert "Added all modified files" in result.content
            assert result.metadata["files_added"] == "all"
            
            # Verify -A flag was used
            first_call = mock_run.call_args_list[0][0][0]
            assert "-A" in first_call
    
    @pytest.mark.asyncio
    async def test_execute_invalid_files_parameter(self, tool):
        """Test handling of invalid files parameter."""
        with patch.object(tool, '_is_git_repo', return_value=True):
            
            result = await tool.execute(files=123, path="/test/repo")  # Invalid type
            
            assert result.status == ToolResultStatus.ERROR
            assert "Files parameter must be string or list" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_git_add_failure(self, tool):
        """Test handling of git add command failure."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "pathspec 'nonexistent.py' did not match any files"
        
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_result):
            
            result = await tool.execute(files="nonexistent.py", path="/test/repo")
            
            assert result.status == ToolResultStatus.ERROR
            assert "Git add failed" in result.error
            assert "pathspec" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_staged_files_display(self, tool, mock_subprocess_add_success):
        """Test display of staged files after add."""
        # Mock status with many staged files
        mock_status = Mock()
        mock_status.returncode = 0
        mock_status.stdout = "\n".join([f"A  file{i}.py" for i in range(15)])  # 15 files
        mock_status.stderr = ""
        
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', side_effect=[mock_subprocess_add_success, mock_status]):
            
            result = await tool.execute(files="test.py", path="/test/repo")
            
            assert result.status == ToolResultStatus.SUCCESS
            assert "Currently staged files" in result.content
            assert "and 5 more files" in result.content  # Should show only first 10


class TestGitCommitTool:
    """Test GitCommitTool implementation."""
    
    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return GitCommitTool()
    
    @pytest.fixture
    def mock_subprocess_commit_success(self):
        """Mock successful git commit subprocess."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "[main abc1234] Add new feature\n 1 file changed, 5 insertions(+)"
        mock_result.stderr = ""
        return mock_result
    
    def test_tool_parameters(self, tool):
        """Test tool parameter definitions."""
        params = tool.parameters
        
        assert len(params) == 4
        
        message_param = next(p for p in params if p.name == "message")
        assert message_param.type == "str"
        assert message_param.required is True
        
        path_param = next(p for p in params if p.name == "path")
        assert path_param.required is False
        assert path_param.default == "."
        
        amend_param = next(p for p in params if p.name == "amend")
        assert amend_param.type == "bool"
        assert amend_param.default is False
        
        add_all_param = next(p for p in params if p.name == "add_all")
        assert add_all_param.type == "bool"
        assert add_all_param.default is False
    
    @pytest.mark.asyncio
    async def test_execute_success(self, tool, mock_subprocess_commit_success):
        """Test successful git commit."""
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_subprocess_commit_success):
            
            result = await tool.execute(message="Add new feature", path="/test/repo")
            
            assert result.status == ToolResultStatus.SUCCESS
            assert "Commit created successfully" in result.content
            assert "Add new feature" in result.content
            assert result.metadata["message"] == "Add new feature"
            assert result.metadata["commit_hash"] == "abc1234"
    
    @pytest.mark.asyncio
    async def test_execute_amend_commit(self, tool, mock_subprocess_commit_success):
        """Test amending last commit."""
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_subprocess_commit_success) as mock_run:
            
            result = await tool.execute(message="Updated message", amend=True, path="/test/repo")
            
            assert result.status == ToolResultStatus.SUCCESS
            assert result.metadata["amend"] is True
            
            # Verify --amend flag was used
            called_cmd = mock_run.call_args[0][0]
            assert "--amend" in called_cmd
    
    @pytest.mark.asyncio
    async def test_execute_add_all_commit(self, tool, mock_subprocess_commit_success):
        """Test commit with add all flag."""
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_subprocess_commit_success) as mock_run:
            
            result = await tool.execute(message="Quick commit", add_all=True, path="/test/repo")
            
            assert result.status == ToolResultStatus.SUCCESS
            assert result.metadata["add_all"] is True
            
            # Verify -a flag was used
            called_cmd = mock_run.call_args[0][0]
            assert "-a" in called_cmd
    
    @pytest.mark.asyncio
    async def test_execute_nothing_to_commit(self, tool):
        """Test handling when nothing to commit."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = "On branch main\nnothing to commit, working tree clean"
        mock_result.stderr = ""
        
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_result):
            
            result = await tool.execute(message="Empty commit", path="/test/repo")
            
            assert result.status == ToolResultStatus.WARNING
            assert "Nothing to commit" in result.content
    
    @pytest.mark.asyncio
    async def test_execute_commit_failure(self, tool):
        """Test handling of commit failure."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Author identity unknown"
        
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_result):
            
            result = await tool.execute(message="Test commit", path="/test/repo")
            
            assert result.status == ToolResultStatus.ERROR
            assert "Git commit failed" in result.error
            assert "Author identity unknown" in result.error
    
    @pytest.mark.asyncio
    async def test_commit_hash_extraction(self, tool):
        """Test extraction of commit hash from output."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "[feature-branch def5678] Implement new algorithm\n 2 files changed, 25 insertions(+), 3 deletions(-)"
        mock_result.stderr = ""
        
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_result):
            
            result = await tool.execute(message="Test commit", path="/test/repo")
            
            assert result.status == ToolResultStatus.SUCCESS
            assert result.metadata["commit_hash"] == "def5678"


class TestGitLogTool:
    """Test GitLogTool implementation."""
    
    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return GitLogTool()
    
    @pytest.fixture
    def mock_subprocess_log_success(self):
        """Mock successful git log subprocess."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """abc1234 Add new feature
def5678 Fix bug in parser
ghi9012 Update documentation
jkl3456 Initial commit"""
        mock_result.stderr = ""
        return mock_result
    
    def test_tool_parameters(self, tool):
        """Test tool parameter definitions."""
        params = tool.parameters
        
        assert len(params) == 4
        
        path_param = next(p for p in params if p.name == "path")
        assert path_param.required is False
        assert path_param.default == "."
        
        limit_param = next(p for p in params if p.name == "limit")
        assert limit_param.type == "int"
        assert limit_param.default == 10
        
        oneline_param = next(p for p in params if p.name == "oneline")
        assert oneline_param.type == "bool"
        assert oneline_param.default is True
        
        file_param = next(p for p in params if p.name == "file_path")
        assert file_param.required is False
    
    @pytest.mark.asyncio
    async def test_execute_success(self, tool, mock_subprocess_log_success):
        """Test successful git log execution."""
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_subprocess_log_success):
            
            result = await tool.execute(path="/test/repo")
            
            assert result.status == ToolResultStatus.SUCCESS
            assert "Git Commit History" in result.content
            assert "Add new feature" in result.content
            assert result.metadata["commit_count"] == 4
            assert result.metadata["oneline"] is True
    
    @pytest.mark.asyncio
    async def test_execute_with_limit(self, tool, mock_subprocess_log_success):
        """Test git log with custom limit."""
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_subprocess_log_success) as mock_run:
            
            result = await tool.execute(path="/test/repo", limit=5)
            
            assert result.status == ToolResultStatus.SUCCESS
            assert result.metadata["limit"] == 5
            
            # Verify --max-count parameter
            called_cmd = mock_run.call_args[0][0]
            assert "--max-count=5" in called_cmd
    
    @pytest.mark.asyncio
    async def test_execute_full_format(self, tool, mock_subprocess_log_success):
        """Test git log without oneline format."""
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_subprocess_log_success) as mock_run:
            
            result = await tool.execute(path="/test/repo", oneline=False)
            
            assert result.status == ToolResultStatus.SUCCESS
            assert result.metadata["oneline"] is False
            
            # Verify --oneline is not in command
            called_cmd = mock_run.call_args[0][0]
            assert "--oneline" not in called_cmd
    
    @pytest.mark.asyncio
    async def test_execute_specific_file(self, tool, mock_subprocess_log_success):
        """Test git log for specific file."""
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_subprocess_log_success) as mock_run:
            
            result = await tool.execute(path="/test/repo", file_path="test.py")
            
            assert result.status == ToolResultStatus.SUCCESS
            assert "for `test.py`" in result.content
            assert result.metadata["file_path"] == "test.py"
            
            # Verify file path in command
            called_cmd = mock_run.call_args[0][0]
            assert "--" in called_cmd
            assert "test.py" in called_cmd
    
    @pytest.mark.asyncio
    async def test_execute_no_commits(self, tool):
        """Test git log when no commits found."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_result):
            
            result = await tool.execute(path="/test/repo")
            
            assert result.status == ToolResultStatus.SUCCESS
            assert "No commits found" in result.content
            assert result.metadata["commit_count"] == 0
    
    @pytest.mark.asyncio
    async def test_execute_git_log_failure(self, tool):
        """Test handling of git log command failure."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "fatal: your current branch 'main' does not have any commits yet"
        
        with patch.object(tool, '_is_git_repo', return_value=True), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_result):
            
            result = await tool.execute(path="/test/repo")
            
            assert result.status == ToolResultStatus.ERROR
            assert "Git log failed" in result.error
            assert "does not have any commits" in result.error


class TestGitToolsIntegration:
    """Test integration between git tools."""
    
    @pytest.fixture
    def status_tool(self):
        return GitStatusTool()
    
    @pytest.fixture
    def diff_tool(self):
        return GitDiffTool()
    
    @pytest.fixture
    def add_tool(self):
        return GitAddTool()
    
    @pytest.fixture
    def commit_tool(self):
        return GitCommitTool()
    
    @pytest.fixture
    def log_tool(self):
        return GitLogTool()
    
    @pytest.fixture
    def mock_git_repo(self, tmp_path):
        """Create mock git repository structure."""
        (tmp_path / ".git").mkdir()
        (tmp_path / "file1.py").write_text("print('hello')")
        (tmp_path / "file2.py").write_text("print('world')")
        return tmp_path
    
    @pytest.mark.asyncio
    async def test_typical_git_workflow(self, status_tool, add_tool, commit_tool, mock_git_repo):
        """Test a typical git workflow: status -> add -> commit."""
        # Mock all subprocess calls for a typical workflow
        mock_status = Mock()
        mock_status.returncode = 0
        mock_status.stdout = " M file1.py\n?? file2.py"
        
        mock_add = Mock()
        mock_add.returncode = 0
        mock_add.stdout = ""
        
        mock_status_after_add = Mock()
        mock_status_after_add.returncode = 0
        mock_status_after_add.stdout = "A  file1.py\nA  file2.py"
        
        mock_commit = Mock()
        mock_commit.returncode = 0
        mock_commit.stdout = "[main abc1234] Add files\n 2 files changed, 2 insertions(+)"
        
        with patch('flowlib.agent.runners.repl.tools.git.subprocess.run', side_effect=[
            mock_status, mock_add, mock_status_after_add, mock_commit
        ]):
            
            # 1. Check status
            status_result = await status_tool.execute(path=str(mock_git_repo), porcelain=True)
            assert status_result.status == ToolResultStatus.SUCCESS
            assert "file1.py" in status_result.metadata["modified_files"]
            assert "file2.py" in status_result.metadata["untracked_files"]
            
            # 2. Add files
            add_result = await add_tool.execute(files=["file1.py", "file2.py"], path=str(mock_git_repo))
            assert add_result.status == ToolResultStatus.SUCCESS
            
            # 3. Commit
            commit_result = await commit_tool.execute(message="Add files", path=str(mock_git_repo))
            assert commit_result.status == ToolResultStatus.SUCCESS
            assert commit_result.metadata["commit_hash"] == "abc1234"
    
    @pytest.mark.asyncio
    async def test_error_handling_consistency(self, status_tool, diff_tool, add_tool, commit_tool, log_tool):
        """Test consistent error handling across all git tools."""
        nonexistent_path = "/this/path/does/not/exist"
        
        # All tools should handle non-git repositories consistently
        with patch('flowlib.agent.runners.repl.tools.git.Path.exists', return_value=False), \
             patch('flowlib.agent.runners.repl.tools.git.subprocess.run') as mock_run:
            
            mock_run.return_value.returncode = 1  # git rev-parse fails
            
            status_result = await status_tool.execute(path=nonexistent_path)
            diff_result = await diff_tool.execute(path=nonexistent_path)
            add_result = await add_tool.execute(files=["test.py"], path=nonexistent_path)
            commit_result = await commit_tool.execute(message="test", path=nonexistent_path)
            log_result = await log_tool.execute(path=nonexistent_path)
            
            assert status_result.status == ToolResultStatus.ERROR
            assert diff_result.status == ToolResultStatus.ERROR
            assert add_result.status == ToolResultStatus.ERROR
            assert commit_result.status == ToolResultStatus.ERROR
            assert log_result.status == ToolResultStatus.ERROR
            
            for result in [status_result, diff_result, add_result, commit_result, log_result]:
                assert "Not a git repository" in result.error
    
    @pytest.mark.asyncio
    async def test_timeout_handling_consistency(self, status_tool, diff_tool, add_tool, commit_tool, log_tool):
        """Test consistent timeout handling across all git tools."""
        with patch('flowlib.agent.runners.repl.tools.git.subprocess.run', side_effect=subprocess.TimeoutExpired("git", 30)):
            
            # Mock git repo check to pass
            with patch.object(status_tool, '_is_git_repo', return_value=True), \
                 patch.object(diff_tool, '_is_git_repo', return_value=True), \
                 patch.object(add_tool, '_is_git_repo', return_value=True), \
                 patch.object(commit_tool, '_is_git_repo', return_value=True), \
                 patch.object(log_tool, '_is_git_repo', return_value=True):
                
                status_result = await status_tool.execute()
                diff_result = await diff_tool.execute()
                add_result = await add_tool.execute(files=["test.py"])
                commit_result = await commit_tool.execute(message="test")
                log_result = await log_tool.execute()
                
                for result in [status_result, diff_result, add_result, commit_result, log_result]:
                    assert result.status == ToolResultStatus.ERROR
                    assert "timed out" in result.error
    
    def test_git_repo_detection_consistency(self, status_tool, diff_tool, add_tool, commit_tool, log_tool, tmp_path):
        """Test consistent git repository detection across all tools."""
        # Test with .git directory
        (tmp_path / ".git").mkdir()
        
        assert status_tool._is_git_repo(str(tmp_path)) is True
        assert diff_tool._is_git_repo(str(tmp_path)) is True
        assert add_tool._is_git_repo(str(tmp_path)) is True
        assert commit_tool._is_git_repo(str(tmp_path)) is True
        assert log_tool._is_git_repo(str(tmp_path)) is True
    
    @pytest.mark.asyncio
    async def test_metadata_consistency(self, status_tool, diff_tool, add_tool, commit_tool, log_tool):
        """Test consistent metadata structure across git tools."""
        # Mock successful operations
        mock_success = Mock()
        mock_success.returncode = 0
        mock_success.stdout = "test output"
        mock_success.stderr = ""
        
        with patch('flowlib.agent.runners.repl.tools.git.subprocess.run', return_value=mock_success):
            
            with patch.object(status_tool, '_is_git_repo', return_value=True), \
                 patch.object(diff_tool, '_is_git_repo', return_value=True), \
                 patch.object(add_tool, '_is_git_repo', return_value=True), \
                 patch.object(commit_tool, '_is_git_repo', return_value=True), \
                 patch.object(log_tool, '_is_git_repo', return_value=True):
                
                status_result = await status_tool.execute(path="/test/repo")
                diff_result = await diff_tool.execute(path="/test/repo")
                add_result = await add_tool.execute(files=["test.py"], path="/test/repo")
                commit_result = await commit_tool.execute(message="test", path="/test/repo")
                log_result = await log_tool.execute(path="/test/repo")
                
                # All should have path in metadata
                for result in [status_result, diff_result, add_result, commit_result, log_result]:
                    assert result.status == ToolResultStatus.SUCCESS
                    assert "path" in result.metadata
                    assert result.metadata["path"].endswith("/test/repo")