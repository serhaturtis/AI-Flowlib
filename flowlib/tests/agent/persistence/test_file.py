"""Tests for file-based persistence implementation."""

import pytest
import os
import json
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Optional
from unittest.mock import patch, mock_open

from flowlib.agent.components.persistence.file import (
    FileStatePersister,
    FileStatePersisterSettings,
    DateTimeEncoder
)
from flowlib.agent.models.state import AgentState
from flowlib.agent.core.errors import StatePersistenceError


class TestDateTimeEncoder:
    """Test custom JSON encoder for datetime objects."""
    
    def test_encode_datetime(self):
        """Test encoding datetime objects."""
        encoder = DateTimeEncoder()
        now = datetime.now()
        
        result = encoder.default(now)
        
        assert result == now.isoformat()
        assert isinstance(result, str)
    
    def test_encode_non_datetime(self):
        """Test encoding non-datetime objects falls back to default."""
        encoder = DateTimeEncoder()
        
        # Should raise TypeError for unsupported types
        with pytest.raises(TypeError):
            encoder.default(object())
    
    def test_json_dumps_with_datetime(self):
        """Test using encoder with json.dumps()."""
        data = {
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "name": "test",
            "value": 42
        }
        
        json_str = json.dumps(data, cls=DateTimeEncoder)
        
        assert "2024-01-01T12:00:00" in json_str
        assert "test" in json_str
        assert "42" in json_str


class TestFileStatePersisterSettings:
    """Test file persister settings."""
    
    def test_default_directory(self):
        """Test default directory setting."""
        settings = FileStatePersisterSettings()
        
        assert settings.directory == "./states"
    
    def test_custom_directory(self):
        """Test custom directory setting."""
        settings = FileStatePersisterSettings(directory="/custom/path")
        
        assert settings.directory == "/custom/path"


class TestFileStatePersister:
    """Test file-based state persister."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def settings(self, temp_dir):
        """Create settings with temporary directory."""
        return FileStatePersisterSettings(directory=temp_dir)
    
    @pytest.fixture
    def persister(self, settings):
        """Create file persister instance."""
        return FileStatePersister(settings=settings)
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample agent state."""
        return AgentState(
            task_id="file_test_task",
            task_description="File persister test task"
        )
    
    def test_persister_initialization_valid(self, settings):
        """Test persister initialization with valid settings."""
        persister = FileStatePersister(settings)
        
        assert persister.name == "file_state_persister"
        assert persister.settings == settings
        assert persister.settings.directory == settings.directory
    
    def test_persister_initialization_no_settings(self):
        """Test persister initialization without settings raises error."""
        with pytest.raises(ValueError, match="requires a 'settings' argument"):
            FileStatePersister(None)
    
    def test_persister_initialization_no_directory(self):
        """Test persister initialization without directory setting."""
        # Create settings without directory
        invalid_settings = type('Settings', (), {})()
        
        with pytest.raises(ValueError, match="requires a 'settings' argument"):
            FileStatePersister(invalid_settings)
    
    @pytest.mark.asyncio
    async def test_initialize_creates_directory(self, temp_dir):
        """Test that initialization creates the directory."""
        # Use a subdirectory that doesn't exist yet
        subdir = os.path.join(temp_dir, "new_states")
        settings = FileStatePersisterSettings(directory=subdir)
        persister = FileStatePersister(settings)
        
        # Directory shouldn't exist yet
        assert not os.path.exists(subdir)
        
        # Initialize should create it
        await persister._initialize()
        
        assert os.path.exists(subdir)
        assert os.path.isdir(subdir)
    
    @pytest.mark.asyncio
    async def test_initialize_existing_directory(self, persister, temp_dir):
        """Test initialization with existing directory."""
        # Directory already exists from fixture
        assert os.path.exists(temp_dir)
        
        # Should not raise error
        await persister._initialize()
        
        assert os.path.exists(temp_dir)
    
    @pytest.mark.asyncio
    async def test_save_state_success(self, persister, sample_state, temp_dir):
        """Test successful state saving."""
        await persister._initialize()
        metadata = {"source": "test", "version": "1.0"}
        
        result = await persister._save_state_impl(sample_state, metadata)
        
        assert result is True
        
        # Check state file exists
        state_file = os.path.join(temp_dir, f"{sample_state.task_id}.json")
        assert os.path.exists(state_file)
        
        # Check metadata file exists
        meta_file = os.path.join(temp_dir, f"{sample_state.task_id}.meta.json")
        assert os.path.exists(meta_file)
        
        # Verify state file content
        with open(state_file, 'r') as f:
            saved_data = json.load(f)
        assert saved_data["task_id"] == sample_state.task_id
        assert saved_data["task_description"] == sample_state.task_description
        
        # Verify metadata file content
        with open(meta_file, 'r') as f:
            saved_metadata = json.load(f)
        assert saved_metadata == metadata
    
    @pytest.mark.asyncio
    async def test_save_state_without_metadata(self, persister, sample_state, temp_dir):
        """Test saving state without metadata."""
        await persister._initialize()
        
        result = await persister._save_state_impl(sample_state, None)
        
        assert result is True
        
        # Check state file exists
        state_file = os.path.join(temp_dir, f"{sample_state.task_id}.json")
        assert os.path.exists(state_file)
        
        # Check metadata file does NOT exist
        meta_file = os.path.join(temp_dir, f"{sample_state.task_id}.meta.json")
        assert not os.path.exists(meta_file)
    
    @pytest.mark.asyncio
    async def test_save_state_with_datetime(self, persister, temp_dir):
        """Test saving state with datetime objects."""
        await persister._initialize()
        
        # Create state and manually set datetime
        state = AgentState(task_id="datetime_test", task_description="Test with datetime")
        state.add_system_message("Test message")  # This should add timestamp
        
        result = await persister._save_state_impl(state)
        
        assert result is True
        
        # Verify file can be read back and contains ISO datetime
        state_file = os.path.join(temp_dir, "datetime_test.json")
        with open(state_file, 'r') as f:
            content = f.read()
        
        # Should contain ISO datetime format
        assert "T" in content  # ISO datetime contains 'T'
    
    @pytest.mark.asyncio
    async def test_save_state_permission_error(self, persister, sample_state):
        """Test save state with file permission error."""
        await persister._initialize()
        
        # Mock open to raise PermissionError
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(StatePersistenceError) as exc_info:
                await persister._save_state_impl(sample_state)
            
            assert exc_info.value.context["operation"] == "save"
            assert exc_info.value.context["task_id"] == sample_state.task_id
            assert "Permission denied" in str(exc_info.value.cause)
    
    @pytest.mark.asyncio
    async def test_load_state_success(self, persister, sample_state, temp_dir):
        """Test successful state loading."""
        await persister._initialize()
        
        # First save a state
        await persister._save_state_impl(sample_state, {"test": "metadata"})
        
        # Then load it
        loaded_state = await persister._load_state_impl(sample_state.task_id)
        
        assert loaded_state is not None
        assert loaded_state.task_id == sample_state.task_id
        assert loaded_state.task_description == sample_state.task_description
    
    @pytest.mark.asyncio
    async def test_load_state_not_found(self, persister, temp_dir):
        """Test loading non-existent state."""
        await persister._initialize()
        
        loaded_state = await persister._load_state_impl("nonexistent_task")
        
        assert loaded_state is None
    
    @pytest.mark.asyncio
    async def test_load_state_invalid_json(self, persister, temp_dir):
        """Test loading state with invalid JSON."""
        await persister._initialize()
        
        # Create file with invalid JSON
        invalid_file = os.path.join(temp_dir, "invalid.json")
        with open(invalid_file, 'w') as f:
            f.write("{ invalid json")
        
        with pytest.raises(StatePersistenceError) as exc_info:
            await persister._load_state_impl("invalid")
        
        assert exc_info.value.context["operation"] == "load"
        assert exc_info.value.context["task_id"] == "invalid"
        assert isinstance(exc_info.value.cause, json.JSONDecodeError)
    
    @pytest.mark.asyncio
    async def test_load_state_file_read_error(self, persister, temp_dir):
        """Test loading state with file read error."""
        await persister._initialize()
        
        # Create file then mock open to raise error
        state_file = os.path.join(temp_dir, "error_task.json")
        with open(state_file, 'w') as f:
            f.write("{}")
        
        with patch("builtins.open", side_effect=IOError("Read error")):
            with pytest.raises(StatePersistenceError) as exc_info:
                await persister._load_state_impl("error_task")
            
            assert exc_info.value.context["operation"] == "load"
            assert "Read error" in str(exc_info.value.cause)
    
    @pytest.mark.asyncio
    async def test_delete_state_success(self, persister, sample_state, temp_dir):
        """Test successful state deletion."""
        await persister._initialize()
        
        # First save a state with metadata
        await persister._save_state_impl(sample_state, {"test": "metadata"})
        
        # Verify files exist
        state_file = os.path.join(temp_dir, f"{sample_state.task_id}.json")
        meta_file = os.path.join(temp_dir, f"{sample_state.task_id}.meta.json")
        assert os.path.exists(state_file)
        assert os.path.exists(meta_file)
        
        # Delete the state
        result = await persister._delete_state_impl(sample_state.task_id)
        
        assert result is True
        assert not os.path.exists(state_file)
        assert not os.path.exists(meta_file)
    
    @pytest.mark.asyncio
    async def test_delete_state_only_state_file(self, persister, sample_state, temp_dir):
        """Test deleting state when only state file exists (no metadata)."""
        await persister._initialize()
        
        # Save state without metadata
        await persister._save_state_impl(sample_state, None)
        
        # Verify only state file exists
        state_file = os.path.join(temp_dir, f"{sample_state.task_id}.json")
        meta_file = os.path.join(temp_dir, f"{sample_state.task_id}.meta.json")
        assert os.path.exists(state_file)
        assert not os.path.exists(meta_file)
        
        # Delete should succeed
        result = await persister._delete_state_impl(sample_state.task_id)
        
        assert result is True
        assert not os.path.exists(state_file)
    
    @pytest.mark.asyncio
    async def test_delete_state_nonexistent(self, persister, temp_dir):
        """Test deleting non-existent state."""
        await persister._initialize()
        
        # Should return True even if file doesn't exist
        result = await persister._delete_state_impl("nonexistent_task")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_state_permission_error(self, persister, sample_state, temp_dir):
        """Test delete state with permission error."""
        await persister._initialize()
        
        # Save a state first
        await persister._save_state_impl(sample_state)
        
        # Mock os.remove to raise PermissionError
        with patch("os.remove", side_effect=PermissionError("Permission denied")):
            with pytest.raises(StatePersistenceError) as exc_info:
                await persister._delete_state_impl(sample_state.task_id)
            
            assert exc_info.value.context["operation"] == "delete"
            assert "Permission denied" in str(exc_info.value.cause)
    
    @pytest.mark.asyncio
    async def test_list_states_empty(self, persister, temp_dir):
        """Test listing states when directory is empty."""
        await persister._initialize()
        
        states = await persister._list_states_impl()
        
        assert states == []
    
    @pytest.mark.asyncio
    async def test_list_states_multiple(self, persister, temp_dir):
        """Test listing multiple states."""
        await persister._initialize()
        
        # Create multiple states
        states = [
            AgentState(task_id="task_1", task_description="Task 1"),
            AgentState(task_id="task_2", task_description="Task 2"),
            AgentState(task_id="task_3", task_description="Task 3")
        ]
        
        # Save states with different metadata
        for i, state in enumerate(states):
            metadata = {"index": str(i), "type": "test"}
            await persister._save_state_impl(state, metadata)
        
        # List all states
        listed_states = await persister._list_states_impl()
        
        assert len(listed_states) == 3
        
        # Check each state metadata
        for state_meta in listed_states:
            assert "task_id" in state_meta
            assert "index" in state_meta
            assert "type" in state_meta
            assert state_meta["type"] == "test"
    
    @pytest.mark.asyncio
    async def test_list_states_with_filter(self, persister, temp_dir):
        """Test listing states with filter criteria."""
        await persister._initialize()
        
        # Create states with different metadata
        await persister._save_state_impl(
            AgentState(task_id="prod_task", task_description="Production task"),
            {"environment": "production", "status": "active"}
        )
        await persister._save_state_impl(
            AgentState(task_id="test_task", task_description="Test task"),
            {"environment": "test", "status": "active"}
        )
        await persister._save_state_impl(
            AgentState(task_id="dev_task", task_description="Development task"),
            {"environment": "development", "status": "inactive"}
        )
        
        # Filter by environment
        prod_states = await persister._list_states_impl({"environment": "production"})
        assert len(prod_states) == 1
        assert prod_states[0]["task_id"] == "prod_task"
        
        # Filter by status
        active_states = await persister._list_states_impl({"status": "active"})
        assert len(active_states) == 2
        task_ids = [s["task_id"] for s in active_states]
        assert "prod_task" in task_ids
        assert "test_task" in task_ids
        
        # Filter with no matches
        no_matches = await persister._list_states_impl({"environment": "staging"})
        assert len(no_matches) == 0
    
    @pytest.mark.asyncio
    async def test_list_states_without_metadata_files(self, persister, temp_dir):
        """Test listing states when metadata files don't exist."""
        await persister._initialize()
        
        # Save state without metadata
        state = AgentState(task_id="no_meta", task_description="No metadata")
        await persister._save_state_impl(state, None)
        
        # List states
        listed_states = await persister._list_states_impl()
        
        assert len(listed_states) == 1
        assert listed_states[0]["task_id"] == "no_meta"
        # Should only have task_id, no additional metadata
        assert list(listed_states[0].keys()) == ["task_id"]
    
    @pytest.mark.asyncio
    async def test_list_states_ignores_meta_files(self, persister, temp_dir):
        """Test that listing ignores .meta.json files in file list."""
        await persister._initialize()
        
        # Save state with metadata
        state = AgentState(task_id="with_meta", task_description="With metadata")
        await persister._save_state_impl(state, {"test": "data"})
        
        # Manually create an orphaned .meta.json file
        orphan_meta = os.path.join(temp_dir, "orphan.meta.json")
        with open(orphan_meta, 'w') as f:
            json.dump({"orphan": "metadata"}, f)
        
        # List states - should only find the real state, not the orphan meta
        listed_states = await persister._list_states_impl()
        
        assert len(listed_states) == 1
        assert listed_states[0]["task_id"] == "with_meta"
    
    @pytest.mark.asyncio
    async def test_list_states_directory_error(self, persister):
        """Test list states with directory access error."""
        await persister._initialize()
        
        # Mock os.listdir to raise error
        with patch("os.listdir", side_effect=OSError("Directory error")):
            with pytest.raises(StatePersistenceError) as exc_info:
                await persister._list_states_impl()
            
            assert exc_info.value.context["operation"] == "list"
            assert "Directory error" in str(exc_info.value.cause)
    
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, persister, temp_dir):
        """Test complete save -> load -> list -> delete lifecycle."""
        await persister._initialize()
        
        # Create and save state
        state = AgentState(task_id="lifecycle_test", task_description="Lifecycle test")
        metadata = {"test": "lifecycle", "created": "2024-01-01"}
        
        # Save
        save_result = await persister._save_state_impl(state, metadata)
        assert save_result is True
        
        # List and verify
        listed_states = await persister._list_states_impl()
        assert len(listed_states) == 1
        assert listed_states[0]["task_id"] == "lifecycle_test"
        assert listed_states[0]["test"] == "lifecycle"
        
        # Load and verify
        loaded_state = await persister._load_state_impl("lifecycle_test")
        assert loaded_state is not None
        assert loaded_state.task_id == "lifecycle_test"
        assert loaded_state.task_description == "Lifecycle test"
        
        # Delete
        delete_result = await persister._delete_state_impl("lifecycle_test")
        assert delete_result is True
        
        # Verify gone
        final_list = await persister._list_states_impl()
        assert len(final_list) == 0
        
        final_load = await persister._load_state_impl("lifecycle_test")
        assert final_load is None
    
    @pytest.mark.asyncio
    async def test_concurrent_file_operations(self, persister, temp_dir):
        """Test concurrent file operations."""
        import asyncio
        
        await persister._initialize()
        
        # Create multiple states for concurrent operations
        states = [
            AgentState(task_id=f"concurrent_{i}", task_description=f"Concurrent task {i}")
            for i in range(5)
        ]
        
        # Save all states concurrently
        save_tasks = [
            persister._save_state_impl(state, {"index": str(i)})
            for i, state in enumerate(states)
        ]
        save_results = await asyncio.gather(*save_tasks)
        assert all(save_results)
        
        # Load all states concurrently
        load_tasks = [persister._load_state_impl(state.task_id) for state in states]
        loaded_states = await asyncio.gather(*load_tasks)
        assert all(loaded is not None for loaded in loaded_states)
        
        # List states
        all_states = await persister._list_states_impl()
        assert len(all_states) == 5
        
        # Delete all states concurrently
        delete_tasks = [persister._delete_state_impl(state.task_id) for state in states]
        delete_results = await asyncio.gather(*delete_tasks)
        assert all(delete_results)
        
        # Verify all gone
        final_states = await persister._list_states_impl()
        assert len(final_states) == 0


class TestFileStatePersisterEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def settings(self, temp_dir):
        """Create settings with temporary directory."""
        return FileStatePersisterSettings(directory=temp_dir)
    
    @pytest.fixture
    def persister(self, settings):
        """Create file persister instance."""
        return FileStatePersister(settings=settings)
    
    @pytest.mark.asyncio
    async def test_save_empty_task_id(self, persister, temp_dir):
        """Test saving state with empty task_id."""
        await persister._initialize()
        
        state = AgentState(task_id="", task_description="Empty task ID")
        
        # AgentState automatically generates a UUID when empty task_id is provided
        # So the file will be named with the generated UUID
        assert state.task_id != ""  # Should have a generated UUID
        
        result = await persister._save_state_impl(state)
        assert result is True
        
        # Check that one file was created with the generated task_id
        files = os.listdir(temp_dir)
        assert len(files) == 1
        assert files[0] == f"{state.task_id}.json"
    
    @pytest.mark.asyncio
    async def test_load_empty_task_id(self, persister):
        """Test loading state with empty task_id."""
        await persister._initialize()
        
        # Should return None (file .json won't exist)
        result = await persister._load_state_impl("")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_save_special_characters_in_task_id(self, persister, temp_dir):
        """Test saving state with special characters in task_id."""
        await persister._initialize()
        
        # Some filesystems may not support certain characters
        state = AgentState(task_id="task/with:special*chars", task_description="Special chars")
        
        # Depending on filesystem, this might fail
        try:
            result = await persister._save_state_impl(state)
            # If it succeeds, verify file exists
            if result:
                # The actual filename depends on OS handling of special chars
                files = os.listdir(temp_dir)
                assert len(files) >= 1  # At least one file should exist
        except StatePersistenceError:
            # Expected on some filesystems
            pass
    
    @pytest.mark.asyncio
    async def test_large_state_serialization(self, persister, temp_dir):
        """Test saving large state data."""
        await persister._initialize()
        
        # Create state with large data
        state = AgentState(task_id="large_state", task_description="Large state test")
        
        # Add many system messages to make it large
        for i in range(1000):
            state.add_system_message(f"Message {i}: " + "x" * 100)
        
        result = await persister._save_state_impl(state)
        assert result is True
        
        # Verify file exists and has reasonable size
        state_file = os.path.join(temp_dir, "large_state.json")
        assert os.path.exists(state_file)
        assert os.path.getsize(state_file) > 1000  # Should be reasonably large
        
        # Verify can be loaded back
        loaded_state = await persister._load_state_impl("large_state")
        assert loaded_state is not None
        assert len(loaded_state.system_messages) == 1000
    
    @pytest.mark.asyncio
    async def test_corrupted_metadata_file(self, persister, temp_dir):
        """Test listing states with corrupted metadata file."""
        await persister._initialize()
        
        # Save normal state
        state = AgentState(task_id="corrupted_meta", task_description="Test")
        await persister._save_state_impl(state, {"good": "metadata"})
        
        # Corrupt the metadata file
        meta_file = os.path.join(temp_dir, "corrupted_meta.meta.json")
        with open(meta_file, 'w') as f:
            f.write("{ corrupted json")
        
        # Listing should handle corrupted metadata gracefully
        with pytest.raises(StatePersistenceError):
            await persister._list_states_impl()
    
    @pytest.mark.asyncio
    async def test_unicode_in_state_data(self, persister, temp_dir):
        """Test saving and loading state with unicode characters."""
        await persister._initialize()
        
        # Create state with unicode characters
        state = AgentState(
            task_id="unicode_test", 
            task_description="Test with unicode: 你好世界 🌍 émojis"
        )
        state.add_user_message("Unicode message: 中文测试 🚀")
        
        metadata = {"unicode_field": "测试数据", "emoji": "🎉"}
        
        # Save with unicode data
        result = await persister._save_state_impl(state, metadata)
        assert result is True
        
        # Load and verify unicode is preserved
        loaded_state = await persister._load_state_impl("unicode_test")
        assert loaded_state is not None
        assert "你好世界" in loaded_state.task_description
        assert "🌍" in loaded_state.task_description
        assert len(loaded_state.user_messages) > 0
        assert "中文测试" in loaded_state.user_messages[0]
        
        # Verify metadata unicode
        listed_states = await persister._list_states_impl()
        unicode_meta = next(s for s in listed_states if s["task_id"] == "unicode_test")
        assert unicode_meta["unicode_field"] == "测试数据"
        assert unicode_meta["emoji"] == "🎉"