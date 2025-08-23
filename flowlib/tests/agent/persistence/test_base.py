"""Tests for base persistence implementation."""

import pytest
from datetime import datetime
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

from flowlib.agent.components.persistence.base import BaseStatePersister, SettingsType
from flowlib.agent.models.state import AgentState
from flowlib.agent.core.errors import StatePersistenceError
from flowlib.providers.core.base import ProviderSettings


class MockStatePersisterSettings(ProviderSettings):
    """Mock settings class for BaseStatePersister testing."""
    test_setting: str = "default_value"
    numeric_setting: int = 42


class ConcreteStatePersister(BaseStatePersister[MockStatePersisterSettings]):
    """Concrete implementation of BaseStatePersister for testing."""
    
    def __init__(self, name: str = "test_persister", settings: Optional[MockStatePersisterSettings] = None):
        if settings is None:
            settings = MockStatePersisterSettings()
        super().__init__(name=name, settings=settings)
        # Initialize test tracking attributes using object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, 'save_calls', [])
        object.__setattr__(self, 'load_calls', [])
        object.__setattr__(self, 'delete_calls', [])
        object.__setattr__(self, 'list_calls', [])
        object.__setattr__(self, 'should_fail', False)
        object.__setattr__(self, 'fail_operation', None)
        object.__setattr__(self, 'stored_states', {})
    
    async def _save_state_impl(
        self,
        state: AgentState,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        self.save_calls.append((state, metadata))
        if self.should_fail and self.fail_operation == "save":
            raise RuntimeError("Mock implementation save failure")
        
        self.stored_states[state.task_id] = state
        return True
    
    async def _load_state_impl(self, task_id: str) -> Optional[AgentState]:
        self.load_calls.append(task_id)
        if self.should_fail and self.fail_operation == "load":
            raise RuntimeError("Mock implementation load failure")
        
        return self.stored_states.get(task_id)
    
    async def _delete_state_impl(self, task_id: str) -> bool:
        self.delete_calls.append(task_id)
        if self.should_fail and self.fail_operation == "delete":
            raise RuntimeError("Mock implementation delete failure")
        
        if task_id in self.stored_states:
            del self.stored_states[task_id]
            return True
        return False
    
    async def _list_states_impl(
        self,
        filter_criteria: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        self.list_calls.append(filter_criteria)
        if self.should_fail and self.fail_operation == "list":
            raise RuntimeError("Mock implementation list failure")
        
        return [
            {"task_id": task_id, "task_description": state.task_description}
            for task_id, state in self.stored_states.items()
        ]


class TestBaseStatePersister:
    """Test BaseStatePersister functionality."""
    
    @pytest.fixture
    def persister_settings(self):
        """Create test persister settings."""
        return MockStatePersisterSettings(
            test_setting="custom_value",
            numeric_setting=100
        )
    
    @pytest.fixture
    def persister(self, persister_settings):
        """Create a concrete persister instance."""
        return ConcreteStatePersister(
            name="test_persister",
            settings=persister_settings
        )
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample agent state."""
        return AgentState(
            task_id="test_task_base",
            task_description="Base persister test task"
        )
    
    def test_persister_initialization(self, persister, persister_settings):
        """Test persister initialization."""
        assert persister.name == "test_persister"
        assert persister.settings == persister_settings
        assert persister.settings.test_setting == "custom_value"
        assert persister.settings.numeric_setting == 100
    
    def test_persister_default_settings(self):
        """Test persister with default settings."""
        persister = ConcreteStatePersister()
        
        assert persister.name == "test_persister"
        assert isinstance(persister.settings, MockStatePersisterSettings)
        assert persister.settings.test_setting == "default_value"
        assert persister.settings.numeric_setting == 42
    
    def test_persister_inheritance(self, persister):
        """Test that persister properly inherits from Provider and Generic."""
        from flowlib.providers.core.base import Provider
        from typing import Generic
        
        assert isinstance(persister, Provider)
        assert hasattr(persister, "name")
        assert hasattr(persister, "settings")
        assert hasattr(persister, "initialized")
    
    @pytest.mark.asyncio
    async def test_save_state_success(self, persister, sample_state):
        """Test successful state saving."""
        metadata = {"source": "test", "timestamp": "2024-01-01"}
        
        # Mock the persister as initialized
        persister._initialized = True
        
        result = await persister.save_state(sample_state, metadata)
        
        assert result is True
        assert len(persister.save_calls) == 1
        assert persister.save_calls[0][0] == sample_state
        assert persister.save_calls[0][1] == metadata
        
        # Verify timestamp was updated
        assert sample_state.updated_at is not None
        assert isinstance(sample_state.updated_at, datetime)
    
    @pytest.mark.asyncio
    async def test_save_state_no_task_id(self, persister):
        """Test saving state without task_id."""
        # Create state with empty task_id - but AgentState auto-generates UUID
        state = AgentState(task_id="", task_description="No task ID")
        persister._initialized = True
        
        # AgentState auto-generates a UUID, so task_id won't be empty
        assert state.task_id != ""
        
        result = await persister.save_state(state)
        
        # Should succeed because task_id was auto-generated
        assert result is True
        assert len(persister.save_calls) == 1
    
    @pytest.mark.asyncio
    async def test_save_state_implementation_failure(self, persister, sample_state):
        """Test save state when implementation raises error."""
        persister._initialized = True
        object.__setattr__(persister, 'should_fail', True)
        object.__setattr__(persister, 'fail_operation', 'save')
        
        with pytest.raises(StatePersistenceError) as exc_info:
            await persister.save_state(sample_state)
        
        assert exc_info.value.context["operation"] == "save"
        assert exc_info.value.context["task_id"] == sample_state.task_id
        assert "Mock implementation save failure" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_load_state_success(self, persister, sample_state):
        """Test successful state loading."""
        persister._initialized = True
        
        # First save a state
        await persister.save_state(sample_state)
        
        # Then load it
        loaded_state = await persister.load_state(sample_state.task_id)
        
        assert loaded_state is not None
        assert loaded_state.task_id == sample_state.task_id
        assert len(persister.load_calls) == 1
        assert persister.load_calls[0] == sample_state.task_id
    
    @pytest.mark.asyncio
    async def test_load_state_not_found(self, persister):
        """Test loading non-existent state."""
        persister._initialized = True
        
        loaded_state = await persister.load_state("nonexistent_task")
        
        assert loaded_state is None
        assert len(persister.load_calls) == 1
        assert persister.load_calls[0] == "nonexistent_task"
    
    @pytest.mark.asyncio
    async def test_load_state_implementation_failure(self, persister):
        """Test load state when implementation raises error."""
        persister._initialized = True
        object.__setattr__(persister, 'should_fail', True)
        object.__setattr__(persister, 'fail_operation', 'load')
        
        with pytest.raises(StatePersistenceError) as exc_info:
            await persister.load_state("test_task")
        
        assert exc_info.value.context["operation"] == "load"
        assert exc_info.value.context["task_id"] == "test_task"
        assert "Mock implementation load failure" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_delete_state_success(self, persister, sample_state):
        """Test successful state deletion."""
        persister._initialized = True
        
        # First save a state
        await persister.save_state(sample_state)
        
        # Then delete it
        result = await persister.delete_state(sample_state.task_id)
        
        assert result is True
        assert len(persister.delete_calls) == 1
        assert persister.delete_calls[0] == sample_state.task_id
        
        # Verify it's gone
        loaded_state = await persister.load_state(sample_state.task_id)
        assert loaded_state is None
    
    @pytest.mark.asyncio
    async def test_delete_state_not_found(self, persister):
        """Test deleting non-existent state."""
        persister._initialized = True
        
        result = await persister.delete_state("nonexistent_task")
        
        assert result is False
        assert len(persister.delete_calls) == 1
        assert persister.delete_calls[0] == "nonexistent_task"
    
    @pytest.mark.asyncio
    async def test_delete_state_implementation_failure(self, persister):
        """Test delete state when implementation raises error."""
        persister._initialized = True
        object.__setattr__(persister, 'should_fail', True)
        object.__setattr__(persister, 'fail_operation', 'delete')
        
        with pytest.raises(StatePersistenceError) as exc_info:
            await persister.delete_state("test_task")
        
        assert exc_info.value.context["operation"] == "delete"
        assert exc_info.value.context["task_id"] == "test_task"
        assert "Mock implementation delete failure" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_list_states_success(self, persister, sample_state):
        """Test successful state listing."""
        persister._initialized = True
        
        # Save a state
        await persister.save_state(sample_state)
        
        # List states
        filter_criteria = {"status": "active"}
        states = await persister.list_states(filter_criteria)
        
        assert len(states) == 1
        assert states[0]["task_id"] == sample_state.task_id
        assert len(persister.list_calls) == 1
        assert persister.list_calls[0] == filter_criteria
    
    @pytest.mark.asyncio
    async def test_list_states_empty(self, persister):
        """Test listing states when none exist."""
        persister._initialized = True
        
        states = await persister.list_states()
        
        assert states == []
        assert len(persister.list_calls) == 1
        assert persister.list_calls[0] is None
    
    @pytest.mark.asyncio
    async def test_list_states_implementation_failure(self, persister):
        """Test list states when implementation raises error."""
        persister._initialized = True
        object.__setattr__(persister, 'should_fail', True)
        object.__setattr__(persister, 'fail_operation', 'list')
        
        with pytest.raises(StatePersistenceError) as exc_info:
            await persister.list_states()
        
        assert exc_info.value.context["operation"] == "list"
        assert "Mock implementation list failure" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_timestamp_management(self, persister, sample_state):
        """Test that timestamps are properly managed."""
        persister._initialized = True
        
        # Record initial timestamp
        initial_time = sample_state.updated_at
        
        # Save state - should update timestamp
        await persister.save_state(sample_state)
        
        # Verify timestamp was updated
        assert sample_state.updated_at != initial_time
        assert sample_state.updated_at is not None
        assert isinstance(sample_state.updated_at, datetime)
    
    @pytest.mark.asyncio
    async def test_error_context_preservation(self, persister, sample_state):
        """Test that error context is preserved in StatePersistenceError."""
        persister._initialized = True
        object.__setattr__(persister, 'should_fail', True)
        object.__setattr__(persister, 'fail_operation', 'save')
        
        try:
            await persister.save_state(sample_state)
        except StatePersistenceError as e:
            assert e.context["operation"] == "save"
            assert e.context["task_id"] == sample_state.task_id
            assert e.message is not None
            assert e.cause is not None
            assert isinstance(e.cause, RuntimeError)
        else:
            pytest.fail("Expected StatePersistenceError to be raised")
    
    def test_abstract_methods_enforcement(self):
        """Test that abstract methods must be implemented."""
        # BaseStatePersister can be instantiated but calling abstract methods raises NotImplementedError
        persister = BaseStatePersister("test", MockStatePersisterSettings())
        
        # The implementation methods should raise NotImplementedError
        import asyncio
        async def test_not_implemented():
            with pytest.raises(NotImplementedError):
                await persister._save_state_impl(Mock(), None)
        
        asyncio.run(test_not_implemented())
    
    @pytest.mark.asyncio
    async def test_multiple_operations_sequence(self, persister):
        """Test sequence of multiple operations."""
        persister._initialized = True
        
        # Create multiple states
        states = [
            AgentState(task_id=f"task_{i}", task_description=f"Task {i}")
            for i in range(3)
        ]
        
        # Save all states
        for state in states:
            result = await persister.save_state(state, {"index": str(states.index(state))})
            assert result is True
        
        # List all states
        all_states = await persister.list_states()
        assert len(all_states) == 3
        
        # Load each state
        for state in states:
            loaded = await persister.load_state(state.task_id)
            assert loaded is not None
            assert loaded.task_id == state.task_id
        
        # Delete one state
        result = await persister.delete_state(states[1].task_id)
        assert result is True
        
        # Verify count reduced
        remaining_states = await persister.list_states()
        assert len(remaining_states) == 2
        
        # Verify specific state is gone
        missing = await persister.load_state(states[1].task_id)
        assert missing is None
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, persister):
        """Test concurrent access to persister."""
        import asyncio
        
        persister._initialized = True
        
        # Create states for concurrent operations
        states = [
            AgentState(task_id=f"concurrent_{i}", task_description=f"Concurrent task {i}")
            for i in range(5)
        ]
        
        # Save all states concurrently
        save_tasks = [persister.save_state(state) for state in states]
        results = await asyncio.gather(*save_tasks)
        assert all(results)
        
        # Load all states concurrently
        load_tasks = [persister.load_state(state.task_id) for state in states]
        loaded_states = await asyncio.gather(*load_tasks)
        assert all(loaded is not None for loaded in loaded_states)
        
        # Verify call counts
        assert len(persister.save_calls) == 5
        assert len(persister.load_calls) == 5


class TestBaseStatePersisterEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def persister(self):
        """Create a concrete persister instance."""
        return ConcreteStatePersister()
    
    @pytest.mark.asyncio
    async def test_save_state_with_none_metadata(self, persister):
        """Test saving state with explicitly None metadata."""
        persister._initialized = True
        state = AgentState(task_id="test_none_meta", task_description="Test")
        
        result = await persister.save_state(state, None)
        
        assert result is True
        assert persister.save_calls[0][1] is None
    
    @pytest.mark.asyncio
    async def test_save_state_with_empty_metadata(self, persister):
        """Test saving state with empty metadata dict."""
        persister._initialized = True
        state = AgentState(task_id="test_empty_meta", task_description="Test")
        
        result = await persister.save_state(state, {})
        
        assert result is True
        assert persister.save_calls[0][1] == {}
    
    @pytest.mark.asyncio
    async def test_load_state_empty_task_id(self, persister):
        """Test loading state with empty task ID."""
        persister._initialized = True
        
        loaded_state = await persister.load_state("")
        
        assert loaded_state is None
        assert len(persister.load_calls) == 1
        assert persister.load_calls[0] == ""
    
    @pytest.mark.asyncio
    async def test_delete_state_empty_task_id(self, persister):
        """Test deleting state with empty task ID."""
        persister._initialized = True
        
        result = await persister.delete_state("")
        
        assert result is False
        assert len(persister.delete_calls) == 1
        assert persister.delete_calls[0] == ""
    
    @pytest.mark.asyncio
    async def test_list_states_with_empty_filter(self, persister):
        """Test listing states with empty filter criteria."""
        persister._initialized = True
        
        states = await persister.list_states({})
        
        assert states == []
        assert len(persister.list_calls) == 1
        assert persister.list_calls[0] == {}
    
    @pytest.mark.asyncio
    async def test_operations_call_implementation_methods(self, persister):
        """Test that public methods call corresponding implementation methods."""
        persister._initialized = True
        state = AgentState(task_id="test_call", task_description="Test")
        
        # Test save calls _save_state_impl
        await persister.save_state(state)
        assert len(persister.save_calls) == 1
        
        # Test load calls _load_state_impl
        await persister.load_state("test_call")
        assert len(persister.load_calls) == 1
        
        # Test delete calls _delete_state_impl
        await persister.delete_state("test_call")
        assert len(persister.delete_calls) == 1
        
        # Test list calls _list_states_impl
        await persister.list_states()
        assert len(persister.list_calls) == 1
    
    def test_settings_type_validation(self):
        """Test that settings type is properly validated."""
        valid_settings = MockStatePersisterSettings(test_setting="valid")
        persister = ConcreteStatePersister(settings=valid_settings)
        
        assert persister.settings.test_setting == "valid"
        assert isinstance(persister.settings, MockStatePersisterSettings)