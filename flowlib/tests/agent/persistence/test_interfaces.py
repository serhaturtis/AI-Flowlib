"""Tests for persistence interfaces."""

import pytest
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, Mock

from flowlib.agent.components.persistence.interfaces import StatePersistenceInterface
from flowlib.agent.models.state import AgentState
from flowlib.agent.core.errors import StatePersistenceError


class MockStatePersister:
    """Mock implementation of StatePersistenceInterface for testing."""
    
    def __init__(self):
        self.saved_states: Dict[str, AgentState] = {}
        self.metadata_store: Dict[str, Dict[str, str]] = {}
        self.should_fail = False
        self.fail_operation = None
    
    async def save_state(
        self,
        state: AgentState,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        if self.should_fail and self.fail_operation == "save":
            raise StatePersistenceError("Mock save failure", "save", state.task_id)
        
        self.saved_states[state.task_id] = state
        if metadata:
            self.metadata_store[state.task_id] = metadata
        return True
    
    async def load_state(self, task_id: str) -> Optional[AgentState]:
        if self.should_fail and self.fail_operation == "load":
            raise StatePersistenceError("Mock load failure", "load", task_id)
        
        return self.saved_states.get(task_id)
    
    async def delete_state(self, task_id: str) -> bool:
        if self.should_fail and self.fail_operation == "delete":
            raise StatePersistenceError("Mock delete failure", "delete", task_id)
        
        if task_id in self.saved_states:
            del self.saved_states[task_id]
            self.metadata_store.pop(task_id, None)
            return True
        return False
    
    async def list_states(
        self,
        filter_criteria: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        if self.should_fail and self.fail_operation == "list":
            raise StatePersistenceError("Mock list failure", "list")
        
        result = []
        for task_id, state in self.saved_states.items():
            metadata = {
                "task_id": task_id,
                "task_description": state.task_description,
                "is_complete": str(state.is_complete),
                "progress": str(state.progress)
            }
            # Add stored metadata
            if task_id in self.metadata_store:
                metadata.update(self.metadata_store[task_id])
            
            # Apply filter if provided
            if filter_criteria:
                matches = all(
                    metadata.get(key) == value 
                    for key, value in filter_criteria.items()
                )
                if matches:
                    result.append(metadata)
            else:
                result.append(metadata)
        
        return result


class TestStatePersistenceInterface:
    """Test StatePersistenceInterface protocol contract."""
    
    @pytest.fixture
    def mock_persister(self):
        """Create a mock state persister."""
        return MockStatePersister()
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample agent state."""
        return AgentState(
            task_id="test_task_123",
            task_description="Test task for persistence testing"
        )
    
    @pytest.fixture
    def another_state(self):
        """Create another sample agent state."""
        return AgentState(
            task_id="test_task_456",
            task_description="Another test task"
        )
    
    @pytest.mark.asyncio
    async def test_interface_protocol_compliance(self, mock_persister):
        """Test that mock persister implements the interface correctly."""
        # Verify the mock persister has all required methods
        assert hasattr(mock_persister, 'save_state')
        assert hasattr(mock_persister, 'load_state')
        assert hasattr(mock_persister, 'delete_state')
        assert hasattr(mock_persister, 'list_states')
        
        # Verify methods are callable
        assert callable(mock_persister.save_state)
        assert callable(mock_persister.load_state)
        assert callable(mock_persister.delete_state)
        assert callable(mock_persister.list_states)
    
    @pytest.mark.asyncio
    async def test_save_state_success(self, mock_persister, sample_state):
        """Test successful state saving."""
        metadata = {"source": "test", "version": "1.0"}
        
        result = await mock_persister.save_state(sample_state, metadata)
        
        assert result is True
        assert sample_state.task_id in mock_persister.saved_states
        assert mock_persister.saved_states[sample_state.task_id] == sample_state
        assert mock_persister.metadata_store[sample_state.task_id] == metadata
    
    @pytest.mark.asyncio
    async def test_save_state_without_metadata(self, mock_persister, sample_state):
        """Test saving state without metadata."""
        result = await mock_persister.save_state(sample_state)
        
        assert result is True
        assert sample_state.task_id in mock_persister.saved_states
        assert sample_state.task_id not in mock_persister.metadata_store
    
    @pytest.mark.asyncio
    async def test_save_state_failure(self, mock_persister, sample_state):
        """Test save state failure handling."""
        mock_persister.should_fail = True
        mock_persister.fail_operation = "save"
        
        with pytest.raises(StatePersistenceError) as exc_info:
            await mock_persister.save_state(sample_state)
        
        assert exc_info.value.context["operation"] == "save"
        assert exc_info.value.context["task_id"] == sample_state.task_id
        assert "Mock save failure" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_load_state_success(self, mock_persister, sample_state):
        """Test successful state loading."""
        # First save the state
        await mock_persister.save_state(sample_state)
        
        # Then load it
        loaded_state = await mock_persister.load_state(sample_state.task_id)
        
        assert loaded_state is not None
        assert loaded_state.task_id == sample_state.task_id
        assert loaded_state.task_description == sample_state.task_description
    
    @pytest.mark.asyncio
    async def test_load_state_not_found(self, mock_persister):
        """Test loading non-existent state."""
        loaded_state = await mock_persister.load_state("nonexistent_task")
        
        assert loaded_state is None
    
    @pytest.mark.asyncio
    async def test_load_state_failure(self, mock_persister):
        """Test load state failure handling."""
        mock_persister.should_fail = True
        mock_persister.fail_operation = "load"
        
        with pytest.raises(StatePersistenceError) as exc_info:
            await mock_persister.load_state("test_task")
        
        assert exc_info.value.context["operation"] == "load"
        assert exc_info.value.context["task_id"] == "test_task"
        assert "Mock load failure" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_delete_state_success(self, mock_persister, sample_state):
        """Test successful state deletion."""
        # First save the state
        await mock_persister.save_state(sample_state, {"test": "metadata"})
        assert sample_state.task_id in mock_persister.saved_states
        assert sample_state.task_id in mock_persister.metadata_store
        
        # Then delete it
        result = await mock_persister.delete_state(sample_state.task_id)
        
        assert result is True
        assert sample_state.task_id not in mock_persister.saved_states
        assert sample_state.task_id not in mock_persister.metadata_store
    
    @pytest.mark.asyncio
    async def test_delete_state_not_found(self, mock_persister):
        """Test deleting non-existent state."""
        result = await mock_persister.delete_state("nonexistent_task")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_delete_state_failure(self, mock_persister):
        """Test delete state failure handling."""
        mock_persister.should_fail = True
        mock_persister.fail_operation = "delete"
        
        with pytest.raises(StatePersistenceError) as exc_info:
            await mock_persister.delete_state("test_task")
        
        assert exc_info.value.context["operation"] == "delete"
        assert exc_info.value.context["task_id"] == "test_task"
        assert "Mock delete failure" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_list_states_empty(self, mock_persister):
        """Test listing states when none exist."""
        states = await mock_persister.list_states()
        
        assert states == []
    
    @pytest.mark.asyncio
    async def test_list_states_multiple(self, mock_persister, sample_state, another_state):
        """Test listing multiple states."""
        # Save two states with metadata
        await mock_persister.save_state(sample_state, {"source": "test1"})
        await mock_persister.save_state(another_state, {"source": "test2"})
        
        states = await mock_persister.list_states()
        
        assert len(states) == 2
        
        # Check first state metadata
        state1_meta = next(s for s in states if s["task_id"] == sample_state.task_id)
        assert state1_meta["task_description"] == sample_state.task_description
        assert state1_meta["source"] == "test1"
        assert state1_meta["is_complete"] == str(sample_state.is_complete)
        assert state1_meta["progress"] == str(sample_state.progress)
        
        # Check second state metadata
        state2_meta = next(s for s in states if s["task_id"] == another_state.task_id)
        assert state2_meta["task_description"] == another_state.task_description
        assert state2_meta["source"] == "test2"
    
    @pytest.mark.asyncio
    async def test_list_states_with_filter(self, mock_persister, sample_state, another_state):
        """Test listing states with filter criteria."""
        # Set different completion status
        sample_state.set_complete("Task finished")
        
        # Save states with different metadata
        await mock_persister.save_state(sample_state, {"environment": "test"})
        await mock_persister.save_state(another_state, {"environment": "prod"})
        
        # Filter by environment
        test_states = await mock_persister.list_states({"environment": "test"})
        assert len(test_states) == 1
        assert test_states[0]["task_id"] == sample_state.task_id
        
        # Filter by completion status
        completed_states = await mock_persister.list_states({"is_complete": "True"})
        assert len(completed_states) == 1
        assert completed_states[0]["task_id"] == sample_state.task_id
        
        # Filter with no matches
        no_matches = await mock_persister.list_states({"environment": "staging"})
        assert len(no_matches) == 0
    
    @pytest.mark.asyncio
    async def test_list_states_failure(self, mock_persister):
        """Test list states failure handling."""
        mock_persister.should_fail = True
        mock_persister.fail_operation = "list"
        
        with pytest.raises(StatePersistenceError) as exc_info:
            await mock_persister.list_states()
        
        assert exc_info.value.context["operation"] == "list"
        assert "Mock list failure" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_full_state_lifecycle(self, mock_persister, sample_state):
        """Test complete state lifecycle: save -> load -> delete."""
        metadata = {"lifecycle_test": "true", "version": "1.0"}
        
        # Save state
        save_result = await mock_persister.save_state(sample_state, metadata)
        assert save_result is True
        
        # Verify it appears in list
        all_states = await mock_persister.list_states()
        assert len(all_states) == 1
        assert all_states[0]["task_id"] == sample_state.task_id
        assert all_states[0]["lifecycle_test"] == "true"
        
        # Load state
        loaded_state = await mock_persister.load_state(sample_state.task_id)
        assert loaded_state is not None
        assert loaded_state.task_id == sample_state.task_id
        
        # Delete state
        delete_result = await mock_persister.delete_state(sample_state.task_id)
        assert delete_result is True
        
        # Verify it's gone
        final_states = await mock_persister.list_states()
        assert len(final_states) == 0
        
        # Verify load returns None
        missing_state = await mock_persister.load_state(sample_state.task_id)
        assert missing_state is None
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_persister):
        """Test concurrent state operations."""
        import asyncio
        
        # Create multiple states
        states = [
            AgentState(task_id=f"concurrent_task_{i}", task_description=f"Task {i}")
            for i in range(5)
        ]
        
        # Save all states concurrently
        save_tasks = [
            mock_persister.save_state(state, {"index": str(i)})
            for i, state in enumerate(states)
        ]
        save_results = await asyncio.gather(*save_tasks)
        assert all(save_results)
        
        # Load all states concurrently
        load_tasks = [mock_persister.load_state(state.task_id) for state in states]
        loaded_states = await asyncio.gather(*load_tasks)
        assert all(loaded_state is not None for loaded_state in loaded_states)
        assert len(loaded_states) == 5
        
        # Verify all states in list
        all_states = await mock_persister.list_states()
        assert len(all_states) == 5
        
        # Delete all states concurrently
        delete_tasks = [mock_persister.delete_state(state.task_id) for state in states]
        delete_results = await asyncio.gather(*delete_tasks)
        assert all(delete_results)
        
        # Verify all gone
        final_states = await mock_persister.list_states()
        assert len(final_states) == 0


class TestInterfaceTypeHints:
    """Test interface type annotations and protocol compliance."""
    
    def test_protocol_signature_compliance(self):
        """Test that interface has correct method signatures."""
        import inspect
        from typing import get_type_hints
        
        # Get the save_state method
        save_method = StatePersistenceInterface.save_state
        save_sig = inspect.signature(save_method)
        
        # Check parameter names and types
        params = list(save_sig.parameters.keys())
        assert "self" in params
        assert "state" in params  
        assert "metadata" in params
        
        # Get type hints
        hints = get_type_hints(save_method)
        assert "return" in hints
        
        # Check other methods exist
        assert hasattr(StatePersistenceInterface, "load_state")
        assert hasattr(StatePersistenceInterface, "delete_state")
        assert hasattr(StatePersistenceInterface, "list_states")
    
    def test_protocol_method_count(self):
        """Test that protocol has expected number of methods."""
        methods = [attr for attr in dir(StatePersistenceInterface) 
                  if not attr.startswith('_') and callable(getattr(StatePersistenceInterface, attr))]
        
        # Should have exactly 4 methods: save_state, load_state, delete_state, list_states
        assert len(methods) == 4
        assert "save_state" in methods
        assert "load_state" in methods
        assert "delete_state" in methods
        assert "list_states" in methods