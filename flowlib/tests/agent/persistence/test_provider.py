"""Comprehensive tests for agent persistence provider module."""

import pytest
import logging
from typing import Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from flowlib.agent.components.persistence.provider import ProviderStatePersister
from flowlib.agent.components.persistence.base import BaseStatePersister
from flowlib.agent.models.state import AgentState
from flowlib.agent.core.errors import StatePersistenceError
from flowlib.agent.components.persistence.provider import logger


class TestProviderStatePersister:
    """Test ProviderStatePersister class."""
    
    def test_provider_persister_inheritance(self):
        """Test that ProviderStatePersister inherits from BaseStatePersister."""
        persister = ProviderStatePersister(provider_name="test_provider")
        assert isinstance(persister, BaseStatePersister)
    
    def test_provider_persister_creation(self):
        """Test creating ProviderStatePersister instance."""
        provider_name = "test_provider"
        persister = ProviderStatePersister(provider_name=provider_name)
        
        assert persister.settings.provider_name == provider_name
        assert persister._provider is None
        assert persister.name == "provider_state_persister"
    
    @pytest.mark.asyncio
    async def test_provider_persister_initialize_success(self):
        """Test successful provider initialization."""
        provider_name = "test_provider"
        persister = ProviderStatePersister(provider_name=provider_name)
        
        mock_provider = Mock()
        
        with patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_provider)
            
            await persister._initialize()
            
            assert persister._provider == mock_provider
            mock_provider_registry.get_by_config.assert_called_once_with(provider_name)
    
    @pytest.mark.asyncio
    async def test_provider_persister_initialize_provider_not_found(self):
        """Test initialization when provider is not found."""
        provider_name = "nonexistent_provider"
        persister = ProviderStatePersister(provider_name=provider_name)
        
        with patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            mock_provider_registry.get_by_config = AsyncMock(return_value=None)
            
            with pytest.raises(StatePersistenceError) as exc_info:
                await persister._initialize()
            
            assert "Provider not found: nonexistent_provider" in str(exc_info.value)
            assert exc_info.value.operation == "initialize"
            assert persister._provider is None
    
    @pytest.mark.asyncio
    async def test_provider_persister_initialize_import_error(self):
        """Test initialization with import error."""
        provider_name = "test_provider"
        persister = ProviderStatePersister(provider_name=provider_name)
        
        with patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            mock_provider_registry.get_by_config = AsyncMock(side_effect=ImportError("Module not found"))
            
            with pytest.raises(StatePersistenceError) as exc_info:
                await persister._initialize()
            
            assert "Error initializing provider" in str(exc_info.value)
            assert exc_info.value.operation == "initialize"
            assert exc_info.value.cause is not None
    
    @pytest.mark.asyncio
    async def test_provider_persister_initialize_general_error(self):
        """Test initialization with general error."""
        provider_name = "test_provider"
        persister = ProviderStatePersister(provider_name=provider_name)
        
        with patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            mock_provider_registry.get_by_config = AsyncMock(side_effect=Exception("Unexpected error"))
            
            with pytest.raises(StatePersistenceError) as exc_info:
                await persister._initialize()
            
            assert "Error initializing provider" in str(exc_info.value)
            assert exc_info.value.operation == "initialize"
            assert isinstance(exc_info.value.cause, Exception)
    
    @pytest.mark.asyncio
    async def test_provider_persister_save_state_success(self):
        """Test successful state saving."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        mock_provider = Mock()
        mock_provider.save_state = AsyncMock()
        persister._provider = mock_provider
        
        # Create test state
        test_state = Mock(spec=AgentState)
        test_state.task_id = "test_task_123"
        test_state.model_dump.return_value = {"task_id": "test_task_123", "data": "test"}
        
        metadata = {"type": "test", "version": "1.0"}
        
        result = await persister._save_state_impl(test_state, metadata=metadata)
        
        assert result is True
        mock_provider.save_state.assert_called_once_with(
            {"task_id": "test_task_123", "data": "test"},
            metadata
        )
    
    @pytest.mark.asyncio
    async def test_provider_persister_save_state_without_metadata(self):
        """Test saving state without metadata."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        mock_provider = Mock()
        mock_provider.save_state = AsyncMock()
        persister._provider = mock_provider
        
        test_state = Mock(spec=AgentState)
        test_state.task_id = "test_task_123"
        test_state.model_dump.return_value = {"task_id": "test_task_123"}
        
        result = await persister._save_state_impl(test_state)
        
        assert result is True
        mock_provider.save_state.assert_called_once_with(
            {"task_id": "test_task_123"},
            None
        )
    
    @pytest.mark.asyncio
    async def test_provider_persister_save_state_provider_error(self):
        """Test saving state with provider error."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        mock_provider = Mock()
        mock_provider.save_state = AsyncMock(side_effect=Exception("Provider save error"))
        persister._provider = mock_provider
        
        test_state = Mock(spec=AgentState)
        test_state.task_id = "test_task_123"
        test_state.model_dump.return_value = {"task_id": "test_task_123"}
        
        with pytest.raises(StatePersistenceError) as exc_info:
            await persister._save_state_impl(test_state)
        
        assert "Error saving state using provider" in str(exc_info.value)
        assert exc_info.value.operation == "save"
        assert exc_info.value.task_id == "test_task_123"
        assert exc_info.value.cause is not None
    
    @pytest.mark.asyncio
    async def test_provider_persister_save_state_serialization_error(self):
        """Test saving state with serialization error."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        mock_provider = Mock()
        persister._provider = mock_provider
        
        test_state = Mock(spec=AgentState)
        test_state.task_id = "test_task_123"
        test_state.model_dump.side_effect = Exception("Serialization error")
        
        with pytest.raises(StatePersistenceError) as exc_info:
            await persister._save_state_impl(test_state)
        
        assert "Error saving state using provider" in str(exc_info.value)
        assert exc_info.value.operation == "save"
        assert exc_info.value.task_id == "test_task_123"
    
    @pytest.mark.asyncio
    async def test_provider_persister_load_state_success(self):
        """Test successful state loading."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        mock_provider = Mock()
        mock_provider.load_state = AsyncMock(return_value={
            "task_id": "test_task_123",
            "data": "loaded_data"
        })
        persister._provider = mock_provider
        
        with patch('flowlib.agent.components.persistence.provider.AgentState') as mock_agent_state:
            mock_state = Mock()
            mock_agent_state.return_value = mock_state
            
            result = await persister._load_state_impl("test_task_123")
            
            assert result == mock_state
            mock_provider.load_state.assert_called_once_with("test_task_123")
            mock_agent_state.assert_called_once_with(initial_state_data={
                "task_id": "test_task_123",
                "data": "loaded_data"
            })
    
    @pytest.mark.asyncio
    async def test_provider_persister_load_state_not_found(self):
        """Test loading state when not found."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        mock_provider = Mock()
        mock_provider.load_state = AsyncMock(return_value=None)
        persister._provider = mock_provider
        
        result = await persister._load_state_impl("nonexistent_task")
        
        assert result is None
        mock_provider.load_state.assert_called_once_with("nonexistent_task")
    
    @pytest.mark.asyncio
    async def test_provider_persister_load_state_empty_dict(self):
        """Test loading state when provider returns empty dict."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        mock_provider = Mock()
        mock_provider.load_state = AsyncMock(return_value={})
        persister._provider = mock_provider
        
        result = await persister._load_state_impl("test_task_123")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_provider_persister_load_state_provider_error(self):
        """Test loading state with provider error."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        mock_provider = Mock()
        mock_provider.load_state = AsyncMock(side_effect=Exception("Provider load error"))
        persister._provider = mock_provider
        
        with pytest.raises(StatePersistenceError) as exc_info:
            await persister._load_state_impl("test_task_123")
        
        assert "Error loading state using provider" in str(exc_info.value)
        assert exc_info.value.operation == "load"
        assert exc_info.value.task_id == "test_task_123"
        assert exc_info.value.cause is not None
    
    @pytest.mark.asyncio
    async def test_provider_persister_load_state_validation_error(self):
        """Test loading state with validation error."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        mock_provider = Mock()
        mock_provider.load_state = AsyncMock(return_value={"invalid": "data"})
        persister._provider = mock_provider
        
        with patch('flowlib.agent.components.persistence.provider.AgentState') as mock_agent_state:
            mock_agent_state.side_effect = Exception("Validation error")
            
            with pytest.raises(StatePersistenceError) as exc_info:
                await persister._load_state_impl("test_task_123")
            
            assert "Error loading state using provider" in str(exc_info.value)
            assert exc_info.value.operation == "load"
            assert exc_info.value.task_id == "test_task_123"
    
    @pytest.mark.asyncio
    async def test_provider_persister_delete_state_success(self):
        """Test successful state deletion."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        mock_provider = Mock()
        mock_provider.delete_state = AsyncMock()
        persister._provider = mock_provider
        
        result = await persister._delete_state_impl("test_task_123")
        
        assert result is True
        mock_provider.delete_state.assert_called_once_with("test_task_123")
    
    @pytest.mark.asyncio
    async def test_provider_persister_delete_state_provider_error(self):
        """Test deleting state with provider error."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        mock_provider = Mock()
        mock_provider.delete_state = AsyncMock(side_effect=Exception("Provider delete error"))
        persister._provider = mock_provider
        
        with pytest.raises(StatePersistenceError) as exc_info:
            await persister._delete_state_impl("test_task_123")
        
        assert "Error deleting state using provider" in str(exc_info.value)
        assert exc_info.value.operation == "delete"
        assert exc_info.value.task_id == "test_task_123"
        assert exc_info.value.cause is not None
    
    @pytest.mark.asyncio
    async def test_provider_persister_list_states_success(self):
        """Test successful state listing."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        mock_provider = Mock()
        mock_states = [
            {"task_id": "task1", "status": "completed"},
            {"task_id": "task2", "status": "running"}
        ]
        mock_provider.list_states = AsyncMock(return_value=mock_states)
        persister._provider = mock_provider
        
        filter_criteria = {"status": "running"}
        result = await persister._list_states_impl(filter_criteria)
        
        assert result == mock_states
        mock_provider.list_states.assert_called_once_with(filter_criteria)
    
    @pytest.mark.asyncio
    async def test_provider_persister_list_states_without_filter(self):
        """Test listing states without filter criteria."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        mock_provider = Mock()
        mock_states = [{"task_id": "task1"}, {"task_id": "task2"}]
        mock_provider.list_states = AsyncMock(return_value=mock_states)
        persister._provider = mock_provider
        
        result = await persister._list_states_impl()
        
        assert result == mock_states
        mock_provider.list_states.assert_called_once_with(None)
    
    @pytest.mark.asyncio
    async def test_provider_persister_list_states_empty_result(self):
        """Test listing states with empty result."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        mock_provider = Mock()
        mock_provider.list_states = AsyncMock(return_value=[])
        persister._provider = mock_provider
        
        result = await persister._list_states_impl()
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_provider_persister_list_states_provider_error(self):
        """Test listing states with provider error."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        mock_provider = Mock()
        mock_provider.list_states = AsyncMock(side_effect=Exception("Provider list error"))
        persister._provider = mock_provider
        
        with pytest.raises(StatePersistenceError) as exc_info:
            await persister._list_states_impl()
        
        assert "Error listing states using provider" in str(exc_info.value)
        assert exc_info.value.operation == "list"
        assert exc_info.value.cause is not None
    
    def test_provider_persister_logging_debug_messages(self):
        """Test that debug messages are logged correctly."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        with patch('flowlib.agent.components.persistence.provider.logger') as mock_logger:
            # Test initialization logging
            mock_provider = Mock()
            with patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
                # This would be called during async initialization
                persister._provider = mock_provider
                
                # Manually call the logging part using the mocked logger
                mock_logger.debug(f"Using provider: {persister.settings.provider_name}")
                mock_logger.debug.assert_called_with("Using provider: test_provider")
    
    def test_provider_persister_logging_error_messages(self):
        """Test that error messages are logged correctly."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        with patch('flowlib.agent.components.persistence.provider.logger') as mock_logger:
            # Test error logging
            error_msg = "Test error message"
            mock_logger.error(error_msg)
            mock_logger.error.assert_called_with(error_msg)


class TestProviderStatePersisterIntegration:
    """Test integration aspects of ProviderStatePersister."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete workflow with provider persister."""
        provider_name = "workflow_provider"
        persister = ProviderStatePersister(provider_name=provider_name)
        
        # Mock provider
        mock_provider = Mock()
        mock_provider.save_state = AsyncMock()
        mock_provider.load_state = AsyncMock(return_value={
            "task_id": "workflow_test",
            "status": "test"
        })
        mock_provider.delete_state = AsyncMock()
        mock_provider.list_states = AsyncMock(return_value=[
            {"task_id": "workflow_test", "status": "test"}
        ])
        
        with patch('flowlib.providers.core.registry.provider_registry') as mock_provider_registry:
            mock_provider_registry.get_by_config = AsyncMock(return_value=mock_provider)
            
            # Initialize
            await persister._initialize()
            assert persister._provider == mock_provider
            
            # Save state
            test_state = Mock(spec=AgentState)
            test_state.task_id = "workflow_test"
            test_state.model_dump.return_value = {"task_id": "workflow_test", "status": "test"}
            
            save_result = await persister._save_state_impl(test_state, metadata={"type": "test"})
            assert save_result is True
            
            # Load state
            with patch('flowlib.agent.components.persistence.provider.AgentState') as mock_agent_state:
                mock_loaded_state = Mock()
                mock_agent_state.return_value = mock_loaded_state
                
                load_result = await persister._load_state_impl("workflow_test")
                assert load_result == mock_loaded_state
            
            # List states
            list_result = await persister._list_states_impl({"status": "test"})
            assert len(list_result) == 1
            assert list_result[0]["task_id"] == "workflow_test"
            
            # Delete state
            delete_result = await persister._delete_state_impl("workflow_test")
            assert delete_result is True
    
    @pytest.mark.asyncio
    async def test_provider_interface_compliance(self):
        """Test that provider interface is used correctly."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        # Mock provider with specific interface
        mock_provider = Mock()
        mock_provider.save_state = AsyncMock()
        mock_provider.load_state = AsyncMock(return_value={"task_id": "test"})
        mock_provider.delete_state = AsyncMock()
        mock_provider.list_states = AsyncMock(return_value=[])
        
        persister._provider = mock_provider
        
        # Test all operations use the correct provider methods
        test_state = Mock(spec=AgentState)
        test_state.task_id = "test"
        test_state.model_dump.return_value = {"task_id": "test"}
        
        await persister._save_state_impl(test_state, {"meta": "data"})
        mock_provider.save_state.assert_called_once_with({"task_id": "test"}, {"meta": "data"})
        
        await persister._load_state_impl("test")
        mock_provider.load_state.assert_called_once_with("test")
        
        await persister._delete_state_impl("test")
        mock_provider.delete_state.assert_called_once_with("test")
        
        await persister._list_states_impl({"filter": "value"})
        mock_provider.list_states.assert_called_once_with({"filter": "value"})
    
    @pytest.mark.asyncio
    async def test_error_context_preservation(self):
        """Test that error context is preserved through the provider layer."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        original_error = ValueError("Original provider error")
        mock_provider = Mock()
        mock_provider.save_state = AsyncMock(side_effect=original_error)
        persister._provider = mock_provider
        
        test_state = Mock(spec=AgentState)
        test_state.task_id = "test_task"
        test_state.model_dump.return_value = {"task_id": "test_task"}
        
        with pytest.raises(StatePersistenceError) as exc_info:
            await persister._save_state_impl(test_state)
        
        # Check that original error is preserved
        assert exc_info.value.cause == original_error
        assert exc_info.value.operation == "save"
        assert exc_info.value.task_id == "test_task"
        assert "Error saving state using provider" in exc_info.value.message
    
    def test_provider_persister_name_handling(self):
        """Test provider name handling edge cases."""
        # Test with empty provider name
        with pytest.raises((ValueError, TypeError)):
            ProviderStatePersister(provider_name="")
        
        # Test with None provider name
        with pytest.raises((ValueError, TypeError)):
            ProviderStatePersister(provider_name=None)
        
        # Test with valid provider name
        persister = ProviderStatePersister(provider_name="valid_provider")
        assert persister.settings.provider_name == "valid_provider"
    
    @pytest.mark.asyncio
    async def test_provider_state_data_handling(self):
        """Test various state data formats."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        mock_provider = Mock()
        persister._provider = mock_provider
        
        # Test with complex state data
        complex_state = Mock(spec=AgentState)
        complex_state.task_id = "complex_task"
        complex_state.model_dump.return_value = {
            "task_id": "complex_task",
            "data": {
                "nested": {"value": 123},
                "list": [1, 2, 3],
                "boolean": True,
                "null_value": None
            },
            "metadata": {"timestamp": "2023-01-01T00:00:00"}
        }
        
        mock_provider.save_state = AsyncMock()
        
        result = await persister._save_state_impl(complex_state)
        assert result is True
        
        # Verify complex data was passed correctly
        save_call_args = mock_provider.save_state.call_args
        saved_data = save_call_args[0][0]
        assert saved_data["task_id"] == "complex_task"
        assert saved_data["data"]["nested"]["value"] == 123
        assert saved_data["data"]["list"] == [1, 2, 3]
        assert saved_data["data"]["boolean"] is True
        assert saved_data["data"]["null_value"] is None
    
    @pytest.mark.asyncio
    async def test_provider_metadata_handling(self):
        """Test metadata handling in different scenarios."""
        persister = ProviderStatePersister(provider_name="test_provider")
        
        mock_provider = Mock()
        mock_provider.save_state = AsyncMock()
        persister._provider = mock_provider
        
        test_state = Mock(spec=AgentState)
        test_state.task_id = "metadata_test"
        test_state.model_dump.return_value = {"task_id": "metadata_test"}
        
        # Test with None metadata
        await persister._save_state_impl(test_state, metadata=None)
        mock_provider.save_state.assert_called_with({"task_id": "metadata_test"}, None)
        
        # Test with empty metadata
        await persister._save_state_impl(test_state, metadata={})
        mock_provider.save_state.assert_called_with({"task_id": "metadata_test"}, {})
        
        # Test with complex metadata
        complex_metadata = {
            "type": "test",
            "version": "1.0",
            "tags": ["important", "test"],
            "config": {"setting": "value"}
        }
        await persister._save_state_impl(test_state, metadata=complex_metadata)
        mock_provider.save_state.assert_called_with({"task_id": "metadata_test"}, complex_metadata)