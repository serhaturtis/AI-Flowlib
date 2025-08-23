"""Tests for persistence utils."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from flowlib.agent.components.persistence.utils import list_saved_states_metadata


class TestPersistenceUtils:
    """Test persistence utility functions."""
    
    @pytest.mark.asyncio
    async def test_list_saved_states_metadata_success(self):
        """Test successful listing of saved states metadata."""
        # Mock persister
        mock_persister = Mock()
        mock_persister.initialize = AsyncMock()
        mock_persister.list_states = AsyncMock(return_value=[
            {"task_id": "task1", "timestamp": "2024-01-01T00:00:00"},
            {"task_id": "task2", "timestamp": "2024-01-01T01:00:00"}
        ])
        mock_persister.shutdown = AsyncMock()
        mock_persister.initialized = True
        
        # Mock factory
        with patch('flowlib.agent.components.persistence.utils.create_state_persister', 
                  return_value=mock_persister) as mock_factory:
            
            result = await list_saved_states_metadata(
                persister_type="file",
                base_path="./test_states"
            )
            
            # Verify factory was called correctly
            mock_factory.assert_called_once_with(
                persister_type="file",
                base_path="./test_states"
            )
            
            # Verify persister methods were called
            mock_persister.initialize.assert_called_once()
            mock_persister.list_states.assert_called_once()
            mock_persister.shutdown.assert_called_once()
            
            # Verify result
            assert len(result) == 2
            assert result[0]["task_id"] == "task1"
            assert result[1]["task_id"] == "task2"
    
    @pytest.mark.asyncio
    async def test_list_saved_states_metadata_empty(self):
        """Test listing when no saved states exist."""
        mock_persister = Mock()
        mock_persister.initialize = AsyncMock()
        mock_persister.list_states = AsyncMock(return_value=[])
        mock_persister.shutdown = AsyncMock()
        mock_persister.initialized = True
        
        with patch('flowlib.agent.components.persistence.utils.create_state_persister',
                  return_value=mock_persister):
            
            result = await list_saved_states_metadata(persister_type="database")
            
            assert result == []
            mock_persister.list_states.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_saved_states_metadata_factory_error(self):
        """Test handling of factory creation error."""
        with patch('flowlib.agent.components.persistence.utils.create_state_persister',
                  side_effect=Exception("Factory error")):
            
            result = await list_saved_states_metadata(
                persister_type="invalid",
                invalid_param="value"
            )
            
            # Should return empty list on error
            assert result == []
    
    @pytest.mark.asyncio
    async def test_list_saved_states_metadata_initialize_error(self):
        """Test handling of persister initialization error."""
        mock_persister = Mock()
        mock_persister.initialize = AsyncMock(side_effect=Exception("Init error"))
        mock_persister.initialized = False
        
        with patch('flowlib.agent.components.persistence.utils.create_state_persister',
                  return_value=mock_persister):
            
            result = await list_saved_states_metadata(persister_type="file")
            
            assert result == []
            mock_persister.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_saved_states_metadata_list_error(self):
        """Test handling of list_states error."""
        mock_persister = Mock()
        mock_persister.initialize = AsyncMock()
        mock_persister.list_states = AsyncMock(side_effect=Exception("List error"))
        mock_persister.shutdown = AsyncMock()
        mock_persister.initialized = True
        
        with patch('flowlib.agent.components.persistence.utils.create_state_persister',
                  return_value=mock_persister):
            
            result = await list_saved_states_metadata(persister_type="database")
            
            assert result == []
            mock_persister.list_states.assert_called_once()
            # Should still attempt shutdown
            mock_persister.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_saved_states_metadata_shutdown_error(self):
        """Test handling of shutdown error."""
        mock_persister = Mock()
        mock_persister.initialize = AsyncMock()
        mock_persister.list_states = AsyncMock(return_value=[{"task_id": "task1"}])
        mock_persister.shutdown = AsyncMock(side_effect=Exception("Shutdown error"))
        mock_persister.initialized = True
        
        with patch('flowlib.agent.components.persistence.utils.create_state_persister',
                  return_value=mock_persister):
            
            # Should still return results even if shutdown fails
            result = await list_saved_states_metadata(persister_type="file")
            
            assert len(result) == 1
            assert result[0]["task_id"] == "task1"
            mock_persister.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_saved_states_metadata_not_initialized(self):
        """Test when persister is not initialized (no shutdown needed)."""
        mock_persister = Mock()
        mock_persister.initialize = AsyncMock()
        mock_persister.list_states = AsyncMock(return_value=[])
        mock_persister.shutdown = AsyncMock()
        mock_persister.initialized = False  # Not initialized
        
        with patch('flowlib.agent.components.persistence.utils.create_state_persister',
                  return_value=mock_persister):
            
            result = await list_saved_states_metadata(persister_type="memory")
            
            assert result == []
            # Shutdown should not be called when not initialized
            mock_persister.shutdown.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_list_saved_states_metadata_various_persisters(self):
        """Test with different persister types and configurations."""
        mock_persister = Mock()
        mock_persister.initialize = AsyncMock()
        mock_persister.list_states = AsyncMock(return_value=[])
        mock_persister.shutdown = AsyncMock()
        mock_persister.initialized = True
        
        with patch('flowlib.agent.components.persistence.utils.create_state_persister',
                  return_value=mock_persister) as mock_factory:
            
            # Test file persister
            await list_saved_states_metadata(
                persister_type="file",
                base_path="/tmp/states"
            )
            mock_factory.assert_called_with(
                persister_type="file",
                base_path="/tmp/states"
            )
            
            # Test database persister
            await list_saved_states_metadata(
                persister_type="database",
                connection_string="sqlite:///test.db",
                table_name="agent_states"
            )
            mock_factory.assert_called_with(
                persister_type="database",
                connection_string="sqlite:///test.db",
                table_name="agent_states"
            )
    
    @pytest.mark.asyncio
    async def test_list_saved_states_metadata_logging(self, caplog):
        """Test that errors are properly logged."""
        with patch('flowlib.agent.components.persistence.utils.create_state_persister',
                  side_effect=Exception("Test error")):
            
            with caplog.at_level("ERROR"):
                result = await list_saved_states_metadata(persister_type="test")
                
                assert result == []
                assert "Failed to list saved states using test persister" in caplog.text
                assert "Test error" in caplog.text