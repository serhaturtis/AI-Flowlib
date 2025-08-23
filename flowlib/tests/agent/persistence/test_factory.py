"""Comprehensive tests for agent persistence factory module."""

import pytest
import logging
from typing import Optional, Union
from unittest.mock import Mock, patch, MagicMock

from flowlib.agent.components.persistence.factory import create_state_persister
from flowlib.agent.components.persistence.file import FileStatePersister, FileStatePersisterSettings
from flowlib.agent.components.persistence.provider import ProviderStatePersister
from flowlib.agent.models.config import StatePersistenceConfig
from flowlib.agent.core.errors import StatePersistenceError


class TestCreateStatePersister:
    """Test create_state_persister factory function."""
    
    def test_create_file_persister_default(self):
        """Test creating file persister with default parameters."""
        with patch('flowlib.agent.components.persistence.factory.FileStatePersister') as mock_file_persister:
            with patch('flowlib.agent.components.persistence.factory.FileStatePersisterSettings') as mock_settings:
                mock_instance = Mock()
                mock_file_persister.return_value = mock_instance
                mock_settings_instance = Mock()
                mock_settings.return_value = mock_settings_instance
                
                result = create_state_persister(persister_type="file")
                
                assert result == mock_instance
                mock_settings.assert_called_once_with(directory="./states")
                mock_file_persister.assert_called_once_with(settings=mock_settings_instance)
    
    def test_create_file_persister_with_directory(self):
        """Test creating file persister with custom directory."""
        with patch('flowlib.agent.components.persistence.factory.FileStatePersister') as mock_file_persister:
            with patch('flowlib.agent.components.persistence.factory.FileStatePersisterSettings') as mock_settings:
                mock_instance = Mock()
                mock_file_persister.return_value = mock_instance
                
                result = create_state_persister(
                    persister_type="file",
                    directory="/custom/path"
                )
                
                assert result == mock_instance
                mock_settings.assert_called_once_with(directory="/custom/path")
    
    def test_create_file_persister_with_base_path(self):
        """Test creating file persister with base_path parameter."""
        with patch('flowlib.agent.components.persistence.factory.FileStatePersister') as mock_file_persister:
            with patch('flowlib.agent.components.persistence.factory.FileStatePersisterSettings') as mock_settings:
                mock_instance = Mock()
                mock_file_persister.return_value = mock_instance
                
                result = create_state_persister(
                    persister_type="file",
                    base_path="/base/path"
                )
                
                assert result == mock_instance
                mock_settings.assert_called_once_with(directory="/base/path")
    
    def test_create_file_persister_base_path_priority(self):
        """Test that base_path takes priority over directory."""
        with patch('flowlib.agent.components.persistence.factory.FileStatePersister') as mock_file_persister:
            with patch('flowlib.agent.components.persistence.factory.FileStatePersisterSettings') as mock_settings:
                mock_instance = Mock()
                mock_file_persister.return_value = mock_instance
                
                result = create_state_persister(
                    persister_type="file",
                    base_path="/base/path",
                    directory="/ignored/path"
                )
                
                assert result == mock_instance
                mock_settings.assert_called_once_with(directory="/base/path")
    
    def test_create_file_persister_missing_directory_error(self):
        """Test error when directory is missing for file persister."""
        with pytest.raises(StatePersistenceError) as exc_info:
            create_state_persister(
                persister_type="file",
                directory=None
            )
        
        assert "Missing directory for file state persister" in str(exc_info.value)
        assert exc_info.value.operation == "create"
    
    def test_create_file_persister_empty_directory_error(self):
        """Test error when directory is empty string."""
        with pytest.raises(StatePersistenceError) as exc_info:
            create_state_persister(
                persister_type="file",
                directory=""
            )
        
        assert "Missing directory for file state persister" in str(exc_info.value)
    
    def test_create_provider_persister_with_provider_name(self):
        """Test creating provider persister with provider_name."""
        with patch('flowlib.agent.components.persistence.factory.ProviderStatePersister') as mock_provider_persister:
            mock_instance = Mock()
            mock_provider_persister.return_value = mock_instance
            
            result = create_state_persister(
                persister_type="provider",
                provider_name="test_provider"
            )
            
            assert result == mock_instance
            mock_provider_persister.assert_called_once_with(provider_name="test_provider")
    
    def test_create_provider_persister_with_provider_id(self):
        """Test creating provider persister with provider_id."""
        with patch('flowlib.agent.components.persistence.factory.ProviderStatePersister') as mock_provider_persister:
            mock_instance = Mock()
            mock_provider_persister.return_value = mock_instance
            
            result = create_state_persister(
                persister_type="provider",
                provider_id="test_provider_id"
            )
            
            assert result == mock_instance
            mock_provider_persister.assert_called_once_with(provider_name="test_provider_id")
    
    def test_create_provider_persister_provider_id_priority(self):
        """Test that provider_id takes priority over provider_name."""
        with patch('flowlib.agent.components.persistence.factory.ProviderStatePersister') as mock_provider_persister:
            mock_instance = Mock()
            mock_provider_persister.return_value = mock_instance
            
            result = create_state_persister(
                persister_type="provider",
                provider_id="priority_provider",
                provider_name="ignored_provider"
            )
            
            assert result == mock_instance
            mock_provider_persister.assert_called_once_with(provider_name="priority_provider")
    
    def test_create_provider_persister_missing_provider_error(self):
        """Test error when provider name is missing."""
        with pytest.raises(StatePersistenceError) as exc_info:
            create_state_persister(persister_type="provider")
        
        assert "Provider name is required for provider persister" in str(exc_info.value)
        assert exc_info.value.operation == "create"
    
    def test_create_provider_persister_empty_provider_error(self):
        """Test error when provider name is empty."""
        with pytest.raises(StatePersistenceError) as exc_info:
            create_state_persister(
                persister_type="provider",
                provider_name=""
            )
        
        assert "Provider name is required for provider persister" in str(exc_info.value)
    
    def test_create_invalid_persister_type(self):
        """Test creating persister with invalid type."""
        result = create_state_persister(persister_type="invalid")
        
        assert result is None
    
    def test_create_persister_with_config_file_type(self):
        """Test creating persister with StatePersistenceConfig for file type."""
        config = Mock(spec=StatePersistenceConfig)
        config.persistence_type = "file"
        config.base_path = "/config/path"
        
        with patch('flowlib.agent.components.persistence.factory.FileStatePersister') as mock_file_persister:
            with patch('flowlib.agent.components.persistence.factory.FileStatePersisterSettings') as mock_settings:
                mock_instance = Mock()
                mock_file_persister.return_value = mock_instance
                mock_settings_instance = Mock()
                mock_settings.return_value = mock_settings_instance
                
                result = create_state_persister(config=config)
                
                assert result == mock_instance
                mock_settings.assert_called_once_with(directory="/config/path")
                mock_file_persister.assert_called_once_with(settings=mock_settings_instance)
    
    def test_create_persister_with_config_provider_type(self):
        """Test creating persister with StatePersistenceConfig for provider type."""
        config = Mock(spec=StatePersistenceConfig)
        config.persistence_type = "provider"
        config.provider_id = "config_provider"
        
        with patch('flowlib.agent.components.persistence.factory.ProviderStatePersister') as mock_provider_persister:
            mock_instance = Mock()
            mock_provider_persister.return_value = mock_instance
            
            result = create_state_persister(config=config)
            
            assert result == mock_instance
            mock_provider_persister.assert_called_once_with(provider_name="config_provider")
    
    def test_create_persister_with_config_file_missing_base_path(self):
        """Test error when config file type is missing base_path."""
        config = Mock(spec=StatePersistenceConfig)
        config.persistence_type = "file"
        config.base_path = None
        
        with pytest.raises(StatePersistenceError) as exc_info:
            create_state_persister(config=config)
        
        assert "Missing base_path for file state persister" in str(exc_info.value)
        assert exc_info.value.operation == "create"
    
    def test_create_persister_with_config_file_empty_base_path(self):
        """Test error when config file type has empty base_path."""
        config = Mock(spec=StatePersistenceConfig)
        config.persistence_type = "file"
        config.base_path = ""
        
        with pytest.raises(StatePersistenceError) as exc_info:
            create_state_persister(config=config)
        
        assert "Missing base_path for file state persister" in str(exc_info.value)
    
    def test_create_persister_config_overrides_type_parameter(self):
        """Test that config type overrides the persister_type parameter."""
        config = Mock(spec=StatePersistenceConfig)
        config.persistence_type = "provider"
        config.provider_id = "config_provider"
        
        with patch('flowlib.agent.components.persistence.factory.ProviderStatePersister') as mock_provider_persister:
            mock_instance = Mock()
            mock_provider_persister.return_value = mock_instance
            
            # persister_type="file" should be ignored when config is provided
            result = create_state_persister(
                persister_type="file",
                config=config
            )
            
            assert result == mock_instance
            mock_provider_persister.assert_called_once_with(provider_name="config_provider")
    
    def test_create_persister_file_creation_error(self):
        """Test error handling when file persister creation fails."""
        with patch('flowlib.agent.components.persistence.factory.FileStatePersister') as mock_file_persister:
            with patch('flowlib.agent.components.persistence.factory.FileStatePersisterSettings') as mock_settings:
                mock_settings.side_effect = Exception("Settings creation failed")
                
                with pytest.raises(StatePersistenceError) as exc_info:
                    create_state_persister(persister_type="file")
                
                assert "Error creating state persister" in str(exc_info.value)
                assert exc_info.value.operation == "create"
                assert exc_info.value.cause is not None
    
    def test_create_persister_provider_creation_error(self):
        """Test error handling when provider persister creation fails."""
        with patch('flowlib.agent.components.persistence.factory.ProviderStatePersister') as mock_provider_persister:
            mock_provider_persister.side_effect = Exception("Provider creation failed")
            
            with pytest.raises(StatePersistenceError) as exc_info:
                create_state_persister(
                    persister_type="provider",
                    provider_name="test_provider"
                )
            
            assert "Error creating state persister" in str(exc_info.value)
            assert exc_info.value.operation == "create"
            assert exc_info.value.cause is not None
    
    def test_create_persister_default_type(self):
        """Test creating persister with default type (file)."""
        with patch('flowlib.agent.components.persistence.factory.FileStatePersister') as mock_file_persister:
            with patch('flowlib.agent.components.persistence.factory.FileStatePersisterSettings'):
                mock_instance = Mock()
                mock_file_persister.return_value = mock_instance
                
                result = create_state_persister()  # No persister_type specified
                
                assert result == mock_instance
    
    def test_create_persister_logging_invalid_type(self):
        """Test that invalid persister type is logged."""
        with patch('flowlib.agent.components.persistence.factory.logger') as mock_logger:
            result = create_state_persister(persister_type="invalid_type")
            
            assert result is None
            mock_logger.warning.assert_called_once_with("Invalid persister type: invalid_type")
    
    def test_create_persister_logging_error(self):
        """Test that errors are logged."""
        with patch('flowlib.agent.components.persistence.factory.FileStatePersister') as mock_file_persister:
            with patch('flowlib.agent.components.persistence.factory.logger') as mock_logger:
                mock_file_persister.side_effect = Exception("Test error")
                
                with pytest.raises(StatePersistenceError):
                    create_state_persister(persister_type="file")
                
                mock_logger.error.assert_called_once()
                assert "Error creating state persister" in mock_logger.error.call_args[0][0]
    
    def test_create_persister_with_extra_kwargs(self):
        """Test creating persister with extra keyword arguments."""
        with patch('flowlib.agent.components.persistence.factory.FileStatePersister') as mock_file_persister:
            with patch('flowlib.agent.components.persistence.factory.FileStatePersisterSettings') as mock_settings:
                mock_instance = Mock()
                mock_file_persister.return_value = mock_instance
                
                # Extra kwargs should be ignored for file persister
                result = create_state_persister(
                    persister_type="file",
                    directory="/test/path",
                    extra_param="ignored",
                    another_param=123
                )
                
                assert result == mock_instance
                mock_settings.assert_called_once_with(directory="/test/path")
    
    def test_create_persister_return_type_annotations(self):
        """Test that the function returns correct types."""
        with patch('flowlib.agent.components.persistence.factory.FileStatePersister') as mock_file_persister:
            with patch('flowlib.agent.components.persistence.factory.FileStatePersisterSettings'):
                mock_file_instance = Mock(spec=FileStatePersister)
                mock_file_persister.return_value = mock_file_instance
                
                result = create_state_persister(persister_type="file")
                assert isinstance(result, type(mock_file_instance))
        
        with patch('flowlib.agent.components.persistence.factory.ProviderStatePersister') as mock_provider_persister:
            mock_provider_instance = Mock(spec=ProviderStatePersister)
            mock_provider_persister.return_value = mock_provider_instance
            
            result = create_state_persister(
                persister_type="provider",
                provider_name="test"
            )
            assert isinstance(result, type(mock_provider_instance))
        
        # Invalid type returns None
        result = create_state_persister(persister_type="invalid")
        assert result is None


class TestFactoryIntegration:
    """Test integration aspects of the factory."""
    
    def test_factory_with_real_config_structure(self):
        """Test factory with realistic config structure."""
        # Create a realistic config object
        from flowlib.agent.models.config import StatePersistenceConfig
        config = StatePersistenceConfig(
            persistence_type="file",
            base_path="/app/states",
            provider_id=None
        )
        
        with patch('flowlib.agent.components.persistence.factory.FileStatePersister') as mock_file_persister:
            with patch('flowlib.agent.components.persistence.factory.FileStatePersisterSettings') as mock_settings:
                mock_instance = Mock()
                mock_file_persister.return_value = mock_instance
                
                result = create_state_persister(config=config)
                
                assert result == mock_instance
                mock_settings.assert_called_once_with(directory="/app/states")
    
    def test_factory_parameter_combinations(self):
        """Test various parameter combinations."""
        test_cases = [
            # File persister cases
            {
                "params": {"persister_type": "file", "directory": "/test1"},
                "expected_directory": "/test1"
            },
            {
                "params": {"persister_type": "file", "base_path": "/test2"},
                "expected_directory": "/test2"
            },
            {
                "params": {"persister_type": "file", "base_path": "/priority", "directory": "/ignored"},
                "expected_directory": "/priority"
            },
            {
                "params": {"persister_type": "file"},
                "expected_directory": "./states"
            }
        ]
        
        for case in test_cases:
            with patch('flowlib.agent.components.persistence.factory.FileStatePersister') as mock_file_persister:
                with patch('flowlib.agent.components.persistence.factory.FileStatePersisterSettings') as mock_settings:
                    mock_instance = Mock()
                    mock_file_persister.return_value = mock_instance
                    
                    result = create_state_persister(**case["params"])
                    
                    assert result == mock_instance
                    mock_settings.assert_called_once_with(directory=case["expected_directory"])
    
    def test_factory_error_propagation(self):
        """Test that errors are properly propagated with context."""
        original_error = ValueError("Original error message")
        
        with patch('flowlib.agent.components.persistence.factory.FileStatePersister') as mock_file_persister:
            mock_file_persister.side_effect = original_error
            
            with pytest.raises(StatePersistenceError) as exc_info:
                create_state_persister(persister_type="file")
            
            # Check that the original error is preserved
            assert exc_info.value.cause == original_error
            assert exc_info.value.operation == "create"
            assert "Error creating state persister" in exc_info.value.message
    
    def test_factory_config_validation(self):
        """Test config object validation."""
        # Test with non-StatePersistenceConfig object
        invalid_config = {"not": "a config object"}
        
        # Should not trigger config path since it's not a StatePersistenceConfig instance
        with patch('flowlib.agent.components.persistence.factory.FileStatePersister') as mock_file_persister:
            with patch('flowlib.agent.components.persistence.factory.FileStatePersisterSettings'):
                mock_instance = Mock()
                mock_file_persister.return_value = mock_instance
                
                result = create_state_persister(
                    persister_type="file",
                    config=invalid_config
                )
                
                assert result == mock_instance
    
    def test_factory_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with None values
        with pytest.raises(StatePersistenceError):
            create_state_persister(
                persister_type="file",
                directory=None,
                base_path=None
            )
        
        # Test with empty string provider name
        with pytest.raises(StatePersistenceError):
            create_state_persister(
                persister_type="provider",
                provider_name="",
                provider_id=""
            )
        
        # Test case sensitivity
        result = create_state_persister(persister_type="FILE")
        assert result is None
        
        result = create_state_persister(persister_type="PROVIDER", provider_name="test")
        assert result is None
    
    def test_factory_documentation_examples(self):
        """Test examples that would be in documentation."""
        # Example 1: Simple file persister
        with patch('flowlib.agent.components.persistence.factory.FileStatePersister') as mock_file_persister:
            with patch('flowlib.agent.components.persistence.factory.FileStatePersisterSettings'):
                mock_instance = Mock()
                mock_file_persister.return_value = mock_instance
                
                persister = create_state_persister(
                    persister_type="file",
                    directory="./my_states"
                )
                assert persister == mock_instance
        
        # Example 2: Provider persister
        with patch('flowlib.agent.components.persistence.factory.ProviderStatePersister') as mock_provider_persister:
            mock_instance = Mock()
            mock_provider_persister.return_value = mock_instance
            
            persister = create_state_persister(
                persister_type="provider",
                provider_name="redis_provider"
            )
            assert persister == mock_instance
        
        # Example 3: Config-based creation
        config = Mock(spec=StatePersistenceConfig)
        config.persistence_type = "file"
        config.base_path = "/app/data/states"
        
        with patch('flowlib.agent.components.persistence.factory.FileStatePersister') as mock_file_persister:
            with patch('flowlib.agent.components.persistence.factory.FileStatePersisterSettings'):
                mock_instance = Mock()
                mock_file_persister.return_value = mock_instance
                
                persister = create_state_persister(config=config)
                assert persister == mock_instance