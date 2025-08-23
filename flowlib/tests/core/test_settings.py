"""Tests for core settings and configuration management."""

import pytest
import os
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from pydantic import ValidationError

from flowlib.core.settings.settings import (
    FlowlibSettings,
    DatabaseSettings,
    RedisSettings,
    VectorDBSettings,
    LLMSettings,
    SecuritySettings,
    MonitoringSettings,
    ConfigurationManager,
    create_settings,
    get_config,
    reload_config,
    validate_config,
    load_environment_config
)


class TestFlowlibSettings:
    """Test FlowlibSettings configuration class."""
    
    def test_default_configuration_values(self):
        """Test default configuration values."""
        settings = FlowlibSettings()
        
        assert settings.environment == "development"
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.default_llm_provider == "llamacpp"
    
    def test_environment_variable_override(self):
        """Test environment variable overrides."""
        with patch.dict(os.environ, {
            'FLOWLIB_ENV': 'production',
            'FLOWLIB_DEBUG': 'true',
            'FLOWLIB_LOG_LEVEL': 'DEBUG',
            'FLOWLIB_MAX_CONCURRENT_FLOWS': '20'
        }):
            settings = FlowlibSettings()
            
            assert settings.environment == "production"
            assert settings.debug is True
            assert settings.log_level == "DEBUG"
            assert settings.max_concurrent_flows == 20
    
    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid log levels
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            settings = FlowlibSettings(log_level=level)
            assert settings.log_level == level
        
        # Invalid log level
        with pytest.raises(ValidationError):
            FlowlibSettings(log_level='INVALID')
    
    def test_positive_integer_validation(self):
        """Test positive integer field validation."""
        # Valid positive values
        settings = FlowlibSettings(
            max_concurrent_flows=5,
            flow_timeout_seconds=60,
            max_memory_entries=1000
        )
        assert settings.max_concurrent_flows == 5
        assert settings.flow_timeout_seconds == 60
        assert settings.max_memory_entries == 1000
        
        # Invalid non-positive values
        with pytest.raises(ValidationError):
            FlowlibSettings(max_concurrent_flows=0)
        
        with pytest.raises(ValidationError):
            FlowlibSettings(flow_timeout_seconds=-1)
    
    def test_directory_creation(self):
        """Test that directories are created automatically."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "test_data"
            config_dir = Path(temp_dir) / "test_config"
            logs_dir = Path(temp_dir) / "test_logs"
            
            settings = FlowlibSettings(
                data_dir=data_dir,
                config_dir=config_dir,
                logs_dir=logs_dir
            )
            
            # Directories should be created
            assert data_dir.exists()
            assert config_dir.exists()
            assert logs_dir.exists()


class TestDatabaseSettings:
    """Test DatabaseSettings configuration class."""
    
    def test_default_configuration_values(self):
        """Test default database configuration values."""
        # Use isolated environment to avoid picking up system variables
        with patch.dict(os.environ, {'PATH': os.environ.get('PATH', '')}, clear=True):
            settings = DatabaseSettings()
            
            assert settings.host == "localhost"
            assert settings.port == 5432
            assert settings.name == "flowlib"
            assert settings.username == "flowlib"
            assert settings.password is None
            assert settings.pool_size == 5
            assert settings.max_overflow == 10
            assert settings.pool_timeout == 30
    
    def test_environment_prefix(self):
        """Test DB_ environment variable prefix."""
        with patch.dict(os.environ, {
            'DB_HOST': 'dbserver',
            'DB_PORT': '3306',
            'DB_NAME': 'myapp',
            'DB_USERNAME': 'dbuser',
            'DB_PASSWORD': 'secret123'
        }):
            settings = DatabaseSettings()
            
            assert settings.host == "dbserver"
            assert settings.port == 3306
            assert settings.name == "myapp"
            assert settings.username == "dbuser"
            assert settings.password == "secret123"


class TestLLMSettings:
    """Test LLMSettings configuration class."""
    
    def test_default_configuration_values(self):
        """Test default LLM configuration values."""
        settings = LLMSettings()
        
        assert settings.model_path is None
        assert settings.model_name == "default"
        assert settings.temperature == 0.7
        assert settings.max_tokens == 1000
        assert settings.top_p == 1.0
        assert settings.context_length == 4096
    
    def test_temperature_validation(self):
        """Test temperature value validation."""
        # Valid temperatures
        settings = LLMSettings(temperature=0.0)
        assert settings.temperature == 0.0
        
        settings = LLMSettings(temperature=2.0)
        assert settings.temperature == 2.0
        
        settings = LLMSettings(temperature=1.0)
        assert settings.temperature == 1.0
        
        # Invalid temperatures
        with pytest.raises(ValidationError):
            LLMSettings(temperature=-0.1)
        
        with pytest.raises(ValidationError):
            LLMSettings(temperature=2.1)
    
    def test_top_p_validation(self):
        """Test top_p value validation."""
        # Valid top_p values
        settings = LLMSettings(top_p=0.0)
        assert settings.top_p == 0.0
        
        settings = LLMSettings(top_p=1.0)
        assert settings.top_p == 1.0
        
        settings = LLMSettings(top_p=0.5)
        assert settings.top_p == 0.5
        
        # Invalid top_p values
        with pytest.raises(ValidationError):
            LLMSettings(top_p=-0.1)
        
        with pytest.raises(ValidationError):
            LLMSettings(top_p=1.1)


class TestSecuritySettings:
    """Test SecuritySettings configuration class."""
    
    def test_required_secret_key(self):
        """Test that secret key is required."""
        # Should work with secret key
        settings = SecuritySettings(secret_key="test-secret-key")
        assert settings.secret_key == "test-secret-key"
        
        # Should fail without secret key
        with pytest.raises(ValidationError):
            SecuritySettings()
    
    def test_default_configuration_values(self):
        """Test default security configuration values."""
        settings = SecuritySettings(secret_key="test-key")
        
        assert settings.encryption_algorithm == "AES-256-GCM"
        assert settings.password_hash_algorithm == "bcrypt"
        assert settings.session_timeout == 3600
        assert settings.max_login_attempts == 5
        assert settings.api_key_required is True
        assert settings.rate_limit_requests == 1000
        assert settings.rate_limit_window == 3600


class TestConfigurationManager:
    """Test ConfigurationManager functionality."""
    
    def test_initialization_default(self):
        """Test configuration manager initialization."""
        config_manager = ConfigurationManager()
        
        # Should have all expected settings sections
        assert 'flowlib' in config_manager._settings
        assert 'database' in config_manager._settings
        assert 'redis' in config_manager._settings
        assert 'vector_db' in config_manager._settings
        assert 'llm' in config_manager._settings
        assert 'monitoring' in config_manager._settings
    
    def test_get_settings(self):
        """Test getting settings by section."""
        config_manager = ConfigurationManager()
        
        flowlib_settings = config_manager.get_settings('flowlib')
        assert isinstance(flowlib_settings, FlowlibSettings)
        
        db_settings = config_manager.get_settings('database')
        assert isinstance(db_settings, DatabaseSettings)
    
    def test_get_settings_nonexistent(self):
        """Test getting non-existent settings section."""
        config_manager = ConfigurationManager()
        
        with pytest.raises(KeyError, match="Settings section 'nonexistent' not found"):
            config_manager.get_settings('nonexistent')
    
    def test_specific_getters(self):
        """Test specific settings getter methods."""
        config_manager = ConfigurationManager()
        
        assert isinstance(config_manager.get_flowlib_settings(), FlowlibSettings)
        assert isinstance(config_manager.get_database_settings(), DatabaseSettings)
        assert isinstance(config_manager.get_redis_settings(), RedisSettings)
        assert isinstance(config_manager.get_vector_db_settings(), VectorDBSettings)
        assert isinstance(config_manager.get_llm_settings(), LLMSettings)
        assert isinstance(config_manager.get_monitoring_settings(), MonitoringSettings)
    
    def test_to_dict(self):
        """Test exporting configuration to dictionary."""
        config_manager = ConfigurationManager()
        
        config_dict = config_manager.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'flowlib' in config_dict
        assert 'database' in config_dict
        
        # Check that nested structure is preserved
        assert isinstance(config_dict['flowlib'], dict)
        assert 'environment' in config_dict['flowlib']


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_settings(self):
        """Test create_settings utility function."""
        settings = create_settings(FlowlibSettings, environment='test', debug=True)
        
        assert isinstance(settings, FlowlibSettings)
        assert settings.environment == 'test'
        assert settings.debug is True
    
    def test_get_config(self):
        """Test get_config utility function."""
        config = get_config()
        
        assert isinstance(config, ConfigurationManager)
        assert hasattr(config, 'get_flowlib_settings')
    
    def test_reload_config(self):
        """Test reload_config utility function."""
        # Should not raise any exceptions
        reload_config()
    
    def test_validate_config(self):
        """Test validate_config utility function."""
        errors = validate_config()
        
        assert isinstance(errors, dict)