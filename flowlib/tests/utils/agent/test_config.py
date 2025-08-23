"""Comprehensive tests for agent configuration utilities."""

import pytest
import tempfile
import os
import yaml
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from flowlib.utils.agent.config import (
    initialize_resources_from_config,
    load_agent_config,
    initialize_providers_from_config
)
from flowlib.agent.models.config import AgentConfig
from flowlib.agent.core.errors import ConfigurationError


# Test fixtures
@pytest.fixture
def sample_agent_config():
    """Create a sample agent configuration."""
    return AgentConfig(
        name="test_agent",
        persona="Test agent for configuration testing",
        provider_name="llamacpp",
        provider_config={
            "llm": {
                "test_llm": {
                    "implementation": "llamacpp",
                    "settings": {
                        "model_path": "/test/model.bin",
                        "context_length": 2048
                    }
                }
            }
        },
        resource_config={
            "model": {
                "test_model": {
                    "provider": "test_provider",
                    "config": {
                        "param1": "value1",
                        "param2": 42
                    }
                }
            }
        }
    )


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file."""
    config_data = {
        "name": "test_agent",
        "persona": "Test agent for file loading",
        "provider_name": "llamacpp",
        "provider_config": {
            "llm": {
                "test_llm": {
                    "implementation": "llamacpp",
                    "settings": {
                        "model_path": "/test/model.bin"
                    }
                }
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    os.unlink(temp_file)


@pytest.fixture
def invalid_yaml_file():
    """Create a temporary file with invalid YAML."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content: [unclosed")
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    os.unlink(temp_file)


class TestInitializeResourcesFromConfig:
    """Test resource initialization from configuration."""
    
    def test_initialize_resources_success(self, sample_agent_config):
        """Test successful resource initialization."""
        with patch('flowlib.utils.agent.config.resource_registry') as mock_registry:
            mock_registry.contains.return_value = False
            
            initialize_resources_from_config(sample_agent_config)
            
            # Verify resource registration
            mock_registry.register.assert_called_once()
            call_args = mock_registry.register.call_args
            assert call_args[1]['name'] == 'test_model'
            assert call_args[1]['resource_type'] == 'model'
            assert call_args[1]['obj'].provider_type == 'test_provider'
    
    def test_initialize_resources_no_config(self):
        """Test initialization with no resource configuration."""
        config = AgentConfig(name="test", persona="test", provider_name="llamacpp")
        
        with patch('flowlib.utils.agent.config.resource_registry') as mock_registry:
            initialize_resources_from_config(config)
            
            # Should not attempt to register anything
            mock_registry.register.assert_not_called()
    
    def test_initialize_resources_resource_exists(self, sample_agent_config):
        """Test initialization when resource already exists."""
        with patch('flowlib.utils.agent.config.resource_registry') as mock_registry:
            mock_registry.contains.return_value = True
            
            initialize_resources_from_config(sample_agent_config)
            
            # Should not register existing resource
            mock_registry.register.assert_not_called()
    
    def test_initialize_resources_invalid_resource_info(self):
        """Test initialization with invalid resource info."""
        config = AgentConfig(
            name="test",
            persona="test",
            provider_name="llamacpp",
            resource_config={
                "model": {
                    "invalid_resource": "not_a_dict"  # Should be dict
                }
            }
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            initialize_resources_from_config(config)
        
        assert "must be a dict" in str(exc_info.value)
        assert "invalid_resource" in str(exc_info.value)
    
    def test_initialize_resources_missing_provider(self):
        """Test initialization with missing provider field."""
        config = AgentConfig(
            name="test",
            persona="test",
            provider_name="llamacpp",
            resource_config={
                "model": {
                    "test_model": {
                        "config": {"param": "value"}
                        # Missing 'provider' field
                    }
                }
            }
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            initialize_resources_from_config(config)
        
        assert "missing required 'provider'" in str(exc_info.value)
    
    def test_initialize_resources_missing_config(self):
        """Test initialization with missing config field."""
        config = AgentConfig(
            name="test",
            persona="test",
            provider_name="llamacpp",
            resource_config={
                "model": {
                    "test_model": {
                        "provider": "test_provider"
                        # Missing 'config' field
                    }
                }
            }
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            initialize_resources_from_config(config)
        
        assert "missing required" in str(exc_info.value)
        assert "'config'" in str(exc_info.value)


class TestLoadAgentConfig:
    """Test agent configuration loading from files."""
    
    def test_load_agent_config_success(self, temp_config_file):
        """Test successful configuration loading."""
        config = load_agent_config(temp_config_file)
        
        assert isinstance(config, AgentConfig)
        assert config.name == "test_agent"
        assert config.persona == "Test agent for file loading"
        assert "llm" in config.provider_config
    
    def test_load_agent_config_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_agent_config("/non/existent/file.yaml")
        
        assert "Configuration file not found" in str(exc_info.value)
    
    def test_load_agent_config_invalid_yaml(self, invalid_yaml_file):
        """Test loading invalid YAML file."""
        with pytest.raises(ConfigurationError) as exc_info:
            load_agent_config(invalid_yaml_file)
        
        assert "Error parsing YAML" in str(exc_info.value)
    
    def test_load_agent_config_empty_file(self):
        """Test loading empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Write nothing to create empty file
            temp_file = f.name
        
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                load_agent_config(temp_file)
            
            assert "Failed to parse YAML or file is empty" in str(exc_info.value)
        finally:
            os.unlink(temp_file)
    
    def test_load_agent_config_invalid_structure(self):
        """Test loading file with invalid configuration structure."""
        invalid_config = {
            "name": "test",
            # Missing required 'persona' and 'provider_name' fields
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_file = f.name
        
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                load_agent_config(temp_file)
            
            assert "Error validating configuration" in str(exc_info.value)
        finally:
            os.unlink(temp_file)
    
    def test_load_agent_config_non_dict_content(self):
        """Test loading file with non-dictionary content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump("not_a_dict", f)
            temp_file = f.name
        
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                load_agent_config(temp_file)
            
            assert "Failed to parse YAML or file is empty" in str(exc_info.value)
        finally:
            os.unlink(temp_file)


class TestInitializeProvidersFromConfig:
    """Test provider initialization from configuration."""
    
    @pytest.mark.asyncio
    async def test_initialize_providers_success(self, sample_agent_config):
        """Test successful provider initialization."""
        mock_settings_class = Mock()
        mock_settings_instance = Mock()
        mock_settings_class.return_value = mock_settings_instance
        
        with patch('flowlib.utils.agent.config.provider_registry') as mock_registry:
            with patch('flowlib.utils.agent.config.create_and_initialize_provider') as mock_create:
                # Setup registry metadata
                mock_registry._factory_metadata = {
                    ("llm", "llamacpp"): {"settings_class": mock_settings_class}
                }
                
                await initialize_providers_from_config(sample_agent_config)
                
                # Verify settings instantiation
                mock_settings_class.assert_called_once_with(
                    model_path="/test/model.bin",
                    context_length=2048
                )
                
                # Verify provider creation
                mock_create.assert_called_once_with(
                    provider_type="llm",
                    name="test_llm",
                    implementation="llamacpp",
                    register=True,
                    settings=mock_settings_instance
                )
    
    @pytest.mark.asyncio
    async def test_initialize_providers_no_config(self):
        """Test initialization with no provider configuration."""
        config = AgentConfig(name="test", persona="test", provider_name="llamacpp")
        
        with patch('flowlib.utils.agent.config.create_and_initialize_provider') as mock_create:
            await initialize_providers_from_config(config)
            
            # Should not attempt to create any providers
            mock_create.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_initialize_providers_missing_implementation(self):
        """Test initialization with missing implementation field."""
        config = AgentConfig(
            name="test",
            persona="test",
            provider_name="llamacpp",
            provider_config={
                "llm": {
                    "test_llm": {
                        "settings": {"param": "value"}
                        # Missing 'implementation' field
                    }
                }
            }
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            await initialize_providers_from_config(config)
        
        assert "missing required 'implementation'" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_initialize_providers_missing_settings(self):
        """Test initialization with missing settings field."""
        config = AgentConfig(
            name="test",
            persona="test",
            provider_name="llamacpp",
            provider_config={
                "llm": {
                    "test_llm": {
                        "implementation": "llamacpp"
                        # Missing 'settings' field
                    }
                }
            }
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            await initialize_providers_from_config(config)
        
        assert "missing required" in str(exc_info.value)
        assert "'settings'" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_initialize_providers_no_settings_class(self):
        """Test initialization when no settings class is registered."""
        config = AgentConfig(
            name="test",
            persona="test",
            provider_name="llamacpp",
            provider_config={
                "llm": {
                    "test_llm": {
                        "implementation": "llamacpp",
                        "settings": {"param": "value"}
                    }
                }
            }
        )
        
        with patch('flowlib.utils.agent.config.provider_registry') as mock_registry:
            # No metadata for this provider type/implementation
            mock_registry._factory_metadata = {}
            
            with pytest.raises(ConfigurationError) as exc_info:
                await initialize_providers_from_config(config)
            
            assert "No registered settings class" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_initialize_providers_settings_instantiation_error(self):
        """Test initialization when settings instantiation fails."""
        config = AgentConfig(
            name="test",
            persona="test",
            provider_name="llamacpp",
            provider_config={
                "llm": {
                    "test_llm": {
                        "implementation": "llamacpp",
                        "settings": {"invalid_param": "value"}
                    }
                }
            }
        )
        
        mock_settings_class = Mock()
        mock_settings_class.side_effect = ValueError("Invalid settings")
        
        with patch('flowlib.utils.agent.config.provider_registry') as mock_registry:
            mock_registry._factory_metadata = {
                ("llm", "llamacpp"): {"settings_class": mock_settings_class}
            }
            
            with pytest.raises(ConfigurationError) as exc_info:
                await initialize_providers_from_config(config)
            
            assert "Failed to instantiate settings" in str(exc_info.value)
            assert "Invalid settings" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_initialize_providers_creation_error(self, sample_agent_config):
        """Test initialization when provider creation fails."""
        mock_settings_class = Mock()
        mock_settings_instance = Mock()
        mock_settings_class.return_value = mock_settings_instance
        
        with patch('flowlib.utils.agent.config.provider_registry') as mock_registry:
            with patch('flowlib.utils.agent.config.create_and_initialize_provider') as mock_create:
                mock_registry._factory_metadata = {
                    ("llm", "llamacpp"): {"settings_class": mock_settings_class}
                }
                mock_create.side_effect = Exception("Provider creation failed")
                
                with pytest.raises(ConfigurationError) as exc_info:
                    await initialize_providers_from_config(sample_agent_config)
                
                assert "Failed to initialize provider" in str(exc_info.value)
                assert "Provider creation failed" in str(exc_info.value)


class TestConfigurationErrorHandling:
    """Test error handling and edge cases."""
    
    def test_configuration_error_with_config_key(self):
        """Test ConfigurationError includes config key information."""
        config = AgentConfig(
            name="test",
            persona="test",
            provider_name="llamacpp",
            resource_config={
                "model": {
                    "bad_resource": "invalid"
                }
            }
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            initialize_resources_from_config(config)
        
        error = exc_info.value
        assert error.config_key == "resource_config.model.bad_resource"
    
    @pytest.mark.asyncio
    async def test_provider_error_with_config_key(self):
        """Test provider configuration error includes config key."""
        config = AgentConfig(
            name="test",
            persona="test",
            provider_name="llamacpp",
            provider_config={
                "llm": {
                    "bad_provider": {
                        "implementation": "unknown"
                        # Missing settings
                    }
                }
            }
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            await initialize_providers_from_config(config)
        
        error = exc_info.value
        assert "provider_config.llm.bad_provider" in error.config_key


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple functions."""
    
    def test_full_configuration_loading_and_initialization(self):
        """Test complete configuration workflow."""
        config_data = {
            "name": "integration_test_agent",
            "persona": "Agent for integration testing",
            "provider_name": "llamacpp",
            "resource_config": {
                "model": {
                    "test_model": {
                        "provider": "test_provider",
                        "config": {"param1": "value1"}
                    }
                }
            },
            "provider_config": {
                "llm": {
                    "test_llm": {
                        "implementation": "llamacpp",
                        "settings": {"model_path": "/test/path"}
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name
        
        try:
            # Load configuration
            config = load_agent_config(temp_file)
            assert config.name == "integration_test_agent"
            
            # Initialize resources
            with patch('flowlib.utils.agent.config.resource_registry') as mock_resource_registry:
                mock_resource_registry.contains.return_value = False
                initialize_resources_from_config(config)
                mock_resource_registry.register.assert_called_once()
            
        finally:
            os.unlink(temp_file)
    
    def test_configuration_loading_with_unicode_content(self):
        """Test configuration loading with Unicode content."""
        config_data = {
            "name": "unicode_test_agent",
            "persona": "Agent with Unicode: 中文, Émojis 🚀, and symbols ñ",
            "provider_name": "llamacpp",
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump(config_data, f, allow_unicode=True)
            temp_file = f.name
        
        try:
            config = load_agent_config(temp_file)
            assert "中文" in config.persona
            assert "🚀" in config.persona
            assert "ñ" in config.persona
        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])