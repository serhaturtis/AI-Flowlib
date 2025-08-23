"""Tests for remote agent configuration loader."""

import pytest
import tempfile
import os
import yaml
from unittest.mock import patch, mock_open

from flowlib.agent.runners.remote.config_loader import load_remote_config
from flowlib.agent.runners.remote.config_models import RemoteConfig, WorkerServiceConfig, CLIToolConfig


class TestLoadRemoteConfig:
    """Test load_remote_config function."""
    
    def test_load_config_from_file(self):
        """Test loading configuration from a YAML file."""
        config_data = {
            "worker": {
                "mq_provider_name": "file_test_mq",
                "state_persister_name": "file_test_persister",
                "task_queue": "file_test_tasks",
                "results_queue": "file_test_results",
                "base_agent_config_path": "/file/test/config.yaml"
            },
            "cli": {
                "mq_provider_name": "file_test_cli_mq",
                "task_queue": "file_test_cli_tasks"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_remote_config(temp_path)
            
            # Verify loaded configuration
            assert config.worker.mq_provider_name == "file_test_mq"
            assert config.worker.state_persister_name == "file_test_persister"
            assert config.worker.task_queue == "file_test_tasks"
            assert config.worker.results_queue == "file_test_results"
            assert config.worker.base_agent_config_path == "/file/test/config.yaml"
            
            assert config.cli.mq_provider_name == "file_test_cli_mq"
            assert config.cli.task_queue == "file_test_cli_tasks"
            
        finally:
            os.unlink(temp_path)
    
    def test_load_config_partial_file(self):
        """Test loading configuration from file with partial data."""
        config_data = {
            "worker": {
                "mq_provider_name": "partial_mq"
                # Other fields should use defaults
            }
            # CLI section missing, should use defaults
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_remote_config(temp_path)
            
            # Verify partial overrides
            assert config.worker.mq_provider_name == "partial_mq"
            assert config.worker.state_persister_name == "redis"  # Default
            assert config.worker.task_queue == "agent_tasks"  # Default
            
            # CLI should use all defaults
            assert config.cli.mq_provider_name == "rabbitmq"
            assert config.cli.task_queue == "agent_tasks"
            
        finally:
            os.unlink(temp_path)
    
    def test_load_config_empty_file(self):
        """Test loading configuration from empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name
        
        try:
            config = load_remote_config(temp_path)
            
            # Should use all defaults
            assert config.worker.mq_provider_name == "rabbitmq"
            assert config.worker.state_persister_name == "redis"
            assert config.cli.mq_provider_name == "rabbitmq"
            assert config.cli.task_queue == "agent_tasks"
            
        finally:
            os.unlink(temp_path)
    
    def test_load_config_nonexistent_file(self):
        """Test loading configuration when file doesn't exist."""
        nonexistent_path = "/tmp/nonexistent_config.yaml"
        
        config = load_remote_config(nonexistent_path)
        
        # Should return default configuration
        assert isinstance(config, RemoteConfig)
        assert config.worker.mq_provider_name == "rabbitmq"
        assert config.worker.state_persister_name == "redis"
        assert config.cli.mq_provider_name == "rabbitmq"
    
    def test_load_config_default_path(self):
        """Test loading configuration with default path."""
        # Mock os.path.exists to return False for default path
        with patch('os.path.exists', return_value=False):
            config = load_remote_config()  # No path provided
            
            # Should return default configuration
            assert isinstance(config, RemoteConfig)
            assert config.worker.mq_provider_name == "rabbitmq"
            assert config.cli.mq_provider_name == "rabbitmq"
    
    def test_load_config_invalid_yaml(self):
        """Test loading configuration from invalid YAML file."""
        invalid_yaml = "invalid: yaml: content: ["
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_path = f.name
        
        try:
            config = load_remote_config(temp_path)
            
            # Should fallback to defaults on parse error
            assert isinstance(config, RemoteConfig)
            assert config.worker.mq_provider_name == "rabbitmq"
            assert config.cli.mq_provider_name == "rabbitmq"
            
        finally:
            os.unlink(temp_path)
    
    def test_load_config_invalid_schema(self):
        """Test loading configuration with invalid schema."""
        invalid_schema_data = {
            "worker": {
                "mq_provider_name": 123,  # Should be string
                "invalid_field": "should_not_exist"
            },
            "cli": "this_should_be_object"  # Should be object
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_schema_data, f)
            temp_path = f.name
        
        try:
            config = load_remote_config(temp_path)
            
            # Should fallback to defaults on validation error
            assert isinstance(config, RemoteConfig)
            assert config.worker.mq_provider_name == "rabbitmq"
            assert config.cli.mq_provider_name == "rabbitmq"
            
        finally:
            os.unlink(temp_path)
    
    def test_load_config_permission_error(self):
        """Test loading configuration when file can't be read."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"worker": {"mq_provider_name": "test"}}, f)
            temp_path = f.name
        
        try:
            # Change permissions to make file unreadable
            os.chmod(temp_path, 0o000)
            
            config = load_remote_config(temp_path)
            
            # Should fallback to defaults on read error
            assert isinstance(config, RemoteConfig)
            assert config.worker.mq_provider_name == "rabbitmq"
            
        finally:
            # Restore permissions and clean up
            os.chmod(temp_path, 0o644)
            os.unlink(temp_path)
    
    @patch('builtins.open', mock_open(read_data="worker:\n  mq_provider_name: mocked_mq"))
    @patch('os.path.exists', return_value=True)
    def test_load_config_with_mocks(self, mock_exists):
        """Test loading configuration using mocks."""
        config = load_remote_config("mocked_path.yaml")
        
        assert config.worker.mq_provider_name == "mocked_mq"
        assert config.worker.state_persister_name == "redis"  # Default
        mock_exists.assert_called_once_with("mocked_path.yaml")
    
    def test_load_config_complex_structure(self):
        """Test loading configuration with complex nested structure."""
        complex_config = {
            "worker": {
                "mq_provider_name": "complex_worker_mq",
                "state_persister_name": "complex_worker_persister",
                "task_queue": "complex.worker.tasks",
                "results_queue": "complex.worker.results",
                "base_agent_config_path": "/etc/complex/worker/agent.yaml"
            },
            "cli": {
                "mq_provider_name": "complex_cli_mq",
                "task_queue": "complex.cli.tasks"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(complex_config, f)
            temp_path = f.name
        
        try:
            config = load_remote_config(temp_path)
            
            # Verify complex structure is preserved
            assert config.worker.mq_provider_name == "complex_worker_mq"
            assert config.worker.state_persister_name == "complex_worker_persister"
            assert config.worker.task_queue == "complex.worker.tasks"
            assert config.worker.results_queue == "complex.worker.results"
            assert config.worker.base_agent_config_path == "/etc/complex/worker/agent.yaml"
            
            assert config.cli.mq_provider_name == "complex_cli_mq"
            assert config.cli.task_queue == "complex.cli.tasks"
            
        finally:
            os.unlink(temp_path)
    
    def test_load_config_unicode_handling(self):
        """Test loading configuration with unicode characters."""
        unicode_config = {
            "worker": {
                "mq_provider_name": "测试_mq",
                "task_queue": "tâches_测试",
                "base_agent_config_path": "/配置/агент.yaml"
            },
            "cli": {
                "mq_provider_name": "🚀_cli_mq",
                "task_queue": "📋_tasks"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump(unicode_config, f, allow_unicode=True)
            temp_path = f.name
        
        try:
            config = load_remote_config(temp_path)
            
            # Verify unicode characters are preserved
            assert config.worker.mq_provider_name == "测试_mq"
            assert config.worker.task_queue == "tâches_测试"
            assert config.worker.base_agent_config_path == "/配置/агент.yaml"
            
            assert config.cli.mq_provider_name == "🚀_cli_mq"
            assert config.cli.task_queue == "📋_tasks"
            
        finally:
            os.unlink(temp_path)


class TestConfigLoaderIntegration:
    """Test configuration loader integration scenarios."""
    
    def test_environment_specific_configs(self):
        """Test loading different configs for different environments."""
        environments = ["development", "staging", "production"]
        
        for env in environments:
            config_data = {
                "worker": {
                    "mq_provider_name": f"{env}_rabbitmq",
                    "state_persister_name": f"{env}_redis",
                    "task_queue": f"{env}_agent_tasks",
                    "results_queue": f"{env}_agent_results"
                },
                "cli": {
                    "mq_provider_name": f"{env}_rabbitmq",
                    "task_queue": f"{env}_agent_tasks"
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config_data, f)
                temp_path = f.name
            
            try:
                config = load_remote_config(temp_path)
                
                assert config.worker.mq_provider_name == f"{env}_rabbitmq"
                assert config.worker.state_persister_name == f"{env}_redis"
                assert config.worker.task_queue == f"{env}_agent_tasks"
                assert config.cli.mq_provider_name == f"{env}_rabbitmq"
                
            finally:
                os.unlink(temp_path)
    
    def test_config_validation_edge_cases(self):
        """Test configuration validation with edge cases."""
        edge_cases = [
            # Empty sections
            {"worker": {}, "cli": {}},
            # Null values
            {"worker": None, "cli": None},
            # Extra unknown fields
            {
                "worker": {"unknown_field": "value"},
                "cli": {"another_unknown": "value"},
                "unknown_section": {"field": "value"}
            }
        ]
        
        for case_data in edge_cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(case_data, f)
                temp_path = f.name
            
            try:
                config = load_remote_config(temp_path)
                
                # Should always return valid config with defaults
                assert isinstance(config, RemoteConfig)
                assert isinstance(config.worker, WorkerServiceConfig)
                assert isinstance(config.cli, CLIToolConfig)
                
            finally:
                os.unlink(temp_path)