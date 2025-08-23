"""Tests for remote agent configuration models."""

import pytest
from flowlib.agent.runners.remote.config_models import (
    WorkerServiceConfig,
    CLIToolConfig,
    RemoteConfig
)


class TestWorkerServiceConfig:
    """Test WorkerServiceConfig model."""
    
    def test_worker_config_defaults(self):
        """Test default worker configuration values."""
        config = WorkerServiceConfig()
        
        assert config.mq_provider_name == "rabbitmq"
        assert config.state_persister_name == "redis"
        assert config.task_queue == "agent_tasks"
        assert config.results_queue == "agent_results"
        assert config.base_agent_config_path == "./agent_config.yaml"
    
    def test_worker_config_custom_values(self):
        """Test worker configuration with custom values."""
        config = WorkerServiceConfig(
            mq_provider_name="kafka_cluster",
            state_persister_name="mongodb_store",
            task_queue="custom_tasks",
            results_queue="custom_results",
            base_agent_config_path="/etc/agent/config.yaml"
        )
        
        assert config.mq_provider_name == "kafka_cluster"
        assert config.state_persister_name == "mongodb_store"
        assert config.task_queue == "custom_tasks"
        assert config.results_queue == "custom_results"
        assert config.base_agent_config_path == "/etc/agent/config.yaml"
    
    def test_worker_config_partial_override(self):
        """Test worker configuration with partial field overrides."""
        config = WorkerServiceConfig(
            task_queue="special_tasks",
            results_queue="special_results"
        )
        
        # Overridden fields
        assert config.task_queue == "special_tasks"
        assert config.results_queue == "special_results"
        
        # Default fields
        assert config.mq_provider_name == "rabbitmq"
        assert config.state_persister_name == "redis"
        assert config.base_agent_config_path == "./agent_config.yaml"
    
    def test_worker_config_serialization(self):
        """Test worker configuration serialization/deserialization."""
        original = WorkerServiceConfig(
            mq_provider_name="test_mq",
            state_persister_name="test_persister",
            task_queue="test_queue",
            results_queue="test_results",
            base_agent_config_path="/test/config.yaml"
        )
        
        # Test model_dump
        data = original.model_dump()
        assert data["mq_provider_name"] == "test_mq"
        assert data["task_queue"] == "test_queue"
        
        # Test model_validate
        restored = WorkerServiceConfig.model_validate(data)
        assert restored.mq_provider_name == original.mq_provider_name
        assert restored.state_persister_name == original.state_persister_name
        assert restored.task_queue == original.task_queue
        assert restored.results_queue == original.results_queue
        assert restored.base_agent_config_path == original.base_agent_config_path
    
    def test_worker_config_validation(self):
        """Test worker configuration field validation."""
        # Valid configuration
        config = WorkerServiceConfig(mq_provider_name="valid_name")
        assert config.mq_provider_name == "valid_name"
        
        # Test with None values (should use defaults)
        config = WorkerServiceConfig()
        assert config.mq_provider_name is not None
        assert config.state_persister_name is not None


class TestCLIToolConfig:
    """Test CLIToolConfig model."""
    
    def test_cli_config_defaults(self):
        """Test default CLI configuration values."""
        config = CLIToolConfig()
        
        assert config.mq_provider_name == "rabbitmq"
        assert config.task_queue == "agent_tasks"
    
    def test_cli_config_custom_values(self):
        """Test CLI configuration with custom values."""
        config = CLIToolConfig(
            mq_provider_name="kafka_producer",
            task_queue="priority_tasks"
        )
        
        assert config.mq_provider_name == "kafka_producer"
        assert config.task_queue == "priority_tasks"
    
    def test_cli_config_serialization(self):
        """Test CLI configuration serialization/deserialization."""
        original = CLIToolConfig(
            mq_provider_name="cli_mq",
            task_queue="cli_queue"
        )
        
        # Test JSON serialization
        json_str = original.model_dump_json()
        assert "cli_mq" in json_str
        assert "cli_queue" in json_str
        
        # Test JSON deserialization
        restored = CLIToolConfig.model_validate_json(json_str)
        assert restored.mq_provider_name == original.mq_provider_name
        assert restored.task_queue == original.task_queue
    
    def test_cli_config_minimal(self):
        """Test CLI configuration with minimal setup."""
        config = CLIToolConfig(mq_provider_name="minimal_mq")
        
        assert config.mq_provider_name == "minimal_mq"
        assert config.task_queue == "agent_tasks"  # Default value


class TestRemoteConfig:
    """Test RemoteConfig root model."""
    
    def test_remote_config_defaults(self):
        """Test default remote configuration."""
        config = RemoteConfig()
        
        # Test worker defaults
        assert config.worker.mq_provider_name == "rabbitmq"
        assert config.worker.state_persister_name == "redis"
        assert config.worker.task_queue == "agent_tasks"
        assert config.worker.results_queue == "agent_results"
        
        # Test CLI defaults
        assert config.cli.mq_provider_name == "rabbitmq"
        assert config.cli.task_queue == "agent_tasks"
    
    def test_remote_config_custom_sections(self):
        """Test remote configuration with custom sections."""
        worker_config = WorkerServiceConfig(
            mq_provider_name="worker_mq",
            task_queue="worker_tasks"
        )
        cli_config = CLIToolConfig(
            mq_provider_name="cli_mq",
            task_queue="cli_tasks"
        )
        
        config = RemoteConfig(
            worker=worker_config,
            cli=cli_config
        )
        
        assert config.worker.mq_provider_name == "worker_mq"
        assert config.worker.task_queue == "worker_tasks"
        assert config.cli.mq_provider_name == "cli_mq"
        assert config.cli.task_queue == "cli_tasks"
    
    def test_remote_config_partial_override(self):
        """Test remote configuration with partial section overrides."""
        worker_data = {
            "mq_provider_name": "override_mq",
            "task_queue": "override_tasks"
        }
        
        config = RemoteConfig(worker=worker_data)
        
        # Overridden worker fields
        assert config.worker.mq_provider_name == "override_mq"
        assert config.worker.task_queue == "override_tasks"
        
        # Default worker fields
        assert config.worker.state_persister_name == "redis"
        assert config.worker.results_queue == "agent_results"
        
        # Default CLI config
        assert config.cli.mq_provider_name == "rabbitmq"
        assert config.cli.task_queue == "agent_tasks"
    
    def test_remote_config_from_dict(self):
        """Test remote configuration creation from dictionary."""
        config_data = {
            "worker": {
                "mq_provider_name": "dict_worker_mq",
                "state_persister_name": "dict_persister",
                "task_queue": "dict_worker_queue",
                "results_queue": "dict_worker_results",
                "base_agent_config_path": "/dict/config.yaml"
            },
            "cli": {
                "mq_provider_name": "dict_cli_mq",
                "task_queue": "dict_cli_queue"
            }
        }
        
        config = RemoteConfig.model_validate(config_data)
        
        # Worker section
        assert config.worker.mq_provider_name == "dict_worker_mq"
        assert config.worker.state_persister_name == "dict_persister"
        assert config.worker.task_queue == "dict_worker_queue"
        assert config.worker.results_queue == "dict_worker_results"
        assert config.worker.base_agent_config_path == "/dict/config.yaml"
        
        # CLI section
        assert config.cli.mq_provider_name == "dict_cli_mq"
        assert config.cli.task_queue == "dict_cli_queue"
    
    def test_remote_config_serialization(self):
        """Test remote configuration serialization/deserialization."""
        original = RemoteConfig(
            worker=WorkerServiceConfig(
                mq_provider_name="serialization_worker_mq",
                task_queue="serialization_worker_queue"
            ),
            cli=CLIToolConfig(
                mq_provider_name="serialization_cli_mq",
                task_queue="serialization_cli_queue"
            )
        )
        
        # Test model_dump
        data = original.model_dump()
        assert data["worker"]["mq_provider_name"] == "serialization_worker_mq"
        assert data["cli"]["mq_provider_name"] == "serialization_cli_mq"
        
        # Test JSON round trip
        json_str = original.model_dump_json()
        restored = RemoteConfig.model_validate_json(json_str)
        
        assert restored.worker.mq_provider_name == original.worker.mq_provider_name
        assert restored.worker.task_queue == original.worker.task_queue
        assert restored.cli.mq_provider_name == original.cli.mq_provider_name
        assert restored.cli.task_queue == original.cli.task_queue
    
    def test_remote_config_empty_sections(self):
        """Test remote configuration with empty sections."""
        config_data = {
            "worker": {},
            "cli": {}
        }
        
        config = RemoteConfig.model_validate(config_data)
        
        # Should use defaults for all fields
        assert config.worker.mq_provider_name == "rabbitmq"
        assert config.worker.state_persister_name == "redis"
        assert config.cli.mq_provider_name == "rabbitmq"
        assert config.cli.task_queue == "agent_tasks"
    
    def test_remote_config_missing_sections(self):
        """Test remote configuration with missing sections."""
        config_data = {}
        
        config = RemoteConfig.model_validate(config_data)
        
        # Should create default sections
        assert config.worker is not None
        assert config.cli is not None
        assert config.worker.mq_provider_name == "rabbitmq"
        assert config.cli.mq_provider_name == "rabbitmq"


class TestConfigIntegration:
    """Test integration between configuration models."""
    
    def test_config_consistency(self):
        """Test configuration consistency across sections."""
        # Create config where worker and CLI use same MQ provider
        shared_mq = "shared_rabbitmq"
        shared_queue = "shared_tasks"
        
        config = RemoteConfig(
            worker=WorkerServiceConfig(
                mq_provider_name=shared_mq,
                task_queue=shared_queue
            ),
            cli=CLIToolConfig(
                mq_provider_name=shared_mq,
                task_queue=shared_queue
            )
        )
        
        assert config.worker.mq_provider_name == config.cli.mq_provider_name
        assert config.worker.task_queue == config.cli.task_queue
    
    def test_config_independence(self):
        """Test that different sections can have independent settings."""
        config = RemoteConfig(
            worker=WorkerServiceConfig(
                mq_provider_name="worker_mq",
                task_queue="worker_queue"
            ),
            cli=CLIToolConfig(
                mq_provider_name="cli_mq",
                task_queue="cli_queue"
            )
        )
        
        # Worker and CLI should be able to use different providers/queues
        assert config.worker.mq_provider_name != config.cli.mq_provider_name
        assert config.worker.task_queue != config.cli.task_queue
    
    def test_config_yaml_compatibility(self):
        """Test configuration compatibility with YAML structure."""
        # Simulate what would come from a YAML file
        yaml_like_data = {
            "worker": {
                "mq_provider_name": "production_rabbitmq",
                "state_persister_name": "production_redis",
                "task_queue": "prod_agent_tasks",
                "results_queue": "prod_agent_results",
                "base_agent_config_path": "/etc/flowlib/agent.yaml"
            },
            "cli": {
                "mq_provider_name": "production_rabbitmq",
                "task_queue": "prod_agent_tasks"
            }
        }
        
        config = RemoteConfig.model_validate(yaml_like_data)
        
        # Verify YAML-like structure is properly parsed
        assert config.worker.mq_provider_name == "production_rabbitmq"
        assert config.worker.base_agent_config_path == "/etc/flowlib/agent.yaml"
        assert config.cli.mq_provider_name == "production_rabbitmq"
        assert config.worker.task_queue == config.cli.task_queue