"""Tests for configuration resource classes."""
import pytest
from pydantic import ValidationError
from typing import Dict, Any

from flowlib.resources.models.config_resource import (
    ProviderConfigResource,
    LLMConfigResource,
    DatabaseConfigResource,
    VectorDBConfigResource,
    CacheConfigResource,
    StorageConfigResource,
    EmbeddingConfigResource,
    GraphDBConfigResource,
    MessageQueueConfigResource,
)


class TestProviderConfigResource:
    """Test base provider configuration resource."""
    
    def test_valid_config(self):
        """Test valid provider configuration."""
        config = ProviderConfigResource(
            name="test_provider",
            type="config",
            provider_type="test_impl",
            settings={"param1": "value1", "param2": 42}
        )
        
        assert config.name == "test_provider"
        assert config.type == "config"
        assert config.provider_type == "test_impl"
        assert config.settings == {"param1": "value1", "param2": 42}
    
    def test_minimal_config(self):
        """Test minimal provider configuration."""
        config = ProviderConfigResource(
            name="minimal",
            type="config",
            provider_type="minimal_impl"
        )
        
        assert config.name == "minimal"
        assert config.provider_type == "minimal_impl"
        assert config.settings == {}
    
    def test_get_provider_type(self):
        """Test getting provider type."""
        config = ProviderConfigResource(
            name="test",
            type="config",
            provider_type="test_provider"
        )
        
        assert config.get_provider_type() == "test_provider"
    
    def test_get_settings(self):
        """Test getting settings."""
        settings = {"key1": "value1", "key2": 123}
        config = ProviderConfigResource(
            name="test",
            type="config",
            provider_type="test_provider",
            settings=settings
        )
        
        assert config.get_settings() == settings
    
    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            ProviderConfigResource(
                name="test",
                type="config"
                # Missing provider_type
            )
        
        assert "provider_type" in str(exc_info.value)
    
    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            ProviderConfigResource(
                name="test",
                type="config",
                provider_type="test",
                extra_field="not_allowed"
            )
        
        assert "extra_field" in str(exc_info.value)


class TestLLMConfigResource:
    """Test LLM configuration resource."""
    
    def test_valid_llm_config(self):
        """Test valid LLM configuration - Provider-level settings only."""
        config = LLMConfigResource(
            name="llamacpp_config",
            type="llm_config",
            provider_type="llamacpp",
            n_threads=8,
            n_batch=512,
            use_gpu=True,
            n_gpu_layers=32,
            verbose=False,
            timeout=120,
            max_concurrent_models=3,
            settings={"custom_param": "value"}
        )
        
        assert config.name == "llamacpp_config"
        assert config.provider_type == "llamacpp"
        assert config.n_threads == 8
        assert config.n_batch == 512
        assert config.use_gpu is True
        assert config.n_gpu_layers == 32
        assert config.verbose is False
        assert config.timeout == 120
        assert config.max_concurrent_models == 3
        assert config.settings == {"custom_param": "value"}
    
    def test_default_llm_config(self):
        """Test default LLM configuration values - Provider-level defaults only."""
        config = LLMConfigResource(
            name="default_llm",
            type="llm_config",
            provider_type="llamacpp"
        )
        
        # Provider infrastructure fields should have proper defaults
        assert config.n_threads is None  # Will use system defaults
        assert config.n_batch is None  # Will use system defaults
        assert config.use_gpu is None  # Will be determined by provider
        assert config.n_gpu_layers is None  # Will use provider defaults
        assert config.verbose is None  # Will use provider defaults
        assert config.timeout is None  # Will use provider defaults
        assert config.max_concurrent_models is None  # Will use provider defaults
    
    def test_get_provider_settings(self):
        """Test getting provider settings - Infrastructure settings only."""
        config = LLMConfigResource(
            name="test_llm",
            type="llm_config",
            provider_type="llamacpp",
            n_threads=4,
            use_gpu=True,
            n_gpu_layers=20,
            timeout=60,
            settings={"custom_param": "value"}
        )
        
        settings = config.get_provider_settings()
        
        # Should include provider infrastructure settings
        assert settings["custom_param"] == "value"
        # Provider-specific settings are accessible but model settings are not here
    
    def test_get_provider_settings_minimal(self):
        """Test provider settings with minimal configuration."""
        config = LLMConfigResource(
            name="test_llm",
            type="llm_config",
            provider_type="llamacpp",
            settings={"base_param": "base_value"}
        )
        
        settings = config.get_provider_settings()
        
        # Should only contain settings dict contents
        assert settings["base_param"] == "base_value"
        # No model-specific generation settings at provider level
    
    def test_provider_field_validation(self):
        """Test provider infrastructure field validation."""
        # Valid provider settings
        config = LLMConfigResource(
            name="test",
            type="llm_config",
            provider_type="llamacpp",
            n_threads=8,
            n_batch=512,
            use_gpu=True
        )
        assert config.n_threads == 8
        assert config.n_batch == 512
        assert config.use_gpu is True
        
        # Provider configs should accept None for optional infrastructure settings
        config_minimal = LLMConfigResource(
            name="minimal",
            type="llm_config",
            provider_type="llamacpp"
        )
        assert config_minimal.n_threads is None
        assert config_minimal.n_batch is None
        assert config_minimal.use_gpu is None
    
    def test_infrastructure_settings_validation(self):
        """Test infrastructure settings validation."""
        # Valid infrastructure settings
        config = LLMConfigResource(
            name="test",
            type="llm_config",
            provider_type="llamacpp",
            timeout=120,
            max_concurrent_models=5
        )
        assert config.timeout == 120
        assert config.max_concurrent_models == 5
        
        # Infrastructure settings can be None (use provider defaults)
        config_defaults = LLMConfigResource(
            name="test_defaults",
            type="llm_config",
            provider_type="llamacpp"
        )
        assert config_defaults.timeout is None
        assert config_defaults.max_concurrent_models is None
    
    def test_gpu_settings_validation(self):
        """Test GPU-related settings validation."""
        # Valid GPU settings
        config = LLMConfigResource(
            name="test",
            type="llm_config",
            provider_type="llamacpp",
            use_gpu=True,
            n_gpu_layers=32
        )
        assert config.use_gpu is True
        assert config.n_gpu_layers == 32
        
        # GPU settings can be None (provider will determine)
        config_auto = LLMConfigResource(
            name="test_auto",
            type="llm_config",
            provider_type="llamacpp"
        )
        assert config_auto.use_gpu is None
        assert config_auto.n_gpu_layers is None


class TestDatabaseConfigResource:
    """Test database configuration resource."""
    
    def test_valid_database_config(self):
        """Test valid database configuration."""
        config = DatabaseConfigResource(
            name="postgres_config",
            type="db_config",
            provider_type="postgresql",
            settings={
                "host": "localhost",
                "port": 5432,
                "database": "testdb",
                "username": "user",
                "password": "pass",
                "pool_size": 10,
                "ssl_mode": "require"
            }
        )
        
        assert config.name == "postgres_config"
        assert config.provider_type == "postgresql"
        assert config.settings["host"] == "localhost"
        assert config.settings["port"] == 5432
        assert config.settings["database"] == "testdb"
        assert config.settings["username"] == "user"
        assert config.settings["password"] == "pass"
        assert config.settings["pool_size"] == 10
    
    def test_minimal_database_config(self):
        """Test minimal database configuration."""
        config = DatabaseConfigResource(
            name="minimal_db",
            type="db_config",
            provider_type="sqlite",
            settings={
                "database": "test.db"
            }
        )
        
        # Get required settings with defaults
        required_settings = config.get_required_settings()
        assert required_settings["database"] == "test.db"
        assert required_settings["host"] == "localhost"  # default
        assert required_settings["port"] == 5432  # default
    
    def test_get_connection_settings(self):
        """Test getting connection settings."""
        config = DatabaseConfigResource(
            name="test_db",
            type="db_config",
            provider_type="postgres",
            settings={
                "host": "db.example.com",
                "port": 5432,
                "database": "mydb",
                "username": "dbuser",
                "password": "dbpass",
                "pool_size": 20,
                "timeout": 30
            }
        )
        
        settings = config.get_required_settings()
        
        assert settings["host"] == "db.example.com"
        assert settings["port"] == 5432
        assert settings["database"] == "mydb"
        assert settings["username"] == "dbuser"
        assert settings["password"] == "dbpass"
        assert settings["pool_size"] == 20
        assert settings["timeout"] == 30
    
    def test_get_connection_settings_no_credentials(self):
        """Test connection settings without credentials."""
        config = DatabaseConfigResource(
            name="test_db",
            type="db_config",
            provider_type="sqlite",
            settings={
                "database": "test.db"
            }
        )
        
        settings = config.get_required_settings()
        
        # Should get defaults for required fields
        assert settings["host"] == "localhost"  # default
        assert settings["port"] == 5432  # default
        assert settings["database"] == "test.db"
    
    def test_settings_structure(self):
        """Test database settings structure."""
        # Valid config with settings
        config = DatabaseConfigResource(
            name="test",
            type="db_config",
            provider_type="postgres",
            settings={"database": "testdb"}
        )
        
        # Settings should be accessible
        assert config.settings["database"] == "testdb"
        
        # Required settings method should provide defaults
        required = config.get_required_settings()
        assert "host" in required
        assert "port" in required
        assert "database" in required


class TestVectorDBConfigResource:
    """Test vector database configuration resource."""
    
    def test_valid_vector_config(self):
        """Test valid vector database configuration."""
        config = VectorDBConfigResource(
            name="chroma_config",
            type="vector_config",
            provider_type="chroma",
            settings={
                "host": "localhost",
                "collection_name": "documents",
                "dimensions": 768,
                "distance_metric": "euclidean",
                "index_type": "hnsw"
            }
        )
        
        assert config.settings["collection_name"] == "documents"
        assert config.settings["dimensions"] == 768
        assert config.settings["distance_metric"] == "euclidean"
        assert config.settings["index_type"] == "hnsw"
    
    def test_default_vector_config(self):
        """Test default vector database values."""
        config = VectorDBConfigResource(
            name="default_vector",
            type="vector_config",
            provider_type="pinecone"
        )
        
        # Should have empty settings by default
        assert config.settings == {}
        
        # Required settings should provide defaults where needed
        required = config.get_required_settings()
        assert isinstance(required, dict)
    
    def test_get_vector_settings(self):
        """Test getting vector settings."""
        config = VectorDBConfigResource(
            name="test_vector",
            type="vector_config",
            provider_type="qdrant",
            settings={
                "url": "http://localhost:6333",
                "collection_name": "test_collection",
                "dimensions": 384,
                "distance_metric": "dot",
                "index_type": "flat"
            }
        )
        
        settings = config.get_required_settings()
        
        assert settings["collection_name"] == "test_collection"
        assert settings["dimensions"] == 384
        assert settings["distance_metric"] == "dot"
        assert settings["index_type"] == "flat"
        assert settings["url"] == "http://localhost:6333"


class TestCacheConfigResource:
    """Test cache configuration resource."""
    
    def test_valid_cache_config(self):
        """Test valid cache configuration."""
        config = CacheConfigResource(
            name="redis_config",
            type="cache_config",
            provider_type="redis",
            settings={
                "host": "redis.example.com",
                "port": 6379,
                "db": 1,
                "ttl": 7200,
                "max_connections": 20,
                "password": "secret"
            }
        )
        
        assert config.settings["host"] == "redis.example.com"
        assert config.settings["port"] == 6379
        assert config.settings["db"] == 1
        assert config.settings["ttl"] == 7200
        assert config.settings["max_connections"] == 20
    
    def test_default_cache_config(self):
        """Test default cache values."""
        config = CacheConfigResource(
            name="default_cache",
            type="cache_config",
            provider_type="memory"
        )
        
        # Should have empty settings by default
        assert config.settings == {}
        
        # Required settings should provide defaults
        required = config.get_required_settings()
        assert required["host"] == "localhost"  # default
        assert required["port"] == 6379  # default
        assert required["db"] == 0  # default
        assert required["ttl"] == 3600  # default
        assert required["max_connections"] == 10  # default
    
    def test_get_cache_settings(self):
        """Test getting cache settings."""
        config = CacheConfigResource(
            name="test_cache",
            type="cache_config",
            provider_type="redis",
            settings={
                "host": "cache.local",
                "port": 6380,
                "db": 2,
                "ttl": 1800,
                "max_connections": 15,
                "ssl": True
            }
        )
        
        settings = config.get_required_settings()
        
        assert settings["host"] == "cache.local"
        assert settings["port"] == 6380
        assert settings["db"] == 2
        assert settings["ttl"] == 1800
        assert settings["max_connections"] == 15
        assert settings["ssl"] is True


class TestStorageConfigResource:
    """Test storage configuration resource."""
    
    def test_valid_storage_config(self):
        """Test valid storage configuration."""
        config = StorageConfigResource(
            name="s3_config",
            type="storage_config",
            provider_type="s3",
            settings={
                "bucket": "my-bucket",
                "region": "us-east-1",
                "access_key": "AKIAEXAMPLE",
                "secret_key": "secret123",
                "endpoint": "https://s3.amazonaws.com",
                "use_ssl": True
            }
        )
        
        assert config.settings["bucket"] == "my-bucket"
        assert config.settings["region"] == "us-east-1"
        assert config.settings["access_key"] == "AKIAEXAMPLE"
        assert config.settings["secret_key"] == "secret123"
        assert config.settings["endpoint"] == "https://s3.amazonaws.com"
    
    def test_minimal_storage_config(self):
        """Test minimal storage configuration."""
        config = StorageConfigResource(
            name="local_storage",
            type="storage_config",
            provider_type="local"
        )
        
        # Should have empty settings by default
        assert config.settings == {}
        
        # Required settings method should work
        required = config.get_required_settings()
        assert isinstance(required, dict)
    
    def test_get_storage_settings(self):
        """Test getting storage settings."""
        config = StorageConfigResource(
            name="test_storage",
            type="storage_config",
            provider_type="minio",
            settings={
                "bucket": "test-bucket",
                "region": "us-west-2",
                "access_key": "minioadmin",
                "secret_key": "minioadmin",
                "endpoint": "http://localhost:9000",
                "path_style": True
            }
        )
        
        settings = config.get_required_settings()
        
        assert settings["bucket"] == "test-bucket"
        assert settings["region"] == "us-west-2"
        assert settings["access_key"] == "minioadmin"
        assert settings["secret_key"] == "minioadmin"
        assert settings["endpoint"] == "http://localhost:9000"
        assert settings["path_style"] is True


class TestEmbeddingConfigResource:
    """Test embedding configuration resource."""
    
    def test_valid_embedding_config(self):
        """Test valid embedding configuration."""
        config = EmbeddingConfigResource(
            name="openai_embeddings",
            type="embedding_config",
            provider_type="openai",
            settings={
                "api_key": "secret",
                "model_name": "text-embedding-ada-002",
                "dimensions": 1536,
                "batch_size": 64,
                "normalize": False
            }
        )
        
        assert config.settings["model_name"] == "text-embedding-ada-002"
        assert config.settings["dimensions"] == 1536
        assert config.settings["batch_size"] == 64
        assert config.settings["normalize"] is False
    
    def test_default_embedding_config(self):
        """Test default embedding values."""
        config = EmbeddingConfigResource(
            name="default_embedding",
            type="embedding_config",
            provider_type="sentence_transformers"
        )
        
        # Should have empty settings by default
        assert config.settings == {}
        
        # Required settings should provide defaults
        required = config.get_required_settings()
        assert required["batch_size"] == 32  # default
        assert required["normalize"] is True  # default
    
    def test_get_embedding_settings(self):
        """Test getting embedding settings."""
        config = EmbeddingConfigResource(
            name="test_embedding",
            type="embedding_config",
            provider_type="llamacpp",
            settings={
                "model_name": "all-MiniLM-L6-v2",
                "dimensions": 384,
                "batch_size": 16,
                "normalize": True,
                "device": "cuda"
            }
        )
        
        settings = config.get_required_settings()
        
        assert settings["model_name"] == "all-MiniLM-L6-v2"
        assert settings["dimensions"] == 384
        assert settings["batch_size"] == 16
        assert settings["normalize"] is True
        assert settings["device"] == "cuda"


class TestGraphDBConfigResource:
    """Test graph database configuration resource."""
    
    def test_valid_graph_config(self):
        """Test valid graph database configuration."""
        config = GraphDBConfigResource(
            name="neo4j_config",
            type="graph_config",
            provider_type="neo4j",
            settings={
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "password",
                "database": "graph",
                "encrypted": True
            }
        )
        
        assert config.settings["uri"] == "bolt://localhost:7687"
        assert config.settings["username"] == "neo4j"
        assert config.settings["password"] == "password"
        assert config.settings["database"] == "graph"
    
    def test_minimal_graph_config(self):
        """Test minimal graph configuration."""
        config = GraphDBConfigResource(
            name="memory_graph",
            type="graph_config",
            provider_type="memory",
            settings={"uri": "memory://"}
        )
        
        # Should have minimal settings
        assert config.settings["uri"] == "memory://"
        assert "username" not in config.settings
        assert "password" not in config.settings
        assert "database" not in config.settings
    
    def test_get_graph_settings(self):
        """Test getting graph settings."""
        config = GraphDBConfigResource(
            name="test_graph",
            type="graph_config",
            provider_type="arangodb",
            settings={
                "uri": "http://localhost:8529",
                "username": "root",
                "password": "admin",
                "database": "test_db",
                "verify_certificate": False
            }
        )
        
        settings = config.get_required_settings()
        
        assert settings["uri"] == "http://localhost:8529"
        assert settings["username"] == "root"
        assert settings["password"] == "admin"
        assert settings["database"] == "test_db"
        assert settings["verify_certificate"] is False
    
    def test_graph_settings_structure(self):
        """Test graph settings structure."""
        config = GraphDBConfigResource(
            name="test",
            type="graph_config",
            provider_type="neo4j",
            settings={"uri": "bolt://localhost:7687"}
        )
        
        # Settings should be accessible
        assert config.settings["uri"] == "bolt://localhost:7687"
        
        # Required settings should provide defaults
        required = config.get_required_settings()
        assert "uri" in required
        # Should use default URI if not provided
        assert required["uri"] == "bolt://localhost:7687"


class TestMessageQueueConfigResource:
    """Test message queue configuration resource."""
    
    def test_valid_mq_config(self):
        """Test valid message queue configuration."""
        config = MessageQueueConfigResource(
            name="rabbitmq_config",
            type="mq_config",
            provider_type="rabbitmq",
            settings={
                "host": "rabbitmq.example.com",
                "port": 5672,
                "username": "guest",
                "password": "guest",
                "vhost": "/",
                "ssl": True
            }
        )
        
        assert config.settings["host"] == "rabbitmq.example.com"
        assert config.settings["port"] == 5672
        assert config.settings["username"] == "guest"
        assert config.settings["password"] == "guest"
        assert config.settings["vhost"] == "/"
    
    def test_minimal_mq_config(self):
        """Test minimal message queue configuration."""
        config = MessageQueueConfigResource(
            name="local_mq",
            type="mq_config",
            provider_type="memory",
            settings={
                "host": "localhost",
                "port": 5672
            }
        )
        
        # Should have minimal required settings
        assert config.settings["host"] == "localhost"
        assert config.settings["port"] == 5672
        assert "username" not in config.settings
        assert "password" not in config.settings
        assert "vhost" not in config.settings
    
    def test_get_mq_settings(self):
        """Test getting message queue settings."""
        config = MessageQueueConfigResource(
            name="test_mq",
            type="mq_config",
            provider_type="kafka",
            settings={
                "host": "kafka.local",
                "port": 9092,
                "username": "kafka_user",
                "password": "kafka_pass",
                "vhost": "test_vhost",
                "security_protocol": "SASL_SSL"
            }
        )
        
        settings = config.get_required_settings()
        
        assert settings["host"] == "kafka.local"
        assert settings["port"] == 9092
        assert settings["username"] == "kafka_user"
        assert settings["password"] == "kafka_pass"
        assert settings["vhost"] == "test_vhost"
        assert settings["security_protocol"] == "SASL_SSL"
    
    def test_mq_settings_structure(self):
        """Test message queue settings structure."""
        config = MessageQueueConfigResource(
            name="test",
            type="mq_config",
            provider_type="rabbitmq",
            settings={
                "host": "localhost",
                "port": 5672
            }
        )
        
        # Settings should be accessible
        assert config.settings["host"] == "localhost"
        assert config.settings["port"] == 5672
        
        # Required settings should provide defaults
        required = config.get_required_settings()
        assert required["host"] == "localhost"
        assert required["port"] == 5672


class TestConfigResourceIntegration:
    """Integration tests for configuration resources."""
    
    def test_config_inheritance(self):
        """Test that all config resources inherit from base."""
        configs = [
            LLMConfigResource(name="test", type="test", provider_type="test"),
            DatabaseConfigResource(name="test", type="test", provider_type="test", 
                                 settings={"host": "localhost", "port": 5432, "database": "test"}),
            VectorDBConfigResource(name="test", type="test", provider_type="test"),
            CacheConfigResource(name="test", type="test", provider_type="test", 
                              settings={"host": "localhost", "port": 6379}),
            StorageConfigResource(name="test", type="test", provider_type="test"),
            EmbeddingConfigResource(name="test", type="test", provider_type="test"),
            GraphDBConfigResource(name="test", type="test", provider_type="test", 
                                settings={"uri": "test://"}),
            MessageQueueConfigResource(name="test", type="test", provider_type="test", 
                                     settings={"host": "localhost", "port": 5672}),
        ]
        
        for config in configs:
            assert isinstance(config, ProviderConfigResource)
            assert hasattr(config, 'get_provider_type')
            assert hasattr(config, 'get_settings')
            assert config.get_provider_type() == "test"
    
    def test_settings_merging(self):
        """Test that specific settings merge with base settings."""
        config = LLMConfigResource(
            name="test_llm",
            type="llm_config",
            provider_type="llamacpp",
            n_threads=8,
            use_gpu=True,
            settings={"api_key": "secret", "custom_param": "value"}
        )
        
        provider_settings = config.get_provider_settings()
        
        # Should have base settings from settings dict
        assert provider_settings["api_key"] == "secret"
        assert provider_settings["custom_param"] == "value"
    
    def test_polymorphic_usage(self):
        """Test using configs polymorphically."""
        configs: list[ProviderConfigResource] = [
            LLMConfigResource(name="llm", type="llm", provider_type="llamacpp"),
            DatabaseConfigResource(name="db", type="db", provider_type="postgres", 
                                 settings={"host": "localhost", "port": 5432, "database": "test"}),
            CacheConfigResource(name="cache", type="cache", provider_type="redis", 
                              settings={"host": "localhost", "port": 6379}),
        ]
        
        for config in configs:
            # All should have base functionality
            assert config.get_provider_type() in ["llamacpp", "postgres", "redis"]
            assert isinstance(config.get_settings(), dict)
            
            # Type-specific functionality
            if isinstance(config, LLMConfigResource):
                assert hasattr(config, 'get_provider_settings')
            elif isinstance(config, DatabaseConfigResource):
                assert hasattr(config, 'get_required_settings')
            elif isinstance(config, CacheConfigResource):
                assert hasattr(config, 'get_required_settings')